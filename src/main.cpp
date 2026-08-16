#include <stdio.h>
#include <assert.h>
#include <span>
#include <chrono>

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/global_control.h>

#include "pcg_basic.h"


std::vector<uint64_t> readBinFile(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f)
        std::terminate();
    uint64_t num = 0;
    if (fread(&num, sizeof(num), 1, f) != 1)
        std::terminate();
    std::vector<uint64_t> data(num);
    if (fread(data.data(), sizeof(data[0]), num, f) != num)
        std::terminate();
    fclose(f);
    return data;
}

//===============================================

//#define SLOW_ASSERT 1
//#define REDUCE_BITS 40
//#define TBB_FORCE_THREADS 1

bool isPot(size_t x) {
    return (x & (x - 1)) == 0;
}

size_t log2up(size_t x) {
    size_t res = 0;
    while ((1 << res) < x)
        res++;
    return res;
}

struct Random {
    pcg32_random_t lower_{};
    pcg32_random_t upper_{};

    void initStream(uint64_t seed, size_t stream) {
        pcg32_srandom_r(&lower_, seed, 2 * stream + 0);
        pcg32_srandom_r(&upper_, seed, 2 * stream + 1);
    }

    size_t generate(size_t maxExclusive) {
        uint64_t x = pcg32_random_r(&upper_);
        x <<= 32;
        x += pcg32_random_r(&lower_);
        return x % maxExclusive;
    }
};


typedef uint64_t Value;
template<class T> using Span = std::span<T>;
template<class T> using Array = std::vector<T>;

struct MultiPivot {
    size_t numBits_ = 0;
    size_t numBuckets_ = 0;
    Array<Value> sorted_;
    Array<Value> tree_;

    void select(
        Span<const Value> arr, size_t numBuckets,
        Random &random, Array<Value> &samplesStore
    ) {
        assert(isPot(numBuckets) && numBuckets >= 2);
        size_t numElems = arr.size();

        size_t numSamples = numBuckets * log2up(numElems) / 5;
        numSamples = std::max(numSamples, numBuckets);

        samplesStore.resize(numSamples);
        Span<Value> samples = samplesStore;
        for (size_t i = 0; i < numSamples; i++) {
            size_t index = random.generate(numElems);
            samples[i] = arr[index];
        }

        std::sort(samples.data(), samples.data() + samples.size());

        sorted_.resize(numBuckets - 1);
        for (size_t i = 1; i <= numBuckets - 1; i++) {
            uint64_t pos = uint64_t(numSamples) * i / numBuckets;
            sorted_[i - 1] = samples[pos];
        }

        size_t numBits = log2up(numBuckets);

        tree_.resize(numBuckets - 1);
        size_t v = 0;
        for (int b = numBits - 1; b >= 0; b--) {
            size_t len = (1 << b);
            for (size_t i = len - 1; i < numBuckets; i += len * 2)
                tree_[v++] = sorted_[i];
        }

        numBuckets_ = numBuckets;
        numBits_ = numBits;        
    }

    inline size_t classifyOne(const Value &value) const {
        size_t res = 0;
        Span<const Value> tree = tree_;
        for (size_t b = 0; b < numBits_; b++) {
            bool isLess = (value < tree[res]);
            res = 2 * res + 1 + size_t(!isLess);
        }
        res -= (numBuckets_ - 1);
        assert(res == numBuckets_ - 1 || (value < sorted_[res]));
        assert(res == 0 || !(value < sorted_[res - 1]));
        return res;
    }
};


struct SharedData;

struct ThreadData {
    Array<size_t> splitsStore_;
    Array<Value> samplesStore_;
    MultiPivot pivot_;
};

struct TaskData {
    size_t first_ = 0;
    size_t len_ = 0;
    size_t world_ = 0;
    size_t numWorkers_ = 0;
    Random random_;

    SharedData* shared_ = nullptr;
};

struct SharedData {
    size_t numElems_ = 0;
    Span<Value> elemsSpans_[2];
    Array<Value> elemsCopyStore_;

    Array<uint8_t> bucketIndexStore_;

    uint64_t randomSeed_ = 0xDEADBEEF01234567ull;
};

void multiPartition(
    Span<const Value> srcElems, const MultiPivot &pivot,
    Span<Value> dstElems, Span<size_t> splits, 
    size_t numWorkers, Span<uint8_t> bucketOf
) {
    size_t numBuckets = pivot.numBuckets_;
    size_t numElems = srcElems.size();
    assert(splits.size() == numBuckets + 1);
    
    std::vector<std::vector<size_t>> localHisto(numWorkers, std::vector<size_t>(numBuckets, 0));
    tbb::parallel_for<size_t>(0, numWorkers, [&](size_t t) {
        size_t l = uint64_t(numElems) * (t + 0) / numWorkers;
        size_t r = uint64_t(numElems) * (t + 1) / numWorkers;
        for (size_t i = l; i < r; i++) {
            size_t b = pivot.classifyOne(srcElems[i]);
            bucketOf[i] = b;
            localHisto[t][b]++;
        }
    });

    std::vector<size_t> globalHisto(numBuckets + 1, 0);
    for (size_t t = 0; t < numWorkers; t++)
        for (size_t b = 0; b < numBuckets; b++)
            globalHisto[b + 1] += localHisto[t][b];

    for (size_t b = 0; b < numBuckets; b++)
        globalHisto[b + 1] += globalHisto[b];
    assert(globalHisto[numBuckets] == numElems);

    for (size_t b = 0; b < numBuckets; b++) {
        size_t tsum = globalHisto[b];
        for (size_t t = 0; t < numWorkers; t++) {
            size_t nsum = tsum + localHisto[t][b];
            localHisto[t][b] = tsum;
            tsum = nsum;
        }
        assert(tsum == globalHisto[b + 1]);
    }
    
    tbb::parallel_for<size_t>(0, numWorkers, [&](size_t t) {
        size_t l = uint64_t(numElems) * (t + 0) / numWorkers;
        size_t r = uint64_t(numElems) * (t + 1) / numWorkers;
        for (size_t i = l; i < r; i++) {
            size_t b = bucketOf[i];
            size_t &pos = localHisto[t][b];
            dstElems[pos++] = srcElems[i];
        }
    });

    for (size_t b = 0; b <= numBuckets; b++)
        splits[b] = globalHisto[b];

#if SLOW_ASSERT
    assert(splits[0] == 0 && splits[numBuckets] == numElems);
    for (size_t b = 0; b < numBuckets; b++) {
        assert(splits[b] <= splits[b + 1]);
        for (size_t i = splits[b]; i < splits[b + 1]; i++) {
            size_t vb = pivot.classifyOne(dstElems[i]);
            assert(vb == b);
        }
    }
#endif    
}


void processRecursive(TaskData task) {
    SharedData *shared = task.shared_;
    //ThreadData *perThread = &shared->perThread_.local();
    ThreadData ptd;
    ThreadData *perThread = &ptd;

    Span<Value> srcElems = shared->elemsSpans_[task.world_].subspan(task.first_, task.len_);
    Span<Value> dstElems = shared->elemsSpans_[task.world_ ^ 1].subspan(task.first_, task.len_);
    Span<uint8_t> bucketOf = Span<uint8_t>(shared->bucketIndexStore_).subspan(task.first_, task.len_);

    if (task.len_ < (1 << 10)) {
        std::sort(srcElems.data(), srcElems.data() + srcElems.size());
        if (task.world_ != 0) {
            for (size_t i = 0; i < srcElems.size(); i++)
                dstElems[i] = srcElems[i];
        }
        return;
    }

    size_t numBuckets = 256;
    perThread->pivot_.select(srcElems, numBuckets, task.random_, perThread->samplesStore_);

    perThread->splitsStore_.resize(numBuckets + 1);
    multiPartition(
        srcElems, perThread->pivot_,
        dstElems, perThread->splitsStore_,
        task.numWorkers_, bucketOf
    );

    tbb::parallel_for<size_t>(0, numBuckets, [&](size_t t) {
        TaskData subTask;
        subTask.shared_ = task.shared_;
        subTask.first_ = task.first_ + perThread->splitsStore_[t];
        subTask.len_ = perThread->splitsStore_[t + 1] - perThread->splitsStore_[t];
        subTask.world_ = task.world_ ^ 1;
        subTask.numWorkers_ = (task.numWorkers_ + numBuckets - 1) / numBuckets;
        if (t == 0)
            subTask.random_ = task.random_;
        else
            subTask.random_.initStream(task.shared_->randomSeed_, subTask.first_);
        processRecursive(subTask);
    });
}

void sort(Value *begin, size_t num) {
    SharedData shared;
    shared.elemsCopyStore_.resize(num);
    shared.bucketIndexStore_.resize(num);
    shared.numElems_ = num;
    shared.elemsSpans_[0] = {begin, num};
    shared.elemsSpans_[1] = shared.elemsCopyStore_;

    TaskData task;
    task.shared_ = &shared;
    task.first_ = 0;
    task.len_ = shared.numElems_;
    task.numWorkers_ = tbb::this_task_arena::max_concurrency() * 3;
    task.world_ = 0;
    task.random_.initStream(shared.randomSeed_, task.first_);

    processRecursive(task);
}

//===============================================

auto getTimestamp() -> auto {
    return std::chrono::steady_clock::now();
}

double getTimeDiff(auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main() {
#ifdef TBB_FORCE_THREADS
    tbb::global_control control(tbb::global_control::max_allowed_parallelism, TBB_FORCE_THREADS);
#endif

    auto t0 = getTimestamp();
    std::vector<uint64_t> input = readBinFile("input.bin");
#ifdef REDUCE_BITS
    for (uint64_t &x : input)
        x >>= REDUCE_BITS;
#endif
    auto t1 = getTimestamp();
    printf("%4.0lf ms : read input\n", getTimeDiff(t0, t1));

    sort(input.data(), input.size());
    auto t2 = getTimestamp();
    printf("%4.0lf ms : sort\n", getTimeDiff(t1, t2));

    std::vector<size_t> badpos;
    for (size_t i = 1; i < input.size(); i++)
        if (input[i] < input[i - 1])
            badpos.push_back(i - 1);

    if (!badpos.empty()) {
        printf("FAILED: %zu of %zu\n", badpos.size(), input.size());
        for (size_t i = 0; i < std::min(int(badpos.size()), 100); i++)
            printf("  [%zu .. %zu]: %zu > %zu\n", badpos[i], badpos[i] + 1, input[badpos[i]], input[badpos[i] + 1]);
        std::terminate();
    }

    assert(std::is_sorted(input.data(), input.data() + input.size()));
    auto t3 = getTimestamp();
    printf("%4.0lf ms : validate\n", getTimeDiff(t2, t3));

    return 0;
}
