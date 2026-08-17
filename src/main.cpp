#include <stdio.h>
#include <assert.h>
#include <span>
#include <chrono>
#include <type_traits>

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

#define CLASSIFY_UNROLL 8
#define SMALLSORT_MAX 32

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

template<class Lambda> void parallelWorkers(size_t numWorkers, Lambda&& lambda) {
    assert(numWorkers > 0);
    if (numWorkers == 1) {
        lambda(0);
    }
    else {
        tbb::parallel_for<size_t>(0, numWorkers, lambda);
    }
}

// requirements:
//   destructor, move constructor: must not throw
template<class T> struct ValueTraits {
    inline void relocateOne(T *dst, const T *src) noexcept {
        *dst = static_cast<T&&>(*src);
        *src.~T();
    }
    void relocateMany(T *dst, const T *src, size_t n) noexcept {
        assert(dst >= src + n || src >= dst + n);
        for (size_t i = 0; i < n; i++)
            relocateOne(&dst[i], &src[i]);
    }
};

// because I need this to work on C++17
template<class T> class Span {
    T *ptr_ = nullptr;
    size_t num_ = 0;

public:
    Span() = default;
    inline Span(T *ptr, size_t num)
        : ptr_(ptr)
        , num_(num)
    {}
    template<class U, std::enable_if_t<std::is_same_v<U, std::add_const_t<T>>, int> = 0>
    inline operator Span<U>() const {
        return {ptr_, num_};
    }

    inline T *data() const { return ptr_; }
    inline size_t size() const { return num_; }

    inline T &operator[] (size_t i) const {
        assert(i < num_);
        return ptr_[i];
    }

    inline Span<T> subspan(size_t start, size_t len) const {
        assert(start + len <= num_);
        return {ptr_ + start, len};
    }
};

// simple reusable array of raw memory
// elements lifetime management is external
template<class T> class Array {
    T *ptr_ = nullptr;
    size_t num_ = 0;
    size_t cap_ = 0;

    void grow(size_t n) {
        assert(n > cap_);
        n = std::max(n, 2 * cap_ + 1);
        if (cap_ > 0)
            operator delete[] (ptr_);
        if (n > 0)
            ptr_ = (T*) operator new[] (n * sizeof(T), std::align_val_t(alignof(T)));
        cap_ = n;
    }

public:
    ~Array() {
        if (cap_ > 0)
            operator delete[] (ptr_);
    }
    Array() = default;

    inline T *data() { return ptr_; }
    inline const T *data() const { return ptr_; }
    inline size_t size() const { return num_; }

    inline T &operator[] (size_t i) {
        assert(i < num_);
        return ptr_[i];
    }
    inline const T &operator[] (size_t i) const {
        assert(i < num_);
        return ptr_[i];
    }

    void resize(size_t n) {
        if (n > cap_)
            grow(n);
        num_ = n;
    }

    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
    Array(const Array&&) = delete;
    Array& operator=(const Array&&) = delete;
};

template<class T> inline Span<T> makeSpan(Array<T> &arr) { return {arr.data(), arr.size()}; }
template<class T> inline Span<const T> makeSpan(const Array<T> &arr) { return {arr.data(), arr.size()}; }


typedef uint64_t Value;

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
        Span<Value> samples = makeSpan(samplesStore);
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
        Span<const Value> tree = makeSpan(tree_);
        for (size_t b = 0; b < numBits_; b++) {
            bool isLess = (value < tree[res]);
            res = 2 * res + 1 + size_t(!isLess);
        }
        res -= (numBuckets_ - 1);
        assert(res == numBuckets_ - 1 || (value < sorted_[res]));
        assert(res == 0 || !(value < sorted_[res - 1]));
        return res;
    }

    template<size_t N> inline void classifyBlock(const Value *value, size_t *res) const {
        Span<const Value> tree = makeSpan(tree_);
        for (size_t i = 0; i < N; i++)
            res[i] = 0;
        for (size_t b = 0; b < numBits_; b++) {
            for (size_t i = 0; i < N; i++) {
                bool isLess = (value[i] < tree_[res[i]]);
                res[i] = 2 * res[i] + 1 + size_t(!isLess);
            }
        }
        for (size_t i = 0; i < N; i++)
            res[i] -= (numBuckets_ - 1);
    }
};

void smallSort(Span<Value> arr) {
#if 0
    std::sort(arr.data(), arr.data() + arr.size());
#else
    for (size_t i = 1; i < arr.size(); i++)
        for (size_t j = 0; j < i; j++)
            if (arr[i] < arr[j])
                std::swap(arr[i], arr[j]);
#endif
}

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
    tbb::task_group taskGroup_;
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
    parallelWorkers(numWorkers, [&](size_t t) {
        size_t l = uint64_t(numElems) * (t + 0) / numWorkers;
        size_t r = uint64_t(numElems) * (t + 1) / numWorkers;
        size_t i = l;
#ifdef CLASSIFY_UNROLL        
        for (; i + CLASSIFY_UNROLL <= r; i += CLASSIFY_UNROLL) {
            size_t bidx[CLASSIFY_UNROLL];
            pivot.classifyBlock<CLASSIFY_UNROLL>(&srcElems[i], bidx);
            for (size_t q = 0; q < CLASSIFY_UNROLL; q++) {
                size_t b = bidx[q];
                bucketOf[i + q] = b;
                localHisto[t][b]++;
            }
        }
#endif        
        for (; i < r; i++) {
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
    
    parallelWorkers(numWorkers, [&](size_t t) {
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

void processBase(const TaskData &task) {
    Span<Value> srcElems = task.shared_->elemsSpans_[task.world_].subspan(task.first_, task.len_);
    smallSort(srcElems);

    if (task.world_ == 0)
        return;

    Span<Value> dstElems = task.shared_->elemsSpans_[task.world_ ^ 1].subspan(task.first_, task.len_);
    for (size_t i = 0; i < srcElems.size(); i++)
        dstElems[i] = srcElems[i];
}

void processRecursive(TaskData task) {
    SharedData *shared = task.shared_;
    //ThreadData *perThread = &shared->perThread_.local();
    ThreadData ptd;
    ThreadData *perThread = &ptd;

    Span<Value> srcElems = shared->elemsSpans_[task.world_].subspan(task.first_, task.len_);
    Span<Value> dstElems = shared->elemsSpans_[task.world_ ^ 1].subspan(task.first_, task.len_);
    Span<uint8_t> bucketOf = makeSpan(shared->bucketIndexStore_).subspan(task.first_, task.len_);

    assert(task.len_ > SMALLSORT_MAX);
    size_t logn = log2up(task.len_);

    size_t numBuckets;
    if (task.len_ <= 4096) {
        // one level: aim for 16 elements per bucket
        size_t logb = logn - 4;
        numBuckets = 1 << logb;
    }
    else {
        // two levels: aim for 16 elements per bucket
        size_t logb = (logn - 4) / 2;
        numBuckets = 1 << logb;
        numBuckets = std::min<size_t>(numBuckets, 256);
    }

    perThread->pivot_.select(srcElems, numBuckets, task.random_, perThread->samplesStore_);

    perThread->splitsStore_.resize(numBuckets + 1);
    multiPartition(
        srcElems, perThread->pivot_,
        dstElems, makeSpan(perThread->splitsStore_),
        task.numWorkers_, bucketOf
    );

    TaskData subTask;
    subTask.shared_ = task.shared_;
    subTask.world_ = task.world_ ^ 1;
    subTask.numWorkers_ = (task.numWorkers_ + numBuckets - 1) / numBuckets;
    for (size_t t = 0; t < numBuckets; t++) {
        subTask.first_ = task.first_ + perThread->splitsStore_[t];
        subTask.len_ = perThread->splitsStore_[t + 1] - perThread->splitsStore_[t];
        if (subTask.len_ <= SMALLSORT_MAX) {
            processBase(subTask);
            continue;
        }
        if (t == 0)
            subTask.random_ = task.random_;
        else
            subTask.random_.initStream(task.shared_->randomSeed_, subTask.first_);
        shared->taskGroup_.run([subTask] {
            processRecursive(subTask);
        });
    }
}

void sort(Value *begin, size_t num) {
    if (num < SMALLSORT_MAX) {
        smallSort({begin, num});
        return;
    }

    SharedData shared;
    shared.elemsCopyStore_.resize(num);
    shared.bucketIndexStore_.resize(num);
    shared.numElems_ = num;
    shared.elemsSpans_[0] = {begin, num};
    shared.elemsSpans_[1] = makeSpan(shared.elemsCopyStore_);

    TaskData task;
    task.shared_ = &shared;
    task.first_ = 0;
    task.len_ = shared.numElems_;
    task.numWorkers_ = tbb::this_task_arena::max_concurrency() * 3;
    task.world_ = 0;
    task.random_.initStream(shared.randomSeed_, task.first_);

    shared.taskGroup_.run_and_wait([&task] {
        processRecursive(task);
    });
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
