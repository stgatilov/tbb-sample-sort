#include <stdio.h>
#include <assert.h>
#include <span>
#include <chrono>
#include <type_traits>

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/parallel_sort.h>

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
#define COLLECT_STATS 1

#define CLASSIFY_UNROLL 8
#define SMALLSORT_MAX 32
#define UNCACHED_MININPUT_BYTES (1 << 20)
#define UNCACHED_BUFFER_BYTES (1 << 10)

#define FORCEINLINE __attribute__((always_inline)) inline
#define NOINLINE __attribute__((noinline))

FORCEINLINE bool isPot(size_t x) {
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
//   lifetime methods must not throw...
template<class T> struct ValueTraits {
    static FORCEINLINE void relocateOne(T &dst, T &src) noexcept {
        new(&dst) T(static_cast<T&&>(src));
        src.~T();
    }
    static FORCEINLINE void destroyOne(T &dst) noexcept {
        dst.~T();
    }
    static FORCEINLINE void constructCopyOne(T &dst, const T &src) noexcept {
        new(&dst) T(src);
    }
    static FORCEINLINE void swapOne(T &dst, T &src) noexcept {
        std::swap(dst, src);
    }
    static FORCEINLINE void constructDefaultOne(T &dst) noexcept {
        static_assert(std::is_integral_v<T>); // never used for elements
        new(&dst) T;
    }

    static NOINLINE void relocateMany(T *dst, T *src, size_t n) noexcept {
        assert(dst >= src + n || src >= dst + n);
        for (size_t i = 0; i < n; i++)
            relocateOne(dst[i], src[i]);
    }
    static NOINLINE void relocateManyUncached(T *dst, T *src, size_t n) noexcept {
        assert(dst >= src + n || src >= dst + n);
        char *bdst = reinterpret_cast<char*>(dst);
        const char *bsrc = reinterpret_cast<const char*>(src);
        size_t bn = n * sizeof(T);
        size_t prefix = std::min(64 - size_t(bdst) % 64, bn);
        memcpy(bdst, bsrc, prefix);
        bdst += prefix;
        bsrc += prefix;
        bn -= prefix;
        while (bn >= 64) {
            _mm_stream_si128((__m128i*)(bdst + 0), _mm_loadu_si128((__m128i*)(bsrc + 0)));
            _mm_stream_si128((__m128i*)(bdst + 16), _mm_loadu_si128((__m128i*)(bsrc + 16)));
            _mm_stream_si128((__m128i*)(bdst + 32), _mm_loadu_si128((__m128i*)(bsrc + 32)));
            _mm_stream_si128((__m128i*)(bdst + 48), _mm_loadu_si128((__m128i*)(bsrc + 48)));
            bdst += 64;
            bsrc += 64;
            bn -= 64;
        }
        memcpy(bdst, bsrc, bn);
    }
    static NOINLINE void destroyMany(T *dst, size_t n) noexcept {
        for (size_t i = 0; i < n; i++)
            destroyOne(dst[i]);
    }
    static NOINLINE void constructDefaultMany(T *dst, size_t n) noexcept {
        for (size_t i = 0; i < n; i++)
            constructDefaultOne(dst[i]);
    }
};

template<class T> struct Allocator {
    static T *allocate(size_t n) {
        if (!n)
            return nullptr;
        return (T*) operator new[] (n * sizeof(T), std::align_val_t(alignof(T)));
    }
    static void deallocate(T *ptr) {
        if (!ptr)
            return;
        operator delete[] (ptr);
    }
};

// because I need this to work on C++17
template<class T> class Span {
    T *ptr_ = nullptr;
    size_t num_ = 0;

public:
    FORCEINLINE Span() = default;
    FORCEINLINE Span(T *ptr, size_t num)
        : ptr_(ptr)
        , num_(num)
    {}
    template<class U, std::enable_if_t<std::is_same_v<U, std::add_const_t<T>>, int> = 0>
    FORCEINLINE operator Span<U>() const {
        return {ptr_, num_};
    }

    FORCEINLINE T *data() const { return ptr_; }
    FORCEINLINE size_t size() const { return num_; }

    FORCEINLINE T &operator[] (size_t i) const {
        assert(i < num_);
        return ptr_[i];
    }

    FORCEINLINE Span<T> subspan(size_t start, size_t len) const {
        assert(start + len <= num_);
        return {ptr_ + start, len};
    }
};

template<class T> class Array {
    T *ptr_ = nullptr;
    size_t num_ = 0;
    size_t cap_ = 0;

    NOINLINE void grow(size_t n) {
        assert(n > cap_);
        assert(num_ == 0);
        n = std::max(n, 2 * cap_ + 1);
        Allocator<T>::deallocate(ptr_);
        ptr_ = Allocator<T>::allocate(n);
        cap_ = n;
    }

public:
    ~Array() {
        clear();
        Allocator<T>::deallocate(ptr_);
    }
    Array() = default;

    void clear() {
        ValueTraits<T>::destroyMany(ptr_, num_);
        num_ = 0;
    }

    void clearReserve(size_t n) {
        clear();
        if (n > cap_)
            grow(n);
    }

    void clearResize(size_t n) {
        clearReserve(n);
        num_ = n;
        ValueTraits<T>::constructDefaultMany(ptr_, num_);
    }

    FORCEINLINE void pushBack(const T& x) {
        assert(num_ < cap_);
        ValueTraits<T>::constructCopyOne(ptr_[num_++], x);
    }

    FORCEINLINE T *data() { return ptr_; }
    FORCEINLINE const T *data() const { return ptr_; }
    FORCEINLINE size_t size() const { return num_; }

    FORCEINLINE T &operator[] (size_t i) {
        assert(i < num_);
        return ptr_[i];
    }
    FORCEINLINE const T &operator[] (size_t i) const {
        assert(i < num_);
        return ptr_[i];
    }

    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
    Array(const Array&&) = delete;
    Array& operator=(const Array&&) = delete;
};

template<class T> inline Span<T> makeSpan(Array<T> &arr) { return {arr.data(), arr.size()}; }
template<class T> inline Span<const T> makeSpan(const Array<T> &arr) { return {arr.data(), arr.size()}; }


typedef uint64_t Value;

void smallSort(Span<Value> arr) {
#if 0
    std::sort(arr.data(), arr.data() + arr.size());
#else
    for (size_t i = 1; i < arr.size(); i++)
        for (size_t j = 0; j < i; j++)
            if (arr[i] < arr[j])
                ValueTraits<Value>::swapOne(arr[i], arr[j]);

    assert(std::is_sorted(arr.data(), arr.data() + arr.size()));                
#endif
}

void quickSort(Span<Value> arr, Random &random) {
#if 0
    std::sort(arr.data(), arr.data() + arr.size());
#else
    if (arr.size() <= SMALLSORT_MAX)
        return smallSort(arr);

    size_t idxA = random.generate(arr.size());
    size_t idxB = random.generate(arr.size());
    size_t idxC = random.generate(arr.size());
    if (arr[idxB] < arr[idxA]) ValueTraits<Value>::swapOne(arr[idxA], arr[idxB]);
    if (arr[idxC] < arr[idxA]) ValueTraits<Value>::swapOne(arr[idxA], arr[idxC]);
    if (arr[idxC] < arr[idxB]) ValueTraits<Value>::swapOne(arr[idxB], arr[idxC]);

    Value pivot = arr[idxB];    // constructCopyOne

    ptrdiff_t l = -1;
    ptrdiff_t r = arr.size();
    while (1) {
        do { l++; } while (arr[l] < pivot);
        do { r--; } while (pivot < arr[r]);
        if (l >= r) break;
        ValueTraits<Value>::swapOne(arr[l], arr[r]);
    }
 
    if (r < arr.size() - 1)
        r++;

    assert(r > 0 && r < arr.size());
    quickSort(arr.subspan(0, r), random);
    quickSort(arr.subspan(r, arr.size() - r), random);

    assert(std::is_sorted(arr.data(), arr.data() + arr.size()));
#endif
}

struct MultiPivot {
    size_t numBits_ = 0;
    size_t numBuckets_ = 0;
    Array<Value> sorted_;
    Array<Value> tree_;

    void select(
        Span<const Value> arr, size_t numBuckets,
        Random &random, Array<Value> &samples
    ) {
        assert(isPot(numBuckets) && numBuckets >= 2);
        size_t numElems = arr.size();

        size_t numSamples = numBuckets * log2up(numElems) / 5;
        numSamples = std::max(numSamples, numBuckets);

        samples.clearReserve(numSamples);
        for (size_t i = 0; i < numSamples; i++) {
            size_t index = random.generate(numElems);
            samples.pushBack(arr[index]);
        }

        quickSort(makeSpan(samples), random);

        sorted_.clearReserve(numBuckets - 1);
        for (size_t i = 1; i <= numBuckets - 1; i++) {
            uint64_t pos = uint64_t(numSamples) * i / numBuckets;
            sorted_.pushBack(samples[pos]);
        }

        size_t numBits = log2up(numBuckets);

        tree_.clearReserve(numBuckets - 1);
        for (int b = numBits - 1; b >= 0; b--) {
            size_t len = (1 << b);
            for (size_t i = len - 1; i < numBuckets; i += len * 2)
                tree_.pushBack(sorted_[i]);
        }

        numBuckets_ = numBuckets;
        numBits_ = numBits;        
    }

    FORCEINLINE size_t classifyOne(const Value &value) const {
        size_t res = 0;
        Span<const Value> tree = makeSpan(tree_);
        for (size_t b = 0; b < numBits_; b++) {
            bool isLess = (value < tree[res]);
            res = 2 * res + 1 + size_t(!isLess);
        }

        res -= (numBuckets_ - 1);
        assert(res == numBuckets_ - 1 || (value < sorted_[res]));
        assert(res == 0 || !(value < sorted_[res - 1]));

        res -= (res > 0 && value == sorted_[res - 1]);
        assert(res < numBuckets_);
        return res;
    }

    template<size_t N> FORCEINLINE void classifyBlock(const Value *value, size_t *res) const {
        Span<const Value> tree = makeSpan(tree_);
        for (size_t i = 0; i < N; i++)
            res[i] = 0;
        for (size_t b = 0; b < numBits_; b++) {
            for (size_t i = 0; i < N; i++) {
                bool isLess = (value[i] < tree_[res[i]]);
                res[i] = 2 * res[i] + 1 + size_t(!isLess);
            }
        }
        for (size_t i = 0; i < N; i++) {
            res[i] -= (numBuckets_ - 1);
            res[i] -= (res[i] > 0 && value[i] == sorted_[res[i] - 1]);
        }
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
    std::unique_ptr<Value[], decltype(&Allocator<Value>::deallocate)> elemsCopyStore_{nullptr, Allocator<Value>::deallocate};

    Array<uint8_t> bucketIndexStore_;

    uint64_t randomSeed_ = 0xDEADBEEF01234567ull;
    tbb::task_group taskGroup_;
#if COLLECT_STATS
    std::atomic_int sizeStats_[64];
#endif
};

void multiPartition(
    Span<Value> srcElems, const MultiPivot &pivot,
    Span<Value> dstElems, Array<size_t> &splits, 
    size_t numWorkers, Span<uint8_t> bucketOf
) {
    size_t numBuckets = pivot.numBuckets_;
    size_t numElems = srcElems.size();
    
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
    
    bool scatterSimple = true;
#ifdef UNCACHED_MININPUT_BYTES
    if (numElems * sizeof(Value) > UNCACHED_MININPUT_BYTES) {
        scatterSimple = false;
        parallelWorkers(numWorkers, [&](size_t t) {
            size_t l = uint64_t(numElems) * (t + 0) / numWorkers;
            size_t r = uint64_t(numElems) * (t + 1) / numWorkers;

            alignas(Value) char buffer[256][UNCACHED_BUFFER_BYTES];
            static_assert(sizeof(Value) <= UNCACHED_BUFFER_BYTES);
            
            uint32_t bufCnt[256];
            for (size_t b = 0; b < numBuckets; b++)
                bufCnt[b] = 0;

            for (size_t i = l; i < r; i++) {
                size_t b = bucketOf[i];
                ValueTraits<Value>::relocateOne(((Value*)buffer[b])[bufCnt[b]++], srcElems[i]);
                if (bufCnt[b] == UNCACHED_BUFFER_BYTES / sizeof(Value)) {
                    size_t &pos = localHisto[t][b];
                    ValueTraits<Value>::relocateManyUncached(&dstElems[pos], ((Value*)buffer[b]), bufCnt[b]);
                    pos += bufCnt[b];
                    bufCnt[b] = 0;
                }
            }

            for (size_t b = 0; b < numBuckets; b++) {
                if (bufCnt[b] == 0)
                    continue;
                size_t &pos = localHisto[t][b];
                ValueTraits<Value>::relocateMany(&dstElems[pos], ((Value*)buffer[b]), bufCnt[b]);
                pos += bufCnt[b];
            }
        });
    }
#endif
    if (scatterSimple) {
        parallelWorkers(numWorkers, [&](size_t t) {
            size_t l = uint64_t(numElems) * (t + 0) / numWorkers;
            size_t r = uint64_t(numElems) * (t + 1) / numWorkers;
            for (size_t i = l; i < r; i++) {
                size_t b = bucketOf[i];
                size_t &pos = localHisto[t][b];
                ValueTraits<Value>::relocateOne(dstElems[pos++], srcElems[i]);
            }
        });
    }

    splits.clearReserve(numBuckets + 1);
    for (size_t b = 0; b <= numBuckets; b++)
        splits.pushBack(globalHisto[b]);

#if SLOW_ASSERT
    assert(splits[0] == 0 && splits[numBuckets] == numElems);
    for (size_t b = 0; b < numBuckets; b++) {
        assert(splits[b] <= splits[b + 1]);
        for (size_t i = splits[b]; i < splits[b + 1]; i++) {
            assert(b == 0 || dstElems[i] >= pivot.sorted_[b - 1]);
            assert(b == numBuckets - 1 || dstElems[i] <= pivot.sorted_[b]);
        }
    }
#endif    
}

void copyBack(const TaskData &task) {
    if (task.world_ == 0)
        return;

    Span<Value> srcElems = task.shared_->elemsSpans_[task.world_].subspan(task.first_, task.len_);
    Span<Value> dstElems = task.shared_->elemsSpans_[task.world_ ^ 1].subspan(task.first_, task.len_);
    ValueTraits<Value>::relocateMany(dstElems.data(), srcElems.data(), srcElems.size());
}

void processBase(const TaskData &task) {
    Span<Value> srcElems = task.shared_->elemsSpans_[task.world_].subspan(task.first_, task.len_);
    smallSort(srcElems);

    copyBack(task);
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
#if COLLECT_STATS
    shared->sizeStats_[logn].fetch_add(1, std::memory_order_relaxed);
#endif

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

    multiPartition(
        srcElems, perThread->pivot_,
        dstElems, perThread->splitsStore_,
        task.numWorkers_, bucketOf
    );

    TaskData subTask;
    Span<const size_t> splits = makeSpan(perThread->splitsStore_);
    subTask.shared_ = task.shared_;
    subTask.world_ = task.world_ ^ 1;
    subTask.numWorkers_ = (task.numWorkers_ + numBuckets - 1) / numBuckets;
    for (size_t b = 0; b < numBuckets; b++) {
        subTask.first_ = task.first_ + splits[b];
        subTask.len_ = splits[b + 1] - splits[b];
        if (subTask.len_ == 0)
            continue;
        if (b > 0 && b < numBuckets - 1 && perThread->pivot_.sorted_[b - 1] == perThread->pivot_.sorted_[b]) {
            for (size_t i = splits[b]; i < splits[b + 1]; i++)
                assert(dstElems[i] == perThread->pivot_.sorted_[b]);
            copyBack(subTask);
            continue;
        }
        if (subTask.len_ <= SMALLSORT_MAX) {
#if COLLECT_STATS
            shared->sizeStats_[log2up(subTask.len_)].fetch_add(1, std::memory_order_relaxed);
#endif
            processBase(subTask);
            continue;
        }
        if (b == 0)
            subTask.random_ = task.random_;
        else
            subTask.random_.initStream(task.shared_->randomSeed_, subTask.first_);
        shared->taskGroup_.run([subTask] {
            processRecursive(subTask);
        });
    }
}

void sampleSort(Value *begin, size_t num) {
    if (num < SMALLSORT_MAX) {
        smallSort({begin, num});
        return;
    }

    SharedData shared;
    shared.elemsCopyStore_.reset(Allocator<Value>::allocate(num));
    shared.bucketIndexStore_.clearResize(num);
    shared.numElems_ = num;
    shared.elemsSpans_[0] = {begin, num};
    shared.elemsSpans_[1] = {shared.elemsCopyStore_.get(), num};

    TaskData task;
    task.shared_ = &shared;
    task.first_ = 0;
    task.len_ = shared.numElems_;
    task.numWorkers_ = tbb::this_task_arena::max_concurrency();
    task.world_ = 0;
    task.random_.initStream(shared.randomSeed_, task.first_);

    shared.taskGroup_.run_and_wait([&task] {
        processRecursive(task);
    });

#if COLLECT_STATS
    for (int i = 1; i < 32; i++)
        printf("L%02d: %7d [%zu..%zu)\n", i, shared.sizeStats_[i].load(), size_t(1) << (i-1), size_t(1) << i);
#endif        
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

#if 1
    sampleSort(input.data(), input.size());
#else
    tbb::parallel_sort(input.data(), input.data() + input.size());
#endif
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
