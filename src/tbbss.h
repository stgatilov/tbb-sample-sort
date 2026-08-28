#include <cstddef>
#include <cstring>
#include <cassert>

#include <memory>
#include <algorithm>
#include <type_traits>

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_group.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/enumerable_thread_specific.h>

#include "pcg_basic.h"


#define TBBSS_CLASSIFY_UNROLL 8
#define TBBSS_SMALLSORT_MAX 32
#define TBBSS_UNCACHED_MININPUT_BYTES (1 << 20)
#define TBBSS_UNCACHED_BUFFER_BYTES (1 << 10)

#ifdef _MSC_VER
    #define TBBSS_FORCEINLINE __forceinline
    #define TBBSS_NOINLINE __declspec(noinline)
#else
    #define TBBSS_FORCEINLINE __attribute__((always_inline)) inline
    #define TBBSS_NOINLINE __attribute__((noinline))
#endif

//#define TBBSS_SLOW_ASSERT 1
//#define TBBSS_COLLECT_STATS 1


#if TBBSS_COLLECT_STATS
    #include <atomic>
#endif

namespace tbbss {

//---------------------------------------------------------

TBBSS_FORCEINLINE bool isPot(size_t x) {
    return (x & (x - 1)) == 0;
}

inline size_t log2up(size_t x) {
    size_t res = 0;
    while ((1 << res) < x)
        res++;
    return res;
}

template<class Lambda> void parallelWorkers(size_t numWorkers, Lambda&& lambda) {
    assert(numWorkers > 0);
    if (numWorkers == 1) {
        lambda(0);
    }
    else {
        // isolation ensures per-thread data is not overwritten
        tbb::this_task_arena::isolate([&]{
            tbb::parallel_for<size_t>(0, numWorkers, lambda);
        });
    }
}

//---------------------------------------------------------

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

//---------------------------------------------------------

void memcpyUncached(char *dst, const char *src, size_t size);

template<class T> TBBSS_FORCEINLINE void memcpyElement(void *dst, const void *src) {
#if 0
    memcpy(dst, src, sizeof(T));
#else
    // note: unaligned access may happen here
    for (size_t i = 0; i < sizeof(T) / 8; i++)
        reinterpret_cast<uint64_t*>(dst)[i] = reinterpret_cast<const uint64_t*>(src)[i];

    if constexpr (sizeof(T) % 8 >= 4) {
        static constexpr size_t Index = 2 * (sizeof(T) / 8);
        reinterpret_cast<uint32_t*>(dst)[Index] = reinterpret_cast<const uint32_t*>(src)[Index];
    }
    if constexpr (sizeof(T) % 4 >= 2) {
        static constexpr size_t Index = 2 * (sizeof(T) / 4);
        reinterpret_cast<uint16_t*>(dst)[Index] = reinterpret_cast<const uint16_t*>(src)[Index];
    }
    if constexpr (sizeof(T) % 2 >= 1) {
        static constexpr size_t Index = 2 * (sizeof(T) / 2);
        reinterpret_cast<uint8_t*>(dst)[Index] = reinterpret_cast<const uint8_t*>(src)[Index];
    }
#endif
}

template<class T> TBBSS_FORCEINLINE void memswapElementConditional(void *dst, void *src, bool doSwap) {
    int64_t mask = -int64_t(doSwap);

    // note: unaligned access may happen here
    for (size_t i = 0; i < sizeof(T) / 8; i++) {
        uint64_t &a = reinterpret_cast<uint64_t*>(dst)[i];
        uint64_t &b = reinterpret_cast<uint64_t*>(src)[i];
        uint64_t delta = (a ^ b) & uint64_t(mask);
        a ^= delta;
        b ^= delta;
    }

    if constexpr (sizeof(T) % 8 >= 4) {
        static constexpr size_t Index = 2 * (sizeof(T) / 8);
        uint32_t &a = reinterpret_cast<uint32_t*>(dst)[Index];
        uint32_t &b = reinterpret_cast<uint32_t*>(src)[Index];
        uint32_t delta = (a ^ b) & uint32_t(mask);
        a ^= delta;
        b ^= delta;
    }
    if constexpr (sizeof(T) % 4 >= 2) {
        static constexpr size_t Index = 2 * (sizeof(T) / 4);
        uint16_t &a = reinterpret_cast<uint16_t*>(dst)[Index];
        uint16_t &b = reinterpret_cast<uint16_t*>(src)[Index];
        uint16_t delta = (a ^ b) & uint16_t(mask);
        a ^= delta;
        b ^= delta;

    }
    if constexpr (sizeof(T) % 2 >= 1) {
        static constexpr size_t Index = 2 * (sizeof(T) / 2);
        uint8_t &a = reinterpret_cast<uint8_t*>(dst)[Index];
        uint8_t &b = reinterpret_cast<uint8_t*>(src)[Index];
        uint8_t delta = (a ^ b) & uint8_t(mask);
        a ^= delta;
        b ^= delta;
    }
}

// "relocate" is Rust-style move
// it moves an element into specified raw memory and makes source memory raw
enum RelocationTrivialness {
    // relocation is done via destructing move: move construct + destroy old
    // BEWARE: the lifetime methods used must not throw exceptions!
    // must be used for types that contain self-references
    // e.g. std::string with small string optimization
    rtNone = 0,
    // relocation can be done by memcpy instead of destructing move
    // is automatically used for trivially copyable types
    // but it can also be used for std::unique_ptr, std::vector, std::set etc.
    rtRelocate = 1,
    // temporary copy of an element can be created using memcpy
    // such copies are used only to run comparator, and they are NOT destroyed
    // is automatically used for trivially copyable types
    // but it can also be used for std::unique_ptr, std::vector, std::set etc.
    rtFork = 2,
};
template<class T> constexpr RelocationTrivialness IsTriviallyRelocatable = (std::is_trivially_copyable_v<T> ? rtFork : rtNone);

template<class T, RelocationTrivialness IsRelocationTrivial = IsTriviallyRelocatable<T>>
struct DefaultValueTraits {
    static TBBSS_FORCEINLINE void relocateOne(T &dst, T &src) noexcept {
        if constexpr (IsRelocationTrivial == rtNone) {
            new(&dst) T(static_cast<T&&>(src));
            src.~T();
        }
        else {
            memcpyElement<T>(&dst, &src);
        }
    }
    static TBBSS_FORCEINLINE void swapOne(T &dst, T &src) noexcept {
        if constexpr (IsRelocationTrivial == rtNone) {
            T temp = static_cast<T&&>(dst);
            dst = static_cast<T&&>(src);
            src = static_cast<T&&>(temp);
        }
        else {
            alignas(T) char temp[sizeof(T)];
            memcpyElement<T>(&temp, &dst);
            memcpyElement<T>(&dst, &src);
            memcpyElement<T>(&src, &temp);
        }
    }

    static TBBSS_FORCEINLINE void swapConditionalOne(T &dst, T &src, bool doSwap) noexcept {
        if constexpr (IsRelocationTrivial == rtNone) {
            if (doSwap)
                swapOne(dst, src);
        }
        else {
            memswapElementConditional<T>(&dst, &src, doSwap);
        }
    }

    static TBBSS_FORCEINLINE void constructCopyOne(T &dst, const T &src) noexcept {
        if constexpr (IsRelocationTrivial != rtFork) {
            new(&dst) T(src);
        }
        else {
            // make temporary shallow clone, which shares same internal resources
            memcpyElement<T>(&dst, &src);
        }
    }
    static TBBSS_FORCEINLINE void destroyOne(T &dst) noexcept {
        if constexpr (IsRelocationTrivial != rtFork) {
            dst.~T();
        }
        else {
            // only called for destroying shallow clones, so don't call destructor
        }
    }

    static void relocateMany(T *dst, T *src, size_t n) noexcept {
        assert(dst >= src + n || src >= dst + n);
        for (size_t i = 0; i < n; i++)
            relocateOne(dst[i], src[i]);
    }
    static void relocateManyUncached(T *dst, T *src, size_t n) noexcept {
        if constexpr (IsRelocationTrivial == rtNone)
            return relocateMany(dst, src, n);
        assert(dst >= src + n || src >= dst + n);
        memcpyUncached(reinterpret_cast<char*>(dst), reinterpret_cast<const char*>(src), n * sizeof(T));
    }
    static void destroyMany(T *dst, size_t n) noexcept {
        for (size_t i = 0; i < n; i++)
            destroyOne(dst[i]);
    }
};

template<class T>
void constructDefaultMany(T *dst, size_t n) noexcept {
    // only used for integers internally, never used for sorted user elements
    static_assert(std::is_integral_v<T>);
    for (size_t i = 0; i < n; i++)
        new(&dst[i]) T;
}

//---------------------------------------------------------

template<class T> struct Allocator {
    static T *allocate(size_t n) {
        if (!n)
            return nullptr;
        static constexpr size_t Alignment = std::max<size_t>(alignof(T), 64);
        return (T*) operator new[] (n * sizeof(T), std::align_val_t(Alignment));
    }
    static void deallocate(T *ptr) {
        if (!ptr)
            return;
        operator delete[] (ptr);
    }
};

//---------------------------------------------------------

template<class T, class ValueTraits> class Array {
    T *ptr_ = nullptr;
    size_t num_ = 0;
    size_t cap_ = 0;

    TBBSS_NOINLINE void grow(size_t n) {
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
        ValueTraits::destroyMany(ptr_, num_);
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
        constructDefaultMany(ptr_, num_);
    }

    void clearZero(size_t n) {
        static_assert(std::is_integral_v<T>); // never used for elements
        clearResize(n);
        for (size_t i = 0; i < num_; i++)
            ptr_[i] = 0;
    }

    TBBSS_FORCEINLINE void pushBack(const T& x) {
        assert(num_ < cap_);
        ValueTraits::constructCopyOne(ptr_[num_++], x);
    }

    TBBSS_FORCEINLINE T *data() { return ptr_; }
    TBBSS_FORCEINLINE const T *data() const { return ptr_; }
    TBBSS_FORCEINLINE size_t size() const { return num_; }

    TBBSS_FORCEINLINE T &operator[] (size_t i) {
        assert(i < num_);
        return ptr_[i];
    }
    TBBSS_FORCEINLINE const T &operator[] (size_t i) const {
        assert(i < num_);
        return ptr_[i];
    }

    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
    Array(const Array&&) = delete;
    Array& operator=(const Array&&) = delete;
};

//---------------------------------------------------------

template<class T> class Span {
    T *ptr_ = nullptr;
    size_t num_ = 0;

public:
    TBBSS_FORCEINLINE Span() = default;
    TBBSS_FORCEINLINE Span(T *ptr, size_t num)
        : ptr_(ptr)
        , num_(num)
    {}
    template<class U, std::enable_if_t<std::is_same_v<U, std::add_const_t<T>>, int> = 0>
    TBBSS_FORCEINLINE operator Span<U>() const {
        return {ptr_, num_};
    }

    TBBSS_FORCEINLINE T *data() const { return ptr_; }
    TBBSS_FORCEINLINE size_t size() const { return num_; }

    TBBSS_FORCEINLINE T &operator[] (size_t i) const {
        assert(i < num_);
        return ptr_[i];
    }

    TBBSS_FORCEINLINE Span<T> subspan(size_t start, size_t len) const {
        assert(start + len <= num_);
        return {ptr_ + start, len};
    }
};

template<class T, class ValueTraits>
inline Span<T> makeSpan(Array<T, ValueTraits> &arr) { return {arr.data(), arr.size()}; }
template<class T, class ValueTraits>
inline Span<const T> makeSpan(const Array<T, ValueTraits> &arr) { return {arr.data(), arr.size()}; }

//---------------------------------------------------------

template<class Value, class Comp, class ValueTraits> void smallSort(Span<Value> arr, const Comp &comp) {
#if 0
    std::sort(arr.data(), arr.data() + arr.size(), comp);
#else
    alignas(Value) char buffer[sizeof(Value)];
    for (size_t i = 1; i < arr.size(); i++) {
        Value &arrI = *reinterpret_cast<Value*>(buffer);
        ValueTraits::relocateOne(arrI, arr[i]);
        for (size_t j = 0; j < i; j++)
            ValueTraits::swapConditionalOne(arrI, arr[j], comp(arrI, arr[j]));
        ValueTraits::relocateOne(arr[i], arrI);
    }

    assert(std::is_sorted(arr.data(), arr.data() + arr.size(), comp));                
#endif
}

//---------------------------------------------------------

template<class Value, class ValueTraits>
struct MultiPivot {
    size_t numBits_ = 0;
    size_t numBuckets_ = 0;
    Array<Value, ValueTraits> sorted_;
    Array<Value, ValueTraits> tree_;

    static size_t selectSamples(Span<Value> arr, size_t numBuckets, Random &random) {
        assert(isPot(numBuckets) && numBuckets >= 2);
        size_t numElems = arr.size();

        size_t numSamples = numBuckets * log2up(numElems) / 5;
        numSamples = std::max(numSamples, numBuckets);

        for (size_t i = 0; i < numSamples; i++) {
            size_t index = i + random.generate(numElems - i);
            ValueTraits::swapOne(arr[i], arr[index]);
        }

        return numSamples;
    }

    void initFromSortedSamples(Span<Value> samples, size_t numBuckets) {
        size_t numSamples = samples.size();

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

    template<class Comp>
    TBBSS_FORCEINLINE size_t classifyOne(const Value &value, const Comp &comp) const {
        size_t res = 0;
        Span<const Value> tree = makeSpan(tree_);
        for (size_t b = 0; b < numBits_; b++) {
            bool isLess = comp(value, tree[res]);
            res = 2 * res + 1 + size_t(!isLess);
        }

        res -= (numBuckets_ - 1);
        assert(res == numBuckets_ - 1 || (value < sorted_[res]));
        assert(res == 0 || !(value < sorted_[res - 1]));

        res -= (res > 0 && value == sorted_[res - 1]);
        assert(res < numBuckets_);
        return res;
    }

    template<size_t N, class Comp>
    TBBSS_FORCEINLINE void classifyBlock(const Value *value, size_t *res, const Comp &comp) const {
        Span<const Value> tree = makeSpan(tree_);
        for (size_t i = 0; i < N; i++)
            res[i] = 0;
        for (size_t b = 0; b < numBits_; b++) {
            for (size_t i = 0; i < N; i++) {
                bool isLess = comp(value[i], tree_[res[i]]);
                res[i] = 2 * res[i] + 1 + size_t(!isLess);
            }
        }
        for (size_t i = 0; i < N; i++) {
            res[i] -= (numBuckets_ - 1);
            res[i] -= (res[i] > 0 && value[i] == sorted_[res[i] - 1]);
        }
    }
};

struct PartitionResult {
    Array<size_t, DefaultValueTraits<size_t>> localHisto_;
    Array<size_t, DefaultValueTraits<size_t>> splits_;
};

template<class Value, class ValueTraits, class Comp>
void multiPartition(
    Span<Value> srcElems, const MultiPivot<Value, ValueTraits> &pivot, const Comp &comp,
    Span<Value> dstElems, PartitionResult &result,
    size_t numWorkers, Span<uint8_t> bucketOf
) {
    size_t numBuckets = pivot.numBuckets_;
    size_t numElems = srcElems.size();
    
    result.localHisto_.clearZero(numWorkers * numBuckets);
    Span<size_t> localHisto = makeSpan(result.localHisto_);

    parallelWorkers(numWorkers, [&pivot, numElems, numBuckets, numWorkers, srcElems, bucketOf, localHisto, &comp](size_t t) {
        size_t l = uint64_t(numElems) * (t + 0) / numWorkers;
        size_t r = uint64_t(numElems) * (t + 1) / numWorkers;
        size_t i = l;
#ifdef TBBSS_CLASSIFY_UNROLL
        for (; i + TBBSS_CLASSIFY_UNROLL <= r; i += TBBSS_CLASSIFY_UNROLL) {
            size_t bidx[TBBSS_CLASSIFY_UNROLL];
            pivot.template classifyBlock<TBBSS_CLASSIFY_UNROLL>(&srcElems[i], bidx, comp);
            for (size_t q = 0; q < TBBSS_CLASSIFY_UNROLL; q++) {
                size_t b = bidx[q];
                bucketOf[i + q] = b;
                localHisto[t * numBuckets + b]++;
            }
        }
#endif        
        for (; i < r; i++) {
            size_t b = pivot.classifyOne(srcElems[i], comp);
            bucketOf[i] = b;
            localHisto[t * numBuckets + b]++;
        }
    });

    result.splits_.clearZero(numBuckets + 1);
    Span<size_t> globalHisto = makeSpan(result.splits_);

    for (size_t t = 0; t < numWorkers; t++)
        for (size_t b = 0; b < numBuckets; b++)
            globalHisto[b + 1] += localHisto[t * numBuckets + b];

    for (size_t b = 0; b < numBuckets; b++)
        globalHisto[b + 1] += globalHisto[b];
    assert(globalHisto[numBuckets] == numElems);

    for (size_t b = 0; b < numBuckets; b++) {
        size_t tsum = globalHisto[b];
        for (size_t t = 0; t < numWorkers; t++) {
            size_t nsum = tsum + localHisto[t * numBuckets + b];
            localHisto[t * numBuckets + b] = tsum;
            tsum = nsum;
        }
        assert(tsum == globalHisto[b + 1]);
    }
    
    bool scatterSimple = true;
#ifdef TBBSS_UNCACHED_MININPUT_BYTES
    if (numElems * sizeof(Value) > TBBSS_UNCACHED_MININPUT_BYTES) {
        scatterSimple = false;
        parallelWorkers(numWorkers, [numElems, numBuckets, numWorkers, srcElems, dstElems, localHisto, bucketOf](size_t t) {
            size_t l = uint64_t(numElems) * (t + 0) / numWorkers;
            size_t r = uint64_t(numElems) * (t + 1) / numWorkers;

            alignas(Value) char buffer[256][TBBSS_UNCACHED_BUFFER_BYTES];
            static_assert(sizeof(Value) <= TBBSS_UNCACHED_BUFFER_BYTES);
            
            uint32_t bufCnt[256];
            for (size_t b = 0; b < numBuckets; b++)
                bufCnt[b] = 0;

            for (size_t i = l; i < r; i++) {
                size_t b = bucketOf[i];
                ValueTraits::relocateOne(((Value*)buffer[b])[bufCnt[b]++], srcElems[i]);
                if (bufCnt[b] == TBBSS_UNCACHED_BUFFER_BYTES / sizeof(Value)) {
                    size_t &pos = localHisto[t * numBuckets + b];
                    ValueTraits::relocateManyUncached(&dstElems[pos], ((Value*)buffer[b]), bufCnt[b]);
                    pos += bufCnt[b];
                    bufCnt[b] = 0;
                }
            }

            for (size_t b = 0; b < numBuckets; b++) {
                if (bufCnt[b] == 0)
                    continue;
                size_t &pos = localHisto[t * numBuckets + b];
                ValueTraits::relocateMany(&dstElems[pos], ((Value*)buffer[b]), bufCnt[b]);
                pos += bufCnt[b];
            }
        });
    }
#endif
    if (scatterSimple) {
        parallelWorkers(numWorkers, [numElems, numBuckets, numWorkers, srcElems, dstElems, localHisto, bucketOf](size_t t) {
            size_t l = uint64_t(numElems) * (t + 0) / numWorkers;
            size_t r = uint64_t(numElems) * (t + 1) / numWorkers;
            for (size_t i = l; i < r; i++) {
                size_t b = bucketOf[i];
                size_t &pos = localHisto[t * numBuckets + b];
                ValueTraits::relocateOne(dstElems[pos++], srcElems[i]);
            }
        });
    }

#if TBBSS_TBBSS_SLOW_ASSERT
    Span<const size_t> splits = globalHisto;
    assert(splits[0] == 0 && splits[numBuckets] == numElems);
    for (size_t b = 0; b < numBuckets; b++) {
        assert(splits[b] <= splits[b + 1]);
        for (size_t i = splits[b]; i < splits[b + 1]; i++) {
            assert(b == 0 || !comp(dstElems[i], pivot.sorted_[b - 1]));
            assert(b == numBuckets - 1 || !comp(pivot.sorted_[b], dstElems[i]));
        }
    }
#endif    
}

//---------------------------------------------------------

template<class Value, class Comp, class ValueTraits>
struct SharedData;

template<class Value, class ValueTraits>
struct ThreadData {
    MultiPivot<Value, ValueTraits> pivot_;
    PartitionResult partition_;
};

template<class Value, class Comp, class ValueTraits>
struct TaskData {
    size_t first_ = 0;
    size_t len_ = 0;
    size_t world_ = 0;
    size_t whome_ = 0;
    size_t numWorkers_ = 0;
    Random random_;

    tbb::task_group *taskGroup_ = nullptr;
    SharedData<Value, Comp, ValueTraits> *shared_ = nullptr;
};

template<class Value, class Comp, class ValueTraits>
struct SharedData {
    size_t numElems_ = 0;
    Span<Value> elemsSpans_[2];
    std::unique_ptr<Value[], decltype(&Allocator<Value>::deallocate)> elemsCopyStore_{nullptr, Allocator<Value>::deallocate};

    Array<uint8_t, DefaultValueTraits<uint8_t>> bucketIndexStore_;
    const Comp *comparator_ = nullptr;

    uint64_t randomSeed_ = 0xDEADBEEF01234567ull;
    tbb::enumerable_thread_specific<ThreadData<Value, ValueTraits>> perThread_;
#if TBBSS_COLLECT_STATS
    std::atomic_int sizeStats_[64];
#endif
};

//---------------------------------------------------------

template<class Value, class Comp, class ValueTraits>
void copyBack(const TaskData<Value, Comp, ValueTraits> &task) {
    if (task.world_ == task.whome_)
        return;

    Span<Value> srcElems = task.shared_->elemsSpans_[task.world_].subspan(task.first_, task.len_);
    Span<Value> dstElems = task.shared_->elemsSpans_[task.whome_].subspan(task.first_, task.len_);
    ValueTraits::relocateMany(dstElems.data(), srcElems.data(), srcElems.size());
}

template<class Value, class Comp, class ValueTraits>
void processBase(const TaskData<Value, Comp, ValueTraits> &task) {
    Span<Value> srcElems = task.shared_->elemsSpans_[task.world_].subspan(task.first_, task.len_);
    smallSort<Value, Comp, ValueTraits>(srcElems, *task.shared_->comparator_);

    copyBack(task);
}

template<class Value, class Comp, class ValueTraits>
void processRecursive(TaskData<Value, Comp, ValueTraits> task) {
    SharedData<Value, Comp, ValueTraits> *shared = task.shared_;

    Span<Value> srcElems = shared->elemsSpans_[task.world_].subspan(task.first_, task.len_);
    Span<Value> dstElems = shared->elemsSpans_[task.world_ ^ 1].subspan(task.first_, task.len_);
    Span<uint8_t> bucketOf = makeSpan(shared->bucketIndexStore_).subspan(task.first_, task.len_);

    assert(task.len_ > TBBSS_SMALLSORT_MAX);
    size_t logn = log2up(task.len_);
#if TBBSS_COLLECT_STATS
    shared->sizeStats_[logn].fetch_add(1, std::memory_order_relaxed);
#endif

    size_t numBuckets;
    if (task.len_ <= (1 << 12)) {
        // one level: aim for 8-16 elements per bucket
        size_t logb = logn - 4;
        numBuckets = 1 << logb;
    }
    else {
        // two levels: aim for 8-16 elements per bucket
        size_t logb = (logn - 4) / 2;
        numBuckets = 1 << logb;
        numBuckets = std::min<size_t>(numBuckets, 256);
    }
    assert(numBuckets >= 2 && numBuckets <= 256 && isPot(numBuckets));

    size_t numSamples = MultiPivot<Value, ValueTraits>::selectSamples(srcElems, numBuckets, task.random_);

    if (numSamples <= TBBSS_SMALLSORT_MAX) {
#if TBBSS_COLLECT_STATS
        shared->sizeStats_[log2up(numSamples)].fetch_add(1, std::memory_order_relaxed);
#endif
        smallSort<Value, Comp, ValueTraits>(srcElems.subspan(0, numSamples), *shared->comparator_);
    }
    else {
        tbb::task_group subTaskGroup;
        TaskData<Value, Comp, ValueTraits> samplesTask;
        samplesTask.shared_ = shared;
        samplesTask.taskGroup_ = &subTaskGroup;
        samplesTask.first_ = task.first_;
        samplesTask.len_ = numSamples;
        samplesTask.world_ = task.world_;
        samplesTask.whome_ = task.world_;
        samplesTask.numWorkers_ = 1;
        samplesTask.random_ = task.random_;
        subTaskGroup.run_and_wait([samplesTask] {
            processRecursive(samplesTask);
        });
    }

    // note: we can't use threadlocal data before sorting samples
    // because the recursive call can overwrite it
    ThreadData<Value, ValueTraits> *perThread = &shared->perThread_.local();
    perThread->pivot_.initFromSortedSamples(srcElems.subspan(0, numSamples), numBuckets);

    multiPartition(
        srcElems, perThread->pivot_, *shared->comparator_,
        dstElems, perThread->partition_,
        task.numWorkers_, bucketOf
    );
    Span<const size_t> splits = makeSpan(perThread->partition_.splits_);

    TaskData<Value, Comp, ValueTraits> subTask;
    subTask.shared_ = task.shared_;
    subTask.taskGroup_ = task.taskGroup_;
    subTask.world_ = task.world_ ^ 1;
    subTask.whome_ = task.whome_;
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

        if (subTask.len_ <= TBBSS_SMALLSORT_MAX) {
#if TBBSS_COLLECT_STATS
            shared->sizeStats_[log2up(subTask.len_)].fetch_add(1, std::memory_order_relaxed);
#endif
            processBase(subTask);
            continue;
        }

        if (b == 0)
            subTask.random_ = task.random_;
        else
            subTask.random_.initStream(task.shared_->randomSeed_, subTask.first_);

        task.taskGroup_->run([subTask] {
            processRecursive(subTask);
        });
    }
}

//---------------------------------------------------------

template<
    class Value,
    class Comp = std::less<Value>,
    class ValueTraits = DefaultValueTraits<Value>
>
void sampleSort(Value *begin, size_t num, const Comp &comp = Comp()) {
    if (num <= TBBSS_SMALLSORT_MAX) {
        smallSort<Value, Comp, ValueTraits>({begin, num}, comp);
        return;
    }

    SharedData<Value, Comp, ValueTraits> shared;
    shared.elemsCopyStore_.reset(Allocator<Value>::allocate(num));
    shared.bucketIndexStore_.clearResize(num);
    shared.numElems_ = num;
    shared.elemsSpans_[0] = {begin, num};
    shared.elemsSpans_[1] = {shared.elemsCopyStore_.get(), num};

    tbb::task_group rootTaskGroup;

    TaskData<Value, Comp, ValueTraits> task;
    task.shared_ = &shared;
    task.taskGroup_ = &rootTaskGroup;
    task.first_ = 0;
    task.len_ = shared.numElems_;
    task.numWorkers_ = tbb::this_task_arena::max_concurrency();
    task.world_ = 0;
    task.whome_ = 0;
    task.random_.initStream(shared.randomSeed_, task.first_);

    rootTaskGroup.run_and_wait([&task] {
        processRecursive(task);
    });

#if TBBSS_COLLECT_STATS
    for (int i = 1; i < 32; i++)
        printf("L%02d: %7d [%zu..%zu)\n", i, shared.sizeStats_[i].load(), size_t(1) << (i-1), size_t(1) << i);
#endif        
}

}
