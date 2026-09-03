/**
 * (C) Copyright Stepan Gatilov 2026.
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#include <cstddef>
#include <cstring>
#include <cassert>

#include <algorithm>
#include <type_traits>

#ifdef _MSC_VER
    #pragma warning(push)
    // unreachable code is OK in templates, especially in DefaultValueTraits
    // turns out it happens inside TBB as well..
    #pragma warning(disable: 4702)
#endif

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_group.h>
#include <oneapi/tbb/task_arena.h>

// maximum number of buckets for multi-partition
#define TBBSS_MAX_BUCKETS 256

// when we have less elements, we sort them using trivial quadratic sort
// note: this is tightly coupled to the strategy of selecting number of buckets in the code!
#define TBBSS_SMALLSORT_MAX 32

// run multi-pivot classification on X elements at once for better ILP and less overhead
// note: manually unrolled version exists only for X == 8,
// other values may get slower if a compiler fails to unroll (MSVC does fail)
#define TBBSS_CLASSIFY_UNROLL 8

#ifndef TBBSS_BRANCHLESS_COMPARESWAP
    // force branchless compare-and-swap via XOR and bitmasking in small sort?
    // this is faster in practice, and compilers sometimes optimize away XOR and retain only CMOV
    #define TBBSS_BRANCHLESS_COMPARESWAP 1
#endif

#ifndef TBBSS_UNCACHED_MININPUT_BYTES
    // scattering many elements across buckets is performed using uncached writes (SSE instructions)
    // in theory, it should reduce memory bandwidth, but in practice, I don't see measurably benefit
    // note: set X == 0 to disable optimization

    // uncached writes are enabled if total size of elements is more than X bytes
    #define TBBSS_UNCACHED_MININPUT_BYTES (1 << 20)
    // every bucket has an Y-byte buffer which accumulates outgoing elements before pushing them to RAM
    #define TBBSS_UNCACHED_BUFFER_BYTES (1 << 10)
#endif

#ifndef TBBSS_LARGE_PAGES
    // if X != 0, then large memory allocations are hinted for "transparent huge pages" with MADV_HUGEPAGE
    // the allocations are also aligned to X bytes, so the only meaningful value is X = (2 << 20)
    #define TBBSS_LARGE_PAGES 0
    // note: this can radically accelerate freeing virtual memory pages at the end of the algorithm,
    // which can otherwise can take 5-10% of wall time
    // however, huge pages have many drawbacks, so I'm afraid this is not usable outside benchmarks
#endif

#ifndef TBBSS_DROP_ASSERTS
    // X = 0: all internal asserts use standard C assert macro (stripped by NDEBUG)
    // X = 1: all internal asserts are stripped from compilation
    #define TBBSS_DROP_ASSERTS 0
#endif

//#define TBBSS_COLLECT_STATS 1


#ifdef _MSC_VER
    #define TBBSS_FORCEINLINE __forceinline
    #define TBBSS_NOINLINE __declspec(noinline)
    #define TBBSS_MAYALIAS
#else
    #define TBBSS_FORCEINLINE __attribute__((always_inline)) inline
    #define TBBSS_NOINLINE __attribute__((noinline))
    #define TBBSS_MAYALIAS __attribute__((may_alias))
#endif

#if TBBSS_DROP_ASSERTS
    #define TBBSS_ASSERT(...) (void(0))
#else
    #define TBBSS_ASSERT(...) assert(__VA_ARGS__)
#endif

#if TBBSS_COLLECT_STATS
    #include <atomic>
    #define TBBSS_STATS_ADD(total, delta) total.fetch_add(delta, std::memory_order_relaxed)
#else
    #define TBBSS_STATS_ADD(total, delta) (void(0))
#endif

namespace tbbss {

//---------------------------------------------------------

TBBSS_FORCEINLINE bool isPot(size_t x) {
    return (x & (x - 1)) == 0;
}

inline size_t log2up(size_t x) {
    size_t res = 0;
    while ((size_t(1) << res) < x)
        res++;
    return res;
}

// unfortunately, TBB is not thread-safe by default due to this issue:
//   https://github.com/uxlfoundation/oneTBB/issues/253
// in order to make it thread-safe, we have to explicitly check for cancellation after every TBB wait
// otherwise TBB can continue a task despite its invariant "waited-upon tasks are finished" is broken
struct TbbCancelException : std::exception {};
inline void propagateCancelProperly() {
    if (tbb::is_current_task_group_canceling())
        throw TbbCancelException();
}

template<class Lambda> void parallelWorkers(size_t numWorkers, Lambda&& lambda) {
    TBBSS_ASSERT(numWorkers > 0);
    if (numWorkers == 1) {
        lambda(0);
    }
    else {
        tbb::parallel_for<size_t>(0, numWorkers, lambda);
        propagateCancelProperly();
    }
}

//---------------------------------------------------------

struct Random {
    //==========================================
    // xoroshiro128++ version 1.0, taken from:
    //   https://prng.di.unimi.it/xoroshiro128plusplus.c

    static TBBSS_FORCEINLINE int64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    uint64_t s[2];

    uint64_t TBBSS_FORCEINLINE next(void) {
        const uint64_t s0 = s[0];
        uint64_t s1 = s[1];
        const uint64_t result = rotl(s0 + s1, 17) + s0;

        s1 ^= s0;
        s[0] = rotl(s0, 49) ^ s1 ^ (s1 << 21); // a, b
        s[1] = rotl(s1, 28); // c

        return result;
    }

    //==========================================

    void init(size_t start, size_t len, uint64_t seed) {
        s[0] = start ^ seed;
        s[1] = len;
        TBBSS_ASSERT(len != 0);
        next();
    }

    size_t generate(size_t maxExclusive) {
        return next() % maxExclusive;
    }
};

//---------------------------------------------------------

void *allocateMemory(size_t bytes, size_t reqAlign);
void deallocateMemory(void *ptr, size_t bytes, size_t reqAlign);

template<class T> struct Allocator {
    static T *allocate(size_t n) {
        if (!n)
            return nullptr;
        return reinterpret_cast<T*>(allocateMemory(n * sizeof(T), alignof(T)));
    }
    static void deallocate(T *ptr, size_t n) {
        if (!n)
            return;
        return deallocateMemory(ptr, n * sizeof(T), alignof(T));
    }
};

template<class T> struct Raw {
    alignas(T) char bytes_[sizeof(T)];

    // "StdVector" fails on Clang with T = std::vector
    // it works if any of this is used:
    //   * rtNone mode
    //   * -fno-strict-vtable-pointers
    //   * C++20 and std::launder here
    // so let's use launder, it is noop anyway
    TBBSS_FORCEINLINE T *data() {
        return std::launder(reinterpret_cast<T *>(bytes_));
    }
    TBBSS_FORCEINLINE const T *data() const {
        return std::launder(reinterpret_cast<const T *>(bytes_));
    }
    TBBSS_FORCEINLINE T &get() {
        return *std::launder(reinterpret_cast<T *>(bytes_));
    }
    TBBSS_FORCEINLINE const T &get() const {
        return *std::launder(reinterpret_cast<const T *>(bytes_));
    }
};
template<class T> constexpr inline bool IsRaw = false;
template<class T> constexpr inline bool IsRaw<Raw<T>> = true;

//---------------------------------------------------------

void memcpyUncached(char *dst, const char *src, size_t size);

template<class T> TBBSS_FORCEINLINE void memcpyElement(void *dst, const void *src) {
#if 0
    memcpy(dst, src, sizeof(T));
#else
    // some people believe that memcpy of constant small size is compiled into optimal code
    // this is not true unfortunately: neither on GCC nor on MSVC

    // without this hack, the algorithm breaks on Clang under strict aliasing rules
    typedef TBBSS_MAYALIAS uint64_t ma_uint64;
    typedef TBBSS_MAYALIAS uint32_t ma_uint32;
    typedef TBBSS_MAYALIAS uint16_t ma_uint16;
    typedef TBBSS_MAYALIAS uint8_t ma_uint8;

    // note: unaligned access may happen here
    for (size_t i = 0; i < sizeof(T) / 8; i++)
        reinterpret_cast<ma_uint64*>(dst)[i] = reinterpret_cast<const ma_uint64*>(src)[i];

    if constexpr (sizeof(T) % 8 >= 4) {
        static constexpr size_t Index = 2 * (sizeof(T) / 8);
        reinterpret_cast<ma_uint32*>(dst)[Index] = reinterpret_cast<const ma_uint32*>(src)[Index];
    }
    if constexpr (sizeof(T) % 4 >= 2) {
        static constexpr size_t Index = 2 * (sizeof(T) / 4);
        reinterpret_cast<ma_uint16*>(dst)[Index] = reinterpret_cast<const ma_uint16*>(src)[Index];
    }
    if constexpr (sizeof(T) % 2 >= 1) {
        static constexpr size_t Index = 2 * (sizeof(T) / 2);
        reinterpret_cast<ma_uint8*>(dst)[Index] = reinterpret_cast<const ma_uint8*>(src)[Index];
    }
#endif
}

#if TBBSS_BRANCHLESS_COMPARESWAP
// this compare-and-swap primitive is almost guaranteed to generate branchless code, even if not perfectly optimal
// in practice, e.g. GCC optimizes away XOR and leaves two CMOV commands
// smallSort works much faster on random inputs with branchless compare-and-swap
template<class T> TBBSS_FORCEINLINE void memswapElementConditional(void *dst, void *src, bool doSwap) {
    int64_t mask = -int64_t(doSwap);

    typedef TBBSS_MAYALIAS uint64_t ma_uint64;
    typedef TBBSS_MAYALIAS uint32_t ma_uint32;
    typedef TBBSS_MAYALIAS uint16_t ma_uint16;
    typedef TBBSS_MAYALIAS uint8_t ma_uint8;

    // note: unaligned access may happen here
    for (size_t i = 0; i < sizeof(T) / 8; i++) {
        ma_uint64 &a = reinterpret_cast<ma_uint64*>(dst)[i];
        ma_uint64 &b = reinterpret_cast<ma_uint64*>(src)[i];
        uint64_t delta = (a ^ b) & uint64_t(mask);
        a ^= delta;
        b ^= delta;
    }

    if constexpr (sizeof(T) % 8 >= 4) {
        static constexpr size_t Index = 2 * (sizeof(T) / 8);
        ma_uint32 &a = reinterpret_cast<ma_uint32*>(dst)[Index];
        ma_uint32 &b = reinterpret_cast<ma_uint32*>(src)[Index];
        uint32_t delta = (a ^ b) & uint32_t(mask);
        a ^= delta;
        b ^= delta;
    }
    if constexpr (sizeof(T) % 4 >= 2) {
        static constexpr size_t Index = 2 * (sizeof(T) / 4);
        ma_uint16 &a = reinterpret_cast<ma_uint16*>(dst)[Index];
        ma_uint16 &b = reinterpret_cast<ma_uint16*>(src)[Index];
        uint16_t delta = (a ^ b) & uint16_t(mask);
        a ^= delta;
        b ^= delta;

    }
    if constexpr (sizeof(T) % 2 >= 1) {
        static constexpr size_t Index = 2 * (sizeof(T) / 2);
        ma_uint8 &a = reinterpret_cast<ma_uint8*>(dst)[Index];
        ma_uint8 &b = reinterpret_cast<ma_uint8*>(src)[Index];
        uint8_t delta = (a ^ b) & uint8_t(mask);
        a ^= delta;
        b ^= delta;
    }
}
#endif

// you can select one of three modes when you run sorting algorithm
// BEWARE: the lifetime methods used must not throw exceptions!
// in rtFork mode, lifetime methods are never used: the algorithm just memcpy-s bytes around
// rtNone mode is also "the least undefined behavior" out of the three...
//
// note: "relocation" is Rust-style move
// it moves an element into specified raw memory and makes source memory raw
enum RelocationTrivialness {
    // relocation is done via destructing move: move construct + destroy old
    // must be used for types that contain self-references
    // e.g. std::string with small string optimization
    rtNone = 0,
    // relocation is done with memcpy instead of destructing move
    // this mode can obviously be used for trivially copyable types
    // but it can also be used for std::unique_ptr, std::vector, std::set etc.
    rtRelocate = 1,
    // temporary copy of an element is created using memcpy too
    // such copies are used only to run comparator, and they are NOT destroyed
    // this mode is automatically used for trivially copyable types
    // but it can also be used for std::unique_ptr, std::vector, std::set etc.
    rtFork = 2,
};

// you can specialize this template to change the default mode globally by type
template<class T> constexpr RelocationTrivialness IsTriviallyRelocatable = (std::is_trivially_copyable_v<T> ? rtFork : rtNone);
// or you can pass non-default ValueTraits template argument into the algorithm
// in this case you can set "IsRelocationTrivial" argument of DefaultValueTraits

// note: the algorithm touches your elements only using these methods and by calling comparator!
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
            Raw<T> temp;
            memcpyElement<T>(&temp, &dst);
            memcpyElement<T>(&dst, &src);
            memcpyElement<T>(&src, &temp);
        }
    }

    static TBBSS_FORCEINLINE void swapConditionalOne(T &dst, T &src, bool doSwap) noexcept {
#if TBBSS_BRANCHLESS_COMPARESWAP        
        if constexpr (IsRelocationTrivial != rtNone) {
            memswapElementConditional<T>(&dst, &src, doSwap);
            return;
        }
#endif
        if (doSwap)
            swapOne(dst, src);
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
        TBBSS_ASSERT(dst >= src + n || src >= dst + n);
        for (size_t i = 0; i < n; i++)
            relocateOne(dst[i], src[i]);
    }
    static void relocateManyUncached(T *dst, T *src, size_t n) noexcept {
        if constexpr (IsRelocationTrivial == rtNone)
            return relocateMany(dst, src, n);
        TBBSS_ASSERT(dst >= src + n || src >= dst + n);
        memcpyUncached(reinterpret_cast<char*>(dst), reinterpret_cast<const char*>(src), n * sizeof(T));
    }
    static void destroyMany(T *dst, size_t n) noexcept {
        for (size_t i = 0; i < n; i++)
            destroyOne(dst[i]);
    }
};

template<class T>
void constructDefaultMany(T *dst, size_t n) noexcept {
    // only used internally, never used for user elements
    static_assert(std::is_integral_v<T> || IsRaw<T>);
    for (size_t i = 0; i < n; i++)
        new(&dst[i]) T;
}

//---------------------------------------------------------

template<class T, class ValueTraits> class Array {
    T *ptr_ = nullptr;
    size_t num_ = 0;
    size_t cap_ = 0;

    TBBSS_NOINLINE void grow(size_t n) {
        TBBSS_ASSERT(n > cap_);
        TBBSS_ASSERT(num_ == 0);
        n = std::max(n, 2 * cap_ + 1);
        Allocator<T>::deallocate(ptr_, cap_);
        ptr_ = Allocator<T>::allocate(n);
        cap_ = n;
    }

public:
    ~Array() {
        clear();
        Allocator<T>::deallocate(ptr_, cap_);
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

    TBBSS_FORCEINLINE void pushBack(const T& x) {
        TBBSS_ASSERT(num_ < cap_);
        ValueTraits::constructCopyOne(ptr_[num_++], x);
    }

    TBBSS_FORCEINLINE T *data() { return ptr_; }
    TBBSS_FORCEINLINE const T *data() const { return ptr_; }
    TBBSS_FORCEINLINE size_t size() const { return num_; }

    TBBSS_FORCEINLINE T &operator[] (size_t i) {
        TBBSS_ASSERT(i < num_);
        return ptr_[i];
    }
    TBBSS_FORCEINLINE const T &operator[] (size_t i) const {
        TBBSS_ASSERT(i < num_);
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
        TBBSS_ASSERT(i < num_);
        return ptr_[i];
    }

    TBBSS_FORCEINLINE Span<T> subspan(size_t start, size_t len) const {
        TBBSS_ASSERT(start + len <= num_);
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
    // (for testing only)
    std::sort(arr.data(), arr.data() + arr.size(), std::reference_wrapper(comp));
#else
    // this is a trivial bubble-like sort / insertion sort as a sorting network:
    //   for i = [0..N) for j = [0..i) CompareAndSwap(A[i], A[j]);
    // surprisingly, it works faster than standard insertion sort for me
    // even with branchy compare-and-swap, it suffers just one extra misprediction per outer iteration
    // it is trivial to verify and generates tiny code
    Raw<Value> buffer;
    for (size_t i = 1; i < arr.size(); i++) {
        Value &arrI = buffer.get();
        ValueTraits::relocateOne(arrI, arr[i]);
        for (size_t j = 0; j < i; j++)
            ValueTraits::swapConditionalOne(arrI, arr[j], comp(arrI, arr[j]));
        ValueTraits::relocateOne(arr[i], arrI);
    }

    TBBSS_ASSERT(std::is_sorted(arr.data(), arr.data() + arr.size(), std::reference_wrapper(comp))); 
#endif
}

//---------------------------------------------------------

template<class Value, class ValueTraits>
struct MultiPivot {
    size_t numBits_ = 0;
    size_t numBuckets_ = 0;
    Raw<Value> sortedStore_[TBBSS_MAX_BUCKETS];
    Raw<Value> treeStore_[TBBSS_MAX_BUCKETS];

    ~MultiPivot() {
        ValueTraits::destroyMany(sortedStore_[0].data(), numBuckets_);
        ValueTraits::destroyMany(treeStore_[0].data(), numBuckets_ - 1);
    }

    static size_t selectSamples(Span<Value> arr, size_t numBuckets, Random &random) {
        TBBSS_ASSERT(isPot(numBuckets) && numBuckets >= 2);
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

        Span<Value> sorted(sortedStore_[0].data(), numBuckets);
        TBBSS_ASSERT(sorted.size() <= std::size(sortedStore_));

        for (size_t i = 1; i <= numBuckets - 1; i++) {
            size_t pos = uint64_t(numSamples) * i / numBuckets;
            ValueTraits::constructCopyOne(sorted[i], samples[pos]);
        }
        // note: we prepend one sentinel element for branchless handling of equality buckets
        ValueTraits::constructCopyOne(sorted[0], sorted[1]);

        size_t numBits = log2up(numBuckets);

        Span<Value> tree(treeStore_[0].data(), numBuckets - 1);
        TBBSS_ASSERT(tree.size() <= std::size(treeStore_));
        size_t filled = 0;
        for (ptrdiff_t b = numBits - 1; b >= 0; b--) {
            size_t len = size_t(1) << b;
            for (size_t i = len - 1; i < numBuckets; i += len * 2)
                ValueTraits::constructCopyOne(tree[filled++], sorted[i + 1]);
        }
        TBBSS_ASSERT(filled == tree.size());

        numBuckets_ = numBuckets;
        numBits_ = numBits;        
    }

    template<class Comp>
    TBBSS_FORCEINLINE size_t classifyOne(const Value &value, const Comp &comp) const {
        size_t res = 0;
        Span<const Value> tree(treeStore_[0].data(), numBuckets_ - 1);
        // find bucket for the element using prepared binary search tree
        for (size_t b = 0; b < numBits_; b++) {
            bool isLess = comp(value, tree[res]);
            res = 2 * res + 1 + size_t(!isLess);    // branchless
        }

        Span<const Value> sorted(sortedStore_[0].data(), numBuckets_);
        res -= (numBuckets_ - 1);
        TBBSS_ASSERT(res == numBuckets_ - 1 || comp(value, sorted[res + 1]));
        TBBSS_ASSERT(res == 0 || !comp(value, sorted[res]));

        // for a bucket [L..R), redirect elements equal to L into the previous bucket
        // whenever some pivot X is duplicated, all elements equal to X land into their own bucket
        // this idea of equality buckets allows us to throw many equal elements out of recursion
        // note: bitwise AND is intentional to make it branchless
        res -= (res > 0) & !comp(sorted[res], value);
        TBBSS_ASSERT(res < numBuckets_);
        return res;
    }

    // classify fixed number of elements at once
    // for less overhead and better ILP
    template<size_t N, class Comp>
    TBBSS_FORCEINLINE void classifyBlock(const Value *value, size_t *res, const Comp &comp) const {
        Span<const Value> tree(treeStore_[0].data(), numBuckets_ - 1);
        for (size_t i = 0; i < N; i++)
            res[i] = 0;
        for (size_t b = 0; b < numBits_; b++) {
            for (size_t i = 0; i < N; i++) {
                bool isLess = comp(value[i], tree[res[i]]);
                res[i] = 2 * res[i] + 1 + size_t(!isLess);
            }
        }
        Span<const Value> sorted(sortedStore_[0].data(), numBuckets_);
        for (size_t i = 0; i < N; i++) {
            res[i] -= (numBuckets_ - 1);
            res[i] -= (res[i] > 0) & !comp(sorted[res[i]], value[i]);
        }
    }

    // sadly, MSVC does not unroll the block loops 
    // so we have to do it manually =(
    template<class Comp>
    TBBSS_FORCEINLINE void classifyBlock8(const Value *value, size_t *res, const Comp &comp) const {
        Span<const Value> tree(treeStore_[0].data(), numBuckets_ - 1);
        #define TBBSS_ITER(i) res[i] = 0
        TBBSS_ITER(0);
        TBBSS_ITER(1);
        TBBSS_ITER(2);
        TBBSS_ITER(3);
        TBBSS_ITER(4);
        TBBSS_ITER(5);
        TBBSS_ITER(6);
        TBBSS_ITER(7);
        #undef TBBSS_ITER
        for (size_t b = 0; b < numBits_; b++) {
            #define TBBSS_ITER(i) res[i] = 2 * res[i] + 1 + size_t(!comp(value[i], tree[res[i]]));
            TBBSS_ITER(0);
            TBBSS_ITER(1);
            TBBSS_ITER(2);
            TBBSS_ITER(3);
            TBBSS_ITER(4);
            TBBSS_ITER(5);
            TBBSS_ITER(6);
            TBBSS_ITER(7);
            #undef TBBSS_ITER
        }
        Span<const Value> sorted(sortedStore_[0].data(), numBuckets_);
        #define TBBSS_ITER(i) res[i] -= (numBuckets_ - 1); res[i] -= (res[i] > 0) & !comp(sorted[res[i]], value[i]);
        TBBSS_ITER(0);
        TBBSS_ITER(1);
        TBBSS_ITER(2);
        TBBSS_ITER(3);
        TBBSS_ITER(4);
        TBBSS_ITER(5);
        TBBSS_ITER(6);
        TBBSS_ITER(7);
        #undef TBBSS_ITER
    }
};

template<class Value, class ValueTraits, class Comp>
void multiPartition(
    Span<Value> srcElems, const MultiPivot<Value, ValueTraits> &pivot, const Comp &comp,
    Span<Value> dstElems, Span<size_t> splits,
    size_t numWorkers, Span<uint8_t> bucketOf
) {
    size_t numBuckets = pivot.numBuckets_;
    size_t numElems = srcElems.size();
    
    size_t localHistoSmallStore[TBBSS_MAX_BUCKETS + 1];
    Span<size_t> localHisto(localHistoSmallStore, numWorkers * numBuckets);
    Array<size_t, DefaultValueTraits<size_t>> localHistoBigStore;
    if (localHisto.size() > std::size(localHistoSmallStore)) {
        localHistoBigStore.clearResize(localHisto.size());
        localHisto = makeSpan(localHistoBigStore);
    }
    for (size_t i = 0; i < localHisto.size(); i++)
        localHisto[i] = 0;

    parallelWorkers(numWorkers, [&pivot, numElems, numBuckets, numWorkers, srcElems, bucketOf, localHisto, &comp](size_t t) {
        size_t l = uint64_t(numElems) * (t + 0) / numWorkers;
        size_t r = uint64_t(numElems) * (t + 1) / numWorkers;
        size_t i = l;
#ifdef TBBSS_CLASSIFY_UNROLL
        for (; i + TBBSS_CLASSIFY_UNROLL <= r; i += TBBSS_CLASSIFY_UNROLL) {
            size_t bidx[TBBSS_CLASSIFY_UNROLL];
#if TBBSS_CLASSIFY_UNROLL == 8
            pivot.classifyBlock8(&srcElems[i], bidx, comp);
#else
            pivot.template classifyBlock<TBBSS_CLASSIFY_UNROLL>(&srcElems[i], bidx, comp);
#endif
            for (size_t q = 0; q < TBBSS_CLASSIFY_UNROLL; q++) {
                size_t b = bidx[q];
                bucketOf[i + q] = uint8_t(b);
                localHisto[t * numBuckets + b]++;
            }
        }
#endif        
        for (; i < r; i++) {
            size_t b = pivot.classifyOne(srcElems[i], comp);
            bucketOf[i] = uint8_t(b);
            localHisto[t * numBuckets + b]++;
        }
    });

    TBBSS_ASSERT(splits.size() == numBuckets + 1);
    for (size_t i = 0; i < splits.size(); i++)
        splits[i] = 0;
    Span<size_t> globalHisto = splits;

    for (size_t t = 0; t < numWorkers; t++)
        for (size_t b = 0; b < numBuckets; b++)
            globalHisto[b + 1] += localHisto[t * numBuckets + b];

    for (size_t b = 0; b < numBuckets; b++)
        globalHisto[b + 1] += globalHisto[b];
    TBBSS_ASSERT(globalHisto[numBuckets] == numElems);

    for (size_t b = 0; b < numBuckets; b++) {
        size_t tsum = globalHisto[b];
        for (size_t t = 0; t < numWorkers; t++) {
            size_t nsum = tsum + localHisto[t * numBuckets + b];
            localHisto[t * numBuckets + b] = tsum;
            tsum = nsum;
        }
        TBBSS_ASSERT(tsum == globalHisto[b + 1]);
    }
    
    bool distributeSimple = true;
#if TBBSS_UNCACHED_MININPUT_BYTES
    if (numElems * sizeof(Value) > TBBSS_UNCACHED_MININPUT_BYTES) {
        // when copying a lot of elements from the array to its mirrored copy,
        // we normally pay 3x bandwidth: read source, write destination, and read destination
        // using uncached writes allows to avoid reading destination cache lines unnecessarily
        distributeSimple = false;
        parallelWorkers(numWorkers, [numElems, numBuckets, numWorkers, srcElems, dstElems, localHisto, bucketOf](size_t t) {
            size_t l = uint64_t(numElems) * (t + 0) / numWorkers;
            size_t r = uint64_t(numElems) * (t + 1) / numWorkers;

            uint32_t bufCnt[TBBSS_MAX_BUCKETS];
            for (size_t b = 0; b < numBuckets; b++)
                bufCnt[b] = 0;

            constexpr size_t Lane = TBBSS_UNCACHED_BUFFER_BYTES / sizeof(Value);
            static_assert(Lane > 0);
            Array<Raw<Value>, DefaultValueTraits<Raw<Value>>> buffer;
            buffer.clearResize(TBBSS_MAX_BUCKETS * Lane);

            for (size_t i = l; i < r; i++) {
                size_t b = bucketOf[i];
                ValueTraits::relocateOne(buffer[b * Lane + (bufCnt[b]++)].get(), srcElems[i]);
                if (bufCnt[b] == Lane) {
                    size_t &pos = localHisto[t * numBuckets + b];
                    ValueTraits::relocateManyUncached(&dstElems[pos], buffer[b * Lane].data(), bufCnt[b]);
                    pos += bufCnt[b];
                    bufCnt[b] = 0;
                }
            }

            for (size_t b = 0; b < numBuckets; b++) {
                if (bufCnt[b] == 0)
                    continue;
                size_t &pos = localHisto[t * numBuckets + b];
                ValueTraits::relocateManyUncached(&dstElems[pos], buffer[b * Lane].data(), bufCnt[b]);
                pos += bufCnt[b];
            }
        });
    }
#endif
    if (distributeSimple) {
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

    Span<const Value> sorted(pivot.sortedStore_[0].data(), numBuckets);
    TBBSS_ASSERT(splits[0] == 0 && splits[numBuckets] == numElems);
    for (size_t b = 0; b < numBuckets; b++) {
        TBBSS_ASSERT(splits[b] <= splits[b + 1]);
        for (size_t i = splits[b]; i < splits[b + 1]; i++) {
            TBBSS_ASSERT(b == 0 || !comp(dstElems[i], sorted[b]));
            TBBSS_ASSERT(b == numBuckets - 1 || !comp(sorted[b + 1], dstElems[i]));
        }
    }
}

//---------------------------------------------------------

template<class Value, class Comp, class ValueTraits>
struct SharedData;

template<class Value, class Comp, class ValueTraits>
struct TaskData {
    size_t first_;
    size_t len_;
    uint16_t world_;
    uint16_t whome_;
    uint32_t numWorkers_;
    tbb::task_group *taskGroup_;
    SharedData<Value, Comp, ValueTraits> *shared_;
};

template<class Value, class Comp, class ValueTraits>
struct SharedData {
    size_t numElems_ = 0;
    Span<Value> elemsSpans_[2];
    Array<Raw<Value>, DefaultValueTraits<Raw<Value>>> elemsCopyStore_;

    Array<uint8_t, DefaultValueTraits<uint8_t>> bucketIndexStore_;
    const Comp *comparator_ = nullptr;

    uint64_t randomSeed_ = 0xDEADBEEF01234567ull;
#if TBBSS_COLLECT_STATS
    std::atomic_int sizeStats_[64] = {};
    std::atomic_int taskStats_ = {};
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

// this is RAII helper around copyBack
// it ensures that elements are copied back into user array even if exception happens
template<class Value, class Comp, class ValueTraits>
struct TaskRunner {
    typedef TaskData<Value, Comp, ValueTraits> Data;

    Data data_;

    void finalize() {
        if (!data_.shared_)
            return;
        copyBack(data_);
        data_.shared_ = nullptr;
    }
    void disable() {
        data_.shared_ = nullptr;
    }

    ~TaskRunner() {
        finalize();
    }
    TaskRunner(const Data &data)
        : data_(data)
    {}

    TaskRunner(TaskRunner &&src) {
        data_ = src.data_;
        src.disable();
    }
    TaskRunner& operator=(TaskRunner &&src) = delete;
    TaskRunner(const TaskRunner&) = delete;
    TaskRunner& operator=(const TaskRunner&) = delete;    

    void operator() () const {
        // TBB allows move-only tasks in task_group, but requires operator() to be const for no reason
        // https://github.com/uxlfoundation/oneTBB/issues/393
        processRecursive(*const_cast<TaskRunner*>(this));
    }
};

template<class Value, class Comp, class ValueTraits>
void processRecursive(TaskRunner<Value, Comp, ValueTraits> &taskRunner) {
    TaskData<Value, Comp, ValueTraits> task = taskRunner.data_;
    SharedData<Value, Comp, ValueTraits> *shared = task.shared_;

    Span<Value> srcElems = shared->elemsSpans_[task.world_].subspan(task.first_, task.len_);
    Span<Value> dstElems = shared->elemsSpans_[task.world_ ^ 1].subspan(task.first_, task.len_);
    Span<uint8_t> bucketOf = makeSpan(shared->bucketIndexStore_).subspan(task.first_, task.len_);

    TBBSS_ASSERT(task.len_ > TBBSS_SMALLSORT_MAX);
    size_t logn = log2up(task.len_);
    TBBSS_STATS_ADD(shared->sizeStats_[logn], 1);

    size_t numBuckets;
    if (task.len_ <= (1 << 12)) {
        // one level: aim for 8-16 elements per bucket
        size_t logb = logn - 4;
        numBuckets = size_t(1) << logb;
    }
    else {
        // two levels: aim for 8-16 elements per bucket
        size_t logb = (logn - 4) / 2;
        numBuckets = size_t(1) << logb;
        numBuckets = std::min<size_t>(numBuckets, TBBSS_MAX_BUCKETS);
    }
    TBBSS_ASSERT(numBuckets >= 2 && numBuckets <= TBBSS_MAX_BUCKETS && isPot(numBuckets));

    // pivots are selected from samples, samples are selected randomly
    // random is good enough, deterministic, but not cryptographically secure
    Random random;
    random.init(task.first_, task.len_, shared->randomSeed_);
    size_t numSamples = MultiPivot<Value, ValueTraits>::selectSamples(srcElems, numBuckets, random);

    // samples are sorted using the same algorithm recursively
    if (numSamples <= TBBSS_SMALLSORT_MAX) {
        TBBSS_STATS_ADD(shared->sizeStats_[log2up(numSamples)], 1);
        smallSort<Value, Comp, ValueTraits>(srcElems.subspan(0, numSamples), *shared->comparator_);
    }
    else {
        // but the algorithm is run in its own task group
        // since we need to wait for its completion here
        tbb::task_group subTaskGroup;
        TaskData<Value, Comp, ValueTraits> samplesTask = {};
        samplesTask.shared_ = shared;
        samplesTask.taskGroup_ = &subTaskGroup;
        samplesTask.first_ = task.first_;
        samplesTask.len_ = numSamples;
        samplesTask.world_ = task.world_;
        samplesTask.whome_ = task.world_;
        samplesTask.numWorkers_ = 1;
        TBBSS_STATS_ADD(shared->taskStats_, 1);
        subTaskGroup.run_and_wait(TaskRunner<Value, Comp, ValueTraits>(samplesTask));
        propagateCancelProperly();
    }

    size_t splitsStore[TBBSS_MAX_BUCKETS + 1];
    Span<size_t> splits(splitsStore, numBuckets + 1);
    TBBSS_ASSERT(splits.size() <= std::size(splitsStore));

    MultiPivot<Value, ValueTraits> pivot;
    pivot.initFromSortedSamples(srcElems.subspan(0, numSamples), numBuckets);

    multiPartition(
        srcElems, pivot, *shared->comparator_,
        dstElems, splits,
        task.numWorkers_, bucketOf
    );

    // drop the RAII guard in the current task
    // note: it is critically important that unwinding cannot not happen until
    // all subtasks are either done or wrapped into TaskRunner-s!
    taskRunner.disable();

    TaskData<Value, Comp, ValueTraits> subTask = {};
    subTask.shared_ = task.shared_;
    subTask.taskGroup_ = task.taskGroup_;
    subTask.world_ = task.world_ ^ 1;
    subTask.whome_ = task.whome_;
    subTask.numWorkers_ = uint32_t((task.numWorkers_ + numBuckets - 1) / numBuckets);

    for (size_t b = 0; b < numBuckets; b++) {
        subTask.first_ = task.first_ + splits[b];
        subTask.len_ = splits[b + 1] - splits[b];

        // empty bucket: nothing to do
        if (subTask.len_ == 0)
            continue;

        // equality bucket: already sorted
        const Comp &comp = *shared->comparator_;
        if (b > 0 && b < numBuckets - 1 && !comp(pivot.sortedStore_[b].get(), pivot.sortedStore_[b + 1].get())) {
            [[maybe_unused]] const Value &ref = pivot.sortedStore_[b].get();
            for (size_t i = splits[b]; i < splits[b + 1]; i++)
                TBBSS_ASSERT(!comp(dstElems[i], ref) && !comp(ref, dstElems[i]));
            copyBack(subTask);
            continue;
        }

        // bucket too small: small-sort inline
        if (subTask.len_ <= TBBSS_SMALLSORT_MAX) {
            TBBSS_STATS_ADD(shared->sizeStats_[log2up(subTask.len_)], 1);
            Span<Value> subElems = shared->elemsSpans_[subTask.world_].subspan(subTask.first_, subTask.len_);
            smallSort<Value, Comp, ValueTraits>(subElems, *shared->comparator_);
            copyBack(subTask);
            continue;
        }

        // start subtask to sort this bucket recursively
        TBBSS_STATS_ADD(shared->taskStats_, 1);
        task.taskGroup_->run(TaskRunner<Value, Comp, ValueTraits>(subTask));
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
    shared.comparator_ = &comp;
    shared.elemsCopyStore_.clearResize(num);
    shared.bucketIndexStore_.clearResize(num);
    shared.numElems_ = num;
    shared.elemsSpans_[0] = {begin, num};
    shared.elemsSpans_[1] = {shared.elemsCopyStore_[0].data(), num};

    tbb::task_group rootTaskGroup;

    TaskData<Value, Comp, ValueTraits> task = {};
    task.shared_ = &shared;
    task.taskGroup_ = &rootTaskGroup;
    task.first_ = 0;
    task.len_ = shared.numElems_;
    task.numWorkers_ = tbb::this_task_arena::max_concurrency();
    task.world_ = 0;
    task.whome_ = 0;

    try {
        TBBSS_STATS_ADD(shared.taskStats_, 1);
        rootTaskGroup.run_and_wait(TaskRunner<Value, Comp, ValueTraits>(task));
    }
    catch (const TbbCancelException&) {
        // this can happen if user canceled algorithm using non-throwing cancel
        // e.g. started it inside tbb::task_group and called "cancel" on it
    }

#if TBBSS_COLLECT_STATS
    printf("TBB tasks: %d\n", shared.taskStats_.load());
    for (int i = 1; i < 32; i++)
        printf("L%02d: %7d [%zu..%zu)\n", i, shared.sizeStats_[i].load(), size_t(1) << (i-1), size_t(1) << i);
#endif        
}

// alternative interface for the lovers of iterators
// note: sorting does not support actual iterators, only raw pointers
template<
    class Value,
    class Comp = std::less<Value>,
    class ValueTraits = DefaultValueTraits<Value>
>
void sampleSortIter(Value *begin, Value *end, const Comp &comp = Comp()) {
    return sampleSort(begin, end - begin, comp);
}

}

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
