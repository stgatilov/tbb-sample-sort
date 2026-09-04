#include "tests_common.h"

#include <mutex>
#include <atomic>
#include <thread>

#ifdef _MSC_VER
    #define x_aligned_alloc(al, sz) _aligned_malloc((sz), (al))
    #define x_aligned_free(ptr) _aligned_free(ptr)
#else
    #define x_aligned_alloc(al, sz) aligned_alloc((al), (sz))
    #define x_aligned_free(ptr) free(ptr)
#endif

struct TrackedAllocation {
    std::align_val_t align = std::align_val_t(0);
    size_t size = 0;
    void *pointer = nullptr;
};
bool operator== (const TrackedAllocation &a, const TrackedAllocation &b) {
    return a.align == b.align && a.size == b.size && a.pointer == b.pointer;
}

constexpr size_t MaxTrackedAllocations = 64 << 10;

std::atomic_bool allocTracking = false;
std::mutex allocMutex;
TrackedAllocation trackedAllocs[MaxTrackedAllocations];
size_t allocWatermark;
size_t nextNewIndex;
size_t failedNewIndex;

void* operator new(size_t count, std::align_val_t al) {
    void *ptr = x_aligned_alloc(size_t(al), count);

    if (allocTracking.load()) {
        std::unique_lock lock(allocMutex);

        if (nextNewIndex++ == failedNewIndex) {
            // simulate failed allocation
            x_aligned_free(ptr);
            throw std::bad_alloc();
        }

        size_t i;
        for (i = 0; i < allocWatermark; i++)
            if (trackedAllocs[i] == TrackedAllocation())
                break;

        if (i == MaxTrackedAllocations) {
            WARN_MESSAGE(false, "tracked allocations overflow");
            std::terminate();
        }

        trackedAllocs[i] = {al, count, ptr};
        if (i == allocWatermark)
            allocWatermark++;
    }

    return ptr;
}

void operator delete(void* ptr, std::size_t sz, std::align_val_t al) noexcept {
    if (allocTracking.load()) {
        std::unique_lock lock(allocMutex);

        size_t i = 0;
        for (i = 0; i < allocWatermark; i++)
            if (trackedAllocs[i] == TrackedAllocation{al, sz, ptr})
                break;
        
        if (i == allocWatermark) {
            WARN_MESSAGE(false, "untracked free allocation");
            std::terminate();
        }
        
        trackedAllocs[i] = TrackedAllocation();
    }

    x_aligned_free(ptr);
}

void checkAllocationsEmpty() {
    size_t i;
    for (i = 0; i < allocWatermark; i++)
        if (!(trackedAllocs[i] == TrackedAllocation()))
            break;

    if (i < allocWatermark) {
        WARN_MESSAGE(false, "tracked allocation still alive");
        std::terminate();
    }
    allocWatermark = 0;
    nextNewIndex = 0;
    failedNewIndex = SIZE_MAX;
}

TEST_CASE("AllocPrecheck") {
    Random random;
    std::uniform_int_distribution<int> distr;
    std::vector<int> arr = generateArray<int>(DEFAULT_ARRAY_SIZE, [&]{
        return distr(random);
    });

    checkAllocationsEmpty();
    allocTracking.store(true);

    tbbss::sampleSort(arr.data(), arr.size());

    // simply verify that we intercept allocations properly
    allocTracking.store(false);
    CHECK_GT(allocWatermark, 0);
    checkAllocationsEmpty();

    checkSorted(arr.data(), arr.size());
}

#if TBBSS_UNCACHED_MININPUT_BYTES
    #define MIN_ALLOCATION_FAILS 10
#else
    // without uncached writes, we allocate too few memory buffers
    // I get only 3 allocation on the tests (elems copy, bucket indexes, local histo on first partition)
    #define MIN_ALLOCATION_FAILS 3
#endif

TEST_CASE("AllocFailsInteger") {
    Random random;
    std::uniform_int_distribution<int> distr;
    std::vector<int> arr = generateArray<int>(32 << 20, [&]{
        return distr(random);
    });

    int numExceptions = 0;
    std::vector<int> temp;
    for (int i = 0; i < 30; i++) {
        temp = arr;

        checkAllocationsEmpty();
        failedNewIndex = i; // some allocation fails
        allocTracking.store(true);

        try {
            tbbss::sampleSort(temp.data(), temp.size());
        }
        catch(std::bad_alloc&) {
            numExceptions++;
        }

        // check that deallocations are correct
        // but don't check that array is sorted
        allocTracking.store(false);
        checkAllocationsEmpty();
    }

    CHECK_GE(numExceptions, MIN_ALLOCATION_FAILS);
}

TEST_CASE("AllocFailsStdVector") {
    typedef std::vector<int> Element;
    Random random;
    std::uniform_int_distribution<int> distrLen(1, 3);
    std::uniform_int_distribution<int> distrElem(0, 1000);
    std::vector<Element> arr = generateArray<Element>(10 << 20, [&]{
        int l = distrLen(random);
        Element res;
        for (int i = 0; i < l; i++)
            res.push_back(distrElem(random));
        return res;
    });

    int numExceptions = 0;
    std::vector<Element> temp;
    for (int i = 0; i < 30; i++) {
        temp = arr;

        checkAllocationsEmpty();
        failedNewIndex = i; // some allocation fails
        allocTracking.store(true);

        try {
            tbbss::sampleSort<
                Element, std::less<Element>,
                // any mode should work fine
                // as long as any called element lifetime methods don't throw
                tbbss::DefaultValueTraits<Element, tbbss::rtNone>
            >(temp.data(), temp.size());
        }
        catch(std::bad_alloc&) {
            numExceptions++;
        }

        // check that deallocations are correct
        // but don't check that array is sorted
        allocTracking.store(false);
        checkAllocationsEmpty();
    }

    CHECK_GE(numExceptions, MIN_ALLOCATION_FAILS);
}

TEST_CASE("TbbCancelStdVector") {
    typedef std::vector<int> Element;
    Random random;
    std::uniform_int_distribution<int> distrLen(1, 3);
    std::uniform_int_distribution<int> distrElem(0, 1000);
    std::vector<Element> arr = generateArray<Element>(10 << 20, [&]{
        int l = distrLen(random);
        Element res;
        for (int i = 0; i < l; i++)
            res.push_back(distrElem(random));
        return res;
    });

    std::chrono::duration<double, std::milli> fullRunTime;
    std::vector<Element> temp;
    for (int i = 0; i < 30; i++) {
        temp = arr;

        auto Tbefore = std::chrono::steady_clock::now();

        tbb::task_group taskGroup;
        taskGroup.run([&] {
            tbbss::sampleSort(temp.data(), temp.size());
        });

        if (i <= 1) {
            taskGroup.wait();
            auto Tafter = std::chrono::steady_clock::now();
            fullRunTime = std::chrono::duration<double, std::milli>(Tafter - Tbefore);
        }
        else {
            // we need to reproduce the case where TbbCancelException is thrown to unwind stack on cancellation
            // it happens either at the beginning (parallel partition) or during samples sorting
            // so we cancel at random times and hope that we catch the proper moment =(
            double ratio = std::uniform_real_distribution<double>(0.0, 1.0)(random);
            std::this_thread::sleep_for(fullRunTime * ratio);
            taskGroup.cancel();
            taskGroup.wait();
        }
    }
}
