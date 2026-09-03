#include "tests_common.h"

#include <condition_variable>
#include <atomic>

#include <tbb/parallel_for.h>
#include <tbb/task_group.h>

TEST_CASE("Simple") {
    Random random;
    std::uniform_int_distribution<int> distr;
    std::vector<int> arr = generateArray<int>(DEFAULT_ARRAY_SIZE, [&]{
        return distr(random);
    });

    checkUnsorted(arr.data(), arr.size());
    tbbss::sampleSort(arr.data(), arr.size());
    checkSorted(arr.data(), arr.size());
}

TEST_CASE("Comparator") {
    Random random;
    std::uniform_int_distribution<int> distrIndex(0, DEFAULT_ARRAY_SIZE - 1);
    std::vector<int> arr = generateArray<int>(DEFAULT_ARRAY_SIZE, [&]{
        return distrIndex(random);
    });

    std::uniform_int_distribution<int> distrKey;
    struct Comparator {
        std::vector<int> keyStore;
        bool operator() (int a, int b) const {
            return keyStore[a] < keyStore[b];
        }
        // for some reason std::sort & std::is_sorted accept comparator by value
        // sampleSort accepts by reference, so let's check we don't copy it accidentally
        Comparator() = default;
        Comparator(const Comparator&) = delete;
        Comparator &operator=(const Comparator&) = delete;
    } comp;
    comp.keyStore = generateArray<int>(DEFAULT_ARRAY_SIZE, [&]{
        return distrKey(random);
    });

    tbbss::sampleSort(arr.data(), arr.size(), comp);
    checkUnsorted(arr.data(), arr.size());
    checkSorted(arr.data(), arr.size(), comp);
}

TEST_CASE("Equal1000") {
    typedef IntegerElement<int16_t, 2> Element;
    Random random;
    std::uniform_int_distribution<int16_t> distr(1000, 2000);
    std::vector<Element> arr = generateArray<Element>(DEFAULT_ARRAY_SIZE, [&]{
        Element x;
        x.key = distr(random);
        fillValue(x, random);
        return x;
    });

    runAndValidateSampleSort(arr);
}

TEST_CASE("Equal17") {
    typedef IntegerElement<char, 3> Element;
    Random random;
    std::uniform_int_distribution<int> distr(-42, -17);
    std::vector<Element> arr = generateArray<Element>(DEFAULT_ARRAY_SIZE, [&]{
        Element x;
        x.key = char(distr(random));
        fillValue(x, random);
        return x;
    });

    runAndValidateSampleSort(arr);
}


TEST_CASE("SortedIncr") {
    Random random;
    std::uniform_int_distribution<int> distr;
    std::vector<int> arr = generateArray<int>(DEFAULT_ARRAY_SIZE, [&]{
        return distr(random);
    });
    std::sort(arr.data(), arr.data() + arr.size());

    runAndValidateSampleSort(arr);
}

TEST_CASE("SortedDecr") {
    Random random;
    std::uniform_int_distribution<int> distr;
    std::vector<int> arr = generateArray<int>(DEFAULT_ARRAY_SIZE, [&]{
        return distr(random);
    });
    std::sort(arr.data(), arr.data() + arr.size(), std::greater<int>());

    runAndValidateSampleSort(arr);
}

TEST_CASE("SortedEqual") {
    Random random;
    std::uniform_int_distribution<int> distr;
    std::vector<int> arr = generateArray<int>(DEFAULT_ARRAY_SIZE, [&]{
        return 42;
    });

    runAndValidateSampleSort(arr);
}

TEST_CASE("StdString") {
    typedef std::string Element;
    Random random;
    std::uniform_int_distribution<int> distr;
    std::vector<Element> arr = generateArray<Element>(DEFAULT_ARRAY_SIZE, [&]{
        std::string res = std::to_string(distr(random));
        // make sure some strings are dynamically allocated
        if ((random() & 3) == 0)
            res = res + res + res + res;
        return res;
    });

    // note: RelocationTrivialness::rtNone is required for std::string
    // since it contains self-references due to SSO
    runAndValidateSampleSort(arr);
}

TEST_CASE("StdVector") {
    typedef std::vector<int> Element;
    Random random;
    std::uniform_int_distribution<int> distrLen(1, 3);
    std::uniform_int_distribution<int> distrElem(0, 1000);
    std::vector<Element> arr = generateArray<Element>(DEFAULT_ARRAY_SIZE, [&]{
        int l = distrLen(random);
        Element res;
        for (int i = 0; i < l; i++)
            res.push_back(distrElem(random));
        return res;
    });

    tbbss::sampleSort<
        Element, std::less<Element>,
        // note: std::vector is trivially relocatable and (normally) copyable
        // any RelocationTrivialness mode should work, although "fork" is preferred
        tbbss::DefaultValueTraits<Element, tbbss::rtRelocate>
    > (arr.data(), arr.size());
    checkSorted(arr.data(), arr.size());
}

TEST_CASE("StdUniquePtr") {
    typedef std::unique_ptr<int> Element;
    Random random;
    std::uniform_int_distribution<int> distr;
    std::vector<Element> arr = generateArray<Element>(DEFAULT_ARRAY_SIZE, [&]{
        return std::make_unique<int>(distr(random));
    });
    auto comparator = [](const Element& a, const Element& b) {
        return *a < *b;
    };

    tbbss::sampleSort<
        Element, std::decay_t<decltype(comparator)>,
        // note: this type is no-copyable, thus "fork" mode is required to compile
        // luckily, std::unique_ptr is indeed trivially relocatable
        tbbss::DefaultValueTraits<Element, tbbss::rtFork>
    > (arr.data(), arr.size(), comparator);
    checkSorted(arr.data(), arr.size(), comparator);
}

TEST_CASE("Determinism") {
    typedef IntegerElement<int16_t, 2> Element;
    Random random;
    std::vector<Element> arr = generateArray<Element>(DEFAULT_ARRAY_SIZE, [&]{
        Element x;
        x.key = random() % 1000 + 1000;
        fillValue(x, random);
        return x;
    });

    auto computeHash = [](const std::vector<Element> &arr) {
        static const uint32_t MOD = (1u << 31) - 1;
        static const uint32_t BASE = 3;
        uint32_t hash = 0;
        for (const Element &x : arr) {
            static_assert(sizeof(x) == 4);
            uint32_t raw = *reinterpret_cast<const uint32_t*>(&x);
            hash = (uint64_t(hash) * BASE + raw) % MOD;
        }
        return hash;
    };

    // make sure we generate input data deterministically across platforms
    uint32_t inputHash = computeHash(arr);
    constexpr uint32_t INPUT_PRERECORDED = 573506037u;
    CHECK_EQ(inputHash, INPUT_PRERECORDED);

    // check that output is exactly the same on repeated runs of the function
    // (input contains many equal elements, otherwise sorted output is unique anyway)
    std::vector<Element> firstOutput;
    for (int r = 0; r < 20; r++) {
        std::vector<Element> copy = arr;
        tbbss::sampleSort(copy.data(), copy.size());
        if (firstOutput.empty())
            firstOutput = copy;
        checkExactlyEqual(firstOutput.data(), copy.data(), arr.size());
    }

    // compare hash of the output against prerecorded hash
    // this ensures determinism across platforms as well
    uint32_t outputHash = computeHash(firstOutput);
    constexpr uint32_t OUTPUT_PRERECORDED = 1051995632u;
    CHECK_EQ(outputHash, OUTPUT_PRERECORDED);
}

TEST_CASE("ParallelSorts") {
    static constexpr int Tasks = 20;
    typedef IntegerElement<int32_t, 0> Element;
    Random random;
    std::uniform_int_distribution<int> distr(0, DEFAULT_ARRAY_SIZE / 3);
    std::vector<Element> arr = generateArray<Element>(DEFAULT_ARRAY_SIZE * Tasks, [&]{
        Element x;
        x.key = distr(random);
        fillValue(x, random);
        return x;
    });

    std::uniform_int_distribution<size_t> distrSplit(0, arr.size() - 1);
    std::vector<size_t> splits = {0, arr.size()};
    for (int t = 0; t < Tasks - 1; t++)
        splits.push_back(distrSplit(random));
    std::sort(splits.begin(), splits.end());

    std::vector<Element> check = arr;
    // check running many independent sorts as TBB tasks
    // inaccurate usage of threadlocal can break this use case
    tbb::parallel_for<int>(0, Tasks, [&](int t) {
        size_t beg = splits[t + 0];
        size_t end = splits[t + 1];
        tbbss::sampleSortIter(arr.data() + beg, arr.data() + end);
    });
    tbb::parallel_for<int>(0, Tasks, [&](int t) {
        size_t beg = splits[t + 0];
        size_t end = splits[t + 1];
        std::sort(check.data() + beg, check.data() + end);
    });
    checkExactlyEqual(arr.data(), check.data(), arr.size());
}

void TbbSleepTest(int numSleeping) {
    Random random;
    std::uniform_int_distribution<int> distr;
    std::vector<int> arr = generateArray<int>(DEFAULT_ARRAY_SIZE, [&]{
        return distr(random);
    });

    tbb::task_group sortGroup, waitGroup;
    std::condition_variable variable;
    std::mutex mutex;
    std::atomic_bool finished = false;
    auto waitUntilFinished = [&] {
        std::unique_lock lock(mutex);
        variable.wait(lock, [&]{ return finished.load(); });
    };

    // block specified number of threads (one of them = main thread)
    assert(numSleeping >= 1);

    for (int t = 0; t < numSleeping - 1; t++) {
        waitGroup.run([&] {
            waitUntilFinished();
        });
    }
    sortGroup.run([&] {
        tbbss::sampleSort(arr.data(), arr.size());
        finished.store(true);
        variable.notify_all();
    });
    // block main thread as well
    waitUntilFinished();

    sortGroup.wait();
    waitGroup.wait();
    checkSorted(arr.data(), arr.size());
}

TEST_CASE("TbbSleep1") {
    // we limit number of actually available TBB threads to 1
    // the algorithm runs more tasks than threads but does not deadlock
    TbbSleepTest(tbb::this_task_arena::max_concurrency() - 1);
}

TEST_CASE("TbbSleep0"
    * doctest::skip()
    * doctest::should_fail()
) {
    // we block all threads, thus sorting task never starts and we get deadlock
    TbbSleepTest(tbb::this_task_arena::max_concurrency());
}

#if SIZE_MAX >= UINT64_MAX  // 64-bit architecture only
TEST_CASE("Index64Bit"
    // uses gigabytes of RAM and takes about a minute (with optimization)
    // hence disabled by default
    * doctest::skip()
) {
    constexpr size_t NumElems = size_t(2) << 32;
    typedef uint16_t Element;

    Random random;
    std::uniform_int_distribution<Element> distr;
    std::vector<Element> arr = generateArray<Element>(NumElems, [&]{
        return distr(random);
    });

    tbbss::sampleSort(arr.data(), arr.size());
    checkSorted(arr.data(), arr.size());
}
#endif

constexpr size_t BENCHMARK_ARRAY_SIZE = 300 * 1000 * 1000;

TEST_CASE("Benchmark") {
    typedef int64_t Element;
    Random random;
    std::vector<Element> arr = generateArray<Element>(BENCHMARK_ARRAY_SIZE, [&]{
        return random();
    });

    std::vector<Element> copy = arr;
    auto Tbefore = std::chrono::steady_clock::now();
    tbbss::sampleSort(arr.data(), arr.size());
    auto Tafter = std::chrono::steady_clock::now();
    double time0 = std::chrono::duration<double, std::milli>(Tafter - Tbefore).count();
    checkSorted(arr.data(), arr.size());

    arr = copy;
    Tbefore = std::chrono::steady_clock::now();
    tbbss::sampleSort(arr.data(), arr.size());
    Tafter = std::chrono::steady_clock::now();
    double time1 = std::chrono::duration<double, std::milli>(Tafter - Tbefore).count();
    checkSorted(arr.data(), arr.size());

    arr = copy;
    Tbefore = std::chrono::steady_clock::now();
    tbbss::sampleSort(arr.data(), arr.size());
    Tafter = std::chrono::steady_clock::now();
    double time2 = std::chrono::duration<double, std::milli>(Tafter - Tbefore).count();
    checkSorted(arr.data(), arr.size());

    printf("Sorting %zuM random 64-bit integers (ms): %0.0lf %0.0lf %0.0lf\n",
        BENCHMARK_ARRAY_SIZE / 1000 / 1000,
        time0, time1, time2
    );
}
