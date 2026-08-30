#include "tests_common.h"

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
    std::uniform_int_distribution<int> distr(1000, 2000);
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
        x.key = distr(random);
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
