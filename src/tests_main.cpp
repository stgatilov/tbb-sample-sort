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

TEST_CASE("StdString") {
    typedef std::string Element;
    Random random;
    std::uniform_int_distribution<int> distr;
    std::vector<Element> arr = generateArray<Element>(DEFAULT_ARRAY_SIZE, [&]{
        return std::to_string(distr(random));
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
        // note: std::vector is trivially relocatable and copyable
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
