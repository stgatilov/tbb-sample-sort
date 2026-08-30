#include <doctest/doctest.h>
#include <tbbss.h>

#include <vector>
#include <random>

#include <fmt/format.h>

//----------------------------------------------------------------------------------------------

typedef std::mt19937_64 Random;

template<class T, class Lambda> std::vector<T> generateArray(size_t num, Lambda &&genElem) {
    std::vector<T> res;
    res.resize(num);
    for (size_t i = 0; i < num; i++)
        res[i] = genElem();
    return res;
}

//----------------------------------------------------------------------------------------------

#pragma pack(push, 1)
template<class Key, size_t ValueBytes> struct IntegerElement {
    Key key;
    char value[ValueBytes];
};
template<class Key> struct IntegerElement<Key, 0> {
    Key key;
};
#pragma pack(pop)

template<class Key, size_t ValueBytes> bool operator< (const IntegerElement<Key, ValueBytes> &a, const IntegerElement<Key, ValueBytes> &b) {
    return a.key < b.key;
}

template<class Key, size_t ValueBytes> int totalCompare(const IntegerElement<Key, ValueBytes> &a, const IntegerElement<Key, ValueBytes> &b) {
    if (a.key != b.key)
        return a.key < b.key ? -1 : 1;
    if constexpr (ValueBytes > 0) {
        for (size_t i = 0; i < ValueBytes; i++)
            if (a.value[i] != b.value[i])
                return a.value[i] < b.value[i] ? -1 : 1;
    }
    return 0;
}

template<class Key, size_t ValueBytes> void fillValue(IntegerElement<Key, ValueBytes> &elem, Random &random) {
    if constexpr (ValueBytes > 0) {
        for (size_t i = 0; i < ValueBytes; i++)
            elem.value[i] = char(uint8_t(random()));
    }
}

//----------------------------------------------------------------------------------------------

template<class T, class Comparator = std::less<T>> void checkSorted(T *begin, size_t n, const Comparator &comp = Comparator()) {
    std::vector<size_t> badpos;
    for (size_t i = 1; i < n; i++)
        if (comp(begin[i], begin[i - 1]))
            badpos.push_back(i - 1);

    CHECK_MESSAGE(badpos.empty(), fmt::format("{:d}/{:d} bad positions; first one at {:d}", badpos.size(), n, badpos.front()));
}

template<class T, class Comparator = std::less<T>> void checkUnsorted(T *begin, size_t n, const Comparator &comp = Comparator()) {
    std::vector<size_t> badpos;
    for (size_t i = 1; i < n; i++)
        if (comp(begin[i], begin[i - 1]))
            badpos.push_back(i - 1);

    WARN_MESSAGE(!badpos.empty(), "array is sorted unexpectedly");
}

template<class T> struct TotalLess {
    bool operator() (const T &a, const T &b) const {
        return totalCompare(a, b) < 0;
    }
};

template<class T> void checkExactlyEqual(T *begin1, T *begin2, size_t n) {
    std::vector<size_t> badpos;
    for (size_t i = 0; i < n; i++)
        if (totalCompare(begin1[i], begin2[i]) != 0)
            badpos.push_back(i);

    CHECK_MESSAGE(badpos.empty(), fmt::format("{:d}/{:d} bad positions; first one at {:d}", badpos.size(), n, badpos.front()));
}

template<class T> void runAndValidateSampleSort(std::vector<T> &arr) {
    std::vector<T> check = arr;

    tbbss::sampleSort(arr.data(), arr.size());

    checkSorted(arr.data(), arr.size());

    std::sort(check.data(), check.data() + check.size(), TotalLess<T>());
    std::sort(arr.data(), arr.data() + arr.size(), TotalLess<T>());
    checkExactlyEqual(arr.data(), check.data(), arr.size());
}

//----------------------------------------------------------------------------------------------

constexpr size_t DEFAULT_ARRAY_SIZE = 1000123;

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

TEST_CASE_TEMPLATE("Element", Element,
    IntegerElement<int8_t, 0>,
    IntegerElement<int16_t, 0>,
    IntegerElement<int16_t, 1>,
    IntegerElement<int32_t, 0>,
    IntegerElement<int32_t, 1>,
    IntegerElement<int32_t, 2>,
    IntegerElement<int32_t, 3>,
    IntegerElement<int64_t, 0>,
    IntegerElement<int64_t, 1>,
    IntegerElement<int64_t, 2>,
    IntegerElement<int64_t, 3>,
    IntegerElement<int64_t, 4>,
    IntegerElement<int64_t, 5>,
    IntegerElement<int64_t, 6>,
    IntegerElement<int64_t, 7>,
    IntegerElement<int64_t, 8>,
    IntegerElement<int64_t, 11>,
    IntegerElement<int64_t, 16>,
    IntegerElement<int64_t, 27>
) {
    // most importantly, we want to test memcpyElement and memswapElementConditional
    static_assert(tbbss::IsTriviallyRelocatable<Element> == tbbss::rtFork);
    Random random;
    std::uniform_int_distribution<int64_t> distr;
    std::vector<Element> arr = generateArray<Element>(DEFAULT_ARRAY_SIZE, [&]{
        Element x;
        x.key = distr(random);
        fillValue(x, random);
        return x;
    });
    runAndValidateSampleSort(arr);
}
