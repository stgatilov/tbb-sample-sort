#pragma once

#include <doctest/doctest.h>
#include <tbbss.h>

#include <stdio.h>
#include <vector>
#include <string>
#include <memory>
#include <random>

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
template<class TKey, size_t ValueBytes> struct IntegerElement {
    typedef TKey Key;
    Key key;
    char value[ValueBytes];
};
template<class TKey> struct IntegerElement<TKey, 0> {
    typedef TKey Key;
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

template<class T, typename = std::enable_if_t<std::is_integral_v<T>>> int totalCompare(const T &a, const T &b) {
    if (a != b)
        return a < b ? -1 : 1;
    return 0;
}

template<class T, typename = std::enable_if_t<std::is_integral_v<T>>> int totalCompare(const std::unique_ptr<T> &a, const std::unique_ptr<T> &b) {
    return totalCompare(*a, *b);
}

inline int totalCompare(const std::string &a, const std::string &b) {
    return a.compare(b);
}

template<class T> struct TotalLess {
    bool operator() (const T &a, const T &b) const {
        return totalCompare(a, b) < 0;
    }
};

//----------------------------------------------------------------------------------------------

template<class T, class Comparator = std::less<T>> void checkSorted(T *begin, size_t n, const Comparator &comp = Comparator()) {
    std::vector<size_t> badpos;
    for (size_t i = 1; i < n; i++)
        if (comp(begin[i], begin[i - 1]))
            badpos.push_back(i - 1);

    char message[256];
    sprintf(message, "%zu/%zu bad positions; first one at %zu", badpos.size(), n, (badpos.empty() ? 0 : badpos.front()));
    CHECK_MESSAGE(badpos.empty(), message);
}

template<class T, class Comparator = std::less<T>> void checkUnsorted(T *begin, size_t n, const Comparator &comp = Comparator()) {
    std::vector<size_t> badpos;
    for (size_t i = 1; i < n; i++)
        if (comp(begin[i], begin[i - 1]))
            badpos.push_back(i - 1);

    WARN_MESSAGE(!badpos.empty(), "array is sorted unexpectedly");
}

template<class T> void checkExactlyEqual(T *begin1, T *begin2, size_t n) {
    std::vector<size_t> badpos;
    for (size_t i = 0; i < n; i++)
        if (totalCompare(begin1[i], begin2[i]) != 0)
            badpos.push_back(i);

    char message[256];
    sprintf(message, "%zu/%zu bad positions; first one at %zu", badpos.size(), n, (badpos.empty() ? 0 : badpos.front()));
    CHECK_MESSAGE(badpos.empty(), message);
}

template<class T> void runAndValidateSampleSort(std::vector<T> &arr) {
    std::vector<T> check = arr;

    tbbss::sampleSort(arr.data(), arr.size());

    checkSorted(arr.data(), arr.size());

    std::sort(check.data(), check.data() + check.size(), TotalLess<T>());
    std::sort(arr.data(), arr.data() + arr.size(), TotalLess<T>());
    checkExactlyEqual(arr.data(), check.data(), arr.size());
}

constexpr inline size_t DEFAULT_ARRAY_SIZE = 1000123;
