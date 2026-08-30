#include <doctest/doctest.h>
#include <tbbss.h>

#include <vector>
#include <random>

template<class T> T *pbegin(std::vector<T> &arr) { return arr.data(); }
template<class T> T *pend(std::vector<T> &arr) { return arr.data() + arr.size(); }

template<class T, class Lambda> std::vector<T> generateArray(size_t num, Lambda &&genElem) {
    std::vector<T> res;
    res.resize(num);
    for (size_t i = 0; i < num; i++)
        res[i] = genElem();
    return res;
}

template<class T, class Comparator> void checkSorted(T *begin, size_t n, const Comparator &comp) {
    for (size_t i = 1; i < n; i++)
        REQUIRE(!comp(begin[i], begin[i-1]));
}

TEST_CASE("Simple") {
    std::mt19937 random;
    std::uniform_int_distribution<int> distr;
    std::vector<int> arr = generateArray<int>(1000000, [&]{ return distr(random); });
    tbbss::sampleSort(arr.data(), arr.size());
    checkSorted(arr.data(), arr.size(), std::less<int>());
}
