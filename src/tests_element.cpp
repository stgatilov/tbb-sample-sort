#include "tests_common.h"

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
