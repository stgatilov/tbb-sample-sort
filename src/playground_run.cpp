#include "tbbss.h"

#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <random>

#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/parallel_sort.h>

#pragma pack(push, 1)
struct Element {
    uint64_t key;
    #if 0
        char value[4];
    #endif

    bool operator< (const Element &b) const {
        return key < b.key;
    }
};
#pragma pack(pop)

std::vector<Element> readBinFile(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f)
        std::terminate();
    uint64_t num64 = 0;
    if (fread(&num64, sizeof(num64), 1, f) != 1)
        std::terminate();
    size_t num = size_t(num64);
    std::vector<Element> data(num);
    if (fread(data.data(), sizeof(data[0]), num, f) != num)
        std::terminate();
    fclose(f);
    return data;
}

std::chrono::steady_clock::time_point getTimestamp() {
    return std::chrono::steady_clock::now();
}

double getTimeDiff(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

//#define TBB_FORCE_THREADS 1
#define TBB_PARALLEL_SORTS 1

int main() {
#ifdef TBB_FORCE_THREADS
    tbb::global_control control(tbb::global_control::max_allowed_parallelism, TBB_FORCE_THREADS);
#endif

    auto t0 = getTimestamp();
    std::vector<Element> input = readBinFile("input.bin");
    auto t1 = getTimestamp();
    printf("%4.0lf ms : read input\n", getTimeDiff(t0, t1));

    size_t numRanges = TBB_PARALLEL_SORTS;
    std::vector<size_t> splits;
    splits.push_back(0);
    splits.push_back(input.size());
    std::mt19937 rnd;
    for (size_t i = 0; i < numRanges - 1; i++)
        splits.push_back(rnd() % input.size());
    std::sort(splits.data(), splits.data() + splits.size());

    tbb::parallel_for<size_t>(0, numRanges, [&](size_t r) {
        size_t beg = splits[r + 0];
        size_t end = splits[r + 1];
#if 1
        tbbss::sampleSort(input.data() + beg, end - beg);
#else
        tbb::parallel_sort(input.data() + beg, input.data() + end);
#endif
    });
    auto t2 = getTimestamp();
    printf("%4.0lf ms : sort[%zu]\n", getTimeDiff(t1, t2), numRanges);

    std::vector<size_t> badpos;
    for (size_t r = 0; r < numRanges; r++) {
        for (size_t i = splits[r] + 1; i < splits[r + 1]; i++)
            if (input[i] < input[i - 1])
                badpos.push_back(i - 1);
    }

    if (!badpos.empty()) {
        typedef unsigned long long uint64t;
        printf("FAILED: %zu of %zu\n", badpos.size(), input.size());
        for (int i = 0; i < std::min(int(badpos.size()), 100); i++)
            printf("  [%zu .. %zu]: %llu > %llu\n", badpos[i], badpos[i] + 1, uint64t(input[badpos[i]].key), uint64t(input[badpos[i] + 1].key));
        std::terminate();
    }

    for (size_t r = 0; r < numRanges; r++) {
        assert(std::is_sorted(input.data() + splits[r], input.data() + splits[r + 1]));
    }
    auto t3 = getTimestamp();
    printf("%4.0lf ms : validate\n", getTimeDiff(t2, t3));

    return 0;
}
