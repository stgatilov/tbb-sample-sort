#include <stdio.h>
#include <assert.h>
#include <span>
#include <chrono>

#include "tbbss.h"

#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/parallel_sort.h>


std::vector<uint64_t> readBinFile(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f)
        std::terminate();
    uint64_t num = 0;
    if (fread(&num, sizeof(num), 1, f) != 1)
        std::terminate();
    std::vector<uint64_t> data(num);
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

//#define REDUCE_BITS 40
//#define TBB_FORCE_THREADS 1

int main() {
#ifdef TBB_FORCE_THREADS
    tbb::global_control control(tbb::global_control::max_allowed_parallelism, TBB_FORCE_THREADS);
#endif

    auto t0 = getTimestamp();
    std::vector<uint64_t> input = readBinFile("input.bin");
#ifdef REDUCE_BITS
    for (uint64_t &x : input)
        x >>= REDUCE_BITS;
#endif
    auto t1 = getTimestamp();
    printf("%4.0lf ms : read input\n", getTimeDiff(t0, t1));

#if 1
    tbbss::sampleSort(input.data(), input.size());
#else
    tbb::parallel_sort(input.data(), input.data() + input.size());
#endif
    auto t2 = getTimestamp();
    printf("%4.0lf ms : sort\n", getTimeDiff(t1, t2));

    std::vector<size_t> badpos;
    for (size_t i = 1; i < input.size(); i++)
        if (input[i] < input[i - 1])
            badpos.push_back(i - 1);

    if (!badpos.empty()) {
        printf("FAILED: %zu of %zu\n", badpos.size(), input.size());
        for (size_t i = 0; i < std::min(int(badpos.size()), 100); i++)
            printf("  [%zu .. %zu]: %zu > %zu\n", badpos[i], badpos[i] + 1, input[badpos[i]], input[badpos[i] + 1]);
        std::terminate();
    }

    assert(std::is_sorted(input.data(), input.data() + input.size()));
    auto t3 = getTimestamp();
    printf("%4.0lf ms : validate\n", getTimeDiff(t2, t3));

    return 0;
}
