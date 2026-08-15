#include <stdio.h>
#include <assert.h>

#include <random>

#include <oneapi/tbb/parallel_for.h>


std::vector<uint64_t> readBinFile(const char *filename) {
    FILE *f = fopen(filename, "rb");
    uint64_t num = 0;
    fread(&num, sizeof(num), 1, f);
    std::vector<uint64_t> data(num);
    fread(data.data(), sizeof(data[0]), num, f);
    fclose(f);
    return data;
}

//===============================================

bool isPot(size_t x) {
    return (x & (x - 1)) == 0;
}

size_t log2up(size_t x) {
    size_t res = 0;
    while ((1 << res) < x)
        res++;
    return res;
}


typedef uint64_t Value;
typedef Value *Iter;
typedef const Value *CIter;

struct Random {
    std::mt19937_64 gen;

    size_t generate(size_t maxExclusive) {
        return gen() % maxExclusive;
    }
};

struct MultiPivot {
    size_t numBits = 0;
    size_t numBuckets = 0;
    std::vector<Value> sorted;
    std::vector<Value> tree;
};

void selectMultiPivot(
    MultiPivot &pivot, Random& random,
    CIter arr, size_t numElems, size_t numBuckets
) {
    assert(isPot(numBuckets) && numBuckets >= 2);

    size_t numSamples = numBuckets * log2up(numElems) / 5;
    numSamples = std::max(numSamples, numBuckets);

    std::vector<Value> samples;
    samples.resize(numSamples);
    for (size_t i = 0; i < numSamples; i++) {
        size_t index = random.generate(numElems);
        samples[i] = arr[index];
    }

    std::sort(samples.data(), samples.data() + samples.size());

    pivot.sorted.resize(numBuckets - 1);
    for (size_t i = 1; i <= numBuckets - 1; i++) {
        uint64_t pos = uint64_t(numSamples) * i / numBuckets;
        pivot.sorted[i - 1] = samples[pos];
    }

    size_t numBits = log2up(numBuckets);

    pivot.tree.resize(numBuckets - 1);
    size_t v = 0;
    for (int b = numBits - 1; b >= 0; b--) {
        size_t len = (1 << b);
        for (size_t i = len - 1; i < numBuckets; i += len * 2)
            pivot.tree[v++] = pivot.sorted[i];
    }

    pivot.numBuckets = numBuckets;
    pivot.numBits = numBits;
}

inline size_t classifyOne(const MultiPivot &pivot, const Value &value) {
    size_t res = 0;
    for (size_t b = 0; b < pivot.numBits; b++) {
        bool isLess = (value < pivot.tree[res]);
        res = 2 * res + 1 + size_t(!isLess);
    }
    res -= (pivot.numBuckets - 1);
    assert(res == pivot.numBuckets - 1 || (value < pivot.sorted[res]));
    assert(res == 0 || !(value < pivot.sorted[res - 1]));
    return res;
}

    
Random myrandom;
std::vector<size_t> splits;
std::vector<uint8_t> bucketOf;
void multiPartition(
    Iter src, size_t num,
    Iter dst
) {
    MultiPivot pivot;
    selectMultiPivot(pivot, myrandom, src, num, 256);

    splits.assign(pivot.numBuckets + 1, 0);
    bucketOf.resize(num);
    for (size_t i = 0; i < num; i++) {
        size_t b = classifyOne(pivot, src[i]);
        bucketOf[i] = b;
        splits[b + 1]++;
    }

    for (size_t b = 1; b <= pivot.numBuckets; b++)
        splits[b] += splits[b - 1];

    for (size_t i = 0; i < num; i++) {
        size_t b = bucketOf[i];
        dst[splits[b]++] = src[i];
    }

    for (size_t b = pivot.numBuckets; b > 0; b--)
        splits[b] = splits[b - 1];
    splits[0] = 0;
}

//===============================================

int main() {
    std::vector<uint64_t> input = readBinFile("input.bin");

    std::vector<uint64_t> temp(input.size());
    multiPartition(input.data(), input.size(), temp.data());

    for (uint64_t b = 0; b + 1 < splits.size(); b++)
        std::sort(temp.data() + splits[b], temp.data() + splits[b + 1]);

    assert(std::is_sorted(temp.data(), temp.data() + temp.size()));

    return 0;
}
