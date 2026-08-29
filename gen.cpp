#include <stdio.h>
#include <random>

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Parameters: sizeof(key) sizeof(value) numberOfElements {keyBits}\n");
        return 123;
    }
    uint32_t keyBytes;
    uint32_t valueBytes;
    unsigned long long numElems;
    sscanf(argv[1], "%u", &keyBytes);
    sscanf(argv[2], "%u", &valueBytes);
    sscanf(argv[3], "%llu", &numElems);

    uint32_t keyBits = 8 * keyBytes;
    if (argc >= 5)
        sscanf(argv[4], "%u", &keyBits);

    FILE *f = fopen("input.bin", "wb");

    fwrite(&numElems, sizeof(numElems), 1, f);

    std::mt19937_64 random;
    for (uint64_t i = 0; i < numElems; i++) {
        uint64_t key = random();
        uint64_t value = random();
        if (keyBits < 8 * keyBytes)
            key &= (1ull << keyBits) - 1;
        fwrite(&key, keyBytes, 1, f);
        fwrite(&value, valueBytes, 1, f);
    }

    fclose(f);
    return 0;
}
