#include <stdio.h>
#include <random>

int main() {
    FILE *f = fopen("input.bin", "wb");

    uint64_t num = 10 * 1000 * 1000;
    fwrite(&num, sizeof(num), 1, f);

    std::mt19937_64 random;
    for (uint64_t i = 0; i < num; i++) {
        uint64_t value = random();
        fwrite(&value, sizeof(value), 1, f);
    }

    fclose(f);
    return 0;
}
