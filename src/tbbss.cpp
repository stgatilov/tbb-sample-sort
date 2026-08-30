/**
 * (C) Copyright Stepan Gatilov 2026.
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#include "tbbss.h"

#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <algorithm>

#include <xmmintrin.h>

#if TBBSS_LARGE_PAGES
    #include <sys/mman.h>
#endif

namespace tbbss {

void memcpyUncached(char *dst, const char *src, size_t size) {
    assert(dst >= src + size || src >= dst + size);

    size_t prefix = std::min(64 - size_t(dst) % 64, size);
    memcpy(dst, src, prefix);
    dst += prefix;
    src += prefix;
    size -= prefix;

    while (size >= 64) {
        _mm_stream_si128((__m128i*)(dst + 0), _mm_loadu_si128((__m128i*)(src + 0)));
        _mm_stream_si128((__m128i*)(dst + 16), _mm_loadu_si128((__m128i*)(src + 16)));
        _mm_stream_si128((__m128i*)(dst + 32), _mm_loadu_si128((__m128i*)(src + 32)));
        _mm_stream_si128((__m128i*)(dst + 48), _mm_loadu_si128((__m128i*)(src + 48)));
        dst += 64;
        src += 64;
        size -= 64;
    }

    memcpy(dst, src, size);
}

static std::pair<size_t, bool> selectAlignment(size_t bytes, size_t reqAlign) {
    size_t baseAlignment = 64;
    bool useLargePages = false;
#if TBBSS_LARGE_PAGES
    if (bytes >= 8 * TBBSS_LARGE_PAGES) {
        useLargePages = true;
        baseAlignment = TBBSS_LARGE_PAGES;
    }
#endif
    size_t alignment = std::max<size_t>(reqAlign, baseAlignment);
    return {alignment, useLargePages};
}

void *allocateMemory(size_t bytes, size_t reqAlign) {
    auto [alignment, useLargePages] = selectAlignment(bytes, reqAlign);
    void *ptr = operator new (bytes, std::align_val_t(alignment));
#if TBBSS_LARGE_PAGES
    if (useLargePages)
        madvise(ptr, bytes, MADV_HUGEPAGE);
#endif
    return ptr;
}

void deallocateMemory(void *ptr, size_t bytes, size_t reqAlign) {
    auto [alignment, useLargePages] = selectAlignment(bytes, reqAlign);
    operator delete (ptr, bytes, std::align_val_t(alignment));
}

}
