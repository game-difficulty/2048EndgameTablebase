#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif

#include "UniqueUtils.h"

#if defined(_WIN32)
#define BOOKSOLVER_EXPORT extern "C" __declspec(dllexport)
#else
#define BOOKSOLVER_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr uint32_t kBookSolverIndexLength = 16777217u;
constexpr int kValueKindU32 = 0;
constexpr int kValueKindU64 = 1;
constexpr int kValueKindF32 = 2;
constexpr int kValueKindF64 = 3;

struct HeaderTransition {
    uint32_t header = 0;
    uint32_t position = 0;
};

inline uint64_t load_key(const char *record) {
    uint64_t key = 0;
    std::memcpy(&key, record, sizeof(uint64_t));
    return key;
}

inline uint32_t load_u32_value(const char *record) {
    uint32_t value = 0;
    std::memcpy(&value, record + sizeof(uint64_t), sizeof(uint32_t));
    return value;
}

inline uint64_t load_u64_value(const char *record) {
    uint64_t value = 0;
    std::memcpy(&value, record + sizeof(uint64_t), sizeof(uint64_t));
    return value;
}

inline float load_f32_value(const char *record) {
    float value = 0.0f;
    std::memcpy(&value, record + sizeof(uint64_t), sizeof(float));
    return value;
}

inline double load_f64_value(const char *record) {
    double value = 0.0;
    std::memcpy(&value, record + sizeof(uint64_t), sizeof(double));
    return value;
}

inline void copy_record(char *dst, const char *src, size_t record_size) {
    if (dst == src) {
        return;
    }
    switch (record_size) {
    case 12: {
        uint64_t key = 0;
        uint32_t value = 0;
        std::memcpy(&key, src, sizeof(uint64_t));
        std::memcpy(&value, src + sizeof(uint64_t), sizeof(uint32_t));
        std::memcpy(dst, &key, sizeof(uint64_t));
        std::memcpy(dst + sizeof(uint64_t), &value, sizeof(uint32_t));
        return;
    }
    case 16: {
        uint64_t lo = 0;
        uint64_t hi = 0;
        std::memcpy(&lo, src, sizeof(uint64_t));
        std::memcpy(&hi, src + sizeof(uint64_t), sizeof(uint64_t));
        std::memcpy(dst, &lo, sizeof(uint64_t));
        std::memcpy(dst + sizeof(uint64_t), &hi, sizeof(uint64_t));
        return;
    }
    default:
        std::memcpy(dst, src, record_size);
        return;
    }
}

inline bool record_passes_threshold(const char *record, int value_kind, uint64_t threshold_bits) {
    switch (value_kind) {
    case kValueKindU32:
        return load_u32_value(record) > static_cast<uint32_t>(threshold_bits);
    case kValueKindU64:
        return load_u64_value(record) > threshold_bits;
    case kValueKindF32: {
        uint32_t bits32 = static_cast<uint32_t>(threshold_bits);
        float threshold = 0.0f;
        std::memcpy(&threshold, &bits32, sizeof(float));
        return load_f32_value(record) > threshold;
    }
    case kValueKindF64: {
        double threshold = 0.0;
        std::memcpy(&threshold, &threshold_bits, sizeof(double));
        return load_f64_value(record) > threshold;
    }
    default:
        return false;
    }
}

inline size_t filter_records_scalar_chunk(const char *input,
                                          size_t begin,
                                          size_t end,
                                          size_t record_size,
                                          int value_kind,
                                          uint64_t threshold_bits,
                                          char *output) {
    size_t out = 0;
    for (size_t i = begin; i < end; ++i) {
        const char *record = input + i * record_size;
        if (!record_passes_threshold(record, value_kind, threshold_bits)) {
            continue;
        }
        if (output != nullptr) {
            std::memcpy(output + out * record_size, record, record_size);
        }
        ++out;
    }
    return out;
}

inline size_t filter_records_scalar_inplace(char *data,
                                            size_t count,
                                            size_t record_size,
                                            int value_kind,
                                            uint64_t threshold_bits) {
    size_t out = 0;
    for (size_t i = 0; i < count; ++i) {
        const char *record = data + i * record_size;
        if (!record_passes_threshold(record, value_kind, threshold_bits)) {
            continue;
        }
        copy_record(data + out * record_size, record, record_size);
        ++out;
    }
    return out;
}

#if defined(__GNUC__) || defined(__clang__)

__attribute__((target("avx512f")))
inline void expand_record_mask_12(unsigned keep_mask,
                                  __mmask16 &mask0,
                                  __mmask16 &mask1,
                                  __mmask16 &mask2) {
    unsigned bits0 = 0;
    unsigned bits1 = 0;
    unsigned bits2 = 0;
    for (unsigned record = 0; record < 16; ++record) {
        if ((keep_mask & (1u << record)) == 0) {
            continue;
        }
        const unsigned base = 3u * record;
        for (unsigned j = 0; j < 3; ++j) {
            const unsigned pos = base + j;
            if (pos < 16) {
                bits0 |= 1u << pos;
            } else if (pos < 32) {
                bits1 |= 1u << (pos - 16);
            } else {
                bits2 |= 1u << (pos - 32);
            }
        }
    }
    mask0 = static_cast<__mmask16>(bits0);
    mask1 = static_cast<__mmask16>(bits1);
    mask2 = static_cast<__mmask16>(bits2);
}

__attribute__((target("avx512f")))
inline void expand_record_mask_16(unsigned keep_mask,
                                  __mmask8 &mask0,
                                  __mmask8 &mask1) {
    unsigned bits0 = 0;
    unsigned bits1 = 0;
    for (unsigned record = 0; record < 8; ++record) {
        if ((keep_mask & (1u << record)) == 0) {
            continue;
        }
        const unsigned pair_bits = 0x3u << (2u * (record & 3u));
        if (record < 4) {
            bits0 |= pair_bits;
        } else {
            bits1 |= pair_bits;
        }
    }
    mask0 = static_cast<__mmask8>(bits0);
    mask1 = static_cast<__mmask8>(bits1);
}

__attribute__((target("avx512f")))
size_t expand_u64_records_avx512_chunk(const uint64_t *input,
                                       size_t count,
                                       char *output,
                                       size_t record_size) {
    size_t i = 0;
    if (record_size == 16) {
        const __m512i zeros = _mm512_setzero_si512();
        const __m512i idx_lo = _mm512_setr_epi64(0, 8, 1, 8, 2, 8, 3, 8);
        const __m512i idx_hi = _mm512_setr_epi64(4, 8, 5, 8, 6, 8, 7, 8);

        for (; i + 8 <= count; i += 8) {
            const __m512i keys =
                _mm512_loadu_si512(reinterpret_cast<const void *>(input + i));
            const __m512i out0 = _mm512_permutex2var_epi64(keys, idx_lo, zeros);
            const __m512i out1 = _mm512_permutex2var_epi64(keys, idx_hi, zeros);
            char *dst = output + i * 16;
            _mm512_storeu_si512(reinterpret_cast<void *>(dst), out0);
            _mm512_storeu_si512(reinterpret_cast<void *>(dst + 64), out1);
        }
    } else if (record_size == 12) {
        const __m512i zeros = _mm512_setzero_si512();
        const __m512i idx0 = _mm512_setr_epi32(
            0, 1, 0, 2, 3, 0, 4, 5, 0, 6, 7, 0, 8, 9, 0, 10
        );
        const __m512i idx1 = _mm512_setr_epi32(
            11, 0, 12, 13, 0, 14, 15, 0, 16, 17, 0, 18, 19, 0, 20, 21
        );
        const __m512i idx2 = _mm512_setr_epi32(
            0, 22, 23, 0, 24, 25, 0, 26, 27, 0, 28, 29, 0, 30, 31, 0
        );

        for (; i + 16 <= count; i += 16) {
            const __m512i keys0 =
                _mm512_loadu_si512(reinterpret_cast<const void *>(input + i));
            const __m512i keys1 =
                _mm512_loadu_si512(reinterpret_cast<const void *>(input + i + 8));

            __m512i out0 = _mm512_permutex2var_epi32(keys0, idx0, keys1);
            __m512i out1 = _mm512_permutex2var_epi32(keys0, idx1, keys1);
            __m512i out2 = _mm512_permutex2var_epi32(keys0, idx2, keys1);

            out0 = _mm512_mask_mov_epi32(out0, 0x4924, zeros);
            out1 = _mm512_mask_mov_epi32(out1, 0x2492, zeros);
            out2 = _mm512_mask_mov_epi32(out2, 0x9249, zeros);

            char *dst = output + i * 12;
            _mm512_storeu_si512(reinterpret_cast<void *>(dst), out0);
            _mm512_storeu_si512(reinterpret_cast<void *>(dst + 64), out1);
            _mm512_storeu_si512(reinterpret_cast<void *>(dst + 128), out2);
        }
    }

    for (; i < count; ++i) {
        std::memcpy(output + i * record_size, input + i, sizeof(uint64_t));
    }

    return count;
}

__attribute__((target("avx512f")))
size_t filter_u32_records_avx512_chunk(const char *input,
                                       size_t begin,
                                       size_t end,
                                       size_t record_size,
                                       uint32_t threshold,
                                       char *output) {
    if (record_size == 16) {
        size_t out = 0;
        const __m512i low32_mask = _mm512_set1_epi64(static_cast<long long>(0xFFFFFFFFu));
        const __m512i threshold_vec = _mm512_set1_epi64(static_cast<long long>(threshold));
        const __m512i value_idx = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);

        const char *chunk_input = input + begin * 16;
        size_t i = 0;
        const size_t count = end - begin;
        for (; i + 8 <= count; i += 8) {
            const char *block = chunk_input + i * 16;
            const __m512i src0 = _mm512_loadu_si512(reinterpret_cast<const void *>(block));
            const __m512i src1 = _mm512_loadu_si512(reinterpret_cast<const void *>(block + 64));
            __m512i values = _mm512_permutex2var_epi64(src0, value_idx, src1);
            values = _mm512_and_si512(values, low32_mask);
            const __mmask8 keep_mask =
                _mm512_cmp_epu64_mask(values, threshold_vec, _MM_CMPINT_GT);
            const size_t keep_count =
                UniqueUtils::popcount_mask(static_cast<unsigned>(keep_mask));
            if (keep_count == 0) {
                continue;
            }
            if (output != nullptr) {
                __mmask8 mask0;
                __mmask8 mask1;
                expand_record_mask_16(static_cast<unsigned>(keep_mask), mask0, mask1);
                char *dst = output + out * 16;
                _mm512_mask_compressstoreu_epi64(static_cast<void *>(dst), mask0, src0);
                dst += UniqueUtils::popcount_mask(static_cast<unsigned>(mask0)) * sizeof(uint64_t);
                _mm512_mask_compressstoreu_epi64(static_cast<void *>(dst), mask1, src1);
            }
            out += keep_count;
        }

        for (; i < count; ++i) {
            const char *record = chunk_input + i * 16;
            if (load_u32_value(record) <= threshold) {
                continue;
            }
            if (output != nullptr) {
                std::memcpy(output + out * 16, record, 16);
            }
            ++out;
        }
        return out;
    }

    if (record_size != 12) {
        return filter_records_scalar_chunk(
            input, begin, end, record_size, kValueKindU32, threshold, output
        );
    }

    size_t out = 0;
    const __m512i threshold_vec = _mm512_set1_epi32(static_cast<int>(threshold));
    const __m512i idx_lo = _mm512_setr_epi32(
        2, 5, 8, 11, 14, 17, 20, 23, 2, 5, 8, 11, 14, 17, 20, 23
    );
    const __m512i idx_hi = _mm512_setr_epi32(
        10, 13, 16, 19, 22, 25, 28, 31, 10, 13, 16, 19, 22, 25, 28, 31
    );

    const char *chunk_input = input + begin * 12;
    size_t i = 0;
    const size_t count = end - begin;
    for (; i + 16 <= count; i += 16) {
        const char *block = chunk_input + i * 12;
        const __m512i src0 = _mm512_loadu_si512(reinterpret_cast<const void *>(block));
        const __m512i src1 = _mm512_loadu_si512(reinterpret_cast<const void *>(block + 64));
        const __m512i src2 = _mm512_loadu_si512(reinterpret_cast<const void *>(block + 128));

        const __m512i values_lo = _mm512_permutex2var_epi32(src0, idx_lo, src1);
        const __m512i values_hi = _mm512_permutex2var_epi32(src1, idx_hi, src2);
        __m512i values = _mm512_castsi256_si512(_mm512_castsi512_si256(values_lo));
        values = _mm512_inserti64x4(values, _mm512_castsi512_si256(values_hi), 1);

        const __mmask16 keep_mask =
            _mm512_cmp_epu32_mask(values, threshold_vec, _MM_CMPINT_GT);
        const size_t keep_count =
            UniqueUtils::popcount_mask(static_cast<unsigned>(keep_mask));
        if (keep_count == 0) {
            continue;
        }
        if (output != nullptr) {
            __mmask16 mask0;
            __mmask16 mask1;
            __mmask16 mask2;
            expand_record_mask_12(static_cast<unsigned>(keep_mask), mask0, mask1, mask2);
            char *dst = output + out * 12;
            _mm512_mask_compressstoreu_epi32(static_cast<void *>(dst), mask0, src0);
            dst += UniqueUtils::popcount_mask(static_cast<unsigned>(mask0)) * sizeof(uint32_t);
            _mm512_mask_compressstoreu_epi32(static_cast<void *>(dst), mask1, src1);
            dst += UniqueUtils::popcount_mask(static_cast<unsigned>(mask1)) * sizeof(uint32_t);
            _mm512_mask_compressstoreu_epi32(static_cast<void *>(dst), mask2, src2);
        }
        out += keep_count;
    }

    for (; i < count; ++i) {
        const char *record = chunk_input + i * 12;
        if (load_u32_value(record) <= threshold) {
            continue;
        }
        if (output != nullptr) {
            std::memcpy(output + out * 12, record, 12);
        }
        ++out;
    }

    return out;
}

__attribute__((target("avx512f")))
size_t filter_u64_records_avx512_chunk(const char *input,
                                       size_t begin,
                                       size_t end,
                                       size_t record_size,
                                       uint64_t threshold,
                                       char *output) {
    if (record_size != 16) {
        return filter_records_scalar_chunk(
            input, begin, end, record_size, kValueKindU64, threshold, output
        );
    }

    size_t out = 0;
    const __m512i threshold_vec = _mm512_set1_epi64(static_cast<long long>(threshold));

    const __m512i value_idx = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    const char *chunk_input = input + begin * 16;
    size_t i = 0;
    const size_t count = end - begin;
    for (; i + 8 <= count; i += 8) {
        const char *block = chunk_input + i * 16;
        const __m512i src0 = _mm512_loadu_si512(reinterpret_cast<const void *>(block));
        const __m512i src1 = _mm512_loadu_si512(reinterpret_cast<const void *>(block + 64));
        const __m512i values = _mm512_permutex2var_epi64(src0, value_idx, src1);
        const __mmask8 keep_mask =
            _mm512_cmp_epu64_mask(values, threshold_vec, _MM_CMPINT_GT);
        const size_t keep_count =
            UniqueUtils::popcount_mask(static_cast<unsigned>(keep_mask));
        if (keep_count == 0) {
            continue;
        }
        if (output != nullptr) {
            __mmask8 mask0;
            __mmask8 mask1;
            expand_record_mask_16(static_cast<unsigned>(keep_mask), mask0, mask1);
            char *dst = output + out * 16;
            _mm512_mask_compressstoreu_epi64(static_cast<void *>(dst), mask0, src0);
            dst += UniqueUtils::popcount_mask(static_cast<unsigned>(mask0)) * sizeof(uint64_t);
            _mm512_mask_compressstoreu_epi64(static_cast<void *>(dst), mask1, src1);
        }
        out += keep_count;
    }

    for (; i < count; ++i) {
        const char *record = chunk_input + i * 16;
        if (load_u64_value(record) <= threshold) {
            continue;
        }
        if (output != nullptr) {
            std::memcpy(output + out * 16, record, 16);
        }
        ++out;
    }

    return out;
}

__attribute__((target("avx512f")))
size_t filter_u32_records_avx512_inplace(char *data,
                                         size_t count,
                                         size_t record_size,
                                         uint32_t threshold) {
    if (record_size == 16) {
        size_t out = 0;
        size_t i = 0;
        const __m512i low32_mask = _mm512_set1_epi64(static_cast<long long>(0xFFFFFFFFu));
        const __m512i threshold_vec = _mm512_set1_epi64(static_cast<long long>(threshold));
        const __m512i value_idx = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);

        for (; i + 8 <= count; i += 8) {
            const char *block = data + i * 16;
            const __m512i src0 = _mm512_loadu_si512(reinterpret_cast<const void *>(block));
            const __m512i src1 = _mm512_loadu_si512(reinterpret_cast<const void *>(block + 64));
            __m512i values = _mm512_permutex2var_epi64(src0, value_idx, src1);
            values = _mm512_and_si512(values, low32_mask);
            const __mmask8 keep_mask =
                _mm512_cmp_epu64_mask(values, threshold_vec, _MM_CMPINT_GT);
            const size_t keep_count =
                UniqueUtils::popcount_mask(static_cast<unsigned>(keep_mask));
            if (keep_count == 0) {
                continue;
            }

            __mmask8 mask0;
            __mmask8 mask1;
            expand_record_mask_16(static_cast<unsigned>(keep_mask), mask0, mask1);
            char *dst = data + out * 16;
            _mm512_mask_compressstoreu_epi64(static_cast<void *>(dst), mask0, src0);
            dst += UniqueUtils::popcount_mask(static_cast<unsigned>(mask0)) * sizeof(uint64_t);
            _mm512_mask_compressstoreu_epi64(static_cast<void *>(dst), mask1, src1);
            out += keep_count;
        }

        for (; i < count; ++i) {
            char *record = data + i * 16;
            if (load_u32_value(record) <= threshold) {
                continue;
            }
            copy_record(data + out * 16, record, 16);
            ++out;
        }

        return out;
    }

    if (record_size != 12) {
        return filter_records_scalar_inplace(
            data, count, record_size, kValueKindU32, threshold
        );
    }

    size_t out = 0;
    size_t i = 0;
    const __m512i threshold_vec = _mm512_set1_epi32(static_cast<int>(threshold));
    const __m512i idx_lo = _mm512_setr_epi32(
        2, 5, 8, 11, 14, 17, 20, 23, 2, 5, 8, 11, 14, 17, 20, 23
    );
    const __m512i idx_hi = _mm512_setr_epi32(
        10, 13, 16, 19, 22, 25, 28, 31, 10, 13, 16, 19, 22, 25, 28, 31
    );

    for (; i + 16 <= count; i += 16) {
        const char *block = data + i * 12;
        const __m512i src0 = _mm512_loadu_si512(reinterpret_cast<const void *>(block));
        const __m512i src1 = _mm512_loadu_si512(reinterpret_cast<const void *>(block + 64));
        const __m512i src2 = _mm512_loadu_si512(reinterpret_cast<const void *>(block + 128));

        const __m512i values_lo = _mm512_permutex2var_epi32(src0, idx_lo, src1);
        const __m512i values_hi = _mm512_permutex2var_epi32(src1, idx_hi, src2);
        __m512i values = _mm512_castsi256_si512(_mm512_castsi512_si256(values_lo));
        values = _mm512_inserti64x4(values, _mm512_castsi512_si256(values_hi), 1);

        const __mmask16 keep_mask =
            _mm512_cmp_epu32_mask(values, threshold_vec, _MM_CMPINT_GT);
        const size_t keep_count =
            UniqueUtils::popcount_mask(static_cast<unsigned>(keep_mask));
        if (keep_count == 0) {
            continue;
        }

        __mmask16 mask0;
        __mmask16 mask1;
        __mmask16 mask2;
        expand_record_mask_12(static_cast<unsigned>(keep_mask), mask0, mask1, mask2);
        char *dst = data + out * 12;
        _mm512_mask_compressstoreu_epi32(static_cast<void *>(dst), mask0, src0);
        dst += UniqueUtils::popcount_mask(static_cast<unsigned>(mask0)) * sizeof(uint32_t);
        _mm512_mask_compressstoreu_epi32(static_cast<void *>(dst), mask1, src1);
        dst += UniqueUtils::popcount_mask(static_cast<unsigned>(mask1)) * sizeof(uint32_t);
        _mm512_mask_compressstoreu_epi32(static_cast<void *>(dst), mask2, src2);
        out += keep_count;
    }

    for (; i < count; ++i) {
        char *record = data + i * 12;
        if (load_u32_value(record) <= threshold) {
            continue;
        }
        copy_record(data + out * 12, record, 12);
        ++out;
    }

    return out;
}

__attribute__((target("avx512f")))
size_t filter_u64_records_avx512_inplace(char *data,
                                         size_t count,
                                         size_t record_size,
                                         uint64_t threshold) {
    if (record_size != 16) {
        return filter_records_scalar_inplace(
            data, count, record_size, kValueKindU64, threshold
        );
    }

    size_t out = 0;
    size_t i = 0;
    const __m512i threshold_vec = _mm512_set1_epi64(static_cast<long long>(threshold));
    const __m512i value_idx = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);

    for (; i + 8 <= count; i += 8) {
        const char *block = data + i * 16;
        const __m512i src0 = _mm512_loadu_si512(reinterpret_cast<const void *>(block));
        const __m512i src1 = _mm512_loadu_si512(reinterpret_cast<const void *>(block + 64));
        const __m512i values = _mm512_permutex2var_epi64(src0, value_idx, src1);
        const __mmask8 keep_mask =
            _mm512_cmp_epu64_mask(values, threshold_vec, _MM_CMPINT_GT);
        const size_t keep_count =
            UniqueUtils::popcount_mask(static_cast<unsigned>(keep_mask));
        if (keep_count == 0) {
            continue;
        }

        __mmask8 mask0;
        __mmask8 mask1;
        expand_record_mask_16(static_cast<unsigned>(keep_mask), mask0, mask1);
        char *dst = data + out * 16;
        _mm512_mask_compressstoreu_epi64(static_cast<void *>(dst), mask0, src0);
        dst += UniqueUtils::popcount_mask(static_cast<unsigned>(mask0)) * sizeof(uint64_t);
        _mm512_mask_compressstoreu_epi64(static_cast<void *>(dst), mask1, src1);
        out += keep_count;
    }

    for (; i < count; ++i) {
        char *record = data + i * 16;
        if (load_u64_value(record) <= threshold) {
            continue;
        }
        copy_record(data + out * 16, record, 16);
        ++out;
    }

    return out;
}

__attribute__((target("avx512f")))
void fill_u32_range_avx512(uint32_t *output,
                           size_t begin,
                           size_t end,
                           uint32_t value) {
    if (begin >= end) {
        return;
    }

    const __m512i fill_vec = _mm512_set1_epi32(static_cast<int>(value));
    size_t i = begin;
    for (; i + 16 <= end; i += 16) {
        _mm512_storeu_si512(reinterpret_cast<void *>(output + i), fill_vec);
    }
    for (; i < end; ++i) {
        output[i] = value;
    }
}

#endif

void expand_u64_records_scalar(const uint64_t *input,
                               size_t count,
                               char *output,
                               size_t record_size) {
    for (size_t i = 0; i < count; ++i) {
        std::memcpy(output + i * record_size, input + i, sizeof(uint64_t));
    }
}

size_t filter_records_native_impl(const void *input_void,
                                  size_t count,
                                  size_t record_size,
                                  int value_kind,
                                  uint64_t threshold_bits,
                                  void *output_void) {
    if (input_void == nullptr || count == 0) {
        return 0;
    }

    const char *input = static_cast<const char *>(input_void);
    char *output = static_cast<char *>(output_void);

    const int max_threads =
#if defined(_OPENMP)
        omp_get_max_threads();
#else
        1;
#endif
    const size_t worker_count = (count >= (1u << 20)) ? static_cast<size_t>(std::max(1, max_threads)) : 1u;

    if (worker_count <= 1) {
#if defined(__GNUC__) || defined(__clang__)
        if (UniqueUtils::cpu_has_avx512_dq_vl()) {
            if (value_kind == kValueKindU32) {
                return filter_u32_records_avx512_chunk(
                    input, 0, count, record_size, static_cast<uint32_t>(threshold_bits), output
                );
            }
            if (value_kind == kValueKindU64) {
                return filter_u64_records_avx512_chunk(
                    input, 0, count, record_size, threshold_bits, output
                );
            }
        }
#endif
        return filter_records_scalar_chunk(
            input, 0, count, record_size, value_kind, threshold_bits, output
        );
    }

    const size_t chunk = (count + worker_count - 1) / worker_count;
    std::vector<size_t> counts(worker_count, 0);
    std::vector<size_t> offsets(worker_count, 0);

#if defined(_OPENMP)
#pragma omp parallel for num_threads(static_cast<int>(worker_count))
#endif
    for (int t = 0; t < static_cast<int>(worker_count); ++t) {
        const size_t begin = static_cast<size_t>(t) * chunk;
        const size_t end = std::min(count, begin + chunk);
        if (begin >= end) {
            counts[static_cast<size_t>(t)] = 0;
            continue;
        }
#if defined(__GNUC__) || defined(__clang__)
        if (UniqueUtils::cpu_has_avx512_dq_vl()) {
            if (value_kind == kValueKindU32) {
                counts[static_cast<size_t>(t)] = filter_u32_records_avx512_chunk(
                    input, begin, end, record_size, static_cast<uint32_t>(threshold_bits), nullptr
                );
                continue;
            }
            if (value_kind == kValueKindU64) {
                counts[static_cast<size_t>(t)] = filter_u64_records_avx512_chunk(
                    input, begin, end, record_size, threshold_bits, nullptr
                );
                continue;
            }
        }
#endif
        counts[static_cast<size_t>(t)] = filter_records_scalar_chunk(
            input, begin, end, record_size, value_kind, threshold_bits, nullptr
        );
    }

    size_t total = 0;
    for (size_t t = 0; t < worker_count; ++t) {
        offsets[t] = total;
        total += counts[t];
    }

    if (output == nullptr) {
        return total;
    }

#if defined(_OPENMP)
#pragma omp parallel for num_threads(static_cast<int>(worker_count))
#endif
    for (int t = 0; t < static_cast<int>(worker_count); ++t) {
        const size_t begin = static_cast<size_t>(t) * chunk;
        const size_t end = std::min(count, begin + chunk);
        if (begin >= end) {
            continue;
        }
        char *chunk_output = output + offsets[static_cast<size_t>(t)] * record_size;
#if defined(__GNUC__) || defined(__clang__)
        if (UniqueUtils::cpu_has_avx512_dq_vl()) {
            if (value_kind == kValueKindU32) {
                filter_u32_records_avx512_chunk(
                    input, begin, end, record_size, static_cast<uint32_t>(threshold_bits), chunk_output
                );
                continue;
            }
            if (value_kind == kValueKindU64) {
                filter_u64_records_avx512_chunk(
                    input, begin, end, record_size, threshold_bits, chunk_output
                );
                continue;
            }
        }
#endif
        filter_records_scalar_chunk(
            input, begin, end, record_size, value_kind, threshold_bits, chunk_output
        );
    }

    return total;
}

size_t filter_records_inplace_impl(void *data_void,
                                   size_t count,
                                   size_t record_size,
                                   int value_kind,
                                   uint64_t threshold_bits) {
    if (data_void == nullptr || count == 0) {
        return 0;
    }

    char *data = static_cast<char *>(data_void);

#if defined(__GNUC__) || defined(__clang__)
    if (UniqueUtils::cpu_has_avx512_dq_vl()) {
        switch (value_kind) {
        case kValueKindU32:
            return filter_u32_records_avx512_inplace(
                data, count, record_size, static_cast<uint32_t>(threshold_bits)
            );
        case kValueKindU64:
            return filter_u64_records_avx512_inplace(
                data, count, record_size, threshold_bits
            );
        default:
            break;
        }
    }
#endif

    return filter_records_scalar_inplace(
        data, count, record_size, value_kind, threshold_bits
    );
}

void fill_u32_range_scalar(uint32_t *output,
                           size_t begin,
                           size_t end,
                           uint32_t value) {
    if (begin >= end) {
        return;
    }
    std::fill(output + begin, output + end, value);
}

void fill_u32_range(uint32_t *output,
                    size_t begin,
                    size_t end,
                    uint32_t value) {
#if defined(__GNUC__) || defined(__clang__)
    if (UniqueUtils::cpu_has_avx512_dq_vl() && end - begin >= 256) {
        fill_u32_range_avx512(output, begin, end, value);
        return;
    }
#endif
    fill_u32_range_scalar(output, begin, end, value);
}

void fill_u32_range_parallel(uint32_t *output,
                             size_t begin,
                             size_t end,
                             uint32_t value) {
    if (begin >= end) {
        return;
    }

    const size_t length = end - begin;
    const int max_threads =
#if defined(_OPENMP)
        omp_get_max_threads();
#else
        1;
#endif
    const size_t worker_count =
        (length >= (1u << 20)) ? static_cast<size_t>(std::max(1, max_threads)) : 1u;

    if (worker_count <= 1) {
        fill_u32_range(output, begin, end, value);
        return;
    }

    const size_t chunk = (length + worker_count - 1) / worker_count;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(static_cast<int>(worker_count))
#endif
    for (int t = 0; t < static_cast<int>(worker_count); ++t) {
        const size_t local_begin = begin + static_cast<size_t>(t) * chunk;
        const size_t local_end = std::min(end, local_begin + chunk);
        if (local_begin >= local_end) {
            continue;
        }
        fill_u32_range(output, local_begin, local_end, value);
    }
}

#if defined(__GNUC__) || defined(__clang__)

__attribute__((target("avx512f")))
inline void fill_transition_segment(uint32_t *output,
                                    uint32_t previous_header,
                                    uint32_t current_header,
                                    uint32_t position) {
    const size_t begin = static_cast<size_t>(previous_header) + 1;
    const size_t end = static_cast<size_t>(current_header) + 1;
    if (begin >= end) {
        return;
    }
    if (end - begin == 1) {
        output[begin] = position;
        return;
    }
    fill_u32_range(output, begin, end, position);
}

__attribute__((target("avx512f")))
void extract_headers_record16_block_avx512(const char *input,
                                           size_t count,
                                           uint32_t *headers) {
    size_t i = 0;
    const __m512i key_select = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i dword_select = _mm512_setr_epi32(
        0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14
    );

    if (count >= 16) {
        const __m512i lo0 = _mm512_loadu_si512(reinterpret_cast<const void *>(input));
        const __m512i lo1 = _mm512_loadu_si512(reinterpret_cast<const void *>(input + 64));
        const __m512i hi0 = _mm512_loadu_si512(reinterpret_cast<const void *>(input + 128));
        const __m512i hi1 = _mm512_loadu_si512(reinterpret_cast<const void *>(input + 192));

        const __m512i keys_lo = _mm512_permutex2var_epi64(lo0, key_select, lo1);
        const __m512i keys_hi = _mm512_permutex2var_epi64(hi0, key_select, hi1);
        const __m512i headers_lo =
            _mm512_permutexvar_epi32(dword_select, _mm512_srli_epi64(keys_lo, 40));
        const __m512i headers_hi =
            _mm512_permutexvar_epi32(dword_select, _mm512_srli_epi64(keys_hi, 40));

        _mm256_storeu_si256(
            reinterpret_cast<__m256i *>(headers),
            _mm512_castsi512_si256(headers_lo)
        );
        _mm256_storeu_si256(
            reinterpret_cast<__m256i *>(headers + 8),
            _mm512_castsi512_si256(headers_hi)
        );
        i = 16;
    }

    for (; i + 8 <= count; i += 8) {
        const char *block = input + i * 16;
        const __m512i lo = _mm512_loadu_si512(reinterpret_cast<const void *>(block));
        const __m512i hi = _mm512_loadu_si512(reinterpret_cast<const void *>(block + 64));
        const __m512i keys = _mm512_permutex2var_epi64(lo, key_select, hi);
        const __m512i header_dwords =
            _mm512_permutexvar_epi32(dword_select, _mm512_srli_epi64(keys, 40));
        _mm256_storeu_si256(
            reinterpret_cast<__m256i *>(headers + i),
            _mm512_castsi512_si256(header_dwords)
        );
    }

    for (; i < count; ++i) {
        headers[i] = static_cast<uint32_t>(load_key(input + i * 16) >> 40);
    }
}

__attribute__((target("avx512f")))
void extract_headers_stride_block_avx512(const char *input,
                                         size_t count,
                                         size_t record_size,
                                         uint32_t *headers) {
    if (record_size == 12 && count >= 16) {
        const __m512i src0 = _mm512_loadu_si512(reinterpret_cast<const void *>(input));
        const __m512i src1 = _mm512_loadu_si512(reinterpret_cast<const void *>(input + 64));
        const __m512i src2 = _mm512_loadu_si512(reinterpret_cast<const void *>(input + 128));
        const __m512i idx_lo = _mm512_setr_epi32(
            1, 4, 7, 10, 13, 16, 19, 22, 1, 4, 7, 10, 13, 16, 19, 22
        );
        const __m512i idx_hi = _mm512_setr_epi32(
            9, 12, 15, 18, 21, 24, 27, 30, 9, 12, 15, 18, 21, 24, 27, 30
        );
        const __m512i headers_lo =
            _mm512_srli_epi32(_mm512_permutex2var_epi32(src0, idx_lo, src1), 8);
        const __m512i headers_hi =
            _mm512_srli_epi32(_mm512_permutex2var_epi32(src1, idx_hi, src2), 8);

        _mm256_storeu_si256(
            reinterpret_cast<__m256i *>(headers),
            _mm512_castsi512_si256(headers_lo)
        );
        _mm256_storeu_si256(
            reinterpret_cast<__m256i *>(headers + 8),
            _mm512_castsi512_si256(headers_hi)
        );
        for (size_t i = 16; i < count; ++i) {
            headers[i] = static_cast<uint32_t>(load_key(input + i * record_size) >> 40);
        }
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        headers[i] = static_cast<uint32_t>(load_key(input + i * record_size) >> 40);
    }
}

__attribute__((target("avx512f")))
void extract_headers_block_avx512(const char *input,
                                  size_t count,
                                  size_t record_size,
                                  uint32_t *headers) {
    if (record_size == 16) {
        extract_headers_record16_block_avx512(input, count, headers);
        return;
    }
    extract_headers_stride_block_avx512(input, count, record_size, headers);
}

__attribute__((target("avx512f")))
void emit_transition_block_avx512(const uint32_t *headers,
                                  size_t count,
                                  size_t base_position,
                                  uint32_t &prev_header,
                                  uint32_t &filled_header,
                                  uint32_t *output) {
    if (count == 0) {
        return;
    }

    if (count < 16) {
        for (size_t lane = 0; lane < count; ++lane) {
            const uint32_t header = headers[lane];
            if (header == prev_header) {
                continue;
            }
            fill_transition_segment(
                output, filled_header, header, static_cast<uint32_t>(base_position + lane)
            );
            prev_header = header;
            filled_header = header;
        }
        return;
    }

    const __m512i shift1_idx =
        _mm512_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
    const __m512i lane_idx =
        _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    alignas(64) uint32_t tmp_headers[16];
    alignas(64) uint32_t tmp_positions[16];

    const __m512i curr =
        _mm512_loadu_si512(reinterpret_cast<const void *>(headers));
    __m512i prev = _mm512_permutexvar_epi32(shift1_idx, curr);
    prev = _mm512_mask_mov_epi32(prev, 0x0001, _mm512_set1_epi32(static_cast<int>(prev_header)));
    const __mmask16 changed = _mm512_cmp_epu32_mask(curr, prev, _MM_CMPINT_NE);
    if (changed == 0) {
        prev_header = headers[15];
        return;
    }

    const __m512i positions =
        _mm512_add_epi32(_mm512_set1_epi32(static_cast<int>(base_position)), lane_idx);
    _mm512_mask_compressstoreu_epi32(static_cast<void *>(tmp_headers), changed, curr);
    _mm512_mask_compressstoreu_epi32(static_cast<void *>(tmp_positions), changed, positions);

    const size_t keep_count =
        UniqueUtils::popcount_mask(static_cast<unsigned>(changed));
    for (size_t k = 0; k < keep_count; ++k) {
        const uint32_t header = tmp_headers[k];
        fill_transition_segment(output, filled_header, header, tmp_positions[k]);
        filled_header = header;
    }
    prev_header = headers[15];
}

#endif

void create_index_streaming(const char *input,
                            size_t count,
                            size_t record_size,
                            uint32_t *output) {
    const uint32_t first_header = static_cast<uint32_t>(load_key(input) >> 40);
    const uint32_t last_header =
        static_cast<uint32_t>(load_key(input + (count - 1) * record_size) >> 40);
    fill_u32_range(output, 0, static_cast<size_t>(first_header) + 1, 0u);

    if (first_header == last_header) {
        fill_u32_range_parallel(
            output,
            static_cast<size_t>(first_header) + 1,
            kBookSolverIndexLength,
            static_cast<uint32_t>(count)
        );
        return;
    }

    const int max_threads =
#if defined(_OPENMP)
        omp_get_max_threads();
#else
        1;
#endif
    const size_t worker_count =
        (count >= (1u << 20)) ? static_cast<size_t>(std::max(1, max_threads)) : 1u;

    auto process_chunk_scalar = [&](size_t begin, size_t end, uint32_t chunk_prev_header) {
        if (begin >= end) {
            return;
        }
        uint32_t prev_header = chunk_prev_header;
        uint32_t filled_header = chunk_prev_header;
        size_t i = begin;
        if (begin == 0) {
            i = 1;
        }
        for (; i < end; ++i) {
            const uint32_t header =
                static_cast<uint32_t>(load_key(input + i * record_size) >> 40);
            if (header == prev_header) {
                continue;
            }
            fill_transition_segment(
                output, filled_header, header, static_cast<uint32_t>(i)
            );
            prev_header = header;
            filled_header = header;
        }
    };

    if (worker_count <= 1) {
#if defined(__GNUC__) || defined(__clang__)
        if (UniqueUtils::cpu_has_avx512_dq_vl()) {
            uint32_t prev_header = first_header;
            uint32_t filled_header = first_header;
            size_t i = 1;
            alignas(64) uint32_t block_headers[16];
            for (; i + 16 <= count; i += 16) {
                extract_headers_block_avx512(
                    input + i * record_size, 16, record_size, block_headers
                );
                emit_transition_block_avx512(
                    block_headers, 16, i, prev_header, filled_header, output
                );
            }
            if (i < count) {
                extract_headers_block_avx512(
                    input + i * record_size, count - i, record_size, block_headers
                );
                emit_transition_block_avx512(
                    block_headers, count - i, i, prev_header, filled_header, output
                );
            }
        } else
#endif
        {
            process_chunk_scalar(0, count, first_header);
        }
    } else {
        const size_t chunk = (count + worker_count - 1) / worker_count;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(static_cast<int>(worker_count))
#endif
        for (int t = 0; t < static_cast<int>(worker_count); ++t) {
            const size_t begin = static_cast<size_t>(t) * chunk;
            const size_t end = std::min(count, begin + chunk);
            if (begin >= end) {
                continue;
            }
            const uint32_t chunk_prev_header =
                (begin == 0)
                    ? first_header
                    : static_cast<uint32_t>(load_key(input + (begin - 1) * record_size) >> 40);

#if defined(__GNUC__) || defined(__clang__)
            if (UniqueUtils::cpu_has_avx512_dq_vl()) {
                uint32_t prev_header = chunk_prev_header;
                uint32_t filled_header = chunk_prev_header;
                size_t i = (begin == 0) ? 1 : begin;
                alignas(64) uint32_t block_headers[16];
                for (; i + 16 <= end; i += 16) {
                    extract_headers_block_avx512(
                        input + i * record_size, 16, record_size, block_headers
                    );
                    emit_transition_block_avx512(
                        block_headers, 16, i, prev_header, filled_header, output
                    );
                }
                if (i < end) {
                    extract_headers_block_avx512(
                        input + i * record_size, end - i, record_size, block_headers
                    );
                    emit_transition_block_avx512(
                        block_headers, end - i, i, prev_header, filled_header, output
                    );
                }
                continue;
            }
#endif
            process_chunk_scalar(begin, end, chunk_prev_header);
        }
    }

    fill_u32_range_parallel(
        output,
        static_cast<size_t>(last_header) + 1,
        kBookSolverIndexLength,
        static_cast<uint32_t>(count)
    );
}

void collect_header_transitions_scalar(const char *input,
                                       size_t count,
                                       size_t record_size,
                                       std::vector<HeaderTransition> &transitions,
                                       uint32_t first_header) {
    transitions.clear();
    transitions.reserve(std::min<size_t>(count / 64 + 1, 1u << 20));

    uint32_t prev_header = first_header;
    for (size_t i = 1; i < count; ++i) {
        const uint32_t header =
            static_cast<uint32_t>(load_key(input + i * record_size) >> 40);
        if (header == prev_header) {
            continue;
        }
        transitions.push_back(
            HeaderTransition{header, static_cast<uint32_t>(i)}
        );
        prev_header = header;
    }
}

void collect_header_transitions_parallel(const char *input,
                                         size_t count,
                                         size_t record_size,
                                         std::vector<HeaderTransition> &transitions,
                                         uint32_t first_header) {
    const int max_threads =
#if defined(_OPENMP)
        omp_get_max_threads();
#else
        1;
#endif
    const size_t worker_count =
        (count >= (1u << 20)) ? static_cast<size_t>(std::max(1, max_threads)) : 1u;

    if (worker_count <= 1) {
        collect_header_transitions_scalar(input, count, record_size, transitions, first_header);
        return;
    }

    const size_t chunk = (count + worker_count - 1) / worker_count;
    std::vector<std::vector<HeaderTransition>> local_transitions(worker_count);

#if defined(_OPENMP)
#pragma omp parallel for num_threads(static_cast<int>(worker_count))
#endif
    for (int t = 0; t < static_cast<int>(worker_count); ++t) {
        const size_t begin = static_cast<size_t>(t) * chunk;
        const size_t end = std::min(count, begin + chunk);
        if (begin >= end) {
            continue;
        }

        auto &local = local_transitions[static_cast<size_t>(t)];
        local.reserve(std::min<size_t>((end - begin) / 64 + 1, 1u << 18));

        uint32_t prev_header = first_header;
        size_t i = 1;
        if (begin > 0) {
            prev_header = static_cast<uint32_t>(
                load_key(input + (begin - 1) * record_size) >> 40
            );
            i = begin;
        }

        for (; i < end; ++i) {
            const uint32_t header =
                static_cast<uint32_t>(load_key(input + i * record_size) >> 40);
            if (header == prev_header) {
                continue;
            }
            local.push_back(
                HeaderTransition{header, static_cast<uint32_t>(i)}
            );
            prev_header = header;
        }
    }

    size_t total_transitions = 0;
    for (const auto &local : local_transitions) {
        total_transitions += local.size();
    }
    transitions.clear();
    transitions.reserve(total_transitions);

    uint32_t prev_header = first_header;
    for (const auto &local : local_transitions) {
        for (const HeaderTransition &transition : local) {
            if (transition.header == prev_header) {
                continue;
            }
            transitions.push_back(transition);
            prev_header = transition.header;
        }
    }
}

void create_index_two_stage(const char *input,
                            size_t count,
                            size_t record_size,
                            uint32_t *output) {
    const uint32_t first_header = static_cast<uint32_t>(load_key(input) >> 40);
    const uint32_t last_header =
        static_cast<uint32_t>(load_key(input + (count - 1) * record_size) >> 40);

    fill_u32_range(output, 0, static_cast<size_t>(first_header) + 1, 0u);

    if (first_header == last_header) {
        fill_u32_range_parallel(
            output,
            static_cast<size_t>(first_header) + 1,
            kBookSolverIndexLength,
            static_cast<uint32_t>(count)
        );
        return;
    }

    std::vector<HeaderTransition> transitions;
    collect_header_transitions_parallel(input, count, record_size, transitions, first_header);

    uint32_t prev_header = first_header;
    for (const HeaderTransition &transition : transitions) {
        fill_u32_range(
            output,
            static_cast<size_t>(prev_header) + 1,
            static_cast<size_t>(transition.header) + 1,
            transition.position
        );
        prev_header = transition.header;
    }

    fill_u32_range_parallel(
        output,
        static_cast<size_t>(last_header) + 1,
        kBookSolverIndexLength,
        static_cast<uint32_t>(count)
    );
}

} // namespace

BOOKSOLVER_EXPORT size_t booksolver_expand_u64_records(const uint64_t *input,
                                                       size_t count,
                                                       void *output,
                                                       size_t record_size) {
    if (input == nullptr || output == nullptr || record_size < sizeof(uint64_t)) {
        return 0;
    }

    const int max_threads =
#if defined(_OPENMP)
        omp_get_max_threads();
#else
        1;
#endif
    const size_t worker_count = (count >= (1u << 20)) ? static_cast<size_t>(std::max(1, max_threads)) : 1u;
    char *output_bytes = static_cast<char *>(output);

    if (worker_count <= 1) {
#if defined(__GNUC__) || defined(__clang__)
        if (UniqueUtils::cpu_has_avx512_dq_vl()) {
            return expand_u64_records_avx512_chunk(input, count, output_bytes, record_size);
        }
#endif
        expand_u64_records_scalar(input, count, output_bytes, record_size);
        return count;
    }

    const size_t chunk = (count + worker_count - 1) / worker_count;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(static_cast<int>(worker_count))
#endif
    for (int t = 0; t < static_cast<int>(worker_count); ++t) {
        const size_t begin = static_cast<size_t>(t) * chunk;
        const size_t end = std::min(count, begin + chunk);
        if (begin >= end) {
            continue;
        }
#if defined(__GNUC__) || defined(__clang__)
        if (UniqueUtils::cpu_has_avx512_dq_vl()) {
            expand_u64_records_avx512_chunk(
                input + begin,
                end - begin,
                output_bytes + begin * record_size,
                record_size
            );
            continue;
        }
#endif
        expand_u64_records_scalar(
            input + begin,
            end - begin,
            output_bytes + begin * record_size,
            record_size
        );
    }
    return count;
}

BOOKSOLVER_EXPORT size_t booksolver_filter_records(const void *input,
                                                   size_t count,
                                                   size_t record_size,
                                                   int value_kind,
                                                   uint64_t threshold_bits,
                                                   void *output) {
    return filter_records_native_impl(
        input, count, record_size, value_kind, threshold_bits, output
    );
}

BOOKSOLVER_EXPORT size_t booksolver_filter_records_inplace(void *data,
                                                           size_t count,
                                                           size_t record_size,
                                                           int value_kind,
                                                           uint64_t threshold_bits) {
    return filter_records_inplace_impl(
        data, count, record_size, value_kind, threshold_bits
    );
}

BOOKSOLVER_EXPORT int booksolver_create_index(const void *input,
                                              size_t count,
                                              size_t record_size,
                                              uint32_t *output) {
    if (input == nullptr || output == nullptr || count == 0 || record_size < sizeof(uint64_t)) {
        return 0;
    }

    create_index_streaming(static_cast<const char *>(input), count, record_size, output);
    return 1;
}
