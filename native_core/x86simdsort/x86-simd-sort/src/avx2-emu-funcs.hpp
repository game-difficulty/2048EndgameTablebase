#ifndef AVX2_EMU_FUNCS
#define AVX2_EMU_FUNCS

#include <array>
#include <utility>

constexpr auto avx2_mask_helper_lut64 = [] {
    std::array<std::array<int64_t, 4>, 16> lut {};
    for (int64_t i = 0; i <= 0xF; i++) {
        std::array<int64_t, 4> entry {};
        for (int j = 0; j < 4; j++) {
            if (((i >> j) & 1) == 1)
                entry[j] = 0xFFFFFFFFFFFFFFFF;
            else
                entry[j] = 0;
        }
        lut[i] = entry;
    }
    return lut;
}();

constexpr auto avx2_compressstore_lut64_gen = [] {
    std::array<std::array<int32_t, 8>, 16> permLut {};
    std::array<std::array<int64_t, 4>, 16> leftLut {};
    for (int64_t i = 0; i <= 0xF; i++) {
        std::array<int32_t, 8> indices {};
        std::array<int64_t, 4> leftEntry = {0, 0, 0, 0};
        int right = 7;
        int left = 0;
        for (int j = 0; j < 4; j++) {
            bool ge = (i >> j) & 1;
            if (ge) {
                indices[right] = 2 * j + 1;
                indices[right - 1] = 2 * j;
                right -= 2;
            }
            else {
                indices[left + 1] = 2 * j + 1;
                indices[left] = 2 * j;
                leftEntry[left / 2] = 0xFFFFFFFFFFFFFFFF;
                left += 2;
            }
        }
        permLut[i] = indices;
        leftLut[i] = leftEntry;
    }
    return std::make_pair(permLut, leftLut);
}();
constexpr auto avx2_compressstore_lut64_perm
        = avx2_compressstore_lut64_gen.first;
constexpr auto avx2_compressstore_lut64_left
        = avx2_compressstore_lut64_gen.second;

X86_SIMD_SORT_INLINE
__m256i convert_int_to_avx2_mask_64bit(int32_t m)
{
    return _mm256_loadu_si256(
            (const __m256i *)avx2_mask_helper_lut64[m].data());
}

X86_SIMD_SORT_INLINE
int32_t convert_avx2_mask_to_int_64bit(__m256i m)
{
    return _mm256_movemask_pd(_mm256_castsi256_pd(m));
}

#endif
