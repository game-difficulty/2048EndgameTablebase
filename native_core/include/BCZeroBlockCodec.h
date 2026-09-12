#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace BC {

// Internal, lossless temporary encoding. Logical boundaries never depend on
// data content. Stored size includes only bitmap + payload, not disk padding.
inline constexpr uint32_t kBCZeroBlockBytes = 1024U * 1024U;
inline constexpr uint32_t kBCZeroBitmapFlag = 0x80000000U;

template<class T> using BCZeroBits = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
template<class T> inline BCZeroBits<T> bc_zero_bits(T value) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8);
    BCZeroBits<T> bits;
    std::memcpy(&bits, &value, sizeof(T));
    return bits;
}

// output has exactly kBCZeroBlockBytes capacity. RAW returns the original span
// to the caller: no memcpy for incompressible input.
template<class T>
uint32_t bc_zero_encode_block(const T* input, uint32_t count, T zero, uint8_t* output,
                              uint32_t alignment) {
    const uint32_t raw = count * sizeof(T);
    if (count > kBCZeroBlockBytes / sizeof(T) || alignment == 0 ||
        (alignment & (alignment - 1U)) != 0 || (count && (!input || !output))) {
        throw std::invalid_argument("BC zero block input invalid");
    }
    const auto zero_bits = bc_zero_bits(zero);
    uint32_t nonzero = 0;
    for (uint32_t i = 0; i < count; ++i) nonzero += bc_zero_bits(input[i]) != zero_bits;
    if (nonzero == 0) return kBCZeroBitmapFlag;
    const uint32_t masks = (count + 63U) / 64U;
    const uint32_t packed = masks * 8U + nonzero * sizeof(T);
    const auto aligned = [alignment](uint32_t n) {
        return (static_cast<uint64_t>(n) + alignment - 1U) & ~static_cast<uint64_t>(alignment - 1U);
    };
    // Require at least 5% physical saving; headers/padding cannot turn a win into expansion.
    if (packed >= raw || aligned(packed) * 100U >= aligned(raw) * 95U) return raw;
    uint32_t cursor = masks * 8U;
    for (uint32_t begin = 0, m = 0; begin < count; begin += 64U, ++m) {
        uint64_t mask = 0;
        const uint32_t n = std::min<uint32_t>(64U, count - begin);
        for (uint32_t j = 0; j < n; ++j) {
            const auto bits = bc_zero_bits(input[begin + j]);
            if (bits != zero_bits) {
                mask |= uint64_t{1} << j;
                std::memcpy(output + cursor, &bits, sizeof(T));
                cursor += sizeof(T);
            }
        }
        std::memcpy(output + m * 8U, &mask, 8U);
    }
    if (cursor != packed) throw std::logic_error("BC zero block encode length mismatch");
    return kBCZeroBitmapFlag | packed;
}

template<class T>
void bc_zero_decode_block(const uint8_t* input, uint32_t encoded, uint32_t count, T zero, T* output) {
    if (count > kBCZeroBlockBytes / sizeof(T) || (count && !output))
        throw std::invalid_argument("BC zero block output invalid");
    const uint32_t bytes = encoded & ~kBCZeroBitmapFlag;
    if (!(encoded & kBCZeroBitmapFlag)) {
        if (bytes != count * sizeof(T) || (bytes && !input))
            throw std::runtime_error("BC zero RAW length mismatch");
        if (bytes && input != reinterpret_cast<const uint8_t*>(output)) std::memcpy(output, input, bytes);
        return;
    }
    if (bytes == 0) { std::fill_n(output, count, zero); return; }
    const uint32_t masks = (count + 63U) / 64U;
    uint32_t cursor = masks * 8U;
    if (!input || bytes < cursor || bytes >= count * sizeof(T) || (bytes - cursor) % sizeof(T))
        throw std::runtime_error("BC zero bitmap length invalid");
    for (uint32_t begin = 0, m = 0; begin < count; begin += 64U, ++m) {
        uint64_t mask;
        std::memcpy(&mask, input + m * 8U, 8U);
        const uint32_t n = std::min<uint32_t>(64U, count - begin);
        if (n < 64U && (mask >> n)) throw std::runtime_error("BC zero bitmap tail invalid");
        for (uint32_t j = 0; j < n; ++j) {
            if ((mask >> j) & 1U) {
                if (cursor > bytes || sizeof(T) > bytes - cursor)
                    throw std::runtime_error("BC zero bitmap payload truncated");
                std::memcpy(output + begin + j, input + cursor, sizeof(T));
                cursor += sizeof(T);
            } else output[begin + j] = zero;
        }
    }
    if (cursor != bytes) throw std::runtime_error("BC zero bitmap trailing payload");
}
} // namespace BC
