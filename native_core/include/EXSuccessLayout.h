#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

// EX offsets remain uint32 units. Only decoded addresses are uint64.
namespace EXSuccessLayout {

constexpr uint32_t kLegacyVersion = 5;
constexpr uint32_t kStridedVersion = 6;

inline uint64_t stride(uint32_t version, uint16_t shift) {
    if (version == kLegacyVersion) return 1;
    if (version != kStridedVersion || shift > 31)
        throw std::runtime_error("unsupported EX success layout");
    return uint64_t{1} << shift;
}

inline uint16_t shift(uint64_t stride) {
    if (!stride || (stride & (stride - 1)) || stride > (uint64_t{1} << 31))
        throw std::runtime_error("invalid EX success stride");
    uint16_t result = 0;
    while (stride > 1) { stride >>= 1; ++result; }
    return result;
}

inline uint32_t version(uint64_t stride) {
    return stride == 1 ? kLegacyVersion : kStridedVersion;
}

inline uint64_t units(uint64_t count, uint64_t stride) {
    return count / stride + (count % stride != 0);
}

inline uint64_t aligned(uint64_t count, uint64_t stride) {
    return units(count, stride) * stride;
}

inline uint64_t decode(uint32_t offset, uint64_t stride) {
    return uint64_t{offset} * stride;
}

inline uint64_t storage_count(uint32_t version, uint16_t shift, uint64_t stored,
                              uint64_t live, uint64_t payload) {
    const uint64_t scale = stride(version, shift);
    const uint64_t count = version == kLegacyVersion ? live : stored;
    if (count < live || count % scale || (payload && payload != count) ||
        count / scale > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("invalid EX success storage count");
    return count;
}

template <typename Count>
uint64_t choose_stride(size_t buckets, Count count, uint64_t live, uint64_t minimum = 1) {
    shift(minimum);
    uint64_t scale = minimum;
    for (;;) {
        uint64_t total = units(live, scale);
        if (total <= std::numeric_limits<uint32_t>::max()) {
            if (scale == 1) return scale;
            total = 0;
            for (size_t i = 0; i < buckets; ++i) total += units(count(i), scale);
            if (total <= std::numeric_limits<uint32_t>::max()) return scale;
        }
        if (scale == (uint64_t{1} << 31))
            throw std::runtime_error("EX success offsets exceed supported stride capacity");
        scale *= 2;
    }
}

} // namespace EXSuccessLayout
