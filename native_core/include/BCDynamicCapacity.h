#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace BC {

inline constexpr uint64_t kBCDynamicBitmapWordLimit =
    static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
inline constexpr uint64_t kBCDynamicHashCapacityLimit = 1ULL << 31U;
inline constexpr uint64_t kBCDynamicHashLoadNumerator = 75U;
inline constexpr uint64_t kBCDynamicHashLoadDenominator = 100U;
inline constexpr uint64_t kBCDynamicBucketEstimateLimit =
    (kBCDynamicHashCapacityLimit * kBCDynamicHashLoadNumerator) /
    kBCDynamicHashLoadDenominator;
inline constexpr uint64_t kBCDynamicAddressAdmissionNumerator = 92U;
inline constexpr uint64_t kBCDynamicAddressAdmissionDenominator = 100U;
inline constexpr uint64_t kBCDynamicBitmapWordAdmissionLimit =
    (kBCDynamicBitmapWordLimit * kBCDynamicAddressAdmissionNumerator) /
    kBCDynamicAddressAdmissionDenominator;
inline constexpr uint64_t kBCDynamicBucketEstimateAdmissionLimit =
    (kBCDynamicBucketEstimateLimit * kBCDynamicAddressAdmissionNumerator) /
    kBCDynamicAddressAdmissionDenominator;
inline constexpr uint64_t kBCDynamicBucketPadding = 4096ULL;
inline constexpr uint64_t kBCDynamicBitmapWordPadding = 512ULL * 64ULL;

enum class BCDynamicAddressLimit {
    None,
    HashCapacity,
    BitmapWords,
    HashAndBitmap,
};

struct BCDynamicCapacityEstimate {
    uint64_t bucket_estimate = 0U;
    uint64_t bitmap_word_estimate = 0U;
    uint64_t hash_required_capacity = 0U;
    uint64_t hash_capacity = 0U;
    bool hash_addressable = true;
    bool bitmap_addressable = true;
    BCDynamicAddressLimit limit = BCDynamicAddressLimit::None;

    [[nodiscard]] bool addressable() const noexcept {
        return hash_addressable && bitmap_addressable;
    }
};

class BCDynamicAddressabilityOverflow : public std::overflow_error {
public:
    explicit BCDynamicAddressabilityOverflow(const std::string &message)
        : std::overflow_error(message) {}
};

[[nodiscard]] inline uint64_t bc_dynamic_scaled_estimate(
    uint64_t value,
    double factor,
    uint64_t padding
) {
    if (!(factor > 0.0) || !std::isfinite(factor)) {
        return std::numeric_limits<uint64_t>::max();
    }
    const double scaled = static_cast<double>(value) * factor;
    const long double max_without_padding = static_cast<long double>(
        std::numeric_limits<uint64_t>::max() - padding);
    if (!std::isfinite(scaled) ||
        static_cast<long double>(scaled) > max_without_padding) {
        return std::numeric_limits<uint64_t>::max();
    }
    return static_cast<uint64_t>(scaled) + padding;
}

[[nodiscard]] inline BCDynamicCapacityEstimate bc_estimate_dynamic_capacity(
    uint64_t bucket_estimate,
    uint64_t bitmap_word_estimate
) {
    BCDynamicCapacityEstimate estimate;
    estimate.bucket_estimate = bucket_estimate;
    estimate.bitmap_word_estimate =
        std::max<uint64_t>(bitmap_word_estimate, kBCDynamicBitmapWordPadding);
    estimate.bitmap_addressable =
        estimate.bitmap_word_estimate <= kBCDynamicBitmapWordLimit;

    if (bucket_estimate > kBCDynamicBucketEstimateLimit) {
        estimate.hash_addressable = false;
        estimate.hash_required_capacity = kBCDynamicHashCapacityLimit + 1U;
        estimate.hash_capacity = kBCDynamicHashCapacityLimit + 1U;
    } else {
        const uint64_t normalized = std::max<uint64_t>(bucket_estimate, 1U);
        estimate.hash_required_capacity =
            (normalized * kBCDynamicHashLoadDenominator +
             (kBCDynamicHashLoadNumerator - 1U)) /
            kBCDynamicHashLoadNumerator;
        uint64_t capacity = 1U;
        while (capacity < estimate.hash_required_capacity) {
            capacity <<= 1U;
        }
        estimate.hash_capacity = std::max<uint64_t>(1024U, capacity);
        estimate.hash_addressable =
            estimate.hash_capacity <= kBCDynamicHashCapacityLimit;
    }

    if (!estimate.hash_addressable && !estimate.bitmap_addressable) {
        estimate.limit = BCDynamicAddressLimit::HashAndBitmap;
    } else if (!estimate.hash_addressable) {
        estimate.limit = BCDynamicAddressLimit::HashCapacity;
    } else if (!estimate.bitmap_addressable) {
        estimate.limit = BCDynamicAddressLimit::BitmapWords;
    }
    return estimate;
}

[[nodiscard]] inline const char *bc_dynamic_address_limit_name(BCDynamicAddressLimit limit) {
    switch (limit) {
    case BCDynamicAddressLimit::None:
        return "none";
    case BCDynamicAddressLimit::HashCapacity:
        return "hash_capacity";
    case BCDynamicAddressLimit::BitmapWords:
        return "bitmap_words";
    case BCDynamicAddressLimit::HashAndBitmap:
        return "hash_and_bitmap";
    }
    return "unknown";
}

} // namespace BC
