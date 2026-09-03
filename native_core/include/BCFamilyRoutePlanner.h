#pragma once

#include "BCDynamicCapacity.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace BC {

enum class BCFamilyGenerationRoute {
    Auto,
    Resident,
    Single,
    Family,
};

struct BCFamilyRouteInputs {
    uint64_t source2_size = 0U;
    uint64_t source4_size = 0U;
    bool has_source4 = false;
    uint64_t source2_bucket_count = 0U;
    uint64_t source2_rank_payload_bytes = 0U;
    uint64_t source4_bucket_count = 0U;
    uint64_t source4_rank_payload_bytes = 0U;
    double resident_dynamic_reserve_factor = 1.0;
    double resident_secondary_dynamic_reserve_factor = 0.0;
    double single_dynamic_reserve_factor = 1.0;
    double single_secondary_dynamic_reserve_factor = 0.0;
    bool has_resident_carry = false;
    bool has_single_carry = false;
    bool has_secondary = true;
    bool evaluate_dynamic_address_space = false;
    uint64_t available_memory_bytes = 0U;
    uint64_t total_memory_bytes = 0U;
    uint32_t fixed_modulus = 0U;
    uint32_t previous_modulus = 0U;
    BCFamilyGenerationRoute previous_route = BCFamilyGenerationRoute::Resident;
    uint32_t resident_upgrade_streak = 0U;
    uint32_t single_upgrade_streak = 0U;
};

struct BCFamilyRouteDecision {
    BCFamilyGenerationRoute route = BCFamilyGenerationRoute::Family;
    uint32_t target_modulus = 29U;
    uint64_t available_memory_bytes = 0U;
    uint64_t route_budget_bytes = 0U;
    uint64_t route_estimated_peak_bytes = 0U;
    uint64_t resident_estimated_peak_bytes = 0U;
    uint64_t single_estimated_peak_bytes = 0U;
    uint64_t family_estimated_peak_bytes = 0U;
    uint64_t resident_dynamic_bitmap_words = 0U;
    uint64_t resident_dynamic_hash_capacity = 0U;
    uint64_t single_dynamic_bitmap_words = 0U;
    uint64_t single_dynamic_hash_capacity = 0U;
    uint64_t route_dynamic_bitmap_words = 0U;
    uint64_t route_dynamic_hash_capacity = 0U;
    bool resident_addressable = true;
    bool single_addressable = true;
    bool resident_admission_addressable = true;
    bool single_admission_addressable = true;
    bool route_addressable = true;
    bool dynamic_address_limited = false;
    bool automatic_route = false;
    uint32_t resident_upgrade_streak = 0U;
    uint32_t single_upgrade_streak = 0U;
};

inline constexpr uint64_t kBCFamilyRouteGiB = 1024ULL * 1024ULL * 1024ULL;

struct BCFamilyDynamicRouteEstimate {
    uint64_t bitmap_words = 0U;
    uint64_t hash_capacity = 0U;
    bool addressable = true;
    bool admission_addressable = true;
    BCDynamicAddressLimit limit = BCDynamicAddressLimit::None;
};

[[nodiscard]] inline uint64_t bc_family_rank_payload_word_estimate(uint64_t bytes) {
    return bytes / sizeof(uint64_t) + 1U;
}

[[nodiscard]] inline BCFamilyDynamicRouteEstimate bc_family_dynamic_route_estimate(
    const BCFamilyRouteInputs &inputs,
    double primary_reserve_factor,
    double secondary_reserve_factor,
    bool has_carry
) {
    BCFamilyDynamicRouteEstimate route;
    if (!inputs.evaluate_dynamic_address_space) {
        return route;
    }

    auto include_state = [&](uint64_t source_buckets, uint64_t source_bitmap_words, double factor) {
        const uint64_t bucket_estimate = bc_dynamic_scaled_estimate(
            source_buckets,
            factor,
            kBCDynamicBucketPadding);
        const uint64_t bitmap_estimate = bc_dynamic_scaled_estimate(
            source_bitmap_words,
            factor,
            kBCDynamicBitmapWordPadding);
        const BCDynamicCapacityEstimate capacity =
            bc_estimate_dynamic_capacity(bucket_estimate, bitmap_estimate);
        route.bitmap_words = std::max(route.bitmap_words, capacity.bitmap_word_estimate);
        route.hash_capacity = std::max(route.hash_capacity, capacity.hash_capacity);
        if (capacity.bucket_estimate > kBCDynamicBucketEstimateAdmissionLimit ||
            capacity.bitmap_word_estimate > kBCDynamicBitmapWordAdmissionLimit) {
            route.admission_addressable = false;
        }
        if (!capacity.addressable()) {
            route.addressable = false;
            if (route.limit == BCDynamicAddressLimit::None) {
                route.limit = capacity.limit;
            } else if (route.limit != capacity.limit) {
                route.limit = BCDynamicAddressLimit::HashAndBitmap;
            }
        }
    };

    if (!has_carry) {
        if (inputs.has_source4) {
            include_state(
                inputs.source4_bucket_count,
                bc_family_rank_payload_word_estimate(inputs.source4_rank_payload_bytes),
                secondary_reserve_factor);
        } else {
            include_state(
                inputs.source2_bucket_count,
                bc_family_rank_payload_word_estimate(inputs.source2_rank_payload_bytes),
                primary_reserve_factor);
        }
    }
    if (inputs.has_secondary) {
        include_state(
            inputs.source2_bucket_count,
            bc_family_rank_payload_word_estimate(inputs.source2_rank_payload_bytes),
            secondary_reserve_factor);
    }
    return route;
}

[[nodiscard]] inline const char *bc_family_route_name(BCFamilyGenerationRoute route) {
    switch (route) {
    case BCFamilyGenerationRoute::Auto:
        return "auto";
    case BCFamilyGenerationRoute::Resident:
        return "resident";
    case BCFamilyGenerationRoute::Single:
        return "single";
    case BCFamilyGenerationRoute::Family:
        return "family";
    }
    return "family";
}

[[nodiscard]] inline BCFamilyGenerationRoute bc_parse_family_route(const std::string &value) {
    if (value == "auto") {
        return BCFamilyGenerationRoute::Auto;
    }
    if (value == "resident") {
        return BCFamilyGenerationRoute::Resident;
    }
    if (value == "single") {
        return BCFamilyGenerationRoute::Single;
    }
    if (value == "family") {
        return BCFamilyGenerationRoute::Family;
    }
    throw std::invalid_argument("--family-route must be auto, resident, single, or family");
}

[[nodiscard]] inline bool bc_family_route_is_faster(
    BCFamilyGenerationRoute candidate,
    BCFamilyGenerationRoute current
) {
    auto rank = [](BCFamilyGenerationRoute route) -> uint32_t {
        switch (route) {
        case BCFamilyGenerationRoute::Resident:
            return 0U;
        case BCFamilyGenerationRoute::Single:
            return 1U;
        case BCFamilyGenerationRoute::Family:
            return 2U;
        case BCFamilyGenerationRoute::Auto:
            return 3U;
        }
        return 3U;
    };
    return rank(candidate) < rank(current);
}

[[nodiscard]] inline uint64_t bc_route_mul_div_ceil_u64(
    uint64_t value,
    uint64_t mul,
    uint64_t div
) {
    if (div == 0U) {
        throw std::invalid_argument("BC route div must be non-zero");
    }
    const long double scaled =
        static_cast<long double>(value) * static_cast<long double>(mul) /
        static_cast<long double>(div);
    if (scaled >= static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
        return std::numeric_limits<uint64_t>::max();
    }
    return static_cast<uint64_t>(scaled + 0.999999999L);
}

[[nodiscard]] inline uint64_t bc_family_route_budget(
    uint64_t available_memory_bytes,
    uint64_t total_memory_bytes
) {
    const uint64_t reserve_by_total = bc_route_mul_div_ceil_u64(total_memory_bytes, 3U, 100U);
    const uint64_t reserve = std::max<uint64_t>(kBCFamilyRouteGiB, reserve_by_total);
    return available_memory_bytes > reserve ? available_memory_bytes - reserve : 0U;
}

[[nodiscard]] inline uint64_t bc_family_estimate_for_modulus(uint64_t largest_source_size, uint32_t modulus) {
    if (modulus == 0U) {
        throw std::invalid_argument("BC family route modulus must be non-zero");
    }
    const uint64_t fixed = kBCFamilyRouteGiB / 4U;
    return fixed + bc_route_mul_div_ceil_u64(largest_source_size, 12U, modulus);
}

[[nodiscard]] inline uint32_t bc_family_route_fixed_modulus(const BCFamilyRouteInputs &inputs) {
    if (inputs.fixed_modulus != 0U) {
        return inputs.fixed_modulus;
    }
    if (inputs.previous_modulus != 0U) {
        return inputs.previous_modulus;
    }
    return 29U;
}

[[nodiscard]] inline BCFamilyRouteDecision bc_plan_family_generation_route(
    const BCFamilyRouteInputs &inputs,
    BCFamilyGenerationRoute requested_route
) {
    const uint64_t s2 = inputs.source2_size;
    const uint64_t s4 = inputs.has_source4 ? inputs.source4_size : inputs.source2_size;
    const uint64_t largest = std::max<uint64_t>(s2, s4);
    const uint64_t delta = largest - std::min<uint64_t>(s2, s4);
    const uint64_t budget =
        bc_family_route_budget(inputs.available_memory_bytes, inputs.total_memory_bytes);
    BCFamilyRouteDecision decision;
    decision.available_memory_bytes = inputs.available_memory_bytes;
    decision.route_budget_bytes = budget;
    decision.resident_estimated_peak_bytes =
        bc_route_mul_div_ceil_u64(largest, 7U, 2U) +
        bc_route_mul_div_ceil_u64(delta, 5U, 2U);
    decision.single_estimated_peak_bytes =
        bc_route_mul_div_ceil_u64(largest, 6U, 5U) + delta;

    decision.target_modulus = bc_family_route_fixed_modulus(inputs);
    decision.family_estimated_peak_bytes =
        bc_family_estimate_for_modulus(largest, decision.target_modulus);

    const BCFamilyDynamicRouteEstimate resident_dynamic =
        bc_family_dynamic_route_estimate(
            inputs,
            inputs.resident_dynamic_reserve_factor,
            inputs.resident_secondary_dynamic_reserve_factor > 0.0
                ? inputs.resident_secondary_dynamic_reserve_factor
                : inputs.resident_dynamic_reserve_factor * 2.0,
            inputs.has_resident_carry);
    const BCFamilyDynamicRouteEstimate single_dynamic =
        bc_family_dynamic_route_estimate(
            inputs,
            inputs.single_dynamic_reserve_factor,
            inputs.single_secondary_dynamic_reserve_factor > 0.0
                ? inputs.single_secondary_dynamic_reserve_factor
                : inputs.single_dynamic_reserve_factor * 2.0,
            inputs.has_single_carry);
    decision.resident_dynamic_bitmap_words = resident_dynamic.bitmap_words;
    decision.resident_dynamic_hash_capacity = resident_dynamic.hash_capacity;
    decision.single_dynamic_bitmap_words = single_dynamic.bitmap_words;
    decision.single_dynamic_hash_capacity = single_dynamic.hash_capacity;
    decision.resident_addressable = resident_dynamic.addressable;
    decision.single_addressable = single_dynamic.addressable;
    decision.resident_admission_addressable = resident_dynamic.admission_addressable;
    decision.single_admission_addressable = single_dynamic.admission_addressable;
    decision.automatic_route = requested_route == BCFamilyGenerationRoute::Auto;

    auto set_route = [&](BCFamilyGenerationRoute route) {
        decision.route = route;
        switch (route) {
        case BCFamilyGenerationRoute::Resident:
            decision.route_estimated_peak_bytes = decision.resident_estimated_peak_bytes;
            decision.route_dynamic_bitmap_words = decision.resident_dynamic_bitmap_words;
            decision.route_dynamic_hash_capacity = decision.resident_dynamic_hash_capacity;
            decision.route_addressable = decision.resident_addressable;
            break;
        case BCFamilyGenerationRoute::Single:
            decision.route_estimated_peak_bytes = decision.single_estimated_peak_bytes;
            decision.route_dynamic_bitmap_words = decision.single_dynamic_bitmap_words;
            decision.route_dynamic_hash_capacity = decision.single_dynamic_hash_capacity;
            decision.route_addressable = decision.single_addressable;
            break;
        case BCFamilyGenerationRoute::Family:
        case BCFamilyGenerationRoute::Auto:
            decision.route = BCFamilyGenerationRoute::Family;
            decision.route_estimated_peak_bytes = decision.family_estimated_peak_bytes;
            decision.route_dynamic_bitmap_words = 0U;
            decision.route_dynamic_hash_capacity = 0U;
            decision.route_addressable = true;
            break;
        }
    };

    if (requested_route != BCFamilyGenerationRoute::Auto) {
        set_route(requested_route);
        return decision;
    }

    const bool resident_memory_fits = decision.resident_estimated_peak_bytes <= budget;
    const bool single_memory_fits = decision.single_estimated_peak_bytes <= budget;
    BCFamilyGenerationRoute selected = BCFamilyGenerationRoute::Family;
    if (resident_memory_fits) {
        if (decision.resident_admission_addressable) {
            selected = BCFamilyGenerationRoute::Resident;
        } else {
            decision.dynamic_address_limited = true;
        }
    } else if (single_memory_fits) {
        if (decision.single_admission_addressable) {
            selected = BCFamilyGenerationRoute::Single;
        } else {
            decision.dynamic_address_limited = true;
        }
    }

    BCFamilyGenerationRoute previous = inputs.previous_route == BCFamilyGenerationRoute::Auto
        ? BCFamilyGenerationRoute::Resident
        : inputs.previous_route;
    decision.resident_upgrade_streak = inputs.resident_upgrade_streak;
    decision.single_upgrade_streak = inputs.single_upgrade_streak;
    if (bc_family_route_is_faster(selected, previous)) {
        if (selected == BCFamilyGenerationRoute::Resident) {
            decision.resident_upgrade_streak = inputs.resident_upgrade_streak + 1U;
            decision.single_upgrade_streak = 0U;
            if (decision.resident_upgrade_streak < 2U) {
                selected = previous;
            }
        } else if (selected == BCFamilyGenerationRoute::Single) {
            decision.single_upgrade_streak = inputs.single_upgrade_streak + 1U;
            decision.resident_upgrade_streak = 0U;
            if (decision.single_upgrade_streak < 2U) {
                selected = previous;
            }
        }
    } else {
        decision.resident_upgrade_streak = selected == BCFamilyGenerationRoute::Resident
            ? decision.resident_upgrade_streak
            : 0U;
        decision.single_upgrade_streak = selected == BCFamilyGenerationRoute::Single
            ? decision.single_upgrade_streak
            : 0U;
    }

    if (selected == BCFamilyGenerationRoute::Resident &&
        decision.resident_estimated_peak_bytes > budget) {
        selected = single_memory_fits && decision.single_admission_addressable
            ? BCFamilyGenerationRoute::Single
            : BCFamilyGenerationRoute::Family;
    }
    if (selected == BCFamilyGenerationRoute::Single &&
        decision.single_estimated_peak_bytes > budget) {
        selected = BCFamilyGenerationRoute::Family;
    }
    if (selected == BCFamilyGenerationRoute::Resident && !decision.resident_admission_addressable) {
        selected = BCFamilyGenerationRoute::Family;
        decision.dynamic_address_limited = true;
    }
    if (selected == BCFamilyGenerationRoute::Single && !decision.single_admission_addressable) {
        selected = BCFamilyGenerationRoute::Family;
        decision.dynamic_address_limited = true;
    }
    set_route(selected);
    return decision;
}

enum class BCSolveRoute {
    Auto,
    Resident,
    Single,
    Family,
};

struct BCSolveRouteInputs {
    uint64_t current_rows = 0U;
    uint64_t future2_live_rows = 0U;
    uint64_t future4_live_rows = 0U;
    uint64_t available_memory_bytes = 0U;
    uint32_t fixed_modulus = 0U;
};

struct BCSolveRouteDecision {
    BCSolveRoute route = BCSolveRoute::Family;
    uint32_t family_modulus = 0U;
    uint64_t available_memory_bytes = 0U;
    uint64_t resident_required_bytes = 0U;
    uint64_t single_required_bytes = 0U;
    uint64_t route_required_bytes = 0U;
};

[[nodiscard]] inline const char *bc_solve_route_name(BCSolveRoute route) {
    switch (route) {
    case BCSolveRoute::Auto:
        return "auto";
    case BCSolveRoute::Resident:
        return "resident";
    case BCSolveRoute::Single:
        return "single";
    case BCSolveRoute::Family:
        return "family";
    }
    return "family";
}

[[nodiscard]] inline BCSolveRoute bc_parse_solve_route(const std::string &value) {
    if (value == "auto") {
        return BCSolveRoute::Auto;
    }
    if (value == "resident") {
        return BCSolveRoute::Resident;
    }
    if (value == "single") {
        return BCSolveRoute::Single;
    }
    if (value == "family") {
        return BCSolveRoute::Family;
    }
    throw std::invalid_argument("--solve-route must be auto, resident, single, or family");
}

[[nodiscard]] inline uint64_t bc_route_saturating_add_u64(uint64_t lhs, uint64_t rhs) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        return std::numeric_limits<uint64_t>::max();
    }
    return lhs + rhs;
}

[[nodiscard]] inline uint64_t bc_route_saturating_mul_u64(uint64_t value, uint64_t multiplier) {
    if (multiplier != 0U && value > std::numeric_limits<uint64_t>::max() / multiplier) {
        return std::numeric_limits<uint64_t>::max();
    }
    return value * multiplier;
}

[[nodiscard]] inline uint64_t bc_solve_resident_required_bytes(
    uint64_t current_rows,
    uint64_t future2_live_rows,
    uint64_t future4_live_rows
) {
    uint64_t rows = bc_route_saturating_add_u64(current_rows, future2_live_rows);
    rows = bc_route_saturating_add_u64(rows, future4_live_rows);
    return bc_route_saturating_add_u64(
        kBCFamilyRouteGiB,
        bc_route_saturating_mul_u64(rows, 5U));
}

[[nodiscard]] inline uint64_t bc_solve_single_required_bytes(
    uint64_t future2_live_rows,
    uint64_t future4_live_rows
) {
    const uint64_t max_future_rows = std::max<uint64_t>(future2_live_rows, future4_live_rows);
    return bc_route_saturating_add_u64(
        kBCFamilyRouteGiB,
        bc_route_saturating_mul_u64(max_future_rows, 6U));
}

[[nodiscard]] inline BCSolveRouteDecision bc_plan_solve_route(
    const BCSolveRouteInputs &inputs,
    BCSolveRoute requested_route = BCSolveRoute::Auto
) {
    if (inputs.fixed_modulus == 0U) {
        throw std::invalid_argument("BC solve route fixed_modulus must be non-zero");
    }
    BCSolveRouteDecision decision;
    decision.family_modulus = inputs.fixed_modulus;
    decision.available_memory_bytes = inputs.available_memory_bytes;
    decision.resident_required_bytes = bc_solve_resident_required_bytes(
        inputs.current_rows,
        inputs.future2_live_rows,
        inputs.future4_live_rows);
    decision.single_required_bytes = bc_solve_single_required_bytes(
        inputs.future2_live_rows,
        inputs.future4_live_rows);

    auto set_route = [&](BCSolveRoute route) {
        decision.route = route == BCSolveRoute::Auto ? BCSolveRoute::Family : route;
        switch (decision.route) {
        case BCSolveRoute::Resident:
            decision.route_required_bytes = decision.resident_required_bytes;
            break;
        case BCSolveRoute::Single:
            decision.route_required_bytes = decision.single_required_bytes;
            break;
        case BCSolveRoute::Family:
        case BCSolveRoute::Auto:
            decision.route = BCSolveRoute::Family;
            decision.route_required_bytes = 0U;
            break;
        }
    };

    if (requested_route != BCSolveRoute::Auto) {
        set_route(requested_route);
        return decision;
    }
    if (inputs.available_memory_bytes >= decision.resident_required_bytes) {
        set_route(BCSolveRoute::Resident);
    } else if (inputs.available_memory_bytes >= decision.single_required_bytes) {
        set_route(BCSolveRoute::Single);
    } else {
        set_route(BCSolveRoute::Family);
    }
    return decision;
}

} // namespace BC
