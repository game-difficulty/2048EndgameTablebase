#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace BC {

inline constexpr uint32_t kBCFamilyRouteMaxPrime = 251U;

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
    uint64_t available_memory_bytes = 0U;
    uint64_t total_memory_bytes = 0U;
    uint32_t previous_modulus = 0U;
    BCFamilyGenerationRoute previous_route = BCFamilyGenerationRoute::Family;
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
    uint32_t resident_upgrade_streak = 0U;
    uint32_t single_upgrade_streak = 0U;
};

inline constexpr uint64_t kBCFamilyRouteGiB = 1024ULL * 1024ULL * 1024ULL;

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

[[nodiscard]] inline const uint32_t *bc_family_route_primes(uint32_t &count) {
    static constexpr uint32_t kPrimes[] = {
        13U, 17U, 19U, 23U, 29U, 31U, 37U, 41U, 43U, 47U,
        53U, 59U, 61U, 67U, 71U, 73U, 79U, 83U, 89U, 97U,
        101U, 103U, 107U, 109U, 113U, 127U, 131U, 137U, 139U, 149U,
        151U, 157U, 163U, 167U, 173U, 179U, 181U, 191U, 193U, 197U,
        199U, 211U, 223U, 227U, 229U, 233U, 239U, 241U,
        kBCFamilyRouteMaxPrime
    };
    count = static_cast<uint32_t>(sizeof(kPrimes) / sizeof(kPrimes[0]));
    return kPrimes;
}

[[nodiscard]] inline bool bc_family_route_is_supported_prime(uint32_t value) {
    uint32_t count = 0U;
    const uint32_t *primes = bc_family_route_primes(count);
    for (uint32_t i = 0U; i < count; ++i) {
        if (primes[i] == value) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline uint64_t bc_family_estimate_for_modulus(uint64_t largest_source_size, uint32_t modulus) {
    if (modulus == 0U) {
        throw std::invalid_argument("BC family route modulus must be non-zero");
    }
    const uint64_t fixed = kBCFamilyRouteGiB / 4U;
    return fixed + bc_route_mul_div_ceil_u64(largest_source_size, 12U, modulus);
}

[[nodiscard]] inline uint32_t bc_choose_family_modulus(
    uint64_t largest_source_size,
    uint64_t budget
) {
    uint32_t count = 0U;
    const uint32_t *primes = bc_family_route_primes(count);
    for (uint32_t i = 0U; i < count; ++i) {
        if (bc_family_estimate_for_modulus(largest_source_size, primes[i]) <= budget) {
            return primes[i];
        }
    }
    return kBCFamilyRouteMaxPrime;
}

[[nodiscard]] inline bool bc_keep_previous_family_modulus(
    uint64_t largest_source_size,
    uint64_t budget,
    uint32_t previous_modulus
) {
    if (previous_modulus == 0U) {
        return false;
    }
    const uint64_t fixed = kBCFamilyRouteGiB / 5U;
    if (budget <= fixed || largest_source_size == 0U) {
        return false;
    }
    const long double k =
        (static_cast<long double>(budget - fixed) *
         static_cast<long double>(previous_modulus)) /
        static_cast<long double>(largest_source_size);
    return k >= 9.0L && k <= 16.0L;
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

    uint32_t modulus = inputs.previous_modulus;
    if (!bc_keep_previous_family_modulus(largest, budget, modulus)) {
        modulus = bc_choose_family_modulus(largest, budget);
    }
    decision.target_modulus = modulus == 0U ? kBCFamilyRouteMaxPrime : modulus;
    decision.family_estimated_peak_bytes =
        bc_family_estimate_for_modulus(largest, decision.target_modulus);

    auto set_route = [&](BCFamilyGenerationRoute route) {
        decision.route = route;
        switch (route) {
        case BCFamilyGenerationRoute::Resident:
            decision.route_estimated_peak_bytes = decision.resident_estimated_peak_bytes;
            break;
        case BCFamilyGenerationRoute::Single:
            decision.route_estimated_peak_bytes = decision.single_estimated_peak_bytes;
            break;
        case BCFamilyGenerationRoute::Family:
        case BCFamilyGenerationRoute::Auto:
            decision.route = BCFamilyGenerationRoute::Family;
            decision.route_estimated_peak_bytes = decision.family_estimated_peak_bytes;
            break;
        }
    };

    if (requested_route != BCFamilyGenerationRoute::Auto) {
        set_route(requested_route);
        return decision;
    }

    const bool resident_fits = decision.resident_estimated_peak_bytes <= budget;
    const bool single_fits = decision.single_estimated_peak_bytes <= budget;
    BCFamilyGenerationRoute selected = BCFamilyGenerationRoute::Family;
    if (resident_fits) {
        selected = BCFamilyGenerationRoute::Resident;
    } else if (single_fits) {
        selected = BCFamilyGenerationRoute::Single;
    }

    BCFamilyGenerationRoute previous = inputs.previous_route == BCFamilyGenerationRoute::Auto
        ? BCFamilyGenerationRoute::Family
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
        selected = single_fits ? BCFamilyGenerationRoute::Single : BCFamilyGenerationRoute::Family;
    }
    if (selected == BCFamilyGenerationRoute::Single &&
        decision.single_estimated_peak_bytes > budget) {
        selected = BCFamilyGenerationRoute::Family;
    }
    set_route(selected);
    return decision;
}

} // namespace BC
