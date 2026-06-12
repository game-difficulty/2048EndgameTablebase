#include "BCFamilyRoutePlanner.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

constexpr uint64_t GiB = BC::kBCFamilyRouteGiB;

void check(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

BC::BCFamilyRouteInputs base_inputs() {
    BC::BCFamilyRouteInputs inputs;
    inputs.source2_size = GiB;
    inputs.source4_size = GiB;
    inputs.has_source4 = true;
    inputs.total_memory_bytes = 0U;
    inputs.available_memory_bytes = 16U * GiB;
    inputs.previous_modulus = 29U;
    inputs.previous_route = BC::BCFamilyGenerationRoute::Family;
    return inputs;
}

void test_explicit_routes_use_requested_estimates() {
    BC::BCFamilyRouteInputs inputs = base_inputs();
    const BC::BCFamilyRouteDecision resident =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Resident);
    check(resident.route == BC::BCFamilyGenerationRoute::Resident, "explicit resident route mismatch");
    check(
        resident.route_estimated_peak_bytes == resident.resident_estimated_peak_bytes,
        "resident route should report resident estimate"
    );

    const BC::BCFamilyRouteDecision single =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Single);
    check(single.route == BC::BCFamilyGenerationRoute::Single, "explicit single route mismatch");
    check(
        single.route_estimated_peak_bytes == single.single_estimated_peak_bytes,
        "single route should report single estimate"
    );
}

void test_family_modulus_selection() {
    const uint64_t largest = 10U * GiB;
    const uint64_t budget_for_47 =
        BC::bc_family_estimate_for_modulus(largest, 47U);
    const uint64_t below_43 =
        BC::bc_family_estimate_for_modulus(largest, 43U) - 1U;
    check(budget_for_47 <= below_43, "test budget ordering should put 47 before 43");
    check(
        BC::bc_choose_family_modulus(largest, below_43) == 47U,
        "family modulus should choose smallest satisfying prime"
    );
    check(
        BC::bc_choose_family_modulus(largest, 0U) == 293U,
        "family modulus should fall back to 293 when no prime fits"
    );
}

void test_previous_modulus_sticky_band() {
    const uint64_t largest = GiB;
    const uint32_t previous = 47U;
    const uint64_t fixed = GiB / 5U;
    const uint64_t sticky_budget = fixed + (12U * largest) / previous;
    const uint64_t too_tight_budget = fixed + (8U * largest) / previous;
    const uint64_t too_loose_budget = fixed + (17U * largest) / previous;
    check(
        BC::bc_keep_previous_family_modulus(largest, sticky_budget, previous),
        "previous modulus should stay in 9..16 k band"
    );
    check(
        !BC::bc_keep_previous_family_modulus(largest, too_tight_budget, previous),
        "previous modulus should adjust below sticky band"
    );
    check(
        !BC::bc_keep_previous_family_modulus(largest, too_loose_budget, previous),
        "previous modulus should adjust above sticky band"
    );
}

void test_auto_upgrade_requires_two_layers() {
    BC::BCFamilyRouteInputs inputs = base_inputs();
    inputs.available_memory_bytes = 16U * GiB;
    inputs.previous_route = BC::BCFamilyGenerationRoute::Family;
    BC::BCFamilyRouteDecision first =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Auto);
    check(first.route == BC::BCFamilyGenerationRoute::Family, "first auto upgrade should be delayed");
    check(first.resident_upgrade_streak == 1U, "first resident upgrade streak mismatch");

    inputs.resident_upgrade_streak = first.resident_upgrade_streak;
    inputs.single_upgrade_streak = first.single_upgrade_streak;
    BC::BCFamilyRouteDecision second =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Auto);
    check(second.route == BC::BCFamilyGenerationRoute::Resident, "second auto upgrade should select resident");
}

void test_auto_downgrade_is_immediate() {
    BC::BCFamilyRouteInputs inputs = base_inputs();
    inputs.available_memory_bytes = 3U * GiB;
    inputs.previous_route = BC::BCFamilyGenerationRoute::Resident;
    BC::BCFamilyRouteDecision decision =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Auto);
    check(decision.route == BC::BCFamilyGenerationRoute::Single, "route downgrade to single should be immediate");
}

void test_auto_route_transitions_cover_three_routes() {
    BC::BCFamilyRouteInputs inputs = base_inputs();
    inputs.available_memory_bytes = 3U * GiB;
    inputs.previous_route = BC::BCFamilyGenerationRoute::Family;
    BC::BCFamilyRouteDecision first_single =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Auto);
    check(first_single.route == BC::BCFamilyGenerationRoute::Family, "first family to single upgrade should wait");
    inputs.single_upgrade_streak = first_single.single_upgrade_streak;
    BC::BCFamilyRouteDecision second_single =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Auto);
    check(second_single.route == BC::BCFamilyGenerationRoute::Single, "second family to single upgrade should switch");

    inputs.previous_route = BC::BCFamilyGenerationRoute::Single;
    inputs.single_upgrade_streak = 0U;
    inputs.resident_upgrade_streak = 0U;
    inputs.available_memory_bytes = 6U * GiB;
    BC::BCFamilyRouteDecision first_resident =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Auto);
    check(first_resident.route == BC::BCFamilyGenerationRoute::Single, "first single to resident upgrade should wait");
    inputs.resident_upgrade_streak = first_resident.resident_upgrade_streak;
    BC::BCFamilyRouteDecision second_resident =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Auto);
    check(second_resident.route == BC::BCFamilyGenerationRoute::Resident, "second single to resident upgrade should switch");

    inputs.previous_route = BC::BCFamilyGenerationRoute::Resident;
    inputs.available_memory_bytes = 3U * GiB;
    BC::BCFamilyRouteDecision down_to_single =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Auto);
    check(down_to_single.route == BC::BCFamilyGenerationRoute::Single, "resident to single downgrade should be immediate");
    inputs.previous_route = BC::BCFamilyGenerationRoute::Single;
    inputs.available_memory_bytes = GiB;
    BC::BCFamilyRouteDecision down_to_family =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Auto);
    check(down_to_family.route == BC::BCFamilyGenerationRoute::Family, "single to family downgrade should be immediate");
}

} // namespace

int main() {
    try {
        test_explicit_routes_use_requested_estimates();
        test_family_modulus_selection();
        test_previous_modulus_sticky_band();
        test_auto_upgrade_requires_two_layers();
        test_auto_downgrade_is_immediate();
        test_auto_route_transitions_cover_three_routes();
        std::cout << "bc_family_route_planner_test passed\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "bc_family_route_planner_test failed: " << ex.what() << '\n';
        return 1;
    }
}
