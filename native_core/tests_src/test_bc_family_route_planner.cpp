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
    inputs.fixed_modulus = 29U;
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

void test_generation_modulus_stays_fixed() {
    BC::BCFamilyRouteInputs inputs = base_inputs();
    inputs.fixed_modulus = 61U;
    inputs.previous_modulus = 29U;
    inputs.available_memory_bytes = 16U * GiB;
    const BC::BCFamilyRouteDecision resident =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Auto);
    check(resident.target_modulus == 61U, "auto generation route should keep fixed modulus");

    inputs.available_memory_bytes = 0U;
    const BC::BCFamilyRouteDecision family =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Auto);
    check(family.route == BC::BCFamilyGenerationRoute::Family, "tight generation route should fall back to family");
    check(family.target_modulus == 61U, "family generation route should keep fixed modulus");

    inputs.fixed_modulus = 100U;
    const BC::BCFamilyRouteDecision non_prime =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Family);
    check(non_prime.target_modulus == 100U, "generation route should not require prime modulus");
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

void test_dynamic_capacity_uint32_boundaries() {
    const BC::BCDynamicCapacityEstimate bitmap_at_limit =
        BC::bc_estimate_dynamic_capacity(1U, BC::kBCDynamicBitmapWordLimit);
    check(bitmap_at_limit.addressable(), "bitmap word limit should be addressable");
    const BC::BCDynamicCapacityEstimate bitmap_over_limit =
        BC::bc_estimate_dynamic_capacity(1U, BC::kBCDynamicBitmapWordLimit + 1U);
    check(!bitmap_over_limit.bitmap_addressable, "bitmap word limit + 1 should be rejected");

    const BC::BCDynamicCapacityEstimate hash_at_limit =
        BC::bc_estimate_dynamic_capacity(BC::kBCDynamicBucketEstimateLimit, 1U);
    check(hash_at_limit.addressable(), "maximum hash bucket estimate should be addressable");
    check(
        hash_at_limit.hash_capacity == BC::kBCDynamicHashCapacityLimit,
        "maximum hash bucket estimate should select the uint32 power-of-two limit"
    );
    const BC::BCDynamicCapacityEstimate hash_over_limit =
        BC::bc_estimate_dynamic_capacity(BC::kBCDynamicBucketEstimateLimit + 1U, 1U);
    check(!hash_over_limit.hash_addressable, "hash bucket estimate over limit should be rejected");
}

void test_auto_generation_address_limit_routes_directly_to_family() {
    BC::BCFamilyRouteInputs inputs = base_inputs();
    inputs.previous_route = BC::BCFamilyGenerationRoute::Resident;
    inputs.evaluate_dynamic_address_space = true;
    inputs.has_resident_carry = true;
    inputs.has_single_carry = true;
    inputs.has_secondary = true;
    inputs.source2_bucket_count = 1000U;
    inputs.source2_rank_payload_bytes =
        (BC::kBCDynamicBitmapWordLimit / 2U) * sizeof(uint64_t);
    inputs.source4_bucket_count = inputs.source2_bucket_count;
    inputs.source4_rank_payload_bytes = inputs.source2_rank_payload_bytes;
    inputs.resident_dynamic_reserve_factor = 2.0;
    inputs.single_dynamic_reserve_factor = 2.0;

    const BC::BCFamilyRouteDecision decision =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Auto);
    check(decision.route == BC::BCFamilyGenerationRoute::Family,
          "auto route should fall directly back to family on uint32 overflow");
    check(decision.dynamic_address_limited,
          "address-limited auto route should report its fallback reason");
    check(!decision.resident_addressable, "resident address estimate should fail");
    check(!decision.single_addressable, "single address estimate should fail");
}

void test_forced_dynamic_route_reports_address_limit_without_silent_reroute() {
    BC::BCFamilyRouteInputs inputs = base_inputs();
    inputs.evaluate_dynamic_address_space = true;
    inputs.has_resident_carry = true;
    inputs.has_secondary = true;
    inputs.source2_rank_payload_bytes =
        (BC::kBCDynamicBitmapWordLimit / 2U) * sizeof(uint64_t);
    inputs.resident_dynamic_reserve_factor = 2.0;
    const BC::BCFamilyRouteDecision decision =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Resident);
    check(decision.route == BC::BCFamilyGenerationRoute::Resident,
          "forced resident route should not be changed silently");
    check(!decision.route_addressable,
          "forced resident route should expose the addressability failure");
}

void test_continuous_single_address_limit_routes_to_family() {
    BC::BCFamilyRouteInputs inputs = base_inputs();
    inputs.available_memory_bytes = 3U * GiB;
    inputs.previous_route = BC::BCFamilyGenerationRoute::Single;
    inputs.evaluate_dynamic_address_space = true;
    inputs.has_resident_carry = true;
    inputs.has_single_carry = true;
    inputs.has_secondary = true;
    inputs.source2_bucket_count = 1000U;
    inputs.source2_rank_payload_bytes =
        (BC::kBCDynamicBitmapWordLimit / 2U) * sizeof(uint64_t);
    inputs.resident_dynamic_reserve_factor = 2.0;
    inputs.single_dynamic_reserve_factor = 2.0;
    const BC::BCFamilyRouteDecision decision =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Auto);
    check(decision.route == BC::BCFamilyGenerationRoute::Family,
          "continuous single next-carry overflow should route to family");
    check(decision.dynamic_address_limited,
          "continuous single overflow should report address limitation");
}

void test_resume_learned_factor_keeps_realistic_transition_addressable() {
    BC::BCFamilyRouteInputs inputs = base_inputs();
    inputs.evaluate_dynamic_address_space = true;
    inputs.previous_route = BC::BCFamilyGenerationRoute::Resident;
    inputs.source4_size = 14746107904ULL;
    inputs.source2_size = 13913313280ULL;
    inputs.source4_rank_payload_bytes = 13935851480ULL;
    inputs.source2_rank_payload_bytes = 13141761568ULL;
    inputs.source4_bucket_count = 1000000U;
    inputs.source2_bucket_count = 1000000U;
    inputs.available_memory_bytes = 128U * GiB;
    inputs.resident_dynamic_reserve_factor = 2.0;
    inputs.single_dynamic_reserve_factor = 2.0;
    const BC::BCFamilyRouteDecision bootstrap =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Auto);
    check(!bootstrap.resident_addressable,
          "default resume factor should reproduce the representative uint32 overflow");
    check(bootstrap.route == BC::BCFamilyGenerationRoute::Family,
          "unlearned representative resume should fall back to family");

    inputs.resident_dynamic_reserve_factor = 1.08;
    inputs.single_dynamic_reserve_factor = 1.08;
    inputs.has_resident_carry = false;
    inputs.has_single_carry = false;
    inputs.has_secondary = true;
    const BC::BCFamilyRouteDecision decision =
        BC::bc_plan_family_generation_route(inputs, BC::BCFamilyGenerationRoute::Auto);
    check(decision.resident_addressable,
          "learned resume factor should keep the representative carry bitmap addressable");
    check(decision.resident_dynamic_bitmap_words <= BC::kBCDynamicBitmapWordLimit,
          "learned resume factor bitmap estimate should stay within uint32 words");
    check(decision.route == BC::BCFamilyGenerationRoute::Resident,
          "learned representative resume should remain on resident route");
}

BC::BCSolveRouteInputs base_solve_inputs() {
    BC::BCSolveRouteInputs inputs;
    inputs.current_rows = 100U;
    inputs.future2_live_rows = 40U;
    inputs.future4_live_rows = 20U;
    inputs.fixed_modulus = 101U;
    return inputs;
}

void test_solve_route_thresholds() {
    BC::BCSolveRouteInputs inputs = base_solve_inputs();
    const uint64_t resident_required = BC::bc_solve_resident_required_bytes(
        inputs.current_rows,
        inputs.future2_live_rows,
        inputs.future4_live_rows);
    const uint64_t single_required = BC::bc_solve_single_required_bytes(
        inputs.future2_live_rows,
        inputs.future4_live_rows);
    check(resident_required == GiB + 5U * 160U, "resident solve memory formula mismatch");
    check(single_required == GiB + 6U * 40U, "single solve memory formula mismatch");

    inputs.available_memory_bytes = resident_required;
    BC::BCSolveRouteDecision resident = BC::bc_plan_solve_route(inputs);
    check(resident.route == BC::BCSolveRoute::Resident, "solve route should choose resident when it fits");
    check(resident.family_modulus == 101U, "resident solve route should keep fixed modulus");

    inputs.available_memory_bytes = resident_required - 1U;
    BC::BCSolveRouteDecision single = BC::bc_plan_solve_route(inputs);
    check(single.route == BC::BCSolveRoute::Single, "solve route should choose single when resident does not fit");
    check(single.route_required_bytes == single_required, "single solve route required bytes mismatch");

    inputs.available_memory_bytes = single_required - 1U;
    BC::BCSolveRouteDecision family = BC::bc_plan_solve_route(inputs);
    check(family.route == BC::BCSolveRoute::Family, "solve route should choose family when single does not fit");
    check(family.route_required_bytes == 0U, "family solve route should not claim a memory threshold");
    check(family.family_modulus == 101U, "family solve route should keep fixed modulus");
}

void test_solve_route_forced_selection_keeps_estimates() {
    BC::BCSolveRouteInputs inputs = base_solve_inputs();
    inputs.available_memory_bytes = 0U;
    const BC::BCSolveRouteDecision resident =
        BC::bc_plan_solve_route(inputs, BC::BCSolveRoute::Resident);
    check(resident.route == BC::BCSolveRoute::Resident, "forced resident solve route mismatch");
    check(
        resident.route_required_bytes == resident.resident_required_bytes,
        "forced resident should report resident requirement"
    );
    const BC::BCSolveRouteDecision single =
        BC::bc_plan_solve_route(inputs, BC::BCSolveRoute::Single);
    check(single.route == BC::BCSolveRoute::Single, "forced single solve route mismatch");
    check(single.family_modulus == 101U, "forced single solve route should keep fixed modulus");
}

} // namespace

int main() {
    try {
        test_explicit_routes_use_requested_estimates();
        test_generation_modulus_stays_fixed();
        test_auto_upgrade_requires_two_layers();
        test_auto_downgrade_is_immediate();
        test_auto_route_transitions_cover_three_routes();
        test_dynamic_capacity_uint32_boundaries();
        test_auto_generation_address_limit_routes_directly_to_family();
        test_forced_dynamic_route_reports_address_limit_without_silent_reroute();
        test_continuous_single_address_limit_routes_to_family();
        test_resume_learned_factor_keeps_realistic_transition_addressable();
        test_solve_route_thresholds();
        test_solve_route_forced_selection_keeps_estimates();
        std::cout << "bc_family_route_planner_test passed\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "bc_family_route_planner_test failed: " << ex.what() << '\n';
        return 1;
    }
}
