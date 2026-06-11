#include "BCFamilyPartitionAnalysis.h"
#include "BCLut.h"

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void check(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Fn>
void expect_throws(Fn &&fn, const char *message) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, message);
}

std::vector<BC::LayerSum> sorted_sums(std::initializer_list<BC::LayerSum> values) {
    std::vector<BC::LayerSum> out(values);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

void test_exact_coord_set_weights() {
    const std::vector<BC::LayerSum> possible = sorted_sums({0U, 2U, 4U, 6U, 8U});
    const BC::BCExactFamilyCoordSet exact =
        BC::build_exact_family_coord_set_for_layer(8U, 2U, possible);
    check((exact.coords == std::vector<BC::FamilyCoord>{0U, 1U, 2U}), "exact coords mismatch");
    check(exact.ordered_side_pair_count.size() == 3U, "exact weights size mismatch");
    check(exact.ordered_side_pair_count[0] == 2U, "coord 0 ordered pair count mismatch");
    check(exact.ordered_side_pair_count[1] == 2U, "coord 1 ordered pair count mismatch");
    check(exact.ordered_side_pair_count[2] == 1U, "center coord ordered pair count mismatch");
}

void test_exact_partition_fanout_matches_t_delta_bound() {
    const std::vector<BC::LayerSum> possible = sorted_sums({0U, 2U, 4U, 6U, 8U, 10U, 12U});
    const BC::BCFamilyPartitionAnalysis analysis =
        BC::analyze_family_partition_transition(
            16U,
            2U,
            2U,
            possible,
            BC::BCFamilyPartitionPolicy::exact()
        );
    check(analysis.policy.kind == BC::BCFamilyPartitionKind::ExactCoord, "wrong policy");
    check(analysis.source_exact_coord_count > 0U, "source exact coord count should be non-zero");
    check(analysis.target_exact_coord_count > 0U, "target exact coord count should be non-zero");
    check(analysis.source_family_count == analysis.source_exact_coord_count, "exact source family count mismatch");
    check(analysis.max_target_fanout <= 2U, "exact partition fanout should stay <=2");
    check(analysis.source_families_with_fanout_gt2 == 0U, "exact partition should not have fanout >2");
}

void test_modulo_partition_groups_coords_and_reports_fanout() {
    const std::vector<BC::LayerSum> possible = sorted_sums({
        0U, 2U, 4U, 6U, 8U, 10U, 12U, 14U, 16U, 18U, 20U, 22U, 24U
    });
    const BC::BCFamilyPartitionPolicy policy = BC::BCFamilyPartitionPolicy::modulo(3U);
    const BC::BCFamilyPartitionAnalysis analysis =
        BC::analyze_family_partition_transition(24U, 2U, 2U, possible, policy);
    check(analysis.policy.kind == BC::BCFamilyPartitionKind::ModuloCoord, "wrong modulo policy");
    check(analysis.source_family_count == 3U, "modulo source family count should equal modulus");
    check(analysis.target_family_count == 3U, "modulo target family count should equal modulus");
    check(analysis.source_active_family_count == 3U, "all modulo source residues should be active");
    check(analysis.max_exact_coords_per_source_family > 1U, "modulo should group multiple exact coords");
    check(analysis.max_target_fanout >= 2U, "modulo fanout should be measured");
    check(analysis.max_target_need_cells_per_pass <= 9U, "modulo need cells cannot exceed full 3x3 matrix");
}

void test_modulo_partition_can_expose_fanout_over_two() {
    const std::vector<BC::LayerSum> possible = sorted_sums({
        0U, 2U, 4U, 6U, 8U, 10U, 12U, 14U, 16U, 18U, 20U, 22U, 24U,
        26U, 28U, 30U, 32U, 34U, 36U, 38U, 40U
    });
    const BC::BCFamilyPartitionAnalysis analysis =
        BC::analyze_family_partition_transition(
            18U,
            2U,
            2U,
            possible,
            BC::BCFamilyPartitionPolicy::modulo(3U)
        );
    check(analysis.source_families_with_fanout_gt2 > 0U, "modulo policy should report fanout >2 cases");
    check(analysis.max_target_fanout > 2U, "modulo max fanout should expose wider fanout");
}

void test_modulo_axis_is_dense_partition_id_axis() {
    const std::vector<BC::LayerSum> possible = sorted_sums({0U, 2U, 4U});
    const BC::BCFamilyTable axis = BC::build_family_partition_axis_for_layer(
        4U,
        2U,
        possible,
        BC::BCFamilyPartitionPolicy::modulo(29U)
    );
    check(axis.family_count() == 29U, "modulo axis should keep full dense modulus");
    check(axis.id_to_coord(28U) == 28U, "modulo axis coord table should store dense ids");
}

void test_runtime_partition_fanout_uses_exact_coord_groups() {
    const std::vector<BC::LayerSum> possible = sorted_sums({
        0U, 2U, 4U, 6U, 8U, 10U, 12U, 14U, 16U, 18U, 20U, 22U, 24U,
        26U, 28U, 30U, 32U, 34U, 36U, 38U, 40U
    });
    const BC::BCFamilyPartitionPolicy policy = BC::BCFamilyPartitionPolicy::modulo(3U);
    const BC::BCFamilyTable source_axis =
        BC::build_family_partition_axis_for_layer(18U, 2U, possible, policy);
    const BC::BCFamilyTable target_axis =
        BC::build_family_partition_axis_for_layer(22U, 2U, possible, policy);
    const BC::BCFamilyPartitionLayerMap source =
        BC::build_family_partition_layer_map(source_axis, possible, policy);
    const BC::BCFamilyPartitionLayerMap target =
        BC::build_family_partition_layer_map(target_axis, possible, policy);
    bool saw_wide_fanout = false;
    for (BC::FamilyId id = 0U; id < source_axis.family_count(); ++id) {
        const std::vector<BC::FamilyId> fanout =
            BC::map_partition_source_family_to_target_families(
                source,
                target_axis,
                target,
                id,
                2U
            );
        if (fanout.size() > 2U) {
            saw_wide_fanout = true;
            expect_throws(
                [&] { (void)BC::checked_partition_fanout2(fanout); },
                "current FamilyChain fanout2 guard should reject wider modulo fanout"
            );
            break;
        }
    }
    check(saw_wide_fanout, "test fixture should expose modulo fanout >2");
}

void test_real_tile_sum_possible_sums_keep_large_sentinel() {
    std::vector<uint8_t> legal_tiles;
    for (uint8_t tile = 0U; tile <= 8U; ++tile) {
        legal_tiles.push_back(tile);
    }
    legal_tiles.push_back(15U);
    const std::array<uint32_t, 16U> tile_values = BC::default_2048_tile_sum_values();
    check(tile_values[15U] == 32768U, "tile 15 should be true value in this test");
    const std::vector<BC::LayerSum> possible =
        BC::build_possible_8tile_sums(legal_tiles, tile_values);
    check(std::binary_search(possible.begin(), possible.end(), 32768U), "possible sums should include one sentinel");
    check(std::binary_search(possible.begin(), possible.end(), 2ULL * 32768U), "possible sums should include two sentinels");

    const BC::LayerSum source_layer_sum = 32768ULL + 64ULL;
    const BC::BCFamilyPartitionAnalysis exact =
        BC::analyze_family_partition_transition(
            source_layer_sum,
            2U,
            2U,
            possible,
            BC::BCFamilyPartitionPolicy::exact()
        );
    const BC::BCFamilyPartitionAnalysis mod29 =
        BC::analyze_family_partition_transition(
            source_layer_sum,
            2U,
            2U,
            possible,
            BC::BCFamilyPartitionPolicy::modulo(29U)
        );
    check(exact.source_family_count == exact.source_exact_coord_count, "exact family count should equal exact coords");
    check(mod29.source_family_count == 29U, "modulo family count should equal modulus");
    check(mod29.source_family_count <= exact.source_family_count, "modulo should control family count for this layer");
}

void test_invalid_inputs() {
    const std::vector<BC::LayerSum> possible = sorted_sums({0U, 2U, 4U});
    expect_throws(
        [&] { (void)BC::BCFamilyPartitionPolicy::modulo(0U); },
        "zero modulus should throw"
    );
    expect_throws(
        [&] {
            (void)BC::build_exact_family_coord_set_for_layer(5U, 2U, possible);
        },
        "non-divisible layer sum should throw"
    );
    expect_throws(
        [&] {
            (void)BC::analyze_family_partition_transition(
                8U,
                2U,
                3U,
                possible,
                BC::BCFamilyPartitionPolicy::exact()
            );
        },
        "target layer with empty exact coords should throw"
    );
}

} // namespace

int main() {
    try {
        test_exact_coord_set_weights();
        test_exact_partition_fanout_matches_t_delta_bound();
        test_modulo_partition_groups_coords_and_reports_fanout();
        test_modulo_partition_can_expose_fanout_over_two();
        test_modulo_axis_is_dense_partition_id_axis();
        test_runtime_partition_fanout_uses_exact_coord_groups();
        test_real_tile_sum_possible_sums_keep_large_sentinel();
        test_invalid_inputs();
    } catch (const std::exception &ex) {
        std::cerr << "bc_family_partition_analysis_test failed: " << ex.what() << '\n';
        return 1;
    }

    std::cout << "bc_family_partition_analysis_test passed\n";
    return 0;
}
