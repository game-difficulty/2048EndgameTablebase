#pragma once

#include "EXADLayer.h"

#include <array>
#include <atomic>
#include <memory>
#include <vector>

namespace EXAD {

struct ReserveFactors {
    double bucket = 2.5;
    double small = 4.0;
    double large = 4.0;
    std::array<uint64_t, bucket_slot_count()> bucket_floor{};
    std::array<uint64_t, bucket_slot_count()> small_floor{};
    std::array<uint64_t, bucket_slot_count()> large_floor{};
};

struct CarryState {
    uint32_t original_board_sum = 0;
    uint32_t threshold_bits = kDefaultThresholdBits;
    uint32_t hash_capacity = 0;
    uint64_t reserved_small_bytes = 0;
    uint64_t reserved_large_words = 0;
    std::unique_ptr<std::atomic<uint64_t>[]> key_array;
    std::unique_ptr<std::atomic<uint32_t>[]> offset_array;
    std::unique_ptr<std::atomic<uint8_t>[]> small_arena;
    std::unique_ptr<std::atomic<uint64_t>[]> large_arena;
    std::atomic<uint32_t> small_cursor_bytes{0};
    std::atomic<uint32_t> large_cursor_words{0};
    std::atomic<uint32_t> bucket_count{0};
    std::atomic<bool> overflowed{false};
    std::atomic<uint32_t> overflow_reason{0};

    CarryState() = default;
    CarryState(const CarryState &) = delete;
    CarryState &operator=(const CarryState &) = delete;
    CarryState(CarryState &&other) noexcept;
    CarryState &operator=(CarryState &&other) noexcept;

    [[nodiscard]] bool empty() const {
        return !key_array;
    }
};

struct CarryLayer {
    uint32_t original_board_sum = 0;
    uint32_t threshold_bits = kDefaultThresholdBits;
    std::array<CarryState, bucket_slot_count()> states{};

    [[nodiscard]] bool empty() const {
        for (const CarryState &state : states) {
            if (!state.empty()) {
                return false;
            }
        }
        return true;
    }
};

struct GenerateStats {
    uint64_t input_live = 0;
    uint64_t arr1_live = 0;
    uint64_t arr2_live = 0;
    uint64_t derive_candidate_count = 0;
    uint64_t derived_output_count = 0;
    uint32_t retry_count = 0;
    double prepare_seconds = 0.0;
    double prepare_estimate_seconds = 0.0;
    double prepare_arr1_seconds = 0.0;
    double prepare_arr2_seconds = 0.0;
    double hashmap_seconds = 0.0;
    double worklist_seconds = 0.0;
    double loop_seconds = 0.0;
    double count_finalize_seconds = 0.0;
    double generate_seconds = 0.0;
    double insert_seconds = 0.0;
    double finalize_seconds = 0.0;
};

struct GeneratePairResult {
    CarryLayer arr1;
    CarryLayer arr2;
    GenerateStats stats;
};

struct DeriveHashState {
    std::vector<uint64_t> next1;
    std::vector<uint64_t> next2;
};

Layer build_layer_from_boards(
    const std::vector<uint64_t> &masked_boards,
    uint32_t original_board_sum,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    const Luts &luts,
    int num_threads,
    ReserveFactors factors = {}
);

CarryLayer carry_from_layer(
    const Layer &layer,
    const Luts &luts,
    int num_threads,
    ReserveFactors factors = {}
);

Layer finalize_carry_layer(
    const CarryLayer &carry,
    const Luts &luts,
    int num_threads
);

GeneratePairResult generate_two_layers_carry(
    const Layer &current,
    const AdvancedPatternSpec &spec,
    const RunOptions &options,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    const Luts &luts,
    int num_threads,
    CarryLayer arr1_seed,
    ReserveFactors factors = {},
    DeriveHashState *derive_hash_state = nullptr
);

Layer validate_layer_streaming(
    Layer layer,
    uint32_t original_board_sum,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    const Luts &luts,
    int num_threads
);

CarryLayer validate_carry_streaming(
    const CarryLayer &carry,
    uint32_t original_board_sum,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    const Luts &luts,
    int num_threads
);

uint64_t carry_bucket_count(const CarryLayer &carry);

} // namespace EXAD
