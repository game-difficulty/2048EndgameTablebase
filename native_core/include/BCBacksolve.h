#pragma once

#include "BCBoardOps.h"
#include "BCPositionFile.h"
#include "BCSuccessIO.h"
#include "FormationRuntime.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace BC {

struct BCBacksolveOptions {
    int num_threads = 0;
    uint32_t canonical_batch_size = 8192U;
    int canonical_symm_mode = static_cast<int>(SymmMode::Full);
    double spawn_rate4 = 0.1;
    int success_target_rank = 0;
    const std::vector<uint8_t> *success_shifts = nullptr;
    const std::vector<uint64_t> *pattern_masks = nullptr;
    BCSuccessDTypeMode dtype = BCSuccessDTypeMode::UInt32;
};

struct BCBacksolveStats {
    uint64_t current_rows = 0U;
    uint64_t queries2 = 0U;
    uint64_t queries4 = 0U;
    uint64_t found2 = 0U;
    uint64_t found4 = 0U;
    uint64_t terminal_success_rows = 0U;
    double future_index_seconds = 0.0;
    double recalc_seconds = 0.0;
    double write_seconds = 0.0;

    [[nodiscard]] double recalc_mbps() const {
        return recalc_seconds > 0.0
            ? static_cast<double>(current_rows) / recalc_seconds / 1.0e6
            : 0.0;
    }
};

struct BCBacksolveResult {
    std::vector<uint8_t> success_bytes;
    BCBacksolveStats stats;
};

class BCFutureValueLayerView {
public:
    BCFutureValueLayerView() = default;

    BCFutureValueLayerView(
        const BCLut &lut,
        const BCPositionLayerReader &position,
        const BCSuccessLayerReader &success
    ) {
        open(lut, position, success);
    }

    void open(
        const BCLut &lut,
        const BCPositionLayerReader &position,
        const BCSuccessLayerReader &success
    );

    [[nodiscard]] bool lookup(uint64_t canonical_board, uint32_t &value_out) const;

private:
    struct DirectEntry {
        uint64_t key = 0U;
        uint32_t rank_payload_offset = 0U;
        uint32_t success_row_offset = 0U;
        BucketBitmapLen bitmap_len = 0U;
        bool occupied = false;
    };

    struct CellIndex {
        BCRankPayloadView rank_payload = {};
        std::vector<uint32_t> values;
        std::vector<DirectEntry> entries;
        uint32_t mask = 0U;
    };

    [[nodiscard]] const DirectEntry *find_entry(const CellIndex &cell, uint64_t key) const;
    [[nodiscard]] bool lookup_encoded(const BCBoardEncodedPosition &encoded, uint32_t &value_out) const;

    const BCLut *lut_ = nullptr;
    const BCPositionLayerReader *position_ = nullptr;
    std::vector<CellIndex> cells_;
};

struct BCBacksolveBatchStats {
    uint64_t rows = 0U;
    uint64_t queries2 = 0U;
    uint64_t queries4 = 0U;
    uint64_t found2 = 0U;
    uint64_t found4 = 0U;
    uint64_t terminal_success_rows = 0U;
};

BCBacksolveBatchStats backsolve_board_batch_uint32(
    const uint64_t *boards,
    const uint32_t *local_rows,
    uint32_t board_count,
    const BCFutureValueLayerView &future2,
    const BCFutureValueLayerView &future4,
    const BCBacksolveOptions &options,
    uint32_t *output_values
);

BCBacksolveResult backsolve_resident_layer(
    const BCLut &lut,
    const BCPositionLayerReader &current,
    const BCPositionLayerReader &future2_position,
    const BCSuccessLayerReader &future2_success,
    const BCPositionLayerReader &future4_position,
    const BCSuccessLayerReader &future4_success,
    const BCBacksolveOptions &options = {}
);

BCBacksolveStats backsolve_resident_layer_to_file(
    const std::filesystem::path &path,
    const BCLut &lut,
    const BCPositionLayerReader &current,
    const BCPositionLayerReader &future2_position,
    const BCSuccessLayerReader &future2_success,
    const BCPositionLayerReader &future4_position,
    const BCSuccessLayerReader &future4_success,
    const BCBacksolveOptions &options = {}
);

} // namespace BC
