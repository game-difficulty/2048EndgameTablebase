#pragma once

#include "BCBoardOps.h"
#include "BCFutureSuccessLookup.h"
#include "BCPositionFile.h"
#include "BCSuccessIO.h"
#include "FormationRuntime.h"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
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

class BCFutureValueLayerView : public BCFutureSuccessLookupView<uint32_t> {
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
    ) {
        if (success.row_width() != 1U) {
            throw std::invalid_argument("BC future success reader row_width must be 1");
        }
        if (success.dtype_mode() != BCSuccessDTypeMode::UInt32) {
            throw std::invalid_argument("BC future success reader dtype must be UInt32");
        }
        BCFutureSuccessLookupView<uint32_t>::open(lut, position, success);
    }
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
