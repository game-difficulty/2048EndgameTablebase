#pragma once

#include "BCFamilyGenerationRunner.h"
#include "BCFamilySolveRunner.h"

#include <cstdint>
#include <iosfwd>

namespace BC {

struct BCSolveStatsTotals {
    uint64_t current_rows = 0U;
    uint64_t live_rows = 0U;
    uint64_t zero_pruned_rows = 0U;
    uint64_t archive_live_rows = 0U;
    uint64_t threshold_pruned_rows = 0U;
    uint64_t position_bytes = 0U;
    uint64_t success_bytes = 0U;
    uint64_t output_position_write_bytes = 0U;
    uint64_t output_success_write_bytes = 0U;
    uint64_t temp_compressed_bytes = 0U;
    uint64_t final_compress_bucket_blocks = 0U;
    uint64_t final_compress_value_blocks = 0U;
    uint64_t final_compress_success_values = 0U;
    uint64_t final_compress_bucket_raw_bytes = 0U;
    uint64_t final_compress_bucket_compressed_bytes = 0U;
    uint64_t final_compress_value_raw_bytes = 0U;
    uint64_t final_compress_value_compressed_bytes = 0U;
    uint64_t final_compress_output_bytes = 0U;
    uint64_t route_available_memory_bytes = 0U;
    uint64_t route_resident_required_bytes = 0U;
    uint64_t route_single_required_bytes = 0U;
    uint64_t route_required_bytes = 0U;
    double open_seconds = 0.0;
    double open_current_position_seconds = 0.0;
    double open_future2_position_seconds = 0.0;
    double open_future2_success_seconds = 0.0;
    double open_future4_position_seconds = 0.0;
    double open_future4_success_seconds = 0.0;
    double descriptor_rows_seconds = 0.0;
    double partition_seconds = 0.0;
    double writer_open_seconds = 0.0;
    double solve_call_seconds = 0.0;
    double spawn4_compute_seconds = 0.0;
    double spawn2_compute_seconds = 0.0;
    double compact_seconds = 0.0;
    double temp_write_seconds = 0.0;
    double temp_read_seconds = 0.0;
    double temp_compress_seconds = 0.0;
    double final_stage_write_seconds = 0.0;
    double final_stage_read_seconds = 0.0;
    double position_write_seconds = 0.0;
    double success_write_seconds = 0.0;
    double writer_close_seconds = 0.0;
    double post_resize_seconds = 0.0;
    double archive_scan_seconds = 0.0;
    double archive_prune_write_seconds = 0.0;
    double final_compress_seconds = 0.0;
    double final_compress_read_seconds = 0.0;
    double final_compress_write_seconds = 0.0;
    double final_compress_worker_seconds = 0.0;
    double total_seconds = 0.0;
};

[[nodiscard]] double bc_stats_mrows_per_sec(uint64_t rows, double seconds);
[[nodiscard]] double bc_stats_mib_per_sec(uint64_t bytes, double seconds);

void accumulate_bc_solve_stats_total(
    BCSolveStatsTotals &total,
    const BCFamilySolveRunLayerMetric &metric
);

[[nodiscard]] BCSolveStatsTotals bc_solve_stats_totals(
    const BCFamilySolveRunResult &result
);

void write_bc_solve_stats_header(std::ostream &out);
void write_bc_solve_stats_row(std::ostream &out, const BCFamilySolveRunLayerMetric &metric);
void write_bc_solve_stats_total_row(std::ostream &out, const BCFamilySolveRunResult &result);

void write_bc_solve_summary(
    std::ostream &out,
    const BCFamilyGenerationRunResult &generation,
    const BCFamilySolveRunResult &solve
);

} // namespace BC
