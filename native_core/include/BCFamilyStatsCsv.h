#pragma once

#include "BCFamilyGenerationRunner.h"
#include "BCFamilySolveRunner.h"

#include <cstdint>
#include <filesystem>
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
    double current_position_read_seconds = 0.0;
    double future2_position_read_seconds = 0.0;
    double future2_success_read_seconds = 0.0;
    double future4_position_read_seconds = 0.0;
    double future4_success_read_seconds = 0.0;
    double future2_index_seconds = 0.0;
    double future4_index_seconds = 0.0;
    double temp_write_seconds = 0.0;
    double temp_read_prepare_seconds = 0.0;
    double temp_read_seconds = 0.0;
    double temp_compress_seconds = 0.0;
    double final_stage_write_seconds = 0.0;
    double final_stage_read_seconds = 0.0;
    double family_temp_prepare_seconds = 0.0;
    double family_plan_seconds = 0.0;
    double family_workspace_prepare_seconds = 0.0;
    double family_current_layout_seconds = 0.0;
    double family_future4_prepare_overhead_seconds = 0.0;
    double family_future2_prepare_overhead_seconds = 0.0;
    double family_future4_lookup_copy_seconds = 0.0;
    double family_future2_lookup_copy_seconds = 0.0;
    double family_spawn4_phase_wall_seconds = 0.0;
    double family_spawn2_phase_wall_seconds = 0.0;
    double family_future_release_seconds = 0.0;
    double family_final_dense_copy_seconds = 0.0;
    double family_compact_value_copy_seconds = 0.0;
    double family_pending_mark_seconds = 0.0;
    double family_output_streamer_open_seconds = 0.0;
    double family_output_finish_seconds = 0.0;
    double family_temp_open_seconds = 0.0;
    double family_temp_close_seconds = 0.0;
    double family_partial_cleanup_seconds = 0.0;
    double family_workspace_release_seconds = 0.0;
    double family_untracked_seconds = 0.0;
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

void ensure_bc_solve_stats_csv_file(const std::filesystem::path &path);
void append_bc_solve_stats_csv_row(
    const std::filesystem::path &path,
    const BCFamilySolveRunLayerMetric &metric
);
void append_bc_solve_stats_csv_total_row(
    const std::filesystem::path &path,
    const BCFamilySolveRunResult &result
);

void write_bc_solve_summary(
    std::ostream &out,
    const BCFamilyGenerationRunResult &generation,
    const BCFamilySolveRunResult &solve
);

} // namespace BC
