#include "BCFamilyStatsCsv.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace {

[[nodiscard]] std::ofstream open_bc_solve_stats_csv_append(
    const std::filesystem::path &path
) {
    std::ofstream out(path, std::ios::out | std::ios::app);
    if (!out) {
        throw std::runtime_error("failed to append BC solve stats CSV: " + path.string());
    }
    out << std::setprecision(12);
    return out;
}

[[nodiscard]] std::string bc_solve_stats_expected_header() {
    std::ostringstream expected_stream;
    BC::write_bc_solve_stats_header(expected_stream);
    std::string expected = expected_stream.str();
    if (!expected.empty() && expected.back() == '\n') {
        expected.pop_back();
    }
    return expected;
}

[[nodiscard]] bool bc_solve_stats_file_has_current_header(
    const std::filesystem::path &path
) {
    std::ifstream existing(path);
    if (!existing) {
        return false;
    }
    const std::string expected = bc_solve_stats_expected_header();
    std::string line;
    while (std::getline(existing, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line == expected) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::filesystem::path bc_solve_stats_current_schema_path(
    const std::filesystem::path &path
) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec ||
        std::filesystem::file_size(path, ec) == 0U || ec ||
        bc_solve_stats_file_has_current_header(path)) {
        return path;
    }
    const std::filesystem::path v2 = path.parent_path() /
        (path.stem().string() + ".v2" + path.extension().string());
    ec.clear();
    if (!std::filesystem::exists(v2, ec) || ec ||
        std::filesystem::file_size(v2, ec) == 0U || ec ||
        bc_solve_stats_file_has_current_header(v2)) {
        return v2;
    }
    throw std::runtime_error(
        "BC solve stats v2 CSV has an incompatible header: " + v2.string());
}

void flush_bc_solve_stats_csv(std::ofstream &out, const std::filesystem::path &path) {
    out.flush();
    if (!out) {
        throw std::runtime_error("failed to write BC solve stats CSV: " + path.string());
    }
}

[[nodiscard]] double bc_family_untracked_solve_seconds(
    const BC::BCFamilySolveRunLayerMetric &metric
) {
    if (!metric.has_family_stats) {
        return 0.0;
    }
    const BC::BCFamilySolveStats &fs = metric.family_stats;
    if (fs.spawn4_phase_wall_seconds == 0.0 && fs.spawn2_phase_wall_seconds == 0.0) {
        return 0.0;
    }
    const double accounted =
        fs.single.temp_prepare_seconds +
        fs.plan_seconds +
        fs.workspace_prepare_seconds +
        fs.temp_open_seconds +
        fs.spawn4_phase_wall_seconds +
        fs.output_streamer_open_seconds +
        fs.mark_empty_seconds +
        fs.spawn2_phase_wall_seconds +
        fs.output_finish_seconds +
        fs.temp_close_seconds +
        fs.temp_compress_seconds +
        fs.single.partial_cleanup_seconds;
    return std::max(0.0, metric.solve_call_seconds - accounted);
}

} // namespace

namespace BC {

double bc_stats_mrows_per_sec(uint64_t rows, double seconds) {
    return seconds > 0.0 ? static_cast<double>(rows) / seconds / 1.0e6 : 0.0;
}

double bc_stats_mib_per_sec(uint64_t bytes, double seconds) {
    return seconds > 0.0
        ? static_cast<double>(bytes) / seconds / 1048576.0
        : 0.0;
}

void accumulate_bc_solve_stats_total(
    BCSolveStatsTotals &total,
    const BCFamilySolveRunLayerMetric &m
) {
    const BCFamilySolveStats &fs = m.family_stats;
    const BCSingleChunkSolveStats &s = fs.single;
    total.current_rows += m.current_rows;
    total.live_rows += m.live_rows;
    total.zero_pruned_rows += m.zero_pruned_rows;
    total.archive_live_rows += m.archive_live_rows;
    total.threshold_pruned_rows += m.threshold_pruned_rows;
    total.position_bytes += m.position_bytes;
    total.success_bytes += m.success_bytes;
    total.output_position_write_bytes += m.output_position_write_bytes;
    total.output_success_write_bytes += m.output_success_write_bytes;
    total.temp_compressed_bytes += m.temp_compressed_bytes;
    total.final_compress_bucket_blocks += m.final_compress_bucket_blocks;
    total.final_compress_value_blocks += m.final_compress_value_blocks;
    total.final_compress_success_values += m.final_compress_success_values;
    total.final_compress_bucket_raw_bytes += m.final_compress_bucket_raw_bytes;
    total.final_compress_bucket_compressed_bytes += m.final_compress_bucket_compressed_bytes;
    total.final_compress_value_raw_bytes += m.final_compress_value_raw_bytes;
    total.final_compress_value_compressed_bytes += m.final_compress_value_compressed_bytes;
    total.final_compress_output_bytes += m.final_compress_output_bytes;
    total.route_available_memory_bytes =
        std::max(total.route_available_memory_bytes, m.route_available_memory_bytes);
    total.route_resident_required_bytes =
        std::max(total.route_resident_required_bytes, m.route_resident_required_bytes);
    total.route_single_required_bytes =
        std::max(total.route_single_required_bytes, m.route_single_required_bytes);
    total.route_required_bytes = std::max(total.route_required_bytes, m.route_required_bytes);
    total.open_seconds += m.open_seconds;
    total.open_current_position_seconds += m.open_current_position_seconds;
    total.open_future2_position_seconds += m.open_future2_position_seconds;
    total.open_future2_success_seconds += m.open_future2_success_seconds;
    total.open_future4_position_seconds += m.open_future4_position_seconds;
    total.open_future4_success_seconds += m.open_future4_success_seconds;
    total.descriptor_rows_seconds += m.descriptor_rows_seconds;
    total.partition_seconds += m.partition_seconds;
    total.writer_open_seconds += m.writer_open_seconds;
    total.solve_call_seconds += m.solve_call_seconds;
    if (m.has_family_stats) {
        total.spawn4_compute_seconds += fs.spawn4_cell_compute_seconds;
        total.spawn2_compute_seconds += fs.spawn2_cell_compute_seconds;
        total.compact_seconds += s.compact_seconds;
        total.current_position_read_seconds += s.current_position_read_seconds;
        total.future2_position_read_seconds += s.future2_position_read_seconds;
        total.future2_success_read_seconds += s.future2_success_read_seconds;
        total.future4_position_read_seconds += s.future4_position_read_seconds;
        total.future4_success_read_seconds += s.future4_success_read_seconds;
        total.future2_index_seconds += s.future2_index_seconds;
        total.future4_index_seconds += s.future4_index_seconds;
        total.temp_write_seconds += fs.temp_write_seconds;
        total.temp_read_prepare_seconds += fs.temp_read_prepare_seconds;
        total.temp_read_seconds += fs.temp_read_seconds;
        total.final_stage_write_seconds += fs.final_stage_write_seconds;
        total.final_stage_read_seconds += fs.final_stage_read_seconds;
        total.family_temp_prepare_seconds += s.temp_prepare_seconds;
        total.family_plan_seconds += fs.plan_seconds;
        total.family_workspace_prepare_seconds += fs.workspace_prepare_seconds;
        total.family_current_layout_seconds += fs.current_layout_seconds;
        total.family_future4_prepare_overhead_seconds += fs.future4_prepare_overhead_seconds;
        total.family_future2_prepare_overhead_seconds += fs.future2_prepare_overhead_seconds;
        total.family_future4_lookup_copy_seconds += fs.future4_lookup_copy_seconds;
        total.family_future2_lookup_copy_seconds += fs.future2_lookup_copy_seconds;
        total.family_spawn4_phase_wall_seconds += fs.spawn4_phase_wall_seconds;
        total.family_spawn2_phase_wall_seconds += fs.spawn2_phase_wall_seconds;
        total.family_future_release_seconds += s.future_release_seconds;
        total.family_final_dense_copy_seconds += fs.final_dense_copy_seconds;
        total.family_compact_value_copy_seconds += fs.compact_value_copy_seconds;
        total.family_pending_mark_seconds += fs.pending_mark_seconds;
        total.family_output_streamer_open_seconds += fs.output_streamer_open_seconds;
        total.family_output_finish_seconds += fs.output_finish_seconds;
        total.family_temp_open_seconds += fs.temp_open_seconds;
        total.family_temp_close_seconds += fs.temp_close_seconds;
        total.family_partial_cleanup_seconds += s.partial_cleanup_seconds;
        total.family_workspace_release_seconds +=
            fs.workspace_release_spawn4_partial_seconds +
            fs.workspace_release_spawn4_scratch_seconds +
            fs.workspace_release_spawn4_temp_values_seconds +
            fs.workspace_release_spawn2_dense_seconds +
            fs.workspace_release_spawn2_prefetch_seconds +
            fs.workspace_release_spawn2_temp_values_seconds;
        total.family_untracked_seconds += bc_family_untracked_solve_seconds(m);
    }
    total.temp_compress_seconds += m.temp_compress_seconds;
    total.position_write_seconds += m.position_write_seconds;
    total.success_write_seconds += m.success_write_seconds;
    total.writer_close_seconds += m.writer_close_seconds;
    total.post_resize_seconds += m.post_resize_seconds;
    total.archive_scan_seconds += m.archive_scan_seconds;
    total.archive_prune_write_seconds += m.archive_prune_write_seconds;
    total.final_compress_seconds += m.final_compress_seconds;
    total.final_compress_read_seconds += m.final_compress_read_seconds;
    total.final_compress_write_seconds += m.final_compress_write_seconds;
    total.final_compress_worker_seconds += m.final_compress_worker_seconds;
    total.total_seconds += m.total_seconds;
}

BCSolveStatsTotals bc_solve_stats_totals(const BCFamilySolveRunResult &result) {
    BCSolveStatsTotals total;
    for (const BCFamilySolveRunLayerMetric &m : result.layers) {
        accumulate_bc_solve_stats_total(total, m);
    }
    return total;
}

void write_bc_solve_stats_header(std::ostream &out) {
    out
        << "kind,ordinal,solve_route,layer_sum,current_rows,live_rows,zero_pruned_rows,"
        << "archive_live_rows,threshold_pruned_rows,position_bytes,success_bytes,"
        << "output_position_write_bytes,output_success_write_bytes,temp_compressed_bytes,"
        << "final_compress_bucket_blocks,final_compress_value_blocks,"
        << "final_compress_success_values,final_compress_bucket_raw_bytes,"
        << "final_compress_bucket_compressed_bytes,final_compress_value_raw_bytes,"
        << "final_compress_value_compressed_bytes,final_compress_output_bytes,"
        << "final_compress_raw_mib_per_sec,"
        << "route_available_memory_bytes,route_resident_required_bytes,"
        << "route_single_required_bytes,route_required_bytes,total_mrows_per_sec,"
        << "solve_call_mrows_per_sec,open_seconds,open_current_position_seconds,"
        << "open_future2_position_seconds,open_future2_success_seconds,"
        << "open_future4_position_seconds,open_future4_success_seconds,"
        << "descriptor_rows_seconds,partition_seconds,writer_open_seconds,solve_call_seconds,"
        << "spawn4_compute_seconds,spawn2_compute_seconds,compact_seconds,"
        << "current_position_read_seconds,future2_position_read_seconds,"
        << "future2_success_read_seconds,future4_position_read_seconds,"
        << "future4_success_read_seconds,future2_index_seconds,future4_index_seconds,"
        << "temp_write_seconds,temp_read_prepare_seconds,temp_read_seconds,temp_compress_seconds,"
        << "final_stage_write_seconds,final_stage_read_seconds,position_write_seconds,"
        << "success_write_seconds,writer_close_seconds,post_resize_seconds,"
        << "archive_scan_seconds,archive_prune_write_seconds,final_compress_seconds,"
        << "final_compress_read_seconds,final_compress_write_seconds,"
        << "final_compress_worker_seconds,family_temp_prepare_seconds,family_plan_seconds,"
        << "family_workspace_prepare_seconds,family_current_layout_seconds,"
        << "family_future4_prepare_overhead_seconds,family_future2_prepare_overhead_seconds,"
        << "family_future4_lookup_copy_seconds,family_future2_lookup_copy_seconds,"
        << "family_spawn4_phase_wall_seconds,family_spawn2_phase_wall_seconds,"
        << "family_future_release_seconds,family_final_dense_copy_seconds,"
        << "family_compact_value_copy_seconds,"
        << "family_pending_mark_seconds,family_output_streamer_open_seconds,"
        << "family_output_finish_seconds,family_temp_open_seconds,family_temp_close_seconds,"
        << "family_partial_cleanup_seconds,family_workspace_release_seconds,"
        << "family_untracked_seconds,total_seconds\n";
}

void write_bc_solve_stats_row(std::ostream &out, const BCFamilySolveRunLayerMetric &m) {
    const BCFamilySolveStats &fs = m.family_stats;
    const BCSingleChunkSolveStats &s = fs.single;
    out
        << m.kind << ','
        << m.ordinal << ','
        << m.solve_route << ','
        << m.layer_sum << ','
        << m.current_rows << ','
        << m.live_rows << ','
        << m.zero_pruned_rows << ','
        << m.archive_live_rows << ','
        << m.threshold_pruned_rows << ','
        << m.position_bytes << ','
        << m.success_bytes << ','
        << m.output_position_write_bytes << ','
        << m.output_success_write_bytes << ','
        << m.temp_compressed_bytes << ','
        << m.final_compress_bucket_blocks << ','
        << m.final_compress_value_blocks << ','
        << m.final_compress_success_values << ','
        << m.final_compress_bucket_raw_bytes << ','
        << m.final_compress_bucket_compressed_bytes << ','
        << m.final_compress_value_raw_bytes << ','
        << m.final_compress_value_compressed_bytes << ','
        << m.final_compress_output_bytes << ','
        << bc_stats_mib_per_sec(
            m.final_compress_bucket_raw_bytes + m.final_compress_value_raw_bytes,
            m.final_compress_seconds) << ','
        << m.route_available_memory_bytes << ','
        << m.route_resident_required_bytes << ','
        << m.route_single_required_bytes << ','
        << m.route_required_bytes << ','
        << bc_stats_mrows_per_sec(m.current_rows, m.total_seconds) << ','
        << bc_stats_mrows_per_sec(m.current_rows, m.solve_call_seconds) << ','
        << m.open_seconds << ','
        << m.open_current_position_seconds << ','
        << m.open_future2_position_seconds << ','
        << m.open_future2_success_seconds << ','
        << m.open_future4_position_seconds << ','
        << m.open_future4_success_seconds << ','
        << m.descriptor_rows_seconds << ','
        << m.partition_seconds << ','
        << m.writer_open_seconds << ','
        << m.solve_call_seconds << ','
        << (m.has_family_stats ? fs.spawn4_cell_compute_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.spawn2_cell_compute_seconds : 0.0) << ','
        << (m.has_family_stats ? s.compact_seconds : 0.0) << ','
        << (m.has_family_stats ? s.current_position_read_seconds : 0.0) << ','
        << (m.has_family_stats ? s.future2_position_read_seconds : 0.0) << ','
        << (m.has_family_stats ? s.future2_success_read_seconds : 0.0) << ','
        << (m.has_family_stats ? s.future4_position_read_seconds : 0.0) << ','
        << (m.has_family_stats ? s.future4_success_read_seconds : 0.0) << ','
        << (m.has_family_stats ? s.future2_index_seconds : 0.0) << ','
        << (m.has_family_stats ? s.future4_index_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.temp_write_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.temp_read_prepare_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.temp_read_seconds : 0.0) << ','
        << m.temp_compress_seconds << ','
        << (m.has_family_stats ? fs.final_stage_write_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.final_stage_read_seconds : 0.0) << ','
        << m.position_write_seconds << ','
        << m.success_write_seconds << ','
        << m.writer_close_seconds << ','
        << m.post_resize_seconds << ','
        << m.archive_scan_seconds << ','
        << m.archive_prune_write_seconds << ','
        << m.final_compress_seconds << ','
        << m.final_compress_read_seconds << ','
        << m.final_compress_write_seconds << ','
        << m.final_compress_worker_seconds << ','
        << (m.has_family_stats ? s.temp_prepare_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.plan_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.workspace_prepare_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.current_layout_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.future4_prepare_overhead_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.future2_prepare_overhead_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.future4_lookup_copy_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.future2_lookup_copy_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.spawn4_phase_wall_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.spawn2_phase_wall_seconds : 0.0) << ','
        << (m.has_family_stats ? s.future_release_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.final_dense_copy_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.compact_value_copy_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.pending_mark_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.output_streamer_open_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.output_finish_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.temp_open_seconds : 0.0) << ','
        << (m.has_family_stats ? fs.temp_close_seconds : 0.0) << ','
        << (m.has_family_stats ? s.partial_cleanup_seconds : 0.0) << ','
        << (m.has_family_stats
            ? fs.workspace_release_spawn4_partial_seconds +
                fs.workspace_release_spawn4_scratch_seconds +
                fs.workspace_release_spawn4_temp_values_seconds +
                fs.workspace_release_spawn2_dense_seconds +
                fs.workspace_release_spawn2_prefetch_seconds +
                fs.workspace_release_spawn2_temp_values_seconds
            : 0.0) << ','
        << bc_family_untracked_solve_seconds(m) << ','
        << m.total_seconds << '\n';
}

void write_bc_solve_stats_total_row(
    std::ostream &out,
    const BCFamilySolveRunResult &result
) {
    const BCSolveStatsTotals total = bc_solve_stats_totals(result);
    out
        << "total,0,,0,"
        << total.current_rows << ','
        << total.live_rows << ','
        << total.zero_pruned_rows << ','
        << total.archive_live_rows << ','
        << total.threshold_pruned_rows << ','
        << total.position_bytes << ','
        << total.success_bytes << ','
        << total.output_position_write_bytes << ','
        << total.output_success_write_bytes << ','
        << total.temp_compressed_bytes << ','
        << total.final_compress_bucket_blocks << ','
        << total.final_compress_value_blocks << ','
        << total.final_compress_success_values << ','
        << total.final_compress_bucket_raw_bytes << ','
        << total.final_compress_bucket_compressed_bytes << ','
        << total.final_compress_value_raw_bytes << ','
        << total.final_compress_value_compressed_bytes << ','
        << total.final_compress_output_bytes << ','
        << bc_stats_mib_per_sec(
            total.final_compress_bucket_raw_bytes + total.final_compress_value_raw_bytes,
            total.final_compress_seconds) << ','
        << total.route_available_memory_bytes << ','
        << total.route_resident_required_bytes << ','
        << total.route_single_required_bytes << ','
        << total.route_required_bytes << ','
        << bc_stats_mrows_per_sec(total.current_rows, total.total_seconds) << ','
        << bc_stats_mrows_per_sec(total.current_rows, total.solve_call_seconds) << ','
        << total.open_seconds << ','
        << total.open_current_position_seconds << ','
        << total.open_future2_position_seconds << ','
        << total.open_future2_success_seconds << ','
        << total.open_future4_position_seconds << ','
        << total.open_future4_success_seconds << ','
        << total.descriptor_rows_seconds << ','
        << total.partition_seconds << ','
        << total.writer_open_seconds << ','
        << total.solve_call_seconds << ','
        << total.spawn4_compute_seconds << ','
        << total.spawn2_compute_seconds << ','
        << total.compact_seconds << ','
        << total.current_position_read_seconds << ','
        << total.future2_position_read_seconds << ','
        << total.future2_success_read_seconds << ','
        << total.future4_position_read_seconds << ','
        << total.future4_success_read_seconds << ','
        << total.future2_index_seconds << ','
        << total.future4_index_seconds << ','
        << total.temp_write_seconds << ','
        << total.temp_read_prepare_seconds << ','
        << total.temp_read_seconds << ','
        << total.temp_compress_seconds << ','
        << total.final_stage_write_seconds << ','
        << total.final_stage_read_seconds << ','
        << total.position_write_seconds << ','
        << total.success_write_seconds << ','
        << total.writer_close_seconds << ','
        << total.post_resize_seconds << ','
        << total.archive_scan_seconds << ','
        << total.archive_prune_write_seconds << ','
        << total.final_compress_seconds << ','
        << total.final_compress_read_seconds << ','
        << total.final_compress_write_seconds << ','
        << total.final_compress_worker_seconds << ','
        << total.family_temp_prepare_seconds << ','
        << total.family_plan_seconds << ','
        << total.family_workspace_prepare_seconds << ','
        << total.family_current_layout_seconds << ','
        << total.family_future4_prepare_overhead_seconds << ','
        << total.family_future2_prepare_overhead_seconds << ','
        << total.family_future4_lookup_copy_seconds << ','
        << total.family_future2_lookup_copy_seconds << ','
        << total.family_spawn4_phase_wall_seconds << ','
        << total.family_spawn2_phase_wall_seconds << ','
        << total.family_future_release_seconds << ','
        << total.family_final_dense_copy_seconds << ','
        << total.family_compact_value_copy_seconds << ','
        << total.family_pending_mark_seconds << ','
        << total.family_output_streamer_open_seconds << ','
        << total.family_output_finish_seconds << ','
        << total.family_temp_open_seconds << ','
        << total.family_temp_close_seconds << ','
        << total.family_partial_cleanup_seconds << ','
        << total.family_workspace_release_seconds << ','
        << total.family_untracked_seconds << ','
        << total.total_seconds << '\n';
}

void ensure_bc_solve_stats_csv_file(const std::filesystem::path &path) {
    if (path.empty()) {
        return;
    }
    const std::filesystem::path current_path = bc_solve_stats_current_schema_path(path);
    if (!current_path.parent_path().empty()) {
        std::filesystem::create_directories(current_path.parent_path());
    }
    std::error_code ec;
    const bool exists = std::filesystem::exists(current_path, ec);
    if (ec) {
        throw std::runtime_error("failed to inspect BC solve stats CSV: " + ec.message());
    }
    uint64_t size = 0U;
    if (exists) {
        size = static_cast<uint64_t>(std::filesystem::file_size(current_path, ec));
        if (ec) {
            throw std::runtime_error("failed to inspect BC solve stats CSV: " + ec.message());
        }
    }
    if (size != 0U) {
        return;
    }
    std::ofstream out(current_path, std::ios::out | std::ios::app);
    if (!out) {
        throw std::runtime_error("failed to open BC solve stats CSV header: " + current_path.string());
    }
    out << std::setprecision(12);
    write_bc_solve_stats_header(out);
    flush_bc_solve_stats_csv(out, current_path);
}

void append_bc_solve_stats_csv_row(
    const std::filesystem::path &path,
    const BCFamilySolveRunLayerMetric &metric
) {
    if (path.empty()) {
        return;
    }
    ensure_bc_solve_stats_csv_file(path);
    const std::filesystem::path current_path = bc_solve_stats_current_schema_path(path);
    std::ofstream out = open_bc_solve_stats_csv_append(current_path);
    write_bc_solve_stats_row(out, metric);
    flush_bc_solve_stats_csv(out, current_path);
}

void append_bc_solve_stats_csv_total_row(
    const std::filesystem::path &path,
    const BCFamilySolveRunResult &result
) {
    if (path.empty()) {
        return;
    }
    ensure_bc_solve_stats_csv_file(path);
    const std::filesystem::path current_path = bc_solve_stats_current_schema_path(path);
    std::ofstream out = open_bc_solve_stats_csv_append(current_path);
    write_bc_solve_stats_total_row(out, result);
    flush_bc_solve_stats_csv(out, current_path);
}

void write_bc_solve_summary(
    std::ostream &out,
    const BCFamilyGenerationRunResult &generation,
    const BCFamilySolveRunResult &solve
) {
    const BCSolveStatsTotals total = bc_solve_stats_totals(solve);
    out << std::setprecision(12)
        << "generation_layers,solve_layers,solve_completed,min_ordinal,max_ordinal,"
        << "current_rows,live_rows,zero_pruned_rows,archive_live_rows,"
        << "threshold_pruned_rows,position_bytes,success_bytes,temp_compressed_bytes,"
        << "solve_call_seconds,total_seconds,total_mrows_per_sec,solve_call_mrows_per_sec,"
        << "temp_compress_seconds,final_compress_seconds,final_compress_read_seconds,"
        << "final_compress_write_seconds,final_compress_worker_seconds,"
        << "final_compress_bucket_blocks,final_compress_value_blocks,"
        << "final_compress_bucket_raw_bytes,final_compress_bucket_compressed_bytes,"
        << "final_compress_value_raw_bytes,final_compress_value_compressed_bytes,"
        << "final_compress_output_bytes,final_compress_raw_mib_per_sec\n"
        << generation.layers.size() << ','
        << solve.layers.size() << ','
        << (solve.completed ? 1 : 0) << ','
        << solve.min_ordinal << ','
        << solve.max_ordinal << ','
        << total.current_rows << ','
        << total.live_rows << ','
        << total.zero_pruned_rows << ','
        << total.archive_live_rows << ','
        << total.threshold_pruned_rows << ','
        << total.position_bytes << ','
        << total.success_bytes << ','
        << total.temp_compressed_bytes << ','
        << total.solve_call_seconds << ','
        << total.total_seconds << ','
        << bc_stats_mrows_per_sec(total.current_rows, total.total_seconds) << ','
        << bc_stats_mrows_per_sec(total.current_rows, total.solve_call_seconds) << ','
        << total.temp_compress_seconds << ','
        << total.final_compress_seconds << ','
        << total.final_compress_read_seconds << ','
        << total.final_compress_write_seconds << ','
        << total.final_compress_worker_seconds << ','
        << total.final_compress_bucket_blocks << ','
        << total.final_compress_value_blocks << ','
        << total.final_compress_bucket_raw_bytes << ','
        << total.final_compress_bucket_compressed_bytes << ','
        << total.final_compress_value_raw_bytes << ','
        << total.final_compress_value_compressed_bytes << ','
        << total.final_compress_output_bytes << ','
        << bc_stats_mib_per_sec(
            total.final_compress_bucket_raw_bytes + total.final_compress_value_raw_bytes,
            total.final_compress_seconds) << '\n';
}

} // namespace BC
