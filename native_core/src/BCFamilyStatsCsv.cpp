#include "BCFamilyStatsCsv.h"

#include <algorithm>
#include <iomanip>
#include <ostream>

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
        total.temp_write_seconds += fs.temp_write_seconds;
        total.temp_read_seconds += fs.temp_read_seconds;
        total.final_stage_write_seconds += fs.final_stage_write_seconds;
        total.final_stage_read_seconds += fs.final_stage_read_seconds;
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
        << "temp_write_seconds,temp_read_seconds,temp_compress_seconds,"
        << "final_stage_write_seconds,final_stage_read_seconds,position_write_seconds,"
        << "success_write_seconds,writer_close_seconds,post_resize_seconds,"
        << "archive_scan_seconds,archive_prune_write_seconds,final_compress_seconds,"
        << "final_compress_read_seconds,final_compress_write_seconds,"
        << "final_compress_worker_seconds,total_seconds\n";
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
        << (m.has_family_stats ? fs.temp_write_seconds : 0.0) << ','
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
        << total.temp_write_seconds << ','
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
        << total.total_seconds << '\n';
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
