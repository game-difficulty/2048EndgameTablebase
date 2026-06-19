#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "BookSolver.h"
#include "BCCompressedResult.h"
#include "BCFamilyGenerationRunner.h"
#include "BCFamilySolveRunner.h"
#include "BCLut.h"
#include "CommonMover.h"
#include "EXADCompressedResult.h"
#include "EXCompressedResult.h"
#include "FormationRuntime.h"
#include "NativeDiagnostics.h"
#include "ReaderRuntime.h"
#include "SymmetryUtils.h"
#include "TrieCompression.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using U64Array = nb::ndarray<const uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

std::vector<uint64_t> to_u64_vector(const U64Array &array) {
    std::vector<uint64_t> result(static_cast<size_t>(array.shape(0)));
    if (!result.empty()) {
        std::memcpy(result.data(), array.data(), result.size() * sizeof(uint64_t));
    }
    return result;
}

std::vector<uint8_t> bc_free_legal_tiles(uint32_t target_rank) {
    if (target_rank >= 15U) {
        throw std::invalid_argument("BC target_rank must be < 15");
    }
    std::vector<uint8_t> legal_tiles;
    legal_tiles.reserve(static_cast<size_t>(target_rank) + 2U);
    for (uint32_t tile = 0U; tile <= target_rank; ++tile) {
        legal_tiles.push_back(static_cast<uint8_t>(tile));
    }
    legal_tiles.push_back(15U);
    return legal_tiles;
}

template <typename T>
T dict_get_or(const nb::dict &options, const char *key, T fallback) {
    nb::str py_key(key);
    if (!options.contains(py_key)) {
        return fallback;
    }
    nb::handle value = options[py_key];
    if (value.is_none()) {
        return fallback;
    }
    return nb::cast<T>(value);
}

BC::BCSuccessDTypeMode parse_bc_success_dtype_binding(const std::string &value) {
    if (value == "uint32") return BC::BCSuccessDTypeMode::UInt32;
    if (value == "uint64") return BC::BCSuccessDTypeMode::UInt64;
    if (value == "float32") return BC::BCSuccessDTypeMode::Float32;
    if (value == "float64") return BC::BCSuccessDTypeMode::Float64;
    if (value == "one-minus-float32" || value == "1-float32") {
        return BC::BCSuccessDTypeMode::OneMinusFloat32;
    }
    if (value == "one-minus-float64" || value == "1-float64") {
        return BC::BCSuccessDTypeMode::OneMinusFloat64;
    }
    throw std::invalid_argument("unsupported BC success dtype: " + value);
}

BC::BCFamilyGenerationRoute parse_bc_generation_route_binding(const std::string &value) {
    return BC::bc_parse_family_route(value);
}

BC::BCSolveRoute parse_bc_solve_route_binding(const std::string &value) {
    return BC::bc_parse_solve_route(value);
}

BC::BCFamilyGenerationRunOptions bc_generation_options_from_dict(const nb::dict &options) {
    BC::BCFamilyGenerationRunOptions run;
    run.pattern = dict_get_or<std::string>(options, "pattern", run.pattern);
    run.target_rank = dict_get_or<uint32_t>(options, "target_rank", run.target_rank);
    run.extra_steps = dict_get_or<uint32_t>(options, "extra_steps", run.extra_steps);
    run.output_dir = dict_get_or<std::string>(options, "generated_dir", run.output_dir.string());
    run.stats_csv = dict_get_or<std::string>(options, "generation_stats_csv", run.stats_csv.string());
    run.num_threads = dict_get_or<int>(options, "threads", run.num_threads);
    run.family_modulus = dict_get_or<uint32_t>(options, "family_modulus", run.family_modulus);
    run.family_route = parse_bc_generation_route_binding(
        dict_get_or<std::string>(options, "family_route", "auto"));
    run.direct_queue_depth = dict_get_or<uint32_t>(
        options,
        "direct_queue_depth",
        run.direct_queue_depth);
    run.batch_size = dict_get_or<uint32_t>(options, "batch_size", run.batch_size);
    run.pending_buffer = dict_get_or<uint32_t>(options, "pending_buffer", run.pending_buffer);
    run.family_work_schedule_chunk = dict_get_or<uint32_t>(
        options,
        "family_work_schedule_chunk",
        run.family_work_schedule_chunk);
    run.family_source_words_per_item = dict_get_or<uint32_t>(
        options,
        "family_source_words_per_item",
        run.family_source_words_per_item);
    run.verify_layer_rows = dict_get_or<bool>(options, "verify_layer_rows", false);
    run.output_inspect = dict_get_or<bool>(options, "output_inspect", false);
    const bool direct_io = dict_get_or<bool>(options, "direct_io", true);
    if (!direct_io) {
        run.family_blob = "buffered";
        run.family_position_io = "buffered";
        run.family_source_io = "buffered";
    }
    run.family_blob = dict_get_or<std::string>(options, "family_blob", run.family_blob);
    run.family_position_io = dict_get_or<std::string>(
        options,
        "family_position_io",
        run.family_position_io);
    run.family_source_io = dict_get_or<std::string>(
        options,
        "family_source_io",
        run.family_source_io);
    return run;
}

BC::BCFamilySolveRunOptions bc_solve_options_from_dict(const nb::dict &options) {
    BC::BCFamilySolveRunOptions run;
    run.generated_position_dir =
        dict_get_or<std::string>(options, "generated_dir", run.generated_position_dir.string());
    run.solved_output_dir =
        dict_get_or<std::string>(options, "solved_dir", run.solved_output_dir.string());
    run.archive_output_dir =
        dict_get_or<std::string>(options, "archive_dir", run.archive_output_dir.string());
    run.prefix = dict_get_or<std::string>(options, "prefix", run.prefix);
    run.target_rank = dict_get_or<uint32_t>(options, "target_rank", run.target_rank);
    run.success_target_rank = dict_get_or<int>(
        options,
        "success_target_rank",
        static_cast<int>(run.target_rank));
    run.spawn_rate4 = dict_get_or<double>(options, "spawn_rate4", run.spawn_rate4);
    run.num_threads = dict_get_or<int>(options, "threads", run.num_threads);
    run.family_modulus = dict_get_or<uint32_t>(options, "family_modulus", run.family_modulus);
    run.solve_route = parse_bc_solve_route_binding(
        dict_get_or<std::string>(options, "solve_route", "auto"));
    run.direct_queue_depth = dict_get_or<uint32_t>(
        options,
        "direct_queue_depth",
        run.direct_queue_depth);
    run.direct_io = dict_get_or<bool>(options, "direct_io", run.direct_io);
    run.keep_direct_padding = dict_get_or<bool>(options, "keep_direct_padding", false);
    run.success_dtype = parse_bc_success_dtype_binding(
        dict_get_or<std::string>(options, "success_dtype", "uint32"));
    run.compress = dict_get_or<bool>(options, "compress", true);
    run.compress_temp_files = dict_get_or<bool>(options, "compress_temp_files", false);
    run.deletion_threshold = dict_get_or<double>(options, "deletion_threshold", 0.0);
    run.relative_deletion_threshold = dict_get_or<double>(
        options,
        "relative_deletion_threshold",
        0.0);
    run.deletion_threshold_signal_path = dict_get_or<std::string>(
        options,
        "deletion_threshold_signal_path",
        "");
    run.resume_from_checkpoint = dict_get_or<bool>(options, "resume", true);
    run.force_restart = dict_get_or<bool>(options, "restart", false);
    return run;
}

[[nodiscard]] double bc_stats_mrows_per_sec(uint64_t rows, double seconds) {
    return seconds > 0.0 ? static_cast<double>(rows) / seconds / 1.0e6 : 0.0;
}

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
    double total_seconds = 0.0;
};

void accumulate_bc_solve_stats_total(
    BCSolveStatsTotals &total,
    const BC::BCFamilySolveRunLayerMetric &m
) {
    const BC::BCFamilySolveStats &fs = m.family_stats;
    const BC::BCSingleChunkSolveStats &s = fs.single;
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
    total.total_seconds += m.total_seconds;
}

[[nodiscard]] BCSolveStatsTotals bc_solve_stats_totals(
    const BC::BCFamilySolveRunResult &result
) {
    BCSolveStatsTotals total;
    for (const BC::BCFamilySolveRunLayerMetric &m : result.layers) {
        accumulate_bc_solve_stats_total(total, m);
    }
    return total;
}

void write_bc_solve_stats_header(std::ostream &out) {
    out
        << "kind,ordinal,solve_route,layer_sum,current_rows,live_rows,zero_pruned_rows,"
        << "archive_live_rows,threshold_pruned_rows,position_bytes,success_bytes,"
        << "output_position_write_bytes,output_success_write_bytes,temp_compressed_bytes,"
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
        << "total_seconds\n";
}

void write_bc_solve_stats_row(std::ostream &out, const BC::BCFamilySolveRunLayerMetric &m) {
    const BC::BCFamilySolveStats &fs = m.family_stats;
    const BC::BCSingleChunkSolveStats &s = fs.single;
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
        << m.total_seconds << '\n';
}

void write_bc_solve_stats_total_row(
    std::ostream &out,
    const BC::BCFamilySolveRunResult &result
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
        << total.total_seconds << '\n';
}

void write_bc_solve_summary(
    std::ostream &out,
    const BC::BCFamilyGenerationRunResult &generation,
    const BC::BCFamilySolveRunResult &solve
) {
    const BCSolveStatsTotals total = bc_solve_stats_totals(solve);
    out << std::setprecision(12)
        << "generation_layers,solve_layers,solve_completed,min_ordinal,max_ordinal,"
        << "current_rows,live_rows,zero_pruned_rows,archive_live_rows,"
        << "threshold_pruned_rows,position_bytes,success_bytes,temp_compressed_bytes,"
        << "solve_call_seconds,total_seconds,total_mrows_per_sec,solve_call_mrows_per_sec,"
        << "temp_compress_seconds,final_compress_seconds\n"
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
        << total.final_compress_seconds << '\n';
}

nb::dict bc_family_build_summary_to_python(
    const BC::BCFamilyGenerationRunResult &generation,
    const BC::BCFamilySolveRunResult &solve
) {
    nb::dict result;
    result["generation_layers"] = generation.layers.size();
    result["solve_layers"] = solve.layers.size();
    result["generation_completed"] = generation.completed;
    result["solve_completed"] = solve.completed;
    result["min_ordinal"] = solve.min_ordinal;
    result["max_ordinal"] = solve.max_ordinal;
    return result;
}

nb::tuple reader_result_to_python(const ReaderMoveResult &result) {
    nb::dict entries;
    for (const auto &entry : result.entries) {
        if (entry.kind == ReaderValueKind::Numeric) {
            entries[nb::str(entry.key.c_str())] = entry.number;
        } else if (entry.kind == ReaderValueKind::String) {
            entries[nb::str(entry.key.c_str())] = nb::str(entry.text.c_str());
        } else {
            entries[nb::str(entry.key.c_str())] = nb::none();
        }
    }
    return nb::make_tuple(entries, result.success_rate_dtype);
}

nb::dict ex_compress_stats_to_python(const EXCompressedResult::CompressStats &stats) {
    nb::dict result;
    result["original_bytes"] = stats.original_bytes;
    result["compressed_bytes"] = stats.compressed_bytes;
    result["bucket_block_count"] = stats.bucket_block_count;
    result["success_block_count"] = stats.success_block_count;
    result["bucket_raw_bytes"] = stats.bucket_raw_bytes;
    result["bucket_compressed_bytes"] = stats.bucket_compressed_bytes;
    result["success_raw_bytes"] = stats.success_raw_bytes;
    result["success_compressed_bytes"] = stats.success_compressed_bytes;
    result["compress_seconds"] = stats.compress_seconds;
    result["ratio"] = stats.ratio();
    result["save_ratio"] = stats.save_ratio();
    return result;
}

nb::dict ex_cold_lookup_to_python(const EXCompressedResult::ColdLookupResult &lookup) {
    nb::dict result;
    result["found"] = lookup.found;
    result["global_dense_index"] = lookup.global_dense_index;
    result["raw_value_bits"] = lookup.raw_value_bits;
    result["numeric_value"] = lookup.numeric_value;
    result["success_kind"] = static_cast<uint32_t>(lookup.success_kind);
    result["bucket_block_raw_bytes"] = lookup.bucket_block_raw_bytes;
    result["success_block_raw_bytes"] = lookup.success_block_raw_bytes;
    result["bucket_block_compressed_bytes"] = lookup.bucket_block_compressed_bytes;
    result["success_block_compressed_bytes"] = lookup.success_block_compressed_bytes;
    return result;
}

nb::dict exad_compress_stats_to_python(const EXADCompressedResult::CompressStats &stats) {
    nb::dict result;
    result["original_bytes"] = stats.original_bytes;
    result["compressed_bytes"] = stats.compressed_bytes;
    result["live_board_count"] = stats.live_board_count;
    result["success_value_count"] = stats.success_value_count;
    result["bucket_block_count"] = stats.bucket_block_count;
    result["value_block_count"] = stats.value_block_count;
    result["bucket_raw_bytes"] = stats.bucket_raw_bytes;
    result["bucket_compressed_bytes"] = stats.bucket_compressed_bytes;
    result["value_raw_bytes"] = stats.value_raw_bytes;
    result["value_compressed_bytes"] = stats.value_compressed_bytes;
    result["compression_seconds"] = stats.compression_seconds;
    const double ratio = stats.original_bytes == 0
        ? 0.0
        : static_cast<double>(stats.compressed_bytes) / static_cast<double>(stats.original_bytes);
    result["ratio"] = ratio;
    result["save_ratio"] = 1.0 - ratio;
    return result;
}

nb::dict exad_cold_lookup_to_python(const EXADCompressedResult::ColdLookupResult &lookup) {
    nb::dict result;
    result["found"] = lookup.found;
    result["value_index"] = lookup.value_index;
    result["local_row"] = lookup.local_row;
    result["raw_value_bits"] = lookup.raw_value_bits;
    result["numeric_value"] = lookup.numeric_value;
    result["success_kind"] = static_cast<uint32_t>(lookup.success_kind);
    result["bucket_block_raw_bytes"] = lookup.bucket_block_raw_bytes;
    result["value_block_raw_bytes"] = lookup.value_block_raw_bytes;
    result["bucket_block_compressed_bytes"] = lookup.bucket_block_compressed_bytes;
    result["value_block_compressed_bytes"] = lookup.value_block_compressed_bytes;
    return result;
}

nb::dict bc_cold_lookup_to_python(const BCCompressedResult::ColdLookupResult &lookup) {
    nb::dict result;
    result["found"] = lookup.found;
    result["dtype"] = lookup.dtype;
    result["row_width"] = lookup.row_width;
    result["raw_value_bits"] = lookup.raw_value_bits;
    result["numeric_value"] = lookup.numeric_value;
    result["cid"] = lookup.cid;
    result["local_success_row"] = lookup.local_success_row;
    result["value_index"] = lookup.value_index;
    result["bucket_block_raw_bytes"] = lookup.bucket_block_raw_bytes;
    result["bucket_block_compressed_bytes"] = lookup.bucket_block_compressed_bytes;
    result["value_block_raw_bytes"] = lookup.value_block_raw_bytes;
    result["value_block_compressed_bytes"] = lookup.value_block_compressed_bytes;
    return result;
}

[[nodiscard]] bool bc_axis_looks_like_modulo_partition(const BC::BCFamilyTable &axis) {
    if (!axis.is_contiguous_range() || axis.axis_base_coord() != 0U ||
        axis.family_count() == 0U) {
        return false;
    }
    for (BC::FamilyId id = 0U; id < axis.family_count(); ++id) {
        if (axis.id_to_coord(id) != id) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] BC::BCBoardEncodedPosition bc_encode_for_point_reader(
    const BC::BCLut &lut,
    const BC::BCFamilyTable &axis,
    uint64_t board
) {
    BC::BCBoardEncodedPosition out = BC::encode_spawned_canonical_board(lut, axis, board);
    if (out.valid || !bc_axis_looks_like_modulo_partition(axis)) {
        return out;
    }

    const BC::BCQuadrantWords q = BC::unpack_board_to_quadrants(board);
    const BC::BCWordDesc &nw_desc = lut.word_desc(q.nw);
    const BC::BCWordDesc &ne_desc = lut.word_desc(q.ne);
    const BC::BCWordDesc &sw_desc = lut.word_desc(q.sw);
    const BC::BCWordDesc &se_desc = lut.word_desc(q.se);
    if (!nw_desc.valid || !ne_desc.valid || !sw_desc.valid || !se_desc.valid) {
        return {};
    }

    const uint64_t total_sum =
        static_cast<uint64_t>(nw_desc.sum) +
        static_cast<uint64_t>(ne_desc.sum) +
        static_cast<uint64_t>(sw_desc.sum) +
        static_cast<uint64_t>(se_desc.sum);
    if (total_sum != axis.layer_sum()) {
        return {};
    }

    BC::FamilyCoord row_coord = 0U;
    BC::FamilyCoord col_coord = 0U;
    if (!BC::bc_min_side_coord_u64(
            static_cast<uint64_t>(nw_desc.sum) + ne_desc.sum,
            static_cast<uint64_t>(sw_desc.sum) + se_desc.sum,
            axis.family_unit(),
            row_coord) ||
        !BC::bc_min_side_coord_u64(
            static_cast<uint64_t>(nw_desc.sum) + sw_desc.sum,
            static_cast<uint64_t>(ne_desc.sum) + se_desc.sum,
            axis.family_unit(),
            col_coord)) {
        return {};
    }

    const BC::BCEncodedKeyRank encoded =
        BC::bc_encode_key_rank_from_descs(lut, q.nw, nw_desc, ne_desc, sw_desc, se_desc);
    if (!encoded.valid) {
        return {};
    }

    const uint32_t family_count = axis.family_count();
    out.row_family = static_cast<BC::FamilyId>(row_coord % family_count);
    out.col_family = static_cast<BC::FamilyId>(col_coord % family_count);
    const uint64_t cid =
        static_cast<uint64_t>(out.row_family) * family_count +
        static_cast<uint32_t>(out.col_family);
    if (cid > std::numeric_limits<BC::CellId>::max()) {
        throw std::overflow_error("BC point reader modulo cid exceeds CellId");
    }
    out.cid = static_cast<BC::CellId>(cid);
    out.key = encoded.key;
    out.rank = encoded.rank;
    out.bitmap_len = encoded.bitmap_len;
    out.count_ne = encoded.count_ne;
    out.count_sw = encoded.count_sw;
    out.count_se = encoded.count_se;
    out.valid = true;
    return out;
}

[[nodiscard]] double bc_numeric_from_raw(BC::BCSuccessDTypeMode mode, uint64_t raw_bits) {
    switch (mode) {
        case BC::BCSuccessDTypeMode::UInt32:
            return static_cast<double>(static_cast<uint32_t>(raw_bits));
        case BC::BCSuccessDTypeMode::UInt64:
            return static_cast<double>(raw_bits);
        case BC::BCSuccessDTypeMode::Float32:
        case BC::BCSuccessDTypeMode::OneMinusFloat32: {
            uint32_t bits = static_cast<uint32_t>(raw_bits);
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            return static_cast<double>(value);
        }
        case BC::BCSuccessDTypeMode::Float64:
        case BC::BCSuccessDTypeMode::OneMinusFloat64: {
            double value = 0.0;
            std::memcpy(&value, &raw_bits, sizeof(value));
            return value;
        }
    }
    return 0.0;
}

template <typename T>
[[nodiscard]] uint64_t bc_raw_bits_from_value(T value) {
    uint64_t raw = 0U;
    std::memcpy(&raw, &value, sizeof(T));
    return raw;
}

[[nodiscard]] uint64_t bc_read_success_raw_bits(
    const BC::BCSuccessLayerReader &success,
    BC::CellId cid,
    uint32_t row,
    uint32_t lane
) {
    switch (success.dtype_mode()) {
        case BC::BCSuccessDTypeMode::UInt32:
            return static_cast<uint64_t>(success.read_value_typed<uint32_t>(cid, row, lane));
        case BC::BCSuccessDTypeMode::UInt64:
            return success.read_value_typed<uint64_t>(cid, row, lane);
        case BC::BCSuccessDTypeMode::Float32:
        case BC::BCSuccessDTypeMode::OneMinusFloat32:
            return bc_raw_bits_from_value(success.read_value_typed<float>(cid, row, lane));
        case BC::BCSuccessDTypeMode::Float64:
        case BC::BCSuccessDTypeMode::OneMinusFloat64:
            return bc_raw_bits_from_value(success.read_value_typed<double>(cid, row, lane));
    }
    return 0U;
}

[[nodiscard]] BC::BCPositionHeader bc_read_exact_position_header(
    const BC::BCBufferedFileReader &file
) {
    std::vector<uint8_t> bytes(BC::kBCPositionHeaderBytes);
    file.read_at_cached_size(0U, bytes.data(), bytes.size());
    BC::BCPositionHeader header = BC::bc_read_header(bytes);
    if (header.magic != BC::kBCPositionMagic ||
        header.format_version != BC::kBCPositionFormatVersion ||
        header.header_bytes != BC::kBCPositionHeaderBytes) {
        throw std::runtime_error("BC exact point lookup position header mismatch");
    }
    if (header.key_mode != BC::kBCPositionKeyModeQ4NwExactNeSwSeSumMaskPrefix256 ||
        header.rank_prefix_bits != BC::kBCRankPrefixBits ||
        header.rank_prefix_type != BC::kBCPositionRankPrefixTypeUint16 ||
        header.rank_payload_align != 8U) {
        throw std::runtime_error("BC exact point lookup unsupported position encoding");
    }
    if (header.family_unit == 0U ||
        header.family_unit > std::numeric_limits<uint16_t>::max() ||
        header.family_count == 0U ||
        header.family_count > std::numeric_limits<uint16_t>::max()) {
        throw std::runtime_error("BC exact point lookup invalid position axis");
    }
    const uint64_t axis_bytes = BC::bc_axis_coord_table_bytes(header.family_count);
    if (header.axis_coord_table_bytes != axis_bytes) {
        throw std::runtime_error("BC exact point lookup axis byte mismatch");
    }
    return header;
}

[[nodiscard]] BC::BCFamilyTable bc_read_exact_position_axis(
    const BC::BCBufferedFileReader &file,
    const BC::BCPositionHeader &header,
    uint64_t *bytes_read = nullptr
) {
    const uint64_t axis_bytes = BC::bc_axis_coord_table_bytes(header.family_count);
    if (axis_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC exact point lookup axis table too large");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(axis_bytes));
    file.read_at_cached_size(BC::kBCPositionHeaderBytes, bytes.data(), axis_bytes);
    if (bytes_read != nullptr) {
        *bytes_read += axis_bytes;
    }
    std::vector<BC::FamilyCoord> coords;
    coords.reserve(static_cast<size_t>(header.family_count));
    for (uint32_t i = 0U; i < header.family_count; ++i) {
        coords.push_back(static_cast<BC::FamilyCoord>(
            BC::bc_load_u32_le(bytes.data() + static_cast<size_t>(i) * sizeof(uint32_t))));
    }
    return BC::BCFamilyTable(
        header.layer_sum,
        static_cast<uint16_t>(header.family_unit),
        coords);
}

[[nodiscard]] BC::BCPositionCellDescriptor bc_read_exact_cell_descriptor(
    const BC::BCBufferedFileReader &file,
    const BC::BCPositionHeader &header,
    BC::CellId cid,
    uint64_t *bytes_read = nullptr
) {
    if (cid >= header.descriptor_count) {
        throw std::out_of_range("BC exact point lookup descriptor cid out of range");
    }
    const uint64_t descriptor_offset = BC::bc_checked_add_u64(
        header.descriptor_table_offset,
        static_cast<uint64_t>(cid) * BC::kBCPositionCellDescriptorBytes,
        "BC exact point lookup descriptor offset overflow");
    std::array<uint8_t, BC::kBCPositionCellDescriptorBytes> bytes{};
    file.read_at_cached_size(descriptor_offset, bytes.data(), bytes.size());
    if (bytes_read != nullptr) {
        *bytes_read += bytes.size();
    }
    return BC::bc_read_cell_descriptor(bytes.data(), bytes.size());
}

[[nodiscard]] BC::BCBucketEntry bc_read_exact_bucket_entry(
    const BC::BCBufferedFileReader &file,
    uint64_t bucket_file_offset,
    uint64_t *bytes_read = nullptr
) {
    std::array<uint8_t, BC::kBCPositionBucketEntryBytes> bytes{};
    file.read_at_cached_size(bucket_file_offset, bytes.data(), bytes.size());
    if (bytes_read != nullptr) {
        *bytes_read += bytes.size();
    }
    return BC::bc_read_bucket_entry(bytes.data());
}

[[nodiscard]] bool bc_exact_find_bucket_entry(
    const BC::BCBufferedFileReader &file,
    const BC::BCPositionHeader &header,
    const BC::BCPositionCellDescriptor &desc,
    uint64_t key,
    BC::BCBucketEntry &bucket,
    uint64_t *bytes_read = nullptr
) {
    if (desc.bucket_count == 0U) {
        return false;
    }
    const uint64_t bucket_bytes =
        static_cast<uint64_t>(desc.bucket_count) * BC::kBCPositionBucketEntryBytes;
    if (desc.bucket_meta_offset > header.bucket_meta_bytes ||
        bucket_bytes > header.bucket_meta_bytes - desc.bucket_meta_offset) {
        throw std::runtime_error("BC exact point lookup bucket range exceeds file metadata");
    }
    const uint64_t bucket_base = BC::bc_checked_add_u64(
        header.bucket_meta_offset,
        desc.bucket_meta_offset,
        "BC exact point lookup bucket base overflow");
    uint32_t lo = 0U;
    uint32_t hi = desc.bucket_count;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2U;
        const uint64_t offset = BC::bc_checked_add_u64(
            bucket_base,
            static_cast<uint64_t>(mid) * BC::kBCPositionBucketEntryBytes,
            "BC exact point lookup bucket entry offset overflow");
        const BC::BCBucketEntry candidate =
            bc_read_exact_bucket_entry(file, offset, bytes_read);
        if (candidate.key < key) {
            lo = mid + 1U;
        } else {
            hi = mid;
        }
    }
    if (lo >= desc.bucket_count) {
        return false;
    }
    const uint64_t offset = BC::bc_checked_add_u64(
        bucket_base,
        static_cast<uint64_t>(lo) * BC::kBCPositionBucketEntryBytes,
        "BC exact point lookup bucket entry offset overflow");
    bucket = bc_read_exact_bucket_entry(file, offset, bytes_read);
    return bucket.key == key;
}

[[nodiscard]] BC::BCLookupResult bc_exact_lookup_bucket_rank_precise(
    const BC::BCBufferedFileReader &file,
    const BC::BCLut &lut,
    const BC::BCPositionHeader &header,
    const BC::BCPositionCellDescriptor &desc,
    const BC::BCBucketEntry &bucket,
    BC::BucketRank rank,
    uint64_t *bytes_read = nullptr
) {
    const uint32_t bitmap_len = BC::bitmap_len_from_key(lut, bucket.key);
    if (rank >= bitmap_len) {
        return {};
    }
    const uint32_t prefix_count = BC::prefix_count_for_bits(bitmap_len);
    const uint32_t bitmap_word_count = BC::words_for_bits(bitmap_len);
    if (prefix_count == 0U || bitmap_word_count == 0U) {
        return {};
    }
    const uint32_t prefix_offset = bucket.rank_payload_offset;
    const uint32_t bitmap_offset = BC::bc_rank_payload_bitmap_offset(prefix_offset, bitmap_len);
    const uint64_t prefix_end =
        static_cast<uint64_t>(prefix_offset) +
        static_cast<uint64_t>(prefix_count) * sizeof(BC::RankPrefix);
    const uint64_t bitmap_end =
        static_cast<uint64_t>(bitmap_offset) +
        static_cast<uint64_t>(bitmap_word_count) * sizeof(uint64_t);
    if (prefix_end > desc.rank_payload_bytes || bitmap_end > desc.rank_payload_bytes) {
        throw std::runtime_error("BC exact point lookup rank payload range exceeds cell");
    }
    if (desc.rank_payload_offset > header.rank_payload_bytes ||
        desc.rank_payload_bytes > header.rank_payload_bytes - desc.rank_payload_offset) {
        throw std::runtime_error("BC exact point lookup cell rank range exceeds file metadata");
    }
    const uint64_t rank_base = BC::bc_checked_add_u64(
        header.rank_payload_offset,
        desc.rank_payload_offset,
        "BC exact point lookup rank base overflow");
    const uint32_t rank_u32 = static_cast<uint32_t>(rank);
    const uint32_t block = std::min<uint32_t>(
        rank_u32 / BC::kBCRankPrefixBits,
        prefix_count - 1U);
    const uint32_t block_first_word =
        (block * BC::kBCRankPrefixBits) / BC::kBCBitmapWordBits;
    const uint32_t target_word = rank_u32 / BC::kBCBitmapWordBits;
    if (target_word < block_first_word || target_word >= bitmap_word_count) {
        throw std::logic_error("BC exact point lookup target bitmap word out of range");
    }

    std::array<uint8_t, sizeof(BC::RankPrefix)> prefix_bytes{};
    const uint64_t prefix_file_offset = BC::bc_checked_add_u64(
        rank_base,
        static_cast<uint64_t>(prefix_offset) +
            static_cast<uint64_t>(block) * sizeof(BC::RankPrefix),
        "BC exact point lookup prefix file offset overflow");
    file.read_at_cached_size(prefix_file_offset, prefix_bytes.data(), prefix_bytes.size());
    if (bytes_read != nullptr) {
        *bytes_read += prefix_bytes.size();
    }
    uint32_t before = BC::load_u16_le(prefix_bytes.data());

    const uint32_t words_to_read = target_word - block_first_word + 1U;
    std::array<uint8_t, 4U * sizeof(uint64_t)> word_bytes{};
    const uint64_t word_file_offset = BC::bc_checked_add_u64(
        rank_base,
        static_cast<uint64_t>(bitmap_offset) +
            static_cast<uint64_t>(block_first_word) * sizeof(uint64_t),
        "BC exact point lookup bitmap word file offset overflow");
    const uint64_t word_read_bytes = static_cast<uint64_t>(words_to_read) * sizeof(uint64_t);
    file.read_at_cached_size(word_file_offset, word_bytes.data(), word_read_bytes);
    if (bytes_read != nullptr) {
        *bytes_read += word_read_bytes;
    }
    for (uint32_t i = 0U; i + 1U < words_to_read; ++i) {
        before += BC::popcount64(
            BC::load_u64_le(word_bytes.data() + static_cast<size_t>(i) * sizeof(uint64_t)));
    }
    const uint64_t target =
        BC::load_u64_le(word_bytes.data() + static_cast<size_t>(words_to_read - 1U) * sizeof(uint64_t));
    const uint32_t bit = rank_u32 & (BC::kBCBitmapWordBits - 1U);
    if (((target >> bit) & 1ULL) == 0ULL) {
        return {};
    }
    if (bit != 0U) {
        before += BC::popcount64(target & ((1ULL << bit) - 1ULL));
    }
    if (bucket.success_row_offset > std::numeric_limits<uint32_t>::max() - before) {
        throw std::overflow_error("BC exact point lookup local success row overflow");
    }
    return BC::BCLookupResult{
        true,
        bucket.success_row_offset + before
    };
}

[[nodiscard]] BC::BCSuccessHeader bc_read_exact_success_header(
    const BC::BCBufferedFileReader &file
) {
    std::vector<uint8_t> bytes(BC::kBCSuccessHeaderBytes);
    file.read_at_cached_size(0U, bytes.data(), bytes.size());
    BC::BCSuccessHeader header = BC::bc_read_success_header(bytes);
    if (header.magic != BC::kBCSuccessMagic ||
        header.format_version != BC::kBCSuccessFormatVersion ||
        header.header_bytes != BC::kBCSuccessHeaderBytes) {
        throw std::runtime_error("BC exact point lookup success header mismatch");
    }
    if (header.row_width == 0U) {
        throw std::runtime_error("BC exact point lookup success row_width is zero");
    }
    (void)BC::bc_success_dtype_from_u32(header.dtype);
    return header;
}

[[nodiscard]] uint64_t bc_exact_read_success_raw_bits_precise(
    const BC::BCBufferedFileReader &file,
    const BC::BCSuccessHeader &header,
    BC::CellId cid,
    uint32_t local_success_row,
    uint32_t lane,
    uint64_t *bytes_read = nullptr,
    uint64_t *value_index_out = nullptr
) {
    if (lane >= header.row_width) {
        throw std::out_of_range("BC exact point lookup lane out of range");
    }
    if (cid >= header.descriptor_count) {
        throw std::out_of_range("BC exact point lookup success cid out of range");
    }
    std::array<uint8_t, BC::kBCSuccessCellValueOffsetBytes> offset_bytes{};
    const uint64_t offset_file = BC::bc_checked_add_u64(
        header.cell_value_offsets_offset,
        static_cast<uint64_t>(cid) * BC::kBCSuccessCellValueOffsetBytes,
        "BC exact point lookup success offset table overflow");
    file.read_at_cached_size(offset_file, offset_bytes.data(), offset_bytes.size());
    if (bytes_read != nullptr) {
        *bytes_read += offset_bytes.size();
    }
    const uint64_t cell_value_offset = BC::load_u64_le(offset_bytes.data());
    const uint64_t local_value_index = BC::bc_checked_add_u64(
        static_cast<uint64_t>(local_success_row) * header.row_width,
        lane,
        "BC exact point lookup local success value index overflow");
    const uint64_t value_index = BC::bc_checked_add_u64(
        cell_value_offset,
        local_value_index,
        "BC exact point lookup success value index overflow");
    const BC::BCSuccessDTypeMode dtype = BC::bc_success_dtype_from_u32(header.dtype);
    const uint32_t value_size = BC::bc_success_dtype_value_size(dtype);
    if (value_size != 0U &&
        value_index > std::numeric_limits<uint64_t>::max() / value_size) {
        throw std::overflow_error("BC exact point lookup success value byte index overflow");
    }
    const uint64_t value_relative_byte_offset =
        value_index * static_cast<uint64_t>(value_size);
    const uint64_t value_byte_offset = BC::bc_checked_add_u64(
        header.payload_offset,
        value_relative_byte_offset,
        "BC exact point lookup success value byte offset overflow");
    const uint64_t payload_relative = value_byte_offset - header.payload_offset;
    if (payload_relative > header.payload_bytes ||
        value_size > header.payload_bytes - payload_relative) {
        throw std::runtime_error("BC exact point lookup success value exceeds payload");
    }
    std::array<uint8_t, sizeof(uint64_t)> value_bytes{};
    file.read_at_cached_size(value_byte_offset, value_bytes.data(), value_size);
    if (bytes_read != nullptr) {
        *bytes_read += value_size;
    }
    if (value_index_out != nullptr) {
        *value_index_out = value_index;
    }
    switch (dtype) {
        case BC::BCSuccessDTypeMode::UInt32:
        case BC::BCSuccessDTypeMode::Float32:
        case BC::BCSuccessDTypeMode::OneMinusFloat32:
            return static_cast<uint64_t>(BC::bc_load_u32_le(value_bytes.data()));
        case BC::BCSuccessDTypeMode::UInt64:
        case BC::BCSuccessDTypeMode::Float64:
        case BC::BCSuccessDTypeMode::OneMinusFloat64:
            return BC::load_u64_le(value_bytes.data());
    }
    return 0U;
}

[[nodiscard]] BCCompressedResult::ColdLookupResult bc_lookup_exact_result_cold_precise(
    const std::filesystem::path &position_path,
    const std::filesystem::path &success_path,
    const BC::BCLut &lut,
    uint64_t board,
    uint32_t lane
) {
    BCCompressedResult::ColdLookupResult result;
    uint64_t position_bytes_read = 0U;
    uint64_t success_bytes_read = 0U;
    BC::BCBufferedFileReader position_file(position_path);
    const BC::BCPositionHeader position_header =
        bc_read_exact_position_header(position_file);
    position_bytes_read += BC::kBCPositionHeaderBytes;
    const BC::BCFamilyTable axis =
        bc_read_exact_position_axis(position_file, position_header, &position_bytes_read);
    const BC::BCBoardEncodedPosition encoded =
        bc_encode_for_point_reader(lut, axis, board);
    if (!encoded.valid || encoded.cid >= position_header.descriptor_count) {
        result.bucket_block_raw_bytes = position_bytes_read;
        return result;
    }
    const BC::BCPositionCellDescriptor desc =
        bc_read_exact_cell_descriptor(position_file, position_header, encoded.cid, &position_bytes_read);
    if (desc.empty() || desc.success_rows == 0U) {
        result.bucket_block_raw_bytes = position_bytes_read;
        return result;
    }
    BC::BCBucketEntry bucket;
    if (!bc_exact_find_bucket_entry(
            position_file,
            position_header,
            desc,
            encoded.key,
            bucket,
            &position_bytes_read)) {
        result.bucket_block_raw_bytes = position_bytes_read;
        return result;
    }
    const BC::BCLookupResult row = bc_exact_lookup_bucket_rank_precise(
        position_file,
        lut,
        position_header,
        desc,
        bucket,
        encoded.rank,
        &position_bytes_read);
    if (!row.found) {
        result.bucket_block_raw_bytes = position_bytes_read;
        return result;
    }
    if (row.local_success_row >= desc.success_rows) {
        throw std::runtime_error("BC exact point lookup local success row exceeds descriptor");
    }

    BC::BCBufferedFileReader success_file(success_path);
    const BC::BCSuccessHeader success_header = bc_read_exact_success_header(success_file);
    success_bytes_read += BC::kBCSuccessHeaderBytes;
    if (success_header.descriptor_count != position_header.descriptor_count ||
        success_header.family_count != position_header.family_count ||
        success_header.position_key_mode != position_header.key_mode ||
        success_header.family_unit != position_header.family_unit ||
        success_header.axis_base_coord != position_header.axis_base_coord ||
        success_header.layer_sum != position_header.layer_sum) {
        throw std::runtime_error("BC exact point lookup position/success metadata mismatch");
    }

    uint64_t value_index = 0U;
    const uint64_t raw_bits = bc_exact_read_success_raw_bits_precise(
        success_file,
        success_header,
        encoded.cid,
        row.local_success_row,
        lane,
        &success_bytes_read,
        &value_index);
    const BC::BCSuccessDTypeMode dtype = BC::bc_success_dtype_from_u32(success_header.dtype);
    result.found = true;
    result.dtype = success_header.dtype;
    result.row_width = success_header.row_width;
    result.raw_value_bits = raw_bits;
    result.numeric_value = bc_numeric_from_raw(dtype, raw_bits);
    result.cid = encoded.cid;
    result.local_success_row = row.local_success_row;
    result.value_index = value_index;
    result.bucket_block_raw_bytes = position_bytes_read;
    result.value_block_raw_bytes = success_bytes_read;
    return result;
}

[[nodiscard]] std::vector<BC::BCPositionCellDescriptor> bc_read_exact_position_descriptors(
    const BC::BCBufferedFileReader &file,
    const BC::BCPositionHeader &header,
    uint64_t *bytes_read = nullptr
) {
    if (header.descriptor_count != 0U &&
        header.descriptor_count >
            std::numeric_limits<uint64_t>::max() / BC::kBCPositionCellDescriptorBytes) {
        throw std::overflow_error("BC exact sample descriptor byte count overflow");
    }
    const uint64_t expected_bytes =
        header.descriptor_count * static_cast<uint64_t>(BC::kBCPositionCellDescriptorBytes);
    if (header.descriptor_table_bytes != expected_bytes) {
        throw std::runtime_error("BC exact sample descriptor table byte mismatch");
    }
    if (expected_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC exact sample descriptor table too large");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(expected_bytes));
    file.read_at_cached_size(header.descriptor_table_offset, bytes.data(), expected_bytes);
    if (bytes_read != nullptr) {
        *bytes_read += expected_bytes;
    }
    std::vector<BC::BCPositionCellDescriptor> descriptors;
    descriptors.reserve(static_cast<size_t>(header.descriptor_count));
    for (uint64_t cid = 0U; cid < header.descriptor_count; ++cid) {
        const size_t offset =
            static_cast<size_t>(cid) * BC::kBCPositionCellDescriptorBytes;
        descriptors.push_back(BC::bc_read_cell_descriptor(
            bytes.data() + offset,
            BC::kBCPositionCellDescriptorBytes));
    }
    return descriptors;
}

[[nodiscard]] uint64_t bc_exact_bucket_base(
    const BC::BCPositionHeader &header,
    const BC::BCPositionCellDescriptor &desc
) {
    const uint64_t bucket_bytes =
        static_cast<uint64_t>(desc.bucket_count) * BC::kBCPositionBucketEntryBytes;
    if (desc.bucket_meta_offset > header.bucket_meta_bytes ||
        bucket_bytes > header.bucket_meta_bytes - desc.bucket_meta_offset) {
        throw std::runtime_error("BC exact bucket range exceeds file metadata");
    }
    return BC::bc_checked_add_u64(
        header.bucket_meta_offset,
        desc.bucket_meta_offset,
        "BC exact bucket base overflow");
}

[[nodiscard]] BC::BCBucketEntry bc_read_exact_bucket_entry_by_index(
    const BC::BCBufferedFileReader &file,
    uint64_t bucket_base,
    uint32_t bucket_index,
    uint64_t *bytes_read = nullptr
) {
    const uint64_t offset = BC::bc_checked_add_u64(
        bucket_base,
        static_cast<uint64_t>(bucket_index) * BC::kBCPositionBucketEntryBytes,
        "BC exact bucket entry offset overflow");
    return bc_read_exact_bucket_entry(file, offset, bytes_read);
}

[[nodiscard]] bool bc_exact_find_bucket_by_success_row(
    const BC::BCBufferedFileReader &file,
    const BC::BCPositionHeader &header,
    const BC::BCPositionCellDescriptor &desc,
    uint32_t local_success_row,
    BC::BCBucketEntry &bucket,
    uint32_t &bucket_row_count,
    uint64_t *bytes_read = nullptr
) {
    if (desc.empty() || desc.bucket_count == 0U || local_success_row >= desc.success_rows) {
        return false;
    }
    const uint64_t bucket_base = bc_exact_bucket_base(header, desc);
    uint32_t lo = 0U;
    uint32_t hi = desc.bucket_count;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2U;
        const BC::BCBucketEntry candidate =
            bc_read_exact_bucket_entry_by_index(file, bucket_base, mid, bytes_read);
        if (candidate.success_row_offset <= local_success_row) {
            lo = mid + 1U;
        } else {
            hi = mid;
        }
    }
    if (lo == 0U) {
        return false;
    }
    const uint32_t bucket_index = lo - 1U;
    bucket = bc_read_exact_bucket_entry_by_index(
        file,
        bucket_base,
        bucket_index,
        bytes_read);
    const uint32_t next_success_row = bucket_index + 1U < desc.bucket_count
        ? bc_read_exact_bucket_entry_by_index(
              file,
              bucket_base,
              bucket_index + 1U,
              bytes_read).success_row_offset
        : desc.success_rows;
    if (next_success_row < bucket.success_row_offset) {
        throw std::runtime_error("BC exact sample bucket success rows are not monotonic");
    }
    if (local_success_row < bucket.success_row_offset ||
        local_success_row >= next_success_row) {
        return false;
    }
    bucket_row_count = next_success_row - bucket.success_row_offset;
    return bucket_row_count != 0U;
}

[[nodiscard]] BC::RankPrefix bc_exact_read_rank_prefix(
    const BC::BCBufferedFileReader &file,
    uint64_t rank_base,
    uint32_t prefix_offset,
    uint32_t prefix_index,
    uint64_t *bytes_read = nullptr
) {
    std::array<uint8_t, sizeof(BC::RankPrefix)> bytes{};
    const uint64_t offset = BC::bc_checked_add_u64(
        rank_base,
        static_cast<uint64_t>(prefix_offset) +
            static_cast<uint64_t>(prefix_index) * sizeof(BC::RankPrefix),
        "BC exact rank prefix offset overflow");
    file.read_at_cached_size(offset, bytes.data(), bytes.size());
    if (bytes_read != nullptr) {
        *bytes_read += bytes.size();
    }
    return static_cast<BC::RankPrefix>(BC::load_u16_le(bytes.data()));
}

[[nodiscard]] bool bc_exact_select_rank_for_bucket_ordinal_precise(
    const BC::BCBufferedFileReader &file,
    const BC::BCLut &lut,
    const BC::BCPositionHeader &header,
    const BC::BCPositionCellDescriptor &desc,
    const BC::BCBucketEntry &bucket,
    uint32_t bucket_row_count,
    uint32_t ordinal,
    BC::BucketRank &rank,
    uint64_t *bytes_read = nullptr
) {
    if (ordinal >= bucket_row_count) {
        return false;
    }
    const uint32_t bitmap_len = BC::bitmap_len_from_key(lut, bucket.key);
    const uint32_t prefix_count = BC::prefix_count_for_bits(bitmap_len);
    const uint32_t bitmap_word_count = BC::words_for_bits(bitmap_len);
    if (prefix_count == 0U || bitmap_word_count == 0U) {
        return false;
    }
    const uint32_t prefix_offset = bucket.rank_payload_offset;
    const uint32_t bitmap_offset = BC::bc_rank_payload_bitmap_offset(prefix_offset, bitmap_len);
    const uint64_t prefix_end =
        static_cast<uint64_t>(prefix_offset) +
        static_cast<uint64_t>(prefix_count) * sizeof(BC::RankPrefix);
    const uint64_t bitmap_end =
        static_cast<uint64_t>(bitmap_offset) +
        static_cast<uint64_t>(bitmap_word_count) * sizeof(uint64_t);
    if (prefix_end > desc.rank_payload_bytes || bitmap_end > desc.rank_payload_bytes) {
        throw std::runtime_error("BC exact sample rank payload range exceeds cell");
    }
    if (desc.rank_payload_offset > header.rank_payload_bytes ||
        desc.rank_payload_bytes > header.rank_payload_bytes - desc.rank_payload_offset) {
        throw std::runtime_error("BC exact sample cell rank range exceeds file metadata");
    }
    const uint64_t rank_base = BC::bc_checked_add_u64(
        header.rank_payload_offset,
        desc.rank_payload_offset,
        "BC exact sample rank base overflow");

    uint32_t lo = 0U;
    uint32_t hi = prefix_count;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2U;
        const uint32_t block_end = mid + 1U < prefix_count
            ? bc_exact_read_rank_prefix(
                  file,
                  rank_base,
                  prefix_offset,
                  mid + 1U,
                  bytes_read)
            : bucket_row_count;
        if (ordinal < block_end) {
            hi = mid;
        } else {
            lo = mid + 1U;
        }
    }
    if (lo >= prefix_count) {
        return false;
    }
    const uint32_t block = lo;
    const uint32_t block_begin = bc_exact_read_rank_prefix(
        file,
        rank_base,
        prefix_offset,
        block,
        bytes_read);
    const uint32_t block_end = block + 1U < prefix_count
        ? bc_exact_read_rank_prefix(
              file,
              rank_base,
              prefix_offset,
              block + 1U,
              bytes_read)
        : bucket_row_count;
    if (ordinal < block_begin || ordinal >= block_end) {
        return false;
    }
    uint32_t remaining = ordinal - block_begin;
    constexpr uint32_t kWordsPerPrefixBlock = BC::kBCRankPrefixBits / BC::kBCBitmapWordBits;
    const uint32_t block_first_word = block * kWordsPerPrefixBlock;
    if (block_first_word >= bitmap_word_count) {
        return false;
    }
    const uint32_t words_to_read =
        std::min<uint32_t>(kWordsPerPrefixBlock, bitmap_word_count - block_first_word);
    std::array<uint8_t, kWordsPerPrefixBlock * sizeof(uint64_t)> word_bytes{};
    const uint64_t word_file_offset = BC::bc_checked_add_u64(
        rank_base,
        static_cast<uint64_t>(bitmap_offset) +
            static_cast<uint64_t>(block_first_word) * sizeof(uint64_t),
        "BC exact sample bitmap word offset overflow");
    const uint64_t word_read_bytes = static_cast<uint64_t>(words_to_read) * sizeof(uint64_t);
    file.read_at_cached_size(word_file_offset, word_bytes.data(), word_read_bytes);
    if (bytes_read != nullptr) {
        *bytes_read += word_read_bytes;
    }
    for (uint32_t word_i = 0U; word_i < words_to_read; ++word_i) {
        const uint32_t global_word = block_first_word + word_i;
        uint64_t word = BC::load_u64_le(
            word_bytes.data() + static_cast<size_t>(word_i) * sizeof(uint64_t));
        if (global_word + 1U == bitmap_word_count && (bitmap_len & 63U) != 0U) {
            word &= (1ULL << (bitmap_len & 63U)) - 1ULL;
        }
        const uint32_t live_in_word = BC::popcount64(word);
        if (remaining >= live_in_word) {
            remaining -= live_in_word;
            continue;
        }
        for (uint32_t bit = 0U; bit < BC::kBCBitmapWordBits; ++bit) {
            const uint32_t rank_candidate = global_word * BC::kBCBitmapWordBits + bit;
            if (rank_candidate >= bitmap_len) {
                return false;
            }
            if (((word >> bit) & 1ULL) == 0ULL) {
                continue;
            }
            if (remaining == 0U) {
                rank = static_cast<BC::BucketRank>(rank_candidate);
                return true;
            }
            --remaining;
        }
        return false;
    }
    return false;
}

[[nodiscard]] uint64_t bc_sample_exact_random_board_precise(
    const std::filesystem::path &position_path,
    const BC::BCLut &lut
) {
    BC::BCBufferedFileReader position_file(position_path);
    const BC::BCPositionHeader header = bc_read_exact_position_header(position_file);
    uint64_t bytes_read = BC::kBCPositionHeaderBytes;
    const std::vector<BC::BCPositionCellDescriptor> descriptors =
        bc_read_exact_position_descriptors(position_file, header, &bytes_read);
    uint64_t live_rows = 0U;
    for (const BC::BCPositionCellDescriptor &desc : descriptors) {
        live_rows += desc.success_rows;
    }
    if (live_rows == 0U) {
        return 0ULL;
    }

    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> row_pick(0U, live_rows - 1U);
    constexpr uint32_t kSampleAttempts = 128U;
    for (uint32_t attempt = 0U; attempt < kSampleAttempts; ++attempt) {
        uint64_t target_row = row_pick(rng);
        BC::CellId cid = 0U;
        const BC::BCPositionCellDescriptor *desc = nullptr;
        for (; cid < descriptors.size(); ++cid) {
            const BC::BCPositionCellDescriptor &candidate = descriptors[cid];
            if (target_row < candidate.success_rows) {
                desc = &candidate;
                break;
            }
            target_row -= candidate.success_rows;
        }
        if (desc == nullptr || desc->empty() || desc->success_rows == 0U) {
            continue;
        }
        const uint32_t local_row = static_cast<uint32_t>(target_row);
        BC::BCBucketEntry bucket;
        uint32_t bucket_row_count = 0U;
        if (!bc_exact_find_bucket_by_success_row(
                position_file,
                header,
                *desc,
                local_row,
                bucket,
                bucket_row_count,
                &bytes_read)) {
            continue;
        }
        BC::BucketRank rank = 0U;
        if (!bc_exact_select_rank_for_bucket_ordinal_precise(
                position_file,
                lut,
                header,
                *desc,
                bucket,
                bucket_row_count,
                local_row - bucket.success_row_offset,
                rank,
                &bytes_read)) {
            continue;
        }
        const BC::BCBucketBoardDecoder decoder(lut, bucket.key);
        return decoder.board(rank);
    }
    return 0ULL;
}

[[nodiscard]] nb::dict bc_exact_miss_dict(uint32_t dtype = 1U) {
    nb::dict result;
    result["found"] = false;
    result["dtype"] = dtype;
    result["row_width"] = 1U;
    result["raw_value_bits"] = 0U;
    result["numeric_value"] = 0.0;
    result["cid"] = 0U;
    result["local_success_row"] = 0U;
    result["value_index"] = 0U;
    result["bucket_block_raw_bytes"] = 0U;
    result["bucket_block_compressed_bytes"] = 0U;
    result["value_block_raw_bytes"] = 0U;
    result["value_block_compressed_bytes"] = 0U;
    return result;
}

[[nodiscard]] bool bc_rank_for_payload_ordinal(
    const BC::BCLut &lut,
    const BC::BCBucketEntry &bucket,
    BC::BCRankPayloadView payload,
    uint32_t ordinal,
    BC::BucketRank &rank
) {
    const uint32_t bitmap_len = BC::bitmap_len_from_key(lut, bucket.key);
    const uint32_t bitmap_word_count = BC::words_for_bits(bitmap_len);
    const uint32_t bitmap_offset =
        BC::bc_rank_payload_bitmap_offset(bucket.rank_payload_offset, bitmap_len);
    const uint64_t bitmap_end =
        static_cast<uint64_t>(bitmap_offset) +
        static_cast<uint64_t>(bitmap_word_count) * sizeof(uint64_t);
    if (bitmap_end > payload.size) {
        throw std::out_of_range("BC exact sample bucket bitmap exceeds rank payload");
    }
    uint32_t remaining = ordinal;
    for (uint32_t word_i = 0U; word_i < bitmap_word_count; ++word_i) {
        uint64_t word = BC::load_u64_le(
            payload.data + bitmap_offset + static_cast<size_t>(word_i) * sizeof(uint64_t));
        if (word_i + 1U == bitmap_word_count && (bitmap_len & 63U) != 0U) {
            word &= (1ULL << (bitmap_len & 63U)) - 1ULL;
        }
        const uint32_t live_in_word = BC::popcount64(word);
        if (remaining >= live_in_word) {
            remaining -= live_in_word;
            continue;
        }
        for (uint32_t bit = 0U; bit < BC::kBCBitmapWordBits; ++bit) {
            if (((word >> bit) & 1ULL) == 0ULL) {
                continue;
            }
            if (remaining == 0U) {
                rank = static_cast<BC::BucketRank>(word_i * 64U + bit);
                return rank < bitmap_len;
            }
            --remaining;
        }
        return false;
    }
    return false;
}

} // namespace

NB_MODULE(formation_core, m) {
    NativeDiagnostics::install_crash_handler("formation_core");

    nb::enum_<SymmMode>(m, "SymmMode")
        .value("Identity", SymmMode::Identity)
        .value("Full", SymmMode::Full)
        .value("Diagonal", SymmMode::Diagonal)
        .value("Horizontal", SymmMode::Horizontal)
        .value("Min33", SymmMode::Min33)
        .value("Min24", SymmMode::Min24)
        .value("Min34", SymmMode::Min34)
        .value("Min34Top", SymmMode::Min34Top);

    nb::class_<PatternSpec>(m, "PatternSpec")
        .def(nb::init<>())
        .def_rw("name", &PatternSpec::name)
        .def_rw("pattern_masks", &PatternSpec::pattern_masks)
        .def_rw("success_shifts", &PatternSpec::success_shifts)
        .def_rw("symm_mode", &PatternSpec::symm_mode)
        .def_rw("physical_transform", &PatternSpec::physical_transform)
        .def_rw("inverse_physical_transform", &PatternSpec::inverse_physical_transform)
        .def_rw("logical_pattern_signature", &PatternSpec::logical_pattern_signature)
        .def_rw("physical_pattern_signature", &PatternSpec::physical_pattern_signature);

    nb::class_<RunOptions>(m, "RunOptions")
        .def(nb::init<>())
        .def_rw("target", &RunOptions::target)
        .def_rw("steps", &RunOptions::steps)
        .def_rw("docheck_step", &RunOptions::docheck_step)
        .def_rw("pathname", &RunOptions::pathname)
        .def_rw("is_free", &RunOptions::is_free)
        .def_rw("is_variant", &RunOptions::is_variant)
        .def_rw("spawn_rate4", &RunOptions::spawn_rate4)
        .def_rw("success_rate_dtype", &RunOptions::success_rate_dtype)
        .def_rw("deletion_threshold", &RunOptions::deletion_threshold)
        .def_rw("relative_deletion_threshold", &RunOptions::relative_deletion_threshold)
        .def_rw("deletion_threshold_signal_path", &RunOptions::deletion_threshold_signal_path)
        .def_rw("compress", &RunOptions::compress)
        .def_rw("compress_temp_files", &RunOptions::compress_temp_files)
        .def_rw("optimal_branch_only", &RunOptions::optimal_branch_only)
        .def_rw("chunked_solve", &RunOptions::chunked_solve)
        .def_rw("num_threads", &RunOptions::num_threads)
        .def_rw("direct_io", &RunOptions::direct_io)
        .def_rw("direct_io_queue_depth", &RunOptions::direct_io_queue_depth)
        .def_rw("direct_io_chunk_mib", &RunOptions::direct_io_chunk_mib);

    nb::class_<AdvancedPatternSpec>(m, "AdvancedPatternSpec")
        .def(nb::init<>())
        .def_rw("name", &AdvancedPatternSpec::name)
        .def_rw("pattern_masks", &AdvancedPatternSpec::pattern_masks)
        .def_rw("success_shifts", &AdvancedPatternSpec::success_shifts)
        .def_rw("symm_mode", &AdvancedPatternSpec::symm_mode)
        .def_rw("physical_transform", &AdvancedPatternSpec::physical_transform)
        .def_rw("inverse_physical_transform", &AdvancedPatternSpec::inverse_physical_transform)
        .def_rw("logical_pattern_signature", &AdvancedPatternSpec::logical_pattern_signature)
        .def_rw("physical_pattern_signature", &AdvancedPatternSpec::physical_pattern_signature)
        .def_rw("num_free_32k", &AdvancedPatternSpec::num_free_32k)
        .def_rw("fixed_32k_shifts", &AdvancedPatternSpec::fixed_32k_shifts)
        .def_rw("small_tile_sum_limit", &AdvancedPatternSpec::small_tile_sum_limit)
        .def_rw("target", &AdvancedPatternSpec::target);

    nb::class_<ClassicBookReader>(m, "ClassicBookReader")
        .def(nb::init<PatternSpec, bool>(), "pattern_spec"_a, "is_variant"_a = false)
        .def(
            "move_on_dic",
            [](ClassicBookReader &reader,
               const std::vector<std::vector<int>> &board,
               const std::vector<std::pair<std::string, std::string>> &path_list,
               const std::string &pattern_full,
               int64_t nums_adjust) {
                return reader_result_to_python(reader.move_on_dic(board, path_list, pattern_full, nums_adjust));
            },
            "board"_a,
            "path_list"_a,
            "pattern_full"_a,
            "nums_adjust"_a
        )
        .def(
            "get_random_state",
            &ClassicBookReader::get_random_state,
            "path_list"_a,
            "pattern_full"_a,
            "spawn_rate4"_a
        );

    nb::class_<AdvancedBookReader>(m, "AdvancedBookReader")
        .def(nb::init<AdvancedPatternSpec, bool>(), "pattern_spec"_a, "is_variant"_a = false)
        .def(
            "move_on_dic",
            [](AdvancedBookReader &reader,
               const std::vector<std::vector<int>> &board,
               const std::vector<std::pair<std::string, std::string>> &path_list,
               const std::string &pattern_full,
               int64_t nums_adjust) {
                return reader_result_to_python(reader.move_on_dic(board, path_list, pattern_full, nums_adjust));
            },
            "board"_a,
            "path_list"_a,
            "pattern_full"_a,
            "nums_adjust"_a
        )
        .def(
            "get_random_state",
            &AdvancedBookReader::get_random_state,
            "path_list"_a,
            "pattern_full"_a,
            "spawn_rate4"_a
        );

    nb::class_<EXADBookReader>(m, "EXADBookReader")
        .def(nb::init<AdvancedPatternSpec, bool>(), "pattern_spec"_a, "is_variant"_a = false)
        .def(
            "move_on_dic",
            [](EXADBookReader &reader,
               const std::vector<std::vector<int>> &board,
               const std::vector<std::pair<std::string, std::string>> &path_list,
               const std::string &pattern_full,
               int64_t nums_adjust) {
                return reader_result_to_python(reader.move_on_dic(board, path_list, pattern_full, nums_adjust));
            },
            "board"_a,
            "path_list"_a,
            "pattern_full"_a,
            "nums_adjust"_a
        )
        .def(
            "get_random_state",
            &EXADBookReader::get_random_state,
            "path_list"_a,
            "pattern_full"_a,
            "spawn_rate4"_a
        );

    nb::class_<EXBookReader>(m, "EXBookReader")
        .def(nb::init<PatternSpec, bool>(), "pattern_spec"_a, "is_variant"_a = false)
        .def(
            "move_on_dic",
            [](EXBookReader &reader,
               const std::vector<std::vector<int>> &board,
               const std::vector<std::pair<std::string, std::string>> &path_list,
               const std::string &pattern_full,
               int64_t nums_adjust) {
                return reader_result_to_python(reader.move_on_dic(board, path_list, pattern_full, nums_adjust));
            },
            "board"_a,
            "path_list"_a,
            "pattern_full"_a,
            "nums_adjust"_a
        )
        .def(
            "get_random_state",
            &EXBookReader::get_random_state,
            "path_list"_a,
            "pattern_full"_a,
            "spawn_rate4"_a
        );

    nb::class_<PatternLayer>(m, "PatternLayer")
        .def(nb::init<>())
        .def_prop_ro("size", &PatternLayer::size)
        .def_prop_ro("empty", &PatternLayer::empty)
        .def_prop_ro("dtype_name", &PatternLayer::dtype_name);

    m.def(
        "get_build_progress",
        []() {
            const BuildProgressSnapshot snapshot = FormationProgress::get_build_progress();
            return nb::make_tuple(snapshot.current, snapshot.total);
        }
    );

    m.def(
        "reset_build_progress",
        [](uint32_t total) {
            FormationProgress::reset_build_progress(total);
        },
        "total"_a = 0U
    );

    m.def(
        "run_bc_family_build",
        [](const nb::dict &options) {
            BC::BCFamilyGenerationRunOptions generation_options =
                bc_generation_options_from_dict(options);
            BC::BCFamilySolveRunOptions solve_options =
                bc_solve_options_from_dict(options);
            const uint32_t expected_layers =
                dict_get_or<uint32_t>(options, "expected_layers", 0U);
            const uint32_t progress_total =
                dict_get_or<uint32_t>(
                    options,
                    "progress_total",
                    expected_layers == 0U ? 0U : expected_layers * 2U);
            const bool skip_generation =
                dict_get_or<bool>(options, "skip_generation", false);
            const std::filesystem::path solve_stats_csv =
                dict_get_or<std::string>(options, "solve_stats_csv", "");
            const std::filesystem::path solve_summary_csv =
                dict_get_or<std::string>(options, "solve_summary_csv", "");

            BC::BCFamilyGenerationRunResult generation_result;
            BC::BCFamilySolveRunResult solve_result;
            {
                nb::gil_scoped_release release;
                FormationProgress::reset_build_progress(progress_total);
                uint32_t generation_progress = 0U;
                uint32_t solve_progress = 0U;

                if (skip_generation) {
                    generation_progress = expected_layers;
                    FormationProgress::update_build_progress(
                        generation_progress,
                        progress_total);
                    generation_result.completed = true;
                } else {
                    generation_result = BC::bc_family_generation_full_run(
                        generation_options,
                        [&](const BC::BCFamilyGenerationRunLayerMetric &) {
                            if (progress_total != 0U) {
                                if (expected_layers == 0U ||
                                    generation_progress < expected_layers) {
                                    ++generation_progress;
                                }
                                FormationProgress::update_build_progress(
                                    generation_progress,
                                    progress_total);
                            }
                        });
                }

                std::unique_ptr<std::ofstream> solve_stats;
                if (!solve_stats_csv.empty()) {
                    if (!solve_stats_csv.parent_path().empty()) {
                        std::filesystem::create_directories(solve_stats_csv.parent_path());
                    }
                    solve_stats = std::make_unique<std::ofstream>(solve_stats_csv);
                    if (!*solve_stats) {
                        throw std::runtime_error("failed to open BC solve stats CSV");
                    }
                    *solve_stats << std::setprecision(12);
                    write_bc_solve_stats_header(*solve_stats);
                }

                solve_result = BC::bc_family_solve_full_run(
                    solve_options,
                    [&](const BC::BCFamilySolveRunLayerMetric &metric) {
                        if (solve_stats) {
                            write_bc_solve_stats_row(*solve_stats, metric);
                            solve_stats->flush();
                        }
                        const bool metric_finishes_archive_layer =
                            metric.kind == "archive" &&
                            (expected_layers == 0U || metric.ordinal < expected_layers);
                        if (progress_total != 0U && metric_finishes_archive_layer) {
                            if (expected_layers == 0U || solve_progress < expected_layers) {
                                ++solve_progress;
                            }
                            const uint32_t base = expected_layers == 0U
                                ? generation_progress
                                : expected_layers;
                            FormationProgress::update_build_progress(
                                base + solve_progress,
                                progress_total);
                        }
                    });
                if (solve_stats) {
                    write_bc_solve_stats_total_row(*solve_stats, solve_result);
                    solve_stats->flush();
                }

                if (!solve_summary_csv.empty()) {
                    if (!solve_summary_csv.parent_path().empty()) {
                        std::filesystem::create_directories(solve_summary_csv.parent_path());
                    }
                    std::ofstream summary(solve_summary_csv);
                    if (!summary) {
                        throw std::runtime_error("failed to open BC solve summary CSV");
                    }
                    write_bc_solve_summary(summary, generation_result, solve_result);
                }

                if (progress_total != 0U) {
                    FormationProgress::update_build_progress(progress_total, progress_total);
                }
            }
            return bc_family_build_summary_to_python(generation_result, solve_result);
        },
        "options"_a
    );

    m.def(
        "trie_compress_book",
        &trie_compress_progress_native,
        "book_path"_a,
        "success_rate_dtype"_a = "uint32"
    );

    m.def(
        "trie_decompress_search",
        [](const std::string &path_prefix, uint64_t board, const std::string &success_rate_dtype) {
            const auto result = trie_decompress_search_native(path_prefix, board, success_rate_dtype);
            return result.value_or(0.0);
        },
        "path_prefix"_a,
        "board"_a,
        "success_rate_dtype"_a
    );

    m.def(
        "find_classic_value",
        [](const std::string &pathname,
           const std::string &filename,
           uint64_t search_key,
           const std::string &success_rate_dtype) {
            bool found = false;
            const double value = find_classic_value_native(pathname, filename, search_key, success_rate_dtype, found);
            return found ? nb::cast(value) : nb::none();
        },
        "pathname"_a,
        "filename"_a,
        "search_key"_a,
        "success_rate_dtype"_a = "uint32"
    );

    m.def("apply_sym_like", &apply_sym_like, "board"_a, "symm_index"_a);

    m.def(
        "run_pattern_generate",
        [](const U64Array &arr_init, const PatternSpec &spec, const RunOptions &options) {
            return run_pattern_generate_cpp(to_u64_vector(arr_init), spec, options);
        },
        "arr_init"_a,
        "pattern_spec"_a,
        "run_options"_a,
        nb::call_guard<nb::gil_scoped_release>()
    );

    m.def(
        "run_pattern_solve",
        &run_pattern_solve_cpp,
        "d1"_a,
        "d2"_a,
        "pattern_spec"_a,
        "run_options"_a,
        nb::call_guard<nb::gil_scoped_release>()
    );

    m.def(
        "run_pattern_build",
        [](const U64Array &arr_init, const PatternSpec &spec, const RunOptions &options) {
            NativeDiagnostics::Scope scope("formation_core.run_pattern_build pattern=" + spec.name);
            run_pattern_build_cpp(to_u64_vector(arr_init), spec, options);
        },
        "arr_init"_a,
        "pattern_spec"_a,
        "run_options"_a,
        nb::call_guard<nb::gil_scoped_release>()
    );

    m.def(
        "run_pattern_build_ad",
        [](const U64Array &arr_init, const AdvancedPatternSpec &spec, const RunOptions &options) {
            NativeDiagnostics::Scope scope("formation_core.run_pattern_build_ad pattern=" + spec.name);
            run_pattern_build_ad_cpp(to_u64_vector(arr_init), spec, options);
        },
        "arr_init"_a,
        "pattern_spec"_a,
        "run_options"_a,
        nb::call_guard<nb::gil_scoped_release>()
    );

    m.def(
        "run_pattern_build_exad",
        [](const U64Array &arr_init, const AdvancedPatternSpec &spec, const RunOptions &options) {
            NativeDiagnostics::Scope scope("formation_core.run_pattern_build_exad pattern=" + spec.name);
            run_pattern_build_exad_cpp(to_u64_vector(arr_init), spec, options);
        },
        "arr_init"_a,
        "pattern_spec"_a,
        "run_options"_a,
        nb::call_guard<nb::gil_scoped_release>()
    );

    m.def(
        "run_pattern_solve_exad",
        [](const U64Array &arr_init, const AdvancedPatternSpec &spec, const RunOptions &options) {
            run_pattern_solve_exad_cpp(to_u64_vector(arr_init), spec, options);
        },
        "arr_init"_a,
        "pattern_spec"_a,
        "run_options"_a,
        nb::call_guard<nb::gil_scoped_release>()
    );

    m.def(
        "run_pattern_build_zmask",
        [](const U64Array &arr_init, const PatternSpec &spec, const RunOptions &options) {
            NativeDiagnostics::Scope scope("formation_core.run_pattern_build_zmask pattern=" + spec.name);
            run_pattern_build_zmask_cpp(to_u64_vector(arr_init), spec, options);
        },
        "arr_init"_a,
        "pattern_spec"_a,
        "run_options"_a,
        nb::call_guard<nb::gil_scoped_release>()
    );

    m.def(
        "run_pattern_solve_zmask",
        [](const U64Array &arr_init, const PatternSpec &spec, const RunOptions &options) {
            run_pattern_solve_zmask_cpp(to_u64_vector(arr_init), spec, options);
        },
        "arr_init"_a,
        "pattern_spec"_a,
        "run_options"_a,
        nb::call_guard<nb::gil_scoped_release>()
    );

    m.def(
        "run_pattern_solve_zmask_single_layer",
        [](const U64Array &arr_init, const PatternSpec &spec, const RunOptions &options, int step) {
            run_pattern_solve_zmask_single_layer_cpp(to_u64_vector(arr_init), spec, options, step);
        },
        "arr_init"_a,
        "pattern_spec"_a,
        "run_options"_a,
        "step"_a,
        nb::call_guard<nb::gil_scoped_release>()
    );

    m.def(
        "compress_ex_zbook_result",
        [](const std::string &zbook_path,
           const std::string &zlut_path,
           const std::string &output_path,
           uint32_t bucket_block_buckets,
           uint32_t success_block_values,
           int compression_level) {
            EXCompressedResult::CompressStats stats;
            {
                nb::gil_scoped_release release;
                stats = EXCompressedResult::compress_zbook_to_ex_result(
                    zbook_path,
                    zlut_path,
                    output_path,
                    bucket_block_buckets,
                    success_block_values,
                    compression_level
                );
            }
            return ex_compress_stats_to_python(stats);
        },
        "zbook_path"_a,
        "zlut_path"_a,
        "output_path"_a,
        "bucket_block_buckets"_a = 4096U,
        "success_block_values"_a = 65536U,
        "compression_level"_a = 5
    );

    m.def(
        "lookup_ex_zbook_result_cold",
        [](const std::string &compressed_path, const std::string &zlut_path, uint64_t board) {
            EXCompressedResult::ColdLookupResult result;
            {
                nb::gil_scoped_release release;
                result = EXCompressedResult::lookup_cold(compressed_path, zlut_path, board);
            }
            return ex_cold_lookup_to_python(result);
        },
        "compressed_path"_a,
        "zlut_path"_a,
        "board"_a
    );

    m.def(
        "lookup_ex_zbook_cold",
        [](const std::string &zbook_path, const std::string &zlut_path, uint64_t board) {
            EXCompressedResult::ColdLookupResult result;
            {
                nb::gil_scoped_release release;
                result = EXCompressedResult::lookup_zbook_cold(zbook_path, zlut_path, board);
            }
            return ex_cold_lookup_to_python(result);
        },
        "zbook_path"_a,
        "zlut_path"_a,
        "board"_a
    );

    m.def(
        "compress_exadbook_result",
        [](const std::string &exadbook_path,
           const std::string &exadlut_path,
           const std::string &output_path,
           uint32_t bucket_block_raw_target_bytes,
           uint32_t success_block_values,
           int compression_level) {
            EXADCompressedResult::CompressStats stats;
            {
                nb::gil_scoped_release release;
                stats = EXADCompressedResult::compress_exad_solved_layer_to_result(
                    exadbook_path,
                    exadlut_path,
                    output_path,
                    bucket_block_raw_target_bytes,
                    success_block_values,
                    compression_level
                );
            }
            return exad_compress_stats_to_python(stats);
        },
        "exadbook_path"_a,
        "exadlut_path"_a,
        "output_path"_a,
        "bucket_block_raw_target_bytes"_a = 512U * 1024U,
        "success_block_values"_a = 65536U,
        "compression_level"_a = 5
    );

    m.def(
        "lookup_exadbook_result_cold",
        [](const std::string &compressed_path,
           const std::string &exadlut_path,
           int ad_key,
           uint64_t canonical_board,
           uint32_t column) {
            EXADCompressedResult::ColdLookupResult result;
            {
                nb::gil_scoped_release release;
                result = EXADCompressedResult::lookup_exad_cold(
                    compressed_path,
                    exadlut_path,
                    ad_key,
                    canonical_board,
                    column
                );
            }
            return exad_cold_lookup_to_python(result);
        },
        "compressed_path"_a,
        "exadlut_path"_a,
        "ad_key"_a,
        "canonical_board"_a,
        "column"_a
    );

    m.def(
        "lookup_exadbook_cold",
        [](const std::string &exadbook_path,
           const std::string &exadlut_path,
           int ad_key,
           uint64_t canonical_board,
           uint32_t column) {
            EXADCompressedResult::ColdLookupResult result;
            {
                nb::gil_scoped_release release;
                result = EXADCompressedResult::lookup_exadbook_cold(
                    exadbook_path,
                    exadlut_path,
                    ad_key,
                    canonical_board,
                    column
                );
            }
            return exad_cold_lookup_to_python(result);
        },
        "exadbook_path"_a,
        "exadlut_path"_a,
        "ad_key"_a,
        "canonical_board"_a,
        "column"_a
    );

    m.def(
        "lookup_bc_compressed_result_cold",
        [](const std::string &compressed_path,
           uint32_t target_rank,
           uint64_t board,
           uint32_t lane) {
            BCCompressedResult::ColdLookupResult result;
            {
                nb::gil_scoped_release release;
                const BC::BCLut lut(bc_free_legal_tiles(target_rank));
                result = BCCompressedResult::lookup_cold(
                    compressed_path,
                    lut,
                    board,
                    lane);
            }
            return bc_cold_lookup_to_python(result);
        },
        "compressed_path"_a,
        "target_rank"_a,
        "board"_a,
        "lane"_a = 0U
    );

    m.def(
        "lookup_bc_exact_result_cold",
        [](const std::string &position_path,
           const std::string &success_path,
           uint32_t target_rank,
           uint64_t board,
           uint32_t lane) {
            BCCompressedResult::ColdLookupResult result;
            {
                nb::gil_scoped_release release;
                const BC::BCLut lut(bc_free_legal_tiles(target_rank));
                result = bc_lookup_exact_result_cold_precise(
                    position_path,
                    success_path,
                    lut,
                    board,
                    lane);
            }
            return bc_cold_lookup_to_python(result);
        },
        "position_path"_a,
        "success_path"_a,
        "target_rank"_a,
        "board"_a,
        "lane"_a = 0U
    );

    m.def(
        "sample_bc_compressed_random_state",
        [](const std::string &compressed_path,
           uint32_t target_rank,
           double spawn_rate4) {
            uint64_t board = 0ULL;
            uint64_t raw_value_bits = 0ULL;
            double numeric_value = 0.0;
            bool ok = false;
            {
                nb::gil_scoped_release release;
                const BC::BCLut lut(bc_free_legal_tiles(target_rank));
                ok = BCCompressedResult::sample_cold(
                    compressed_path,
                    lut,
                    board,
                    raw_value_bits,
                    numeric_value);
            }
            (void)raw_value_bits;
            (void)numeric_value;
            return ok ? gen_new_num(board, static_cast<float>(spawn_rate4)).first : 0ULL;
        },
        "compressed_path"_a,
        "target_rank"_a = 8U,
        "spawn_rate4"_a = 0.1
    );

    m.def(
        "sample_bc_exact_random_state",
        [](const std::string &position_path,
           uint32_t target_rank,
           double spawn_rate4) {
            uint64_t board = 0ULL;
            {
                nb::gil_scoped_release release;
                const BC::BCLut lut(bc_free_legal_tiles(target_rank));
                board = bc_sample_exact_random_board_precise(position_path, lut);
            }
            return board != 0ULL ? gen_new_num(board, static_cast<float>(spawn_rate4)).first : 0ULL;
        },
        "position_path"_a,
        "target_rank"_a = 8U,
        "spawn_rate4"_a = 0.1
    );
}
