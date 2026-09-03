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
#include "BCFamilyStatsCsv.h"
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

std::vector<std::filesystem::path> dict_get_path_vector(const nb::dict &options, const char *key) {
    std::vector<std::filesystem::path> paths;
    nb::str py_key(key);
    if (!options.contains(py_key)) {
        return paths;
    }
    nb::handle value = options[py_key];
    if (value.is_none()) {
        return paths;
    }
    for (const std::string &path : nb::cast<std::vector<std::string>>(value)) {
        if (!path.empty()) {
            paths.emplace_back(path);
        }
    }
    return paths;
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
    run.seed_boards = dict_get_or<std::vector<uint64_t>>(
        options,
        "seed_boards",
        run.seed_boards);
    run.pattern_masks = dict_get_or<std::vector<uint64_t>>(
        options,
        "pattern_masks",
        run.pattern_masks);
    run.success_shifts = dict_get_or<std::vector<uint8_t>>(
        options,
        "success_shifts",
        run.success_shifts);
    run.canonical_symm_mode = dict_get_or<int>(
        options,
        "canonical_symm_mode",
        run.canonical_symm_mode);
    run.success_check_min_source_layer_sum = dict_get_or<uint32_t>(
        options,
        "success_check_min_source_layer_sum",
        run.success_check_min_source_layer_sum);
    run.output_dir = dict_get_or<std::string>(options, "generated_dir", run.output_dir.string());
    run.output_dirs = dict_get_path_vector(options, "generated_dirs");
    run.stats_csv = dict_get_or<std::string>(options, "generation_stats_csv", run.stats_csv.string());
    run.num_threads = dict_get_or<int>(options, "threads", run.num_threads);
    run.family_modulus = dict_get_or<uint32_t>(options, "family_modulus", run.family_modulus);
    run.family_route = parse_bc_generation_route_binding(
        dict_get_or<std::string>(options, "family_route", "auto"));
    run.direct_queue_depth = dict_get_or<uint32_t>(
        options,
        "direct_queue_depth",
        run.direct_queue_depth);
    run.direct_io_chunk_mib = dict_get_or<uint32_t>(
        options,
        "direct_io_chunk_mib",
        run.direct_io_chunk_mib);
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
    run.compress_temp_files = dict_get_or<bool>(
        options,
        "compress_temp_files",
        run.compress_temp_files);
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
    run.generated_position_dirs = dict_get_path_vector(options, "generated_dirs");
    run.solved_output_dirs = dict_get_path_vector(options, "solved_dirs");
    run.archive_output_dirs = dict_get_path_vector(options, "archive_dirs");
    run.prefix = dict_get_or<std::string>(options, "prefix", run.prefix);
    run.target_rank = dict_get_or<uint32_t>(options, "target_rank", run.target_rank);
    run.success_target_rank = dict_get_or<int>(
        options,
        "success_target_rank",
        static_cast<int>(run.target_rank));
    run.pattern_masks = dict_get_or<std::vector<uint64_t>>(
        options,
        "pattern_masks",
        run.pattern_masks);
    run.success_shifts = dict_get_or<std::vector<uint8_t>>(
        options,
        "success_shifts",
        run.success_shifts);
    run.canonical_symm_mode = dict_get_or<int>(
        options,
        "canonical_symm_mode",
        run.canonical_symm_mode);
    run.spawn_rate4 = dict_get_or<double>(options, "spawn_rate4", run.spawn_rate4);
    run.num_threads = dict_get_or<int>(options, "threads", run.num_threads);
    run.family_modulus = dict_get_or<uint32_t>(options, "family_modulus", run.family_modulus);
    run.solve_route = parse_bc_solve_route_binding(
        dict_get_or<std::string>(options, "solve_route", "auto"));
    run.direct_queue_depth = dict_get_or<uint32_t>(
        options,
        "direct_queue_depth",
        run.direct_queue_depth);
    run.direct_io_chunk_mib = dict_get_or<uint32_t>(
        options,
        "direct_io_chunk_mib",
        run.direct_io_chunk_mib);
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
        .def_rw("cold_pathnames", &RunOptions::cold_pathnames)
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

    nb::class_<BCBookReader>(m, "BCBookReader")
        .def(nb::init<PatternSpec, uint32_t, bool>(), "pattern_spec"_a, "target_rank"_a, "is_variant"_a = false)
        .def(
            "move_on_dic",
            [](BCBookReader &reader,
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
            &BCBookReader::get_random_state,
            "path_list"_a,
            "pattern_full"_a,
            "spawn_rate4"_a,
            "nums_adjust"_a = 0
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
            const uint32_t generation_resume_count =
                dict_get_or<uint32_t>(options, "generation_resume_count", 0U);
            const std::filesystem::path solve_stats_csv =
                dict_get_or<std::string>(options, "solve_stats_csv", "");
            const std::filesystem::path solve_summary_csv =
                dict_get_or<std::string>(options, "solve_summary_csv", "");

            BC::BCFamilyGenerationRunResult generation_result;
            BC::BCFamilySolveRunResult solve_result;
            {
                nb::gil_scoped_release release;
                FormationProgress::reset_build_progress(progress_total);
                uint32_t generation_progress = expected_layers == 0U
                    ? generation_resume_count
                    : std::min(generation_resume_count, expected_layers);
                uint32_t solve_progress = 0U;

                if (skip_generation) {
                    generation_progress = expected_layers;
                    FormationProgress::update_build_progress(
                        generation_progress,
                        progress_total);
                    generation_result.completed = true;
                } else {
                    if (progress_total != 0U && generation_progress != 0U) {
                        FormationProgress::update_build_progress(
                            generation_progress,
                            progress_total);
                    }
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

                if (!solve_stats_csv.empty()) {
                    BC::ensure_bc_solve_stats_csv_file(solve_stats_csv);
                }

                solve_result = BC::bc_family_solve_full_run(
                    solve_options,
                    [&](const BC::BCFamilySolveRunLayerMetric &metric) {
                        BC::append_bc_solve_stats_csv_row(solve_stats_csv, metric);
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
                BC::append_bc_solve_stats_csv_total_row(solve_stats_csv, solve_result);

                if (!solve_summary_csv.empty()) {
                    if (!solve_summary_csv.parent_path().empty()) {
                        std::filesystem::create_directories(solve_summary_csv.parent_path());
                    }
                    std::ofstream summary(solve_summary_csv);
                    if (!summary) {
                        throw std::runtime_error("failed to open BC solve summary CSV");
                    }
                    BC::write_bc_solve_summary(summary, generation_result, solve_result);
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
        "success_rate_dtype"_a = "uint32",
        "output_book_path"_a = ""
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
                result = BCRuntime::lookup_compressed_result_cached(
                    compressed_path,
                    target_rank,
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
                result = BCRuntime::lookup_exact_result_cached(
                    position_path,
                    success_path,
                    target_rank,
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
                ok = BCRuntime::sample_compressed_result_cached(
                    compressed_path,
                    target_rank,
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
                board = BCRuntime::sample_exact_random_board_cached(position_path, target_rank);
            }
            return board != 0ULL ? gen_new_num(board, static_cast<float>(spawn_rate4)).first : 0ULL;
        },
        "position_path"_a,
        "target_rank"_a = 8U,
        "spawn_rate4"_a = 0.1
    );
}
