#include <cstring>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "BookSolver.h"
#include "EXADCompressedResult.h"
#include "EXCompressedResult.h"
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

} // namespace

NB_MODULE(formation_core, m) {
    NDIAG_INSTALL();

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
            NDIAG_START_RUN("run_pattern_build_exad", spec, options, static_cast<uint64_t>(arr_init.shape(0)));
            try {
                run_pattern_build_exad_cpp(to_u64_vector(arr_init), spec, options);
            } catch (const std::exception &e) {
                NDIAG_EXCEPTION("run_pattern_build_exad", e.what());
                throw;
            } catch (...) {
                NDIAG_EXCEPTION("run_pattern_build_exad", "unknown exception");
                throw;
            }
        },
        "arr_init"_a,
        "pattern_spec"_a,
        "run_options"_a,
        nb::call_guard<nb::gil_scoped_release>()
    );

    m.def(
        "run_pattern_solve_exad",
        [](const U64Array &arr_init, const AdvancedPatternSpec &spec, const RunOptions &options) {
            NDIAG_START_RUN("run_pattern_solve_exad", spec, options, static_cast<uint64_t>(arr_init.shape(0)));
            try {
                run_pattern_solve_exad_cpp(to_u64_vector(arr_init), spec, options);
            } catch (const std::exception &e) {
                NDIAG_EXCEPTION("run_pattern_solve_exad", e.what());
                throw;
            } catch (...) {
                NDIAG_EXCEPTION("run_pattern_solve_exad", "unknown exception");
                throw;
            }
        },
        "arr_init"_a,
        "pattern_spec"_a,
        "run_options"_a,
        nb::call_guard<nb::gil_scoped_release>()
    );

    m.def(
        "run_pattern_build_zmask",
        [](const U64Array &arr_init, const PatternSpec &spec, const RunOptions &options) {
            NDIAG_START_RUN("run_pattern_build_zmask", spec, options, static_cast<uint64_t>(arr_init.shape(0)));
            try {
                run_pattern_build_zmask_cpp(to_u64_vector(arr_init), spec, options);
            } catch (const std::exception &e) {
                NDIAG_EXCEPTION("run_pattern_build_zmask", e.what());
                throw;
            } catch (...) {
                NDIAG_EXCEPTION("run_pattern_build_zmask", "unknown exception");
                throw;
            }
        },
        "arr_init"_a,
        "pattern_spec"_a,
        "run_options"_a,
        nb::call_guard<nb::gil_scoped_release>()
    );

    m.def(
        "run_pattern_solve_zmask",
        [](const U64Array &arr_init, const PatternSpec &spec, const RunOptions &options) {
            NDIAG_START_RUN("run_pattern_solve_zmask", spec, options, static_cast<uint64_t>(arr_init.shape(0)));
            try {
                run_pattern_solve_zmask_cpp(to_u64_vector(arr_init), spec, options);
            } catch (const std::exception &e) {
                NDIAG_EXCEPTION("run_pattern_solve_zmask", e.what());
                throw;
            } catch (...) {
                NDIAG_EXCEPTION("run_pattern_solve_zmask", "unknown exception");
                throw;
            }
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
}
