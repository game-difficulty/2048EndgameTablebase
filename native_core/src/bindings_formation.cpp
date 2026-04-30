#include <cstring>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "BookSolver.h"
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

} // namespace

NB_MODULE(formation_core, m) {
    nb::enum_<SymmMode>(m, "SymmMode")
        .value("Identity", SymmMode::Identity)
        .value("Full", SymmMode::Full)
        .value("Diagonal", SymmMode::Diagonal)
        .value("Horizontal", SymmMode::Horizontal)
        .value("Min33", SymmMode::Min33)
        .value("Min24", SymmMode::Min24)
        .value("Min34", SymmMode::Min34);

    nb::class_<PatternSpec>(m, "PatternSpec")
        .def(nb::init<>())
        .def_rw("name", &PatternSpec::name)
        .def_rw("pattern_masks", &PatternSpec::pattern_masks)
        .def_rw("success_shifts", &PatternSpec::success_shifts)
        .def_rw("symm_mode", &PatternSpec::symm_mode);

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
        .def_rw("symm_mode", &AdvancedPatternSpec::symm_mode)
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
}
