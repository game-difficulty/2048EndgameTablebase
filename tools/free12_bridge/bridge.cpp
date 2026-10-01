// Temporary free11 -> free12-4096 bridge. All board and value work is native.
#include "BookSolver.h"
#include "EXADBuilder.h"
#include "EXADSolvedLayer.h"
#include "Calculator.h"
#include <iostream>
#include <numeric>
#include <omp.h>

// BookGeneratorEXAD.cpp also defines an unused build-and-solve wrapper. Keep
// this standalone helper independent of Python/nanobind and fail if that wrapper
// is accidentally used. run.py calls the actual production solver binding.
void run_pattern_solve_exad_cpp(const std::vector<uint64_t> &, const AdvancedPatternSpec &, const RunOptions &) {
    throw std::logic_error("Use run.py solve for the production EXAD backward solver");
}

namespace {
constexpr uint64_t seed12 = 0x011111111111ffffULL;
constexpr uint32_t base12 = 4U * 32768U + 22U;
constexpr uint32_t base11 = 5U * 32768U + 20U;
constexpr uint32_t scale = 4000000000U;
constexpr size_t slot5 = bucket_to_index(5);

void require(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}

AdvancedPatternSpec spec12(uint32_t stsl, uint64_t signature) {
    AdvancedPatternSpec s;
    s.name = "free12";
    s.target = 12;
    s.num_free_32k = 4;
    s.small_tile_sum_limit = stsl;
    s.symm_mode = static_cast<int>(SymmMode::Full);
    for (uint8_t i = 0; i < 64; i += 4) s.success_shifts.push_back(i);
    s.logical_pattern_signature = s.physical_pattern_signature = signature;
    return s;
}

RunOptions options12(const std::string &prefix, int last, int threads) {
    RunOptions o;
    o.target = 12;
    o.steps = last + 3; // Boundary last must be below steps-2 in the solver.
    o.docheck_step = 2048 - (base12 % 4096) / 2;
    o.pathname = prefix;
    o.is_free = true;
    o.num_threads = threads;
    o.deletion_threshold = 0.05;
    return o;
}

EXAD::Luts target_lut(const std::string &prefix, const AdvancedPatternSpec &s, int threads) {
    PatternSpec p;
    p.name = s.name;
    p.success_shifts = s.success_shifts;
    p.symm_mode = s.symm_mode;
    const auto config = EXAD::make_exad_lut_tile_limit_config(12, {seed12}, p, true, false);
    const auto path = EXAD::lut_file_path(prefix);
    if (NativePath::exists(path)) {
        auto lut = EXAD::read_lut_file(path);
        require(ZMaskFrozen::tile_limit_configs_equal(lut.config, config) &&
                lut.physical_transform == 0 && lut.inverse_physical_transform == 0 &&
                lut.logical_pattern_signature == s.logical_pattern_signature &&
                lut.physical_pattern_signature == s.physical_pattern_signature,
                "Existing target LUT does not match this run");
        return lut;
    }
    auto lut = EXAD::build_luts(config, threads);
    lut.logical_pattern_signature = s.logical_pattern_signature;
    lut.physical_pattern_signature = s.physical_pattern_signature;
    EXAD::write_lut_file(path + ".bridge-writing", lut);
    FileIOUtils::finalize_temporary_file(path + ".bridge-writing", path);
    return lut;
}

template<class Layer>
void check_metadata(const Layer &layer, const EXAD::Luts &lut) {
    require(layer.lut_signature == lut.config_signature &&
            layer.physical_transform == 0 && layer.inverse_physical_transform == 0 &&
            layer.logical_pattern_signature == lut.logical_pattern_signature &&
            layer.physical_pattern_signature == lut.physical_pattern_signature,
            "Layer/LUT metadata mismatch (only identity physical transform is supported)");
}

EXAD::SolvedLayer<uint32_t> read_solved(const std::string &path, const EXAD::Luts &lut) {
    auto layer = EXAD::read_solved_layer_file<uint32_t>(path, EXAD::DTypeMode::UInt32);
    require(EXAD::solved_serialized_size(layer) == NativePath::file_size(path),
            "Truncated or overlong solved file: " + path);
    check_metadata(layer, lut);
    return layer;
}

// Each callback receives the persisted row ordinal; never infer it from a
// sorted board array or collapse/reorder AD lanes.
template<class Fn>
void visit_set(const EXAD::BoardSet &set, const EXAD::Luts &lut, int threads, Fn fn) {
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads)
    for (int64_t b = 0; b < static_cast<int64_t>(set.buckets.size()); ++b) {
        const auto &bucket = set.buckets[b];
        const auto group = EXAD::lut_group_index(EXAD::bucket_key_semantic_sum(bucket.key));
        const auto count = lut.size_table[group];
        const auto unrank = lut.offset_table[group];
        const auto prefix = EXAD::bucket_key_prefix36(bucket.key) << 28U;
        uint64_t row = bucket.dense_offset;
        const bool small = count <= set.threshold_bits;
        const auto units = small ? ZMaskFrozen::bytes_for_bits(count) : ZMaskFrozen::words_for_bits(count);
        for (uint32_t u = 0; u < units; ++u) {
            uint64_t bits = small ? set.small_bitmap_bytes[bucket.bitmap_offset + u]
                                  : set.large_bitmap_words[bucket.bitmap_offset + u];
            while (bits) {
                uint32_t bit = 0;
#if defined(__GNUC__) || defined(__clang__)
                bit = static_cast<uint32_t>(__builtin_ctzll(bits));
#else
                while (((bits >> bit) & 1ULL) == 0) ++bit;
#endif
                const auto rank = u * (small ? 8U : 64U) + bit;
                if (rank >= count) break;
                fn(prefix | lut.unrank_array[unrank + rank], row++);
                bits &= bits - 1;
            }
        }
    }
}

void write_temp(const std::string &path, const EXAD::Layer &layer) {
    EXAD::write_layer_file(path + ".bridge-writing", layer);
    FileIOUtils::finalize_temporary_file(path + ".bridge-writing", path);
}

void write_solved(const std::string &path, const EXAD::SolvedLayer<uint32_t> &layer) {
    EXAD::write_solved_layer_file(path + ".bridge-writing", layer);
    FileIOUtils::finalize_temporary_file(path + ".bridge-writing", path);
}

void prepare(const std::string &source, const std::string &source_lut,
             const std::string &prefix, int step, const AdvancedPatternSpec &spec,
             int threads, size_t sample_limit) {
    auto lut = target_lut(prefix, spec, threads);
    std::vector<uint64_t> boards;
    uint64_t kept = 0;
    {
        const auto src_lut = EXAD::read_lut_file(source_lut);
        auto src = read_solved(source, src_lut);
        require(src.original_board_sum - 32768U + 1024U == base12 + 2U * step,
                "Seed layer sum does not match the target step");
        boards.resize(src.live_board_count, 0);
        for (size_t slot = 0; slot < src.sets.size(); ++slot) {
            if (src.sets[slot].empty()) continue;
            require(src.row_width[slot] == EXAD::solved_derive_size_for_bucket(
                bucket_key_min() + static_cast<int>(slot), 5), "Unexpected source AD row width");
            visit_set(src.sets[slot], src_lut, threads, [&](uint64_t board, uint64_t row) {
                const auto *v = EXAD::row_ptr(src, slot, row);
                if (EXAD::row_has_value_above(v, src.row_width[slot], 1600000000U))
                    boards[src.slot_row_base[slot] + row] = board;
            });
        }
        boards.erase(std::remove(boards.begin(), boards.end(), 0ULL), boards.end());
        kept = boards.size();
        std::cout << "seed step=" << step << " input_rows=" << src.live_board_count
                  << " kept_max_gt_0.4=" << kept << std::endl;
    }
    // Only the smoke test uses a cap, in its own directory.
    if (sample_limit && boards.size() > sample_limit) boards.resize(sample_limit);
    auto masker = FormationAD::init_masker(spec);
    auto layer = EXAD::build_layer_from_boards(boards, base12 + 2U * step,
        masker.tiles_combination_table, masker.param, lut, threads);
    require(layer.live_board_count == boards.size(), "Seed rebuild lost or duplicated masked rows");
    write_temp(EXAD::layer_file_path(prefix, step), layer);
    // A valid empty layer0 prevents the standard generator from expanding seeds.
    const auto zero_path = EXAD::layer_file_path(prefix, 0);
    if (!NativePath::exists(zero_path)) {
        auto zero = EXAD::build_layer_from_boards({}, base12, masker.tiles_combination_table,
                                                masker.param, lut, threads);
        write_temp(zero_path, zero);
    }
    std::cout << "prepared rows=" << layer.live_board_count << " original_sum="
              << layer.original_board_sum << std::endl;
}

void boundary(const std::string &source, const std::string &source_lut,
              const std::string &prefix, int step, const AdvancedPatternSpec &spec, int threads) {
    const auto lut = target_lut(prefix, spec, threads);
    const auto src_lut = EXAD::read_lut_file(source_lut);
    auto src = read_solved(source, src_lut);
    require(src.original_board_sum == base12 + 2U * step + 32768U - 2048U,
            "Boundary source/target sums are not aligned");
    for (size_t slot = 0; slot < src.sets.size(); ++slot)
        require(src.sets[slot].empty() || (slot == slot5 && src.row_width[slot] == 1),
                "Boundary source is no longer scalar; this shortcut cannot be used");
    EXAD::build_direct_indexes(src, src_lut);
    EXAD::Layer layer;
    uint64_t input_rows = 0;
    {
        EXAD::LayerSlotReader in(EXAD::layer_file_path(prefix, step));
        const auto info = in.info();
        check_metadata(info, lut);
        require(info.original_board_sum == base12 + 2U * step, "Boundary target sum mismatch");
        layer.original_board_sum = info.original_board_sum;
        layer.threshold_bits = info.threshold_bits;
        layer.lut_signature = info.lut_signature;
        layer.logical_pattern_signature = info.logical_pattern_signature;
        layer.physical_pattern_signature = info.physical_pattern_signature;
        input_rows = info.live_board_count;
        for (size_t slot = 0; slot < bucket_slot_count(); ++slot) {
            EXAD::BoardSet set;
            require(in.read_next(set), "Missing target slot");
            if (slot == slot5) layer.sets[slot] = std::move(set);
        }
    }
    // Only five-F rows can match these scalar source layers. The other slots
    // are zero and are discarded before allocating their wide success arrays.
    layer.live_board_count = layer.sets[slot5].live_board_count;
    const auto param = FormationAD::build_mask_param(spec);
    auto out = EXAD::make_solved_layer_from_generation<uint32_t>(std::move(layer), param,
                                                               EXAD::DTypeMode::UInt32, 0);
    EXAD::fill_success_values(out.success_values, 0U, threads);
    std::vector<uint64_t> invalid_counts(threads), found_counts(threads);
    visit_set(out.sets[slot5], lut, threads, [&](uint64_t board, uint64_t row) {
        const auto stats = FormationAD::tile_sum_and_32k_count(board, param);
        if (stats.count_32k != 5 || out.original_board_sum - 4U * 32768U - stats.total_sum != 2048U) {
            ++invalid_counts[omp_get_thread_num()];
            return;
        }
        if (const auto *v = EXAD::lookup_row_ptr(src, src_lut, 5, board)) {
            std::fill_n(EXAD::row_ptr(out, slot5, row), 5, *v);
            ++found_counts[omp_get_thread_num()];
        }
    });
    const auto invalid = std::accumulate(invalid_counts.begin(), invalid_counts.end(), uint64_t{0});
    const auto found = std::accumulate(found_counts.begin(), found_counts.end(), uint64_t{0});
    require(invalid == 0, "Unexpected exposed tile or non-2048 row in boundary slot5");
    require(out.row_width[slot5] == 5, "Target boundary must have five lanes");
    const auto candidates = out.live_board_count;
    out = EXAD::compact_solved_layer(out, lut, 0U, threads);
    write_solved(EXAD::solved_file_path(prefix, step), out);
    std::cout << "boundary step=" << step << " input_rows=" << input_rows
              << " merged_2048=" << candidates << " source_found=" << found
              << " positive_rows=" << out.live_board_count << " lanes=5" << std::endl;
}

// Tiny deterministic fixtures exercise real generation, boundary conversion,
// missing/zero/unmerged rows, and the Python production backward solver.
void fixtures(const std::string &prefix, const AdvancedPatternSpec &spec, int threads) {
    const auto lut = target_lut(prefix, spec, threads);
    auto masker = FormationAD::init_masker(spec);
    auto make = [&](const std::vector<uint64_t> &boards, int step) {
        return EXAD::build_layer_from_boards(boards, base12 + 2U * step,
                    masker.tiles_combination_table, masker.param, lut, threads);
    };
    const uint64_t initial = Calculator::canonical_full(0x01111111111fffffULL);
    write_temp(EXAD::layer_file_path(prefix, 0), make({}, 0));
    write_temp(EXAD::layer_file_path(prefix, 1023), make({initial}, 1023));
    write_temp(EXAD::layer_file_path(prefix, 1024), make({}, 1024));
    ensure_exad_temp_through_cpp({seed12}, spec, options12(prefix, 1026, threads), 1026);

    PatternSpec base;
    base.name = "free11";
    base.symm_mode = static_cast<int>(SymmMode::Full);
    base.success_shifts = spec.success_shifts;
    auto sl = EXAD::build_luts(EXAD::make_exad_lut_tile_limit_config(11,
                    {0x01111111111fffffULL}, base, true, false), threads);
    sl.logical_pattern_signature = sl.physical_pattern_signature = 77;
    const auto sl_path = prefix + "fixture-source.exadlut";
    EXAD::write_lut_file(sl_path, sl);
    auto ss = spec;
    ss.name = "free11";
    ss.num_free_32k = 5;
    ss.target = 11;
    auto sm = FormationAD::init_masker(ss);
    for (int step : {1025, 1026}) {
        auto generated = EXAD::read_layer_file(EXAD::layer_file_path(prefix, step));
        std::vector<uint64_t> boards;
        EXAD::for_each_live_board(generated, lut, [&](int8_t key, uint64_t board) {
            require(key == 5, "Fixture unexpectedly derived");
            boards.push_back(board);
        });
        require(boards.size() > 3, "Fixture generation incomplete");
        auto sg = EXAD::build_layer_from_boards(boards, generated.original_board_sum + 32768U - 2048U,
                            sm.tiles_combination_table, sm.param, sl, threads);
        auto solved = EXAD::make_solved_layer_from_generation<uint32_t>(std::move(sg), sm.param,
                                                                        EXAD::DTypeMode::UInt32, 0);
        const uint32_t rate = step == 1025 ? 3200000000U : 2800000000U;
        EXAD::fill_success_values(solved.success_values, rate, threads);
        const auto source = prefix + "fixture-source-" + std::to_string(step) + ".exadbook";

        if (step == 1025) {
            // One stored zero, one absent row, plus an unmerged pair of 1024s.
            const auto absent = boards.front();
            EXAD::build_direct_indexes(solved, sl);
            auto lookup = EXAD::lookup_row(solved, sl, 5, absent);
            *EXAD::row_ptr(solved, slot5, lookup.local_row) = 0;
            solved = EXAD::compact_solved_layer(solved, sl, 0U, threads);
            solved.success_values[0] = 0;
            write_solved(source, solved);
            boards.push_back(Calculator::canonical_full(0xffffaa2222220000ULL));
            auto mixed = make(boards, step);
            require(mixed.live_board_count == boards.size(), "Fixture unmerged row missing");
            write_temp(EXAD::layer_file_path(prefix, step), mixed);
            boundary(source, sl_path, prefix, step, spec, threads);
            auto result = read_solved(EXAD::solved_file_path(prefix, step), lut);
            require(result.live_board_count + 3 == boards.size(), "Missing/zero/unmerged filtering failed");
            for (auto value : result.success_values) require(value == rate, "Boundary five-lane fill failed");
            // Restore complete constant futures for a simple independent oracle.
            write_temp(EXAD::layer_file_path(prefix, step), generated);
            boards.pop_back();
            auto restored = EXAD::build_layer_from_boards(boards, generated.original_board_sum + 32768U - 2048U,
                                sm.tiles_combination_table, sm.param, sl, threads);
            solved = EXAD::make_solved_layer_from_generation<uint32_t>(std::move(restored), sm.param,
                                                                        EXAD::DTypeMode::UInt32, 0);
            EXAD::fill_success_values(solved.success_values, rate, threads);
        }
        write_solved(source, solved);
        boundary(source, sl_path, prefix, step, spec, threads);
    }
    const uint32_t below[] = {0U, 1600000000U};
    const uint32_t above[] = {0U, 1600000001U};
    require(!EXAD::row_has_value_above(below, 2, 1600000000U) &&
             EXAD::row_has_value_above(above, 2, 1600000000U), "Strict row-max filter failed");
    std::cout << "fixture boundary checks passed" << std::endl;
}
} // namespace

int main(int argc, char **argv) {
    try {
        require(argc >= 7, "Usage: free12_bridge MODE PREFIX STEP STSL SIGNATURE THREADS [SOURCE LUT [SAMPLE_LIMIT]]");
        const std::string mode = argv[1], prefix = argv[2];
        const int step = std::stoi(argv[3]), threads = std::stoi(argv[6]);
        require(step >= 0 && threads > 0, "Invalid step/threads");
        omp_set_num_threads(threads);
        const auto spec = spec12(std::stoul(argv[4]), std::stoull(argv[5]));
        if (mode == "prepare" || mode == "boundary") {
            require(argc >= 9, "Missing source file/LUT");
            if (mode == "prepare") prepare(argv[7], argv[8], prefix, step, spec, threads,
                                           argc > 9 ? std::stoull(argv[9]) : 0);
            else boundary(argv[7], argv[8], prefix, step, spec, threads);
        } else if (mode == "generate") {
            auto options = options12(prefix, step, threads);
            ensure_exad_temp_through_cpp({seed12}, spec, options, step);
        } else if (mode == "prune") {
            auto lut = target_lut(prefix, spec, threads);
            const auto path = EXAD::solved_file_path(prefix, step);
            auto layer = read_solved(path, lut);
            layer = EXAD::compact_solved_layer(layer, lut, scale / 20U, threads);
            write_solved(path, layer);
        } else if (mode == "fixtures") {
            fixtures(prefix, spec, threads);
        } else throw std::runtime_error("Unknown mode: " + mode);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "bridge error: " << e.what() << std::endl;
        return 1;
    }
}
