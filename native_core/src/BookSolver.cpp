#include "BookSolver.h"

#include "AdaptiveIndex.h"
#include "BookGenerator.h"
#include "BoardMover.h"
#include "Calculator.h"
#include "CompressionBridge.h"
#include "FileIOUtils.h"
#include "Formation.h"
#include "HybridSearch.h"
#include "VBoardMover.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <nanobind/nanobind.h>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <sstream>

namespace fs = std::filesystem;
namespace nb = nanobind;

namespace {

template <typename T> using LayerVector = std::vector<SuccessEntry<T>>;

template <typename T> struct SplitLayer {
    std::vector<uint64_t> boards;
    std::unique_ptr<T[]> success;
    size_t length = 0;

    [[nodiscard]] bool empty() const {
        return length == 0U;
    }

    [[nodiscard]] size_t size() const {
        return length;
    }
};

template <typename T> SplitLayer<T> allocate_split_layer(size_t size) {
    SplitLayer<T> layer;
    layer.boards.resize(size);
    if (size > 0U) {
        layer.success = std::unique_ptr<T[]>(new T[size]);
    }
    layer.length = size;
    return layer;
}

template <typename T> SplitLayer<T> make_split_layer_from_raw(std::vector<uint64_t> boards) {
    SplitLayer<T> layer;
    layer.length = boards.size();
    layer.boards = std::move(boards);
    if (layer.length > 0U) {
        layer.success = std::unique_ptr<T[]>(new T[layer.length]);
    }
    return layer;
}

template <typename T> SplitLayer<T> split_layer_from_entries(const LayerVector<T> &entries) {
    SplitLayer<T> layer = allocate_split_layer<T>(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        layer.boards[i] = entries[i].board;
        layer.success[i] = entries[i].success;
    }
    return layer;
}

template <typename T> constexpr size_t io_chunk_entries() {
    constexpr size_t target_bytes = 1U << 20U;
    constexpr size_t chunk = target_bytes / sizeof(SuccessEntry<T>);
    return chunk > 0U ? chunk : 1U;
}

double wall_time_seconds() {
    return omp_get_wtime();
}

double round_to_2(double value) {
    return std::round(value * 100.0) / 100.0;
}

void debug_log(const std::string &message) {
    try {
        nb::gil_scoped_acquire acquire;
        nb::module_::import_("Config").attr("logger").attr("debug")(nb::str(message.c_str()));
    } catch (...) {
    }
}

void log_recalculate_performance(int step_index, double t0, double t1, double t2, double t3, size_t length) {
    if (t3 <= t0) {
        return;
    }
    double total = t3 - t0;
    std::ostringstream speed_stream;
    speed_stream << "step " << step_index << " recalculated: "
                 << round_to_2(static_cast<double>(length) / total / 1e6)
                 << " mbps";
    debug_log(speed_stream.str());

    std::ostringstream phase_stream;
    phase_stream << "index/solve/remove: "
                 << round_to_2((t1 - t0) / total) << "/"
                 << round_to_2((t2 - t1) / total) << "/"
                 << round_to_2((t3 - t2) / total);
    debug_log(phase_stream.str());
}

int effective_num_threads(const RunOptions &options) {
    if (options.num_threads > 0) {
        return options.num_threads;
    }
    return std::max(4, std::min(32, omp_get_max_threads()));
}

uint64_t apply_canonical(uint64_t board, int symm_mode) {
    switch (static_cast<SymmMode>(symm_mode)) {
        case SymmMode::Full:
            return Calculator::canonical_full(board);
        case SymmMode::Diagonal:
            return Calculator::canonical_diagonal(board);
        case SymmMode::Horizontal:
            return Calculator::canonical_horizontal(board);
        case SymmMode::Min33:
            return Calculator::canonical_min33(board);
        case SymmMode::Min24:
            return Calculator::canonical_min24(board);
        case SymmMode::Min34:
            return Calculator::canonical_min34(board);
        case SymmMode::Identity:
        default:
            return Calculator::canonical_identity(board);
    }
}

std::string now_string() {
    std::time_t now = std::time(nullptr);
    std::tm local_time{};
#ifdef _WIN32
    localtime_s(&local_time, &now);
#else
    localtime_r(&now, &local_time);
#endif
    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

template <typename T> LayerVector<T> read_layer_file(const std::string &path) {
    return FileIOUtils::read_binary_vector<SuccessEntry<T>>(path);
}

template <typename T> SplitLayer<T> read_split_layer_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }
    const size_t size = static_cast<size_t>(file.tellg());
    if (size == 0U) {
        return {};
    }
    file.seekg(0, std::ios::beg);
    const size_t count = size / sizeof(SuccessEntry<T>);
    SplitLayer<T> layer = allocate_split_layer<T>(count);
    std::unique_ptr<SuccessEntry<T>[]> buffer(new SuccessEntry<T>[io_chunk_entries<T>()]);

    size_t offset = 0U;
    while (offset < count) {
        const size_t current = std::min(io_chunk_entries<T>(), count - offset);
        const size_t bytes = current * sizeof(SuccessEntry<T>);
        FileIOUtils::read_exact(file, buffer.get(), bytes, path);
        for (size_t j = 0; j < current; ++j) {
            layer.boards[offset + j] = buffer[j].board;
            layer.success[offset + j] = buffer[j].success;
        }
        offset += current;
    }
    return layer;
}

template <typename T> void write_layer_file(const std::string &path, const LayerVector<T> &data) {
    FileIOUtils::write_binary_vector(path, data);
}

template <typename T> void write_layer_file(const std::string &path, const SplitLayer<T> &data) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        return;
    }
    if (data.empty()) {
        return;
    }
    std::unique_ptr<SuccessEntry<T>[]> buffer(new SuccessEntry<T>[io_chunk_entries<T>()]);
    size_t offset = 0U;
    while (offset < data.size()) {
        const size_t current = std::min(io_chunk_entries<T>(), data.size() - offset);
        for (size_t j = 0; j < current; ++j) {
            buffer[j].board = data.boards[offset + j];
            buffer[j].success = data.success[offset + j];
        }
        FileIOUtils::write_exact(file, buffer.get(), current * sizeof(SuccessEntry<T>), path);
        offset += current;
    }
}

std::vector<uint64_t> read_raw_file(const std::string &path) {
    return FileIOUtils::read_binary_vector<uint64_t>(path);
}

template <typename T> LayerVector<T> final_situation_process(
    const std::vector<uint64_t> &boards,
    const PatternSpec &spec,
    int target
) {
    LayerVector<T> result(boards.size());
    T max_scale = max_scale_value<T>();
    int num_threads = std::max(1, omp_get_max_threads());
    #pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < static_cast<int64_t>(boards.size()); ++i) {
        result[static_cast<size_t>(i)].board = boards[static_cast<size_t>(i)];
        result[static_cast<size_t>(i)].success =
            is_success_by_shifts(boards[static_cast<size_t>(i)], target, spec.success_shifts) ? max_scale : zero_value<T>();
    }
    return result;
}

template <typename T> std::pair<PatternLayer, PatternLayer> final_steps(
    bool started,
    const std::vector<uint64_t> &d0,
    const std::vector<uint64_t> &d1,
    const PatternSpec &spec,
    const RunOptions &options
) {
    LayerVector<T> layer0;
    LayerVector<T> layer1;
    if (started) {
        layer0 = final_situation_process<T>(d0, spec, options.target);
        layer1 = final_situation_process<T>(d1, spec, options.target);
        write_layer_file(options.pathname + std::to_string(options.steps - 2) + ".book", layer0);
        write_layer_file(options.pathname + std::to_string(options.steps - 1) + ".book", layer1);
    }
    auto raw_path = options.pathname + std::to_string(options.steps - 2);
    if (fs::exists(raw_path)) {
        fs::remove(raw_path);
    }
    return {make_pattern_layer(std::move(layer0)), make_pattern_layer(std::move(layer1))};
}

template <typename T> void ensure_stats_header(const RunOptions &options) {
    auto path = options.pathname + "stats.txt";
    if (fs::exists(path)) {
        return;
    }
    std::ofstream file(path, std::ios::app);
    file << "layer,length,max success rate,speed,deletion_threshold,time\n";
}

} // namespace

namespace {

template <typename T> AdaptiveIndex::Index create_index(const SplitLayer<T> &arr, int num_threads) {
    if (arr.empty()) {
        return {};
    }
    AdaptiveIndex::Config config;
    config.l2_split_threshold = 64U;
    config.l3_split_threshold = 64U;
    config.num_threads = num_threads;
    return AdaptiveIndex::build_experimental_hybrid(arr.boards.data(), static_cast<uint64_t>(arr.size()), config);
}

template <typename T> T binary_search_arr(
    const SplitLayer<T> &arr,
    T zero_val,
    uint64_t target,
    uint64_t low,
    uint64_t high
) {
    if (low > high) {
        return zero_val;
    }
    const size_t begin = static_cast<size_t>(low);
    const size_t length = static_cast<size_t>(high - low + 1U);
    const size_t pos = HybridSearch::exact_search(arr.boards.data() + begin, length, target);
    if (pos == HybridSearch::kNotFound) {
        return zero_val;
    }
    return arr.success[begin + pos];
}

template <typename T> T search_arr(
    const SplitLayer<T> &arr,
    uint64_t board,
    const AdaptiveIndex::Index *index,
    T zero_val
) {
    if (arr.empty()) {
        return zero_val;
    }
    if (!index || index->empty()) {
        return binary_search_arr(arr, zero_val, board, 0U, static_cast<uint64_t>(arr.size() - 1));
    }
    AdaptiveIndex::Range range = index->locate(board);
    if (range.empty()) {
        return zero_val;
    }
    return binary_search_arr(arr, zero_val, board, range.begin, range.end - 1U);
}

template <typename T> std::pair<T, uint64_t> search_arr2(
    const SplitLayer<T> &arr,
    uint64_t board,
    const AdaptiveIndex::Index *index,
    T zero_val
) {
    if (arr.empty()) {
        return {zero_val, 0U};
    }
    uint64_t low = 0U;
    uint64_t high = static_cast<uint64_t>(arr.size() - 1);
    if (index && !index->empty()) {
        AdaptiveIndex::Range range = index->locate(board);
        if (range.empty()) {
            return {zero_val, 0U};
        }
        low = range.begin;
        high = range.end - 1U;
    }

    if (low > high) {
        return {zero_val, 0U};
    }
    const size_t begin = static_cast<size_t>(low);
    const size_t length = static_cast<size_t>(high - low + 1U);
    const size_t pos = HybridSearch::exact_search(arr.boards.data() + begin, length, board);
    if (pos == HybridSearch::kNotFound) {
        return {zero_val, 0U};
    }
    return {arr.success[begin + pos], begin + pos};
}

template <typename T> size_t compact_live_entries(SplitLayer<T> &arr, T threshold, bool shrink_boards) {
    size_t count = 0;
    for (size_t i = 0; i < arr.size(); ++i) {
        if (arr.success[i] > threshold) {
            arr.boards[count] = arr.boards[i];
            arr.success[count] = arr.success[i];
            ++count;
        }
    }
    arr.length = count;
    if (shrink_boards) {
        arr.boards.resize(count);
    }
    return count;
}

template <typename T> size_t compact_by_mask(SplitLayer<T> &arr, const std::vector<uint8_t> &mask, bool shrink_boards) {
    size_t count = 0U;
    const size_t limit = std::min(arr.size(), mask.size());
    for (size_t i = 0; i < limit; ++i) {
        if (mask[i]) {
            arr.boards[count] = arr.boards[i];
            arr.success[count] = arr.success[i];
            ++count;
        }
    }
    arr.length = count;
    if (shrink_boards) {
        arr.boards.resize(count);
    }
    return count;
}

template <typename T, typename Mover>
void recalculate_layer(
    SplitLayer<T> &arr0,
    const SplitLayer<T> &arr1,
    const SplitLayer<T> &arr2,
    const PatternSpec &spec,
    const RunOptions &options,
    const AdaptiveIndex::Index *ind1,
    const AdaptiveIndex::Index *ind2,
    bool do_check
) {
    T max_scale = max_scale_value<T>();
    T zero_val = zero_value<T>();
    int num_threads = effective_num_threads(options);

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1024)
    for (int64_t k = 0; k < static_cast<int64_t>(arr0.size()); ++k) {
        uint64_t board = arr0.boards[static_cast<size_t>(k)];
        if (do_check && is_success_by_shifts(board, options.target, spec.success_shifts)) {
            arr0.success[static_cast<size_t>(k)] = max_scale;
            continue;
        }

        double success_probability = 0.0;
        int empty_slots = 0;
        for (int i = 0; i < 16; ++i) {
            if (((board >> (4 * i)) & 0xFULL) != 0) {
                continue;
            }
            ++empty_slots;

            uint64_t spawn2 = board | (1ULL << (4 * i));
            T best2 = zero_val;
            auto moves2 = Mover::move_all_dir(spawn2);
            uint64_t boards2[4] = {std::get<0>(moves2), std::get<1>(moves2), std::get<2>(moves2), std::get<3>(moves2)};
            for (uint64_t new_board : boards2) {
                if (new_board != spawn2 && is_pattern(new_board, spec.pattern_masks)) {
                    best2 = std::max(best2, search_arr(arr1, apply_canonical(new_board, spec.symm_mode), ind1, zero_val));
                }
            }
            success_probability += static_cast<double>(best2) * (1.0 - options.spawn_rate4);

            uint64_t spawn4 = board | (2ULL << (4 * i));
            T best4 = zero_val;
            auto moves4 = Mover::move_all_dir(spawn4);
            uint64_t boards4[4] = {std::get<0>(moves4), std::get<1>(moves4), std::get<2>(moves4), std::get<3>(moves4)};
            for (uint64_t new_board : boards4) {
                if (new_board != spawn4 && is_pattern(new_board, spec.pattern_masks)) {
                    best4 = std::max(best4, search_arr(arr2, apply_canonical(new_board, spec.symm_mode), ind2, zero_val));
                }
            }
            success_probability += static_cast<double>(best4) * options.spawn_rate4;
        }

        arr0.success[static_cast<size_t>(k)] =
            empty_slots > 0 ? static_cast<T>(success_probability / static_cast<double>(empty_slots)) : zero_val;
    }
}

} // namespace

namespace {

template <typename T>
bool handle_restart_recalculate(
    int i,
    SplitLayer<T> &d1,
    SplitLayer<T> &d2,
    bool &started,
    const RunOptions &options
) {
    auto path_i = options.pathname + std::to_string(i);
    if (fs::exists(path_i + ".book")) {
        debug_log("skipping step " + std::to_string(i));
        if (options.compress && !options.optimal_branch_only) {
            maybe_do_compress_classic(options.pathname + std::to_string(i + 2) + ".book", options.success_rate_dtype);
        }
        return false;
    }
    if (fs::exists(path_i + ".z") || fs::exists(path_i + ".book.7z")) {
        debug_log("skipping step " + std::to_string(i));
        return false;
    }
    if (!started) {
        started = true;
        if (i != options.steps - 3 || d1.empty() || d2.empty()) {
            d1 = read_split_layer_file<T>(options.pathname + std::to_string(i + 1) + ".book");
            d2 = read_split_layer_file<T>(options.pathname + std::to_string(i + 2) + ".book");
        }
    }
    return true;
}

template <typename T, typename Mover>
void recalculate_process_impl(
    SplitLayer<T> d1,
    SplitLayer<T> d2,
    const PatternSpec &spec,
    const RunOptions &options
) {
    ensure_stats_header<T>(options);
    bool started = false;
    AdaptiveIndex::Index ind1;
    T zero_val = zero_value<T>();
    T max_scale = max_scale_value<T>();
    bool has_index1 = false;
    const int index_threads = effective_num_threads(options);
    const uint32_t progress_total = classic_build_progress_total(options);
    const uint32_t solve_progress_total = build_progress_total(options);

    for (int i = options.steps - 3; i >= 0; --i) {
        if (!handle_restart_recalculate(i, d1, d2, started, options)) {
            continue;
        }
        FormationProgress::update_build_progress(
            solve_progress_total - static_cast<uint32_t>(i) - 2U,
            progress_total
        );

        if (options.compress_temp_files) {
            maybe_decompress_with_7z(options.pathname + std::to_string(i) + ".7z");
        }
        std::vector<uint64_t> raw_layer = read_raw_file(options.pathname + std::to_string(i));
        double t0 = wall_time_seconds();
        SplitLayer<T> d0 = make_split_layer_from_raw<T>(std::move(raw_layer));

        AdaptiveIndex::Index ind2;
        if (has_index1) {
            ind2 = std::move(ind1);
        } else if (d2.size() >= 100000) {
            ind2 = create_index(d2, index_threads);
        }
        if (d1.size() < 100000) {
            ind1 = {};
            has_index1 = false;
        } else {
            ind1 = create_index(d1, index_threads);
            has_index1 = true;
        }
        double t1 = wall_time_seconds();

        recalculate_layer<T, Mover>(d0, d1, d2, spec, options, ind1.empty() ? nullptr : &ind1, ind2.empty() ? nullptr : &ind2, i > options.docheck_step);
        size_t length = d0.size();
        double t2 = wall_time_seconds();
        compact_live_entries(d0, zero_val, true);
        double t3 = wall_time_seconds();
        log_recalculate_performance(i, t0, t1, t2, t3, length);

        if (!d0.empty()) {
            T max_success_raw = d0.success[0];
            for (size_t k = 1; k < d0.size(); ++k) {
                if (d0.success[k] > max_success_raw) {
                    max_success_raw = d0.success[k];
                }
            }
            double max_success = static_cast<double>(max_success_raw - zero_val) / static_cast<double>(max_scale - zero_val);
            std::ofstream stats(options.pathname + "stats.txt", std::ios::app);
            stats << i << "," << length << "," << max_success << ","
                  << round_to_2(static_cast<double>(length) / std::max(t3 - t0, 0.01) / 1e6) << " mbps,"
                  << options.deletion_threshold << "," << now_string() << "\n";
        }

        write_layer_file(options.pathname + std::to_string(i) + ".book", d0);
        auto raw_path = options.pathname + std::to_string(i);
        if (fs::exists(raw_path)) {
            fs::remove(raw_path);
        }

        if (options.deletion_threshold > 0.0) {
            T threshold = static_cast<T>(options.deletion_threshold * static_cast<double>(max_scale - zero_val) + static_cast<double>(zero_val));
            compact_live_entries(d2, threshold, false);
            write_layer_file(options.pathname + std::to_string(i + 2) + ".book", d2);
        }
        if (options.compress_temp_files && options.optimal_branch_only) {
            maybe_compress_with_7z(options.pathname + std::to_string(i + 2) + ".book");
        } else if (options.compress && !options.optimal_branch_only) {
            maybe_do_compress_classic(options.pathname + std::to_string(i + 2) + ".book", options.success_rate_dtype);
        }

        if (i > 0) {
            d2 = std::move(d1);
            d1 = std::move(d0);
        }
    }
}

} // namespace

namespace {

template <typename T, typename Mover>
void find_optimal_branches(
    const SplitLayer<T> &arr0,
    const SplitLayer<T> &arr1,
    std::vector<uint8_t> &result,
    const PatternSpec &spec,
    const AdaptiveIndex::Index *index,
    int new_value,
    T zero_val
) {
    for (size_t row = 0; row < arr0.size(); ++row) {
        uint64_t board = arr0.boards[row];
        for (int i = 0; i < 16; ++i) {
            if (((board >> (4 * i)) & 0xFULL) != 0) {
                continue;
            }
            uint64_t spawned = board | (static_cast<uint64_t>(new_value) << (4 * i));
            T best = zero_val;
            uint64_t best_index = 0;
            auto moved = Mover::move_all_dir(spawned);
            uint64_t boards[4] = {std::get<0>(moved), std::get<1>(moved), std::get<2>(moved), std::get<3>(moved)};
            for (uint64_t new_board : boards) {
                if (new_board == spawned || !is_pattern(new_board, spec.pattern_masks)) {
                    continue;
                }
                auto [rate, pos] = search_arr2(arr1, apply_canonical(new_board, spec.symm_mode), index, zero_val);
                if (rate > best) {
                    best = rate;
                    best_index = pos;
                }
            }
            if (best_index < result.size()) {
                result[best_index] = 1;
            }
        }
    }
}

template <typename T>
bool handle_restart_opt_only(
    int i,
    bool &started,
    SplitLayer<T> &d0,
    SplitLayer<T> &d1,
    const RunOptions &options
) {
    auto optlayer_path = options.pathname + "optlayer";
    if (started) {
        std::ofstream out(optlayer_path, std::ios::trunc);
        out << (i - 1);
        return true;
    }

    int current_layer = i - 1;
    std::ifstream in(optlayer_path);
    if (in) {
        in >> current_layer;
    }
    if (current_layer >= i) {
        return false;
    }
    if (i > 20 && d0.empty()) {
        const std::string d0_path = options.pathname + std::to_string(i - 2) + ".book";
        const std::string d1_path = options.pathname + std::to_string(i - 1) + ".book";
        if (!fs::exists(d0_path) || !fs::exists(d1_path)) {
            return false;
        }
        d0 = read_split_layer_file<T>(d0_path);
        d1 = read_split_layer_file<T>(d1_path);
        started = true;
        return true;
    }
    return false;
}

template <typename T, typename Mover>
void keep_only_optimal_branches_impl(const PatternSpec &spec, const RunOptions &options) {
    SplitLayer<T> d0;
    SplitLayer<T> d1;
    bool started = false;
    T zero_val = zero_value<T>();
    const uint32_t progress_total = classic_build_progress_total(options);
    const uint32_t solve_progress_total = build_progress_total(options);

    for (int i = 0; i < options.steps; ++i) {
        FormationProgress::update_build_progress(
            solve_progress_total + static_cast<uint32_t>(i) + 1U,
            progress_total
        );
        const bool process_step = handle_restart_opt_only(i, started, d0, d1, options);
        if (options.compress_temp_files) {
            maybe_decompress_with_7z(options.pathname + std::to_string(i) + ".book.7z");
        }
        if (i > 20 && process_step) {
            SplitLayer<T> d2 = read_split_layer_file<T>(options.pathname + std::to_string(i) + ".book");
            AdaptiveIndex::Index index = create_index(d2, effective_num_threads(options));
            std::vector<uint8_t> mask(d2.size(), 0);
            find_optimal_branches<T, Mover>(d0, d2, mask, spec, index.empty() ? nullptr : &index, 2, zero_val);
            find_optimal_branches<T, Mover>(d1, d2, mask, spec, index.empty() ? nullptr : &index, 1, zero_val);

            compact_by_mask(d2, mask, true);
            write_layer_file(options.pathname + std::to_string(i) + ".book", d2);
            d0 = std::move(d1);
            d1 = std::move(d2);
            debug_log("step " + std::to_string(i) + " retains only the optimal branch\n");
        }
        if (options.compress) {
            maybe_do_compress_classic(options.pathname + std::to_string(i - 2) + ".book", options.success_rate_dtype);
        }
    }

    if (options.compress) {
        maybe_do_compress_classic(options.pathname + std::to_string(options.steps - 2) + ".book", options.success_rate_dtype);
        maybe_do_compress_classic(options.pathname + std::to_string(options.steps - 1) + ".book", options.success_rate_dtype);
    }

    auto optlayer_path = options.pathname + "optlayer";
    if (fs::exists(optlayer_path)) {
        fs::remove(optlayer_path);
    }
}

template <typename T> std::tuple<bool, PatternLayer, PatternLayer> run_pattern_generate_impl(
    const std::vector<uint64_t> &arr_init,
    const PatternSpec &spec,
    const RunOptions &options
) {
    auto [started, d0, d1] = BookGenerator::generate_process(arr_init, spec, options);
    auto [layer0, layer1] = final_steps<T>(started, d0, d1, spec, options);
    return {started, std::move(layer0), std::move(layer1)};
}

template <typename T, typename Mover> void run_pattern_solve_impl(
    const PatternLayer &d1,
    const PatternLayer &d2,
    const PatternSpec &spec,
    const RunOptions &options
) {
    recalculate_process_impl<T, Mover>(
        split_layer_from_entries(layer_as<T>(d1)),
        split_layer_from_entries(layer_as<T>(d2)),
        spec,
        options
    );
    if (options.optimal_branch_only) {
        keep_only_optimal_branches_impl<T, Mover>(spec, options);
    }
}

} // namespace

std::tuple<bool, PatternLayer, PatternLayer> run_pattern_generate_cpp(
    const std::vector<uint64_t> &arr_init,
    const PatternSpec &spec,
    const RunOptions &options
) {
    switch (success_rate_kind_from_name(options.success_rate_dtype)) {
        case SuccessRateKind::UInt64:
            return run_pattern_generate_impl<uint64_t>(arr_init, spec, options);
        case SuccessRateKind::Float32:
            return run_pattern_generate_impl<float>(arr_init, spec, options);
        case SuccessRateKind::Float64:
            return run_pattern_generate_impl<double>(arr_init, spec, options);
        case SuccessRateKind::UInt32:
        default:
            return run_pattern_generate_impl<uint32_t>(arr_init, spec, options);
    }
}

void run_pattern_solve_cpp(
    const PatternLayer &d1,
    const PatternLayer &d2,
    const PatternSpec &spec,
    const RunOptions &options
) {
    switch (success_rate_kind_from_name(options.success_rate_dtype)) {
        case SuccessRateKind::UInt64:
            if (options.is_variant) {
                run_pattern_solve_impl<uint64_t, VBoardMover>(d1, d2, spec, options);
            } else {
                run_pattern_solve_impl<uint64_t, BoardMover>(d1, d2, spec, options);
            }
            break;
        case SuccessRateKind::Float32:
            if (options.is_variant) {
                run_pattern_solve_impl<float, VBoardMover>(d1, d2, spec, options);
            } else {
                run_pattern_solve_impl<float, BoardMover>(d1, d2, spec, options);
            }
            break;
        case SuccessRateKind::Float64:
            if (options.is_variant) {
                run_pattern_solve_impl<double, VBoardMover>(d1, d2, spec, options);
            } else {
                run_pattern_solve_impl<double, BoardMover>(d1, d2, spec, options);
            }
            break;
        case SuccessRateKind::UInt32:
        default:
            if (options.is_variant) {
                run_pattern_solve_impl<uint32_t, VBoardMover>(d1, d2, spec, options);
            } else {
                run_pattern_solve_impl<uint32_t, BoardMover>(d1, d2, spec, options);
            }
            break;
    }
}

void run_pattern_build_cpp(
    const std::vector<uint64_t> &arr_init,
    const PatternSpec &spec,
    const RunOptions &options
) {
    FormationProgress::reset_build_progress(classic_build_progress_total(options));
    auto [started, d1, d2] = run_pattern_generate_cpp(arr_init, spec, options);
    (void) started;
    run_pattern_solve_cpp(d1, d2, spec, options);
}
