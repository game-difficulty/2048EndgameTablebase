#include "BookSolver.h"

#include "AdaptiveIndex.h"
#include "BoardMaskerAD.h"
#include "BoardMover.h"
#include "BoardMoverAD.h"
#include "Calculator.h"
#include "CompressionBridge.h"
#include "Formation.h"
#include "HybridSearch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <atomic>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif
#include <nanobind/nanobind.h>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace nb = nanobind;

namespace {

template <typename T> struct MatrixBucket {
    std::vector<T> data;
    size_t rows = 0;
    size_t cols = 0;

    void resize(size_t new_rows, size_t new_cols) {
        rows = new_rows;
        cols = new_cols;
        data.resize(rows * cols);
    }

    [[nodiscard]] bool empty() const {
        return rows == 0 || cols == 0 || data.empty();
    }

    [[nodiscard]] T *row(size_t index) {
        return data.data() + index * cols;
    }

    [[nodiscard]] const T *row(size_t index) const {
        return data.data() + index * cols;
    }
};

template <typename T> using BookStore = BucketStore<MatrixBucket<T>>;
using IndexStore = BucketStore<std::vector<uint64_t>>;
using PrefixStore = BucketStore<AdaptiveIndex::Index>;

constexpr uint64_t kPendingMatchIndex = 0xFFFFFFFFFFFFFFFEULL;

struct MatchCache {
    size_t map_length = 0;
    size_t derive_size = 0;
    std::unique_ptr<std::atomic<uint64_t>[]> match_index;
    std::vector<uint32_t> match_mat;

    MatchCache() = default;

    MatchCache(size_t map_length_, size_t derive_size_)
        : map_length(map_length_),
          derive_size(derive_size_),
          match_mat(map_length_ * derive_size_, 0U) {
        match_index = std::make_unique<std::atomic<uint64_t>[]>(map_length_);
        for (size_t i = 0; i < map_length_; ++i) {
            match_index[i].store(std::numeric_limits<uint64_t>::max(), std::memory_order_relaxed);
        }
    }

    MatchCache(MatchCache &&other) noexcept
        : map_length(other.map_length),
          derive_size(other.derive_size),
          match_index(std::move(other.match_index)),
          match_mat(std::move(other.match_mat)) {
        other.map_length = 0;
        other.derive_size = 0;
    }

    MatchCache &operator=(MatchCache &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        map_length = other.map_length;
        derive_size = other.derive_size;
        match_index = std::move(other.match_index);
        match_mat = std::move(other.match_mat);
        other.map_length = 0;
        other.derive_size = 0;
        return *this;
    }

    MatchCache(const MatchCache &) = delete;
    MatchCache &operator=(const MatchCache &) = delete;

    [[nodiscard]] uint32_t *row(size_t index) {
        return match_mat.data() + index * derive_size;
    }

    [[nodiscard]] uint64_t key(size_t index) const {
        return match_index[index].load(std::memory_order_acquire);
    }

    [[nodiscard]] bool try_claim_empty(size_t index) {
        uint64_t expected = std::numeric_limits<uint64_t>::max();
        return match_index[index].compare_exchange_strong(
            expected,
            kPendingMatchIndex,
            std::memory_order_acq_rel,
            std::memory_order_acquire
        );
    }

    void publish(size_t index, uint64_t key_value, const std::vector<uint32_t> &ranked_array) {
        std::copy(ranked_array.begin(), ranked_array.end(), row(index));
        match_index[index].store(key_value, std::memory_order_release);
    }
};

template <typename T>
MatchCache &get_shared_match_cache(
    std::unordered_map<uint32_t, MatchCache> &match_dict,
    size_t derive_size
);

template <typename T> struct AdSolveWorkspace {
    std::vector<uint64_t> derived_boards;
    std::vector<uint64_t> moved_boards;
    std::vector<uint64_t> matched_boards;
    std::array<uint8_t, 16> positions{};
    std::vector<size_t> matched_positions;
    std::vector<uint32_t> sorted_indices;
    std::vector<uint32_t> ranked_array;
    std::vector<T> optimal_values;
    std::vector<T> temp_values;
    std::vector<double> success_probability;
    std::vector<uint64_t> kept_indices;
    std::vector<T> kept_values;
};

constexpr size_t kNotFoundIndex = std::numeric_limits<size_t>::max();
constexpr uint64_t kEmptyMatchIndex = 0xFFFFFFFFFFFFFFFFULL;
constexpr std::array<uint8_t, 16> kPosRev = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
constexpr std::array<uint32_t, 19> kFactorials = {
    1U, 1U, 2U, 6U, 24U, 120U, 720U, 5040U, 40320U, 362880U,
    3628800U, 39916800U, 1U, 1U, 1U, 1U, 1U, 1U, 1U
};

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

uint64_t apply_sym_like(uint64_t board, int symm_index) {
    switch (symm_index) {
        case 1:
            return Calculator::ReverseLR(board);
        case 2:
            return Calculator::ReverseUD(board);
        case 3:
            return Calculator::ReverseUL(board);
        case 4:
            return Calculator::ReverseUR(board);
        case 5:
            return Calculator::Rotate180(board);
        case 6:
            return Calculator::RotateL(board);
        case 7:
            return Calculator::RotateR(board);
        case 0:
        default:
            return board;
    }
}

template <typename Transform>
inline void sym_arr_like_transform(std::vector<uint64_t> &boards, Transform transform) {
    uint64_t *data = boards.data();
    const ptrdiff_t count = static_cast<ptrdiff_t>(boards.size());
    #pragma omp simd
    for (ptrdiff_t i = 0; i < count; ++i) {
        data[i] = transform(data[i]);
    }
}

std::pair<uint64_t, int> apply_sym_pair(uint64_t board, int symm_mode) {
    switch (static_cast<SymmMode>(symm_mode)) {
        case SymmMode::Full:
            return Calculator::canonical_full_pair(board);
        case SymmMode::Diagonal:
            return Calculator::canonical_diagonal_pair(board);
        case SymmMode::Horizontal:
            return Calculator::canonical_horizontal_pair(board);
        case SymmMode::Identity:
        case SymmMode::Min33:
        case SymmMode::Min24:
        case SymmMode::Min34:
        default:
            return Calculator::canonical_identity_pair(board);
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

uint64_t available_memory_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<uint64_t>(status.ullAvailPhys);
    }
    return 0ULL;
#else
    struct sysinfo info {};
    if (sysinfo(&info) == 0) {
        return static_cast<uint64_t>(info.freeram) * static_cast<uint64_t>(info.mem_unit);
    }
    return 0ULL;
#endif
}

template <typename T> std::vector<T> read_binary_vector(const std::string &path) {
    std::vector<T> data;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return data;
    }
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    data.resize(size / sizeof(T));
    if (!data.empty()) {
        file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size));
    }
    return data;
}

template <typename T> void write_binary_vector(const std::string &path, const std::vector<T> &data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return;
    }
    if (!data.empty()) {
        out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    }
}

template <typename T> void ensure_stats_header(const RunOptions &options) {
    const std::string path = options.pathname + "stats.txt";
    if (fs::exists(path)) {
        return;
    }
    std::ofstream file(path, std::ios::app);
    file << "layer,length,max success rate,speed,deletion_threshold,time\n";
}

uint32_t board_sum(uint64_t board) {
    uint32_t result = 0;
    for (int shift = 0; shift < 64; shift += 4) {
        uint64_t value = (board >> static_cast<uint64_t>(shift)) & 0xFULL;
        if (value == 0ULL) {
            continue;
        }
        if (value == 0xFULL) {
            result += (1U << 15U);
        } else {
            result += (1U << static_cast<uint32_t>(value));
        }
    }
    return result;
}

size_t derive_size_for_bucket(int count_32k, uint8_t num_free_32k) {
    if (count_32k < 0) {
        return static_cast<size_t>(kFactorials[static_cast<size_t>(-count_32k - 2)] / kFactorials[num_free_32k]);
    }
    if (count_32k > 15) {
        return static_cast<size_t>(kFactorials[static_cast<size_t>(count_32k - 16)] / kFactorials[num_free_32k]);
    }
    return static_cast<size_t>(kFactorials[static_cast<size_t>(count_32k)] / kFactorials[num_free_32k]);
}

template <typename T> void clear_book_store(BookStore<T> &book_store) {
    for (int key = bucket_key_min(); key <= bucket_key_max(); ++key) {
        book_store.at(key) = MatrixBucket<T>{};
    }
}

void clear_index_store(IndexStore &index_store) {
    for (int key = bucket_key_min(); key <= bucket_key_max(); ++key) {
        index_store.at(key).clear();
    }
}

void clear_prefix_store(PrefixStore &prefix_store) {
    for (int key = bucket_key_min(); key <= bucket_key_max(); ++key) {
        prefix_store.at(key) = AdaptiveIndex::Index{};
    }
}

template <typename T>
void dict_tofile(
    const BookStore<T> &book_dict,
    const IndexStore &ind_dict,
    const RunOptions &options,
    int step,
    bool compress_indices
) {
    const fs::path folder_path = options.pathname + std::to_string(step) + "b";
    fs::create_directories(folder_path);

    for (int key = bucket_key_min(); key <= bucket_key_max(); ++key) {
        const fs::path ind_filename = folder_path / (std::to_string(key) + ".i");
        const fs::path book_filename = folder_path / (std::to_string(key) + ".b");
        const auto &indices = ind_dict.at(key);
        const auto &book_bucket = book_dict.at(key);
        if (indices.empty()) {
            if (fs::exists(ind_filename)) {
                fs::remove(ind_filename);
            }
            if (fs::exists(book_filename)) {
                fs::remove(book_filename);
            }
            continue;
        }
        write_binary_vector<uint64_t>(ind_filename.string(), indices);
        write_binary_vector<T>(book_filename.string(), book_bucket.data);
    }

    if (compress_indices && options.compress) {
        maybe_do_compress_ad(folder_path.string());
    }
}

template <typename T>
void dict_fromfile(
    const RunOptions &options,
    int step,
    BookStore<T> &book_dict,
    IndexStore &ind_dict
) {
    clear_book_store(book_dict);
    clear_index_store(ind_dict);

    const fs::path folder_path = options.pathname + std::to_string(step) + "b";
    if (!fs::exists(folder_path)) {
        throw std::runtime_error("Missing AD solve bucket folder: " + folder_path.string());
    }

    for (const auto &entry : fs::directory_iterator(folder_path)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const fs::path path = entry.path();
        const std::string ext = path.extension().string();
        if (ext == ".zi") {
            const int key = std::stoi(path.stem().string());
            auto decompressed = maybe_decompress_uint64_array(path.string());
            if (!decompressed.empty()) {
                ind_dict.at(key) = std::move(decompressed);
            }
            continue;
        }
        if (ext != ".i") {
            continue;
        }
        const int key = std::stoi(path.stem().string());
        ind_dict.at(key) = read_binary_vector<uint64_t>(path.string());
    }

    for (const auto &entry : fs::directory_iterator(folder_path)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const fs::path path = entry.path();
        if (path.extension() != ".b") {
            continue;
        }
        const int key = std::stoi(path.stem().string());
        const auto &indices = ind_dict.at(key);
        if (indices.empty()) {
            continue;
        }
        auto values = read_binary_vector<T>(path.string());
        MatrixBucket<T> bucket;
        bucket.rows = indices.size();
        bucket.cols = bucket.rows == 0 ? 0 : (values.empty() ? 0 : values.size() / bucket.rows);
        bucket.data = std::move(values);
        book_dict.at(key) = std::move(bucket);
    }
}

std::unique_ptr<PrefixStore> create_index_ad(const IndexStore &ind_dict, int num_threads) {
    auto result = std::make_unique<PrefixStore>();
    clear_prefix_store(*result);
    #pragma omp parallel for num_threads(num_threads)
    for (int key_offset = 0; key_offset < static_cast<int>(bucket_slot_count()); ++key_offset) {
        int key = bucket_key_min() + key_offset;
        const auto &arr = ind_dict.at(key);
        if (arr.size() <= 100000U) {
            continue;
        }
        AdaptiveIndex::Config config;
        config.l2_split_threshold = 64U;
        config.l3_split_threshold = 64U;
        config.num_threads = 1;
        result->at(key) = AdaptiveIndex::build_experimental_hybrid(arr, config);
    }
    return result;
}

std::vector<int> write_ind_chunked(const IndexStore &ind_dict, const RunOptions &options, int step) {
    const fs::path folder_path = options.pathname + std::to_string(step) + "bt";
    if (fs::exists(folder_path)) {
        fs::remove_all(folder_path);
    }
    fs::create_directories(folder_path);

    std::vector<int> keys;
    keys.reserve(bucket_slot_count());
    for (int key = bucket_key_min(); key <= bucket_key_max(); ++key) {
        const auto &indices = ind_dict.at(key);
        if (indices.empty()) {
            continue;
        }
        write_binary_vector<uint64_t>((folder_path / (std::to_string(key) + ".i")).string(), indices);
        keys.push_back(key);
    }
    return keys;
}

size_t binary_search_arr_ad(const std::vector<uint64_t> &arr, uint64_t target, uint64_t low, uint64_t high) {
    if (arr.empty()) {
        return kNotFoundIndex;
    }
    const size_t begin = static_cast<size_t>(low);
    const size_t end = static_cast<size_t>(high) + 1U;
    const size_t pos = HybridSearch::exact_search(arr.data() + begin, end - begin, target);
    if (pos == HybridSearch::kNotFound) {
        return kNotFoundIndex;
    }
    return begin + pos;
}

size_t search_arr_ad(const std::vector<uint64_t> &arr, uint64_t board, const AdaptiveIndex::Index &prefix_index) {
    if (arr.empty()) {
        return kNotFoundIndex;
    }
    if (prefix_index.empty()) {
        return binary_search_arr_ad(arr, board, 0U, static_cast<uint64_t>(arr.size() - 1U));
    }
    AdaptiveIndex::Range range = prefix_index.locate(board);
    if (range.empty()) {
        return kNotFoundIndex;
    }
    return binary_search_arr_ad(arr, board, range.begin, range.end - 1U);
}

template <typename T> std::pair<size_t, T> length_count(const BookStore<T> &book_dict) {
    size_t length = 0;
    T max_rate = zero_value<T>();
    for (int key = bucket_key_min(); key <= bucket_key_max(); ++key) {
        const auto &bucket = book_dict.at(key);
        length += bucket.rows * bucket.cols;
        for (const T value : bucket.data) {
            if (value > max_rate) {
                max_rate = value;
            }
        }
    }
    return {length, max_rate};
}

template <typename T>
void remove_died_ad(
    BookStore<T> &book_dict,
    IndexStore &ind_dict,
    T deletion_threshold,
    AdSolveWorkspace<T> &workspace
) {
    for (int key = bucket_key_min(); key <= bucket_key_max(); ++key) {
        auto &indices = ind_dict.at(key);
        auto &bucket = book_dict.at(key);
        if (indices.empty() || bucket.rows == 0 || bucket.cols == 0) {
            continue;
        }
        auto &kept_indices = workspace.kept_indices;
        auto &kept_values = workspace.kept_values;
        kept_indices.clear();
        kept_values.clear();
        kept_indices.reserve(indices.size());
        kept_values.reserve(bucket.data.size());
        for (size_t row = 0; row < bucket.rows; ++row) {
            const T *values = bucket.row(row);
            bool keep = false;
            for (size_t col = 0; col < bucket.cols; ++col) {
                if (values[col] > deletion_threshold) {
                    keep = true;
                    break;
                }
            }
            if (!keep) {
                continue;
            }
            kept_indices.push_back(indices[row]);
            kept_values.insert(kept_values.end(), values, values + bucket.cols);
        }
        indices = std::move(kept_indices);
        bucket.rows = indices.size();
        bucket.data = std::move(kept_values);
        if (bucket.rows == 0) {
            bucket.cols = 0;
        }
    }
}

std::pair<uint64_t, uint64_t> replace_val(uint64_t encoded_board) {
    uint64_t replace_value = 0xFULL;
    for (int shift = 0; shift < 64; shift += 4) {
        uint64_t encoded_num = (encoded_board >> static_cast<uint64_t>(shift)) & 0xFULL;
        if (encoded_num == 0xFULL) {
            encoded_board &= ~(0xFULL << static_cast<uint64_t>(shift));
            encoded_board |= (replace_value << static_cast<uint64_t>(shift));
            --replace_value;
        }
    }
    return {encoded_board, replace_value};
}

uint64_t ind_match(uint64_t encoded_board, uint64_t replacement_value) {
    uint64_t index = 0;
    uint64_t base = 0xFULL - replacement_value;
    for (int shift = 60; shift >= 0; shift -= 4) {
        uint64_t encoded_num = (encoded_board >> static_cast<uint64_t>(shift)) & 0xFULL;
        if (encoded_num > replacement_value) {
            index *= base;
            index += (encoded_num - replacement_value);
        }
    }
    return index;
}

std::vector<uint64_t> reverse_arr(const std::vector<uint64_t> &boards) {
    std::vector<uint64_t> result(boards.size());
    for (size_t i = 0; i < boards.size(); ++i) {
        result[i] = FormationAD::reverse(boards[i]);
    }
    return result;
}

void move_arr_into(const std::vector<uint64_t> &boards, int direction, std::vector<uint64_t> &result) {
    result.resize(boards.size());
    for (size_t i = 0; i < boards.size(); ++i) {
        result[i] = BoardMover::move_board(boards[i], direction);
    }
}

void sym_arr_like(std::vector<uint64_t> &boards, int symm_index) {
    if (symm_index == 0 || boards.empty()) {
        return;
    }
    switch (symm_index) {
        case 1:
            sym_arr_like_transform(boards, [](uint64_t board) { return Calculator::ReverseLR(board); });
            return;
        case 2:
            sym_arr_like_transform(boards, [](uint64_t board) { return Calculator::ReverseUD(board); });
            return;
        case 3:
            sym_arr_like_transform(boards, [](uint64_t board) { return Calculator::ReverseUL(board); });
            return;
        case 4:
            sym_arr_like_transform(boards, [](uint64_t board) { return Calculator::ReverseUR(board); });
            return;
        case 5:
            sym_arr_like_transform(boards, [](uint64_t board) { return Calculator::Rotate180(board); });
            return;
        case 6:
            sym_arr_like_transform(boards, [](uint64_t board) { return Calculator::RotateL(board); });
            return;
        case 7:
            sym_arr_like_transform(boards, [](uint64_t board) { return Calculator::RotateR(board); });
            return;
        default:
            return;
    }
}

void match_arr_into(const std::vector<uint64_t> &boards, AdSolveWorkspace<uint32_t> &workspace, std::vector<uint32_t> &ranked) {
    auto &sorted_indices = workspace.sorted_indices;
    sorted_indices.resize(boards.size());
    for (size_t i = 0; i < boards.size(); ++i) {
        sorted_indices[i] = static_cast<uint32_t>(i);
    }
    std::sort(
        sorted_indices.begin(),
        sorted_indices.end(),
        [&boards](uint32_t lhs, uint32_t rhs) {
            if (boards[lhs] == boards[rhs]) {
                return lhs < rhs;
            }
            return boards[lhs] < boards[rhs];
        }
    );
    ranked.resize(boards.size());
    for (size_t i = 0; i < sorted_indices.size(); ++i) {
        ranked[sorted_indices[i]] = static_cast<uint32_t>(i);
    }
}

template <typename T>
void match_arr_into(const std::vector<uint64_t> &boards, AdSolveWorkspace<T> &workspace, std::vector<uint32_t> &ranked) {
    auto &sorted_indices = workspace.sorted_indices;
    sorted_indices.resize(boards.size());
    for (size_t i = 0; i < boards.size(); ++i) {
        sorted_indices[i] = static_cast<uint32_t>(i);
    }
    std::sort(
        sorted_indices.begin(),
        sorted_indices.end(),
        [&boards](uint32_t lhs, uint32_t rhs) {
            if (boards[lhs] == boards[rhs]) {
                return lhs < rhs;
            }
            return boards[lhs] < boards[rhs];
        }
    );
    ranked.resize(boards.size());
    for (size_t i = 0; i < sorted_indices.size(); ++i) {
        ranked[sorted_indices[i]] = static_cast<uint32_t>(i);
    }
}

struct DispatchResult {
    bool need_process = false;
    ArrayView<const uint8_t> tiles_combinations{};
    uint8_t pos_rank = 0;
    uint64_t pos_32k = 0;
    uint8_t tile_value = 0;
    int8_t count32k = 0;
};

void process_derived_into(
    uint64_t board_after_spawn,
    uint32_t board_sum_after_spawn,
    int direction,
    int symm_index,
    const FormationAD::TilesCombinationTable &tiles_table,
    const FormationAD::PermutationTable &permutation_table,
    const AdvancedMaskParam &param,
    std::vector<uint64_t> &out,
    std::vector<uint64_t> &scratch
) {
    FormationAD::unmask_board_into(
        board_after_spawn, board_sum_after_spawn, tiles_table, permutation_table, param, scratch
    );
    move_arr_into(scratch, direction + 1, out);
    sym_arr_like(out, symm_index);
}

template <typename T>
DispatchResult dispatch_mnt_osr_ad_arr(
    uint64_t unmasked_board,
    uint32_t original_board_sum,
    std::vector<T> &osr,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    T max_scale
) {
    auto stats = FormationAD::tile_sum_and_32k_count4(unmasked_board, param);
    if (stats.is_success) {
        std::fill(osr.begin(), osr.end(), max_scale);
        return {};
    }
    uint32_t large_tiles_sum = original_board_sum
        - stats.total_sum
        - (static_cast<uint32_t>(param.num_free_32k + param.num_fixed_32k) << 15U);
    auto tiles = FormationAD::tiles_combination_view(
        tiles_table,
        static_cast<uint8_t>(large_tiles_sum >> 6U),
        static_cast<uint8_t>(stats.count_32k - static_cast<int8_t>(param.num_free_32k))
    );
    if (tiles.empty()) {
        return {};
    }
    return {true, tiles, stats.pos_rank, stats.pos_bitmap, stats.merged_tile, stats.count_32k};
}

template <typename T>
void update_osr_ad_arr(
    const MatrixBucket<T> &book_bucket,
    uint64_t board,
    const std::vector<uint64_t> &indices,
    const AdaptiveIndex::Index &prefix_index,
    std::vector<T> &osr,
    ArrayView<const uint32_t> ranked_array
) {
    size_t mid = search_arr_ad(indices, board, prefix_index);
    if (mid == kNotFoundIndex) {
        return;
    }
    const T *src = book_bucket.row(mid);
    for (size_t i = 0; i < ranked_array.size; ++i) {
        T value = src[ranked_array[i]];
        if (value > osr[i]) {
            osr[i] = value;
        }
    }
}

template <typename T>
T update_osr_ad(
    const MatrixBucket<T> &book_bucket,
    uint64_t board,
    const std::vector<uint64_t> &indices,
    const AdaptiveIndex::Index &prefix_index,
    T osr
) {
    size_t mid = search_arr_ad(indices, board, prefix_index);
    if (mid == kNotFoundIndex) {
        return osr;
    }
    return std::max(osr, book_bucket.row(mid)[0]);
}

template <typename T>
void update_mnt_osr_ad_arr3(
    const BookStore<T> &book_dict,
    uint64_t unmasked_board,
    int8_t count_32k,
    const IndexStore &ind_dict,
    const PrefixStore &indind_dict,
    ArrayView<const uint32_t> ranked_array,
    std::vector<T> &osr
) {
    const int target_count = static_cast<int>(count_32k - 3 + 16);
    size_t mid = search_arr_ad(ind_dict.at(target_count), unmasked_board, indind_dict.at(target_count));
    if (mid == kNotFoundIndex) {
        return;
    }
    const T *row = book_dict.at(target_count).row(mid);
    for (size_t i = 0; i < ranked_array.size; ++i) {
        T value = row[ranked_array[i]];
        if (value > osr[i]) {
            osr[i] = value;
        }
    }
}

template <typename T>
void update_mnt_osr_ad_arr2(
    const BookStore<T> &book_dict,
    uint64_t board,
    uint8_t pos_rank,
    int8_t count_32k,
    const IndexStore &ind_dict,
    const PrefixStore &indind_dict,
    ArrayView<const uint32_t> ranked_array,
    std::vector<T> &osr,
    const FormationAD::PermutationTable &permutation_table,
    const AdvancedMaskParam &param
) {
    size_t mid = search_arr_ad(ind_dict.at(count_32k), board, indind_dict.at(count_32k));
    if (mid == kNotFoundIndex) {
        return;
    }
    auto subset = FormationAD::permutation_first_subset(
        permutation_table,
        static_cast<uint8_t>(count_32k),
        static_cast<uint8_t>(count_32k - static_cast<int8_t>(param.num_free_32k)),
        pos_rank
    );
    const T *row = book_dict.at(count_32k).row(mid);
    if (subset.size < ranked_array.size) {
        // TODO(ad-parity): verify whether any real pattern can violate the expected permutation filter cardinality.
        return;
    }
    for (size_t i = 0; i < ranked_array.size; ++i) {
        T value = row[subset[ranked_array[i]]];
        if (value > osr[i]) {
            osr[i] = value;
        }
    }
}

template <typename T>
void update_mnt_osr_ad_arr1(
    const BookStore<T> &book_dict,
    uint64_t unmasked_board,
    const IndexStore &ind_dict,
    const PrefixStore &indind_dict,
    const std::vector<uint64_t> &board_derived,
    std::vector<T> &osr,
    const AdvancedPatternSpec &spec,
    ArrayView<const uint8_t> tiles_combinations,
    uint8_t pos_rank,
    uint64_t pos_32k,
    uint8_t tile_value,
    int8_t count_32k,
    AdSolveWorkspace<T> &workspace
) {
    FormationAD::extract_f_positions_compact(pos_32k, workspace.positions.data());
    const auto &positions = workspace.positions;
    for (int j = 0; j < count_32k; ++j) {
        if (j == static_cast<int>(pos_rank)) {
            continue;
        }
        const uint64_t pos = static_cast<uint64_t>(positions[static_cast<size_t>(j)]);
        uint64_t unmasked_b_j = (unmasked_board & ~(0xFULL << pos)) | (static_cast<uint64_t>(tiles_combinations[0]) << pos);
        auto [canonical_board, symm_index] = apply_sym_pair(unmasked_b_j, spec.symm_mode);

        const int bucket_key = -count_32k;
        size_t mid = search_arr_ad(ind_dict.at(bucket_key), canonical_board, indind_dict.at(bucket_key));
        if (mid == kNotFoundIndex) {
            continue;
        }

        auto &matched_positions = workspace.matched_positions;
        auto &matched_boards = workspace.matched_boards;
        matched_positions.clear();
        matched_boards.clear();
        matched_positions.reserve(board_derived.size());
        matched_boards.reserve(board_derived.size());
        for (size_t index = 0; index < board_derived.size(); ++index) {
            if (((board_derived[index] >> pos) & 0xFULL) == static_cast<uint64_t>(tile_value)) {
                matched_positions.push_back(index);
                matched_boards.push_back(board_derived[index]);
            }
        }
        if (matched_boards.empty()) {
            continue;
        }
        sym_arr_like(matched_boards, symm_index);
        auto &ranked_array = workspace.ranked_array;
        match_arr_into(matched_boards, workspace, ranked_array);
        const T *row = book_dict.at(bucket_key).row(mid);
        for (size_t index = 0; index < ranked_array.size(); ++index) {
            T value = row[ranked_array[index]];
            size_t position = matched_positions[index];
            if (value > osr[position]) {
                osr[position] = value;
            }
        }
    }
}

std::tuple<uint8_t, uint8_t, int8_t, uint64_t> find_3x64_pos(uint64_t unmasked_board, const AdvancedMaskParam &param) {
    uint8_t pos_rank = 0;
    uint8_t pos_rank64 = 0;
    uint8_t pos_rank128 = 0;
    uint64_t pos_64 = 0;
    for (int shift = 60; shift >= 0; shift -= 4) {
        uint64_t tile_value = (unmasked_board >> static_cast<uint64_t>(shift)) & 0xFULL;
        if (tile_value == 0xFULL) {
            if (((param.pos_fixed_32k_mask >> static_cast<uint64_t>(shift)) & 0xFULL) == 0ULL) {
                ++pos_rank;
            }
        } else if (tile_value == 6ULL) {
            pos_rank64 = pos_rank;
            pos_64 = static_cast<uint64_t>(shift);
            ++pos_rank;
        } else if (tile_value == 7ULL) {
            pos_rank128 = pos_rank;
            ++pos_rank;
        }
    }
    return {pos_rank64, pos_rank128, static_cast<int8_t>(pos_rank), pos_64};
}

uint32_t permutations_mapping_364(uint8_t x, uint8_t y, int8_t n) {
    if (x < y) {
        return static_cast<uint32_t>(2 * n * x - x * x - 2 * x + y - 1);
    }
    return static_cast<uint32_t>(2 * n * y - y * y - 3 * y + n + x - 2);
}

template <typename T>
T update_mnt_osr_364_ad(
    const BookStore<T> &book_dict,
    uint64_t board,
    uint64_t unmasked_board,
    const IndexStore &ind_dict,
    const PrefixStore &indind_dict,
    T osr,
    const AdvancedMaskParam &param
) {
    auto [pos_rank64, pos_rank128, pos_rank, pos_64] = find_3x64_pos(unmasked_board, param);
    board |= (0xFULL << pos_64);
    size_t mid = search_arr_ad(ind_dict.at(pos_rank), board, indind_dict.at(pos_rank));
    if (mid == kNotFoundIndex) {
        return osr;
    }
    uint32_t mapping = permutations_mapping_364(pos_rank64, pos_rank128, pos_rank);
    return std::max(osr, book_dict.at(pos_rank).row(mid)[mapping]);
}

template <typename T>
void update_mnt_osr_364_arr_ad(
    const BookStore<T> &book_dict,
    uint64_t board,
    uint64_t unmasked_board,
    const IndexStore &ind_dict,
    const PrefixStore &indind_dict,
    std::vector<T> &osr,
    ArrayView<const uint32_t> ranked_array,
    const FormationAD::PermutationTable &permutation_table,
    const AdvancedMaskParam &param
) {
    auto [pos_rank64, pos_rank128, pos_rank, pos_64] = find_3x64_pos(unmasked_board, param);
    board |= (0xFULL << pos_64);
    size_t mid = search_arr_ad(ind_dict.at(pos_rank), board, indind_dict.at(pos_rank));
    if (mid == kNotFoundIndex) {
        return;
    }
    auto subset = FormationAD::permutation_pair_subset(
        permutation_table,
        static_cast<uint8_t>(pos_rank),
        static_cast<uint8_t>(pos_rank - static_cast<int8_t>(param.num_free_32k)),
        pos_rank64,
        pos_rank128
    );
    const T *row = book_dict.at(pos_rank).row(mid);
    if (subset.size < ranked_array.size) {
        // TODO(ad-parity): verify whether any real pattern can violate the expected 3x64 permutation filter cardinality.
        return;
    }
    for (size_t i = 0; i < ranked_array.size; ++i) {
        T value = row[subset[ranked_array[i]]];
        if (value > osr[i]) {
            osr[i] = value;
        }
    }
}

template <typename T>
T update_mnt_osr_ad(
    const BookStore<T> &book_dict,
    uint64_t board,
    uint64_t unmasked_board,
    const IndexStore &ind_dict,
    const PrefixStore &indind_dict,
    T osr,
    uint32_t board_sum_after_spawn,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    T max_scale
) {
    auto stats = FormationAD::tile_sum_and_32k_count4(unmasked_board, param);
    if (stats.is_success) {
        return max_scale;
    }
    uint32_t large_tiles_sum = board_sum_after_spawn
        - stats.total_sum
        - (static_cast<uint32_t>(param.num_free_32k + param.num_fixed_32k) << 15U);
    auto tiles = FormationAD::tiles_combination_view(
        tiles_table,
        static_cast<uint8_t>(large_tiles_sum >> 6U),
        static_cast<uint8_t>(stats.count_32k - static_cast<int8_t>(param.num_free_32k))
    );
    if (tiles.empty()) {
        return osr;
    }
    if (stats.merged_tile_found == 2U) {
        return osr;
    }
    int bucket_key = stats.count_32k;
    if (tiles.size > 1 && tiles[0] == tiles[1]) {
        if (tiles.size > 2 && tiles[0] == tiles[2]) {
            bucket_key = static_cast<int8_t>(stats.count_32k + 16 - 3);
            board = unmasked_board;
        } else {
            bucket_key = static_cast<int8_t>(-stats.count_32k);
            uint64_t tile = static_cast<uint64_t>(tiles[0]);
            uint64_t tiles_all_positions = tile * 0x1111111111111111ULL;
            board = (board & (~stats.pos_bitmap)) | (tiles_all_positions & stats.pos_bitmap);
        }
    }
    size_t mid = search_arr_ad(ind_dict.at(bucket_key), board, indind_dict.at(bucket_key));
    if (mid == kNotFoundIndex) {
        return osr;
    }
    const T *row = book_dict.at(bucket_key).row(mid);
    if (bucket_key > 15) {
        return std::max(osr, row[0]);
    }
    if (bucket_key > 0) {
        return std::max(osr, row[stats.pos_rank]);
    }
    return std::max(osr, row[0]);
}

template <typename T>
void solve_optimal_success_rate_arr_into(
    uint64_t board,
    uint64_t new_value,
    int spawn_pos,
    uint64_t rep_t,
    uint64_t rep_t_rev,
    size_t derive_size,
    const AdvancedPatternSpec &spec,
    uint64_t rep_v,
    int8_t count_32k,
    const BookStore<T> &book_dict,
    const IndexStore &ind_dict,
    const PrefixStore &indind_dict,
    uint32_t board_sum_after_spawn,
    MatchCache &match_cache,
    const MatrixBucket<T> &book_bucket,
    const std::vector<uint64_t> &ind_arr,
    const AdaptiveIndex::Index &indind_arr,
    const FormationAD::TilesCombinationTable &tiles_table,
    const FormationAD::PermutationTable &permutation_table,
    const AdvancedMaskParam &param,
    T zero_val,
    T max_scale,
    AdSolveWorkspace<T> &workspace,
    std::vector<T> &optimal_success_rate
) {
    uint64_t board_with_spawn = board | (new_value << static_cast<uint64_t>(4 * spawn_pos));
    uint64_t rep_t_gen = rep_t | (new_value << static_cast<uint64_t>(4 * spawn_pos));
    uint64_t board_rev = FormationAD::reverse(board_with_spawn);
    optimal_success_rate.assign(derive_size, zero_val);
    auto moves = FormationAD::m_move_all_dir2(board_with_spawn, board_rev);
    (void)rep_t_rev;

    for (int direction = 0; direction < 4; ++direction) {
        uint64_t moved_board = moves[static_cast<size_t>(direction)].board;
        bool mask_new_tile = moves[static_cast<size_t>(direction)].mask_new_tile;
        if (moved_board == board_with_spawn || !is_pattern(moved_board, spec.pattern_masks)) {
            continue;
        }
        auto [canonical_board, symm_index] = apply_sym_pair(moved_board, spec.symm_mode);
        uint64_t rep_t_gen_m = BoardMover::move_board(rep_t_gen, direction + 1);
        rep_t_gen_m = apply_sym_like(rep_t_gen_m, symm_index);
        uint64_t match_ind = ind_match(rep_t_gen_m, rep_v);
        size_t hashed_match_ind = static_cast<size_t>(match_ind % static_cast<uint64_t>(match_cache.map_length));

        auto load_or_compute_ranked = [&](bool store_if_empty) -> ArrayView<const uint32_t> {
            if (match_cache.key(hashed_match_ind) == match_ind) {
                return {match_cache.row(hashed_match_ind), derive_size};
            }
            process_derived_into(
                board_with_spawn,
                board_sum_after_spawn,
                direction,
                symm_index,
                tiles_table,
                permutation_table,
                param,
                workspace.moved_boards,
                workspace.derived_boards
            );
            match_arr_into(workspace.moved_boards, workspace, workspace.ranked_array);
            if (store_if_empty && match_cache.try_claim_empty(hashed_match_ind)) {
                match_cache.publish(hashed_match_ind, match_ind, workspace.ranked_array);
                return {match_cache.row(hashed_match_ind), derive_size};
            }
            if (match_cache.key(hashed_match_ind) == match_ind) {
                return {match_cache.row(hashed_match_ind), derive_size};
            }
            return {workspace.ranked_array.data(), workspace.ranked_array.size()};
        };

        if (mask_new_tile) {
            uint64_t unmasked_newt = BoardMover::move_board(board_with_spawn, direction + 1);
            unmasked_newt = apply_sym_like(unmasked_newt, symm_index);
            if (count_32k > 15) {
                ArrayView<const uint32_t> ranked_array = load_or_compute_ranked(true);
                update_mnt_osr_364_arr_ad(
                    book_dict,
                    canonical_board,
                    unmasked_newt,
                    ind_dict,
                    indind_dict,
                    optimal_success_rate,
                    ranked_array,
                    permutation_table,
                    param
                );
                continue;
            }

            DispatchResult dispatch = dispatch_mnt_osr_ad_arr(
                unmasked_newt, board_sum_after_spawn, optimal_success_rate, tiles_table, param, max_scale
            );
            if (!dispatch.need_process) {
                continue;
            }
            if (
                dispatch.tiles_combinations.size > 2 &&
                dispatch.tiles_combinations[0] == dispatch.tiles_combinations[1] &&
                dispatch.tiles_combinations[0] == dispatch.tiles_combinations[2]
            ) {
                ArrayView<const uint32_t> ranked_array = load_or_compute_ranked(true);
                update_mnt_osr_ad_arr3(
                    book_dict,
                    unmasked_newt,
                    dispatch.count32k,
                    ind_dict,
                    indind_dict,
                    ranked_array,
                    optimal_success_rate
                );
            } else if (
                dispatch.tiles_combinations.size > 1 &&
                dispatch.tiles_combinations[0] == dispatch.tiles_combinations[1]
            ) {
                process_derived_into(
                    board_with_spawn,
                    board_sum_after_spawn,
                    direction,
                    symm_index,
                    tiles_table,
                    permutation_table,
                    param,
                    workspace.moved_boards,
                    workspace.derived_boards
                );
                update_mnt_osr_ad_arr1(
                    book_dict,
                    unmasked_newt,
                    ind_dict,
                    indind_dict,
                    workspace.moved_boards,
                    optimal_success_rate,
                    spec,
                    dispatch.tiles_combinations,
                    dispatch.pos_rank,
                    dispatch.pos_32k,
                    dispatch.tile_value,
                    dispatch.count32k,
                    workspace
                );
            } else {
                ArrayView<const uint32_t> ranked_array = load_or_compute_ranked(true);
                update_mnt_osr_ad_arr2(
                    book_dict,
                    canonical_board,
                    dispatch.pos_rank,
                    dispatch.count32k,
                    ind_dict,
                    indind_dict,
                    ranked_array,
                    optimal_success_rate,
                    permutation_table,
                    param
                );
            }
            continue;
        }

        ArrayView<const uint32_t> ranked_array = load_or_compute_ranked(true);
        update_osr_ad_arr(book_bucket, canonical_board, ind_arr, indind_arr, optimal_success_rate, ranked_array);
    }
}

template <typename T>
T solve_optimal_success_rate(
    uint64_t board,
    uint64_t new_value,
    int spawn_pos,
    uint32_t board_sum_after_spawn,
    const AdvancedPatternSpec &spec,
    int8_t count_32k,
    const BookStore<T> &book_dict,
    const IndexStore &ind_dict,
    const PrefixStore &indind_dict,
    const MatrixBucket<T> &book_bucket,
    const std::vector<uint64_t> &ind_arr,
    const AdaptiveIndex::Index &indind_arr,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    T zero_val,
    T max_scale
) {
    uint64_t board_with_spawn = board | (new_value << static_cast<uint64_t>(4 * spawn_pos));
    uint64_t board_rev = FormationAD::reverse(board_with_spawn);
    T optimal_success_rate = zero_val;
    auto moves = FormationAD::m_move_all_dir2(board_with_spawn, board_rev);

    for (int direction = 0; direction < 4; ++direction) {
        uint64_t moved_board = moves[static_cast<size_t>(direction)].board;
        bool mask_new_tile = moves[static_cast<size_t>(direction)].mask_new_tile;
        if (moved_board == board_with_spawn || !is_pattern(moved_board, spec.pattern_masks)) {
            continue;
        }
        auto [canonical_board, symm_index] = apply_sym_pair(moved_board, spec.symm_mode);
        if (mask_new_tile) {
            uint64_t unmasked_newt = BoardMover::move_board(board_with_spawn, direction + 1);
            unmasked_newt = apply_sym_like(unmasked_newt, symm_index);
            if (count_32k < 16) {
                optimal_success_rate = update_mnt_osr_ad(
                    book_dict,
                    canonical_board,
                    unmasked_newt,
                    ind_dict,
                    indind_dict,
                    optimal_success_rate,
                    board_sum_after_spawn,
                    tiles_table,
                    param,
                    max_scale
                );
            } else {
                optimal_success_rate = update_mnt_osr_364_ad(
                    book_dict,
                    canonical_board,
                    unmasked_newt,
                    ind_dict,
                    indind_dict,
                    optimal_success_rate,
                    param
                );
            }
        } else {
            optimal_success_rate = update_osr_ad(book_bucket, canonical_board, ind_arr, indind_arr, optimal_success_rate);
        }
    }
    return optimal_success_rate;
}

template <typename T>
void expand_ad(
    const std::vector<uint64_t> &arr,
    uint32_t original_board_sum,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    BookStore<T> &book_dict,
    IndexStore &ind_dict,
    bool expand_book_dict,
    int num_threads
) {
    clear_book_store(book_dict);
    clear_index_store(ind_dict);
    std::vector<int8_t> count32ks(arr.size(), -1);

    #pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < static_cast<int64_t>(arr.size()); ++i) {
        uint64_t board = arr[static_cast<size_t>(i)];
        auto stats = FormationAD::tile_sum_and_32k_count2(board, param);
        if (stats.count_32k == static_cast<int8_t>(param.num_free_32k)) {
            count32ks[static_cast<size_t>(i)] = stats.count_32k;
            continue;
        }
        uint32_t large_tiles_sum = original_board_sum
            - stats.total_sum
            - (static_cast<uint32_t>(param.num_free_32k + param.num_fixed_32k) << 15U);
        auto tiles = FormationAD::tiles_combination_view(
            tiles_table,
            static_cast<uint8_t>(large_tiles_sum >> 6U),
            static_cast<uint8_t>(stats.count_32k - static_cast<int8_t>(param.num_free_32k))
        );
        if (tiles.empty()) {
            count32ks[static_cast<size_t>(i)] = -1;
            continue;
        }
        if (stats.count_32k - static_cast<int8_t>(param.num_free_32k) == 1) {
            count32ks[static_cast<size_t>(i)] = stats.count_32k;
            continue;
        }
        if (tiles.size > 1 && tiles[0] == tiles[1]) {
            if (tiles.size > 2 && tiles[0] == tiles[2]) {
                count32ks[static_cast<size_t>(i)] = static_cast<int8_t>(stats.count_32k - 3 + 16);
            } else {
                count32ks[static_cast<size_t>(i)] = static_cast<int8_t>(-stats.count_32k);
            }
            continue;
        }
        count32ks[static_cast<size_t>(i)] = stats.count_32k;
    }

    std::array<size_t, bucket_slot_count()> counts{};
    for (int8_t count_32k : count32ks) {
        if (count_32k == -1 || count_32k < bucket_key_min() || count_32k > bucket_key_max()) {
            continue;
        }
        ++counts[bucket_to_index(count_32k)];
    }

    for (int key = bucket_key_min(); key <= bucket_key_max(); ++key) {
        size_t rows = counts[bucket_to_index(key)];
        ind_dict.at(key).resize(rows);
        if (expand_book_dict) {
            book_dict.at(key).resize(rows, derive_size_for_bucket(key, param.num_free_32k));
        }
    }

    std::array<size_t, bucket_slot_count()> offsets{};
    for (size_t i = 0; i < arr.size(); ++i) {
        int8_t key = count32ks[i];
        if (key == -1 || key < bucket_key_min() || key > bucket_key_max()) {
            continue;
        }
        size_t offset = offsets[bucket_to_index(key)]++;
        ind_dict.at(key)[offset] = arr[i];
    }
}

template <typename T>
void recalculate_ad(
    BookStore<T> &book_dict0,
    const IndexStore &ind_dict0,
    const BookStore<T> &book_dict1,
    const IndexStore &ind_dict1,
    const PrefixStore &indind_dict1,
    const BookStore<T> &book_dict2,
    const IndexStore &ind_dict2,
    const PrefixStore &indind_dict2,
    const FormationAD::TilesCombinationTable &tiles_table,
    const FormationAD::PermutationTable &permutation_table,
    const AdvancedMaskParam &param,
    uint32_t original_board_sum,
    const AdvancedPatternSpec &spec,
    T max_scale,
    T zero_val,
    double spawn_rate4,
    int num_threads,
    std::unordered_map<uint32_t, MatchCache> &match_dict
) {
    for (int key = bucket_key_min(); key <= bucket_key_max(); ++key) {
        const auto &indices = ind_dict0.at(key);
        auto &book_bucket0 = book_dict0.at(key);
        if (indices.empty() || book_bucket0.rows == 0 || book_bucket0.cols == 0) {
            continue;
        }

        const size_t derive_size = book_bucket0.cols;
        MatchCache &match_cache = get_shared_match_cache<T>(match_dict, derive_size);
        std::vector<AdSolveWorkspace<T>> thread_workspaces(static_cast<size_t>(num_threads));
        const auto &book_bucket1 = book_dict1.at(key);
        const auto &ind_arr1 = ind_dict1.at(key);
        const auto &indind_arr1 = indind_dict1.at(key);
        const auto &book_bucket2 = book_dict2.at(key);
        const auto &ind_arr2 = ind_dict2.at(key);
        const auto &indind_arr2 = indind_dict2.at(key);

        int chunk_count = std::max(
            std::min(1024, static_cast<int>((indices.size() * static_cast<size_t>(std::llround(std::log2(static_cast<double>(derive_size + 1U))))) / 1048576ULL)),
            1
        );
        size_t chunk_size = indices.size() / static_cast<size_t>(chunk_count);

        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            size_t start = chunk_size * static_cast<size_t>(chunk);
            size_t end = chunk == chunk_count - 1 ? indices.size() : chunk_size * static_cast<size_t>(chunk + 1);
            #pragma omp parallel for num_threads(num_threads)
            for (int64_t k = static_cast<int64_t>(start); k < static_cast<int64_t>(end); ++k) {
                const size_t thread_index = static_cast<size_t>(omp_get_thread_num());
                AdSolveWorkspace<T> &workspace = thread_workspaces[thread_index];
                uint64_t board = indices[static_cast<size_t>(k)];
                if (derive_size == 1U) {
                    double success_probability = 0.0;
                    int empty_slots = 0;
                    for (int pos = 0; pos < 16; ++pos) {
                        if (((board >> static_cast<uint64_t>(4 * pos)) & 0xFULL) != 0ULL) {
                            continue;
                        }
                        ++empty_slots;
                        T success2 = solve_optimal_success_rate(
                            board,
                            1ULL,
                            pos,
                            original_board_sum + 2U,
                            spec,
                            static_cast<int8_t>(key),
                            book_dict1,
                            ind_dict1,
                            indind_dict1,
                            book_bucket1,
                            ind_arr1,
                            indind_arr1,
                            tiles_table,
                            param,
                            zero_val,
                            max_scale
                        );
                        success_probability += static_cast<double>(success2) * (1.0 - spawn_rate4);

                        T success4 = solve_optimal_success_rate(
                            board,
                            2ULL,
                            pos,
                            original_board_sum + 4U,
                            spec,
                            static_cast<int8_t>(key),
                            book_dict2,
                            ind_dict2,
                            indind_dict2,
                            book_bucket2,
                            ind_arr2,
                            indind_arr2,
                            tiles_table,
                            param,
                            zero_val,
                            max_scale
                        );
                        success_probability += static_cast<double>(success4) * spawn_rate4;
                    }
                    book_bucket0.row(static_cast<size_t>(k))[0] = empty_slots == 0
                        ? zero_val
                        : static_cast<T>(success_probability / static_cast<double>(empty_slots));
                    continue;
                }

                auto [rep_t, rep_v] = replace_val(board);
                uint64_t rep_t_rev = FormationAD::reverse(rep_t);
                auto &success_probability = workspace.success_probability;
                success_probability.assign(derive_size, 0.0);
                int empty_slots = 0;
                for (int pos = 0; pos < 16; ++pos) {
                    if (((board >> static_cast<uint64_t>(4 * pos)) & 0xFULL) != 0ULL) {
                        continue;
                    }
                    ++empty_slots;
                    solve_optimal_success_rate_arr_into(
                        board,
                        1ULL,
                        pos,
                        rep_t,
                        rep_t_rev,
                        derive_size,
                        spec,
                        rep_v,
                        static_cast<int8_t>(key),
                        book_dict1,
                        ind_dict1,
                        indind_dict1,
                        original_board_sum + 2U,
                        match_cache,
                        book_bucket1,
                        ind_arr1,
                        indind_arr1,
                        tiles_table,
                        permutation_table,
                        param,
                        zero_val,
                        max_scale,
                        workspace,
                        workspace.optimal_values
                    );
                    for (size_t idx = 0; idx < derive_size; ++idx) {
                        success_probability[idx] += static_cast<double>(workspace.optimal_values[idx]) * (1.0 - spawn_rate4);
                    }

                    solve_optimal_success_rate_arr_into(
                        board,
                        2ULL,
                        pos,
                        rep_t,
                        rep_t_rev,
                        derive_size,
                        spec,
                        rep_v,
                        static_cast<int8_t>(key),
                        book_dict2,
                        ind_dict2,
                        indind_dict2,
                        original_board_sum + 4U,
                        match_cache,
                        book_bucket2,
                        ind_arr2,
                        indind_arr2,
                        tiles_table,
                        permutation_table,
                        param,
                        zero_val,
                        max_scale,
                        workspace,
                        workspace.temp_values
                    );
                    for (size_t idx = 0; idx < derive_size; ++idx) {
                        success_probability[idx] += static_cast<double>(workspace.temp_values[idx]) * spawn_rate4;
                    }
                }
                T *dst = book_bucket0.row(static_cast<size_t>(k));
                for (size_t idx = 0; idx < derive_size; ++idx) {
                    dst[idx] = empty_slots == 0
                        ? zero_val
                        : static_cast<T>(success_probability[idx] / static_cast<double>(empty_slots));
                }
            }
        }
    }
}

template <typename T>
MatchCache &get_shared_match_cache(
    std::unordered_map<uint32_t, MatchCache> &match_dict,
    size_t derive_size
) {
    const uint32_t key = static_cast<uint32_t>(derive_size);
    auto it = match_dict.find(key);
    if (it == match_dict.end()) {
        const size_t map_length = derive_size < 5000U ? 33331U : 11113U;
        it = match_dict.emplace(key, MatchCache(map_length, derive_size)).first;
    }
    return it->second;
}

template <typename T>
void recalculate_ad_chunk(
    int8_t bucket_key,
    const std::vector<uint64_t> &positions_chunk,
    MatrixBucket<T> &book_chunk,
    const BookStore<T> &book_dict,
    const IndexStore &ind_dict,
    const PrefixStore &indind_dict,
    std::unordered_map<uint32_t, MatchCache> &match_dict,
    const FormationAD::TilesCombinationTable &tiles_table,
    const FormationAD::PermutationTable &permutation_table,
    const AdvancedMaskParam &param,
    uint32_t original_board_sum,
    const AdvancedPatternSpec &spec,
    T max_scale,
    T zero_val,
    bool is_gen2_step,
    double spawn_rate4,
    int num_threads
) {
    const auto &book_bucket = book_dict.at(bucket_key);
    const auto &ind_arr = ind_dict.at(bucket_key);
    const auto &indind_arr = indind_dict.at(bucket_key);
    const size_t derive_size = book_chunk.cols;
    if (derive_size == 0 || positions_chunk.empty()) {
        return;
    }
    MatchCache &match_cache = get_shared_match_cache<T>(match_dict, derive_size);
    std::vector<AdSolveWorkspace<T>> thread_workspaces(static_cast<size_t>(num_threads));

    const uint64_t new_value = is_gen2_step ? 1ULL : 2ULL;
    const double probability = is_gen2_step ? (1.0 - spawn_rate4) : spawn_rate4;
    const uint32_t board_sum_after_spawn = original_board_sum + (is_gen2_step ? 2U : 4U);

    int chunk_count = std::max(
        std::min(
            1024,
            static_cast<int>(
                (positions_chunk.size() * static_cast<size_t>(std::llround(std::log2(static_cast<double>(derive_size + 1U)))))
                / 1048576ULL
            )
        ),
        1
    );
    size_t chunk_size = positions_chunk.size() / static_cast<size_t>(chunk_count);

    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        size_t start = chunk_size * static_cast<size_t>(chunk);
        size_t end = chunk == chunk_count - 1 ? positions_chunk.size() : chunk_size * static_cast<size_t>(chunk + 1);
        #pragma omp parallel for num_threads(num_threads)
        for (int64_t row_index = static_cast<int64_t>(start); row_index < static_cast<int64_t>(end); ++row_index) {
            const size_t thread_index = static_cast<size_t>(omp_get_thread_num());
            AdSolveWorkspace<T> &workspace = thread_workspaces[thread_index];
            const uint64_t board = positions_chunk[static_cast<size_t>(row_index)];
            if (derive_size == 1U) {
                double success_probability = 0.0;
                int empty_slots = 0;
                for (int pos = 0; pos < 16; ++pos) {
                    if (((board >> static_cast<uint64_t>(4 * pos)) & 0xFULL) != 0ULL) {
                        continue;
                    }
                    ++empty_slots;
                    T success = solve_optimal_success_rate(
                        board,
                        new_value,
                        pos,
                        board_sum_after_spawn,
                        spec,
                        bucket_key,
                        book_dict,
                        ind_dict,
                        indind_dict,
                        book_bucket,
                        ind_arr,
                        indind_arr,
                        tiles_table,
                        param,
                        zero_val,
                        max_scale
                    );
                    success_probability += static_cast<double>(success);
                }
                T *dst = book_chunk.row(static_cast<size_t>(row_index));
                if (empty_slots == 0) {
                    dst[0] = is_gen2_step ? static_cast<T>(static_cast<double>(dst[0]) * spawn_rate4) : zero_val;
                } else if (is_gen2_step) {
                    dst[0] = static_cast<T>(
                        static_cast<double>(dst[0]) * spawn_rate4
                        + (success_probability * probability / static_cast<double>(empty_slots))
                    );
                } else {
                    dst[0] = static_cast<T>(success_probability / static_cast<double>(empty_slots));
                }
                continue;
            }

            auto [rep_t, rep_v] = replace_val(board);
            uint64_t rep_t_rev = FormationAD::reverse(rep_t);
            auto &success_probability = workspace.success_probability;
            success_probability.assign(derive_size, 0.0);
            int empty_slots = 0;
            for (int pos = 0; pos < 16; ++pos) {
                if (((board >> static_cast<uint64_t>(4 * pos)) & 0xFULL) != 0ULL) {
                    continue;
                }
                ++empty_slots;
                solve_optimal_success_rate_arr_into(
                    board,
                    new_value,
                    pos,
                    rep_t,
                    rep_t_rev,
                    derive_size,
                    spec,
                    rep_v,
                    bucket_key,
                    book_dict,
                    ind_dict,
                    indind_dict,
                    board_sum_after_spawn,
                    match_cache,
                    book_bucket,
                    ind_arr,
                    indind_arr,
                    tiles_table,
                    permutation_table,
                    param,
                    zero_val,
                    max_scale,
                    workspace,
                    workspace.optimal_values
                );
                for (size_t idx = 0; idx < derive_size; ++idx) {
                    success_probability[idx] += static_cast<double>(workspace.optimal_values[idx]);
                }
            }
            T *dst = book_chunk.row(static_cast<size_t>(row_index));
            if (empty_slots == 0) {
                if (is_gen2_step) {
                    for (size_t idx = 0; idx < derive_size; ++idx) {
                        dst[idx] = static_cast<T>(static_cast<double>(dst[idx]) * spawn_rate4);
                    }
                } else {
                    std::fill(dst, dst + derive_size, zero_val);
                }
                continue;
            }
            if (is_gen2_step) {
                for (size_t idx = 0; idx < derive_size; ++idx) {
                    dst[idx] = static_cast<T>(
                        static_cast<double>(dst[idx]) * spawn_rate4
                        + (success_probability[idx] * probability / static_cast<double>(empty_slots))
                    );
                }
            } else {
                for (size_t idx = 0; idx < derive_size; ++idx) {
                    dst[idx] = static_cast<T>(success_probability[idx] / static_cast<double>(empty_slots));
                }
            }
        }
    }
}

template <typename T>
void iter_ind_dict4(
    const std::vector<int> &ind_dict0_keys,
    const BookStore<T> &book_dict2,
    const IndexStore &ind_dict2,
    const PrefixStore &indind_dict2,
    std::unordered_map<uint32_t, MatchCache> &match_dict,
    const FormationAD::TilesCombinationTable &tiles_table,
    const FormationAD::PermutationTable &permutation_table,
    const AdvancedMaskParam &param,
    uint32_t original_board_sum,
    const AdvancedPatternSpec &spec,
    T max_scale,
    T zero_val,
    double spawn_rate4,
    const RunOptions &options,
    int step,
    double &timer_acc,
    size_t &counter_acc,
    T &max_rate,
    int num_threads
) {
    const fs::path folder = options.pathname + std::to_string(step) + "bt";
    for (int key : ind_dict0_keys) {
        std::vector<uint64_t> positions_array = read_binary_vector<uint64_t>((folder / (std::to_string(key) + ".i")).string());
        if (positions_array.empty()) {
            continue;
        }
        const size_t derive_size = derive_size_for_bucket(key, param.num_free_32k);
        const uint64_t free_mem = available_memory_bytes();
        const uint64_t chunk_size = std::max<uint64_t>(1ULL << 28, static_cast<uint64_t>(static_cast<double>(free_mem) * 0.9 / static_cast<double>(sizeof(T))));
        const size_t chunk_length = std::max<size_t>(1U, static_cast<size_t>(chunk_size / std::max<size_t>(derive_size, 1U)));
        std::ofstream out((folder / (std::to_string(key) + ".b")).string(), std::ios::binary | std::ios::trunc);
        for (size_t start_idx = 0; start_idx < positions_array.size(); start_idx += chunk_length) {
            const double t0 = wall_time_seconds();
            const size_t end_idx = std::min(start_idx + chunk_length, positions_array.size());
            std::vector<uint64_t> positions_chunk(
                positions_array.begin() + static_cast<std::ptrdiff_t>(start_idx),
                positions_array.begin() + static_cast<std::ptrdiff_t>(end_idx)
            );
            MatrixBucket<T> book_chunk;
            book_chunk.resize(positions_chunk.size(), derive_size);
            std::fill(book_chunk.data.begin(), book_chunk.data.end(), static_cast<T>(0));

            recalculate_ad_chunk(
                static_cast<int8_t>(key),
                positions_chunk,
                book_chunk,
                book_dict2,
                ind_dict2,
                indind_dict2,
                match_dict,
                tiles_table,
                permutation_table,
                param,
                original_board_sum,
                spec,
                max_scale,
                zero_val,
                false,
                spawn_rate4,
                num_threads
            );
            const double t1 = wall_time_seconds();
            if (t1 > t0) {
                std::ostringstream oss;
                oss << "step " << step
                    << ", gen4, count_32k " << key
                    << ", chunk " << (start_idx / chunk_length)
                    << ", done "
                    << round_to_2(static_cast<double>(positions_chunk.size() * derive_size) / (t1 - t0) / 1e6)
                    << " mbps";
                debug_log(oss.str());
            }
            timer_acc += (t1 - t0);
            counter_acc += positions_chunk.size() * derive_size;
            if (!book_chunk.data.empty()) {
                out.write(reinterpret_cast<const char *>(book_chunk.data.data()),
                          static_cast<std::streamsize>(book_chunk.data.size() * sizeof(T)));
            }
        }
    }
    (void)max_rate;
}

template <typename T>
void iter_ind_dict2(
    const std::vector<int> &ind_dict0_keys,
    const BookStore<T> &book_dict1,
    const IndexStore &ind_dict1,
    const PrefixStore &indind_dict1,
    std::unordered_map<uint32_t, MatchCache> &match_dict,
    const FormationAD::TilesCombinationTable &tiles_table,
    const FormationAD::PermutationTable &permutation_table,
    const AdvancedMaskParam &param,
    uint32_t original_board_sum,
    const AdvancedPatternSpec &spec,
    T max_scale,
    T zero_val,
    double spawn_rate4,
    const RunOptions &options,
    int step,
    double &timer_acc,
    size_t &counter_acc,
    T &max_rate,
    int num_threads
) {
    const fs::path folder = options.pathname + std::to_string(step) + "bt";
    for (int key : ind_dict0_keys) {
        std::vector<uint64_t> positions_array = read_binary_vector<uint64_t>((folder / (std::to_string(key) + ".i")).string());
        if (positions_array.empty()) {
            continue;
        }
        const fs::path book_path = folder / (std::to_string(key) + ".b");
        if (!fs::exists(book_path)) {
            continue;
        }
        const size_t derive_size = derive_size_for_bucket(key, param.num_free_32k);
        const uint64_t free_mem = available_memory_bytes();
        const uint64_t chunk_size = std::max<uint64_t>(1ULL << 28, static_cast<uint64_t>(static_cast<double>(free_mem) * 0.9 / static_cast<double>(sizeof(T))));
        const size_t chunk_length = std::max<size_t>(1U, static_cast<size_t>(chunk_size / std::max<size_t>(derive_size, 1U)));

        std::fstream file(book_path, std::ios::binary | std::ios::in | std::ios::out);
        for (size_t start_idx = 0; start_idx < positions_array.size(); start_idx += chunk_length) {
            const double t0 = wall_time_seconds();
            const size_t end_idx = std::min(start_idx + chunk_length, positions_array.size());
            std::vector<uint64_t> positions_chunk(
                positions_array.begin() + static_cast<std::ptrdiff_t>(start_idx),
                positions_array.begin() + static_cast<std::ptrdiff_t>(end_idx)
            );
            MatrixBucket<T> book_chunk;
            book_chunk.resize(positions_chunk.size(), derive_size);

            const std::streamoff offset = static_cast<std::streamoff>(start_idx * derive_size * sizeof(T));
            file.seekg(offset, std::ios::beg);
            if (!book_chunk.data.empty()) {
                file.read(reinterpret_cast<char *>(book_chunk.data.data()),
                          static_cast<std::streamsize>(book_chunk.data.size() * sizeof(T)));
            }

            recalculate_ad_chunk(
                static_cast<int8_t>(key),
                positions_chunk,
                book_chunk,
                book_dict1,
                ind_dict1,
                indind_dict1,
                match_dict,
                tiles_table,
                permutation_table,
                param,
                original_board_sum,
                spec,
                max_scale,
                zero_val,
                true,
                spawn_rate4,
                num_threads
            );
            const double t1 = wall_time_seconds();
            if (t1 > t0) {
                std::ostringstream oss;
                oss << "step " << step
                    << ", gen2, count_32k " << key
                    << ", chunk " << (start_idx / chunk_length)
                    << ", done "
                    << round_to_2(static_cast<double>(positions_chunk.size() * derive_size) / (t1 - t0) / 1e6)
                    << " mbps";
                debug_log(oss.str());
            }
            timer_acc += (t1 - t0);
            counter_acc += positions_chunk.size() * derive_size;
            if (!book_chunk.data.empty()) {
                max_rate = std::max(max_rate, *std::max_element(book_chunk.data.begin(), book_chunk.data.end()));
                file.seekp(offset, std::ios::beg);
                file.write(reinterpret_cast<const char *>(book_chunk.data.data()),
                           static_cast<std::streamsize>(book_chunk.data.size() * sizeof(T)));
                file.flush();
            }
        }
    }
}

template <typename T>
void recalculate_process_ad_chunked_impl(
    const std::vector<uint64_t> &arr_init,
    const AdvancedPatternSpec &spec,
    const RunOptions &options,
    bool started_from_generate
) {
    ensure_stats_header<T>(options);
    const AdvancedMaskParam param = FormationAD::build_mask_param(spec);
    const FormationAD::MaskerContext masker = FormationAD::init_masker(spec);
    const uint32_t ini_board_sum = arr_init.empty() ? 0U : board_sum(arr_init.front());
    const T max_scale = max_scale_value<T>();
    const T zero_val = zero_value<T>();
    const T deletion_threshold = static_cast<T>(options.deletion_threshold * static_cast<double>(max_scale - zero_val) + zero_val);
    const int num_threads = effective_num_threads(options);

    BookStore<T> book_dict1;
    BookStore<T> book_dict2;
    IndexStore ind_dict1;
    IndexStore ind_dict2;
    std::unique_ptr<PrefixStore> indind_dict1;
    std::unique_ptr<PrefixStore> indind_dict2;
    clear_book_store(book_dict1);
    clear_book_store(book_dict2);
    clear_index_store(ind_dict1);
    clear_index_store(ind_dict2);

    if (started_from_generate) {
        fs::create_directories(options.pathname + std::to_string(options.steps - 1) + "b");
        fs::create_directories(options.pathname + std::to_string(options.steps - 2) + "b");
    }
    const std::string final_raw_path = options.pathname + std::to_string(options.steps - 2);
    if (fs::exists(final_raw_path)) {
        fs::remove(final_raw_path);
    }

    std::unordered_map<uint32_t, MatchCache> match_dict;
    bool started = false;
    const uint32_t progress_total = build_progress_total(options);

    for (int step = options.steps - 3; step >= 0; --step) {
        const std::string solved_folder = options.pathname + std::to_string(step) + "b";
        if (fs::exists(solved_folder)) {
            if (options.compress) {
                maybe_do_compress_ad(options.pathname + std::to_string(step + 2) + "b");
            }
            debug_log("skipping step " + std::to_string(step));
            continue;
        }

        if (!started) {
            started = true;
            if (step != options.steps - 3 || !started_from_generate) {
                dict_fromfile(options, step + 2, book_dict2, ind_dict2);
                AdSolveWorkspace<T> remove_workspace;
                remove_died_ad(book_dict2, ind_dict2, zero_val, remove_workspace);
            }
        }
        FormationProgress::update_build_progress(
            progress_total - static_cast<uint32_t>(step) - 2U,
            progress_total
        );

        const uint32_t original_board_sum = static_cast<uint32_t>(2 * step) + ini_board_sum;
        if (options.compress_temp_files) {
            maybe_decompress_with_7z(options.pathname + std::to_string(step) + ".7z");
        }
        std::vector<uint64_t> d0 = read_binary_vector<uint64_t>(options.pathname + std::to_string(step));

        BookStore<T> book_dict0_dummy;
        IndexStore ind_dict0;
        clear_book_store(book_dict0_dummy);
        clear_index_store(ind_dict0);
        expand_ad(
            d0,
            original_board_sum,
            masker.tiles_combination_table,
            param,
            book_dict0_dummy,
            ind_dict0,
            false,
            num_threads
        );
        d0.clear();
        d0.shrink_to_fit();

        std::vector<int> ind_dict0_keys = write_ind_chunked(ind_dict0, options, step);
        clear_index_store(ind_dict0);

        if (indind_dict1) {
            indind_dict2 = std::move(indind_dict1);
        } else {
            indind_dict2 = create_index_ad(ind_dict2, num_threads);
        }

        double timer_acc = 0.0;
        size_t counter_acc = 0;
        T max_rate = zero_val;

        iter_ind_dict4(
            ind_dict0_keys,
            book_dict2,
            ind_dict2,
            *indind_dict2,
            match_dict,
            masker.tiles_combination_table,
            masker.permutation_table,
            param,
            original_board_sum,
            spec,
            max_scale,
            zero_val,
            options.spawn_rate4,
            options,
            step,
            timer_acc,
            counter_acc,
            max_rate,
            num_threads
        );

        if (options.deletion_threshold > 0.0) {
            AdSolveWorkspace<T> remove_workspace2;
            remove_died_ad(book_dict2, ind_dict2, deletion_threshold, remove_workspace2);
        }
        dict_tofile(book_dict2, ind_dict2, options, step + 2, true);

        dict_fromfile(options, step + 1, book_dict1, ind_dict1);
        AdSolveWorkspace<T> remove_workspace1;
        remove_died_ad(book_dict1, ind_dict1, zero_val, remove_workspace1);
        indind_dict1 = create_index_ad(ind_dict1, num_threads);

        iter_ind_dict2(
            ind_dict0_keys,
            book_dict1,
            ind_dict1,
            *indind_dict1,
            match_dict,
            masker.tiles_combination_table,
            masker.permutation_table,
            param,
            original_board_sum,
            spec,
            max_scale,
            zero_val,
            options.spawn_rate4,
            options,
            step,
            timer_acc,
            counter_acc,
            max_rate,
            num_threads
        );

        const fs::path temp_folder = options.pathname + std::to_string(step) + "bt";
        const fs::path final_folder = options.pathname + std::to_string(step) + "b";
        if (fs::exists(final_folder)) {
            fs::remove_all(final_folder);
        }
        fs::rename(temp_folder, final_folder);

        const std::string raw_path = options.pathname + std::to_string(step);
        if (fs::exists(raw_path)) {
            fs::remove(raw_path);
        }

        const double avg_speed = round_to_2(static_cast<double>(counter_acc) / std::max(timer_acc, 0.000001) / 2e6);
        {
            std::ostringstream oss;
            oss << "step " << step << " done, solving avg " << avg_speed << " mbps\n";
            debug_log(oss.str());
        }
        {
            std::ofstream file(options.pathname + "stats.txt", std::ios::app);
            file << step << ","
                 << counter_acc << ","
                 << static_cast<double>(max_rate - zero_val) / static_cast<double>(max_scale - zero_val) << ","
                 << avg_speed << " mbps,"
                 << options.deletion_threshold << ","
                 << now_string() << "\n";
        }

        book_dict2 = std::move(book_dict1);
        ind_dict2 = std::move(ind_dict1);
    }
}

template <typename T>
void recalculate_process_ad_impl(
    const std::vector<uint64_t> &arr_init,
    const AdvancedPatternSpec &spec,
    const RunOptions &options,
    bool started_from_generate
) {
    ensure_stats_header<T>(options);
    const AdvancedMaskParam param = FormationAD::build_mask_param(spec);
    const FormationAD::MaskerContext masker = FormationAD::init_masker(spec);
    const uint32_t ini_board_sum = arr_init.empty() ? 0U : board_sum(arr_init.front());
    const T max_scale = max_scale_value<T>();
    const T zero_val = zero_value<T>();
    const T deletion_threshold = static_cast<T>(options.deletion_threshold * static_cast<double>(max_scale - zero_val) + zero_val);
    const int num_threads = effective_num_threads(options);

    BookStore<T> book_dict1;
    BookStore<T> book_dict2;
    IndexStore ind_dict1;
    IndexStore ind_dict2;
    std::unique_ptr<PrefixStore> indind_dict1;
    std::unique_ptr<PrefixStore> indind_dict2;
    clear_book_store(book_dict1);
    clear_book_store(book_dict2);
    clear_index_store(ind_dict1);
    clear_index_store(ind_dict2);

    if (started_from_generate) {
        fs::create_directories(options.pathname + std::to_string(options.steps - 1) + "b");
        fs::create_directories(options.pathname + std::to_string(options.steps - 2) + "b");
    }
    const std::string final_raw_path = options.pathname + std::to_string(options.steps - 2);
    if (fs::exists(final_raw_path)) {
        fs::remove(final_raw_path);
    }

    std::unordered_map<uint32_t, MatchCache> match_dict;
    bool started = false;
    bool has_prefix1 = false;
    const uint32_t progress_total = build_progress_total(options);

    for (int step = options.steps - 3; step >= 0; --step) {
        const std::string solved_folder = options.pathname + std::to_string(step) + "b";
        if (fs::exists(solved_folder)) {
            if (options.compress) {
                maybe_do_compress_ad(options.pathname + std::to_string(step + 2) + "b");
            }
            debug_log("skipping step " + std::to_string(step));
            continue;
        }

        if (!started) {
            started = true;
            if (step != options.steps - 3 || !started_from_generate) {
                dict_fromfile(options, step + 1, book_dict1, ind_dict1);
                dict_fromfile(options, step + 2, book_dict2, ind_dict2);
            }
        }
        FormationProgress::update_build_progress(
            progress_total - static_cast<uint32_t>(step) - 2U,
            progress_total
        );

        if (options.compress_temp_files) {
            maybe_decompress_with_7z(options.pathname + std::to_string(step) + ".7z");
        }
        std::vector<uint64_t> d0 = read_binary_vector<uint64_t>(options.pathname + std::to_string(step));
        double t0 = wall_time_seconds();

        BookStore<T> book_dict0;
        IndexStore ind_dict0;
        clear_book_store(book_dict0);
        clear_index_store(ind_dict0);
        expand_ad(
            d0,
            static_cast<uint32_t>(2 * step) + ini_board_sum,
            masker.tiles_combination_table,
            param,
            book_dict0,
            ind_dict0,
            true,
            num_threads
        );
        d0.clear();
        d0.shrink_to_fit();

        if (has_prefix1) {
            indind_dict2 = std::move(indind_dict1);
        } else {
            indind_dict2 = create_index_ad(ind_dict2, num_threads);
        }
        indind_dict1 = create_index_ad(ind_dict1, num_threads);
        has_prefix1 = true;
        double t1 = wall_time_seconds();

        recalculate_ad(
            book_dict0,
            ind_dict0,
            book_dict1,
            ind_dict1,
            *indind_dict1,
            book_dict2,
            ind_dict2,
            *indind_dict2,
            masker.tiles_combination_table,
            masker.permutation_table,
            param,
            static_cast<uint32_t>(2 * step) + ini_board_sum,
            spec,
            max_scale,
            zero_val,
            options.spawn_rate4,
            num_threads,
            match_dict
        );
        auto [length, max_rate] = length_count(book_dict0);
        double normalized_max_rate = static_cast<double>(max_rate - zero_val) / static_cast<double>(max_scale - zero_val);
        double t2 = wall_time_seconds();

        if (options.deletion_threshold > 0.0) {
            AdSolveWorkspace<T> remove_workspace2;
            remove_died_ad(book_dict2, ind_dict2, deletion_threshold, remove_workspace2);
        }
        AdSolveWorkspace<T> remove_workspace0;
        remove_died_ad(book_dict0, ind_dict0, zero_val, remove_workspace0);
        double t3 = wall_time_seconds();
        log_recalculate_performance(step, t0, t1, t2, t3, length);

        {
            std::ofstream file(options.pathname + "stats.txt", std::ios::app);
            file << step << ","
                 << length << ","
                 << normalized_max_rate << ","
                 << round_to_2(static_cast<double>(length) / std::max(t3 - t0, 0.01) / 1e6) << " mbps,"
                 << options.deletion_threshold << ","
                 << now_string() << "\n";
        }

        if (options.deletion_threshold > 0.0 || options.compress) {
            dict_tofile(book_dict2, ind_dict2, options, step + 2, true);
        }
        dict_tofile(book_dict0, ind_dict0, options, step, false);

        const std::string raw_path = options.pathname + std::to_string(step);
        if (fs::exists(raw_path)) {
            fs::remove(raw_path);
        }
        debug_log("step " + std::to_string(step) + " written\n");

        book_dict2 = std::move(book_dict1);
        ind_dict2 = std::move(ind_dict1);
        book_dict1 = std::move(book_dict0);
        ind_dict1 = std::move(ind_dict0);
    }
}

} // namespace

void run_pattern_solve_ad_cpp(
    const std::vector<uint64_t> &arr_init,
    const AdvancedPatternSpec &spec,
    const RunOptions &options,
    bool started_from_generate
) {
    (void)HybridSearch::mode();
    switch (success_rate_kind_from_name(options.success_rate_dtype)) {
        case SuccessRateKind::UInt64:
            if (options.chunked_solve) {
                recalculate_process_ad_chunked_impl<uint64_t>(arr_init, spec, options, started_from_generate);
            } else {
                recalculate_process_ad_impl<uint64_t>(arr_init, spec, options, started_from_generate);
            }
            return;
        case SuccessRateKind::Float32:
            if (options.chunked_solve) {
                recalculate_process_ad_chunked_impl<float>(arr_init, spec, options, started_from_generate);
            } else {
                recalculate_process_ad_impl<float>(arr_init, spec, options, started_from_generate);
            }
            return;
        case SuccessRateKind::Float64:
            if (options.chunked_solve) {
                recalculate_process_ad_chunked_impl<double>(arr_init, spec, options, started_from_generate);
            } else {
                recalculate_process_ad_impl<double>(arr_init, spec, options, started_from_generate);
            }
            return;
        case SuccessRateKind::UInt32:
        default:
            if (options.chunked_solve) {
                recalculate_process_ad_chunked_impl<uint32_t>(arr_init, spec, options, started_from_generate);
            } else {
                recalculate_process_ad_impl<uint32_t>(arr_init, spec, options, started_from_generate);
            }
            return;
    }
}
