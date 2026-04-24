#include "BookSolver.h"

#include "BoardMaskerAD.h"
#include "BoardMoverAD.h"
#include "BookGenerator.h"
#include "BookGeneratorUtils.h"
#include "Calculator.h"
#include "CompressionBridge.h"
#include "Formation.h"
#include "HybridSearch.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <immintrin.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <numeric>
#include <omp.h>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
namespace nb = nanobind;

namespace {

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

void log_generation_performance(int step_index, double t0, double t1, double t2, double t3, size_t layer_size) {
    if (t3 <= t0) {
        return;
    }
    double total = t3 - t0;
    std::ostringstream speed_stream;
    speed_stream << "step " << step_index << " generated: "
                 << round_to_2(static_cast<double>(layer_size) / total / 1e6)
                 << " mbps";
    debug_log(speed_stream.str());

    std::ostringstream phase_stream;
    phase_stream << "generate/sort/deduplicate: "
                 << round_to_2((t1 - t0) / total) << "/"
                 << round_to_2((t2 - t1) / total) << "/"
                 << round_to_2((t3 - t2) / total) << "\n";
    debug_log(phase_stream.str());
}

void clear_u64_buffer(std::vector<uint64_t> &buffer, int num_threads) {
    if (buffer.empty()) {
        return;
    }
    if (num_threads <= 1 || buffer.size() < 1048576ULL) {
        std::fill(buffer.begin(), buffer.end(), 0ULL);
        return;
    }
    #pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < static_cast<int64_t>(buffer.size()); ++i) {
        buffer[static_cast<size_t>(i)] = 0ULL;
    }
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

std::vector<uint64_t> read_raw_file(const std::string &path) {
    std::vector<uint64_t> data;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return data;
    }
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    data.resize(size / sizeof(uint64_t));
    if (!data.empty()) {
        file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size));
    }
    return data;
}

void write_raw_file(const std::string &path, const std::vector<uint64_t> &data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (out && !data.empty()) {
        out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(uint64_t)));
    }
}

std::vector<std::vector<double>> load_length_factors(const std::string &path, double default_value) {
    std::vector<std::vector<double>> result;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            if (!value.empty()) {
                row.push_back(std::stod(value));
            }
        }
        if (!row.empty()) {
            result.push_back(std::move(row));
        }
    }
    if (result.empty()) {
        result.push_back({default_value, default_value, default_value});
    }
    return result;
}

void save_length_factors(const std::string &path, const std::vector<std::vector<double>> &lists) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        return;
    }
    out << std::fixed << std::setprecision(6);
    for (const auto &row : lists) {
        for (size_t i = 0; i < row.size(); ++i) {
            out << row[i];
            if (i + 1 != row.size()) {
                out << ",";
            }
        }
        out << "\n";
    }
}

struct InitParams {
    double length_factor = 3.2;
    std::vector<double> length_factors;
    std::vector<std::vector<double>> length_factors_list;
    std::string length_factors_list_path;
    double length_factor_multiplier = 2.0;
    size_t segment_size = 0;
};

InitParams initialize_parameters_internal(const std::string &pathname, bool isfree) {
    InitParams params;
    fs::path base_path(pathname);
    params.length_factors_list_path = (base_path.parent_path() / "length_factors_list.txt").string();
    params.length_factors_list = load_length_factors(params.length_factors_list_path, 3.2);
    params.length_factors = BookGenerator::harmonic_mean_by_column(params.length_factors_list);
    if (params.length_factors.empty()) {
        params.length_factors = {3.2, 3.2, 3.2};
    }
    params.length_factor = BookGenerator::predict_next_length_factor_quadratic(params.length_factors) * 1.2;
    double ram_gb = std::round(BookGenerator::get_system_memory_gb());
    size_t base_unit = isfree ? 5120000ULL : 8192000ULL;
    params.segment_size = base_unit * static_cast<size_t>(std::max(1.0, ram_gb - 5.0));
    return params;
}

std::vector<std::vector<double>> split_length_factor_list(const std::vector<std::vector<double>> &length_factor_list) {
    std::vector<std::vector<double>> result;
    if (length_factor_list.empty()) {
        return result;
    }
    std::vector<double> head = length_factor_list.front();
    for (double &value : head) {
        value *= 1.5;
    }
    result.push_back(std::move(head));
    for (const auto &row : length_factor_list) {
        result.push_back(row);
        result.push_back(row);
    }
    if (!result.empty()) {
        result.pop_back();
    }
    return result;
}

std::vector<std::vector<double>> reverse_split_length_factor_list(const std::vector<std::vector<double>> &length_factor_list) {
    std::vector<std::vector<double>> result;
    for (const auto &row : length_factor_list) {
        std::vector<double> reduced;
        for (size_t i = 0; i < row.size(); i += 2) {
            reduced.push_back(row[i]);
        }
        result.push_back(std::move(reduced));
    }
    return result;
}

std::vector<size_t> allocate_seg(const std::vector<std::vector<double>> &length_factors_list, size_t arr_length) {
    if (length_factors_list.size() == 1) {
        return {0, arr_length};
    }
    std::vector<double> factors;
    factors.reserve(length_factors_list.size());
    for (const auto &row : length_factors_list) {
        factors.push_back(row.empty() ? 1.5 : row.back());
    }
    for (double &value : factors) {
        value = 1.0 / (value + 0.2);
    }
    double total = std::accumulate(factors.begin(), factors.end(), 0.0);
    std::vector<size_t> result = {0};
    double cumulative = 0.0;
    for (double value : factors) {
        cumulative += value / total;
        result.push_back(static_cast<size_t>(cumulative * static_cast<double>(arr_length)));
    }
    result.back() = arr_length;
    return result;
}

struct RestartResult {
    bool run = false;
    bool started = false;
    std::vector<uint64_t> d0;
    std::vector<uint64_t> d1;
};

struct DeriveResult {
    bool is_valid = false;
    bool is_derived = false;
    uint8_t count = 0;
    std::array<uint64_t, 120> boards{};
};

void validate_length_and_balance(
    size_t len_d0,
    size_t len_d2,
    size_t len_d1t,
    const std::vector<size_t> &counts1,
    const std::vector<size_t> &counts2,
    double length_factor,
    bool is_big
) {
    if (len_d0 < 99999 || len_d2 < 99999 || counts1.empty() || counts2.empty()) {
        return;
    }
    size_t length_needed = std::max(
        *std::max_element(counts1.begin(), counts1.end()),
        *std::max_element(counts2.begin(), counts2.end())
    ) * counts1.size();
    double length_factor_actual = static_cast<double>(length_needed) / static_cast<double>(len_d0);
    size_t length = std::max<size_t>(6999999ULL, static_cast<size_t>(static_cast<double>(len_d0) * length_factor));
    if (length_needed > length) {
        std::ostringstream oss;
        oss << "length multiplier " << length_factor << ", need " << length_factor_actual;
        throw std::runtime_error(oss.str());
    }
    if (is_big || len_d0 == 0) {
        return;
    }
    std::ostringstream oss;
    oss << "length " << len_d1t << ", " << len_d2
        << ", Using " << round_to_2(length_factor)
        << ", Need " << round_to_2(length_factor_actual);
    debug_log(oss.str());
}

std::tuple<std::vector<double>, std::vector<std::vector<double>>> update_parameters(
    size_t len_d0,
    size_t len_d2,
    std::vector<double> length_factors,
    const std::string &path
) {
    if (!length_factors.empty()) {
        length_factors.erase(length_factors.begin());
    }
    length_factors.push_back(static_cast<double>(len_d2) / static_cast<double>(len_d0 + 1));
    std::vector<std::vector<double>> list = {length_factors};
    save_length_factors(path, list);
    return {length_factors, list};
}

std::pair<std::vector<std::vector<double>>, double> update_parameters_big(
    const std::vector<uint64_t> &actual_lengths2,
    const std::vector<uint64_t> &actual_lengths1,
    int n,
    std::vector<std::vector<double>> length_factors_list
) {
    double sum2 = std::accumulate(actual_lengths2.begin(), actual_lengths2.end(), 0.0);
    double sum1 = std::accumulate(actual_lengths1.begin(), actual_lengths1.end(), 0.0);
    double mean_ratio2 = 0.0;
    double mean_ratio1 = 0.0;
    double max_ratio2 = 0.0;
    double max_ratio1 = 0.0;
    if (sum2 > 0.0) {
        for (uint64_t value : actual_lengths2) {
            double ratio = static_cast<double>(value) / sum2;
            mean_ratio2 += ratio;
            max_ratio2 = std::max(max_ratio2, ratio);
        }
        mean_ratio2 /= static_cast<double>(actual_lengths2.size());
    }
    if (sum1 > 0.0) {
        for (uint64_t value : actual_lengths1) {
            double ratio = static_cast<double>(value) / sum1;
            mean_ratio1 += ratio;
            max_ratio1 = std::max(max_ratio1, ratio);
        }
        mean_ratio1 /= static_cast<double>(actual_lengths1.size());
    }
    double length_factor_multiplier = std::max(
        mean_ratio2 > 0.0 ? max_ratio2 / mean_ratio2 : 1.0,
        mean_ratio1 > 0.0 ? max_ratio1 / mean_ratio1 : 1.0
    );

    double total_ram = std::round(BookGenerator::get_system_memory_gb());
    double threshold_big = 20971520.0 * (total_ram * 0.75);
    double threshold_small = 524288.0 * (total_ram * 0.75);
    double mean_actual2 = actual_lengths2.empty()
        ? 0.0
        : std::accumulate(actual_lengths2.begin(), actual_lengths2.end(), 0.0) / static_cast<double>(actual_lengths2.size());
    if (mean_actual2 * n > threshold_big) {
        length_factors_list = split_length_factor_list(length_factors_list);
    }
    if (mean_actual2 * n < threshold_small) {
        length_factors_list = reverse_split_length_factor_list(length_factors_list);
    }
    return {length_factors_list, length_factor_multiplier};
}

RestartResult handle_restart_ad(int step_index, const std::string &pathname, const std::vector<uint64_t> &arr_init, bool started) {
    const std::string path_i = pathname + std::to_string(step_index);
    const std::string path_i_plus_1 = pathname + std::to_string(step_index + 1);
    const std::string path_i_minus_1 = pathname + std::to_string(step_index - 1);
    if ((fs::exists(path_i_plus_1) && fs::exists(path_i)) ||
        (fs::exists(path_i_plus_1 + "b") && fs::exists(path_i)) ||
        (fs::exists(path_i_plus_1 + ".z") && fs::exists(path_i)) ||
        fs::exists(path_i + "b") ||
        fs::exists(path_i + ".z") ||
        fs::exists(path_i + "b.7z") ||
        fs::exists(path_i + ".7z")) {
        debug_log("skipping step " + std::to_string(step_index));
        return {};
    }
    if (step_index == 1) {
        write_raw_file(path_i_minus_1, arr_init);
        return {true, true, arr_init, {}};
    }
    if (!started) {
        return {true, true, read_raw_file(path_i_minus_1), read_raw_file(path_i)};
    }
    return {true, true, {}, {}};
}

uint8_t derive_3x64(uint64_t masked, const std::array<uint64_t, 16> &pos_32k, int8_t count_32k, std::array<uint64_t, 120> &out) {
    uint8_t count = 0;
    for (int idx = 0; idx < count_32k; ++idx) {
        uint64_t pos = pos_32k[static_cast<size_t>(idx)];
        uint64_t tile_value = (masked >> pos) & 0xFULL;
        if (tile_value == 6ULL) {
            continue;
        }
        uint64_t derived = masked & ~(0xFULL << pos);
        derived |= (6ULL << pos);
        out[static_cast<size_t>(count++)] = derived;
    }
    return count;
}

DeriveResult derive(uint64_t board, uint32_t original_board_sum, const FormationAD::TilesCombinationTable &tiles_table, const AdvancedMaskParam &param) {
    FormationAD::TileCount3Result stats = FormationAD::tile_sum_and_32k_count3(board, param);
    if (stats.total_sum >= param.small_tile_sum_limit + 64U) {
        return {};
    }
    uint32_t large_tiles_sum = original_board_sum - stats.total_sum
        - (static_cast<uint32_t>(param.num_free_32k + param.num_fixed_32k) << 15U);
    auto tiles_combinations = FormationAD::tiles_combination_view(
        tiles_table,
        static_cast<uint8_t>(large_tiles_sum >> 6U),
        static_cast<uint8_t>(stats.count_32k - static_cast<int8_t>(param.num_free_32k))
    );
    if (tiles_combinations.empty()) {
        return {};
    }
    if (tiles_combinations[tiles_combinations.size - 1] == param.target) {
        return {};
    }
    if (stats.count_32k - static_cast<int8_t>(param.num_free_32k) < 2) {
        DeriveResult result;
        result.is_valid = true;
        return result;
    }

    if (tiles_combinations[0] == tiles_combinations[1]) {
        if (stats.total_sum >= param.small_tile_sum_limit) {
            return {};
        }
        if (tiles_combinations.size > 2 && tiles_combinations[0] == tiles_combinations[2]) {
            if (stats.total_sum >= param.small_tile_sum_limit - 64U) {
                return {};
            }
            if (stats.tile64_count == 0) {
                DeriveResult result;
                result.is_valid = true;
                result.is_derived = true;
                return result;
            }
            DeriveResult result;
            result.is_valid = true;
            result.is_derived = true;
            result.count = derive_3x64(board, stats.pos_32k, stats.count_32k, result.boards);
            return result;
        }
        DeriveResult result;
        result.is_valid = true;
        result.is_derived = true;
        for (int pos1 = 0; pos1 < stats.count_32k - 1; ++pos1) {
            for (int pos2 = pos1 + 1; pos2 < stats.count_32k; ++pos2) {
                uint64_t derived = board & ~(0xFULL << stats.pos_32k[static_cast<size_t>(pos1)]);
                derived |= static_cast<uint64_t>(tiles_combinations[0]) << stats.pos_32k[static_cast<size_t>(pos1)];
                derived &= ~(0xFULL << stats.pos_32k[static_cast<size_t>(pos2)]);
                derived |= static_cast<uint64_t>(tiles_combinations[0]) << stats.pos_32k[static_cast<size_t>(pos2)];
                result.boards[static_cast<size_t>(result.count++)] = derived;
            }
        }
        return result;
    }

    if (stats.masked_board != board) {
        DeriveResult result;
        result.is_valid = true;
        result.is_derived = true;
        result.count = 1;
        result.boards[0] = stats.masked_board;
        return result;
    }
    DeriveResult result;
    result.is_valid = true;
    return result;
}

std::vector<uint64_t> validate_layer(const std::vector<uint64_t> &arr, uint32_t original_board_sum, const FormationAD::TilesCombinationTable &tiles_table, const AdvancedMaskParam &param, int num_threads) {
    std::vector<uint8_t> valid(arr.size(), 0);
    #pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < static_cast<int64_t>(arr.size()); ++i) {
        valid[static_cast<size_t>(i)] = FormationAD::validate(arr[static_cast<size_t>(i)], original_board_sum, tiles_table, param) ? 1U : 0U;
    }
    std::vector<uint64_t> result;
    result.reserve(arr.size());
    for (size_t i = 0; i < arr.size(); ++i) {
        if (valid[i] != 0U) {
            result.push_back(arr[i]);
        }
    }
    return result;
}

struct GenBoardsAdResult {
    std::vector<uint64_t> arr1;
    std::vector<uint64_t> arr2;
    std::vector<size_t> counts1;
    std::vector<size_t> counts2;
};

using GenBoardsAdFn = GenBoardsAdResult (*)(
    ArrayView<const uint64_t>,
    const AdvancedPatternSpec &,
    std::vector<uint64_t> &,
    std::vector<uint64_t> &,
    uint32_t,
    const FormationAD::TilesCombinationTable &,
    const AdvancedMaskParam &,
    int,
    double,
    bool
);

enum class AdGenMode : uint8_t {
    Scalar = 0,
    AVX512 = 1,
};

struct GenBoardsAdDispatch {
    AdGenMode mode = AdGenMode::Scalar;
    GenBoardsAdFn fn = nullptr;
};

constexpr size_t kAdGenBatchSize = 128;

enum class BufferedBaseKind : uint8_t {
    Plain = 0,
    Derive = 1,
};

struct BufferedAcceptedBoards {
    std::array<uint64_t, kAdGenBatchSize> boards{};
    std::array<uint8_t, kAdGenBatchSize> kinds{};
    size_t count = 0;
};

void process_buffered_accepted_boards(
    const BufferedAcceptedBoards &accepted,
    uint32_t original_board_sum,
    uint32_t spawn_delta,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    const AdvancedPatternSpec &spec,
    uint64_t *target_arr,
    size_t &counter
) {
    for (size_t accepted_index = 0; accepted_index < accepted.count; ++accepted_index) {
        const uint64_t canon = accepted.boards[accepted_index];
        if (accepted.kinds[accepted_index] == static_cast<uint8_t>(BufferedBaseKind::Plain)) {
            target_arr[counter++] = canon;
            continue;
        }
        DeriveResult derived = derive(canon, original_board_sum + spawn_delta, tiles_table, param);
        if (!derived.is_valid) {
            continue;
        }
        if (!derived.is_derived) {
            target_arr[counter++] = canon;
            continue;
        }
        for (uint8_t derived_index = 0; derived_index < derived.count; ++derived_index) {
            const uint64_t derived_board = derived.boards[static_cast<size_t>(derived_index)];
            if (is_pattern(derived_board, spec.pattern_masks)) {
                target_arr[counter++] = apply_canonical(derived_board, spec.symm_mode);
            }
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512dq,avx512bw,avx512vl")))
#endif
inline __m512i simd_hash_ad_avx512(__m512i v) {
    __m512i v_xor = _mm512_xor_epi64(v, _mm512_srli_epi64(v, 27));
    __m512i v_mul = _mm512_mullo_epi64(v_xor, _mm512_set1_epi64(0x1A85EC53ULL));
    __m512i v_mix = _mm512_add_epi64(_mm512_add_epi64(v_mul, _mm512_srli_epi64(v, 23)), v);
    __m512i v_mix_xor = _mm512_xor_epi64(v_mix, _mm512_srli_epi64(v_mix, 27));
    __m512i v_mix_mul = _mm512_mullo_epi64(v_mix_xor, _mm512_set1_epi64(0x1A85EC53ULL));
    return _mm512_add_epi64(_mm512_add_epi64(v_mix_mul, _mm512_srli_epi64(v_mix, 23)), v_mix);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512dq,avx512bw,avx512vl")))
#endif
void flush_candidate_buffer_avx512(
    const uint64_t *board_buffer,
    const uint8_t *kind_buffer,
    size_t active_count,
    uint64_t *hashmap,
    uint64_t hashmask,
    BufferedAcceptedBoards &accepted
) {
    accepted.count = 0;
    for (size_t i = 0; i < active_count; i += 8) {
        const size_t remaining = std::min<size_t>(8, active_count - i);
        const __mmask8 active_mask = static_cast<__mmask8>((1u << static_cast<unsigned>(remaining)) - 1u);
        const __m512i states = _mm512_maskz_loadu_epi64(active_mask, board_buffer + i);
        const __m512i hashes = simd_hash_ad_avx512(states);
        const __m512i idx = _mm512_and_epi64(hashes, _mm512_set1_epi64(static_cast<long long>(hashmask)));
        alignas(64) std::array<uint64_t, 8> idx_values{};
        _mm512_storeu_si512(idx_values.data(), idx);

        for (size_t lane = 0; lane < remaining; ++lane) {
            const uint64_t board = board_buffer[i + lane];
            const size_t hash_index = static_cast<size_t>(idx_values[lane]);
            if (hashmap[hash_index] == board) {
                continue;
            }
            hashmap[hash_index] = board;
            accepted.boards[accepted.count] = board;
            accepted.kinds[accepted.count] = kind_buffer[i + lane];
            ++accepted.count;
        }
    }
}

void flush_candidate_buffer_scalar(
    const uint64_t *board_buffer,
    const uint8_t *kind_buffer,
    size_t active_count,
    uint64_t *hashmap,
    uint64_t hashmask,
    BufferedAcceptedBoards &accepted
) {
    accepted.count = 0;
    std::array<uint64_t, kAdGenBatchSize> hashed_idx{};
    for (size_t i = 0; i < active_count; ++i) {
        hashed_idx[i] = BookGeneratorUtils::hash_board(board_buffer[i]) & hashmask;
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(hashmap + static_cast<size_t>(hashed_idx[i]), 1, 1);
#endif
    }

    for (size_t i = 0; i < active_count; ++i) {
        constexpr size_t kPrefetchDistance = 8;
        if (i + kPrefetchDistance < active_count) {
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(hashmap + static_cast<size_t>(hashed_idx[i + kPrefetchDistance]), 1, 1);
#endif
        }
        const uint64_t board = board_buffer[i];
        const size_t hash_index = static_cast<size_t>(hashed_idx[i]);
        if (hashmap[hash_index] == board) {
            continue;
        }
        hashmap[hash_index] = board;
        accepted.boards[accepted.count] = board;
        accepted.kinds[accepted.count] = kind_buffer[i];
        ++accepted.count;
    }
}

GenBoardsAdResult gen_boards_ad_scalar(
    ArrayView<const uint64_t> arr0,
    const AdvancedPatternSpec &spec,
    std::vector<uint64_t> &hashmap1,
    std::vector<uint64_t> &hashmap2,
    uint32_t original_board_sum,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    int n,
    double length_factor,
    bool isfree
) {
    const size_t min_length = isfree ? 9999999ULL : 6999999ULL;
    const size_t length = std::max(min_length, static_cast<size_t>(static_cast<double>(arr0.size) * length_factor));
    auto arr1 = std::unique_ptr<uint64_t[]>(new uint64_t[length]);
    auto arr2 = std::unique_ptr<uint64_t[]>(new uint64_t[length]);
    std::vector<size_t> starts(static_cast<size_t>(n));
    std::vector<size_t> c1(static_cast<size_t>(n));
    std::vector<size_t> c2(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        starts[static_cast<size_t>(i)] = (length / static_cast<size_t>(n)) * static_cast<size_t>(i);
        c1[static_cast<size_t>(i)] = starts[static_cast<size_t>(i)];
        c2[static_cast<size_t>(i)] = starts[static_cast<size_t>(i)];
    }
    uint64_t hashmap1_mask = static_cast<uint64_t>(hashmap1.size() - 1);
    uint64_t hashmap2_mask = static_cast<uint64_t>(hashmap2.size() - 1);

    const size_t total_tasks = arr0.size;
    const size_t chunk_size = std::min<size_t>(1000000ULL, total_tasks / static_cast<size_t>(n * 5) + 1ULL) * static_cast<size_t>(n);
    const size_t chunks_count = (total_tasks + chunk_size - 1) / chunk_size;

    #pragma omp parallel for num_threads(n)
    for (int s = 0; s < n; ++s) {
        size_t c1t = starts[static_cast<size_t>(s)];
        size_t c2t = starts[static_cast<size_t>(s)];
        std::array<uint64_t, kAdGenBatchSize> buffer1{};
        std::array<uint8_t, kAdGenBatchSize> kinds1{};
        std::array<uint64_t, kAdGenBatchSize> buffer2{};
        std::array<uint8_t, kAdGenBatchSize> kinds2{};
        size_t buffer1_count = 0;
        size_t buffer2_count = 0;
        BufferedAcceptedBoards accepted{};

        auto flush_spawn2 = [&]() {
            if (buffer1_count == 0) {
                return;
            }
            flush_candidate_buffer_scalar(buffer1.data(), kinds1.data(), buffer1_count, hashmap1.data(), hashmap1_mask, accepted);
            process_buffered_accepted_boards(accepted, original_board_sum, 2U, tiles_table, param, spec, arr1.get(), c1t);
            buffer1_count = 0;
        };

        auto flush_spawn4 = [&]() {
            if (buffer2_count == 0) {
                return;
            }
            flush_candidate_buffer_scalar(buffer2.data(), kinds2.data(), buffer2_count, hashmap2.data(), hashmap2_mask, accepted);
            process_buffered_accepted_boards(accepted, original_board_sum, 4U, tiles_table, param, spec, arr2.get(), c2t);
            buffer2_count = 0;
        };

        for (size_t chunk = 0; chunk < chunks_count; ++chunk) {
            size_t chunk_start = chunk * chunk_size;
            size_t chunk_end = std::min(chunk_start + chunk_size, total_tasks);
            size_t thread_start = chunk_start + static_cast<size_t>(s) * chunk_size / static_cast<size_t>(n);
            size_t thread_end = thread_start + chunk_size / static_cast<size_t>(n);
            size_t start = std::max(thread_start, chunk_start);
            size_t end = std::min(thread_end, chunk_end);
            if (!(start < end)) {
                continue;
            }

            for (size_t b = start; b < end; ++b) {
                uint64_t t = arr0[b];
                for (int i = 0; i < 16; ++i) {
                    if (((t >> static_cast<uint64_t>(4 * i)) & 0xFULL) != 0ULL) {
                        continue;
                    }
                    uint64_t t1 = t | (1ULL << static_cast<uint64_t>(4 * i));
                    uint64_t t1_rev = FormationAD::reverse(t1);
                    uint64_t md2 = FormationAD::m_move_down(t1, t1_rev);
                    uint64_t mr2 = FormationAD::m_move_right(t1);
                    auto [ml2, mnt_h2] = FormationAD::m_move_left2(t1);
                    auto [mu2, mnt_v2] = FormationAD::m_move_up2(t1, t1_rev);
                    const uint64_t moves2_boards[4] = {ml2, mr2, mu2, md2};
                    const bool moves2_mask_new_tile[4] = {mnt_h2, mnt_h2, mnt_v2, mnt_v2};
                    for (int move_index = 0; move_index < 4; ++move_index) {
                        uint64_t newt = moves2_boards[move_index];
                        if (newt == t1 || !is_pattern(newt, spec.pattern_masks)) {
                            continue;
                        }
                        buffer1[buffer1_count] = apply_canonical(newt, spec.symm_mode);
                        kinds1[buffer1_count] = static_cast<uint8_t>(
                            moves2_mask_new_tile[move_index] ? BufferedBaseKind::Derive : BufferedBaseKind::Plain
                        );
                        ++buffer1_count;
                        if (buffer1_count == kAdGenBatchSize) {
                            flush_spawn2();
                        }
                    }

                    t1 = t | (2ULL << static_cast<uint64_t>(4 * i));
                    uint64_t t1_rev4 = FormationAD::reverse(t1);
                    uint64_t md4 = FormationAD::m_move_down(t1, t1_rev4);
                    uint64_t mr4 = FormationAD::m_move_right(t1);
                    auto [ml4, mnt_h4] = FormationAD::m_move_left2(t1);
                    auto [mu4, mnt_v4] = FormationAD::m_move_up2(t1, t1_rev4);
                    const uint64_t moves4_boards[4] = {ml4, mr4, mu4, md4};
                    const bool moves4_mask_new_tile[4] = {mnt_h4, mnt_h4, mnt_v4, mnt_v4};
                    for (int move_index = 0; move_index < 4; ++move_index) {
                        uint64_t newt = moves4_boards[move_index];
                        if (newt == t1 || !is_pattern(newt, spec.pattern_masks)) {
                            continue;
                        }
                        buffer2[buffer2_count] = apply_canonical(newt, spec.symm_mode);
                        kinds2[buffer2_count] = static_cast<uint8_t>(
                            moves4_mask_new_tile[move_index] ? BufferedBaseKind::Derive : BufferedBaseKind::Plain
                        );
                        ++buffer2_count;
                        if (buffer2_count == kAdGenBatchSize) {
                            flush_spawn4();
                        }
                    }
                }
            }
        }
        flush_spawn2();
        flush_spawn4();
        c1[static_cast<size_t>(s)] = c1t;
        c2[static_cast<size_t>(s)] = c2t;
    }

    size_t total_arr1 = BookGeneratorUtils::merge_inplace(arr1.get(), c1, starts);
    size_t total_arr2 = BookGeneratorUtils::merge_inplace(arr2.get(), c2, starts);
    std::vector<uint64_t> arr1_vec(total_arr1);
    std::vector<uint64_t> arr2_vec(total_arr2);
    if (total_arr1 > 0) {
        std::memcpy(arr1_vec.data(), arr1.get(), total_arr1 * sizeof(uint64_t));
    }
    if (total_arr2 > 0) {
        std::memcpy(arr2_vec.data(), arr2.get(), total_arr2 * sizeof(uint64_t));
    }

    std::vector<size_t> counts1(static_cast<size_t>(n));
    std::vector<size_t> counts2(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        counts1[static_cast<size_t>(i)] = c1[static_cast<size_t>(i)] - starts[static_cast<size_t>(i)];
        counts2[static_cast<size_t>(i)] = c2[static_cast<size_t>(i)] - starts[static_cast<size_t>(i)];
    }
    return {std::move(arr1_vec), std::move(arr2_vec), std::move(counts1), std::move(counts2)};
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512dq,avx512bw,avx512vl")))
#endif
GenBoardsAdResult gen_boards_ad_avx512(
    ArrayView<const uint64_t> arr0,
    const AdvancedPatternSpec &spec,
    std::vector<uint64_t> &hashmap1,
    std::vector<uint64_t> &hashmap2,
    uint32_t original_board_sum,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    int n,
    double length_factor,
    bool isfree
) {
    const size_t min_length = isfree ? 9999999ULL : 6999999ULL;
    const size_t length = std::max(min_length, static_cast<size_t>(static_cast<double>(arr0.size) * length_factor));
    auto arr1 = std::unique_ptr<uint64_t[]>(new uint64_t[length]);
    auto arr2 = std::unique_ptr<uint64_t[]>(new uint64_t[length]);
    std::vector<size_t> starts(static_cast<size_t>(n));
    std::vector<size_t> c1(static_cast<size_t>(n));
    std::vector<size_t> c2(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        starts[static_cast<size_t>(i)] = (length / static_cast<size_t>(n)) * static_cast<size_t>(i);
        c1[static_cast<size_t>(i)] = starts[static_cast<size_t>(i)];
        c2[static_cast<size_t>(i)] = starts[static_cast<size_t>(i)];
    }
    const uint64_t hashmap1_mask = static_cast<uint64_t>(hashmap1.size() - 1);
    const uint64_t hashmap2_mask = static_cast<uint64_t>(hashmap2.size() - 1);

    const size_t total_tasks = arr0.size;
    const size_t chunk_size = std::min<size_t>(1000000ULL, total_tasks / static_cast<size_t>(n * 5) + 1ULL) * static_cast<size_t>(n);
    const size_t chunks_count = (total_tasks + chunk_size - 1) / chunk_size;

    #pragma omp parallel for num_threads(n)
    for (int s = 0; s < n; ++s) {
        size_t c1t = starts[static_cast<size_t>(s)];
        size_t c2t = starts[static_cast<size_t>(s)];
        std::array<uint64_t, kAdGenBatchSize> buffer1{};
        std::array<uint8_t, kAdGenBatchSize> kinds1{};
        std::array<uint64_t, kAdGenBatchSize> buffer2{};
        std::array<uint8_t, kAdGenBatchSize> kinds2{};
        size_t buffer1_count = 0;
        size_t buffer2_count = 0;
        BufferedAcceptedBoards accepted{};

        auto flush_spawn2 = [&]() {
            if (buffer1_count == 0) {
                return;
            }
            flush_candidate_buffer_avx512(buffer1.data(), kinds1.data(), buffer1_count, hashmap1.data(), hashmap1_mask, accepted);
            process_buffered_accepted_boards(accepted, original_board_sum, 2U, tiles_table, param, spec, arr1.get(), c1t);
            buffer1_count = 0;
        };

        auto flush_spawn4 = [&]() {
            if (buffer2_count == 0) {
                return;
            }
            flush_candidate_buffer_avx512(buffer2.data(), kinds2.data(), buffer2_count, hashmap2.data(), hashmap2_mask, accepted);
            process_buffered_accepted_boards(accepted, original_board_sum, 4U, tiles_table, param, spec, arr2.get(), c2t);
            buffer2_count = 0;
        };

        for (size_t chunk = 0; chunk < chunks_count; ++chunk) {
            const size_t chunk_start = chunk * chunk_size;
            const size_t chunk_end = std::min(chunk_start + chunk_size, total_tasks);
            const size_t thread_start = chunk_start + static_cast<size_t>(s) * chunk_size / static_cast<size_t>(n);
            const size_t thread_end = thread_start + chunk_size / static_cast<size_t>(n);
            const size_t start = std::max(thread_start, chunk_start);
            const size_t end = std::min(thread_end, chunk_end);
            if (!(start < end)) {
                continue;
            }

            for (size_t b = start; b < end; ++b) {
                const uint64_t t = arr0[b];
                for (int i = 0; i < 16; ++i) {
                    if (((t >> static_cast<uint64_t>(4 * i)) & 0xFULL) != 0ULL) {
                        continue;
                    }

                    uint64_t t1 = t | (1ULL << static_cast<uint64_t>(4 * i));
                    const uint64_t t1_rev = FormationAD::reverse(t1);
                    const uint64_t md2 = FormationAD::m_move_down(t1, t1_rev);
                    const uint64_t mr2 = FormationAD::m_move_right(t1);
                    const auto [ml2, mnt_h2] = FormationAD::m_move_left2(t1);
                    const auto [mu2, mnt_v2] = FormationAD::m_move_up2(t1, t1_rev);
                    const uint64_t moves2_boards[4] = {ml2, mr2, mu2, md2};
                    const bool moves2_mask_new_tile[4] = {mnt_h2, mnt_h2, mnt_v2, mnt_v2};
                    for (int move_index = 0; move_index < 4; ++move_index) {
                        const uint64_t newt = moves2_boards[move_index];
                        if (newt == t1 || !is_pattern(newt, spec.pattern_masks)) {
                            continue;
                        }
                        buffer1[buffer1_count] = apply_canonical(newt, spec.symm_mode);
                        kinds1[buffer1_count] = static_cast<uint8_t>(
                            moves2_mask_new_tile[move_index] ? BufferedBaseKind::Derive : BufferedBaseKind::Plain
                        );
                        ++buffer1_count;
                        if (buffer1_count == kAdGenBatchSize) {
                            flush_spawn2();
                        }
                    }

                    t1 = t | (2ULL << static_cast<uint64_t>(4 * i));
                    const uint64_t t1_rev4 = FormationAD::reverse(t1);
                    const uint64_t md4 = FormationAD::m_move_down(t1, t1_rev4);
                    const uint64_t mr4 = FormationAD::m_move_right(t1);
                    const auto [ml4, mnt_h4] = FormationAD::m_move_left2(t1);
                    const auto [mu4, mnt_v4] = FormationAD::m_move_up2(t1, t1_rev4);
                    const uint64_t moves4_boards[4] = {ml4, mr4, mu4, md4};
                    const bool moves4_mask_new_tile[4] = {mnt_h4, mnt_h4, mnt_v4, mnt_v4};
                    for (int move_index = 0; move_index < 4; ++move_index) {
                        const uint64_t newt = moves4_boards[move_index];
                        if (newt == t1 || !is_pattern(newt, spec.pattern_masks)) {
                            continue;
                        }
                        buffer2[buffer2_count] = apply_canonical(newt, spec.symm_mode);
                        kinds2[buffer2_count] = static_cast<uint8_t>(
                            moves4_mask_new_tile[move_index] ? BufferedBaseKind::Derive : BufferedBaseKind::Plain
                        );
                        ++buffer2_count;
                        if (buffer2_count == kAdGenBatchSize) {
                            flush_spawn4();
                        }
                    }
                }
            }
        }

        flush_spawn2();
        flush_spawn4();
        c1[static_cast<size_t>(s)] = c1t;
        c2[static_cast<size_t>(s)] = c2t;
    }

    const size_t total_arr1 = BookGeneratorUtils::merge_inplace(arr1.get(), c1, starts);
    const size_t total_arr2 = BookGeneratorUtils::merge_inplace(arr2.get(), c2, starts);
    std::vector<uint64_t> arr1_vec(total_arr1);
    std::vector<uint64_t> arr2_vec(total_arr2);
    if (total_arr1 > 0) {
        std::memcpy(arr1_vec.data(), arr1.get(), total_arr1 * sizeof(uint64_t));
    }
    if (total_arr2 > 0) {
        std::memcpy(arr2_vec.data(), arr2.get(), total_arr2 * sizeof(uint64_t));
    }

    std::vector<size_t> counts1(static_cast<size_t>(n));
    std::vector<size_t> counts2(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        counts1[static_cast<size_t>(i)] = c1[static_cast<size_t>(i)] - starts[static_cast<size_t>(i)];
        counts2[static_cast<size_t>(i)] = c2[static_cast<size_t>(i)] - starts[static_cast<size_t>(i)];
    }
    return {std::move(arr1_vec), std::move(arr2_vec), std::move(counts1), std::move(counts2)};
}

bool cpu_has_ad_gen_avx512_uncached() {
#if defined(__x86_64__) || defined(__i386)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl");
#elif defined(_M_X64) || defined(_M_IX86)
    return HybridSearch::mode() == HybridSearch::Mode::AVX512;
#else
    return false;
#endif
}

GenBoardsAdDispatch resolve_gen_boards_ad_dispatch() {
    if (cpu_has_ad_gen_avx512_uncached()) {
        return {AdGenMode::AVX512, gen_boards_ad_avx512};
    }
    return {AdGenMode::Scalar, gen_boards_ad_scalar};
}

const GenBoardsAdDispatch &gen_boards_ad_dispatch() {
    static const GenBoardsAdDispatch cached = resolve_gen_boards_ad_dispatch();
    return cached;
}

GenBoardsAdResult gen_boards_ad(
    ArrayView<const uint64_t> arr0,
    const AdvancedPatternSpec &spec,
    std::vector<uint64_t> &hashmap1,
    std::vector<uint64_t> &hashmap2,
    uint32_t original_board_sum,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    int n,
    double length_factor,
    bool isfree
) {
    static const GenBoardsAdFn fn = gen_boards_ad_dispatch().fn;
    return fn(arr0, spec, hashmap1, hashmap2, original_board_sum, tiles_table, param, n, length_factor, isfree);
}

struct GenBoardsBigAdResult {
    std::vector<std::vector<uint64_t>> arr1s;
    std::vector<std::vector<uint64_t>> arr2s;
    std::vector<std::vector<double>> length_factors_list;
    double length_factor_multiplier = 1.5;
    double t0 = 0.0;
    double gen_time = 0.0;
    double t2 = 0.0;
};

GenBoardsBigAdResult gen_boards_big_ad(
    const std::vector<uint64_t> &arr0,
    const AdvancedPatternSpec &spec,
    std::vector<uint64_t> &hashmap1,
    std::vector<uint64_t> &hashmap2,
    uint32_t board_sum,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    int n,
    std::vector<std::vector<double>> length_factors_list,
    double length_factor_multiplier,
    bool isfree
) {
    size_t segs_count = length_factors_list.size();
    std::vector<std::vector<uint64_t>> arr1s;
    std::vector<std::vector<uint64_t>> arr2s;
    std::vector<uint64_t> actual_lengths1(segs_count * static_cast<size_t>(n), 0);
    std::vector<uint64_t> actual_lengths2(segs_count * static_cast<size_t>(n), 0);
    std::vector<size_t> seg_start_end = allocate_seg(length_factors_list, arr0.size());

    double t0 = wall_time_seconds();
    double gen_time = 0.0;
    for (size_t seg_index = 0; seg_index < segs_count; ++seg_index) {
        double t_segment = wall_time_seconds();
        size_t start_index = seg_start_end[seg_index];
        size_t end_index = seg_start_end[seg_index + 1];
        ArrayView<const uint64_t> arr0t{arr0.data() + start_index, end_index - start_index};

        std::vector<double> length_factors = length_factors_list[seg_index];
        double length_factor = BookGenerator::predict_next_length_factor_quadratic(length_factors);
        length_factor *= arr0t.size > static_cast<size_t>(1e8) ? 1.25 : 1.5;
        length_factor *= length_factor_multiplier;

        GenBoardsAdResult result = gen_boards_ad(arr0t, spec, hashmap1, hashmap2, board_sum, tiles_table, param, n, length_factor, isfree);
        validate_length_and_balance(arr0t.size, result.arr2.size(), result.arr1.size(), result.counts1, result.counts2, length_factor, true);

        for (int idx = 0; idx < n; ++idx) {
            actual_lengths2[seg_index * static_cast<size_t>(n) + static_cast<size_t>(idx)] = result.counts2[static_cast<size_t>(idx)];
            actual_lengths1[seg_index * static_cast<size_t>(n) + static_cast<size_t>(idx)] = result.counts1[static_cast<size_t>(idx)];
        }
        if (!length_factors.empty()) {
            length_factors.erase(length_factors.begin());
        }
        length_factors.push_back(static_cast<double>(result.arr2.size()) / static_cast<double>(1 + arr0t.size));
        length_factors_list[seg_index] = length_factors;
        gen_time += wall_time_seconds() - t_segment;

        BookGeneratorUtils::sort_array(result.arr1.data(), result.arr1.size(), n);
        BookGeneratorUtils::sort_array(result.arr2.data(), result.arr2.size(), n);
        result.arr1.resize(BookGeneratorUtils::parallel_unique(result.arr1.data(), result.arr1.size(), n));
        result.arr2.resize(BookGeneratorUtils::parallel_unique(result.arr2.data(), result.arr2.size(), n));
        arr1s.push_back(std::move(result.arr1));
        arr2s.push_back(std::move(result.arr2));
    }

    auto [updated_factors, updated_multiplier] = update_parameters_big(actual_lengths2, actual_lengths1, n, std::move(length_factors_list));
    double t2 = wall_time_seconds();
    return {std::move(arr1s), std::move(arr2s), std::move(updated_factors), updated_multiplier, t0, gen_time, t2};
}

std::tuple<bool, std::vector<uint64_t>, std::vector<uint64_t>> generate_process_ad(
    const std::vector<uint64_t> &arr_init_raw,
    const AdvancedPatternSpec &spec,
    const RunOptions &options,
    const FormationAD::MaskerContext &masker
) {
    bool started = false;
    std::vector<uint64_t> d0;
    std::vector<uint64_t> d1;
    std::vector<uint64_t> hashmap1;
    std::vector<uint64_t> hashmap2;
    const int n = options.num_threads > 0 ? options.num_threads : std::max(4, std::min(32, omp_get_max_threads()));
    InitParams init_params = initialize_parameters_internal(options.pathname, options.is_free);

    std::vector<uint64_t> arr_init = arr_init_raw;
    uint32_t ini_board_sum = 0;
    if (!arr_init.empty()) {
        uint64_t board = arr_init.front();
        for (int i = 0; i < 16; ++i) {
            uint64_t tile = (board >> static_cast<uint64_t>(4 * i)) & 0xFULL;
            if (tile > 0ULL) {
                ini_board_sum = static_cast<uint32_t>(ini_board_sum + (1U << tile));
            }
        }
    }
    for (uint64_t &board : arr_init) {
        board = FormationAD::mask_board(board);
    }

    for (int i = 1; i < options.steps - 1; ++i) {
        RestartResult restart = handle_restart_ad(i, options.pathname, arr_init, started);
        if (!restart.run) {
            continue;
        }
        started = restart.started;
        if (!restart.d0.empty()) {
            d0 = std::move(restart.d0);
        }
        if (!restart.d1.empty()) {
            d1 = std::move(restart.d1);
        }

        uint32_t board_sum = static_cast<uint32_t>(2 * i + ini_board_sum - 2);
        if (d0.size() < init_params.segment_size) {
            double t0 = wall_time_seconds();
            double length_factor = BookGenerator::predict_next_length_factor_quadratic(init_params.length_factors);
            length_factor *= d0.size() > static_cast<size_t>(1e8) ? 1.33 : 1.5;
            length_factor *= init_params.length_factor_multiplier;
            if (hashmap1.empty()) {
                BookGenerator::update_hashmap_length(hashmap1, d0.size());
                BookGenerator::update_hashmap_length(hashmap2, d0.size());
            }
            GenBoardsAdResult result = gen_boards_ad({d0.data(), d0.size()}, spec, hashmap1, hashmap2, board_sum, masker.tiles_combination_table, masker.param, n, length_factor, options.is_free);
            validate_length_and_balance(d0.size(), result.arr2.size(), result.arr1.size(), result.counts1, result.counts2, length_factor, false);
            double t1 = wall_time_seconds();

            BookGeneratorUtils::sort_array(result.arr1.data(), result.arr1.size(), n);
            BookGeneratorUtils::sort_array(result.arr2.data(), result.arr2.size(), n);
            auto [new_length_factors, new_length_factors_list] =
                update_parameters(d0.size(), result.arr2.size(), init_params.length_factors, init_params.length_factors_list_path);
            init_params.length_factors = std::move(new_length_factors);
            init_params.length_factors_list = std::move(new_length_factors_list);
            double mean_count = result.counts2.empty()
                ? 0.0
                : std::accumulate(result.counts2.begin(), result.counts2.end(), 0.0) / static_cast<double>(result.counts2.size());
            init_params.length_factor_multiplier = mean_count > 0.0
                ? static_cast<double>(*std::max_element(result.counts2.begin(), result.counts2.end())) / mean_count
                : 1.0;
            double t2 = wall_time_seconds();

            result.arr1.resize(BookGeneratorUtils::parallel_unique(result.arr1.data(), result.arr1.size(), n));
            result.arr2.resize(BookGeneratorUtils::parallel_unique(result.arr2.data(), result.arr2.size(), n));
            std::vector<uint64_t> pivots;
            for (int pt = 1; pt < n; ++pt) {
                if (d0.empty()) {
                    pivots.push_back(static_cast<uint64_t>(pt) * (1ULL << 50) / static_cast<uint64_t>(n));
                } else {
                    pivots.push_back(d0[static_cast<size_t>(pt) * d0.size() / static_cast<size_t>(n)]);
                }
            }
            std::vector<std::vector<uint64_t>> d1_inputs = {std::move(d1), std::move(result.arr1)};
            d1 = BookGeneratorUtils::merge_deduplicate_all_concat(d1_inputs, pivots, n);
            d0 = std::move(d1);
            d1 = std::move(result.arr2);
            double t3 = wall_time_seconds();
            log_generation_performance(i, t0, t1, t2, t3, d0.size());

            if (!hashmap1.empty()) {
                BookGenerator::update_hashmap_length(hashmap1, d1.size());
                BookGenerator::update_hashmap_length(hashmap2, d1.size());
            }
        } else {
            if (hashmap1.empty()) {
                size_t capacity = BookGeneratorUtils::largest_power_of_2(
                    static_cast<uint64_t>(20971520ULL * std::max(1.0, std::round(BookGenerator::get_system_memory_gb()) * 0.75))
                );
                hashmap1.assign(capacity, 0);
                hashmap2.assign(capacity, 0);
            }
            GenBoardsBigAdResult big_result = gen_boards_big_ad(
                d0, spec, hashmap1, hashmap2, board_sum, masker.tiles_combination_table, masker.param, n,
                init_params.length_factors_list, init_params.length_factor_multiplier, options.is_free
            );

            std::vector<uint64_t> pivots;
            for (int pt = 1; pt < n; ++pt) {
                if (d0.empty()) {
                    pivots.push_back(static_cast<uint64_t>(pt) * (1ULL << 50) / static_cast<uint64_t>(n));
                } else {
                    pivots.push_back(d0[static_cast<size_t>(pt) * d0.size() / static_cast<size_t>(n)]);
                }
            }
            big_result.arr1s.push_back(std::move(d1));
            d1 = BookGeneratorUtils::merge_deduplicate_all_concat(big_result.arr2s, pivots, n);
            d0 = BookGeneratorUtils::merge_deduplicate_all_concat(big_result.arr1s, pivots, n);
            double t3 = wall_time_seconds();
            log_generation_performance(i, big_result.t0, big_result.t0 + big_result.gen_time, big_result.t2, t3, d0.size());
            init_params.length_factors_list = std::move(big_result.length_factors_list);
            init_params.length_factor_multiplier = big_result.length_factor_multiplier;
            init_params.length_factors = BookGenerator::harmonic_mean_by_column(init_params.length_factors_list);
            save_length_factors(init_params.length_factors_list_path, init_params.length_factors_list);
        }

        if (((i + static_cast<int>(ini_board_sum % 64U / 2U)) % 32) == ((static_cast<int>(masker.param.small_tile_sum_limit / 2U)) % 32) + 1) {
            d0 = validate_layer(d0, board_sum + 2U, masker.tiles_combination_table, masker.param, n);
            d1 = validate_layer(d1, board_sum + 4U, masker.tiles_combination_table, masker.param, n);
            debug_log("validate step " + std::to_string(i));
        }

        clear_u64_buffer(hashmap1, n);
        write_raw_file(options.pathname + std::to_string(i), d0);
        if (options.compress_temp_files && i > 5) {
            maybe_compress_with_7z(options.pathname + std::to_string(i - 2));
        }
        std::swap(hashmap1, hashmap2);
    }
    return {started, std::move(d0), std::move(d1)};
}

void run_python_ad_solver_bridge(
    const std::vector<uint64_t> &arr_init,
    const AdvancedPatternSpec &spec,
    const RunOptions &options
) {
    nb::gil_scoped_acquire acquire;
    nb::object helper = nb::module_::import_("egtb_core.BookBuilder").attr("run_pattern_solve_ad_bridge");
    helper(
        spec.name,
        nb::cast(arr_init),
        spec.symm_mode,
        spec.num_free_32k,
        spec.fixed_32k_shifts,
        static_cast<int>(spec.target),
        options.steps,
        options.pathname,
        options.is_free,
        options.spawn_rate4
    );
}

} // namespace

void run_pattern_build_ad_cpp(
    const std::vector<uint64_t> &arr_init,
    const AdvancedPatternSpec &spec,
    const RunOptions &options
) {
    bool started = false;
    {
        FormationAD::MaskerContext masker = FormationAD::init_masker(spec);
        auto [generated_started, d0, d1] = generate_process_ad(arr_init, spec, options, masker);
        started = generated_started;
        d0.clear();
        d0.shrink_to_fit();
        d1.clear();
        d1.shrink_to_fit();
    }
    run_pattern_solve_ad_cpp(arr_init, spec, options, started);
}
