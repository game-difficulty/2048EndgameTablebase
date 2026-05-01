#include "BookGenerator.h"

#include "BoardMover.h"
#include "Calculator.h"
#include "CompressionBridge.h"
#include "FileIOUtils.h"
#include "Formation.h"
#include "UniqueUtils.h"
#include "VBoardMover.h"
#include <immintrin.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <nanobind/nanobind.h>
#include <numeric>
#include <omp.h>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

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

bool supports_avx512() {
    return UniqueUtils::cpu_has_avx512_dq_bw_vl();
}

size_t success_entry_size_for_dtype(const std::string &name) {
    switch (success_rate_kind_from_name(name)) {
        case SuccessRateKind::UInt64:
            return sizeof(SuccessEntry<uint64_t>);
        case SuccessRateKind::Float32:
            return sizeof(SuccessEntry<float>);
        case SuccessRateKind::Float64:
            return sizeof(SuccessEntry<double>);
        case SuccessRateKind::UInt32:
        default:
            return sizeof(SuccessEntry<uint32_t>);
    }
}

bool is_valid_restart_file(const std::string &path, uint64_t alignment) {
    std::error_code ec;
    if (!fs::exists(path, ec) || ec) {
        return false;
    }
    if (!fs::is_regular_file(path, ec) || ec) {
        return false;
    }
    const auto size = fs::file_size(path, ec);
    if (ec || size == 0U) {
        return false;
    }
    return alignment == 0U || (size % alignment) == 0U;
}

void remove_invalid_restart_file(const std::string &path, uint64_t alignment) {
    std::error_code ec;
    if (!fs::exists(path, ec) || ec) {
        return;
    }
    if (is_valid_restart_file(path, alignment)) {
        return;
    }
    fs::remove(path, ec);
    if (ec) {
        throw std::runtime_error("failed to remove invalid restart file: " + path);
    }
    debug_log("removed invalid restart file: " + path);
}

__attribute__((target("avx512f,avx512dq,avx512bw,avx512vl")))
inline __m512i simd_hash(__m512i v) {
    // 严格匹配 BookGeneratorUtils::hash_board 的两轮混合逻辑
    __m512i v_xor = _mm512_xor_epi64(v, _mm512_srli_epi64(v, 27));
    __m512i v_mul = _mm512_mullo_epi64(v_xor, _mm512_set1_epi64(0x1A85EC53ULL));
    __m512i v_mix = _mm512_add_epi64(_mm512_add_epi64(v_mul, _mm512_srli_epi64(v, 23)), v);
    __m512i v_mix_xor = _mm512_xor_epi64(v_mix, _mm512_srli_epi64(v_mix, 27));
    __m512i v_mix_mul = _mm512_mullo_epi64(v_mix_xor, _mm512_set1_epi64(0x1A85EC53ULL));
    return _mm512_add_epi64(_mm512_add_epi64(v_mix_mul, _mm512_srli_epi64(v_mix, 23)), v_mix);
}

__attribute__((target("avx512f,avx512dq,avx512bw,avx512vl")))
inline void process_buf_avx512(uint64_t *buf, uint64_t *hmap, uint64_t *target_arr, size_t &counter, uint64_t mask) {
    #pragma GCC unroll 16
    for (int i = 0; i < 128; i += 8) {
        __m512i v_states = _mm512_loadu_epi64(&buf[i]);
        __m512i v_h = simd_hash(v_states);
        __m512i v_idx = _mm512_and_epi64(v_h, _mm512_set1_epi64(mask));
        __m512i v_table = _mm512_i64gather_epi64(v_idx, hmap, 8);
        __mmask8 matches = _mm512_cmpeq_epi64_mask(v_table, v_states);
        __mmask8 not_dup = ~matches;
        _mm512_mask_i64scatter_epi64(hmap, not_dup, v_idx, v_states, 8);
        int count = __builtin_popcount(static_cast<unsigned int>(not_dup));
        _mm512_mask_compressstoreu_epi64(target_arr + counter, not_dup, v_states);
        counter += static_cast<size_t>(count);
    }
}

void log_performance(int step_index, double t0, double t1, double t2, double t3, size_t layer_size) {
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

constexpr size_t kScalarBufferedBatchSize = 128;

inline void process_buf_scalar(
    const uint64_t *buf,
    size_t active_count,
    uint64_t *hmap,
    uint64_t *target_arr,
    size_t &counter,
    uint64_t mask
) {
    std::array<uint64_t, kScalarBufferedBatchSize> hashed_idx{};
    for (size_t i = 0; i < active_count; ++i) {
        hashed_idx[i] = BookGeneratorUtils::hash_board(buf[i]) & mask;
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(hmap + hashed_idx[i], 1, 1);
#endif
    }

    for (size_t i = 0; i < active_count; ++i) {
        const size_t prefetch_distance = 8;
        if (i + prefetch_distance < active_count) {
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(hmap + hashed_idx[i + prefetch_distance], 1, 1);
#endif
        }
        const uint64_t board = buf[i];
        const size_t hashed = static_cast<size_t>(hashed_idx[i]);
        if (hmap[hashed] == board) {
            continue;
        }
        hmap[hashed] = board;
        target_arr[counter++] = board;
    }
}

double solve_quadratic_and_predict(const std::vector<double> &y) {
    size_t n = y.size();
    if (n < 3) {
        return y.empty() ? 3.0 : y.back();
    }

    double s0 = static_cast<double>(n);
    double s1 = 0.0;
    double s2 = 0.0;
    double s3 = 0.0;
    double s4 = 0.0;
    double sy = 0.0;
    double sxy = 0.0;
    double sx2y = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double x = static_cast<double>(i);
        double x2 = x * x;
        double x3 = x2 * x;
        double x4 = x3 * x;
        s1 += x;
        s2 += x2;
        s3 += x3;
        s4 += x4;
        sy += y[i];
        sxy += x * y[i];
        sx2y += x2 * y[i];
    }

    auto det3x3 = [](double a, double b, double c, double d, double e, double f, double g, double h, double i) {
        return a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h;
    };

    double det = det3x3(s4, s3, s2, s3, s2, s1, s2, s1, s0);
    if (std::abs(det) < 1e-9) {
        return y.back();
    }

    double det_a = det3x3(sx2y, s3, s2, sxy, s2, s1, sy, s1, s0);
    double det_b = det3x3(s4, sx2y, s2, s3, sxy, s1, s2, sy, s0);
    double det_c = det3x3(s4, s3, sx2y, s3, s2, sxy, s2, s1, sy);
    double a = det_a / det;
    double b = det_b / det;
    double c = det_c / det;
    double next_x = static_cast<double>(n);
    return a * next_x * next_x + b * next_x + c;
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

struct InitParams {
    double length_factor = 3.2;
    std::vector<double> length_factors;
    std::vector<std::vector<double>> length_factors_list;
    std::string length_factors_list_path;
    double length_factor_multiplier = 2.0;
    size_t segment_size = 0;
};

InitParams initialize_parameters_internal(int num_threads, const std::string &pathname, bool isfree) {
    (void) num_threads;
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

} // namespace

namespace {

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

std::string format_percent_array(const std::vector<size_t> &counts) {
    double sum = std::accumulate(counts.begin(), counts.end(), 0.0);
    std::ostringstream oss;
    oss << "array([";
    for (size_t i = 0; i < counts.size(); ++i) {
        double value = sum > 0.0 ? static_cast<double>(counts[i]) / sum : 0.0;
        oss << std::fixed << std::setprecision(5) << value;
        if (i + 1 != counts.size()) {
            oss << ", ";
        }
    }
    oss << "])";
    return oss.str();
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
    double length_factor_actual = static_cast<double>(length_needed) / static_cast<double>(len_d0 == 0 ? 1 : len_d0);
    size_t allocated_length = std::max<size_t>(6999999ULL, static_cast<size_t>(static_cast<double>(len_d0) * length_factor));
    if (length_needed > allocated_length) {
        throw std::out_of_range("The length multiplier is not big enough. Please restart.");
    }
    std::ostringstream oss;
    oss << "length " << len_d1t << ", " << len_d2
        << ", Using " << round_to_2(length_factor)
        << ", Need " << round_to_2(length_factor_actual);
    debug_log(oss.str());
    if (!is_big) {
        debug_log("Segmentation1_ac " + format_percent_array(counts2));
        debug_log("Segmentation2_ac " + format_percent_array(counts1));
    }
}

std::tuple<bool, std::vector<uint64_t>, std::vector<uint64_t>> handle_restart(
    int step_index,
    const RunOptions &options,
    const std::vector<uint64_t> &arr_init,
    bool started,
    FileIOUtils::DirectIoConfig io_config
) {
    auto path_i = options.pathname + std::to_string(step_index);
    auto path_i_plus_1 = options.pathname + std::to_string(step_index + 1);
    auto path_i_minus_1 = options.pathname + std::to_string(step_index - 1);
    const uint64_t raw_alignment = sizeof(uint64_t);
    const uint64_t book_alignment = success_entry_size_for_dtype(options.success_rate_dtype);

    remove_invalid_restart_file(path_i, raw_alignment);
    remove_invalid_restart_file(path_i_plus_1, raw_alignment);
    remove_invalid_restart_file(path_i + ".book", book_alignment);
    remove_invalid_restart_file(path_i_plus_1 + ".book", book_alignment);
    remove_invalid_restart_file(path_i + ".z", 0U);
    remove_invalid_restart_file(path_i + ".book.7z", 0U);
    remove_invalid_restart_file(path_i + ".7z", 0U);
    remove_invalid_restart_file(path_i_plus_1 + ".z", 0U);

    if ((is_valid_restart_file(path_i_plus_1, raw_alignment) && is_valid_restart_file(path_i, raw_alignment)) ||
        (is_valid_restart_file(path_i_plus_1 + ".book", book_alignment) && is_valid_restart_file(path_i, raw_alignment)) ||
        (is_valid_restart_file(path_i_plus_1 + ".z", 0U) && is_valid_restart_file(path_i, raw_alignment)) ||
        is_valid_restart_file(path_i + ".book", book_alignment) ||
        is_valid_restart_file(path_i + ".z", 0U) ||
        is_valid_restart_file(path_i + ".book.7z", 0U) ||
        is_valid_restart_file(path_i + ".7z", 0U)) {
        debug_log("skipping step " + std::to_string(step_index));
        return {false, {}, {}};
    }

    if (step_index == 1) {
        FileIOUtils::write_binary_vector_direct(path_i_minus_1, arr_init, io_config);
        return {true, arr_init, {}};
    }

    if (!started) {
        auto read_raw = [&io_config](const std::string &path) {
            return FileIOUtils::read_binary_vector_direct<uint64_t>(path, io_config);
        };
        return {true, read_raw(path_i_minus_1), read_raw(path_i)};
    }

    return {true, {}, {}};
}

bool should_use_simple_path(
    const std::vector<uint64_t> &d0,
    const std::vector<uint64_t> &arr_init,
    const std::vector<double> &length_factors
) {
    return d0.size() < 10000 ||
           (!arr_init.empty() &&
            (arr_init[0] == 0xffff00000000ffffULL || arr_init[0] == 0x000f000f000fffffULL)) ||
           std::find(length_factors.begin(), length_factors.end(), 3.2) != length_factors.end();
}

std::vector<size_t> allocate_seg(const std::vector<std::vector<double>> &length_factors_list, size_t arr_length) {
    if (length_factors_list.size() <= 1) {
        return {0, arr_length};
    }
    std::vector<double> weights;
    weights.reserve(length_factors_list.size());
    double total_weight = 0.0;
    for (const auto &factors : length_factors_list) {
        double factor = factors.empty() ? 1.5 : factors.back();
        double weight = 1.0 / (factor + 0.2);
        weights.push_back(weight);
        total_weight += weight;
    }
    std::vector<size_t> segments;
    segments.reserve(length_factors_list.size() + 1);
    segments.push_back(0);
    double cumulative = 0.0;
    for (size_t i = 0; i + 1 < weights.size(); ++i) {
        cumulative += weights[i];
        segments.push_back(static_cast<size_t>((cumulative / total_weight) * static_cast<double>(arr_length)));
    }
    segments.push_back(arr_length);
    return segments;
}

} // namespace

namespace BookGenerator {

template <typename Mover>
GenBoardsResult gen_boards_internal_naive(
    const uint64_t *arr0, size_t arr0_size,
    int target,
    const PatternSpec &spec,
    uint64_t *hashmap1, size_t hashmap1_size,
    uint64_t *hashmap2, size_t hashmap2_size,
    uint64_t *arr1, uint64_t *arr2,
    size_t capacity,
    int num_threads,
    bool do_check,
    bool isfree
) {
    (void) isfree;
    uint64_t hashmask1 = hashmap1_size - 1;
    uint64_t hashmask2 = hashmap2_size - 1;
    std::vector<size_t> starts(num_threads);
    std::vector<size_t> c1(num_threads);
    std::vector<size_t> c2(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        starts[i] = (capacity / static_cast<size_t>(num_threads)) * static_cast<size_t>(i);
    }

    size_t chunk_size = std::min<size_t>(1000000ULL, arr0_size / (static_cast<size_t>(num_threads) * 5ULL) + 1ULL) * static_cast<size_t>(num_threads);
    size_t chunks_count = (arr0_size + chunk_size - 1) / chunk_size;

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_index = omp_get_thread_num();
        size_t c1t = starts[thread_index];
        size_t c2t = starts[thread_index];
        std::array<uint64_t, kScalarBufferedBatchSize> buffer1{};
        std::array<uint64_t, kScalarBufferedBatchSize> buffer2{};
        size_t b1_ptr = 0;
        size_t b2_ptr = 0;

        auto flush_buf1 = [&]() {
            if (b1_ptr == 0) {
                return;
            }
            process_buf_scalar(buffer1.data(), b1_ptr, hashmap1, arr1, c1t, hashmask1);
            b1_ptr = 0;
        };

        auto flush_buf2 = [&]() {
            if (b2_ptr == 0) {
                return;
            }
            process_buf_scalar(buffer2.data(), b2_ptr, hashmap2, arr2, c2t, hashmask2);
            b2_ptr = 0;
        };

        for (size_t chunk = 0; chunk < chunks_count; ++chunk) {
            size_t chunk_start = chunk * chunk_size;
            size_t chunk_end = std::min(chunk_start + chunk_size, arr0_size);
            size_t thread_start = chunk_start + static_cast<size_t>(thread_index) * (chunk_size / static_cast<size_t>(num_threads));
            size_t thread_end = thread_start + (chunk_size / static_cast<size_t>(num_threads));
            size_t start = std::max(thread_start, chunk_start);
            size_t end = std::min(thread_end, chunk_end);
            if (start >= end) {
                continue;
            }

            for (size_t b = start; b < end; ++b) {
                uint64_t board = arr0[b];
                if (do_check && is_success_by_shifts(board, target, spec.success_shifts)) {
                    continue;
                }
                for (int i = 0; i < 16; ++i) {
                    if (((board >> (4 * i)) & 0xFULL) != 0) {
                        continue;
                    }

                    uint64_t spawn2 = board | (1ULL << (4 * i));
                    auto moves2 = Mover::move_all_dir(spawn2);
                    uint64_t boards2[4] = {std::get<0>(moves2), std::get<1>(moves2), std::get<2>(moves2), std::get<3>(moves2)};
                    for (uint64_t new_board : boards2) {
                        if (new_board != spawn2 && is_pattern(new_board, spec.pattern_masks)) {
                            buffer1[b1_ptr++] = apply_canonical(new_board, spec.symm_mode);
                            if (b1_ptr == kScalarBufferedBatchSize) {
                                flush_buf1();
                            }
                        }
                    }

                    uint64_t spawn4 = board | (2ULL << (4 * i));
                    auto moves4 = Mover::move_all_dir(spawn4);
                    uint64_t boards4[4] = {std::get<0>(moves4), std::get<1>(moves4), std::get<2>(moves4), std::get<3>(moves4)};
                    for (uint64_t new_board : boards4) {
                        if (new_board != spawn4 && is_pattern(new_board, spec.pattern_masks)) {
                            buffer2[b2_ptr++] = apply_canonical(new_board, spec.symm_mode);
                            if (b2_ptr == kScalarBufferedBatchSize) {
                                flush_buf2();
                            }
                        }
                    }
                }
            }
        }

        flush_buf1();
        flush_buf2();

        c1[thread_index] = c1t;
        c2[thread_index] = c2t;
    }

    GenBoardsResult result;
    result.total_arr1 = BookGeneratorUtils::merge_inplace(arr1, c1, starts);
    result.total_arr2 = BookGeneratorUtils::merge_inplace(arr2, c2, starts);
    result.counts1.resize(num_threads);
    result.counts2.resize(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        result.counts1[i] = c1[i] - starts[i];
        result.counts2[i] = c2[i] - starts[i];
    }
    return result;
}

template <typename Mover>
__attribute__((target("avx512f,avx512dq,avx512bw,avx512vl")))
GenBoardsResult gen_boards_internal_avx512(
    const uint64_t *arr0, size_t arr0_size,
    int target,
    const PatternSpec &spec,
    uint64_t *hashmap1, size_t hashmap1_size,
    uint64_t *hashmap2, size_t hashmap2_size,
    uint64_t *arr1, uint64_t *arr2,
    size_t capacity,
    int num_threads,
    bool do_check,
    bool isfree
) {
    (void) isfree;
    uint64_t hashmask1 = hashmap1_size - 1;
    uint64_t hashmask2 = hashmap2_size - 1;
    std::vector<size_t> starts(num_threads);
    std::vector<size_t> c1(num_threads);
    std::vector<size_t> c2(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        starts[i] = (capacity / static_cast<size_t>(num_threads)) * static_cast<size_t>(i);
    }

    const int BATCH_SIZE = 128;

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_index = omp_get_thread_num();
        size_t c1t = starts[thread_index];
        size_t c2t = starts[thread_index];

        uint64_t buffer1[BATCH_SIZE], buffer2[BATCH_SIZE];
        int b1_ptr = 0, b2_ptr = 0;

        auto process_buf = [&](uint64_t* buf, int& ptr, uint64_t* hmap, uint64_t* target_arr, size_t& counter, uint64_t mask) {
            process_buf_avx512(buf, hmap, target_arr, counter, mask);
            ptr = 0;
        };

        #pragma omp for schedule(dynamic, 1024)
        for (int64_t b = 0; b < (int64_t)arr0_size; ++b) {
            uint64_t board = arr0[b];
            if (do_check && is_success_by_shifts(board, target, spec.success_shifts)) {
                continue;
            }
            for (int i = 0; i < 16; ++i) {
                if (((board >> (4 * i)) & 0xFULL) != 0) {
                    continue;
                }

                uint64_t spawn2 = board | (1ULL << (4 * i));
                auto moves2 = Mover::move_all_dir(spawn2);
                uint64_t boards2[4] = {std::get<0>(moves2), std::get<1>(moves2), std::get<2>(moves2), std::get<3>(moves2)};
                for (uint64_t new_board : boards2) {
                    if (new_board != spawn2 && is_pattern(new_board, spec.pattern_masks)) {
                        buffer1[b1_ptr++] = apply_canonical(new_board, spec.symm_mode);
                        if (b1_ptr == BATCH_SIZE) process_buf(buffer1, b1_ptr, hashmap1, arr1, c1t, hashmask1);
                    }
                }

                uint64_t spawn4 = board | (2ULL << (4 * i));
                auto moves4 = Mover::move_all_dir(spawn4);
                uint64_t boards4[4] = {std::get<0>(moves4), std::get<1>(moves4), std::get<2>(moves4), std::get<3>(moves4)};
                for (uint64_t new_board : boards4) {
                    if (new_board != spawn4 && is_pattern(new_board, spec.pattern_masks)) {
                        buffer2[b2_ptr++] = apply_canonical(new_board, spec.symm_mode);
                        if (b2_ptr == BATCH_SIZE) process_buf(buffer2, b2_ptr, hashmap2, arr2, c2t, hashmask2);
                    }
                }
            }
        }
        
        // 尾部清理
        if (b1_ptr > 0) {
            uint64_t dummy = buffer1[0];
            while (b1_ptr < BATCH_SIZE) buffer1[b1_ptr++] = dummy;
            process_buf(buffer1, b1_ptr, hashmap1, arr1, c1t, hashmask1);
        }
        if (b2_ptr > 0) {
            uint64_t dummy = buffer2[0];
            while (b2_ptr < BATCH_SIZE) buffer2[b2_ptr++] = dummy;
            process_buf(buffer2, b2_ptr, hashmap2, arr2, c2t, hashmask2);
        }

        c1[thread_index] = c1t;
        c2[thread_index] = c2t;
    }

    GenBoardsResult result;
    result.total_arr1 = BookGeneratorUtils::merge_inplace(arr1, c1, starts);
    result.total_arr2 = BookGeneratorUtils::merge_inplace(arr2, c2, starts);
    result.counts1.resize(num_threads);
    result.counts2.resize(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        result.counts1[i] = c1[i] - starts[i];
        result.counts2[i] = c2[i] - starts[i];
    }
    return result;
}

template <typename Mover>
GenBoardsResult gen_boards(
    const uint64_t *arr0, size_t arr0_size,
    int target,
    const PatternSpec &spec,
    uint64_t *hashmap1, size_t hashmap1_size,
    uint64_t *hashmap2, size_t hashmap2_size,
    uint64_t *arr1, uint64_t *arr2,
    size_t capacity,
    int num_threads,
    bool do_check,
    bool isfree
) {
    if (supports_avx512()) {
        return gen_boards_internal_avx512<Mover>(
            arr0, arr0_size, target, spec, hashmap1, hashmap1_size, hashmap2, hashmap2_size,
            arr1, arr2, capacity, num_threads, do_check, isfree
        );
    }
    return gen_boards_internal_naive<Mover>(
        arr0, arr0_size, target, spec, hashmap1, hashmap1_size, hashmap2, hashmap2_size,
        arr1, arr2, capacity, num_threads, do_check, isfree
    );
}

template <typename Mover>
std::tuple<std::vector<uint64_t>, std::vector<uint64_t>> gen_boards_simple(
    const uint64_t *arr0, size_t arr0_size,
    int target,
    const PatternSpec &spec,
    bool do_check,
    bool isfree
) {
    size_t length = isfree ? std::max<size_t>(arr0_size * 8ULL, 1999999ULL) : std::max<size_t>(arr0_size * 6ULL, 9999999ULL);
    std::vector<uint64_t> arr_out1;
    std::vector<uint64_t> arr_out2;

    #pragma omp parallel for num_threads(2)
    for (int p = 1; p <= 2; ++p) {
        std::vector<uint64_t> arr;
        arr.reserve(length);
        for (size_t b = 0; b < arr0_size; ++b) {
            uint64_t board = arr0[b];
            if (do_check && is_success_by_shifts(board, target, spec.success_shifts)) {
                continue;
            }
            for (int i = 0; i < 16; ++i) {
                if (((board >> (4 * i)) & 0xFULL) != 0) {
                    continue;
                }
                uint64_t spawned = board | (static_cast<uint64_t>(p) << (4 * i));
                auto moved = Mover::move_all_dir(spawned);
                uint64_t boards[4] = {std::get<0>(moved), std::get<1>(moved), std::get<2>(moved), std::get<3>(moved)};
                for (uint64_t new_board : boards) {
                    if (new_board != spawned && is_pattern(new_board, spec.pattern_masks)) {
                        arr.push_back(apply_canonical(new_board, spec.symm_mode));
                    }
                }
            }
        }
        if (p == 1) {
            arr_out1 = std::move(arr);
        } else {
            arr_out2 = std::move(arr);
        }
    }

    return {std::move(arr_out1), std::move(arr_out2)};
}

GenBoardsBigResult
gen_boards_big(
    const std::vector<uint64_t> &arr0,
    int target,
    const PatternSpec &spec,
    std::vector<uint64_t> &hashmap1,
    std::vector<uint64_t> &hashmap2,
    int num_threads,
    std::vector<std::vector<double>> &length_factors_list,
    double &length_factor_multiplier,
    bool do_check,
    bool isfree,
    bool is_variant
) {
    size_t segs_count = length_factors_list.size();
    GenBoardsBigResult big_result;
    big_result.t0 = wall_time_seconds();
    double gen_time = 0.0;
    std::vector<size_t> actual_lengths1(segs_count * static_cast<size_t>(num_threads), 0);
    std::vector<size_t> actual_lengths2(segs_count * static_cast<size_t>(num_threads), 0);
    std::vector<size_t> seg_limits = allocate_seg(length_factors_list, arr0.size());

    for (size_t seg_index = 0; seg_index < segs_count; ++seg_index) {
        double seg_t0 = wall_time_seconds();
        size_t start_index = seg_limits[seg_index];
        size_t end_index = seg_limits[seg_index + 1];
        size_t len = end_index - start_index;
        double length_factor = predict_next_length_factor_quadratic(length_factors_list[seg_index]);
        length_factor *= len > static_cast<size_t>(1e8) ? 1.25 : 1.5;
        length_factor *= length_factor_multiplier;
        if (std::isnan(length_factor)) {
            length_factor = 3.0;
        }

        size_t min_length = isfree ? 9999999ULL : 6999999ULL;
        size_t capacity = std::max(min_length, static_cast<size_t>(static_cast<double>(len) * length_factor));
        auto arr1_ptr = std::make_unique<uint64_t[]>(capacity);
        auto arr2_ptr = std::make_unique<uint64_t[]>(capacity);

        GenBoardsResult segment_result = is_variant
            ? gen_boards<VBoardMover>(arr0.data() + start_index, len, target, spec, hashmap1.data(), hashmap1.size(), hashmap2.data(), hashmap2.size(), arr1_ptr.get(), arr2_ptr.get(), capacity, num_threads, do_check, isfree)
            : gen_boards<BoardMover>(arr0.data() + start_index, len, target, spec, hashmap1.data(), hashmap1.size(), hashmap2.data(), hashmap2.size(), arr1_ptr.get(), arr2_ptr.get(), capacity, num_threads, do_check, isfree);

        validate_length_and_balance(
            len,
            segment_result.total_arr2,
            segment_result.total_arr1,
            segment_result.counts1,
            segment_result.counts2,
            length_factor,
            true
        );
        std::copy(segment_result.counts1.begin(), segment_result.counts1.end(), actual_lengths1.begin() + static_cast<std::ptrdiff_t>(seg_index * static_cast<size_t>(num_threads)));
        std::copy(segment_result.counts2.begin(), segment_result.counts2.end(), actual_lengths2.begin() + static_cast<std::ptrdiff_t>(seg_index * static_cast<size_t>(num_threads)));

        if (!length_factors_list[seg_index].empty()) {
            length_factors_list[seg_index].erase(length_factors_list[seg_index].begin());
        }
        length_factors_list[seg_index].push_back(static_cast<double>(segment_result.total_arr2) / static_cast<double>(1 + len));
        gen_time += wall_time_seconds() - seg_t0;

        BookGeneratorUtils::sort_array(arr1_ptr.get(), segment_result.total_arr1, num_threads);
        BookGeneratorUtils::sort_array(arr2_ptr.get(), segment_result.total_arr2, num_threads);
        big_result.arr1s.emplace_back(
            arr1_ptr.get(),
            arr1_ptr.get() + BookGeneratorUtils::parallel_unique(arr1_ptr.get(), segment_result.total_arr1, num_threads)
        );
        big_result.arr2s.emplace_back(
            arr2_ptr.get(),
            arr2_ptr.get() + BookGeneratorUtils::parallel_unique(arr2_ptr.get(), segment_result.total_arr2, num_threads)
        );
    }

    auto get_stats = [](const std::vector<size_t> &values) {
        double sum = static_cast<double>(std::accumulate(values.begin(), values.end(), 0ULL));
        double max_value = values.empty() ? 0.0 : static_cast<double>(*std::max_element(values.begin(), values.end()));
        double mean = values.empty() ? 0.0 : sum / static_cast<double>(values.size());
        return std::make_tuple(sum, max_value, mean);
    };
    auto [sum2, max2, mean2] = get_stats(actual_lengths2);
    auto [sum1, max1, mean1] = get_stats(actual_lengths1);
    double mean_percent2 = actual_lengths2.empty() ? 1.0 : 1.0 / static_cast<double>(actual_lengths2.size());
    double mean_percent1 = actual_lengths1.empty() ? 1.0 : 1.0 / static_cast<double>(actual_lengths1.size());
    length_factor_multiplier = 1.0;
    if (sum2 > 0.0) {
        length_factor_multiplier = std::max(length_factor_multiplier, (max2 / sum2) / mean_percent2);
    }
    if (sum1 > 0.0) {
        length_factor_multiplier = std::max(length_factor_multiplier, (max1 / sum1) / mean_percent1);
    }

    double threshold_gb = std::round(get_system_memory_gb()) * 0.75;
    if (mean2 * num_threads > 20971520.0 * threshold_gb && !length_factors_list.empty()) {
        std::vector<std::vector<double>> new_list;
        std::vector<double> first = length_factors_list.front();
        for (double &value : first) {
            value *= 1.5;
        }
        new_list.push_back(first);
        for (const auto &row : length_factors_list) {
            new_list.push_back(row);
            new_list.push_back(row);
        }
        if (!new_list.empty()) {
            new_list.pop_back();
        }
        length_factors_list = std::move(new_list);
    }
    if (mean2 * num_threads < 524288.0 * threshold_gb && !length_factors_list.empty()) {
        std::vector<std::vector<double>> new_list;
        for (const auto &row : length_factors_list) {
            std::vector<double> reduced;
            for (size_t i = 0; i < row.size(); i += 2) {
                reduced.push_back(row[i]);
            }
            new_list.push_back(std::move(reduced));
        }
        length_factors_list = std::move(new_list);
    }

    big_result.hashmap1 = std::move(hashmap1);
    big_result.hashmap2 = std::move(hashmap2);
    big_result.t1 = big_result.t0 + gen_time;
    big_result.t2 = wall_time_seconds();
    return big_result;
}

double predict_next_length_factor_quadratic(const std::vector<double> &length_factors) {
    if (length_factors.empty()) {
        return 3.0;
    }
    double last = length_factors.back();
    bool all_close = std::all_of(length_factors.begin(), length_factors.end(), [&](double value) {
        return std::abs(value - last) <= 0.1;
    });
    if (all_close) {
        return last;
    }
    double next_value = solve_quadratic_and_predict(length_factors);
    double mean = std::accumulate(length_factors.begin(), length_factors.end(), 0.0) / static_cast<double>(length_factors.size());
    double result = std::max(next_value, mean);
    result = std::min(result, last * 2.5);
    return std::isnan(result) ? 3.0 : result;
}

std::vector<double> harmonic_mean_by_column(const std::vector<std::vector<double>> &matrix) {
    if (matrix.empty() || matrix.front().empty()) {
        return {};
    }
    size_t num_columns = matrix.front().size();
    std::vector<double> harmonic_means;
    harmonic_means.reserve(num_columns);
    for (size_t col = 0; col < num_columns; ++col) {
        std::vector<double> non_zero_values;
        for (const auto &row : matrix) {
            if (col < row.size() && row[col] != 0.0) {
                non_zero_values.push_back(row[col]);
            }
        }
        if (non_zero_values.empty()) {
            harmonic_means.push_back(1.5);
            continue;
        }
        double reciprocal_sum = 0.0;
        for (double value : non_zero_values) {
            reciprocal_sum += 1.0 / value;
        }
        harmonic_means.push_back(static_cast<double>(non_zero_values.size()) / reciprocal_sum);
    }
    return harmonic_means;
}

double get_system_memory_gb() {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return static_cast<double>(status.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return static_cast<double>(pages * page_size) / (1024.0 * 1024.0 * 1024.0);
#endif
}

void update_hashmap_length(std::vector<uint64_t> &hashmap, size_t current_arr_size) {
    size_t length = std::max<size_t>(BookGeneratorUtils::largest_power_of_2(current_arr_size), 1048576ULL);
    if (hashmap.size() >= length) {
        return;
    }
    if (hashmap.empty()) {
        hashmap.assign(length, 0);
        return;
    }
    size_t old_size = hashmap.size();
    std::vector<uint64_t> new_hashmap(old_size * 2);
    std::memcpy(new_hashmap.data(), hashmap.data(), old_size * sizeof(uint64_t));
    std::memcpy(new_hashmap.data() + old_size, hashmap.data(), old_size * sizeof(uint64_t));
    hashmap = std::move(new_hashmap);
}

std::tuple<bool, std::vector<uint64_t>, std::vector<uint64_t>> generate_process(
    const std::vector<uint64_t> &arr_init,
    const PatternSpec &spec,
    const RunOptions &options
) {
    bool started = false;
    std::vector<uint64_t> d0;
    std::vector<uint64_t> d1;
    std::vector<uint64_t> hashmap1;
    std::vector<uint64_t> hashmap2;
    int num_threads = options.num_threads > 0 ? options.num_threads : std::max(4, std::min(32, omp_get_max_threads()));
    const uint32_t progress_total = classic_build_progress_total(options);
    auto init_params = initialize_parameters_internal(num_threads, options.pathname, options.is_free);
    const FileIOUtils::DirectIoConfig io_config = FileIOUtils::direct_io_config_from_options(options);

    for (int i = 1; i < options.steps - 1; ++i) {
        auto [run, restart_d0, restart_d1] = handle_restart(i, options, arr_init, started, io_config);
        if (!restart_d0.empty()) {
            d0 = std::move(restart_d0);
        }
        if (!restart_d1.empty()) {
            d1 = std::move(restart_d1);
        }
        if (!run) {
            continue;
        }
        started = true;
        FormationProgress::update_build_progress(static_cast<uint32_t>(i), progress_total);
        bool do_check = i > options.docheck_step;

        if (d0.size() < init_params.segment_size) {
            double t0 = wall_time_seconds();
            double t1 = t0;
            double t2 = t0;
            std::vector<uint64_t> d1t;
            std::vector<uint64_t> d2;
            std::vector<size_t> generation_counts2;
            bool use_simple_path = should_use_simple_path(d0, arr_init, init_params.length_factors);
            if (use_simple_path) {
                debug_log("step " + std::to_string(i) + " path: simple");
                std::tie(d1t, d2) = options.is_variant
                    ? gen_boards_simple<VBoardMover>(d0.data(), d0.size(), options.target, spec, do_check, options.is_free)
                    : gen_boards_simple<BoardMover>(d0.data(), d0.size(), options.target, spec, do_check, options.is_free);
                t1 = wall_time_seconds();
            } else {
                debug_log(
                    "step " + std::to_string(i) + " path: normal-" +
                    std::string(supports_avx512() ? "avx512" : "buffered-scalar")
                );
                double length_factor = predict_next_length_factor_quadratic(init_params.length_factors);
                length_factor *= d0.size() > static_cast<size_t>(1e8) ? 1.25 : 1.5;
                length_factor *= init_params.length_factor_multiplier;
                if (std::isnan(length_factor)) {
                    length_factor = 3.0;
                }
                if (hashmap1.empty()) {
                    update_hashmap_length(hashmap1, d0.size());
                    update_hashmap_length(hashmap2, d0.size());
                }

                size_t min_length = options.is_free ? 9999999ULL : 6999999ULL;
                size_t capacity = std::max(min_length, static_cast<size_t>(static_cast<double>(d0.size()) * length_factor));
                auto arr1_ptr = std::make_unique<uint64_t[]>(capacity);
                auto arr2_ptr = std::make_unique<uint64_t[]>(capacity);
                GenBoardsResult result = options.is_variant
                    ? gen_boards<VBoardMover>(d0.data(), d0.size(), options.target, spec, hashmap1.data(), hashmap1.size(), hashmap2.data(), hashmap2.size(), arr1_ptr.get(), arr2_ptr.get(), capacity, num_threads, do_check, options.is_free)
                    : gen_boards<BoardMover>(d0.data(), d0.size(), options.target, spec, hashmap1.data(), hashmap1.size(), hashmap2.data(), hashmap2.size(), arr1_ptr.get(), arr2_ptr.get(), capacity, num_threads, do_check, options.is_free);
                t1 = wall_time_seconds();

                validate_length_and_balance(
                    d0.size(),
                    result.total_arr2,
                    result.total_arr1,
                    result.counts1,
                    result.counts2,
                    length_factor,
                    false
                );
                d1t.assign(arr1_ptr.get(), arr1_ptr.get() + result.total_arr1);
                d2.assign(arr2_ptr.get(), arr2_ptr.get() + result.total_arr2);
                generation_counts2 = std::move(result.counts2);
            }

            std::tie(init_params.length_factors, init_params.length_factors_list) =
                update_parameters(d0.size(), d2.size(), init_params.length_factors, init_params.length_factors_list_path);
            if (generation_counts2.empty()) {
                // Python keeps counts2 initialized to ones on the simple path, so the multiplier collapses to 1.0.
                init_params.length_factor_multiplier = 1.0;
            } else {
                double mean_count = std::accumulate(generation_counts2.begin(), generation_counts2.end(), 0.0)
                    / static_cast<double>(generation_counts2.size());
                init_params.length_factor_multiplier = mean_count > 0.0
                    ? static_cast<double>(*std::max_element(generation_counts2.begin(), generation_counts2.end())) / mean_count
                    : 5.0;
            }

            BookGeneratorUtils::sort_array(d1t.data(), d1t.size(), num_threads);
            BookGeneratorUtils::sort_array(d2.data(), d2.size(), num_threads);
            d1t.resize(BookGeneratorUtils::parallel_unique(d1t.data(), d1t.size(), num_threads));
            d2.resize(BookGeneratorUtils::parallel_unique(d2.data(), d2.size(), num_threads));
            t2 = wall_time_seconds();

            std::vector<uint64_t> pivots;
            for (int pt = 1; pt < num_threads; ++pt) {
                if (d0.empty()) {
                    pivots.push_back(static_cast<uint64_t>(pt) * (1ULL << 50) / static_cast<uint64_t>(num_threads));
                } else {
                    pivots.push_back(d0[static_cast<size_t>(pt) * d0.size() / static_cast<size_t>(num_threads)]);
                }
            }
            std::vector<std::vector<uint64_t>> d1_inputs = {std::move(d1), std::move(d1t)};
            d1 = BookGeneratorUtils::concatenate(BookGeneratorUtils::merge_deduplicate_all(d1_inputs, pivots, num_threads));
            double t3 = wall_time_seconds();
            log_performance(i, t0, t1, t2, t3, d1.size());
            d0 = std::move(d1);
            d1 = std::move(d2);
            if (!hashmap1.empty()) {
                update_hashmap_length(hashmap1, d1.size());
                update_hashmap_length(hashmap2, d1.size());
            }
        } else {
            debug_log("step " + std::to_string(i) + " path: big");
            if (hashmap1.empty()) {
                size_t capacity = BookGeneratorUtils::largest_power_of_2(
                    20971520ULL * static_cast<size_t>(std::max(1.0, get_system_memory_gb() * 0.75))
                );
                hashmap1.assign(capacity, 0);
                hashmap2.assign(capacity, 0);
            }

            auto big_result = gen_boards_big(
                d0, options.target, spec, hashmap1, hashmap2, num_threads,
                init_params.length_factors_list, init_params.length_factor_multiplier,
                do_check, options.is_free, options.is_variant);
            hashmap1 = std::move(big_result.hashmap1);
            hashmap2 = std::move(big_result.hashmap2);

            std::vector<uint64_t> pivots;
            for (int pt = 1; pt < num_threads; ++pt) {
                pivots.push_back(d0[static_cast<size_t>(pt) * d0.size() / static_cast<size_t>(num_threads)]);
            }
            big_result.arr1s.push_back(std::move(d1));
            d1 = BookGeneratorUtils::concatenate(BookGeneratorUtils::merge_deduplicate_all(big_result.arr2s, pivots, num_threads));
            d0 = BookGeneratorUtils::concatenate(BookGeneratorUtils::merge_deduplicate_all(big_result.arr1s, pivots, num_threads));
            double t3 = wall_time_seconds();
            log_performance(i, big_result.t0, big_result.t1, big_result.t2, t3, d0.size());
            init_params.length_factors = harmonic_mean_by_column(init_params.length_factors_list);
            save_length_factors(init_params.length_factors_list_path, init_params.length_factors_list);
        }

        FileIOUtils::write_binary_vector_direct(options.pathname + std::to_string(i), d0, io_config);
        if (options.compress_temp_files && i > 5) {
            maybe_compress_with_7z(options.pathname + std::to_string(i - 2));
        }
        std::swap(hashmap1, hashmap2);
    }

    return {started, std::move(d0), std::move(d1)};
}

} // namespace BookGenerator
