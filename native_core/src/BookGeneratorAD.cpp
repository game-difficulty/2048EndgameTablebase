#include "BookSolver.h"

#include "BoardMaskerAD.h"
#include "BoardMoverAD.h"
#include "BookGenerator.h"
#include "BookGeneratorUtils.h"
#include "Calculator.h"
#include "CanonicalBatch.h"
#include "CompressionBridge.h"
#include "FileIOUtils.h"
#include "Formation.h"
#include "HybridSearch.h"
#include "NativeDiagnostics.h"
#include "UniqueUtils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
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

std::string now_string() {
    const auto now = std::chrono::system_clock::now();
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ) % 1000;
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
#ifdef _WIN32
    localtime_s(&local_time, &now_time);
#else
    localtime_r(&now_time, &local_time);
#endif
    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S")
        << '.' << std::setw(3) << std::setfill('0') << millis.count();
    return oss.str();
}

double throughput_mbps_for(uint64_t count, double seconds) {
    return seconds > 0.0 ? static_cast<double>(count) / seconds / 1e6 : 0.0;
}

std::string ad_generate_stats_file_path(const RunOptions &options) {
    return options.pathname + "ad_generate_stats.csv";
}

std::string ad_generate_stats_header() {
    return "stage,step,input_live,arr1_raw,arr2_raw,arr1_unique,arr2_unique,output_live,length_factor,total_seconds,throughput_mbps,compute_seconds,compute_throughput_mbps,generate_seconds,sort_unique_seconds,merge_seconds,validate_seconds,write_seconds,time";
}

void ensure_ad_generate_stats_header(const RunOptions &options) {
    const std::string path = ad_generate_stats_file_path(options);
    if (fs::exists(path)) {
        std::ifstream in(path);
        std::string first_line;
        if (std::getline(in, first_line) && first_line == ad_generate_stats_header()) {
            return;
        }
        in.close();
        std::error_code ec;
        fs::remove(path, ec);
    }
    std::ofstream file(path, std::ios::app);
    file << ad_generate_stats_header() << "\n";
}

struct AdGenerateStatsRecord {
    std::string stage;
    int step = -1;
    uint64_t input_live = 0U;
    uint64_t arr1_raw = 0U;
    uint64_t arr2_raw = 0U;
    uint64_t arr1_unique = 0U;
    uint64_t arr2_unique = 0U;
    uint64_t output_live = 0U;
    double length_factor = 0.0;
    double generate_seconds = 0.0;
    double sort_unique_seconds = 0.0;
    double merge_seconds = 0.0;
    double validate_seconds = 0.0;
    double write_seconds = 0.0;
};

void append_ad_generate_stats_record(
    const RunOptions &options,
    const AdGenerateStatsRecord &record
) {
    ensure_ad_generate_stats_header(options);
    const double compute_seconds =
        record.generate_seconds + record.sort_unique_seconds + record.merge_seconds + record.validate_seconds;
    const double total_seconds = compute_seconds + record.write_seconds;
    std::ofstream file(ad_generate_stats_file_path(options), std::ios::app);
    file << record.stage << ","
         << record.step << ","
         << record.input_live << ","
         << record.arr1_raw << ","
         << record.arr2_raw << ","
         << record.arr1_unique << ","
         << record.arr2_unique << ","
         << record.output_live << ","
         << std::fixed << std::setprecision(6)
         << record.length_factor << ","
         << total_seconds << ","
         << throughput_mbps_for(record.output_live, total_seconds) << ","
         << compute_seconds << ","
         << throughput_mbps_for(record.output_live, compute_seconds) << ","
         << record.generate_seconds << ","
         << record.sort_unique_seconds << ","
         << record.merge_seconds << ","
         << record.validate_seconds << ","
         << record.write_seconds << ","
         << now_string() << "\n";
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
        case SymmMode::Min34Top:
            return Calculator::canonical_min34_top(board);
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
        case SymmMode::Min34Top:
        default:
            return Calculator::canonical_identity_pair(board);
    }
}

std::vector<uint64_t> read_raw_file(const std::string &path, FileIOUtils::DirectIoConfig config) {
    const double t0 = wall_time_seconds();
    std::vector<uint64_t> data;
    if (fs::exists(path)) {
        data = FileIOUtils::read_binary_vector_direct<uint64_t>(path, config);
    } else {
        data = read_temp_uint64_archive(path + ".7z");
    }
    const double t1 = wall_time_seconds();
    const double elapsed = std::max(t1 - t0, 1e-12);
    const uint64_t bytes = static_cast<uint64_t>(data.size()) * static_cast<uint64_t>(sizeof(uint64_t));
    std::ostringstream oss;
    oss << "ad raw read path=" << path
        << " bytes=" << bytes
        << " direct_io=" << (config.enabled ? 1 : 0)
        << " qd=" << config.queue_depth
        << " chunk_mib=" << config.chunk_mib
        << " seconds=" << round_to_2(elapsed)
        << " gibps=" << round_to_2(static_cast<double>(bytes) / elapsed / 1024.0 / 1024.0 / 1024.0);
    debug_log(oss.str());
    return data;
}

void write_raw_file(
    const std::string &path,
    const std::vector<uint64_t> &data,
    FileIOUtils::DirectIoConfig config,
    bool compressed = false
) {
    const double t0 = wall_time_seconds();
    if (compressed) {
        if (!write_temp_uint64_archive(path + ".7z", data, 1)) {
            throw std::runtime_error("failed to write compressed temp layer: " + path + ".7z");
        }
        std::error_code ec;
        fs::remove(path, ec);
    } else {
        FileIOUtils::write_binary_vector_direct(path, data, config);
    }
    const double t1 = wall_time_seconds();
    const double elapsed = std::max(t1 - t0, 1e-12);
    const uint64_t bytes = static_cast<uint64_t>(data.size()) * static_cast<uint64_t>(sizeof(uint64_t));
    std::ostringstream oss;
    oss << "ad raw write path=" << path
        << " bytes=" << bytes
        << " direct_io=" << (config.enabled ? 1 : 0)
        << " qd=" << config.queue_depth
        << " chunk_mib=" << config.chunk_mib
        << " seconds=" << round_to_2(elapsed)
        << " gibps=" << round_to_2(static_cast<double>(bytes) / elapsed / 1024.0 / 1024.0 / 1024.0);
    debug_log(oss.str());
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

size_t capacity_from_factor(size_t input_size, double length_factor, size_t min_length, size_t capacity_floor);
double effective_length_factor_for_capacity(size_t input_size, double length_factor, size_t capacity_floor);

void validate_length_and_balance(
    size_t len_d0,
    size_t len_d2,
    size_t len_d1t,
    const std::vector<size_t> &counts1,
    const std::vector<size_t> &counts2,
    double length_factor,
    bool is_big,
    size_t spill_arr1 = 0,
    size_t spill_arr2 = 0,
    bool exact_gather_arr1 = false,
    bool exact_gather_arr2 = false,
    size_t capacity_floor = 0
) {
    if (len_d0 < 99999 || len_d2 < 99999 || counts1.empty() || counts2.empty()) {
        return;
    }
    size_t length_needed = std::max(
        *std::max_element(counts1.begin(), counts1.end()),
        *std::max_element(counts2.begin(), counts2.end())
    ) * counts1.size();
    double length_factor_actual = static_cast<double>(length_needed) / static_cast<double>(len_d0);
    size_t length = capacity_from_factor(len_d0, length_factor, 6999999ULL, capacity_floor);
    const double reported_length_factor =
        effective_length_factor_for_capacity(len_d0, length_factor, capacity_floor);
    const bool used_spill = spill_arr1 > 0 || spill_arr2 > 0;
    if (!used_spill && length_needed > length) {
        std::ostringstream oss;
        oss << "length multiplier " << reported_length_factor << ", need " << length_factor_actual;
        throw std::runtime_error(oss.str());
    }
    if (is_big || len_d0 == 0) {
        return;
    }
    std::ostringstream oss;
    oss << "length " << len_d1t << ", " << len_d2
        << ", Using " << round_to_2(reported_length_factor)
        << ", Need " << round_to_2(length_factor_actual);
    if (used_spill) {
        oss << ", Spill1 " << spill_arr1
            << (exact_gather_arr1 ? "(gather)" : "(tail)")
            << ", Spill2 " << spill_arr2
            << (exact_gather_arr2 ? "(gather)" : "(tail)");
    }
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

RestartResult handle_restart_ad(
    int step_index,
    const std::string &pathname,
    const std::vector<uint64_t> &arr_init,
    bool started,
    FileIOUtils::DirectIoConfig io_config,
    bool compress_temp_files
) {
    const std::string path_i = pathname + std::to_string(step_index);
    const std::string path_i_plus_1 = pathname + std::to_string(step_index + 1);
    const std::string path_i_minus_1 = pathname + std::to_string(step_index - 1);
    const bool has_readable_book_archive = is_readable_temp_archive(path_i + "b.7z");
    const bool has_readable_raw_archive = is_readable_temp_archive(path_i + ".7z");
    if ((fs::exists(path_i_plus_1) && fs::exists(path_i)) ||
        (fs::exists(path_i_plus_1 + "b") && fs::exists(path_i)) ||
        (fs::exists(path_i_plus_1 + ".z") && fs::exists(path_i)) ||
        fs::exists(path_i + "b") ||
        fs::exists(path_i + ".z") ||
        has_readable_book_archive ||
        has_readable_raw_archive) {
        debug_log("skipping step " + std::to_string(step_index));
        return {};
    }
    if (step_index == 1) {
        write_raw_file(path_i_minus_1, arr_init, io_config, compress_temp_files);
        return {true, true, arr_init, {}};
    }
    if (!started) {
        return {true, true, read_raw_file(path_i_minus_1, io_config), read_raw_file(path_i, io_config)};
    }
    return {true, true, {}, {}};
}

uint8_t derive_3x64(uint64_t masked, const std::array<uint64_t, 16> &pos_32k, int8_t count_32k, std::array<uint64_t, 120> &out) {
    uint8_t count = 0;
    for (int idx = 0; idx < count_32k; ++idx) {
        const uint64_t pos = pos_32k[static_cast<size_t>(idx)];
        const uint64_t tile_value = (masked >> pos) & 0xFULL;
        if (tile_value == 6ULL) {
            continue;
        }
        const uint64_t delta = ((tile_value ^ 6ULL) << pos);
        out[static_cast<size_t>(count++)] = masked ^ delta;
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
        const uint64_t target_value = static_cast<uint64_t>(tiles_combinations[0]);
        std::array<uint64_t, 16> deltas{};
        for (int idx = 0; idx < stats.count_32k; ++idx) {
            const uint64_t pos = stats.pos_32k[static_cast<size_t>(idx)];
            const uint64_t tile_value = (board >> pos) & 0xFULL;
            deltas[static_cast<size_t>(idx)] = ((tile_value ^ target_value) << pos);
        }
        for (int pos1 = 0; pos1 < stats.count_32k - 1; ++pos1) {
            const uint64_t base = board ^ deltas[static_cast<size_t>(pos1)];
            for (int pos2 = pos1 + 1; pos2 < stats.count_32k; ++pos2) {
                result.boards[static_cast<size_t>(result.count++)] = base ^ deltas[static_cast<size_t>(pos2)];
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
    size_t total_arr1 = 0;
    size_t total_arr2 = 0;
    std::vector<size_t> counts1;
    std::vector<size_t> counts2;
    std::unique_ptr<uint64_t[]> finalized_arr1;
    std::unique_ptr<uint64_t[]> finalized_arr2;
    size_t spill_arr1 = 0;
    size_t spill_arr2 = 0;
    bool exact_gather_arr1 = false;
    bool exact_gather_arr2 = false;
};

size_t capacity_from_factor(size_t input_size, double length_factor, size_t min_length, size_t capacity_floor) {
    size_t predicted = 0;
    if (std::isfinite(length_factor) && length_factor > 0.0) {
        const long double scaled = static_cast<long double>(input_size) * static_cast<long double>(length_factor);
        predicted = scaled >= static_cast<long double>(std::numeric_limits<size_t>::max())
            ? std::numeric_limits<size_t>::max()
            : static_cast<size_t>(scaled);
    }
    return std::max(std::max(min_length, predicted), capacity_floor);
}

double effective_length_factor_for_capacity(size_t input_size, double length_factor, size_t capacity_floor) {
    if (capacity_floor == 0U || input_size == 0U) {
        return length_factor;
    }
    return std::max(length_factor, static_cast<double>(capacity_floor) / static_cast<double>(input_size));
}

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
    size_t,
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
constexpr size_t kAdMaxDerivedBoards = 120;
constexpr size_t kSpillChunkLength = 1048576ULL;

struct SpillChunks {
    std::vector<std::unique_ptr<uint64_t[]>> chunks;
    std::vector<size_t> used;
    size_t total_count = 0;

    void append_one(uint64_t value) {
        if (chunks.empty() || used.back() == kSpillChunkLength) {
            chunks.push_back(std::make_unique<uint64_t[]>(kSpillChunkLength));
            used.push_back(0);
        }
        chunks.back()[used.back()++] = value;
        ++total_count;
    }

    void append_many(const uint64_t *src, size_t count) {
        while (count > 0) {
            if (chunks.empty() || used.back() == kSpillChunkLength) {
                chunks.push_back(std::make_unique<uint64_t[]>(kSpillChunkLength));
                used.push_back(0);
            }
            const size_t remaining = kSpillChunkLength - used.back();
            const size_t to_copy = std::min(remaining, count);
            std::memcpy(chunks.back().get() + used.back(), src, to_copy * sizeof(uint64_t));
            used.back() += to_copy;
            total_count += to_copy;
            src += to_copy;
            count -= to_copy;
        }
    }

    void copy_into(uint64_t *dst) const {
        size_t offset = 0;
        for (size_t i = 0; i < chunks.size(); ++i) {
            if (used[i] == 0) {
                continue;
            }
            std::memcpy(dst + offset, chunks[i].get(), used[i] * sizeof(uint64_t));
            offset += used[i];
        }
    }
};

struct ThreadWriteSink {
    uint64_t *primary = nullptr;
    size_t primary_begin = 0;
    size_t primary_pos = 0;
    size_t primary_end = 0;
    SpillChunks spill;

    ThreadWriteSink() = default;

    ThreadWriteSink(uint64_t *primary_arr, size_t begin, size_t end)
        : primary(primary_arr), primary_begin(begin), primary_pos(begin), primary_end(end) {}

    size_t primary_written() const {
        return primary_pos - primary_begin;
    }

    size_t total_written() const {
        return primary_written() + spill.total_count;
    }

    void append_one(uint64_t value) {
        if (primary_pos < primary_end) {
            primary[primary_pos++] = value;
            return;
        }
        spill.append_one(value);
    }

    void append_many(const uint64_t *src, size_t count) {
        const size_t primary_remaining = primary_end - primary_pos;
        const size_t to_primary = std::min(primary_remaining, count);
        if (to_primary > 0) {
            std::memcpy(primary + primary_pos, src, to_primary * sizeof(uint64_t));
            primary_pos += to_primary;
            src += to_primary;
            count -= to_primary;
        }
        if (count > 0) {
            spill.append_many(src, count);
        }
    }
};

struct FinalizedGeneratedArray {
    std::unique_ptr<uint64_t[]> owned;
    size_t total = 0;
    size_t spill_total = 0;
    bool exact_gather = false;

    uint64_t *data(uint64_t *primary_arr) const {
        return owned ? owned.get() : primary_arr;
    }
};

std::vector<size_t> build_segment_starts(size_t capacity, int num_threads) {
    std::vector<size_t> starts(static_cast<size_t>(num_threads));
    for (int i = 0; i < num_threads; ++i) {
        starts[static_cast<size_t>(i)] = (capacity / static_cast<size_t>(num_threads)) * static_cast<size_t>(i);
    }
    return starts;
}

std::vector<ThreadWriteSink> create_write_sinks(uint64_t *primary_arr, const std::vector<size_t> &starts, size_t capacity) {
    std::vector<ThreadWriteSink> sinks;
    sinks.reserve(starts.size());
    for (size_t i = 0; i < starts.size(); ++i) {
        const size_t begin = starts[i];
        const size_t end = (i + 1 < starts.size()) ? starts[i + 1] : capacity;
        sinks.emplace_back(primary_arr, begin, end);
    }
    return sinks;
}

std::vector<size_t> collect_total_counts(const std::vector<ThreadWriteSink> &sinks) {
    std::vector<size_t> counts(sinks.size(), 0);
    for (size_t i = 0; i < sinks.size(); ++i) {
        counts[i] = sinks[i].total_written();
    }
    return counts;
}

FinalizedGeneratedArray finalize_generated_array(
    uint64_t *primary_arr,
    size_t capacity,
    const std::vector<size_t> &starts,
    const std::vector<ThreadWriteSink> &sinks
) {
    std::vector<size_t> primary_ends(starts.size(), 0);
    size_t spill_total = 0;
    for (size_t i = 0; i < sinks.size(); ++i) {
        primary_ends[i] = sinks[i].primary_pos;
        spill_total += sinks[i].spill.total_count;
    }

    const size_t primary_total = BookGeneratorUtils::merge_inplace(primary_arr, primary_ends, starts);
    FinalizedGeneratedArray result;
    result.total = primary_total;
    result.spill_total = spill_total;
    if (spill_total == 0) {
        return result;
    }

    if (capacity - primary_total >= spill_total) {
        size_t offset = primary_total;
        for (const auto &sink : sinks) {
            sink.spill.copy_into(primary_arr + offset);
            offset += sink.spill.total_count;
        }
        result.total = offset;
        return result;
    }

    result.exact_gather = true;
    result.owned = std::make_unique<uint64_t[]>(primary_total + spill_total);
    if (primary_total > 0) {
        std::memcpy(result.owned.get(), primary_arr, primary_total * sizeof(uint64_t));
    }
    size_t offset = primary_total;
    for (const auto &sink : sinks) {
        sink.spill.copy_into(result.owned.get() + offset);
        offset += sink.spill.total_count;
    }
    result.total = offset;
    return result;
}

enum class BufferedBaseKind : uint8_t {
    Plain = 0,
    Derive = 1,
};

struct BufferedAcceptedBoards {
    std::array<uint64_t, kAdGenBatchSize> boards{};
    std::array<uint8_t, kAdGenBatchSize> kinds{};
    size_t count = 0;
};

template <typename TransformFn>
void transform_boards_batch(const uint64_t *src, uint64_t *dst, size_t count, TransformFn transform) {
    #pragma omp simd
    for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
        dst[static_cast<size_t>(i)] = transform(src[static_cast<size_t>(i)]);
    }
}

void apply_canonical_batch(const uint64_t *src, uint64_t *dst, size_t count, int symm_mode) {
    CanonicalBatch::canonicalize_by_mode(src, dst, count, symm_mode);
}

size_t filter_pattern_matching_boards(
    const uint64_t *src,
    size_t count,
    const std::vector<uint64_t> &masks,
    uint64_t *dst
) {
    if (count == 0) {
        return 0;
    }
    if (masks.empty()) {
        std::memcpy(dst, src, count * sizeof(uint64_t));
        return count;
    }

    std::array<uint8_t, kAdMaxDerivedBoards> keep{};
    for (uint64_t mask : masks) {
        #pragma omp simd
        for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
            keep[static_cast<size_t>(i)] = static_cast<uint8_t>(
                keep[static_cast<size_t>(i)] |
                static_cast<uint8_t>(((src[static_cast<size_t>(i)] & mask) == mask) ? 1U : 0U)
            );
        }
    }

    size_t kept_count = 0;
    for (size_t i = 0; i < count; ++i) {
        if (keep[i] != 0U) {
            dst[kept_count++] = src[i];
        }
    }
    return kept_count;
}

void process_buffered_accepted_boards(
    const BufferedAcceptedBoards &accepted,
    uint32_t original_board_sum,
    uint32_t spawn_delta,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    const AdvancedPatternSpec &spec,
    uint64_t *hashmap,
    uint64_t hashmask,
    ThreadWriteSink &sink
) {
    std::array<uint64_t, kAdMaxDerivedBoards> filtered_derived{};
    std::array<uint64_t, kAdMaxDerivedBoards> hashed_idx{};
    for (size_t accepted_index = 0; accepted_index < accepted.count; ++accepted_index) {
        const uint64_t canon = accepted.boards[accepted_index];
        if (accepted.kinds[accepted_index] == static_cast<uint8_t>(BufferedBaseKind::Plain)) {
            sink.append_one(canon);
            continue;
        }
        DeriveResult derived = derive(canon, original_board_sum + spawn_delta, tiles_table, param);
        if (!derived.is_valid) {
            continue;
        }
        if (!derived.is_derived) {
            sink.append_one(canon);
            continue;
        }
        const size_t filtered_count = filter_pattern_matching_boards(
            derived.boards.data(),
            derived.count,
            spec.pattern_masks,
            filtered_derived.data()
        );
        if (filtered_count == 0) {
            continue;
        }
        apply_canonical_batch(filtered_derived.data(), filtered_derived.data(), filtered_count, spec.symm_mode);
        for (size_t i = 0; i < filtered_count; ++i) {
            hashed_idx[i] = BookGeneratorUtils::hash_board(filtered_derived[i]) & hashmask;
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(hashmap + static_cast<size_t>(hashed_idx[i]), 1, 1);
#endif
        }
        for (size_t i = 0; i < filtered_count; ++i) {
            constexpr size_t kPrefetchDistance = 8;
            if (i + kPrefetchDistance < filtered_count) {
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(hashmap + static_cast<size_t>(hashed_idx[i + kPrefetchDistance]), 1, 1);
#endif
            }
            const uint64_t board = filtered_derived[i];
            const size_t hash_index = static_cast<size_t>(hashed_idx[i]);
            if (hashmap[hash_index] == board) {
                continue;
            }
            hashmap[hash_index] = board;
            sink.append_one(board);
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
    size_t capacity_floor,
    bool isfree
) {
    const size_t min_length = isfree ? 9999999ULL : 6999999ULL;
    const size_t length = capacity_from_factor(arr0.size, length_factor, min_length, capacity_floor);
    auto arr1 = std::unique_ptr<uint64_t[]>(new uint64_t[length]);
    auto arr2 = std::unique_ptr<uint64_t[]>(new uint64_t[length]);
    std::vector<size_t> starts = build_segment_starts(length, n);
    std::vector<ThreadWriteSink> sinks1 = create_write_sinks(arr1.get(), starts, length);
    std::vector<ThreadWriteSink> sinks2 = create_write_sinks(arr2.get(), starts, length);
    uint64_t hashmap1_mask = static_cast<uint64_t>(hashmap1.size() - 1);
    uint64_t hashmap2_mask = static_cast<uint64_t>(hashmap2.size() - 1);

    const size_t total_tasks = arr0.size;
    const size_t chunk_size = std::min<size_t>(1000000ULL, total_tasks / static_cast<size_t>(n * 5) + 1ULL) * static_cast<size_t>(n);
    const size_t chunks_count = (total_tasks + chunk_size - 1) / chunk_size;

    #pragma omp parallel for num_threads(n)
    for (int s = 0; s < n; ++s) {
        ThreadWriteSink &sink1 = sinks1[static_cast<size_t>(s)];
        ThreadWriteSink &sink2 = sinks2[static_cast<size_t>(s)];
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
            CanonicalBatch::canonicalize_inplace(buffer1.data(), buffer1_count, spec.symm_mode);
            flush_candidate_buffer_scalar(buffer1.data(), kinds1.data(), buffer1_count, hashmap1.data(), hashmap1_mask, accepted);
            process_buffered_accepted_boards(
                accepted,
                original_board_sum,
                2U,
                tiles_table,
                param,
                spec,
                hashmap1.data(),
                hashmap1_mask,
                sink1
            );
            buffer1_count = 0;
        };

        auto flush_spawn4 = [&]() {
            if (buffer2_count == 0) {
                return;
            }
            CanonicalBatch::canonicalize_inplace(buffer2.data(), buffer2_count, spec.symm_mode);
            flush_candidate_buffer_scalar(buffer2.data(), kinds2.data(), buffer2_count, hashmap2.data(), hashmap2_mask, accepted);
            process_buffered_accepted_boards(
                accepted,
                original_board_sum,
                4U,
                tiles_table,
                param,
                spec,
                hashmap2.data(),
                hashmap2_mask,
                sink2
            );
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
                        buffer1[buffer1_count] = newt;
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
                        buffer2[buffer2_count] = newt;
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
    }

    FinalizedGeneratedArray finalized_arr1 = finalize_generated_array(arr1.get(), length, starts, sinks1);
    FinalizedGeneratedArray finalized_arr2 = finalize_generated_array(arr2.get(), length, starts, sinks2);
    return {
        finalized_arr1.total,
        finalized_arr2.total,
        collect_total_counts(sinks1),
        collect_total_counts(sinks2),
        finalized_arr1.owned ? std::move(finalized_arr1.owned) : std::move(arr1),
        finalized_arr2.owned ? std::move(finalized_arr2.owned) : std::move(arr2),
        finalized_arr1.spill_total,
        finalized_arr2.spill_total,
        finalized_arr1.exact_gather,
        finalized_arr2.exact_gather
    };
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
    size_t capacity_floor,
    bool isfree
) {
    const size_t min_length = isfree ? 9999999ULL : 6999999ULL;
    const size_t length = capacity_from_factor(arr0.size, length_factor, min_length, capacity_floor);
    auto arr1 = std::unique_ptr<uint64_t[]>(new uint64_t[length]);
    auto arr2 = std::unique_ptr<uint64_t[]>(new uint64_t[length]);
    std::vector<size_t> starts = build_segment_starts(length, n);
    std::vector<ThreadWriteSink> sinks1 = create_write_sinks(arr1.get(), starts, length);
    std::vector<ThreadWriteSink> sinks2 = create_write_sinks(arr2.get(), starts, length);
    const uint64_t hashmap1_mask = static_cast<uint64_t>(hashmap1.size() - 1);
    const uint64_t hashmap2_mask = static_cast<uint64_t>(hashmap2.size() - 1);

    const size_t total_tasks = arr0.size;
    const size_t chunk_size = std::min<size_t>(1000000ULL, total_tasks / static_cast<size_t>(n * 5) + 1ULL) * static_cast<size_t>(n);
    const size_t chunks_count = (total_tasks + chunk_size - 1) / chunk_size;

    #pragma omp parallel for num_threads(n)
    for (int s = 0; s < n; ++s) {
        ThreadWriteSink &sink1 = sinks1[static_cast<size_t>(s)];
        ThreadWriteSink &sink2 = sinks2[static_cast<size_t>(s)];
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
            CanonicalBatch::canonicalize_inplace(buffer1.data(), buffer1_count, spec.symm_mode);
            flush_candidate_buffer_avx512(buffer1.data(), kinds1.data(), buffer1_count, hashmap1.data(), hashmap1_mask, accepted);
            process_buffered_accepted_boards(
                accepted,
                original_board_sum,
                2U,
                tiles_table,
                param,
                spec,
                hashmap1.data(),
                hashmap1_mask,
                sink1
            );
            buffer1_count = 0;
        };

        auto flush_spawn4 = [&]() {
            if (buffer2_count == 0) {
                return;
            }
            CanonicalBatch::canonicalize_inplace(buffer2.data(), buffer2_count, spec.symm_mode);
            flush_candidate_buffer_avx512(buffer2.data(), kinds2.data(), buffer2_count, hashmap2.data(), hashmap2_mask, accepted);
            process_buffered_accepted_boards(
                accepted,
                original_board_sum,
                4U,
                tiles_table,
                param,
                spec,
                hashmap2.data(),
                hashmap2_mask,
                sink2
            );
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
                        buffer1[buffer1_count] = newt;
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
                        buffer2[buffer2_count] = newt;
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
    }

    FinalizedGeneratedArray finalized_arr1 = finalize_generated_array(arr1.get(), length, starts, sinks1);
    FinalizedGeneratedArray finalized_arr2 = finalize_generated_array(arr2.get(), length, starts, sinks2);
    return {
        finalized_arr1.total,
        finalized_arr2.total,
        collect_total_counts(sinks1),
        collect_total_counts(sinks2),
        finalized_arr1.owned ? std::move(finalized_arr1.owned) : std::move(arr1),
        finalized_arr2.owned ? std::move(finalized_arr2.owned) : std::move(arr2),
        finalized_arr1.spill_total,
        finalized_arr2.spill_total,
        finalized_arr1.exact_gather,
        finalized_arr2.exact_gather
    };
}

bool cpu_has_ad_gen_avx512_uncached() {
    return UniqueUtils::cpu_has_avx512_dq_bw_vl();
}

bool env_flag_enabled(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return value[0] != '0' && value[0] != 'f' && value[0] != 'F' && value[0] != 'n' && value[0] != 'N';
}

bool ad_validate_step_trigger(int step, uint32_t ini_board_sum, const AdvancedMaskParam &param) {
    return ((step + static_cast<int>(ini_board_sum % 64U / 2U)) % 32) ==
        ((static_cast<int>(param.small_tile_sum_limit / 2U)) % 32) + 1;
}

struct PostValidateCapacityFloor {
    size_t capacity = 0;
    int remaining_layers = 0;
};

size_t post_validate_capacity_floor_from_layer(size_t layer_size) {
    constexpr long double kPostValidateFloorMargin = 1.20L;
    const long double scaled = static_cast<long double>(layer_size) * kPostValidateFloorMargin;
    return scaled >= static_cast<long double>(std::numeric_limits<size_t>::max())
        ? std::numeric_limits<size_t>::max()
        : static_cast<size_t>(std::ceil(scaled));
}

size_t current_post_validate_capacity_floor(const PostValidateCapacityFloor &floor) {
    return floor.remaining_layers > 0 ? floor.capacity : 0U;
}

size_t proportional_capacity_floor(size_t total_floor, size_t part_size, size_t total_size) {
    if (total_floor == 0U || part_size == 0U || total_size == 0U) {
        return 0U;
    }
    const long double scaled =
        static_cast<long double>(total_floor) *
        static_cast<long double>(part_size) /
        static_cast<long double>(total_size);
    return scaled >= static_cast<long double>(std::numeric_limits<size_t>::max())
        ? std::numeric_limits<size_t>::max()
        : static_cast<size_t>(std::ceil(scaled));
}

GenBoardsAdDispatch resolve_gen_boards_ad_dispatch() {
    if (env_flag_enabled("NATIVE_DISABLE_AD_GEN_AVX512")) {
        return {AdGenMode::Scalar, gen_boards_ad_scalar};
    }
#ifndef NDEBUG
    if (!env_flag_enabled("NATIVE_FORCE_AD_GEN_AVX512")) {
        return {AdGenMode::Scalar, gen_boards_ad_scalar};
    }
#endif
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
    size_t capacity_floor,
    bool isfree
) {
    static const GenBoardsAdFn fn = gen_boards_ad_dispatch().fn;
    return fn(arr0, spec, hashmap1, hashmap2, original_board_sum, tiles_table, param, n, length_factor, capacity_floor, isfree);
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
    bool isfree,
    size_t capacity_floor
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
        length_factor *= arr0t.size > static_cast<size_t>(1e8) ? 1.15 : 1.2;
        length_factor *= length_factor_multiplier;
        const size_t segment_capacity_floor =
            proportional_capacity_floor(capacity_floor, arr0t.size, arr0.size());

        GenBoardsAdResult result = gen_boards_ad(
            arr0t,
            spec,
            hashmap1,
            hashmap2,
            board_sum,
            tiles_table,
            param,
            n,
            length_factor,
            segment_capacity_floor,
            isfree
        );
        validate_length_and_balance(
            arr0t.size,
            result.total_arr2,
            result.total_arr1,
            result.counts1,
            result.counts2,
            length_factor,
            true,
            result.spill_arr1,
            result.spill_arr2,
            result.exact_gather_arr1,
            result.exact_gather_arr2,
            segment_capacity_floor
        );

        for (int idx = 0; idx < n; ++idx) {
            actual_lengths2[seg_index * static_cast<size_t>(n) + static_cast<size_t>(idx)] = result.counts2[static_cast<size_t>(idx)];
            actual_lengths1[seg_index * static_cast<size_t>(n) + static_cast<size_t>(idx)] = result.counts1[static_cast<size_t>(idx)];
        }
        if (!length_factors.empty()) {
            length_factors.erase(length_factors.begin());
        }
        length_factors.push_back(static_cast<double>(result.total_arr2) / static_cast<double>(1 + arr0t.size));
        length_factors_list[seg_index] = length_factors;
        gen_time += wall_time_seconds() - t_segment;

        uint64_t *segment_arr1 = result.finalized_arr1.get();
        uint64_t *segment_arr2 = result.finalized_arr2.get();
        auto [unique_arr1_length, unique_arr2_length] = BookGeneratorUtils::sort_and_unique_two_arrays_concurrently(
            segment_arr1,
            result.total_arr1,
            segment_arr2,
            result.total_arr2,
            n
        );
        arr1s.emplace_back(segment_arr1, segment_arr1 + unique_arr1_length);
        arr2s.emplace_back(segment_arr2, segment_arr2 + unique_arr2_length);
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
    const uint32_t progress_total = build_progress_total(options);
    const FileIOUtils::DirectIoConfig io_config = FileIOUtils::direct_io_config_from_options(options);
    ensure_ad_generate_stats_header(options);
    AdGenerateStatsRecord total_record;
    total_record.stage = "_total";
    PostValidateCapacityFloor post_validate_floor;

    for (int i = 1; i < options.steps - 1; ++i) {
        RestartResult restart = handle_restart_ad(i, options.pathname, arr_init, started, io_config, options.compress_temp_files);
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
        NativeDiagnostics::mark(
            "AD.generate step begin step=" + std::to_string(i) +
            " live=" + std::to_string(d0.size())
        );
        FormationProgress::update_build_progress(static_cast<uint32_t>(i), progress_total);

        uint32_t board_sum = static_cast<uint32_t>(2 * i + ini_board_sum - 2);
        AdGenerateStatsRecord stats_record;
        bool has_stats_record = false;
        stats_record.step = i;
        stats_record.input_live = static_cast<uint64_t>(d0.size());
        const size_t capacity_floor = current_post_validate_capacity_floor(post_validate_floor);
        if (post_validate_floor.remaining_layers > 0) {
            --post_validate_floor.remaining_layers;
        }
        if (d0.size() < init_params.segment_size) {
            double t0 = wall_time_seconds();
            double length_factor = BookGenerator::predict_next_length_factor_quadratic(init_params.length_factors);
            length_factor *= d0.size() > static_cast<size_t>(1e8) ? 1.15 : 1.2;
            length_factor *= init_params.length_factor_multiplier;
            stats_record.stage = "normal";
            stats_record.length_factor =
                effective_length_factor_for_capacity(d0.size(), length_factor, capacity_floor);
            NativeDiagnostics::mark(
                "AD.generate path step=" + std::to_string(i) +
                " stage=normal length_factor=" + std::to_string(stats_record.length_factor)
            );
            if (hashmap1.empty()) {
                BookGenerator::update_hashmap_length(hashmap1, d0.size());
                BookGenerator::update_hashmap_length(hashmap2, d0.size());
            }
            NativeDiagnostics::mark("AD.generate gen_boards_ad begin step=" + std::to_string(i));
            GenBoardsAdResult result = gen_boards_ad(
                {d0.data(), d0.size()},
                spec,
                hashmap1,
                hashmap2,
                board_sum,
                masker.tiles_combination_table,
                masker.param,
                n,
                length_factor,
                capacity_floor,
                options.is_free
            );
            NativeDiagnostics::mark("AD.generate gen_boards_ad done step=" + std::to_string(i));
            validate_length_and_balance(
                d0.size(),
                result.total_arr2,
                result.total_arr1,
                result.counts1,
                result.counts2,
                length_factor,
                false,
                result.spill_arr1,
                result.spill_arr2,
                result.exact_gather_arr1,
                result.exact_gather_arr2,
                capacity_floor
            );
            double t1 = wall_time_seconds();

            auto [new_length_factors, new_length_factors_list] =
                update_parameters(d0.size(), result.total_arr2, init_params.length_factors, init_params.length_factors_list_path);
            init_params.length_factors = std::move(new_length_factors);
            init_params.length_factors_list = std::move(new_length_factors_list);
            double mean_count = result.counts2.empty()
                ? 0.0
                : std::accumulate(result.counts2.begin(), result.counts2.end(), 0.0) / static_cast<double>(result.counts2.size());
            init_params.length_factor_multiplier = mean_count > 0.0
                ? static_cast<double>(*std::max_element(result.counts2.begin(), result.counts2.end())) / mean_count
                : 1.0;
            uint64_t *result_arr1 = result.finalized_arr1.get();
            uint64_t *result_arr2 = result.finalized_arr2.get();
            NativeDiagnostics::mark("AD.generate sort_unique begin step=" + std::to_string(i));
            auto [unique_arr1_length, unique_arr2_length] = BookGeneratorUtils::sort_and_unique_two_arrays_concurrently(
                result_arr1,
                result.total_arr1,
                result_arr2,
                result.total_arr2,
                n
            );
            NativeDiagnostics::mark("AD.generate sort_unique done step=" + std::to_string(i));
            stats_record.arr1_raw = static_cast<uint64_t>(result.total_arr1);
            stats_record.arr2_raw = static_cast<uint64_t>(result.total_arr2);
            stats_record.arr1_unique = static_cast<uint64_t>(unique_arr1_length);
            stats_record.arr2_unique = static_cast<uint64_t>(unique_arr2_length);
            std::vector<uint64_t> next_d1t(result_arr1, result_arr1 + unique_arr1_length);
            std::vector<uint64_t> next_d2(result_arr2, result_arr2 + unique_arr2_length);
            double t2 = wall_time_seconds();
            std::vector<uint64_t> pivots;
            for (int pt = 1; pt < n; ++pt) {
                if (d0.empty()) {
                    pivots.push_back(static_cast<uint64_t>(pt) * (1ULL << 50) / static_cast<uint64_t>(n));
                } else {
                    pivots.push_back(d0[static_cast<size_t>(pt) * d0.size() / static_cast<size_t>(n)]);
                }
            }
            std::vector<std::vector<uint64_t>> d1_inputs = {std::move(d1), std::move(next_d1t)};
            NativeDiagnostics::mark("AD.generate merge begin step=" + std::to_string(i));
            d1 = BookGeneratorUtils::merge_deduplicate_all_concat(d1_inputs, pivots, n);
            NativeDiagnostics::mark("AD.generate merge done step=" + std::to_string(i));
            d0 = std::move(d1);
            d1 = std::move(next_d2);
            double t3 = wall_time_seconds();
            stats_record.output_live = static_cast<uint64_t>(d0.size());
            stats_record.generate_seconds = t1 - t0;
            stats_record.sort_unique_seconds = t2 - t1;
            stats_record.merge_seconds = t3 - t2;
            has_stats_record = true;

            if (!hashmap1.empty()) {
                BookGenerator::update_hashmap_length(hashmap1, d1.size());
                BookGenerator::update_hashmap_length(hashmap2, d1.size());
            }
        } else {
            NativeDiagnostics::mark(
                "AD.generate path step=" + std::to_string(i) +
                " stage=big live=" + std::to_string(d0.size())
            );
            if (hashmap1.empty()) {
                size_t capacity = BookGeneratorUtils::largest_power_of_2(
                    static_cast<uint64_t>(20971520ULL * std::max(1.0, std::round(BookGenerator::get_system_memory_gb()) * 0.75))
                );
                hashmap1.assign(capacity, 0);
                hashmap2.assign(capacity, 0);
            }
            GenBoardsBigAdResult big_result = gen_boards_big_ad(
                d0, spec, hashmap1, hashmap2, board_sum, masker.tiles_combination_table, masker.param, n,
                init_params.length_factors_list, init_params.length_factor_multiplier, options.is_free, capacity_floor
            );
            stats_record.stage = "big";

            std::vector<uint64_t> pivots;
            for (int pt = 1; pt < n; ++pt) {
                if (d0.empty()) {
                    pivots.push_back(static_cast<uint64_t>(pt) * (1ULL << 50) / static_cast<uint64_t>(n));
                } else {
                    pivots.push_back(d0[static_cast<size_t>(pt) * d0.size() / static_cast<size_t>(n)]);
                }
            }
            uint64_t big_arr1_raw = 0U;
            uint64_t big_arr2_raw = 0U;
            for (const auto &part : big_result.arr1s) {
                big_arr1_raw += static_cast<uint64_t>(part.size());
            }
            for (const auto &part : big_result.arr2s) {
                big_arr2_raw += static_cast<uint64_t>(part.size());
            }
            big_result.arr1s.push_back(std::move(d1));
            d1 = BookGeneratorUtils::merge_deduplicate_all_concat(big_result.arr2s, pivots, n);
            d0 = BookGeneratorUtils::merge_deduplicate_all_concat(big_result.arr1s, pivots, n);
            double t3 = wall_time_seconds();
            stats_record.arr1_raw = big_arr1_raw;
            stats_record.arr2_raw = big_arr2_raw;
            stats_record.arr1_unique = static_cast<uint64_t>(d0.size());
            stats_record.arr2_unique = static_cast<uint64_t>(d1.size());
            stats_record.output_live = static_cast<uint64_t>(d0.size());
            stats_record.generate_seconds = big_result.gen_time;
            stats_record.sort_unique_seconds = big_result.t2 - (big_result.t0 + big_result.gen_time);
            stats_record.merge_seconds = t3 - big_result.t2;
            has_stats_record = true;
            init_params.length_factors_list = std::move(big_result.length_factors_list);
            init_params.length_factor_multiplier = big_result.length_factor_multiplier;
            init_params.length_factors = BookGenerator::harmonic_mean_by_column(init_params.length_factors_list);
            save_length_factors(init_params.length_factors_list_path, init_params.length_factors_list);
        }

        const double validate_t0 = wall_time_seconds();
        if (ad_validate_step_trigger(i, ini_board_sum, masker.param)) {
            post_validate_floor.capacity = post_validate_capacity_floor_from_layer(d0.size());
            post_validate_floor.remaining_layers = 3;
            d0 = validate_layer(d0, board_sum + 2U, masker.tiles_combination_table, masker.param, n);
            d1 = validate_layer(d1, board_sum + 4U, masker.tiles_combination_table, masker.param, n);
            debug_log("validate step " + std::to_string(i));
        }
        stats_record.validate_seconds = wall_time_seconds() - validate_t0;

        clear_u64_buffer(hashmap1, n);
        const double write_t0 = wall_time_seconds();
        NativeDiagnostics::mark("AD.generate write begin step=" + std::to_string(i));
        write_raw_file(options.pathname + std::to_string(i), d0, io_config, options.compress_temp_files);
        NativeDiagnostics::mark("AD.generate write done step=" + std::to_string(i));
        stats_record.write_seconds = wall_time_seconds() - write_t0;
        if (has_stats_record) {
            append_ad_generate_stats_record(options, stats_record);
            total_record.input_live += stats_record.input_live;
            total_record.arr1_raw += stats_record.arr1_raw;
            total_record.arr2_raw += stats_record.arr2_raw;
            total_record.arr1_unique += stats_record.arr1_unique;
            total_record.arr2_unique += stats_record.arr2_unique;
            total_record.output_live += stats_record.output_live;
            total_record.generate_seconds += stats_record.generate_seconds;
            total_record.sort_unique_seconds += stats_record.sort_unique_seconds;
            total_record.merge_seconds += stats_record.merge_seconds;
            total_record.validate_seconds += stats_record.validate_seconds;
            total_record.write_seconds += stats_record.write_seconds;
        }
        std::swap(hashmap1, hashmap2);
    }
    total_record.length_factor = total_record.input_live > 0U
        ? static_cast<double>(total_record.output_live) / static_cast<double>(total_record.input_live)
        : 0.0;
    append_ad_generate_stats_record(options, total_record);
    return {started, std::move(d0), std::move(d1)};
}

void run_python_ad_solver_bridge(
    const std::vector<uint64_t> &arr_init,
    const AdvancedPatternSpec &spec,
    const RunOptions &options
) {
    nb::gil_scoped_acquire acquire;
    nb::object helper = nb::module_::import_("engine_core.BookBuilder").attr("run_pattern_solve_ad_bridge");
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
    FormationProgress::reset_build_progress(build_progress_total(options));
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
