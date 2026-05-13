#include "BookSolver.h"

#include "AdaptiveIndex.h"
#include "BookGenerator.h"
#include "BoardMover.h"
#include "Calculator.h"
#include "CompressionBridge.h"
#include "FileIOUtils.h"
#include "Formation.h"
#include "HybridSearch.h"
#include "NativeLzma.h"
#include "UniqueUtils.h"
#include "VBoardMover.h"

#include <algorithm>
#include <cmath>
#include <cstring>
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
#include <type_traits>

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

constexpr int kOptimalBranchOnlyStartStep = 21;

template <typename T> SplitLayer<T> split_layer_from_entry_bytes(const uint8_t *bytes, size_t size) {
    if ((size % sizeof(SuccessEntry<T>)) != 0U) {
        throw std::runtime_error("misaligned entry bytes for split layer");
    }
    const size_t entry_count = size / sizeof(SuccessEntry<T>);
    SplitLayer<T> layer = allocate_split_layer<T>(entry_count);
    if (entry_count == 0U) {
        return layer;
    }
    std::unique_ptr<SuccessEntry<T>[]> buffer(new SuccessEntry<T>[io_chunk_entries<T>()]);
    size_t offset = 0U;
    while (offset < entry_count) {
        const size_t current = std::min(io_chunk_entries<T>(), entry_count - offset);
        std::memcpy(
            buffer.get(),
            bytes + offset * sizeof(SuccessEntry<T>),
            current * sizeof(SuccessEntry<T>)
        );
        for (size_t i = 0; i < current; ++i) {
            layer.boards[offset + i] = buffer[i].board;
            layer.success[offset + i] = buffer[i].success;
        }
        offset += current;
    }
    return layer;
}

template <typename T> std::vector<uint8_t> split_layer_to_entry_bytes(const SplitLayer<T> &data) {
    std::vector<uint8_t> bytes(data.size() * sizeof(SuccessEntry<T>));
    if (data.empty()) {
        return bytes;
    }
    std::unique_ptr<SuccessEntry<T>[]> buffer(new SuccessEntry<T>[io_chunk_entries<T>()]);
    size_t offset = 0U;
    while (offset < data.size()) {
        const size_t current = std::min(io_chunk_entries<T>(), data.size() - offset);
        for (size_t i = 0; i < current; ++i) {
            buffer[i].board = data.boards[offset + i];
            buffer[i].success = data.success[offset + i];
        }
        std::memcpy(
            bytes.data() + offset * sizeof(SuccessEntry<T>),
            buffer.get(),
            current * sizeof(SuccessEntry<T>)
        );
        offset += current;
    }
    return bytes;
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

bool is_xz_magic_header(const std::vector<uint8_t> &bytes) {
    static constexpr uint8_t kMagic[] = {0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00};
    return bytes.size() >= sizeof(kMagic) &&
           std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) == 0;
}

void remove_file_if_exists(const std::string &path) {
    std::error_code ec;
    fs::remove(path, ec);
}

void remove_temp_raw_layer_files(const std::string &path) {
    remove_file_if_exists(path);
    remove_file_if_exists(path + ".7z");
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

double throughput_mbps_for(uint64_t count, double seconds) {
    return seconds > 0.0 ? static_cast<double>(count) / seconds / 1e6 : 0.0;
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

template <typename T>
LayerVector<T> read_layer_file(const std::string &path, FileIOUtils::DirectIoConfig config = {}) {
    return FileIOUtils::read_binary_vector_direct<SuccessEntry<T>>(path, config);
}

template <typename T>
SplitLayer<T> read_split_layer_file(const std::string &path, FileIOUtils::DirectIoConfig config = {}) {
    const FileIOUtils::DirectIoConfig normalized = FileIOUtils::normalize_direct_io_config(config);
    if (!normalized.enabled) {
        return split_layer_from_entries<T>(read_layer_file<T>(path, config));
    }

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }
    const size_t size = FileIOUtils::checked_tellg(file, path);
    if (size % sizeof(SuccessEntry<T>) != 0U) {
        throw std::runtime_error("misaligned binary file size: " + path);
    }

    const size_t entry_count = size / sizeof(SuccessEntry<T>);
    SplitLayer<T> layer = allocate_split_layer<T>(entry_count);
    if (entry_count == 0U) {
        return layer;
    }

    FileIOUtils::DirectSequentialReader reader(path, static_cast<uint64_t>(size), normalized);
    std::unique_ptr<SuccessEntry<T>[]> buffer(new SuccessEntry<T>[io_chunk_entries<T>()]);
    size_t offset = 0U;
    while (offset < entry_count) {
        const size_t current = std::min(io_chunk_entries<T>(), entry_count - offset);
        reader.read(buffer.get(), current * sizeof(SuccessEntry<T>));
        for (size_t i = 0; i < current; ++i) {
            layer.boards[offset + i] = buffer[i].board;
            layer.success[offset + i] = buffer[i].success;
        }
        offset += current;
    }
    reader.close();
    return layer;
}

template <typename T>
SplitLayer<T> read_split_layer_file_or_archive(const std::string &path, FileIOUtils::DirectIoConfig config = {}) {
    if (fs::exists(path)) {
        return read_split_layer_file<T>(path, config);
    }
    const std::string archive_path = path + ".7z";
    std::vector<uint8_t> decompressed;
    std::vector<uint8_t> header = FileIOUtils::read_binary_bytes_range(archive_path, 0, 6);
    if (is_xz_magic_header(header)) {
        std::vector<uint8_t> archive_bytes = FileIOUtils::read_binary_bytes(archive_path);
        decompressed = decompress_xz_block_native(archive_bytes.data(), archive_bytes.size());
        if (decompressed.empty() && !archive_bytes.empty()) {
            return {};
        }
    } else if (!decompress_7z_archive_to_bytes_streaming(archive_path, decompressed)) {
        return {};
    }
    return split_layer_from_entry_bytes<T>(decompressed.data(), decompressed.size());
}

template <typename T>
void write_layer_file(
    const std::string &path,
    const LayerVector<T> &data,
    FileIOUtils::DirectIoConfig config = {}
) {
    FileIOUtils::write_binary_vector_direct(path, data, config);
}

template <typename T>
void write_layer_file(
    const std::string &path,
    const SplitLayer<T> &data,
    FileIOUtils::DirectIoConfig config = {}
) {
    FileIOUtils::DirectAppendWriter out(
        path,
        static_cast<uint64_t>(data.size()) * static_cast<uint64_t>(sizeof(SuccessEntry<T>)),
        config
    );
    if (!data.empty()) {
        std::unique_ptr<SuccessEntry<T>[]> buffer(new SuccessEntry<T>[io_chunk_entries<T>()]);
        size_t offset = 0U;
        while (offset < data.size()) {
            const size_t current = std::min(io_chunk_entries<T>(), data.size() - offset);
            for (size_t j = 0; j < current; ++j) {
                buffer[j].board = data.boards[offset + j];
                buffer[j].success = data.success[offset + j];
            }
            out.append(buffer.get(), current * sizeof(SuccessEntry<T>));
            offset += current;
        }
    }
    out.close();
}

template <typename T>
void write_split_layer_archive_file(const std::string &archive_path, const SplitLayer<T> &data, int lvl = 1) {
    std::vector<uint8_t> bytes = split_layer_to_entry_bytes(data);
    const std::string entry_name = fs::path(archive_path).stem().string() + ".book";
    if (!compress_bytes_to_7z_archive_streaming(bytes.data(), bytes.size(), archive_path, entry_name, lvl)) {
        throw std::runtime_error("failed to write archived split layer: " + archive_path);
    }
    std::error_code ec;
    fs::remove(fs::path(archive_path).replace_extension().string(), ec);
}

std::vector<uint64_t> read_raw_file(const std::string &path, FileIOUtils::DirectIoConfig config = {}) {
    if (fs::exists(path)) {
        return FileIOUtils::read_binary_vector_direct<uint64_t>(path, config);
    }
    return read_temp_uint64_archive(path + ".7z");
}

template <typename T> LayerVector<T> final_situation_process(
    const std::vector<uint64_t> &boards,
    const PatternSpec &spec,
    int target,
    const std::string &success_rate_dtype
) {
    LayerVector<T> result(boards.size());
    const T max_scale = max_scale_value_for_dtype<T>(success_rate_dtype);
    const T zero_val = zero_value_for_dtype<T>(success_rate_dtype);
    int num_threads = std::max(1, omp_get_max_threads());
    #pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < static_cast<int64_t>(boards.size()); ++i) {
        result[static_cast<size_t>(i)].board = boards[static_cast<size_t>(i)];
        result[static_cast<size_t>(i)].success =
            is_success_by_shifts(boards[static_cast<size_t>(i)], target, spec.success_shifts) ? max_scale : zero_val;
    }
    size_t count = 0U;
    for (size_t i = 0; i < result.size(); ++i) {
        if (result[i].success > zero_val) {
            result[count++] = result[i];
        }
    }
    result.resize(count);
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
        layer0 = final_situation_process<T>(d0, spec, options.target, options.success_rate_dtype);
        layer1 = final_situation_process<T>(d1, spec, options.target, options.success_rate_dtype);
        const FileIOUtils::DirectIoConfig io_config = FileIOUtils::direct_io_config_from_options(options);
        write_layer_file(options.pathname + std::to_string(options.steps - 2) + ".book", layer0, io_config);
        write_layer_file(options.pathname + std::to_string(options.steps - 1) + ".book", layer1, io_config);
    }
    auto raw_path = options.pathname + std::to_string(options.steps - 2);
    remove_temp_raw_layer_files(raw_path);
    return {make_pattern_layer(std::move(layer0)), make_pattern_layer(std::move(layer1))};
}

std::string classic_solve_stats_file_path(const RunOptions &options) {
    return options.pathname + "classic_solve_stats.csv";
}

std::string classic_solve_stats_header() {
    return "stage,step,input_live,post_zero_live,future1_live,future2_live,future2_post_threshold_live,deletion_threshold,max_success,total_seconds,throughput_mbps,compute_seconds,compute_throughput_mbps,current_read_seconds,index_seconds,recalculate_seconds,zero_compact_seconds,max_scan_seconds,current_write_seconds,future_compact_seconds,future_write_seconds,compress_seconds,time";
}

void ensure_classic_solve_stats_header(const RunOptions &options) {
    {
        std::error_code ec;
        fs::remove(options.pathname + "stats.txt", ec);
    }
    const std::string path = classic_solve_stats_file_path(options);
    if (fs::exists(path)) {
        std::ifstream in(path);
        std::string first_line;
        if (std::getline(in, first_line) && first_line == classic_solve_stats_header()) {
            return;
        }
        in.close();
        std::error_code ec;
        fs::remove(path, ec);
    }
    std::ofstream file(path, std::ios::app);
    file << classic_solve_stats_header() << "\n";
}

struct ClassicSolveStatsRecord {
    std::string stage = "solve";
    int step = -1;
    uint64_t input_live = 0U;
    uint64_t post_zero_live = 0U;
    uint64_t future1_live = 0U;
    uint64_t future2_live = 0U;
    uint64_t future2_post_threshold_live = 0U;
    double deletion_threshold = 0.0;
    double max_success = 0.0;
    double current_read_seconds = 0.0;
    double index_seconds = 0.0;
    double recalculate_seconds = 0.0;
    double zero_compact_seconds = 0.0;
    double max_scan_seconds = 0.0;
    double current_write_seconds = 0.0;
    double future_compact_seconds = 0.0;
    double future_write_seconds = 0.0;
    double compress_seconds = 0.0;
};

void append_classic_solve_stats_record(
    const RunOptions &options,
    const ClassicSolveStatsRecord &record
) {
    ensure_classic_solve_stats_header(options);
    const double compute_seconds =
        record.index_seconds + record.recalculate_seconds + record.zero_compact_seconds
        + record.max_scan_seconds + record.future_compact_seconds;
    const double total_seconds =
        record.current_read_seconds + compute_seconds + record.current_write_seconds
        + record.future_write_seconds + record.compress_seconds;
    std::ofstream file(classic_solve_stats_file_path(options), std::ios::app);
    file << record.stage << ","
         << record.step << ","
         << record.input_live << ","
         << record.post_zero_live << ","
         << record.future1_live << ","
         << record.future2_live << ","
         << record.future2_post_threshold_live << ","
         << std::fixed << std::setprecision(6)
         << record.deletion_threshold << ","
         << record.max_success << ","
         << total_seconds << ","
         << throughput_mbps_for(record.input_live, total_seconds) << ","
         << compute_seconds << ","
         << throughput_mbps_for(record.input_live, compute_seconds) << ","
         << record.current_read_seconds << ","
         << record.index_seconds << ","
         << record.recalculate_seconds << ","
         << record.zero_compact_seconds << ","
         << record.max_scan_seconds << ","
         << record.current_write_seconds << ","
         << record.future_compact_seconds << ","
         << record.future_write_seconds << ","
         << record.compress_seconds << ","
         << now_string() << "\n";
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

template <typename T> size_t compact_live_entries_scalar_impl(SplitLayer<T> &arr, T threshold, bool shrink_boards) {
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

template <typename T> size_t compact_by_mask_scalar_impl(SplitLayer<T> &arr, const std::vector<uint8_t> &mask, bool shrink_boards) {
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

template <typename T>
void copy_selected_runs_soa(
    SplitLayer<T> &arr,
    size_t src_index,
    size_t &dst_index,
    uint32_t keep_mask,
    unsigned lane_count
) {
    while (keep_mask != 0U) {
        const unsigned start = static_cast<unsigned>(__builtin_ctz(keep_mask));
        unsigned run_length = 1U;
        while (start + run_length < lane_count && ((keep_mask >> (start + run_length)) & 1U) != 0U) {
            ++run_length;
        }
        std::memmove(
            arr.boards.data() + dst_index,
            arr.boards.data() + src_index + static_cast<size_t>(start),
            static_cast<size_t>(run_length) * sizeof(uint64_t)
        );
        std::memmove(
            arr.success.get() + dst_index,
            arr.success.get() + src_index + static_cast<size_t>(start),
            static_cast<size_t>(run_length) * sizeof(T)
        );
        dst_index += static_cast<size_t>(run_length);
        const uint32_t run_bits = static_cast<uint32_t>(((1ULL << run_length) - 1ULL) << start);
        keep_mask &= ~run_bits;
    }
}

inline uint32_t keep_mask_from_bytes_scalar(const uint8_t *mask, unsigned lane_count) {
    uint32_t keep_mask = 0U;
    for (unsigned lane = 0; lane < lane_count; ++lane) {
        if (mask[lane] != 0U) {
            keep_mask |= (1U << lane);
        }
    }
    return keep_mask;
}

#if defined(__GNUC__) || defined(__clang__)
template <typename T>
__attribute__((target("avx2")))
size_t compact_live_entries_avx2_impl(SplitLayer<T> &arr, T threshold, bool shrink_boards) {
    size_t out = 0U;
    size_t i = 0U;

    if constexpr (std::is_same_v<T, uint32_t>) {
        constexpr unsigned kLanes = 8U;
        const __m256i bias = _mm256_set1_epi32(static_cast<int>(0x80000000U));
        const __m256i threshold_vec = _mm256_xor_si256(_mm256_set1_epi32(static_cast<int>(threshold)), bias);
        for (; i + kLanes <= arr.size(); i += kLanes) {
            const __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(arr.success.get() + i));
            const __m256i biased = _mm256_xor_si256(values, bias);
            const __m256i cmp = _mm256_cmpgt_epi32(biased, threshold_vec);
            const uint32_t keep_mask = static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(cmp)));
            if (keep_mask == 0U) {
                continue;
            }
            if (keep_mask == 0xFFU) {
                if (out != i) {
                    std::memmove(arr.boards.data() + out, arr.boards.data() + i, kLanes * sizeof(uint64_t));
                    std::memmove(arr.success.get() + out, arr.success.get() + i, kLanes * sizeof(T));
                }
                out += kLanes;
                continue;
            }
            copy_selected_runs_soa(arr, i, out, keep_mask, kLanes);
        }
    } else if constexpr (std::is_same_v<T, float>) {
        constexpr unsigned kLanes = 8U;
        const __m256 threshold_vec = _mm256_set1_ps(threshold);
        for (; i + kLanes <= arr.size(); i += kLanes) {
            const __m256 values = _mm256_loadu_ps(arr.success.get() + i);
            const __m256 cmp = _mm256_cmp_ps(values, threshold_vec, _CMP_GT_OQ);
            const uint32_t keep_mask = static_cast<uint32_t>(_mm256_movemask_ps(cmp));
            if (keep_mask == 0U) {
                continue;
            }
            if (keep_mask == 0xFFU) {
                if (out != i) {
                    std::memmove(arr.boards.data() + out, arr.boards.data() + i, kLanes * sizeof(uint64_t));
                    std::memmove(arr.success.get() + out, arr.success.get() + i, kLanes * sizeof(T));
                }
                out += kLanes;
                continue;
            }
            copy_selected_runs_soa(arr, i, out, keep_mask, kLanes);
        }
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        constexpr unsigned kLanes = 4U;
        const __m256i bias = _mm256_set1_epi64x(static_cast<long long>(0x8000000000000000ULL));
        const __m256i threshold_vec = _mm256_xor_si256(
            _mm256_set1_epi64x(static_cast<long long>(threshold)),
            bias
        );
        for (; i + kLanes <= arr.size(); i += kLanes) {
            const __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(arr.success.get() + i));
            const __m256i biased = _mm256_xor_si256(values, bias);
            const __m256i cmp = _mm256_cmpgt_epi64(biased, threshold_vec);
            const uint32_t keep_mask = static_cast<uint32_t>(_mm256_movemask_pd(_mm256_castsi256_pd(cmp)));
            if (keep_mask == 0U) {
                continue;
            }
            if (keep_mask == 0xFU) {
                if (out != i) {
                    std::memmove(arr.boards.data() + out, arr.boards.data() + i, kLanes * sizeof(uint64_t));
                    std::memmove(arr.success.get() + out, arr.success.get() + i, kLanes * sizeof(T));
                }
                out += kLanes;
                continue;
            }
            copy_selected_runs_soa(arr, i, out, keep_mask, kLanes);
        }
    } else if constexpr (std::is_same_v<T, double>) {
        constexpr unsigned kLanes = 4U;
        const __m256d threshold_vec = _mm256_set1_pd(threshold);
        for (; i + kLanes <= arr.size(); i += kLanes) {
            const __m256d values = _mm256_loadu_pd(arr.success.get() + i);
            const __m256d cmp = _mm256_cmp_pd(values, threshold_vec, _CMP_GT_OQ);
            const uint32_t keep_mask = static_cast<uint32_t>(_mm256_movemask_pd(cmp));
            if (keep_mask == 0U) {
                continue;
            }
            if (keep_mask == 0xFU) {
                if (out != i) {
                    std::memmove(arr.boards.data() + out, arr.boards.data() + i, kLanes * sizeof(uint64_t));
                    std::memmove(arr.success.get() + out, arr.success.get() + i, kLanes * sizeof(T));
                }
                out += kLanes;
                continue;
            }
            copy_selected_runs_soa(arr, i, out, keep_mask, kLanes);
        }
    }

    for (; i < arr.size(); ++i) {
        if (arr.success[i] > threshold) {
            arr.boards[out] = arr.boards[i];
            arr.success[out] = arr.success[i];
            ++out;
        }
    }
    arr.length = out;
    if (shrink_boards) {
        arr.boards.resize(out);
    }
    return out;
}

template <typename T>
__attribute__((target("avx512f,avx512dq,avx512bw,avx512vl")))
size_t compact_live_entries_avx512_impl(SplitLayer<T> &arr, T threshold, bool shrink_boards) {
    size_t out = 0U;
    size_t i = 0U;

    if constexpr (std::is_same_v<T, uint32_t>) {
        constexpr unsigned kLanes = 16U;
        const __m512i threshold_vec = _mm512_set1_epi32(static_cast<int>(threshold));
        for (; i + kLanes <= arr.size(); i += kLanes) {
            const __m512i values = _mm512_loadu_si512(reinterpret_cast<const void *>(arr.success.get() + i));
            const __mmask16 keep_mask = _mm512_cmp_epu32_mask(values, threshold_vec, _MM_CMPINT_GT);
            if (keep_mask == 0U) {
                continue;
            }
            if (keep_mask == 0xFFFFU && out == i) {
                out += kLanes;
                continue;
            }
            _mm512_mask_compressstoreu_epi32(arr.success.get() + out, keep_mask, values);
            const __m512i boards_lo = _mm512_loadu_si512(reinterpret_cast<const void *>(arr.boards.data() + i));
            const __m512i boards_hi = _mm512_loadu_si512(reinterpret_cast<const void *>(arr.boards.data() + i + 8U));
            const __mmask8 keep_lo = static_cast<__mmask8>(keep_mask & 0xFFU);
            const __mmask8 keep_hi = static_cast<__mmask8>((keep_mask >> 8U) & 0xFFU);
            _mm512_mask_compressstoreu_epi64(arr.boards.data() + out, keep_lo, boards_lo);
            const size_t out_hi = out + UniqueUtils::popcount_mask(static_cast<unsigned>(keep_lo));
            _mm512_mask_compressstoreu_epi64(arr.boards.data() + out_hi, keep_hi, boards_hi);
            out += UniqueUtils::popcount_mask(static_cast<unsigned>(keep_mask));
        }
    } else if constexpr (std::is_same_v<T, float>) {
        constexpr unsigned kLanes = 16U;
        const __m512 threshold_vec = _mm512_set1_ps(threshold);
        for (; i + kLanes <= arr.size(); i += kLanes) {
            const __m512 values = _mm512_loadu_ps(arr.success.get() + i);
            const __mmask16 keep_mask = _mm512_cmp_ps_mask(values, threshold_vec, _CMP_GT_OQ);
            if (keep_mask == 0U) {
                continue;
            }
            if (keep_mask == 0xFFFFU && out == i) {
                out += kLanes;
                continue;
            }
            _mm512_mask_compressstoreu_ps(arr.success.get() + out, keep_mask, values);
            const __m512i boards_lo = _mm512_loadu_si512(reinterpret_cast<const void *>(arr.boards.data() + i));
            const __m512i boards_hi = _mm512_loadu_si512(reinterpret_cast<const void *>(arr.boards.data() + i + 8U));
            const __mmask8 keep_lo = static_cast<__mmask8>(keep_mask & 0xFFU);
            const __mmask8 keep_hi = static_cast<__mmask8>((keep_mask >> 8U) & 0xFFU);
            _mm512_mask_compressstoreu_epi64(arr.boards.data() + out, keep_lo, boards_lo);
            const size_t out_hi = out + UniqueUtils::popcount_mask(static_cast<unsigned>(keep_lo));
            _mm512_mask_compressstoreu_epi64(arr.boards.data() + out_hi, keep_hi, boards_hi);
            out += UniqueUtils::popcount_mask(static_cast<unsigned>(keep_mask));
        }
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        constexpr unsigned kLanes = 8U;
        const __m512i threshold_vec = _mm512_set1_epi64(static_cast<long long>(threshold));
        for (; i + kLanes <= arr.size(); i += kLanes) {
            const __m512i values = _mm512_loadu_si512(reinterpret_cast<const void *>(arr.success.get() + i));
            const __mmask8 keep_mask = _mm512_cmp_epu64_mask(values, threshold_vec, _MM_CMPINT_GT);
            if (keep_mask == 0U) {
                continue;
            }
            if (keep_mask == 0xFFU && out == i) {
                out += kLanes;
                continue;
            }
            _mm512_mask_compressstoreu_epi64(arr.success.get() + out, keep_mask, values);
            const __m512i boards = _mm512_loadu_si512(reinterpret_cast<const void *>(arr.boards.data() + i));
            _mm512_mask_compressstoreu_epi64(arr.boards.data() + out, keep_mask, boards);
            out += UniqueUtils::popcount_mask(static_cast<unsigned>(keep_mask));
        }
    } else if constexpr (std::is_same_v<T, double>) {
        constexpr unsigned kLanes = 8U;
        const __m512d threshold_vec = _mm512_set1_pd(threshold);
        for (; i + kLanes <= arr.size(); i += kLanes) {
            const __m512d values = _mm512_loadu_pd(arr.success.get() + i);
            const __mmask8 keep_mask = _mm512_cmp_pd_mask(values, threshold_vec, _CMP_GT_OQ);
            if (keep_mask == 0U) {
                continue;
            }
            if (keep_mask == 0xFFU && out == i) {
                out += kLanes;
                continue;
            }
            _mm512_mask_compressstoreu_pd(arr.success.get() + out, keep_mask, values);
            const __m512i boards = _mm512_loadu_si512(reinterpret_cast<const void *>(arr.boards.data() + i));
            _mm512_mask_compressstoreu_epi64(arr.boards.data() + out, keep_mask, boards);
            out += UniqueUtils::popcount_mask(static_cast<unsigned>(keep_mask));
        }
    }

    for (; i < arr.size(); ++i) {
        if (arr.success[i] > threshold) {
            arr.boards[out] = arr.boards[i];
            arr.success[out] = arr.success[i];
            ++out;
        }
    }
    arr.length = out;
    if (shrink_boards) {
        arr.boards.resize(out);
    }
    return out;
}

template <typename T>
__attribute__((target("avx2")))
size_t compact_by_mask_avx2_impl(SplitLayer<T> &arr, const std::vector<uint8_t> &mask, bool shrink_boards) {
    constexpr unsigned kLanes = 8U;
    size_t out = 0U;
    size_t i = 0U;
    const size_t limit = std::min(arr.size(), mask.size());
    for (; i + kLanes <= limit; i += kLanes) {
        const uint32_t keep_mask = keep_mask_from_bytes_scalar(mask.data() + i, kLanes);
        if (keep_mask == 0U) {
            continue;
        }
        if (keep_mask == 0xFFU) {
            if (out != i) {
                std::memmove(arr.boards.data() + out, arr.boards.data() + i, kLanes * sizeof(uint64_t));
                std::memmove(arr.success.get() + out, arr.success.get() + i, kLanes * sizeof(T));
            }
            out += kLanes;
            continue;
        }
        copy_selected_runs_soa(arr, i, out, keep_mask, kLanes);
    }
    for (; i < limit; ++i) {
        if (mask[i] != 0U) {
            arr.boards[out] = arr.boards[i];
            arr.success[out] = arr.success[i];
            ++out;
        }
    }
    arr.length = out;
    if (shrink_boards) {
        arr.boards.resize(out);
    }
    return out;
}

template <typename T>
__attribute__((target("avx512f,avx512dq,avx512bw,avx512vl")))
size_t compact_by_mask_avx512_impl(SplitLayer<T> &arr, const std::vector<uint8_t> &mask, bool shrink_boards) {
    size_t out = 0U;
    size_t i = 0U;
    const size_t limit = std::min(arr.size(), mask.size());

    if constexpr (std::is_same_v<T, uint32_t>) {
        constexpr unsigned kLanes = 16U;
        for (; i + kLanes <= limit; i += kLanes) {
            const __mmask16 keep_mask = _mm_cmpneq_epi8_mask(
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(mask.data() + i)),
                _mm_setzero_si128()
            );
            if (keep_mask == 0U) {
                continue;
            }
            if (keep_mask == 0xFFFFU && out == i) {
                out += kLanes;
                continue;
            }
            const __m512i values = _mm512_loadu_si512(reinterpret_cast<const void *>(arr.success.get() + i));
            _mm512_mask_compressstoreu_epi32(arr.success.get() + out, keep_mask, values);
            const __m512i boards_lo = _mm512_loadu_si512(reinterpret_cast<const void *>(arr.boards.data() + i));
            const __m512i boards_hi = _mm512_loadu_si512(reinterpret_cast<const void *>(arr.boards.data() + i + 8U));
            const __mmask8 keep_lo = static_cast<__mmask8>(keep_mask & 0xFFU);
            const __mmask8 keep_hi = static_cast<__mmask8>((keep_mask >> 8U) & 0xFFU);
            _mm512_mask_compressstoreu_epi64(arr.boards.data() + out, keep_lo, boards_lo);
            const size_t out_hi = out + UniqueUtils::popcount_mask(static_cast<unsigned>(keep_lo));
            _mm512_mask_compressstoreu_epi64(arr.boards.data() + out_hi, keep_hi, boards_hi);
            out += UniqueUtils::popcount_mask(static_cast<unsigned>(keep_mask));
        }
    } else if constexpr (std::is_same_v<T, float>) {
        constexpr unsigned kLanes = 16U;
        for (; i + kLanes <= limit; i += kLanes) {
            const __mmask16 keep_mask = _mm_cmpneq_epi8_mask(
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(mask.data() + i)),
                _mm_setzero_si128()
            );
            if (keep_mask == 0U) {
                continue;
            }
            if (keep_mask == 0xFFFFU && out == i) {
                out += kLanes;
                continue;
            }
            const __m512 values = _mm512_loadu_ps(arr.success.get() + i);
            _mm512_mask_compressstoreu_ps(arr.success.get() + out, keep_mask, values);
            const __m512i boards_lo = _mm512_loadu_si512(reinterpret_cast<const void *>(arr.boards.data() + i));
            const __m512i boards_hi = _mm512_loadu_si512(reinterpret_cast<const void *>(arr.boards.data() + i + 8U));
            const __mmask8 keep_lo = static_cast<__mmask8>(keep_mask & 0xFFU);
            const __mmask8 keep_hi = static_cast<__mmask8>((keep_mask >> 8U) & 0xFFU);
            _mm512_mask_compressstoreu_epi64(arr.boards.data() + out, keep_lo, boards_lo);
            const size_t out_hi = out + UniqueUtils::popcount_mask(static_cast<unsigned>(keep_lo));
            _mm512_mask_compressstoreu_epi64(arr.boards.data() + out_hi, keep_hi, boards_hi);
            out += UniqueUtils::popcount_mask(static_cast<unsigned>(keep_mask));
        }
    } else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, double>) {
        constexpr unsigned kLanes = 8U;
        for (; i + kLanes <= limit; i += kLanes) {
            const uint32_t keep_mask32 = keep_mask_from_bytes_scalar(mask.data() + i, kLanes);
            const __mmask8 keep_mask = static_cast<__mmask8>(keep_mask32);
            if (keep_mask == 0U) {
                continue;
            }
            if (keep_mask == 0xFFU && out == i) {
                out += kLanes;
                continue;
            }
            const __m512i boards = _mm512_loadu_si512(reinterpret_cast<const void *>(arr.boards.data() + i));
            _mm512_mask_compressstoreu_epi64(arr.boards.data() + out, keep_mask, boards);
            if constexpr (std::is_same_v<T, uint64_t>) {
                const __m512i values = _mm512_loadu_si512(reinterpret_cast<const void *>(arr.success.get() + i));
                _mm512_mask_compressstoreu_epi64(arr.success.get() + out, keep_mask, values);
            } else {
                const __m512d values = _mm512_loadu_pd(arr.success.get() + i);
                _mm512_mask_compressstoreu_pd(arr.success.get() + out, keep_mask, values);
            }
            out += UniqueUtils::popcount_mask(static_cast<unsigned>(keep_mask));
        }
    }

    for (; i < limit; ++i) {
        if (mask[i] != 0U) {
            arr.boards[out] = arr.boards[i];
            arr.success[out] = arr.success[i];
            ++out;
        }
    }
    arr.length = out;
    if (shrink_boards) {
        arr.boards.resize(out);
    }
    return out;
}
#endif

template <typename T>
size_t compact_live_entries(SplitLayer<T> &arr, T threshold, bool shrink_boards) {
#if defined(__GNUC__) || defined(__clang__)
    using Fn = size_t (*)(SplitLayer<T> &, T, bool);
    static const Fn fn = []() -> Fn {
        if (UniqueUtils::cpu_has_avx512_dq_bw_vl()) {
            return &compact_live_entries_avx512_impl<T>;
        }
        if (UniqueUtils::cpu_has_avx2()) {
            return &compact_live_entries_avx2_impl<T>;
        }
        return &compact_live_entries_scalar_impl<T>;
    }();
    return fn(arr, threshold, shrink_boards);
#else
    return compact_live_entries_scalar_impl(arr, threshold, shrink_boards);
#endif
}

template <typename T>
size_t compact_by_mask(SplitLayer<T> &arr, const std::vector<uint8_t> &mask, bool shrink_boards) {
#if defined(__GNUC__) || defined(__clang__)
    using Fn = size_t (*)(SplitLayer<T> &, const std::vector<uint8_t> &, bool);
    static const Fn fn = []() -> Fn {
        if (UniqueUtils::cpu_has_avx512_dq_bw_vl()) {
            return &compact_by_mask_avx512_impl<T>;
        }
        if (UniqueUtils::cpu_has_avx2()) {
            return &compact_by_mask_avx2_impl<T>;
        }
        return &compact_by_mask_scalar_impl<T>;
    }();
    return fn(arr, mask, shrink_boards);
#else
    return compact_by_mask_scalar_impl(arr, mask, shrink_boards);
#endif
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
    T max_scale = max_scale_value_for_dtype<T>(options.success_rate_dtype);
    T zero_val = zero_value_for_dtype<T>(options.success_rate_dtype);
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
    const uint64_t book_alignment = sizeof(SuccessEntry<T>);
    const bool allow_opt_temp_archive =
        options.optimal_branch_only &&
        options.compress_temp_files &&
        i >= kOptimalBranchOnlyStartStep;

    remove_invalid_restart_file(path_i + ".book", book_alignment);
    remove_invalid_restart_file(path_i + ".z", 0U);
    remove_invalid_restart_file(path_i + ".book.7z", 0U);

    if (is_valid_restart_file(path_i + ".book", book_alignment)) {
        debug_log("skipping step " + std::to_string(i));
        if (options.compress && !options.optimal_branch_only) {
            maybe_do_compress_classic(options.pathname + std::to_string(i + 2) + ".book", options.success_rate_dtype);
        }
        return false;
    }
    if (is_valid_restart_file(path_i + ".z", 0U) ||
        (allow_opt_temp_archive && is_valid_restart_file(path_i + ".book.7z", 0U))) {
        debug_log("skipping step " + std::to_string(i));
        return false;
    }
    if (!started) {
        started = true;
        if (i != options.steps - 3 || d1.empty() || d2.empty()) {
            const FileIOUtils::DirectIoConfig io_config = FileIOUtils::direct_io_config_from_options(options);
            d1 = read_split_layer_file_or_archive<T>(options.pathname + std::to_string(i + 1) + ".book", io_config);
            d2 = read_split_layer_file_or_archive<T>(options.pathname + std::to_string(i + 2) + ".book", io_config);
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
    ensure_classic_solve_stats_header(options);
    ClassicSolveStatsRecord total_record;
    total_record.stage = "_total";
    total_record.deletion_threshold = options.deletion_threshold;
    bool started = false;
    AdaptiveIndex::Index ind1;
    T zero_val = zero_value_for_dtype<T>(options.success_rate_dtype);
    T max_scale = max_scale_value_for_dtype<T>(options.success_rate_dtype);
    bool has_index1 = false;
    const int index_threads = effective_num_threads(options);
    const uint32_t progress_total = classic_build_progress_total(options);
    const uint32_t solve_progress_total = build_progress_total(options);
    const FileIOUtils::DirectIoConfig io_config = FileIOUtils::direct_io_config_from_options(options);

    for (int i = options.steps - 3; i >= 0; --i) {
        FormationProgress::update_build_progress(
            solve_progress_total - static_cast<uint32_t>(i) - 2U,
            progress_total
        );
        if (!handle_restart_recalculate(i, d1, d2, started, options)) {
            continue;
        }

        const double read_t0 = wall_time_seconds();
        std::vector<uint64_t> raw_layer = read_raw_file(options.pathname + std::to_string(i), io_config);
        const double read_t1 = wall_time_seconds();
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

        double max_success = 0.0;
        const double max_scan_t0 = wall_time_seconds();
        if (!d0.empty()) {
            T max_success_raw = d0.success[0];
            for (size_t k = 1; k < d0.size(); ++k) {
                if (d0.success[k] > max_success_raw) {
                    max_success_raw = d0.success[k];
                }
            }
            max_success = static_cast<double>(max_success_raw - zero_val) / static_cast<double>(max_scale - zero_val);
        }
        const double max_scan_t1 = wall_time_seconds();

        const double current_write_t0 = wall_time_seconds();
        write_layer_file(options.pathname + std::to_string(i) + ".book", d0, io_config);
        const double current_write_t1 = wall_time_seconds();
        remove_file_if_exists(options.pathname + std::to_string(i) + ".book.7z");
        auto raw_path = options.pathname + std::to_string(i);
        remove_temp_raw_layer_files(raw_path);
        const int future_layer = i + 2;
        const std::string future_book_path = options.pathname + std::to_string(future_layer) + ".book";
        const std::string future_archive_path = future_book_path + ".7z";
        const bool use_opt_temp_archive =
            options.compress_temp_files &&
            options.optimal_branch_only &&
            future_layer >= kOptimalBranchOnlyStartStep;

        double future_compact_seconds = 0.0;
        double future_write_seconds = 0.0;
        double compress_seconds = 0.0;
        const uint64_t future2_live_before = static_cast<uint64_t>(d2.size());
        if (options.deletion_threshold > 0.0) {
            T threshold = static_cast<T>(options.deletion_threshold * static_cast<double>(max_scale - zero_val) + static_cast<double>(zero_val));
            const double future_compact_t0 = wall_time_seconds();
            compact_live_entries(d2, threshold, false);
            const double future_compact_t1 = wall_time_seconds();
            future_compact_seconds += future_compact_t1 - future_compact_t0;
            const double future_write_t0 = wall_time_seconds();
            if (use_opt_temp_archive) {
                write_split_layer_archive_file(future_archive_path, d2, 1);
            } else {
                write_layer_file(future_book_path, d2, io_config);
                remove_file_if_exists(future_archive_path);
            }
            const double future_write_t1 = wall_time_seconds();
            future_write_seconds += future_write_t1 - future_write_t0;
        }
        if (use_opt_temp_archive) {
            if (options.deletion_threshold <= 0.0) {
                const double future_write_t0 = wall_time_seconds();
                write_split_layer_archive_file(future_archive_path, d2, 1);
                future_write_seconds += wall_time_seconds() - future_write_t0;
            }
        } else if (options.compress && !options.optimal_branch_only) {
            const double compress_t0 = wall_time_seconds();
            maybe_do_compress_classic(future_book_path, options.success_rate_dtype);
            compress_seconds += wall_time_seconds() - compress_t0;
        }

        ClassicSolveStatsRecord record;
        record.step = i;
        record.input_live = static_cast<uint64_t>(length);
        record.post_zero_live = static_cast<uint64_t>(d0.size());
        record.future1_live = static_cast<uint64_t>(d1.size());
        record.future2_live = future2_live_before;
        record.future2_post_threshold_live = static_cast<uint64_t>(d2.size());
        record.deletion_threshold = options.deletion_threshold;
        record.max_success = max_success;
        record.current_read_seconds = read_t1 - read_t0;
        record.index_seconds = t1 - t0;
        record.recalculate_seconds = t2 - t1;
        record.zero_compact_seconds = t3 - t2;
        record.max_scan_seconds = max_scan_t1 - max_scan_t0;
        record.current_write_seconds = current_write_t1 - current_write_t0;
        record.future_compact_seconds = future_compact_seconds;
        record.future_write_seconds = future_write_seconds;
        record.compress_seconds = compress_seconds;
        append_classic_solve_stats_record(options, record);
        total_record.input_live += record.input_live;
        total_record.post_zero_live += record.post_zero_live;
        total_record.future1_live += record.future1_live;
        total_record.future2_live += record.future2_live;
        total_record.future2_post_threshold_live += record.future2_post_threshold_live;
        total_record.max_success = std::max(total_record.max_success, record.max_success);
        total_record.current_read_seconds += record.current_read_seconds;
        total_record.index_seconds += record.index_seconds;
        total_record.recalculate_seconds += record.recalculate_seconds;
        total_record.zero_compact_seconds += record.zero_compact_seconds;
        total_record.max_scan_seconds += record.max_scan_seconds;
        total_record.current_write_seconds += record.current_write_seconds;
        total_record.future_compact_seconds += record.future_compact_seconds;
        total_record.future_write_seconds += record.future_write_seconds;
        total_record.compress_seconds += record.compress_seconds;

        if (i > 0) {
            d2 = std::move(d1);
            d1 = std::move(d0);
        }
    }
    append_classic_solve_stats_record(options, total_record);
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
    if (i >= kOptimalBranchOnlyStartStep && d0.empty()) {
        const std::string d0_path = options.pathname + std::to_string(i - 2) + ".book";
        const std::string d1_path = options.pathname + std::to_string(i - 1) + ".book";
        if ((!fs::exists(d0_path) && !fs::exists(d0_path + ".7z")) ||
            (!fs::exists(d1_path) && !fs::exists(d1_path + ".7z"))) {
            return false;
        }
        const FileIOUtils::DirectIoConfig io_config = FileIOUtils::direct_io_config_from_options(options);
        d0 = read_split_layer_file_or_archive<T>(d0_path, io_config);
        d1 = read_split_layer_file_or_archive<T>(d1_path, io_config);
        if (!fs::exists(d0_path)) {
            write_layer_file(d0_path, d0, io_config);
            remove_file_if_exists(d0_path + ".7z");
        }
        if (!fs::exists(d1_path)) {
            write_layer_file(d1_path, d1, io_config);
            remove_file_if_exists(d1_path + ".7z");
        }
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
    T zero_val = zero_value_for_dtype<T>(options.success_rate_dtype);
    const uint32_t progress_total = classic_build_progress_total(options);
    const uint32_t solve_progress_total = build_progress_total(options);

    for (int i = 0; i < options.steps; ++i) {
        FormationProgress::update_build_progress(
            solve_progress_total + static_cast<uint32_t>(i) + 1U,
            progress_total
        );
        const bool process_step = handle_restart_opt_only(i, started, d0, d1, options);
        if (i >= kOptimalBranchOnlyStartStep && process_step) {
            const std::string d2_path = options.pathname + std::to_string(i) + ".book";
            SplitLayer<T> d2 = read_split_layer_file_or_archive<T>(
                d2_path,
                FileIOUtils::direct_io_config_from_options(options)
            );
            AdaptiveIndex::Index index = create_index(d2, effective_num_threads(options));
            std::vector<uint8_t> mask(d2.size(), 0);
            find_optimal_branches<T, Mover>(d0, d2, mask, spec, index.empty() ? nullptr : &index, 2, zero_val);
            find_optimal_branches<T, Mover>(d1, d2, mask, spec, index.empty() ? nullptr : &index, 1, zero_val);

            compact_by_mask(d2, mask, true);
            write_layer_file(d2_path, d2, FileIOUtils::direct_io_config_from_options(options));
            remove_file_if_exists(d2_path + ".7z");
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
