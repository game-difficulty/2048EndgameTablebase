#include "BookGenerator.h"

#include "BoardMover.h"
#include "Calculator.h"
#include "CanonicalBatch.h"
#include "CompressionBridge.h"
#include "FileIOUtils.h"
#include "Formation.h"
#include "NativeDiagnostics.h"
#include "UniqueUtils.h"
#include "VBoardMover.h"
#include <immintrin.h>

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
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

std::string classic_generate_stats_file_path(const RunOptions &options) {
    return options.pathname + "classic_generate_stats.csv";
}

std::string classic_generate_stats_header() {
    return "stage,step,input_live,arr1_raw,arr2_raw,arr1_unique,arr2_unique,output_live,length_factor,total_seconds,throughput_mbps,compute_seconds,compute_throughput_mbps,generate_seconds,sort_unique_seconds,merge_seconds,validate_seconds,write_seconds,time";
}

void ensure_classic_generate_stats_header(const RunOptions &options) {
    const std::string path = classic_generate_stats_file_path(options);
    if (NativePath::exists(path)) {
        std::ifstream in(NativePath::from_utf8(path));
        std::string first_line;
        if (std::getline(in, first_line) && first_line == classic_generate_stats_header()) {
            return;
        }
        in.close();
        std::error_code ec;
        NativePath::remove(path, ec);
    }
    std::ofstream file(NativePath::from_utf8(path), std::ios::app);
    file << classic_generate_stats_header() << "\n";
}

struct ClassicGenerateStatsRecord {
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

void append_classic_generate_stats_record(
    const RunOptions &options,
    const ClassicGenerateStatsRecord &record
) {
    ensure_classic_generate_stats_header(options);
    const double compute_seconds =
        record.generate_seconds + record.sort_unique_seconds + record.merge_seconds + record.validate_seconds;
    const double total_seconds = compute_seconds + record.write_seconds;
    std::ofstream file(NativePath::from_utf8(classic_generate_stats_file_path(options)), std::ios::app);
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

bool supports_avx512() {
    const char *disable = std::getenv("NATIVE_DISABLE_CLASSIC_GEN_AVX512");
    if (disable != nullptr && disable[0] != '\0' &&
        disable[0] != '0' && disable[0] != 'f' && disable[0] != 'F' &&
        disable[0] != 'n' && disable[0] != 'N') {
        return false;
    }
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
    (void)alignment;
    return StoragePaths::filename_exists(path);
}

void remove_invalid_restart_file(const std::string &path, uint64_t alignment) {
    std::error_code ec;
    if (!NativePath::exists(path, ec) || ec) {
        return;
    }
    if (is_valid_restart_file(path, alignment)) {
        return;
    }
    NativePath::remove(path, ec);
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

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512dq,avx512bw,avx512vl")))
#endif
inline void load_hash_chunk_avx512(
    const uint64_t *buf,
    int offset,
    __mmask8 active_mask,
    __m512i v_mask,
    __m512i &states,
    __m512i &idx
) {
    states = _mm512_maskz_loadu_epi64(active_mask, buf + offset);
    idx = _mm512_and_epi64(simd_hash(states), v_mask);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512dq,avx512bw,avx512vl")))
#endif
inline void prefetch_hash_slots_avx512(uint64_t *hmap, __m512i idx) {
#if defined(__GNUC__) || defined(__clang__)
    alignas(64) uint64_t idx_values[8];
    _mm512_store_si512(reinterpret_cast<__m512i *>(idx_values), idx);
    #pragma GCC unroll 8
    for (int lane = 0; lane < 8; ++lane) {
        __builtin_prefetch(hmap + idx_values[lane], 1, 0);
    }
#else
    (void)hmap;
    (void)idx;
#endif
}

struct ThreadWriteSink;

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512dq,avx512bw,avx512vl")))
#endif
void append_masked_to_sink_avx512(ThreadWriteSink &sink, __m512i states, __mmask8 mask, size_t accepted_count);

__attribute__((target("avx512f,avx512dq,avx512bw,avx512vl")))
inline void process_buf_avx512(
    const uint64_t *buf,
    size_t active_count,
    uint64_t *hmap,
    ThreadWriteSink &sink,
    uint64_t mask
) {
    constexpr int kLanesPerChunk = 8;
    constexpr int kPrefetchChunksAhead = 3;
    const int chunk_count = static_cast<int>((active_count + static_cast<size_t>(kLanesPerChunk) - 1U) / static_cast<size_t>(kLanesPerChunk));
    if (chunk_count == 0) {
        return;
    }

    const __m512i v_mask = _mm512_set1_epi64(static_cast<long long>(mask));

    alignas(64) __m512i state_queue[kPrefetchChunksAhead];
    alignas(64) __m512i idx_queue[kPrefetchChunksAhead];
    __mmask8 active_mask_queue[kPrefetchChunksAhead]{};

    for (int chunk = 0; chunk < std::min(chunk_count, kPrefetchChunksAhead); ++chunk) {
        const size_t offset = static_cast<size_t>(chunk * kLanesPerChunk);
        const size_t remaining = std::min(active_count - offset, static_cast<size_t>(kLanesPerChunk));
        active_mask_queue[chunk] = static_cast<__mmask8>((1u << static_cast<unsigned>(remaining)) - 1u);
        load_hash_chunk_avx512(buf, chunk * kLanesPerChunk, active_mask_queue[chunk], v_mask, state_queue[chunk], idx_queue[chunk]);
        prefetch_hash_slots_avx512(hmap, idx_queue[chunk]);
    }

    #pragma GCC unroll 16
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        const int slot = chunk % kPrefetchChunksAhead;
        const __m512i v_states = state_queue[slot];
        const __m512i v_idx = idx_queue[slot];
        const __mmask8 active_mask = active_mask_queue[slot];

        const int future_chunk = chunk + kPrefetchChunksAhead;
        if (future_chunk < chunk_count) {
            const size_t offset = static_cast<size_t>(future_chunk * kLanesPerChunk);
            const size_t remaining = std::min(active_count - offset, static_cast<size_t>(kLanesPerChunk));
            active_mask_queue[slot] = static_cast<__mmask8>((1u << static_cast<unsigned>(remaining)) - 1u);
            load_hash_chunk_avx512(
                buf,
                future_chunk * kLanesPerChunk,
                active_mask_queue[slot],
                v_mask,
                state_queue[slot],
                idx_queue[slot]
            );
            prefetch_hash_slots_avx512(hmap, idx_queue[slot]);
        }

        __m512i v_table = _mm512_mask_i64gather_epi64(_mm512_setzero_si512(), active_mask, v_idx, hmap, 8);
        __mmask8 matches = _mm512_mask_cmpeq_epi64_mask(active_mask, v_table, v_states);
        __mmask8 not_dup = active_mask & static_cast<__mmask8>(~matches);
        _mm512_mask_i64scatter_epi64(hmap, not_dup, v_idx, v_states, 8);
        const size_t accepted_count = static_cast<size_t>(__builtin_popcount(static_cast<unsigned int>(not_dup)));
        append_masked_to_sink_avx512(sink, v_states, not_dup, accepted_count);
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

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512dq,avx512bw,avx512vl")))
#endif
void append_masked_to_sink_avx512(ThreadWriteSink &sink, __m512i states, __mmask8 mask, size_t accepted_count) {
    if (accepted_count == 0) {
        return;
    }
    const size_t primary_remaining = sink.primary_end - sink.primary_pos;
    if (primary_remaining >= accepted_count) {
        _mm512_mask_compressstoreu_epi64(sink.primary + sink.primary_pos, mask, states);
        sink.primary_pos += accepted_count;
        return;
    }
    alignas(64) uint64_t compacted[8];
    _mm512_mask_compressstoreu_epi64(compacted, mask, states);
    sink.append_many(compacted, accepted_count);
}

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

inline void process_buf_scalar(
    const uint64_t *buf,
    size_t active_count,
    uint64_t *hmap,
    ThreadWriteSink &sink,
    uint64_t mask
) {
    std::array<uint64_t, kScalarBufferedBatchSize> hashed_idx{};
    for (size_t i = 0; i < active_count; ++i) {
        hashed_idx[i] = BookGeneratorUtils::hash_board(buf[i]) & mask;
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(hmap + hashed_idx[i], 1, 0);
#endif
    }

    for (size_t i = 0; i < active_count; ++i) {
        const size_t prefetch_distance = 8;
        if (i + prefetch_distance < active_count) {
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(hmap + hashed_idx[i + prefetch_distance], 1, 0);
#endif
        }
        const uint64_t board = buf[i];
        const size_t hashed = static_cast<size_t>(hashed_idx[i]);
        if (hmap[hashed] == board) {
            continue;
        }
        hmap[hashed] = board;
        sink.append_one(board);
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
        case SymmMode::Min34Top:
            return Calculator::canonical_min34_top(board);
        case SymmMode::Identity:
        default:
            return Calculator::canonical_identity(board);
    }
}

std::vector<std::vector<double>> load_length_factors(const std::string &path, double default_value) {
    std::vector<std::vector<double>> result;
    std::ifstream file(NativePath::from_utf8(path));
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
    fs::path base_path = NativePath::from_utf8(pathname);
    params.length_factors_list_path = NativePath::to_utf8_string(base_path.parent_path() / "length_factors_list.txt");
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
    std::ofstream out(NativePath::from_utf8(path), std::ios::trunc);
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
    bool is_big,
    size_t spill_arr1 = 0,
    size_t spill_arr2 = 0,
    bool exact_gather_arr1 = false,
    bool exact_gather_arr2 = false
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
    const bool used_spill = spill_arr1 > 0 || spill_arr2 > 0;
    if (!used_spill && length_needed > allocated_length) {
        throw std::out_of_range("The length multiplier is not big enough. Please restart.");
    }
    std::ostringstream oss;
    oss << "length " << len_d1t << ", " << len_d2
        << ", Using " << round_to_2(length_factor)
        << ", Need " << round_to_2(length_factor_actual);
    if (used_spill) {
        oss << ", Spill1 " << spill_arr1
            << (exact_gather_arr1 ? "(gather)" : "(tail)")
            << ", Spill2 " << spill_arr2
            << (exact_gather_arr2 ? "(gather)" : "(tail)");
    }
    debug_log(oss.str());
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
    auto read_temp_layer = [&options, &io_config](int step) {
        const std::string raw_path = StoragePaths::existing_path_for(options, step, "", false);
        if (!raw_path.empty()) {
            return FileIOUtils::read_binary_vector_direct<uint64_t>(raw_path, io_config);
        }
        const std::string archive_path = StoragePaths::existing_path_for(options, step, ".7z", false);
        if (!archive_path.empty()) {
            return read_temp_uint64_archive(archive_path);
        }
        return std::vector<uint64_t>{};
    };
    auto write_temp_layer = [&io_config, &options](int step, const std::vector<uint64_t> &data) {
        const uint64_t raw_bytes = static_cast<uint64_t>(data.size()) * static_cast<uint64_t>(sizeof(uint64_t));
        if (options.compress_temp_files) {
            auto lease = StoragePaths::reserve_write_path(
                options,
                StoragePaths::ArtifactRole::Hot,
                step,
                ".7z",
                StoragePaths::scale_ratio(raw_bytes, 25ULL)
            );
            if (!write_temp_uint64_archive(lease.path(), data, 1)) {
                throw std::runtime_error("failed to write compressed temp layer: " + lease.path());
            }
            std::error_code ec;
            (void)ec;
            StoragePaths::remove_all_candidates(options, step, "", false);
            lease.release();
            return;
        }
        FileIOUtils::write_routed_binary_vector_direct(
            options,
            StoragePaths::ArtifactRole::Hot,
            step,
            "",
            data,
            io_config,
            raw_bytes
        );
        StoragePaths::remove_all_candidates(options, step, ".7z", false);
    };

    remove_invalid_restart_file(path_i, raw_alignment);
    remove_invalid_restart_file(path_i_plus_1, raw_alignment);
    remove_invalid_restart_file(path_i + ".book", book_alignment);
    remove_invalid_restart_file(path_i_plus_1 + ".book", book_alignment);
    remove_invalid_restart_file(path_i + ".z", 0U);
    remove_invalid_restart_file(path_i + ".book.7z", 0U);
    remove_invalid_restart_file(path_i + ".7z", 0U);
    remove_invalid_restart_file(path_i_plus_1 + ".z", 0U);

    const bool current_compressed_exists =
        StoragePaths::filename_exists_any(options, step_index, ".z", true);
    const bool next_compressed_exists =
        StoragePaths::filename_exists_any(options, step_index + 1, ".z", true);
    const bool current_raw_exists =
        StoragePaths::filename_exists_any(options, step_index, "", false);
    const bool next_raw_exists =
        StoragePaths::filename_exists_any(options, step_index + 1, "", false);
    const bool current_book_exists =
        StoragePaths::filename_exists_any(options, step_index, ".book", false);
    const bool next_book_exists =
        StoragePaths::filename_exists_any(options, step_index + 1, ".book", false);
    const bool current_book_archive_exists =
        StoragePaths::filename_exists_any(options, step_index, ".book.7z", false);
    const bool current_raw_archive_exists =
        StoragePaths::filename_exists_any(options, step_index, ".7z", false);
    if ((next_raw_exists && current_raw_exists) ||
        (next_book_exists && current_raw_exists) ||
        (next_compressed_exists && current_raw_exists) ||
        current_book_exists ||
        current_compressed_exists ||
        current_book_archive_exists ||
        current_raw_archive_exists) {
        debug_log("skipping step " + std::to_string(step_index));
        return {false, {}, {}};
    }

    if (step_index == 1) {
        write_temp_layer(step_index - 1, arr_init);
        return {true, arr_init, {}};
    }

    if (!started) {
        return {true, read_temp_layer(step_index - 1), read_temp_layer(step_index)};
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
    std::vector<size_t> starts = build_segment_starts(capacity, num_threads);
    std::vector<ThreadWriteSink> sinks1 = create_write_sinks(arr1, starts, capacity);
    std::vector<ThreadWriteSink> sinks2 = create_write_sinks(arr2, starts, capacity);

    size_t chunk_size = std::min<size_t>(1000000ULL, arr0_size / (static_cast<size_t>(num_threads) * 5ULL) + 1ULL) * static_cast<size_t>(num_threads);
    size_t chunks_count = (arr0_size + chunk_size - 1) / chunk_size;

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_index = omp_get_thread_num();
        ThreadWriteSink &sink1 = sinks1[static_cast<size_t>(thread_index)];
        ThreadWriteSink &sink2 = sinks2[static_cast<size_t>(thread_index)];
        std::array<uint64_t, kScalarBufferedBatchSize> buffer1{};
        std::array<uint64_t, kScalarBufferedBatchSize> buffer2{};
        size_t b1_ptr = 0;
        size_t b2_ptr = 0;

        auto flush_buf1 = [&]() {
            if (b1_ptr == 0) {
                return;
            }
            CanonicalBatch::canonicalize_inplace(buffer1.data(), b1_ptr, spec.symm_mode);
            process_buf_scalar(buffer1.data(), b1_ptr, hashmap1, sink1, hashmask1);
            b1_ptr = 0;
        };

        auto flush_buf2 = [&]() {
            if (b2_ptr == 0) {
                return;
            }
            CanonicalBatch::canonicalize_inplace(buffer2.data(), b2_ptr, spec.symm_mode);
            process_buf_scalar(buffer2.data(), b2_ptr, hashmap2, sink2, hashmask2);
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
                            buffer1[b1_ptr++] = new_board;
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
                            buffer2[b2_ptr++] = new_board;
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
    }

    GenBoardsResult result;
    FinalizedGeneratedArray finalized_arr1 = finalize_generated_array(arr1, capacity, starts, sinks1);
    FinalizedGeneratedArray finalized_arr2 = finalize_generated_array(arr2, capacity, starts, sinks2);
    result.total_arr1 = finalized_arr1.total;
    result.total_arr2 = finalized_arr2.total;
    result.counts1 = collect_total_counts(sinks1);
    result.counts2 = collect_total_counts(sinks2);
    result.finalized_arr1 = std::move(finalized_arr1.owned);
    result.finalized_arr2 = std::move(finalized_arr2.owned);
    result.spill_arr1 = finalized_arr1.spill_total;
    result.spill_arr2 = finalized_arr2.spill_total;
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
    std::vector<size_t> starts = build_segment_starts(capacity, num_threads);
    std::vector<ThreadWriteSink> sinks1 = create_write_sinks(arr1, starts, capacity);
    std::vector<ThreadWriteSink> sinks2 = create_write_sinks(arr2, starts, capacity);

    const int BATCH_SIZE = 128;

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_index = omp_get_thread_num();
        ThreadWriteSink &sink1 = sinks1[static_cast<size_t>(thread_index)];
        ThreadWriteSink &sink2 = sinks2[static_cast<size_t>(thread_index)];

        uint64_t buffer1[BATCH_SIZE], buffer2[BATCH_SIZE];
        int b1_ptr = 0, b2_ptr = 0;

        auto process_buf = [&](uint64_t *buf, int &ptr, uint64_t *hmap, ThreadWriteSink &sink, uint64_t mask) {
            CanonicalBatch::canonicalize_inplace(buf, static_cast<size_t>(ptr), spec.symm_mode);
            if (ptr == BATCH_SIZE) {
                process_buf_avx512(buf, static_cast<size_t>(ptr), hmap, sink, mask);
            } else {
                process_buf_scalar(buf, static_cast<size_t>(ptr), hmap, sink, mask);
            }
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
                        buffer1[b1_ptr++] = new_board;
                        if (b1_ptr == BATCH_SIZE) process_buf(buffer1, b1_ptr, hashmap1, sink1, hashmask1);
                    }
                }

                uint64_t spawn4 = board | (2ULL << (4 * i));
                auto moves4 = Mover::move_all_dir(spawn4);
                uint64_t boards4[4] = {std::get<0>(moves4), std::get<1>(moves4), std::get<2>(moves4), std::get<3>(moves4)};
                for (uint64_t new_board : boards4) {
                    if (new_board != spawn4 && is_pattern(new_board, spec.pattern_masks)) {
                        buffer2[b2_ptr++] = new_board;
                        if (b2_ptr == BATCH_SIZE) process_buf(buffer2, b2_ptr, hashmap2, sink2, hashmask2);
                    }
                }
            }
        }
        
        // 尾部清理
        if (b1_ptr > 0) {
            process_buf(buffer1, b1_ptr, hashmap1, sink1, hashmask1);
        }
        if (b2_ptr > 0) {
            process_buf(buffer2, b2_ptr, hashmap2, sink2, hashmask2);
        }

    }

    GenBoardsResult result;
    FinalizedGeneratedArray finalized_arr1 = finalize_generated_array(arr1, capacity, starts, sinks1);
    FinalizedGeneratedArray finalized_arr2 = finalize_generated_array(arr2, capacity, starts, sinks2);
    result.total_arr1 = finalized_arr1.total;
    result.total_arr2 = finalized_arr2.total;
    result.counts1 = collect_total_counts(sinks1);
    result.counts2 = collect_total_counts(sinks2);
    result.finalized_arr1 = std::move(finalized_arr1.owned);
    result.finalized_arr2 = std::move(finalized_arr2.owned);
    result.spill_arr1 = finalized_arr1.spill_total;
    result.spill_arr2 = finalized_arr2.spill_total;
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
        length_factor *= len > static_cast<size_t>(1e8) ? 1.15 : 1.2;
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
            true,
            segment_result.spill_arr1,
            segment_result.spill_arr2,
            segment_result.finalized_arr1 != nullptr,
            segment_result.finalized_arr2 != nullptr
        );
        std::copy(segment_result.counts1.begin(), segment_result.counts1.end(), actual_lengths1.begin() + static_cast<std::ptrdiff_t>(seg_index * static_cast<size_t>(num_threads)));
        std::copy(segment_result.counts2.begin(), segment_result.counts2.end(), actual_lengths2.begin() + static_cast<std::ptrdiff_t>(seg_index * static_cast<size_t>(num_threads)));

        if (!length_factors_list[seg_index].empty()) {
            length_factors_list[seg_index].erase(length_factors_list[seg_index].begin());
        }
        length_factors_list[seg_index].push_back(static_cast<double>(segment_result.total_arr2) / static_cast<double>(1 + len));
        gen_time += wall_time_seconds() - seg_t0;

        uint64_t *segment_arr1 = segment_result.finalized_arr1 ? segment_result.finalized_arr1.get() : arr1_ptr.get();
        uint64_t *segment_arr2 = segment_result.finalized_arr2 ? segment_result.finalized_arr2.get() : arr2_ptr.get();
        auto [unique_arr1_length, unique_arr2_length] = BookGeneratorUtils::sort_and_unique_two_arrays_concurrently(
            segment_arr1,
            segment_result.total_arr1,
            segment_arr2,
            segment_result.total_arr2,
            num_threads
        );
        big_result.arr1s.emplace_back(
            segment_arr1,
            segment_arr1 + unique_arr1_length
        );
        big_result.arr2s.emplace_back(
            segment_arr2,
            segment_arr2 + unique_arr2_length
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
    ensure_classic_generate_stats_header(options);
    ClassicGenerateStatsRecord total_record;
    total_record.stage = "_total";

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
        NativeDiagnostics::mark(
            "Classic.generate step begin step=" + std::to_string(i) +
            " live=" + std::to_string(d0.size())
        );
        FormationProgress::update_build_progress(static_cast<uint32_t>(i), progress_total);
        bool do_check = i > options.docheck_step;
        ClassicGenerateStatsRecord stats_record;
        bool has_stats_record = false;
        stats_record.step = i;
        stats_record.input_live = static_cast<uint64_t>(d0.size());

        if (d0.size() < init_params.segment_size) {
            double t0 = wall_time_seconds();
            double t1 = t0;
            double t2 = t0;
            std::vector<uint64_t> d1t;
            std::vector<uint64_t> d2;
            std::vector<size_t> generation_counts2;
            std::unique_ptr<uint64_t[]> arr1_ptr;
            std::unique_ptr<uint64_t[]> arr2_ptr;
            uint64_t *sort_arr1 = nullptr;
            uint64_t *sort_arr2 = nullptr;
            size_t sort_len1 = 0;
            size_t sort_len2 = 0;
            GenBoardsResult generated_result;
            bool use_simple_path = should_use_simple_path(d0, arr_init, init_params.length_factors);
            stats_record.stage = use_simple_path
                ? "simple"
                : std::string("normal-") + (supports_avx512() ? "avx512" : "buffered-scalar");
            NativeDiagnostics::mark(
                "Classic.generate path step=" + std::to_string(i) +
                " stage=" + stats_record.stage
            );
            if (use_simple_path) {
                debug_log("step " + std::to_string(i) + " path: simple");
                std::tie(d1t, d2) = options.is_variant
                    ? gen_boards_simple<VBoardMover>(d0.data(), d0.size(), options.target, spec, do_check, options.is_free)
                    : gen_boards_simple<BoardMover>(d0.data(), d0.size(), options.target, spec, do_check, options.is_free);
                sort_arr1 = d1t.data();
                sort_len1 = d1t.size();
                sort_arr2 = d2.data();
                sort_len2 = d2.size();
                t1 = wall_time_seconds();
            } else {
                debug_log("step " + std::to_string(i) + " path: " + stats_record.stage);
                double length_factor = predict_next_length_factor_quadratic(init_params.length_factors);
                length_factor *= d0.size() > static_cast<size_t>(1e8) ? 1.15 : 1.2;
                length_factor *= init_params.length_factor_multiplier;
                if (std::isnan(length_factor)) {
                    length_factor = 3.0;
                }
                stats_record.length_factor = length_factor;
                if (hashmap1.empty()) {
                    update_hashmap_length(hashmap1, d0.size());
                    update_hashmap_length(hashmap2, d0.size());
                }

                size_t min_length = options.is_free ? 9999999ULL : 6999999ULL;
                size_t capacity = std::max(min_length, static_cast<size_t>(static_cast<double>(d0.size()) * length_factor));
                NativeDiagnostics::mark(
                    "Classic.generate allocate normal step=" + std::to_string(i) +
                    " capacity=" + std::to_string(capacity)
                );
                arr1_ptr = std::make_unique<uint64_t[]>(capacity);
                arr2_ptr = std::make_unique<uint64_t[]>(capacity);
                NativeDiagnostics::mark("Classic.generate gen_boards begin step=" + std::to_string(i));
                generated_result = options.is_variant
                    ? gen_boards<VBoardMover>(d0.data(), d0.size(), options.target, spec, hashmap1.data(), hashmap1.size(), hashmap2.data(), hashmap2.size(), arr1_ptr.get(), arr2_ptr.get(), capacity, num_threads, do_check, options.is_free)
                    : gen_boards<BoardMover>(d0.data(), d0.size(), options.target, spec, hashmap1.data(), hashmap1.size(), hashmap2.data(), hashmap2.size(), arr1_ptr.get(), arr2_ptr.get(), capacity, num_threads, do_check, options.is_free);
                NativeDiagnostics::mark("Classic.generate gen_boards done step=" + std::to_string(i));
                t1 = wall_time_seconds();

                validate_length_and_balance(
                    d0.size(),
                    generated_result.total_arr2,
                    generated_result.total_arr1,
                    generated_result.counts1,
                    generated_result.counts2,
                    length_factor,
                    false,
                    generated_result.spill_arr1,
                    generated_result.spill_arr2,
                    generated_result.finalized_arr1 != nullptr,
                    generated_result.finalized_arr2 != nullptr
                );
                sort_arr1 = generated_result.finalized_arr1 ? generated_result.finalized_arr1.get() : arr1_ptr.get();
                sort_len1 = generated_result.total_arr1;
                sort_arr2 = generated_result.finalized_arr2 ? generated_result.finalized_arr2.get() : arr2_ptr.get();
                sort_len2 = generated_result.total_arr2;
                generation_counts2 = generated_result.counts2;
            }

            std::tie(init_params.length_factors, init_params.length_factors_list) =
                update_parameters(d0.size(), sort_len2, init_params.length_factors, init_params.length_factors_list_path);
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

            NativeDiagnostics::mark("Classic.generate sort_unique begin step=" + std::to_string(i));
            auto [unique_d1t_length, unique_d2_length] = BookGeneratorUtils::sort_and_unique_two_arrays_concurrently(
                sort_arr1, sort_len1,
                sort_arr2, sort_len2,
                num_threads
            );
            NativeDiagnostics::mark("Classic.generate sort_unique done step=" + std::to_string(i));
            stats_record.arr1_raw = static_cast<uint64_t>(sort_len1);
            stats_record.arr2_raw = static_cast<uint64_t>(sort_len2);
            stats_record.arr1_unique = static_cast<uint64_t>(unique_d1t_length);
            stats_record.arr2_unique = static_cast<uint64_t>(unique_d2_length);
            if (use_simple_path) {
                d1t.resize(unique_d1t_length);
                d2.resize(unique_d2_length);
            } else {
                d1t.assign(sort_arr1, sort_arr1 + unique_d1t_length);
                d2.assign(sort_arr2, sort_arr2 + unique_d2_length);
            }
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
            NativeDiagnostics::mark("Classic.generate merge begin step=" + std::to_string(i));
            d1 = BookGeneratorUtils::concatenate(BookGeneratorUtils::merge_deduplicate_all(d1_inputs, pivots, num_threads));
            NativeDiagnostics::mark("Classic.generate merge done step=" + std::to_string(i));
            double t3 = wall_time_seconds();
            stats_record.output_live = static_cast<uint64_t>(d1.size());
            stats_record.generate_seconds = t1 - t0;
            stats_record.sort_unique_seconds = t2 - t1;
            stats_record.merge_seconds = t3 - t2;
            has_stats_record = true;
            d0 = std::move(d1);
            d1 = std::move(d2);
            if (!hashmap1.empty()) {
                update_hashmap_length(hashmap1, d1.size());
                update_hashmap_length(hashmap2, d1.size());
            }
        } else {
            NativeDiagnostics::mark(
                "Classic.generate path step=" + std::to_string(i) +
                " stage=big live=" + std::to_string(d0.size())
            );
            debug_log("step " + std::to_string(i) + " path: big");
            stats_record.stage = "big";
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
            uint64_t big_arr1_raw = 0U;
            uint64_t big_arr2_raw = 0U;
            for (const auto &part : big_result.arr1s) {
                big_arr1_raw += static_cast<uint64_t>(part.size());
            }
            for (const auto &part : big_result.arr2s) {
                big_arr2_raw += static_cast<uint64_t>(part.size());
            }
            big_result.arr1s.push_back(std::move(d1));
            d1 = BookGeneratorUtils::concatenate(BookGeneratorUtils::merge_deduplicate_all(big_result.arr2s, pivots, num_threads));
            d0 = BookGeneratorUtils::concatenate(BookGeneratorUtils::merge_deduplicate_all(big_result.arr1s, pivots, num_threads));
            double t3 = wall_time_seconds();
            stats_record.arr1_raw = big_arr1_raw;
            stats_record.arr2_raw = big_arr2_raw;
            stats_record.arr1_unique = static_cast<uint64_t>(d0.size());
            stats_record.arr2_unique = static_cast<uint64_t>(d1.size());
            stats_record.output_live = static_cast<uint64_t>(d0.size());
            stats_record.generate_seconds = big_result.t1 - big_result.t0;
            stats_record.sort_unique_seconds = big_result.t2 - big_result.t1;
            stats_record.merge_seconds = t3 - big_result.t2;
            has_stats_record = true;
            init_params.length_factors = harmonic_mean_by_column(init_params.length_factors_list);
            save_length_factors(init_params.length_factors_list_path, init_params.length_factors_list);
        }

        const double validate_t0 = wall_time_seconds();
        NativeDiagnostics::mark("Classic.generate write begin step=" + std::to_string(i));
        const uint64_t raw_bytes = static_cast<uint64_t>(d0.size()) * static_cast<uint64_t>(sizeof(uint64_t));
        if (options.compress_temp_files) {
            stats_record.validate_seconds += wall_time_seconds() - validate_t0;
            const double write_t0 = wall_time_seconds();
            auto lease = StoragePaths::reserve_write_path(
                options,
                StoragePaths::ArtifactRole::Hot,
                i,
                ".7z",
                StoragePaths::scale_ratio(raw_bytes, 25ULL)
            );
            if (!write_temp_uint64_archive(lease.path(), d0, 1)) {
                throw std::runtime_error("failed to write compressed temp layer: " + lease.path());
            }
            std::error_code ec;
            (void)ec;
            StoragePaths::remove_all_candidates(options, i, "", false);
            lease.release();
            stats_record.write_seconds = wall_time_seconds() - write_t0;
        } else {
            stats_record.validate_seconds += wall_time_seconds() - validate_t0;
            const double write_t0 = wall_time_seconds();
            FileIOUtils::write_routed_binary_vector_direct(
                options,
                StoragePaths::ArtifactRole::Hot,
                i,
                "",
                d0,
                io_config,
                raw_bytes
            );
            StoragePaths::remove_all_candidates(options, i, ".7z", false);
            stats_record.write_seconds = wall_time_seconds() - write_t0;
        }
        NativeDiagnostics::mark("Classic.generate write done step=" + std::to_string(i));
        if (has_stats_record) {
            append_classic_generate_stats_record(options, stats_record);
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
    append_classic_generate_stats_record(options, total_record);

    return {started, std::move(d0), std::move(d1)};
}

} // namespace BookGenerator
