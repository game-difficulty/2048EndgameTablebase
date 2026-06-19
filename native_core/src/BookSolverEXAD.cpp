#include "BookSolver.h"

#include "CanonicalBatch.h"
#include "EXADCompressedResult.h"
#include "EXADSolvedLayer.h"
#include "HybridSearch.h"
#include "PathUtils.h"

// EXAD recalculate still reuses AD's semantic helper routines (derive ranking,
// mask-new-tile dispatch, permutation matching, and workspaces).  The storage
// and lookup path below is EXAD-native and does not call AD's matrix solver.
#define run_pattern_solve_ad_cpp run_pattern_solve_ad_cpp_for_exad_core
#include "BookSolverAD.cpp"
#undef run_pattern_solve_ad_cpp

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {

namespace fs = std::filesystem;

template <typename T>
inline void prefetch_success_row(const T *row) {
#if defined(__GNUC__) || defined(__clang__)
    if (row != nullptr) {
        __builtin_prefetch(row, 0, 1);
    }
#else
    (void)row;
#endif
}

template <typename T>
uint64_t solved_layer_bitmap_bits(const EXAD::SolvedLayer<T> &layer) {
    uint64_t bits = 0;
    for (const EXAD::BoardSet &set : layer.sets) {
        bits += set.aligned_bitmap_bits;
    }
    return bits;
}

constexpr uint32_t kEXADScalarBatchSize = 256U;
constexpr size_t kEXADScalarBestSlots = static_cast<size_t>(kEXADScalarBatchSize) * 16U;
constexpr size_t kEXADScalarQueryReserve = static_cast<size_t>(kEXADScalarBatchSize) * 16U * 4U;
constexpr uint32_t kEXADVectorMaxBatchSize = 256U;
constexpr uint32_t kEXADVectorMaxWidth = 128U;
constexpr size_t kEXADVectorTargetBestSlots = kEXADScalarBestSlots * 8U;
constexpr size_t kEXADMaxMatchCacheCells = 64ULL * 1024ULL * 1024ULL;
static_assert(kEXADVectorMaxWidth <= std::numeric_limits<uint16_t>::max());
static_assert(kEXADVectorMaxBatchSize * 16U - 1U <= std::numeric_limits<uint16_t>::max());

inline bool exad_vector_batch_enabled(size_t width) {
    // Very large AD vector rows are valid but intentionally stay on the scalar
    // vector path; the batched query metadata stores width/ref compactly.
    return width > 1U &&
        width <= kEXADVectorMaxWidth &&
        width <= std::numeric_limits<uint16_t>::max();
}

inline uint32_t exad_vector_batch_size(size_t width) {
    const size_t denom = std::max<size_t>(1U, 16U * width);
    const size_t by_width = std::max<size_t>(1U, kEXADVectorTargetBestSlots / denom);
    return static_cast<uint32_t>(std::min<size_t>(kEXADVectorMaxBatchSize, by_width));
}

inline size_t exad_match_cache_map_length(size_t derive_size) {
    if (derive_size == 0U) {
        return 0U;
    }
    constexpr std::array<size_t, 8> kPrimeLengths = {
        33331U, 11113U, 4099U, 2053U, 1021U, 509U, 251U, 127U
    };
    const size_t max_length = std::max<size_t>(1U, kEXADMaxMatchCacheCells / derive_size);
    for (size_t length : kPrimeLengths) {
        if (length <= max_length) {
            return length;
        }
    }
    return 0U;
}

inline bool exad_match_cache_enabled(size_t derive_size) {
    if (derive_size <= 1U || derive_size > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    const size_t map_length = exad_match_cache_map_length(derive_size);
    return map_length != 0U &&
        map_length <= kEXADMaxMatchCacheCells / derive_size;
}

inline bool match_cache_ready(const MatchCache &cache, size_t derive_size) {
    return cache.map_length != 0U && cache.derive_size >= derive_size;
}

template <typename T>
MatchCache &get_exad_match_cache(
    std::unordered_map<uint32_t, MatchCache> &match_dict,
    size_t derive_size
) {
    const uint32_t key = static_cast<uint32_t>(derive_size);
    auto it = match_dict.find(key);
    if (it == match_dict.end()) {
        const size_t map_length = exad_match_cache_map_length(derive_size);
        if (map_length == 0U) {
            throw std::runtime_error("EXAD match cache requested for unsupported row width");
        }
        it = match_dict.emplace(key, MatchCache(map_length, derive_size)).first;
    }
    return it->second;
}

template <typename T>
struct EXADScalarBatchWorkspace {
    std::vector<EXAD::PreparedQuery> queries1;
    std::vector<EXAD::PreparedQuery> queries2;
    std::vector<uint64_t> canonical_candidates1;
    std::vector<uint64_t> canonical_candidates2;
    std::vector<uint16_t> candidate_refs1;
    std::vector<uint16_t> candidate_refs2;
    std::array<uint16_t, kEXADScalarBatchSize> empty_masks{};
    std::array<T, kEXADScalarBestSlots> best2{};
    std::array<T, kEXADScalarBestSlots> best4{};

    EXADScalarBatchWorkspace() {
        queries1.reserve(kEXADScalarQueryReserve);
        queries2.reserve(kEXADScalarQueryReserve);
        canonical_candidates1.reserve(kEXADScalarQueryReserve);
        canonical_candidates2.reserve(kEXADScalarQueryReserve);
        candidate_refs1.reserve(kEXADScalarQueryReserve);
        candidate_refs2.reserve(kEXADScalarQueryReserve);
    }
};

struct EXADVectorQuery {
    EXAD::PreparedQuery query{};
    uint32_t column_offset = 0;
};

template <typename T>
struct EXADVectorBatchWorkspace {
    std::vector<EXADVectorQuery> queries1;
    std::vector<EXADVectorQuery> queries2;
    std::vector<uint32_t> columns1;
    std::vector<uint32_t> columns2;
    std::vector<uint16_t> empty_masks;
    std::vector<T> best2;
    std::vector<T> best4;
    std::vector<double> success_probability;

    void prepare(uint32_t width, uint32_t board_count, T zero_val) {
        queries1.clear();
        queries2.clear();
        columns1.clear();
        columns2.clear();
        const size_t query_reserve = static_cast<size_t>(board_count) * 16U * 4U;
        const size_t best_slots = static_cast<size_t>(board_count) * 16U * width;
        if (queries1.capacity() < query_reserve) {
            queries1.reserve(query_reserve);
            queries2.reserve(query_reserve);
        }
        const size_t column_reserve = query_reserve * width;
        if (columns1.capacity() < column_reserve) {
            columns1.reserve(column_reserve);
            columns2.reserve(column_reserve);
        }
        empty_masks.resize(board_count);
        best2.resize(best_slots);
        best4.resize(best_slots);
        std::fill(best2.begin(), best2.end(), zero_val);
        std::fill(best4.begin(), best4.end(), zero_val);
        success_probability.resize(width);
    }
};

std::string exad_solve_stats_file_path(const RunOptions &options) {
    return options.pathname + "exad_solve_stats.csv";
}

std::string exad_solve_stats_header() {
    return "stage,layout,direct_index_type,step,input_rows,input_values,post_zero_rows,post_zero_values,"
           "deletion_threshold,relative_deletion_threshold,effective_deletion_threshold,"
           "current_retained_ratio,future_threshold_values_before,"
           "future_threshold_values_after,future_threshold_retained_ratio,max_success,total_seconds,"
           "throughput_mbps,compute_seconds,compute_throughput_mbps,"
           "current_read_seconds,current_build_seconds,future_read_seconds,future_index_seconds,recalculate_seconds,"
           "zero_compact_seconds,current_write_seconds,future_compact_seconds,future_write_seconds,metadata_bytes,"
           "success_bytes,bitmap_density,compress_seconds,time";
}

void ensure_exad_solve_stats_header(const RunOptions &options) {
    const std::string path = exad_solve_stats_file_path(options);
    if (NativePath::exists(path)) {
        std::ifstream in(NativePath::from_utf8(path));
        std::string first_line;
        if (std::getline(in, first_line) && first_line == exad_solve_stats_header()) {
            return;
        }
        in.close();
        std::error_code ec;
        NativePath::remove(path, ec);
    }
    std::ofstream file(NativePath::from_utf8(path), std::ios::app);
    file << exad_solve_stats_header() << "\n";
}

struct EXADSolveStatsRecord {
    std::string stage = "solve";
    int step = -1;
    uint64_t input_rows = 0;
    uint64_t input_values = 0;
    uint64_t post_zero_rows = 0;
    uint64_t post_zero_values = 0;
    double deletion_threshold = 0.0;
    double relative_deletion_threshold = 0.0;
    double effective_deletion_threshold = 0.0;
    double current_retained_ratio = 0.0;
    uint64_t future_threshold_values_before = 0;
    uint64_t future_threshold_values_after = 0;
    double future_threshold_retained_ratio = 0.0;
    double max_success = 0.0;
    double current_read_seconds = 0.0;
    double current_build_seconds = 0.0;
    double future_read_seconds = 0.0;
    double future_index_seconds = 0.0;
    double recalculate_seconds = 0.0;
    double zero_compact_seconds = 0.0;
    double current_write_seconds = 0.0;
    double future_compact_seconds = 0.0;
    double future_write_seconds = 0.0;
    double compress_seconds = 0.0;
    uint64_t metadata_bytes = 0;
    uint64_t success_bytes = 0;
    double bitmap_density = 0.0;
};

void append_exad_solve_stats_record(
    const RunOptions &options,
    const EXADSolveStatsRecord &record
) {
    ensure_exad_solve_stats_header(options);
    const double compute_seconds =
        record.current_build_seconds + record.future_index_seconds + record.recalculate_seconds
        + record.zero_compact_seconds + record.future_compact_seconds;
    const double total_seconds =
        record.current_read_seconds + record.future_read_seconds + compute_seconds
        + record.current_write_seconds + record.future_write_seconds + record.compress_seconds;
    std::ofstream file(NativePath::from_utf8(exad_solve_stats_file_path(options)), std::ios::app);
    file << record.stage << ","
         << "exad_prefix36_suffix28_solved,per_ad_bucket_entry,"
         << record.step << ","
         << record.input_rows << ","
         << record.input_values << ","
         << record.post_zero_rows << ","
         << record.post_zero_values << ","
         << std::fixed << std::setprecision(6)
         << record.deletion_threshold << ","
         << record.relative_deletion_threshold << ","
         << record.effective_deletion_threshold << ","
         << record.current_retained_ratio << ","
         << record.future_threshold_values_before << ","
         << record.future_threshold_values_after << ","
         << record.future_threshold_retained_ratio << ","
         << record.max_success << ","
         << total_seconds << ","
         << throughput_mbps_for(record.input_values, total_seconds) << ","
         << compute_seconds << ","
         << throughput_mbps_for(record.input_values, compute_seconds) << ","
         << record.current_read_seconds << ","
         << record.current_build_seconds << ","
         << record.future_read_seconds << ","
         << record.future_index_seconds << ","
         << record.recalculate_seconds << ","
         << record.zero_compact_seconds << ","
         << record.current_write_seconds << ","
         << record.future_compact_seconds << ","
         << record.future_write_seconds << ","
         << record.metadata_bytes << ","
         << record.success_bytes << ","
         << record.bitmap_density << ","
         << record.compress_seconds << ","
         << now_string() << "\n";
}

constexpr char kEXADSlotChunkMagic[8] = {'E', 'X', 'A', 'D', '7', 'S', 'L', 'C'};
constexpr uint32_t kEXADSlotChunkVersion = 2U;

struct EXADSlotChunkHeader {
    char magic[8];
    uint32_t version = kEXADSlotChunkVersion;
    uint32_t dtype_mode = 0;
    uint32_t value_size = 0;
    uint32_t slot = 0;
    uint32_t original_board_sum = 0;
    uint32_t threshold_bits = 0;
    uint32_t row_width = 0;
    uint8_t physical_transform = 0;
    uint8_t inverse_physical_transform = 0;
    uint16_t reserved_transform = 0;
    uint64_t lut_signature = 0;
    uint64_t logical_pattern_signature = 0;
    uint64_t physical_pattern_signature = 0;
    uint64_t live_board_count = 0;
    uint64_t exact_bitmap_bits = 0;
    uint64_t aligned_bitmap_bits = 0;
    uint64_t bucket_count = 0;
    uint64_t small_bitmap_bytes = 0;
    uint64_t large_bitmap_words = 0;
    uint64_t success_value_count = 0;
};

struct EXADSlotChunkManifest {
    std::string path;
    uint32_t slot = 0;
    uint32_t original_board_sum = 0;
    uint32_t threshold_bits = 0;
    uint32_t row_width = 0;
    uint8_t physical_transform = 0;
    uint8_t inverse_physical_transform = 0;
    uint64_t lut_signature = 0;
    uint64_t logical_pattern_signature = 0;
    uint64_t physical_pattern_signature = 0;
    uint64_t live_board_count = 0;
    uint64_t exact_bitmap_bits = 0;
    uint64_t aligned_bitmap_bits = 0;
    uint64_t bucket_count = 0;
    uint64_t small_bitmap_bytes = 0;
    uint64_t large_bitmap_words = 0;
    uint64_t success_value_count = 0;
};

struct EXADChunkMergeSummary {
    uint64_t post_zero_rows = 0;
    uint64_t post_zero_values = 0;
    uint64_t metadata_bytes = 0;
    uint64_t success_bytes = 0;
    uint64_t bitmap_bits = 0;
    double bitmap_density = 0.0;
};

struct EXADCurrentChunkRange {
    uint32_t bucket_begin = 0;
    uint32_t bucket_end = 0;
    uint64_t rows = 0;
};

std::string exad_chunk_dir_path(const RunOptions &options, int step) {
    return EXAD::solved_file_path(options.pathname, step) + ".chunks.tmp";
}

std::string exad_chunk_writing_path(const RunOptions &options, int step) {
    return EXAD::solved_file_path(options.pathname, step) + ".writing";
}

std::string exad_slot_chunk_path(const std::string &chunk_dir, size_t slot) {
    std::ostringstream stream;
    stream << "slot_" << std::setw(2) << std::setfill('0') << slot << ".exadslot";
    return (fs::path(chunk_dir) / stream.str()).string();
}

std::string exad_slot_part_chunk_path(
    const std::string &chunk_dir,
    size_t slot,
    size_t part,
    const char *suffix
) {
    std::ostringstream stream;
    stream << "slot_" << std::setw(2) << std::setfill('0') << slot
           << "_part_" << std::setw(4) << std::setfill('0') << part
           << ".exadslot";
    if (suffix != nullptr && *suffix != '\0') {
        stream << suffix;
    }
    return (fs::path(chunk_dir) / stream.str()).string();
}

bool env_flag_enabled_local(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    return std::strcmp(value, "0") != 0 &&
        std::strcmp(value, "false") != 0 &&
        std::strcmp(value, "FALSE") != 0;
}

bool exad_force_standard_chunked_solve() {
    return env_flag_enabled_local("EXAD_FORCE_CHUNKED_SOLVE") ||
        env_flag_enabled_local("EXAD_FORCE_STANDARD_CHUNKED_SOLVE");
}

bool env_step_matches(const char *name, int step) {
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return false;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && *end == '\0' && parsed == static_cast<long>(step);
}

uint64_t env_u64_or_zero(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return 0ULL;
    }
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        return 0ULL;
    }
    return static_cast<uint64_t>(parsed);
}

template <typename T>
uint64_t exad_current_chunk_row_budget(uint32_t row_width) {
    const uint64_t forced_rows = env_u64_or_zero("EXAD_TEST_CURRENT_CHUNK_ROWS");
    if (forced_rows != 0ULL) {
        return forced_rows;
    }
    uint64_t free_mem = available_memory_bytes();
    const uint64_t forced_memory_mib = env_u64_or_zero("EXAD_TEST_AVAILABLE_MEMORY_MIB");
    if (forced_memory_mib != 0ULL) {
        constexpr uint64_t kMiB = 1024ULL * 1024ULL;
        free_mem = forced_memory_mib > std::numeric_limits<uint64_t>::max() / kMiB
            ? std::numeric_limits<uint64_t>::max()
            : forced_memory_mib * kMiB;
    }
    const uint64_t value_slots = std::max<uint64_t>(
        1ULL << 28,
        static_cast<uint64_t>(static_cast<double>(free_mem) * 0.9 / static_cast<double>(sizeof(T)))
    );
    return std::max<uint64_t>(1ULL, value_slots / std::max<uint32_t>(row_width, 1U));
}

void maybe_throw_exad_chunked_test_stop(const char *name, int step, const char *stage) {
    if (!env_step_matches(name, step)) {
        return;
    }
    throw std::runtime_error(
        std::string("EXAD chunked solve test stop at step ") +
        std::to_string(step) + " (" + stage + ")"
    );
}

void cleanup_exad_chunk_work_files(const RunOptions &options, int step) {
    std::error_code ec;
    const std::string chunk_dir = exad_chunk_dir_path(options, step);
    const std::string writing_path = exad_chunk_writing_path(options, step);
    NativePath::remove_all(chunk_dir, ec);
    NativePath::remove(writing_path, ec);
    NativePath::remove(FileIOUtils::temp_write_path(writing_path), ec);
}

std::string exad_compressed_file_path(const RunOptions &options, int step) {
    return options.pathname + std::to_string(step) + EXADCompressedResult::kCompressedLayerFileExtension;
}

bool exad_compressed_file_is_fresh(const std::string &source_path, const std::string &output_path) {
    std::error_code ec;
    if (!NativePath::exists(output_path, ec)) {
        return false;
    }
    const auto out_time = fs::last_write_time(NativePath::from_utf8(output_path), ec);
    if (ec) {
        return false;
    }
    const auto src_time = fs::last_write_time(NativePath::from_utf8(source_path), ec);
    if (ec) {
        return false;
    }
    return out_time >= src_time;
}

bool exad_compressed_file_exists(const RunOptions &options, int step) {
    std::error_code ec;
    return NativePath::exists(exad_compressed_file_path(options, step), ec);
}

bool exad_raw_solved_file_exists(const RunOptions &options, int step) {
    return EXAD::solved_file_exists(EXAD::solved_file_path(options.pathname, step));
}

bool exad_solved_output_exists(const RunOptions &options, int step) {
    return exad_raw_solved_file_exists(options, step) ||
        exad_compressed_file_exists(options, step);
}

struct EXADSolvePlan {
    int first_step = -1;
};

EXADSolvePlan make_exad_solve_plan(const RunOptions &options) {
    EXADSolvePlan plan;
    const int last_solve_step = options.steps - 3;
    if (last_solve_step < 0) {
        return plan;
    }
    plan.first_step = last_solve_step;
    return plan;
}

void remove_exad_solved_file_after_compression(const std::string &solved_path,
                                               const std::string &compressed_path) {
    if (!exad_compressed_file_is_fresh(solved_path, compressed_path)) {
        throw std::runtime_error("EXAD compressed layer is not fresh after compression: " + compressed_path);
    }
    std::error_code ec;
    NativePath::remove(solved_path, ec);
    if (ec) {
        throw std::runtime_error("failed to remove EXAD solved layer after compression: " + solved_path);
    }
}

double maybe_compress_exad_solved_file(const RunOptions &options, int step, bool remove_source_after_compress = false) {
    if (!options.compress) {
        return 0.0;
    }
    const std::string solved_path = EXAD::solved_file_path(options.pathname, step);
    const std::string compressed_path = exad_compressed_file_path(options, step);
    if (exad_compressed_file_is_fresh(solved_path, compressed_path)) {
        if (remove_source_after_compress) {
            remove_exad_solved_file_after_compression(solved_path, compressed_path);
        }
        return 0.0;
    }
    const double t0 = wall_time_seconds();
    EXADCompressedResult::compress_exad_solved_layer_to_result(
        solved_path,
        EXAD::lut_file_path(options.pathname),
        compressed_path
    );
    const double elapsed = wall_time_seconds() - t0;
    if (remove_source_after_compress) {
        remove_exad_solved_file_after_compression(solved_path, compressed_path);
    }
    return elapsed;
}

double compress_all_exad_solved_files(const RunOptions &options, bool remove_source_after_compress) {
    if (!options.compress || options.steps < 3) {
        return 0.0;
    }
    double elapsed = 0.0;
    for (int step = 0; step <= options.steps - 3; ++step) {
        const std::string solved_path = EXAD::solved_file_path(options.pathname, step);
        if (!EXAD::solved_file_exists(solved_path)) {
            continue;
        }
        elapsed += maybe_compress_exad_solved_file(options, step, remove_source_after_compress);
    }
    return elapsed;
}

template <typename T>
double compress_exad_solved_layer_from_memory(
    const RunOptions &options,
    int step,
    const EXAD::SolvedLayer<T> &layer,
    const EXAD::Luts &luts
) {
    if (!options.compress) {
        return 0.0;
    }
    const std::string solved_path = EXAD::solved_file_path(options.pathname, step);
    const std::string compressed_path = exad_compressed_file_path(options, step);
    const std::string temp_path = compressed_path + ".tmp";
    std::error_code ec;
    NativePath::remove(temp_path, ec);
    const double t0 = wall_time_seconds();
    EXADCompressedResult::compress_exad_solved_layer_to_result_from_memory(
        layer,
        luts,
        solved_path,
        temp_path
    );
    FileIOUtils::finalize_temporary_file(temp_path, compressed_path);
    NativePath::remove(solved_path, ec);
    if (ec) {
        throw std::runtime_error("failed to remove EXAD solved layer after in-memory compression: " + solved_path);
    }
    return wall_time_seconds() - t0;
}

uint64_t exad_file_size_or_throw(const std::string &path, const char *kind) {
    std::error_code ec;
    const uintmax_t size = NativePath::file_size(path, ec);
    if (ec) {
        throw std::runtime_error(std::string("failed to determine EXAD ") + kind + " file size: " + path);
    }
    return static_cast<uint64_t>(size);
}

void read_chunk_exact(
    FileIOUtils::DirectSequentialReader &in,
    void *dst,
    size_t bytes,
    const std::string &
) {
    if (bytes != 0U) {
        in.read(dst, bytes);
    }
}

template <typename T>
EXADSlotChunkManifest manifest_from_chunk_header(
    const EXADSlotChunkHeader &header,
    const std::string &path,
    EXAD::DTypeMode expected_mode,
    size_t expected_slot
) {
    if (std::memcmp(header.magic, kEXADSlotChunkMagic, sizeof(header.magic)) != 0 ||
        header.version != kEXADSlotChunkVersion) {
        throw std::runtime_error("invalid EXAD slot chunk magic/version: " + path);
    }
    const EXAD::DTypeMode file_mode = static_cast<EXAD::DTypeMode>(header.dtype_mode);
    if (file_mode != expected_mode || !EXAD::dtype_matches_type<T>(file_mode) || header.value_size != sizeof(T)) {
        throw std::runtime_error("EXAD slot chunk dtype mismatch: " + path);
    }
    if (header.slot != expected_slot || expected_slot >= bucket_slot_count()) {
        throw std::runtime_error("EXAD slot chunk slot mismatch: " + path);
    }
    if (header.row_width != 0U) {
        const uint64_t expected_values = header.live_board_count * static_cast<uint64_t>(header.row_width);
        if (header.live_board_count != 0U &&
            expected_values / header.live_board_count != static_cast<uint64_t>(header.row_width)) {
            throw std::runtime_error("EXAD slot chunk value count overflow: " + path);
        }
        if (header.success_value_count != expected_values) {
            throw std::runtime_error("EXAD slot chunk value count mismatch: " + path);
        }
    } else if (header.success_value_count != 0U) {
        throw std::runtime_error("EXAD slot chunk has values with zero row width: " + path);
    }

    EXADSlotChunkManifest manifest;
    manifest.path = path;
    manifest.slot = header.slot;
    manifest.original_board_sum = header.original_board_sum;
    manifest.threshold_bits = header.threshold_bits;
    manifest.row_width = header.row_width;
    manifest.physical_transform = header.physical_transform;
    manifest.inverse_physical_transform = header.inverse_physical_transform;
    manifest.lut_signature = header.lut_signature;
    manifest.logical_pattern_signature = header.logical_pattern_signature;
    manifest.physical_pattern_signature = header.physical_pattern_signature;
    manifest.live_board_count = header.live_board_count;
    manifest.exact_bitmap_bits = header.exact_bitmap_bits;
    manifest.aligned_bitmap_bits = header.aligned_bitmap_bits;
    manifest.bucket_count = header.bucket_count;
    manifest.small_bitmap_bytes = header.small_bitmap_bytes;
    manifest.large_bitmap_words = header.large_bitmap_words;
    manifest.success_value_count = header.success_value_count;
    return manifest;
}

template <typename T>
EXADSlotChunkManifest read_slot_chunk_manifest(
    const std::string &path,
    EXAD::DTypeMode expected_mode,
    size_t expected_slot
) {
    std::ifstream in(NativePath::from_utf8(path), std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open EXAD slot chunk: " + path);
    }
    EXADSlotChunkHeader header{};
    FileIOUtils::read_exact(in, &header, sizeof(header), path);
    return manifest_from_chunk_header<T>(header, path, expected_mode, expected_slot);
}

template <typename T>
uint64_t slot_chunk_serialized_size(const EXAD::SolvedLayer<T> &layer, size_t slot) {
    const EXAD::BoardSet &set = layer.sets[slot];
    return sizeof(EXADSlotChunkHeader)
        + static_cast<uint64_t>(set.buckets.size()) * sizeof(EXAD::BucketEntry)
        + static_cast<uint64_t>(set.small_bitmap_bytes.size())
        + static_cast<uint64_t>(set.large_bitmap_words.size()) * sizeof(uint64_t)
        + static_cast<uint64_t>(layer.success_values.size()) * sizeof(T);
}

template <typename T>
void write_slot_chunk_file(
    const std::string &path,
    const EXAD::SolvedLayer<T> &layer,
    size_t slot,
    FileIOUtils::DirectIoConfig config
) {
    const EXAD::BoardSet &set = layer.sets[slot];
    EXADSlotChunkHeader header{};
    std::memcpy(header.magic, kEXADSlotChunkMagic, sizeof(header.magic));
    header.dtype_mode = static_cast<uint32_t>(layer.dtype_mode);
    header.value_size = sizeof(T);
    header.slot = static_cast<uint32_t>(slot);
    header.original_board_sum = layer.original_board_sum;
    header.threshold_bits = layer.threshold_bits;
    header.row_width = layer.row_width[slot];
    header.physical_transform = layer.physical_transform;
    header.inverse_physical_transform = layer.inverse_physical_transform;
    header.lut_signature = layer.lut_signature;
    header.logical_pattern_signature = layer.logical_pattern_signature;
    header.physical_pattern_signature = layer.physical_pattern_signature;
    header.live_board_count = set.live_board_count;
    header.exact_bitmap_bits = set.exact_bitmap_bits;
    header.aligned_bitmap_bits = set.aligned_bitmap_bits;
    header.bucket_count = set.buckets.size();
    header.small_bitmap_bytes = set.small_bitmap_bytes.size();
    header.large_bitmap_words = set.large_bitmap_words.size();
    header.success_value_count = layer.success_values.size();

    FileIOUtils::DirectAppendWriter out(path, slot_chunk_serialized_size(layer, slot), config);
    out.append(&header, sizeof(header));
    if (!set.buckets.empty()) {
        out.append(set.buckets.data(), set.buckets.size() * sizeof(EXAD::BucketEntry));
    }
    if (!set.small_bitmap_bytes.empty()) {
        out.append(set.small_bitmap_bytes.data(), set.small_bitmap_bytes.size());
    }
    if (!set.large_bitmap_words.empty()) {
        out.append(set.large_bitmap_words.data(), set.large_bitmap_words.size() * sizeof(uint64_t));
    }
    if (!layer.success_values.empty()) {
        out.append(layer.success_values.data(), layer.success_values.size() * sizeof(T));
    }
    out.close();
}

template <typename T>
EXAD::SolvedLayer<T> read_slot_chunk_layer(
    const std::string &path,
    EXAD::DTypeMode expected_mode,
    size_t expected_slot,
    FileIOUtils::DirectIoConfig config
) {
    FileIOUtils::DirectSequentialReader reader(
        path,
        exad_file_size_or_throw(path, "slot chunk"),
        config
    );
    EXADSlotChunkHeader header{};
    read_chunk_exact(reader, &header, sizeof(header), path);
    (void)manifest_from_chunk_header<T>(header, path, expected_mode, expected_slot);
    EXAD::SolvedLayer<T> layer;
    layer.dtype_mode = expected_mode;
    layer.original_board_sum = header.original_board_sum;
    layer.threshold_bits = header.threshold_bits;
    layer.lut_signature = header.lut_signature;
    layer.physical_transform = header.physical_transform;
    layer.inverse_physical_transform = header.inverse_physical_transform;
    layer.logical_pattern_signature = header.logical_pattern_signature;
    layer.physical_pattern_signature = header.physical_pattern_signature;
    layer.live_board_count = header.live_board_count;
    layer.row_width[expected_slot] = header.row_width;
    layer.slot_row_base[expected_slot] = 0;
    layer.slot_value_base[expected_slot] = 0;
    EXAD::BoardSet &set = layer.sets[expected_slot];
    set.threshold_bits = header.threshold_bits;
    set.live_board_count = header.live_board_count;
    set.exact_bitmap_bits = header.exact_bitmap_bits;
    set.aligned_bitmap_bits = header.aligned_bitmap_bits;
    set.buckets.resize(static_cast<size_t>(header.bucket_count));
    set.small_bitmap_bytes.resize(static_cast<size_t>(header.small_bitmap_bytes));
    set.large_bitmap_words.resize(static_cast<size_t>(header.large_bitmap_words));
    layer.success_values.resize(static_cast<size_t>(header.success_value_count));
    if (!set.buckets.empty()) {
        read_chunk_exact(reader, set.buckets.data(), set.buckets.size() * sizeof(EXAD::BucketEntry), path);
    }
    if (!set.small_bitmap_bytes.empty()) {
        read_chunk_exact(reader, set.small_bitmap_bytes.data(), set.small_bitmap_bytes.size(), path);
    }
    if (!set.large_bitmap_words.empty()) {
        read_chunk_exact(reader, set.large_bitmap_words.data(), set.large_bitmap_words.size() * sizeof(uint64_t), path);
    }
    if (!layer.success_values.empty()) {
        read_chunk_exact(reader, layer.success_values.data(), layer.success_values.size() * sizeof(T), path);
    }
    reader.close();
    return layer;
}

uint64_t exad_bucket_live_rows(const EXAD::BoardSet &set, uint32_t bucket_idx) {
    if (bucket_idx >= set.buckets.size()) {
        return 0ULL;
    }
    return EXAD::bucket_dense_count(set, bucket_idx);
}

std::vector<EXADCurrentChunkRange> exad_current_chunk_ranges(
    const EXAD::BoardSet &set,
    uint64_t row_budget
) {
    std::vector<EXADCurrentChunkRange> ranges;
    if (set.buckets.empty()) {
        ranges.push_back({});
        return ranges;
    }
    row_budget = std::max<uint64_t>(1ULL, row_budget);
    uint64_t bucket_budget = env_u64_or_zero("EXAD_TEST_CURRENT_CHUNK_BUCKETS");
    if (bucket_budget == 0ULL) {
        bucket_budget = std::numeric_limits<uint64_t>::max();
    }

    EXADCurrentChunkRange current;
    current.bucket_begin = 0;
    current.bucket_end = 0;
    current.rows = 0;
    auto flush = [&]() {
        if (current.bucket_end > current.bucket_begin) {
            ranges.push_back(current);
        }
    };

    for (uint32_t bucket_idx = 0; bucket_idx < static_cast<uint32_t>(set.buckets.size()); ++bucket_idx) {
        const uint64_t rows = exad_bucket_live_rows(set, bucket_idx);
        const uint64_t bucket_count = static_cast<uint64_t>(current.bucket_end - current.bucket_begin);
        if (current.bucket_end > current.bucket_begin &&
            (current.rows + rows > row_budget || bucket_count >= bucket_budget)) {
            flush();
            current.bucket_begin = bucket_idx;
            current.bucket_end = bucket_idx;
            current.rows = 0;
        }
        current.bucket_end = bucket_idx + 1U;
        current.rows += rows;
    }
    flush();
    if (ranges.empty()) {
        ranges.push_back({});
    }
    return ranges;
}

EXAD::BoardSet make_board_set_bucket_range(
    const EXAD::BoardSet &set,
    const EXAD::Luts &luts,
    uint32_t bucket_begin,
    uint32_t bucket_end
) {
    EXAD::BoardSet out;
    out.threshold_bits = set.threshold_bits;
    bucket_begin = std::min<uint32_t>(bucket_begin, static_cast<uint32_t>(set.buckets.size()));
    bucket_end = std::min<uint32_t>(bucket_end, static_cast<uint32_t>(set.buckets.size()));
    if (bucket_end < bucket_begin) {
        bucket_end = bucket_begin;
    }
    out.buckets.reserve(bucket_end - bucket_begin);
    uint32_t dense_offset = 0;
    for (uint32_t bucket_idx = bucket_begin; bucket_idx < bucket_end; ++bucket_idx) {
        const EXAD::BucketEntry &bucket = set.buckets[bucket_idx];
        const uint32_t group = EXAD::lut_group_index(EXAD::bucket_key_semantic_sum(bucket.key));
        if (group >= luts.size_table.size()) {
            throw std::runtime_error("EXAD current chunk bucket semantic group is outside LUT");
        }
        const uint32_t valid_count = luts.size_table[group];
        EXAD::BucketEntry out_bucket{};
        out_bucket.key = bucket.key;
        out_bucket.dense_offset = dense_offset;
        out.exact_bitmap_bits += valid_count;
        if (valid_count <= set.threshold_bits) {
            const size_t bytes = ZMaskFrozen::bytes_for_bits(valid_count);
            out_bucket.bitmap_offset = static_cast<uint32_t>(out.small_bitmap_bytes.size());
            if (bytes != 0U) {
                const auto *begin = set.small_bitmap_bytes.data() + bucket.bitmap_offset;
                out.small_bitmap_bytes.insert(out.small_bitmap_bytes.end(), begin, begin + bytes);
            }
            out.aligned_bitmap_bits += static_cast<uint64_t>(bytes) * 8ULL;
        } else {
            const size_t words = ZMaskFrozen::words_for_bits(valid_count);
            out_bucket.bitmap_offset = static_cast<uint32_t>(out.large_bitmap_words.size());
            if (words != 0U) {
                const auto *begin = set.large_bitmap_words.data() + bucket.bitmap_offset;
                out.large_bitmap_words.insert(out.large_bitmap_words.end(), begin, begin + words);
            }
            out.aligned_bitmap_bits += static_cast<uint64_t>(words) * 64ULL;
        }
        const uint32_t rows = EXAD::bucket_dense_count(set, bucket_idx);
        dense_offset += rows;
        out.live_board_count += rows;
        out.buckets.push_back(out_bucket);
    }
    return out;
}

template <typename T>
EXAD::SolvedLayer<T> make_solved_layer_from_slot(
    EXAD::BoardSet &&set,
    const EXAD::LayerFileInfo &info,
    size_t slot,
    const AdvancedMaskParam &param,
    EXAD::DTypeMode mode,
    T fill_value,
    int num_threads
);

template <typename T>
EXAD::SolvedLayer<T> make_solved_layer_from_slot_range(
    const EXAD::BoardSet &set,
    const EXAD::Luts &luts,
    const EXAD::LayerFileInfo &info,
    size_t slot,
    const AdvancedMaskParam &param,
    EXAD::DTypeMode mode,
    T fill_value,
    int num_threads,
    const EXADCurrentChunkRange &range
) {
    EXAD::BoardSet chunk_set = make_board_set_bucket_range(set, luts, range.bucket_begin, range.bucket_end);
    return make_solved_layer_from_slot<T>(
        std::move(chunk_set),
        info,
        slot,
        param,
        mode,
        fill_value,
        num_threads
    );
}

template <typename T>
EXAD::SolvedLayer<T> make_solved_layer_from_slot(
    EXAD::BoardSet &&set,
    const EXAD::LayerFileInfo &info,
    size_t slot,
    const AdvancedMaskParam &param,
    EXAD::DTypeMode mode,
    T fill_value,
    int num_threads
) {
    if (slot >= bucket_slot_count()) {
        throw std::runtime_error("EXAD slot out of range while building chunked solved layer");
    }
    EXAD::SolvedLayer<T> layer;
    layer.original_board_sum = info.original_board_sum;
    layer.threshold_bits = info.threshold_bits;
    layer.lut_signature = info.lut_signature;
    layer.physical_transform = info.physical_transform;
    layer.inverse_physical_transform = info.inverse_physical_transform;
    layer.logical_pattern_signature = info.logical_pattern_signature;
    layer.physical_pattern_signature = info.physical_pattern_signature;
    layer.dtype_mode = mode;
    layer.live_board_count = set.live_board_count;
    layer.sets[slot] = std::move(set);
    const int ad_key = bucket_key_min() + static_cast<int>(slot);
    const uint32_t width = static_cast<uint32_t>(EXAD::solved_derive_size_for_bucket(ad_key, param.num_free_32k));
    layer.row_width[slot] = width;
    const uint64_t value_count = layer.live_board_count * static_cast<uint64_t>(width);
    if (layer.live_board_count != 0U && value_count / layer.live_board_count != static_cast<uint64_t>(width)) {
        throw std::runtime_error("EXAD chunked solved layer value count overflow");
    }
    layer.success_values.resize(static_cast<size_t>(value_count));
    EXAD::fill_success_values(layer.success_values, fill_value, num_threads);
    return layer;
}

void discard_reader_bytes(
    FileIOUtils::DirectSequentialReader &reader,
    uint64_t bytes,
    const std::string &path
) {
    std::vector<uint8_t> buffer(8ULL * 1024ULL * 1024ULL);
    while (bytes != 0U) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(bytes, buffer.size()));
        read_chunk_exact(reader, buffer.data(), chunk, path);
        bytes -= static_cast<uint64_t>(chunk);
    }
}

void copy_reader_bytes(
    FileIOUtils::DirectSequentialReader &reader,
    FileIOUtils::DirectAppendWriter &writer,
    uint64_t bytes,
    const std::string &path
) {
    std::vector<uint8_t> buffer(8ULL * 1024ULL * 1024ULL);
    while (bytes != 0U) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(bytes, buffer.size()));
        read_chunk_exact(reader, buffer.data(), chunk, path);
        writer.append(buffer.data(), chunk);
        bytes -= static_cast<uint64_t>(chunk);
    }
}

template <typename T>
EXADSlotChunkHeader read_slot_chunk_header_direct(
    FileIOUtils::DirectSequentialReader &reader,
    const std::string &path,
    EXAD::DTypeMode expected_mode,
    size_t expected_slot
) {
    EXADSlotChunkHeader header{};
    read_chunk_exact(reader, &header, sizeof(header), path);
    (void)manifest_from_chunk_header<T>(header, path, expected_mode, expected_slot);
    return header;
}

template <typename T>
void append_slot_chunk_metadata(
    FileIOUtils::DirectAppendWriter &out,
    const EXADSlotChunkManifest &manifest,
    EXAD::DTypeMode expected_mode,
    FileIOUtils::DirectIoConfig config
) {
    (void)config;
    std::ifstream in(NativePath::from_utf8(manifest.path), std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open EXAD slot chunk metadata: " + manifest.path);
    }
    EXADSlotChunkHeader header{};
    FileIOUtils::read_exact(in, &header, sizeof(header), manifest.path);
    (void)manifest_from_chunk_header<T>(header, manifest.path, expected_mode, manifest.slot);
    uint64_t bytes = static_cast<uint64_t>(manifest.bucket_count) * sizeof(EXAD::BucketEntry)
        + manifest.small_bitmap_bytes
        + manifest.large_bitmap_words * sizeof(uint64_t);
    std::vector<uint8_t> buffer(8ULL * 1024ULL * 1024ULL);
    while (bytes != 0U) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(bytes, buffer.size()));
        FileIOUtils::read_exact(in, buffer.data(), chunk, manifest.path);
        out.append(buffer.data(), chunk);
        bytes -= static_cast<uint64_t>(chunk);
    }
    // The success payload is appended in a second pass so the final file keeps
    // the standard EXAD solved-layer layout without materializing all slots.
}

template <typename T>
void append_slot_chunk_success(
    FileIOUtils::DirectAppendWriter &out,
    const EXADSlotChunkManifest &manifest,
    EXAD::DTypeMode expected_mode,
    FileIOUtils::DirectIoConfig config
) {
    (void)config;
    std::ifstream in(NativePath::from_utf8(manifest.path), std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open EXAD slot chunk success: " + manifest.path);
    }
    EXADSlotChunkHeader header{};
    FileIOUtils::read_exact(in, &header, sizeof(header), manifest.path);
    (void)manifest_from_chunk_header<T>(header, manifest.path, expected_mode, manifest.slot);
    const uint64_t metadata_bytes = static_cast<uint64_t>(manifest.bucket_count) * sizeof(EXAD::BucketEntry)
        + manifest.small_bitmap_bytes
        + manifest.large_bitmap_words * sizeof(uint64_t);
    in.seekg(static_cast<std::streamoff>(metadata_bytes), std::ios::cur);
    if (!in) {
        throw std::runtime_error("failed to seek EXAD slot chunk success: " + manifest.path);
    }
    uint64_t bytes = manifest.success_value_count * sizeof(T);
    std::vector<uint8_t> buffer(8ULL * 1024ULL * 1024ULL);
    while (bytes != 0U) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(bytes, buffer.size()));
        FileIOUtils::read_exact(in, buffer.data(), chunk, manifest.path);
        out.append(buffer.data(), chunk);
        bytes -= static_cast<uint64_t>(chunk);
    }
}

void append_slot_chunk_file_bytes(
    FileIOUtils::DirectAppendWriter &out,
    const std::string &path,
    uint64_t offset,
    uint64_t bytes
) {
    if (bytes == 0U) {
        return;
    }
    std::ifstream in(NativePath::from_utf8(path), std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open EXAD slot chunk segment: " + path);
    }
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in) {
        throw std::runtime_error("failed to seek EXAD slot chunk segment: " + path);
    }
    std::vector<uint8_t> buffer(8ULL * 1024ULL * 1024ULL);
    while (bytes != 0U) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(bytes, buffer.size()));
        FileIOUtils::read_exact(in, buffer.data(), chunk, path);
        out.append(buffer.data(), chunk);
        bytes -= static_cast<uint64_t>(chunk);
    }
}

template <typename T>
EXADSlotChunkManifest merge_slot_part_chunks_to_slot_chunk(
    const std::vector<std::string> &part_paths,
    const std::string &merged_path,
    size_t slot,
    EXAD::DTypeMode dtype_mode,
    const EXAD::Luts &luts,
    FileIOUtils::DirectIoConfig config
) {
    if (part_paths.empty()) {
        throw std::runtime_error("missing EXAD slot part chunks before slot merge");
    }

    std::vector<EXADSlotChunkManifest> parts;
    parts.reserve(part_paths.size());
    uint64_t total_rows = 0;
    uint64_t total_bucket_count = 0;
    uint64_t total_small_bytes = 0;
    uint64_t total_large_words = 0;
    uint64_t total_success_values = 0;
    uint64_t total_exact_bits = 0;
    uint64_t total_aligned_bits = 0;
    uint64_t row_cursor = 0;
    uint64_t small_cursor = 0;
    uint64_t large_cursor = 0;
    std::vector<EXAD::BucketEntry> merged_buckets;

    for (const std::string &path : part_paths) {
        EXADSlotChunkManifest manifest = read_slot_chunk_manifest<T>(path, dtype_mode, slot);
        if (!parts.empty()) {
            const EXADSlotChunkManifest &base = parts.front();
            if (manifest.original_board_sum != base.original_board_sum ||
                manifest.threshold_bits != base.threshold_bits ||
                manifest.row_width != base.row_width ||
                manifest.physical_transform != base.physical_transform ||
                manifest.inverse_physical_transform != base.inverse_physical_transform ||
                manifest.lut_signature != base.lut_signature ||
                manifest.logical_pattern_signature != base.logical_pattern_signature ||
                manifest.physical_pattern_signature != base.physical_pattern_signature) {
                throw std::runtime_error("EXAD slot part metadata mismatch before slot merge: " + path);
            }
        }

        std::ifstream in(NativePath::from_utf8(path), std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open EXAD slot part metadata: " + path);
        }
        EXADSlotChunkHeader header{};
        FileIOUtils::read_exact(in, &header, sizeof(header), path);
        (void)manifest_from_chunk_header<T>(header, path, dtype_mode, slot);
        std::vector<EXAD::BucketEntry> buckets(static_cast<size_t>(manifest.bucket_count));
        if (!buckets.empty()) {
            FileIOUtils::read_exact(in, buckets.data(), buckets.size() * sizeof(EXAD::BucketEntry), path);
        }
        if (row_cursor > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::runtime_error("EXAD slot part row offset exceeds uint32 range");
        }
        if (small_cursor > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
            large_cursor > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::runtime_error("EXAD slot part bitmap offset exceeds uint32 range");
        }
        for (EXAD::BucketEntry &bucket : buckets) {
            const uint32_t group = EXAD::lut_group_index(EXAD::bucket_key_semantic_sum(bucket.key));
            if (group >= luts.size_table.size()) {
                throw std::runtime_error("EXAD slot part bucket semantic group is outside LUT");
            }
            const uint32_t valid_count = luts.size_table[group];
            if (bucket.dense_offset > std::numeric_limits<uint32_t>::max() - static_cast<uint32_t>(row_cursor)) {
                throw std::runtime_error("EXAD slot part dense offset overflow");
            }
            bucket.dense_offset += static_cast<uint32_t>(row_cursor);
            if (valid_count <= manifest.threshold_bits) {
                if (bucket.bitmap_offset >
                    std::numeric_limits<uint32_t>::max() - static_cast<uint32_t>(small_cursor)) {
                    throw std::runtime_error("EXAD slot part small bitmap offset overflow");
                }
                bucket.bitmap_offset += static_cast<uint32_t>(small_cursor);
            } else {
                if (bucket.bitmap_offset >
                    std::numeric_limits<uint32_t>::max() - static_cast<uint32_t>(large_cursor)) {
                    throw std::runtime_error("EXAD slot part large bitmap offset overflow");
                }
                bucket.bitmap_offset += static_cast<uint32_t>(large_cursor);
            }
        }
        merged_buckets.insert(merged_buckets.end(), buckets.begin(), buckets.end());

        total_rows += manifest.live_board_count;
        total_bucket_count += manifest.bucket_count;
        total_small_bytes += manifest.small_bitmap_bytes;
        total_large_words += manifest.large_bitmap_words;
        total_success_values += manifest.success_value_count;
        total_exact_bits += manifest.exact_bitmap_bits;
        total_aligned_bits += manifest.aligned_bitmap_bits;
        row_cursor += manifest.live_board_count;
        small_cursor += manifest.small_bitmap_bytes;
        large_cursor += manifest.large_bitmap_words;
        if (row_cursor > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
            small_cursor > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
            large_cursor > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::runtime_error("EXAD merged slot offsets exceed uint32 range");
        }
        parts.push_back(std::move(manifest));
    }

    const EXADSlotChunkManifest &base = parts.front();
    const uint64_t expected_success_values = total_rows * static_cast<uint64_t>(base.row_width);
    if (base.row_width != 0U && expected_success_values / static_cast<uint64_t>(base.row_width) != total_rows) {
        throw std::runtime_error("EXAD merged slot success value count overflow");
    }
    if (expected_success_values != total_success_values) {
        throw std::runtime_error("EXAD slot part success value count mismatch before slot merge");
    }
    if (total_bucket_count != static_cast<uint64_t>(merged_buckets.size())) {
        throw std::runtime_error("EXAD slot part bucket count mismatch before slot merge");
    }

    EXADSlotChunkHeader header{};
    std::memcpy(header.magic, kEXADSlotChunkMagic, sizeof(header.magic));
    header.dtype_mode = static_cast<uint32_t>(dtype_mode);
    header.value_size = sizeof(T);
    header.slot = static_cast<uint32_t>(slot);
    header.original_board_sum = base.original_board_sum;
    header.threshold_bits = base.threshold_bits;
    header.row_width = base.row_width;
    header.physical_transform = base.physical_transform;
    header.inverse_physical_transform = base.inverse_physical_transform;
    header.lut_signature = base.lut_signature;
    header.logical_pattern_signature = base.logical_pattern_signature;
    header.physical_pattern_signature = base.physical_pattern_signature;
    header.live_board_count = total_rows;
    header.exact_bitmap_bits = total_exact_bits;
    header.aligned_bitmap_bits = total_aligned_bits;
    header.bucket_count = total_bucket_count;
    header.small_bitmap_bytes = total_small_bytes;
    header.large_bitmap_words = total_large_words;
    header.success_value_count = total_success_values;

    const uint64_t serialized_size =
        sizeof(EXADSlotChunkHeader) +
        total_bucket_count * sizeof(EXAD::BucketEntry) +
        total_small_bytes +
        total_large_words * sizeof(uint64_t) +
        total_success_values * sizeof(T);
    FileIOUtils::DirectAppendWriter out(merged_path, serialized_size, config);
    out.append(&header, sizeof(header));
    if (!merged_buckets.empty()) {
        out.append(merged_buckets.data(), merged_buckets.size() * sizeof(EXAD::BucketEntry));
    }
    for (const EXADSlotChunkManifest &manifest : parts) {
        const uint64_t offset = sizeof(EXADSlotChunkHeader) +
            manifest.bucket_count * sizeof(EXAD::BucketEntry);
        append_slot_chunk_file_bytes(out, manifest.path, offset, manifest.small_bitmap_bytes);
    }
    for (const EXADSlotChunkManifest &manifest : parts) {
        const uint64_t offset = sizeof(EXADSlotChunkHeader) +
            manifest.bucket_count * sizeof(EXAD::BucketEntry) +
            manifest.small_bitmap_bytes;
        append_slot_chunk_file_bytes(
            out,
            manifest.path,
            offset,
            manifest.large_bitmap_words * sizeof(uint64_t)
        );
    }
    for (const EXADSlotChunkManifest &manifest : parts) {
        const uint64_t offset = sizeof(EXADSlotChunkHeader) +
            manifest.bucket_count * sizeof(EXAD::BucketEntry) +
            manifest.small_bitmap_bytes +
            manifest.large_bitmap_words * sizeof(uint64_t);
        append_slot_chunk_file_bytes(
            out,
            manifest.path,
            offset,
            manifest.success_value_count * sizeof(T)
        );
    }
    out.close();
    return read_slot_chunk_manifest<T>(merged_path, dtype_mode, slot);
}

template <typename T>
EXADChunkMergeSummary merge_slot_chunks_to_solved_file(
    const std::string &solved_path,
    const std::string &writing_path,
    EXAD::DTypeMode dtype_mode,
    uint32_t original_board_sum,
    uint32_t threshold_bits,
    uint64_t lut_signature,
    uint8_t physical_transform,
    uint8_t inverse_physical_transform,
    uint64_t logical_pattern_signature,
    uint64_t physical_pattern_signature,
    const std::array<EXADSlotChunkManifest, bucket_slot_count()> &manifests,
    FileIOUtils::DirectIoConfig config
) {
    std::array<EXAD::detail::SolvedSlotHeader, bucket_slot_count()> slots{};
    uint64_t row_cursor = 0;
    uint64_t value_cursor = 0;
    uint64_t metadata_payload_bytes = 0;
    uint64_t total_aligned_bits = 0;
    for (size_t slot = 0; slot < manifests.size(); ++slot) {
        const EXADSlotChunkManifest &manifest = manifests[slot];
        if (manifest.path.empty()) {
            throw std::runtime_error("missing EXAD slot chunk before merge: slot " + std::to_string(slot));
        }
        if (manifest.slot != slot ||
            manifest.original_board_sum != original_board_sum ||
            manifest.threshold_bits != threshold_bits ||
            manifest.lut_signature != lut_signature ||
            manifest.physical_transform != physical_transform ||
            manifest.inverse_physical_transform != inverse_physical_transform ||
            manifest.logical_pattern_signature != logical_pattern_signature ||
            manifest.physical_pattern_signature != physical_pattern_signature) {
            throw std::runtime_error("EXAD slot chunk metadata mismatch before merge: " + manifest.path);
        }
        slots[slot].bucket_count = manifest.bucket_count;
        slots[slot].small_bitmap_bytes = manifest.small_bitmap_bytes;
        slots[slot].large_bitmap_words = manifest.large_bitmap_words;
        slots[slot].live_board_count = manifest.live_board_count;
        slots[slot].exact_bitmap_bits = manifest.exact_bitmap_bits;
        slots[slot].aligned_bitmap_bits = manifest.aligned_bitmap_bits;
        slots[slot].row_base = row_cursor;
        slots[slot].value_base = value_cursor;
        slots[slot].row_width = manifest.row_width;
        row_cursor += manifest.live_board_count;
        value_cursor += manifest.success_value_count;
        metadata_payload_bytes += manifest.bucket_count * sizeof(EXAD::BucketEntry)
            + manifest.small_bitmap_bytes
            + manifest.large_bitmap_words * sizeof(uint64_t);
        total_aligned_bits += manifest.aligned_bitmap_bits;
    }

    EXAD::detail::SolvedFileHeader header{};
    std::memcpy(header.magic, EXAD::detail::kSolvedMagic, sizeof(header.magic));
    header.dtype_mode = static_cast<uint32_t>(dtype_mode);
    header.value_size = sizeof(T);
    header.original_board_sum = original_board_sum;
    header.threshold_bits = threshold_bits;
    header.physical_transform = physical_transform;
    header.inverse_physical_transform = inverse_physical_transform;
    header.lut_signature = lut_signature;
    header.logical_pattern_signature = logical_pattern_signature;
    header.physical_pattern_signature = physical_pattern_signature;
    header.live_board_count = row_cursor;
    header.success_value_count = value_cursor;

    const uint64_t serialized_size = sizeof(EXAD::detail::SolvedFileHeader)
        + sizeof(EXAD::detail::SolvedSlotHeader) * bucket_slot_count()
        + metadata_payload_bytes
        + value_cursor * sizeof(T);

    std::error_code ec;
    NativePath::remove(writing_path, ec);
    NativePath::remove(FileIOUtils::temp_write_path(writing_path), ec);
    FileIOUtils::DirectIoConfig final_merge_config = config;
    final_merge_config.enabled = false;
    FileIOUtils::DirectAppendWriter out(writing_path, serialized_size, final_merge_config);
    out.append(&header, sizeof(header));
    out.append(slots.data(), slots.size() * sizeof(slots[0]));
    for (const EXADSlotChunkManifest &manifest : manifests) {
        append_slot_chunk_metadata<T>(out, manifest, dtype_mode, config);
    }
    for (const EXADSlotChunkManifest &manifest : manifests) {
        append_slot_chunk_success<T>(out, manifest, dtype_mode, config);
    }
    out.close();

    FileIOUtils::finalize_temporary_file(writing_path, solved_path);

    EXADChunkMergeSummary summary;
    summary.post_zero_rows = row_cursor;
    summary.post_zero_values = value_cursor;
    summary.metadata_bytes = sizeof(uint64_t) * bucket_slot_count() * 2ULL
        + sizeof(uint32_t) * bucket_slot_count()
        + metadata_payload_bytes;
    summary.success_bytes = value_cursor * sizeof(T);
    summary.bitmap_bits = total_aligned_bits;
    summary.bitmap_density = total_aligned_bits == 0U
        ? 0.0
        : static_cast<double>(row_cursor) / static_cast<double>(total_aligned_bits);
    return summary;
}

PatternSpec exad_make_base_pattern_spec(const AdvancedPatternSpec &spec) {
    PatternSpec base;
    base.name = spec.name;
    base.pattern_masks = spec.pattern_masks;
    base.success_shifts = spec.success_shifts;
    base.symm_mode = spec.symm_mode;
    base.physical_transform = spec.physical_transform;
    base.inverse_physical_transform = spec.inverse_physical_transform;
    base.logical_pattern_signature = spec.logical_pattern_signature;
    base.physical_pattern_signature = spec.physical_pattern_signature;
    return base;
}

EXAD::Luts exad_load_or_build_luts(
    const std::vector<uint64_t> &seed_boards,
    const AdvancedPatternSpec &spec,
    const RunOptions &options,
    int num_threads,
    FileIOUtils::DirectIoConfig io_config
) {
    const std::string path = EXAD::lut_file_path(options.pathname);
    const PatternSpec base = exad_make_base_pattern_spec(spec);
    const ZMaskFrozen::TileLimitConfig config = EXAD::make_exad_lut_tile_limit_config(
        options.target,
        seed_boards,
        base,
        options.is_free,
        options.is_variant
    );
    if (NativePath::exists(path)) {
        EXAD::Luts luts = EXAD::read_lut_file(path, io_config);
        if (ZMaskFrozen::tile_limit_configs_equal(luts.config, config) &&
            luts.physical_transform == spec.physical_transform &&
            luts.inverse_physical_transform == spec.inverse_physical_transform &&
            luts.logical_pattern_signature == spec.logical_pattern_signature &&
            luts.physical_pattern_signature == spec.physical_pattern_signature) {
            EXAD::initialize_runtime_tables(luts);
            return luts;
        }
    }
    EXAD::Luts luts = EXAD::build_luts(config, num_threads);
    luts.physical_transform = spec.physical_transform;
    luts.inverse_physical_transform = spec.inverse_physical_transform;
    luts.logical_pattern_signature = spec.logical_pattern_signature;
    luts.physical_pattern_signature = spec.physical_pattern_signature;
    EXAD::write_lut_file(path, luts, io_config);
    return luts;
}

template <typename T>
T update_osr_exad_arr(
    const EXAD::SolvedLayer<T> &layer,
    const EXAD::Luts &luts,
    int ad_key,
    uint64_t board,
    uint32_t column,
    T osr
) {
    const T *row = EXAD::lookup_row_ptr(layer, luts, ad_key, board);
    if (row == nullptr || column >= layer.row_width[bucket_to_index(ad_key)]) {
        return osr;
    }
    prefetch_success_row(row);
    return std::max(osr, row[column]);
}

template <typename T>
void update_osr_exad_ranked(
    const EXAD::SolvedLayer<T> &layer,
    const EXAD::Luts &luts,
    int ad_key,
    uint64_t board,
    ArrayView<const uint32_t> ranked_array,
    std::vector<T> &osr
) {
    const T *row = EXAD::lookup_row_ptr(layer, luts, ad_key, board);
    if (row == nullptr) {
        return;
    }
    prefetch_success_row(row);
    const uint32_t width = layer.row_width[bucket_to_index(ad_key)];
    for (size_t i = 0; i < ranked_array.size; ++i) {
        const uint32_t col = ranked_array[i];
        if (col >= width) {
            continue;
        }
        T value = row[col];
        if (value > osr[i]) {
            osr[i] = value;
        }
    }
}

template <typename T>
void update_mnt_osr_exad_arr3(
    const EXAD::SolvedLayer<T> &layer,
    const EXAD::Luts &luts,
    uint64_t unmasked_board,
    int8_t count_32k,
    ArrayView<const uint32_t> ranked_array,
    std::vector<T> &osr
) {
    const int target_count = static_cast<int>(count_32k - 3 + 16);
    update_osr_exad_ranked(layer, luts, target_count, unmasked_board, ranked_array, osr);
}

template <typename T>
void update_mnt_osr_exad_arr2(
    const EXAD::SolvedLayer<T> &layer,
    const EXAD::Luts &luts,
    uint64_t board,
    uint8_t pos_rank,
    int8_t count_32k,
    ArrayView<const uint32_t> ranked_array,
    std::vector<T> &osr,
    const FormationAD::PermutationTable &permutation_table,
    const AdvancedMaskParam &param
) {
    const T *row = EXAD::lookup_row_ptr(layer, luts, count_32k, board);
    if (row == nullptr) {
        return;
    }
    prefetch_success_row(row);
    auto subset = FormationAD::permutation_first_subset(
        permutation_table,
        static_cast<uint8_t>(count_32k),
        static_cast<uint8_t>(count_32k - static_cast<int8_t>(param.num_free_32k)),
        pos_rank
    );
    if (subset.size < ranked_array.size) {
        return;
    }
    const uint32_t width = layer.row_width[bucket_to_index(count_32k)];
    for (size_t i = 0; i < ranked_array.size; ++i) {
        const uint32_t col = subset[ranked_array[i]];
        if (col >= width) {
            continue;
        }
        T value = row[col];
        if (value > osr[i]) {
            osr[i] = value;
        }
    }
}

template <typename T>
void update_mnt_osr_exad_arr1(
    const EXAD::SolvedLayer<T> &layer,
    const EXAD::Luts &luts,
    uint64_t unmasked_board,
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
    struct Candidate {
        uint64_t pos = 0;
        int symm_index = 0;
        const T *row = nullptr;
    };
    std::array<Candidate, 16> candidates{};
    size_t candidate_count = 0;
    const int bucket_key = -count_32k;
    for (int j = 0; j < count_32k; ++j) {
        if (j == static_cast<int>(pos_rank)) {
            continue;
        }
        const uint64_t pos = static_cast<uint64_t>(positions[static_cast<size_t>(j)]);
        uint64_t unmasked_b_j =
            (unmasked_board & ~(0xFULL << pos)) | (static_cast<uint64_t>(tiles_combinations[0]) << pos);
        auto [canonical_board, symm_index] = canonical_pair_by_mode(unmasked_b_j, spec.symm_mode);

        const T *row = EXAD::lookup_row_ptr(layer, luts, bucket_key, canonical_board);
        if (row == nullptr) {
            continue;
        }
        if (candidate_count < candidates.size()) {
            candidates[candidate_count++] = Candidate{pos, symm_index, row};
        }
    }

    for (size_t candidate_i = 0; candidate_i < candidate_count; ++candidate_i) {
        prefetch_success_row(candidates[candidate_i].row);
    }

    for (size_t candidate_i = 0; candidate_i < candidate_count; ++candidate_i) {
        const uint64_t pos = candidates[candidate_i].pos;
        const int symm_index = candidates[candidate_i].symm_index;
        const T *row = candidates[candidate_i].row;
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
        const uint32_t width = layer.row_width[bucket_to_index(bucket_key)];
        for (size_t index = 0; index < ranked_array.size(); ++index) {
            const uint32_t col = ranked_array[index];
            if (col >= width) {
                continue;
            }
            T value = row[col];
            size_t position = matched_positions[index];
            if (value > osr[position]) {
                osr[position] = value;
            }
        }
    }
}

template <typename T>
T update_mnt_osr_364_exad(
    const EXAD::SolvedLayer<T> &layer,
    const EXAD::Luts &luts,
    uint64_t board,
    uint64_t unmasked_board,
    T osr,
    const AdvancedMaskParam &param
) {
    auto [pos_rank64, pos_rank128, pos_rank, pos_64] = find_3x64_pos(unmasked_board, param);
    board |= (0xFULL << pos_64);
    const T *row = EXAD::lookup_row_ptr(layer, luts, pos_rank, board);
    if (row == nullptr) {
        return osr;
    }
    prefetch_success_row(row);
    uint32_t mapping = permutations_mapping_364(pos_rank64, pos_rank128, pos_rank);
    if (mapping >= layer.row_width[bucket_to_index(pos_rank)]) {
        return osr;
    }
    return std::max(osr, row[mapping]);
}

template <typename T>
void update_mnt_osr_364_arr_exad(
    const EXAD::SolvedLayer<T> &layer,
    const EXAD::Luts &luts,
    uint64_t board,
    uint64_t unmasked_board,
    std::vector<T> &osr,
    ArrayView<const uint32_t> ranked_array,
    const FormationAD::PermutationTable &permutation_table,
    const AdvancedMaskParam &param
) {
    auto [pos_rank64, pos_rank128, pos_rank, pos_64] = find_3x64_pos(unmasked_board, param);
    board |= (0xFULL << pos_64);
    const T *row = EXAD::lookup_row_ptr(layer, luts, pos_rank, board);
    if (row == nullptr) {
        return;
    }
    prefetch_success_row(row);
    auto subset = FormationAD::permutation_pair_subset(
        permutation_table,
        static_cast<uint8_t>(pos_rank),
        static_cast<uint8_t>(pos_rank - static_cast<int8_t>(param.num_free_32k)),
        pos_rank64,
        pos_rank128
    );
    if (subset.size < ranked_array.size) {
        return;
    }
    const uint32_t width = layer.row_width[bucket_to_index(pos_rank)];
    for (size_t i = 0; i < ranked_array.size; ++i) {
        const uint32_t col = subset[ranked_array[i]];
        if (col >= width) {
            continue;
        }
        T value = row[col];
        if (value > osr[i]) {
            osr[i] = value;
        }
    }
}

template <typename T>
T update_mnt_osr_exad(
    const EXAD::SolvedLayer<T> &layer,
    const EXAD::Luts &luts,
    uint64_t board,
    uint64_t unmasked_board,
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
    if (tiles.empty() || stats.merged_tile_found == 2U) {
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
    const T *row = EXAD::lookup_row_ptr(layer, luts, bucket_key, board);
    if (row == nullptr) {
        return osr;
    }
    prefetch_success_row(row);
    if (bucket_key > 15) {
        return std::max(osr, row[0]);
    }
    if (bucket_key > 0) {
        if (stats.pos_rank >= layer.row_width[bucket_to_index(bucket_key)]) {
            return osr;
        }
        return std::max(osr, row[stats.pos_rank]);
    }
    return std::max(osr, row[0]);
}

template <typename T>
void solve_optimal_success_rate_arr_into_exad(
    uint64_t board,
    uint64_t new_value,
    int spawn_pos,
    uint64_t rep_t,
    uint64_t rep_t_rev,
    size_t derive_size,
    const AdvancedPatternSpec &spec,
    uint64_t rep_v,
    int8_t count_32k,
    const EXAD::SolvedLayer<T> &future,
    const EXAD::Luts &luts,
    uint32_t board_sum_after_spawn,
    MatchCache &match_cache,
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
        auto [canonical_board, symm_index] = canonical_pair_by_mode(moved_board, spec.symm_mode);
        uint64_t rep_t_gen_m = BoardMover::move_board(rep_t_gen, direction + 1);
        rep_t_gen_m = apply_sym_like(rep_t_gen_m, symm_index);
        uint64_t match_ind = ind_match(rep_t_gen_m, rep_v);
        const bool use_match_cache = match_cache_ready(match_cache, derive_size);
        size_t hashed_match_ind = 0U;
        if (use_match_cache) {
            hashed_match_ind = static_cast<size_t>(match_ind % static_cast<uint64_t>(match_cache.map_length));
        }

        auto load_or_compute_ranked = [&](bool store_if_empty) -> ArrayView<const uint32_t> {
            if (use_match_cache && match_cache.key(hashed_match_ind) == match_ind) {
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
            if (use_match_cache &&
                store_if_empty &&
                workspace.ranked_array.size() == derive_size &&
                match_cache.try_claim_empty(hashed_match_ind)) {
                match_cache.publish(hashed_match_ind, match_ind, workspace.ranked_array);
                return {match_cache.row(hashed_match_ind), derive_size};
            }
            if (use_match_cache && match_cache.key(hashed_match_ind) == match_ind) {
                return {match_cache.row(hashed_match_ind), derive_size};
            }
            return {workspace.ranked_array.data(), workspace.ranked_array.size()};
        };

        if (mask_new_tile) {
            uint64_t unmasked_newt = BoardMover::move_board(board_with_spawn, direction + 1);
            unmasked_newt = apply_sym_like(unmasked_newt, symm_index);
            if (count_32k > 15) {
                ArrayView<const uint32_t> ranked_array = load_or_compute_ranked(true);
                update_mnt_osr_364_arr_exad(
                    future, luts, canonical_board, unmasked_newt, optimal_success_rate, ranked_array,
                    permutation_table, param
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
                update_mnt_osr_exad_arr3(
                    future, luts, unmasked_newt, dispatch.count32k, ranked_array, optimal_success_rate
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
                update_mnt_osr_exad_arr1(
                    future, luts, unmasked_newt, workspace.moved_boards, optimal_success_rate, spec,
                    dispatch.tiles_combinations, dispatch.pos_rank, dispatch.pos_32k, dispatch.tile_value,
                    dispatch.count32k, workspace
                );
            } else {
                ArrayView<const uint32_t> ranked_array = load_or_compute_ranked(true);
                update_mnt_osr_exad_arr2(
                    future, luts, canonical_board, dispatch.pos_rank, dispatch.count32k, ranked_array,
                    optimal_success_rate, permutation_table, param
                );
            }
            continue;
        }

        ArrayView<const uint32_t> ranked_array = load_or_compute_ranked(true);
        update_osr_exad_ranked(future, luts, count_32k, canonical_board, ranked_array, optimal_success_rate);
    }
}

template <typename T>
T solve_optimal_success_rate_exad(
    uint64_t board,
    uint64_t new_value,
    int spawn_pos,
    uint32_t board_sum_after_spawn,
    const AdvancedPatternSpec &spec,
    int8_t count_32k,
    const EXAD::SolvedLayer<T> &future,
    const EXAD::Luts &luts,
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
        auto [canonical_board, symm_index] = canonical_pair_by_mode(moved_board, spec.symm_mode);
        if (mask_new_tile) {
            uint64_t unmasked_newt = BoardMover::move_board(board_with_spawn, direction + 1);
            unmasked_newt = apply_sym_like(unmasked_newt, symm_index);
            if (count_32k < 16) {
                optimal_success_rate = update_mnt_osr_exad(
                    future,
                    luts,
                    canonical_board,
                    unmasked_newt,
                    optimal_success_rate,
                    board_sum_after_spawn,
                    tiles_table,
                    param,
                    max_scale
                );
            } else {
                optimal_success_rate = update_mnt_osr_364_exad(
                    future, luts, canonical_board, unmasked_newt, optimal_success_rate, param
                );
            }
        } else {
            optimal_success_rate = update_osr_exad_arr(future, luts, count_32k, canonical_board, 0U, optimal_success_rate);
        }
    }
    return optimal_success_rate;
}

template <typename T>
void collect_scalar_spawn_queries_exad(
    uint64_t board,
    uint64_t new_value,
    int spawn_pos,
    uint32_t board_sum_after_spawn,
    const AdvancedPatternSpec &spec,
    int8_t count_32k,
    const EXAD::SolvedLayer<T> &future,
    const EXAD::Luts &luts,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    T max_scale,
    uint16_t ref,
    std::vector<EXAD::PreparedQuery> &queries,
    std::vector<uint64_t> &canonical_candidates,
    std::vector<uint16_t> &candidate_refs,
    T *best
) {
    uint64_t board_with_spawn = board | (new_value << static_cast<uint64_t>(4 * spawn_pos));
    uint64_t board_rev = FormationAD::reverse(board_with_spawn);
    auto moves = FormationAD::m_move_all_dir2(board_with_spawn, board_rev);

    for (int direction = 0; direction < 4; ++direction) {
        const uint64_t moved_board = moves[static_cast<size_t>(direction)].board;
        const bool mask_new_tile = moves[static_cast<size_t>(direction)].mask_new_tile;
        if (moved_board == board_with_spawn || !is_pattern(moved_board, spec.pattern_masks)) {
            continue;
        }
        if (!mask_new_tile) {
            canonical_candidates.push_back(moved_board);
            candidate_refs.push_back(ref);
            continue;
        }

        auto [canonical_board, symm_index] = canonical_pair_by_mode(moved_board, spec.symm_mode);
        uint64_t unmasked_newt = BoardMover::move_board(board_with_spawn, direction + 1);
        unmasked_newt = apply_sym_like(unmasked_newt, symm_index);
        if (count_32k >= 16) {
            auto [pos_rank64, pos_rank128, pos_rank, pos_64] = find_3x64_pos(unmasked_newt, param);
            uint64_t lookup_board = canonical_board | (0xFULL << pos_64);
            const uint32_t column = permutations_mapping_364(pos_rank64, pos_rank128, pos_rank);
            EXAD::PreparedQuery query;
            if (EXAD::prepare_query(luts, pos_rank, lookup_board, ref, column, query)) {
                queries.push_back(query);
            }
            continue;
        }

        auto stats = FormationAD::tile_sum_and_32k_count4(unmasked_newt, param);
        if (stats.is_success) {
            best[ref] = max_scale;
            continue;
        }
        uint32_t large_tiles_sum = board_sum_after_spawn
            - stats.total_sum
            - (static_cast<uint32_t>(param.num_free_32k + param.num_fixed_32k) << 15U);
        auto tiles = FormationAD::tiles_combination_view(
            tiles_table,
            static_cast<uint8_t>(large_tiles_sum >> 6U),
            static_cast<uint8_t>(stats.count_32k - static_cast<int8_t>(param.num_free_32k))
        );
        if (tiles.empty() || stats.merged_tile_found == 2U) {
            continue;
        }
        int bucket_key = stats.count_32k;
        uint64_t lookup_board = canonical_board;
        uint32_t column = stats.pos_rank;
        if (tiles.size > 1 && tiles[0] == tiles[1]) {
            column = 0U;
            if (tiles.size > 2 && tiles[0] == tiles[2]) {
                bucket_key = static_cast<int8_t>(stats.count_32k + 16 - 3);
                lookup_board = unmasked_newt;
            } else {
                bucket_key = static_cast<int8_t>(-stats.count_32k);
                const uint64_t tile = static_cast<uint64_t>(tiles[0]);
                const uint64_t tiles_all_positions = tile * 0x1111111111111111ULL;
                lookup_board = (canonical_board & (~stats.pos_bitmap)) | (tiles_all_positions & stats.pos_bitmap);
            }
        }
        EXAD::PreparedQuery query;
        if (EXAD::prepare_query(luts, bucket_key, lookup_board, ref, column, query)) {
            queries.push_back(query);
        }
    }
}

template <typename Fn>
bool append_vector_query_columns(
    const EXAD::PreparedQuery &prepared,
    uint32_t width,
    std::vector<EXADVectorQuery> &queries,
    std::vector<uint32_t> &columns,
    Fn &&column_at
) {
    if (width == 0U || width > std::numeric_limits<uint16_t>::max()) {
        return false;
    }
    if (columns.size() > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    const uint32_t offset = static_cast<uint32_t>(columns.size());
    for (uint32_t i = 0; i < width; ++i) {
        columns.push_back(static_cast<uint32_t>(column_at(i)));
    }
    EXADVectorQuery vector_query;
    vector_query.query = prepared;
    vector_query.column_offset = offset;
    queries.push_back(vector_query);
    return true;
}

bool append_ranked_vector_query(
    const EXAD::PreparedQuery &prepared,
    ArrayView<const uint32_t> ranked_array,
    uint32_t width,
    std::vector<EXADVectorQuery> &queries,
    std::vector<uint32_t> &columns
) {
    if (ranked_array.size < width) {
        return false;
    }
    return append_vector_query_columns(prepared, width, queries, columns, [&](uint32_t i) {
        return ranked_array[i];
    });
}

bool append_subset_vector_query(
    const EXAD::PreparedQuery &prepared,
    ArrayView<const uint32_t> ranked_array,
    ArrayView<const uint32_t> subset,
    uint32_t width,
    std::vector<EXADVectorQuery> &queries,
    std::vector<uint32_t> &columns
) {
    if (ranked_array.size < width) {
        return false;
    }
    const size_t old_size = columns.size();
    if (old_size > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    for (uint32_t i = 0; i < width; ++i) {
        const uint32_t ranked = ranked_array[i];
        if (ranked >= subset.size) {
            columns.resize(old_size);
            return false;
        }
        columns.push_back(subset[ranked]);
    }
    EXADVectorQuery vector_query;
    vector_query.query = prepared;
    vector_query.column_offset = static_cast<uint32_t>(old_size);
    queries.push_back(vector_query);
    return true;
}

template <uint32_t StaticWidth = 0U, typename T>
uint64_t lookup_reduce_vector_queries_exad(
    const EXAD::SolvedLayer<T> &layer,
    const std::vector<EXADVectorQuery> &queries,
    const std::vector<uint32_t> &columns,
    uint32_t vector_width,
    T *best
) {
    uint64_t found = 0U;
    const uint32_t count = static_cast<uint32_t>(queries.size());
    for (uint32_t base = 0; base < count; base += EXAD::kLookupBatchSize) {
        const uint32_t block_count = std::min<uint32_t>(EXAD::kLookupBatchSize, count - base);
        std::array<const EXAD::DirectEntry *, EXAD::kLookupBatchSize> entries{};
        std::array<const T *, EXAD::kLookupBatchSize> rows{};
        std::array<uint32_t, EXAD::kLookupBatchSize> hash_slots{};

        for (uint32_t i = 0; i < block_count; ++i) {
            const EXAD::PreparedQuery &query = queries[base + i].query;
            if (query.valid == 0U || query.slot >= bucket_slot_count()) {
                continue;
            }
            const EXAD::DirectEntryIndex &index = layer.direct_entry_indices[query.slot];
            if (index.empty()) {
                continue;
            }
            hash_slots[i] = static_cast<uint32_t>(EXAD::mix_key64(query.key)) & index.mask;
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(&index.entries[hash_slots[i]], 0, 1);
#endif
        }

        for (uint32_t i = 0; i < block_count; ++i) {
            const EXAD::PreparedQuery &query = queries[base + i].query;
            if (query.valid == 0U || query.slot >= bucket_slot_count()) {
                continue;
            }
            const EXAD::DirectEntryIndex &index = layer.direct_entry_indices[query.slot];
            if (index.empty()) {
                continue;
            }
            uint32_t hash_slot = hash_slots[i];
            for (;;) {
                const EXAD::DirectEntry &entry = index.entries[hash_slot];
                if (entry.key == EXAD::kInvalidBucketKey) {
                    break;
                }
                if (entry.key == query.key) {
                    entries[i] = &entry;
                    break;
                }
                hash_slot = (hash_slot + 1U) & index.mask;
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(&index.entries[hash_slot], 0, 1);
#endif
            }
        }

        for (uint32_t i = 0; i < block_count; ++i) {
            const EXAD::DirectEntry *entry = entries[i];
            if (entry == nullptr) {
                continue;
            }
            const EXAD::PreparedQuery &query = queries[base + i].query;
            const EXAD::BoardSet &set = layer.sets[query.slot];
            if (query.valid_count <= set.threshold_bits) {
                const uint32_t byte_idx = query.rank >> 3U;
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(&set.small_bitmap_bytes[entry->bitmap_offset + byte_idx], 0, 1);
#endif
            } else {
                const uint32_t word_idx = query.rank >> 6U;
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(&set.large_bitmap_words[entry->bitmap_offset + word_idx], 0, 1);
                __builtin_prefetch(&layer.large_rank_bases[query.slot][entry->bitmap_offset + word_idx], 0, 1);
#endif
            }
        }

        for (uint32_t i = 0; i < block_count; ++i) {
            const EXAD::DirectEntry *entry = entries[i];
            if (entry == nullptr) {
                continue;
            }
            const EXAD::PreparedQuery &query = queries[base + i].query;
            const EXAD::BoardSet &set = layer.sets[query.slot];
            uint32_t ordinal = 0;
            if (!EXAD::dense_ordinal_for_rank_fast(
                    set,
                    *entry,
                    query.valid_count,
                    query.rank,
                    layer.large_rank_bases[query.slot],
                    ordinal)) {
                continue;
            }
            const uint64_t local_row = static_cast<uint64_t>(entry->dense_offset) + ordinal;
            rows[i] = EXAD::row_ptr(layer, query.slot, local_row);
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(rows[i], 0, 1);
#endif
        }

        for (uint32_t i = 0; i < block_count; ++i) {
            const T *row = rows[i];
            if (row == nullptr) {
                continue;
            }
            const EXADVectorQuery &vector_query = queries[base + i];
            const uint32_t active_width = vector_width;
            const uint32_t loop_width = StaticWidth == 0U ? vector_width : StaticWidth;
            T *dst = best + static_cast<size_t>(vector_query.query.ref) * active_width;
            const uint32_t row_width = layer.row_width[vector_query.query.slot];
            const uint32_t *query_columns = columns.data() + vector_query.column_offset;
            for (uint32_t col_i = 0; col_i < loop_width; ++col_i) {
                if (col_i >= active_width) {
                    break;
                }
                const uint32_t column = query_columns[col_i];
                if (column >= row_width) {
                    continue;
                }
                T value = row[column];
                if (value > dst[col_i]) {
                    dst[col_i] = value;
                }
            }
            ++found;
        }
    }
    return found;
}

template <typename T>
uint64_t lookup_reduce_vector_queries_by_width_exad(
    const EXAD::SolvedLayer<T> &layer,
    const std::vector<EXADVectorQuery> &queries,
    const std::vector<uint32_t> &columns,
    uint32_t width,
    T *best
) {
    if (width <= 8U) {
        return lookup_reduce_vector_queries_exad<8U>(layer, queries, columns, width, best);
    }
    if (width <= 16U) {
        return lookup_reduce_vector_queries_exad<16U>(layer, queries, columns, width, best);
    }
    if (width <= 32U) {
        return lookup_reduce_vector_queries_exad<32U>(layer, queries, columns, width, best);
    }
    if (width <= 64U) {
        return lookup_reduce_vector_queries_exad<64U>(layer, queries, columns, width, best);
    }
    if (width <= 128U) {
        return lookup_reduce_vector_queries_exad<128U>(layer, queries, columns, width, best);
    }
    return lookup_reduce_vector_queries_exad(layer, queries, columns, width, best);
}

template <typename T>
void collect_vector_spawn_queries_exad(
    uint64_t board,
    uint64_t new_value,
    int spawn_pos,
    uint64_t rep_t,
    uint64_t rep_v,
    const AdvancedPatternSpec &spec,
    int8_t count_32k,
    const EXAD::SolvedLayer<T> &future,
    const EXAD::Luts &luts,
    uint32_t board_sum_after_spawn,
    MatchCache &match_cache,
    const FormationAD::TilesCombinationTable &tiles_table,
    const FormationAD::PermutationTable &permutation_table,
    const AdvancedMaskParam &param,
    T zero_val,
    T max_scale,
    AdSolveWorkspace<T> &workspace,
    uint16_t ref,
    uint32_t width,
    std::vector<EXADVectorQuery> &queries,
    std::vector<uint32_t> &columns,
    T *best
) {
    uint64_t board_with_spawn = board | (new_value << static_cast<uint64_t>(4 * spawn_pos));
    uint64_t rep_t_gen = rep_t | (new_value << static_cast<uint64_t>(4 * spawn_pos));
    uint64_t board_rev = FormationAD::reverse(board_with_spawn);
    auto moves = FormationAD::m_move_all_dir2(board_with_spawn, board_rev);
    bool mask_values_used = false;

    auto ensure_mask_values = [&]() -> std::vector<T> & {
        if (!mask_values_used) {
            workspace.optimal_values.assign(width, zero_val);
            mask_values_used = true;
        }
        return workspace.optimal_values;
    };

    for (int direction = 0; direction < 4; ++direction) {
        uint64_t moved_board = moves[static_cast<size_t>(direction)].board;
        bool mask_new_tile = moves[static_cast<size_t>(direction)].mask_new_tile;
        if (moved_board == board_with_spawn || !is_pattern(moved_board, spec.pattern_masks)) {
            continue;
        }
        auto [canonical_board, symm_index] = canonical_pair_by_mode(moved_board, spec.symm_mode);
        uint64_t rep_t_gen_m = BoardMover::move_board(rep_t_gen, direction + 1);
        rep_t_gen_m = apply_sym_like(rep_t_gen_m, symm_index);
        uint64_t match_ind = ind_match(rep_t_gen_m, rep_v);
        const bool use_match_cache = match_cache_ready(match_cache, width);
        size_t hashed_match_ind = 0U;
        if (use_match_cache) {
            hashed_match_ind = static_cast<size_t>(match_ind % static_cast<uint64_t>(match_cache.map_length));
        }

        auto load_or_compute_ranked = [&](bool store_if_empty) -> ArrayView<const uint32_t> {
            if (use_match_cache && match_cache.key(hashed_match_ind) == match_ind) {
                return {match_cache.row(hashed_match_ind), width};
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
            if (use_match_cache &&
                store_if_empty &&
                workspace.ranked_array.size() == width &&
                match_cache.try_claim_empty(hashed_match_ind)) {
                match_cache.publish(hashed_match_ind, match_ind, workspace.ranked_array);
                return {match_cache.row(hashed_match_ind), width};
            }
            if (use_match_cache && match_cache.key(hashed_match_ind) == match_ind) {
                return {match_cache.row(hashed_match_ind), width};
            }
            return {workspace.ranked_array.data(), workspace.ranked_array.size()};
        };

        if (mask_new_tile) {
            uint64_t unmasked_newt = BoardMover::move_board(board_with_spawn, direction + 1);
            unmasked_newt = apply_sym_like(unmasked_newt, symm_index);
            if (count_32k > 15) {
                ArrayView<const uint32_t> ranked_array = load_or_compute_ranked(true);
                auto [pos_rank64, pos_rank128, pos_rank, pos_64] = find_3x64_pos(unmasked_newt, param);
                uint64_t lookup_board = canonical_board | (0xFULL << pos_64);
                auto subset = FormationAD::permutation_pair_subset(
                    permutation_table,
                    static_cast<uint8_t>(pos_rank),
                    static_cast<uint8_t>(pos_rank - static_cast<int8_t>(param.num_free_32k)),
                    pos_rank64,
                    pos_rank128
                );
                if (ranked_array.size >= width) {
                    EXAD::PreparedQuery prepared;
                    if (EXAD::prepare_query(luts, pos_rank, lookup_board, ref, 0U, prepared)) {
                        append_subset_vector_query(prepared, ranked_array, subset, width, queries, columns);
                    }
                }
                continue;
            }

            auto stats = FormationAD::tile_sum_and_32k_count4(unmasked_newt, param);
            if (stats.is_success) {
                T *dst = best + static_cast<size_t>(ref) * width;
                std::fill(dst, dst + width, max_scale);
                continue;
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
                continue;
            }
            if (
                tiles.size > 2 &&
                tiles[0] == tiles[1] &&
                tiles[0] == tiles[2]
            ) {
                ArrayView<const uint32_t> ranked_array = load_or_compute_ranked(true);
                if (ranked_array.size >= width) {
                    const int bucket_key = static_cast<int8_t>(stats.count_32k + 16 - 3);
                    EXAD::PreparedQuery prepared;
                    if (EXAD::prepare_query(luts, bucket_key, unmasked_newt, ref, 0U, prepared)) {
                        append_ranked_vector_query(prepared, ranked_array, width, queries, columns);
                    }
                }
            } else if (
                tiles.size > 1 &&
                tiles[0] == tiles[1]
            ) {
                std::vector<T> &mask_values = ensure_mask_values();
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
                update_mnt_osr_exad_arr1(
                    future, luts, unmasked_newt, workspace.moved_boards, mask_values, spec,
                    tiles, stats.pos_rank, stats.pos_bitmap, stats.merged_tile,
                    stats.count_32k, workspace
                );
            } else {
                ArrayView<const uint32_t> ranked_array = load_or_compute_ranked(true);
                auto subset = FormationAD::permutation_first_subset(
                    permutation_table,
                    static_cast<uint8_t>(stats.count_32k),
                    static_cast<uint8_t>(stats.count_32k - static_cast<int8_t>(param.num_free_32k)),
                    stats.pos_rank
                );
                if (ranked_array.size >= width) {
                    EXAD::PreparedQuery prepared;
                    if (EXAD::prepare_query(luts, stats.count_32k, canonical_board, ref, 0U, prepared)) {
                        append_subset_vector_query(prepared, ranked_array, subset, width, queries, columns);
                    }
                }
            }
            continue;
        }

        ArrayView<const uint32_t> ranked_array = load_or_compute_ranked(true);
        if (ranked_array.size < width) {
            continue;
        }
        EXAD::PreparedQuery prepared;
        if (!EXAD::prepare_query(luts, count_32k, canonical_board, ref, 0U, prepared)) {
            continue;
        }
        append_ranked_vector_query(prepared, ranked_array, width, queries, columns);
    }

    if (mask_values_used) {
        const std::vector<T> &mask_values = workspace.optimal_values;
        T *dst = best + static_cast<size_t>(ref) * width;
        for (uint32_t i = 0; i < width; ++i) {
            if (mask_values[i] > dst[i]) {
                dst[i] = mask_values[i];
            }
        }
    }
}

template <typename T>
void recalculate_exad_scalar_batch(
    EXAD::SolvedLayer<T> &current,
    size_t current_slot,
    const uint64_t *boards,
    const uint64_t *local_rows,
    uint32_t board_count,
    const EXAD::SolvedLayer<T> &future1,
    const EXAD::SolvedLayer<T> &future2,
    const EXAD::Luts &luts,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    uint32_t original_board_sum,
    const AdvancedPatternSpec &spec,
    int8_t count_32k,
    T max_scale,
    T zero_val,
    double spawn_rate4,
    EXADScalarBatchWorkspace<T> &workspace
) {
    if (board_count == 0U) {
        return;
    }
    workspace.queries1.clear();
    workspace.queries2.clear();
    workspace.canonical_candidates1.clear();
    workspace.canonical_candidates2.clear();
    workspace.candidate_refs1.clear();
    workspace.candidate_refs2.clear();
    const size_t best_slots = static_cast<size_t>(board_count) * 16U;
    std::fill(workspace.best2.begin(), workspace.best2.begin() + best_slots, zero_val);
    std::fill(workspace.best4.begin(), workspace.best4.begin() + best_slots, zero_val);

    auto prepare_canonical_queries = [&](
        std::vector<uint64_t> &canonical_candidates,
        std::vector<uint16_t> &candidate_refs,
        std::vector<EXAD::PreparedQuery> &queries
    ) {
        if (canonical_candidates.empty()) {
            return;
        }
        CanonicalBatch::canonicalize_inplace(
            canonical_candidates.data(),
            canonical_candidates.size(),
            spec.symm_mode
        );
        for (size_t i = 0; i < canonical_candidates.size(); ++i) {
            EXAD::PreparedQuery query;
            if (EXAD::prepare_query(luts, count_32k, canonical_candidates[i], candidate_refs[i], 0U, query)) {
                queries.push_back(query);
            }
        }
    };

    for (uint32_t board_slot = 0; board_slot < board_count; ++board_slot) {
        const uint64_t board = boards[board_slot];
        uint32_t empty_mask = empty_cell_mask16(board);
        workspace.empty_masks[board_slot] = static_cast<uint16_t>(empty_mask);
        while (empty_mask != 0U) {
            const int pos = static_cast<int>(ctz_u32(empty_mask));
            empty_mask &= empty_mask - 1U;
            const uint16_t ref = static_cast<uint16_t>(board_slot * 16U + static_cast<uint32_t>(pos));
            collect_scalar_spawn_queries_exad(
                board,
                1ULL,
                pos,
                original_board_sum + 2U,
                spec,
                count_32k,
                future1,
                luts,
                tiles_table,
                param,
                max_scale,
                ref,
                workspace.queries1,
                workspace.canonical_candidates1,
                workspace.candidate_refs1,
                workspace.best2.data()
            );
            collect_scalar_spawn_queries_exad(
                board,
                2ULL,
                pos,
                original_board_sum + 4U,
                spec,
                count_32k,
                future2,
                luts,
                tiles_table,
                param,
                max_scale,
                ref,
                workspace.queries2,
                workspace.canonical_candidates2,
                workspace.candidate_refs2,
                workspace.best4.data()
            );
        }
    }

    prepare_canonical_queries(workspace.canonical_candidates1, workspace.candidate_refs1, workspace.queries1);
    prepare_canonical_queries(workspace.canonical_candidates2, workspace.candidate_refs2, workspace.queries2);
    EXAD::lookup_reduce_prepared_queries(
        future1,
        workspace.queries1.data(),
        static_cast<uint32_t>(workspace.queries1.size()),
        workspace.best2.data()
    );
    EXAD::lookup_reduce_prepared_queries(
        future2,
        workspace.queries2.data(),
        static_cast<uint32_t>(workspace.queries2.size()),
        workspace.best4.data()
    );

    for (uint32_t board_slot = 0; board_slot < board_count; ++board_slot) {
        double success_probability = 0.0;
        uint32_t empty_count = 0U;
        uint32_t empty_mask = workspace.empty_masks[board_slot];
        while (empty_mask != 0U) {
            const uint32_t pos = ctz_u32(empty_mask);
            empty_mask &= empty_mask - 1U;
            const size_t best_index = static_cast<size_t>(board_slot) * 16U + pos;
            success_probability += static_cast<double>(workspace.best2[best_index]) * (1.0 - spawn_rate4);
            success_probability += static_cast<double>(workspace.best4[best_index]) * spawn_rate4;
            ++empty_count;
        }
        T *dst = EXAD::row_ptr(current, current_slot, local_rows[board_slot]);
        dst[0] = empty_count == 0U
            ? zero_val
            : static_cast<T>(success_probability / static_cast<double>(empty_count));
    }
}

template <uint32_t MaxWidth, typename T>
void finish_vector_batch_rows_bounded(
    EXAD::SolvedLayer<T> &current,
    size_t current_slot,
    const uint64_t *local_rows,
    uint32_t board_count,
    uint32_t width,
    double spawn_rate4,
    T zero_val,
    EXADVectorBatchWorkspace<T> &batch_workspace
) {
    for (uint32_t board_slot = 0; board_slot < board_count; ++board_slot) {
        std::array<double, MaxWidth> success_probability{};
        uint32_t empty_count = 0U;
        uint32_t empty_mask = batch_workspace.empty_masks[board_slot];
        while (empty_mask != 0U) {
            const uint32_t pos = ctz_u32(empty_mask);
            empty_mask &= empty_mask - 1U;
            const size_t best_index = (static_cast<size_t>(board_slot) * 16U + pos) * width;
            for (uint32_t idx = 0; idx < MaxWidth; ++idx) {
                if (idx >= width) {
                    break;
                }
                success_probability[idx] +=
                    static_cast<double>(batch_workspace.best2[best_index + idx]) * (1.0 - spawn_rate4);
                success_probability[idx] +=
                    static_cast<double>(batch_workspace.best4[best_index + idx]) * spawn_rate4;
            }
            ++empty_count;
        }
        T *dst = EXAD::row_ptr(current, current_slot, local_rows[board_slot]);
        if (empty_count == 0U) {
            std::fill(dst, dst + width, zero_val);
            continue;
        }
        const double inv_empty = 1.0 / static_cast<double>(empty_count);
        for (uint32_t idx = 0; idx < MaxWidth; ++idx) {
            if (idx >= width) {
                break;
            }
            dst[idx] = static_cast<T>(success_probability[idx] * inv_empty);
        }
    }
}

template <typename T>
void finish_vector_batch_rows_exad(
    EXAD::SolvedLayer<T> &current,
    size_t current_slot,
    const uint64_t *local_rows,
    uint32_t board_count,
    uint32_t width,
    double spawn_rate4,
    T zero_val,
    EXADVectorBatchWorkspace<T> &batch_workspace
) {
    if (width <= 8U) {
        finish_vector_batch_rows_bounded<8U>(
            current, current_slot, local_rows, board_count, width, spawn_rate4, zero_val, batch_workspace
        );
    } else if (width <= 16U) {
        finish_vector_batch_rows_bounded<16U>(
            current, current_slot, local_rows, board_count, width, spawn_rate4, zero_val, batch_workspace
        );
    } else if (width <= 32U) {
        finish_vector_batch_rows_bounded<32U>(
            current, current_slot, local_rows, board_count, width, spawn_rate4, zero_val, batch_workspace
        );
    } else if (width <= 64U) {
        finish_vector_batch_rows_bounded<64U>(
            current, current_slot, local_rows, board_count, width, spawn_rate4, zero_val, batch_workspace
        );
    } else {
        finish_vector_batch_rows_bounded<128U>(
            current, current_slot, local_rows, board_count, width, spawn_rate4, zero_val, batch_workspace
        );
    }
}

template <typename T>
void recalculate_exad_vector_batch(
    EXAD::SolvedLayer<T> &current,
    size_t current_slot,
    const uint64_t *boards,
    const uint64_t *local_rows,
    uint32_t board_count,
    const EXAD::SolvedLayer<T> &future1,
    const EXAD::SolvedLayer<T> &future2,
    const EXAD::Luts &luts,
    const FormationAD::TilesCombinationTable &tiles_table,
    const FormationAD::PermutationTable &permutation_table,
    const AdvancedMaskParam &param,
    uint32_t original_board_sum,
    const AdvancedPatternSpec &spec,
    int8_t count_32k,
    T max_scale,
    T zero_val,
    double spawn_rate4,
    MatchCache &match_cache,
    AdSolveWorkspace<T> &ad_workspace,
    uint32_t width,
    EXADVectorBatchWorkspace<T> &batch_workspace
) {
    if (board_count == 0U) {
        return;
    }
    batch_workspace.prepare(width, board_count, zero_val);

    for (uint32_t board_slot = 0; board_slot < board_count; ++board_slot) {
        const uint64_t board = boards[board_slot];
        auto [rep_t, rep_v] = replace_val(board);
        uint32_t empty_mask = empty_cell_mask16(board);
        batch_workspace.empty_masks[board_slot] = static_cast<uint16_t>(empty_mask);
        while (empty_mask != 0U) {
            const int pos = static_cast<int>(ctz_u32(empty_mask));
            empty_mask &= empty_mask - 1U;
            const uint16_t ref = static_cast<uint16_t>(board_slot * 16U + static_cast<uint32_t>(pos));
            collect_vector_spawn_queries_exad(
                board,
                1ULL,
                pos,
                rep_t,
                rep_v,
                spec,
                count_32k,
                future1,
                luts,
                original_board_sum + 2U,
                match_cache,
                tiles_table,
                permutation_table,
                param,
                zero_val,
                max_scale,
                ad_workspace,
                ref,
                width,
                batch_workspace.queries1,
                batch_workspace.columns1,
                batch_workspace.best2.data()
            );
            collect_vector_spawn_queries_exad(
                board,
                2ULL,
                pos,
                rep_t,
                rep_v,
                spec,
                count_32k,
                future2,
                luts,
                original_board_sum + 4U,
                match_cache,
                tiles_table,
                permutation_table,
                param,
                zero_val,
                max_scale,
                ad_workspace,
                ref,
                width,
                batch_workspace.queries2,
                batch_workspace.columns2,
                batch_workspace.best4.data()
            );
        }
    }

    lookup_reduce_vector_queries_by_width_exad(
        future1,
        batch_workspace.queries1,
        batch_workspace.columns1,
        width,
        batch_workspace.best2.data()
    );
    lookup_reduce_vector_queries_by_width_exad(
        future2,
        batch_workspace.queries2,
        batch_workspace.columns2,
        width,
        batch_workspace.best4.data()
    );
    finish_vector_batch_rows_exad(
        current,
        current_slot,
        local_rows,
        board_count,
        width,
        spawn_rate4,
        zero_val,
        batch_workspace
    );
}

template <typename T>
void recalculate_exad_scalar_batch_single_future(
    EXAD::SolvedLayer<T> &current,
    size_t current_slot,
    const uint64_t *boards,
    const uint64_t *local_rows,
    uint32_t board_count,
    const EXAD::SolvedLayer<T> &future,
    const EXAD::Luts &luts,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    uint32_t board_sum_after_spawn,
    const AdvancedPatternSpec &spec,
    int8_t count_32k,
    uint64_t new_value,
    double contribution_weight,
    T max_scale,
    T zero_val,
    EXADScalarBatchWorkspace<T> &workspace
) {
    if (board_count == 0U) {
        return;
    }
    workspace.queries1.clear();
    workspace.canonical_candidates1.clear();
    workspace.candidate_refs1.clear();
    const size_t best_slots = static_cast<size_t>(board_count) * 16U;
    std::fill(workspace.best2.begin(), workspace.best2.begin() + best_slots, zero_val);

    for (uint32_t board_slot = 0; board_slot < board_count; ++board_slot) {
        const uint64_t board = boards[board_slot];
        uint32_t empty_mask = empty_cell_mask16(board);
        workspace.empty_masks[board_slot] = static_cast<uint16_t>(empty_mask);
        while (empty_mask != 0U) {
            const int pos = static_cast<int>(ctz_u32(empty_mask));
            empty_mask &= empty_mask - 1U;
            const uint16_t ref = static_cast<uint16_t>(board_slot * 16U + static_cast<uint32_t>(pos));
            collect_scalar_spawn_queries_exad(
                board,
                new_value,
                pos,
                board_sum_after_spawn,
                spec,
                count_32k,
                future,
                luts,
                tiles_table,
                param,
                max_scale,
                ref,
                workspace.queries1,
                workspace.canonical_candidates1,
                workspace.candidate_refs1,
                workspace.best2.data()
            );
        }
    }

    if (!workspace.canonical_candidates1.empty()) {
        CanonicalBatch::canonicalize_inplace(
            workspace.canonical_candidates1.data(),
            workspace.canonical_candidates1.size(),
            spec.symm_mode
        );
        for (size_t i = 0; i < workspace.canonical_candidates1.size(); ++i) {
            EXAD::PreparedQuery query;
            if (EXAD::prepare_query(
                    luts,
                    count_32k,
                    workspace.canonical_candidates1[i],
                    workspace.candidate_refs1[i],
                    0U,
                    query)) {
                workspace.queries1.push_back(query);
            }
        }
    }
    EXAD::lookup_reduce_prepared_queries(
        future,
        workspace.queries1.data(),
        static_cast<uint32_t>(workspace.queries1.size()),
        workspace.best2.data()
    );

    for (uint32_t board_slot = 0; board_slot < board_count; ++board_slot) {
        uint32_t empty_count = 0U;
        double contribution = 0.0;
        uint32_t empty_mask = workspace.empty_masks[board_slot];
        while (empty_mask != 0U) {
            const uint32_t pos = ctz_u32(empty_mask);
            empty_mask &= empty_mask - 1U;
            const size_t best_index = static_cast<size_t>(board_slot) * 16U + pos;
            contribution += static_cast<double>(workspace.best2[best_index]) * contribution_weight;
            ++empty_count;
        }
        T *dst = EXAD::row_ptr(current, current_slot, local_rows[board_slot]);
        if (empty_count == 0U) {
            dst[0] = zero_val;
        } else {
            dst[0] = static_cast<T>(
                static_cast<double>(dst[0]) + contribution / static_cast<double>(empty_count)
            );
        }
    }
}

template <typename T>
void add_vector_batch_rows_single_future(
    EXAD::SolvedLayer<T> &current,
    size_t current_slot,
    const uint64_t *local_rows,
    uint32_t board_count,
    uint32_t width,
    double contribution_weight,
    T zero_val,
    EXADVectorBatchWorkspace<T> &batch_workspace
) {
    for (uint32_t board_slot = 0; board_slot < board_count; ++board_slot) {
        uint32_t empty_count = 0U;
        uint32_t empty_mask = batch_workspace.empty_masks[board_slot];
        T *dst = EXAD::row_ptr(current, current_slot, local_rows[board_slot]);
        if (empty_mask == 0U) {
            std::fill(dst, dst + width, zero_val);
            continue;
        }
        std::fill(
            batch_workspace.success_probability.begin(),
            batch_workspace.success_probability.end(),
            0.0
        );
        while (empty_mask != 0U) {
            const uint32_t pos = ctz_u32(empty_mask);
            empty_mask &= empty_mask - 1U;
            const size_t best_index = (static_cast<size_t>(board_slot) * 16U + pos) * width;
            for (uint32_t idx = 0; idx < width; ++idx) {
                batch_workspace.success_probability[idx] +=
                    static_cast<double>(batch_workspace.best2[best_index + idx]) * contribution_weight;
            }
            ++empty_count;
        }
        const double inv_empty = 1.0 / static_cast<double>(empty_count);
        for (uint32_t idx = 0; idx < width; ++idx) {
            dst[idx] = static_cast<T>(
                static_cast<double>(dst[idx]) + batch_workspace.success_probability[idx] * inv_empty
            );
        }
    }
}

template <typename T>
void recalculate_exad_vector_batch_single_future(
    EXAD::SolvedLayer<T> &current,
    size_t current_slot,
    const uint64_t *boards,
    const uint64_t *local_rows,
    uint32_t board_count,
    const EXAD::SolvedLayer<T> &future,
    const EXAD::Luts &luts,
    const FormationAD::TilesCombinationTable &tiles_table,
    const FormationAD::PermutationTable &permutation_table,
    const AdvancedMaskParam &param,
    uint32_t board_sum_after_spawn,
    const AdvancedPatternSpec &spec,
    int8_t count_32k,
    uint64_t new_value,
    double contribution_weight,
    T max_scale,
    T zero_val,
    MatchCache &match_cache,
    AdSolveWorkspace<T> &ad_workspace,
    uint32_t width,
    EXADVectorBatchWorkspace<T> &batch_workspace
) {
    if (board_count == 0U) {
        return;
    }
    batch_workspace.prepare(width, board_count, zero_val);

    for (uint32_t board_slot = 0; board_slot < board_count; ++board_slot) {
        const uint64_t board = boards[board_slot];
        auto [rep_t, rep_v] = replace_val(board);
        uint32_t empty_mask = empty_cell_mask16(board);
        batch_workspace.empty_masks[board_slot] = static_cast<uint16_t>(empty_mask);
        while (empty_mask != 0U) {
            const int pos = static_cast<int>(ctz_u32(empty_mask));
            empty_mask &= empty_mask - 1U;
            const uint16_t ref = static_cast<uint16_t>(board_slot * 16U + static_cast<uint32_t>(pos));
            collect_vector_spawn_queries_exad(
                board,
                new_value,
                pos,
                rep_t,
                rep_v,
                spec,
                count_32k,
                future,
                luts,
                board_sum_after_spawn,
                match_cache,
                tiles_table,
                permutation_table,
                param,
                zero_val,
                max_scale,
                ad_workspace,
                ref,
                width,
                batch_workspace.queries1,
                batch_workspace.columns1,
                batch_workspace.best2.data()
            );
        }
    }

    lookup_reduce_vector_queries_by_width_exad(
        future,
        batch_workspace.queries1,
        batch_workspace.columns1,
        width,
        batch_workspace.best2.data()
    );
    add_vector_batch_rows_single_future(
        current,
        current_slot,
        local_rows,
        board_count,
        width,
        contribution_weight,
        zero_val,
        batch_workspace
    );
}

template <typename T>
void recalculate_exad_direct_single_future(
    EXAD::SolvedLayer<T> &current,
    const EXAD::SolvedLayer<T> &future,
    uint64_t new_value,
    double contribution_weight,
    const EXAD::Luts &luts,
    const FormationAD::TilesCombinationTable &tiles_table,
    const FormationAD::PermutationTable &permutation_table,
    const AdvancedMaskParam &param,
    uint32_t board_sum_after_spawn,
    const AdvancedPatternSpec &spec,
    T max_scale,
    T zero_val,
    bool do_check,
    int target,
    int num_threads,
    std::unordered_map<uint32_t, MatchCache> &match_dict
) {
    for (int key = bucket_key_min(); key <= bucket_key_max(); ++key) {
        const size_t slot = bucket_to_index(key);
        const EXAD::BoardSet &set = current.sets[slot];
        const size_t derive_size = current.row_width[slot];
        if (set.live_board_count == 0 || derive_size == 0) {
            continue;
        }

        MatchCache disabled_match_cache;
        MatchCache *match_cache = nullptr;
        if (derive_size != 1U) {
            match_cache = exad_match_cache_enabled(derive_size)
                ? &get_exad_match_cache<T>(match_dict, derive_size)
                : &disabled_match_cache;
        }
        std::vector<AdSolveWorkspace<T>> thread_workspaces(static_cast<size_t>(num_threads));
        std::vector<EXADScalarBatchWorkspace<T>> scalar_workspaces;
        std::vector<EXADVectorBatchWorkspace<T>> vector_workspaces;
        const bool use_vector_batch = exad_vector_batch_enabled(derive_size);
        const uint32_t vector_batch_size = use_vector_batch ? exad_vector_batch_size(derive_size) : 0U;
        if (derive_size == 1U) {
            scalar_workspaces.resize(static_cast<size_t>(num_threads));
        } else if (use_vector_batch) {
            vector_workspaces.resize(static_cast<size_t>(num_threads));
        }
        const int chunk_count = std::max(
            std::min(
                1024,
                static_cast<int>(
                    (set.buckets.size() * static_cast<size_t>(std::llround(std::log2(static_cast<double>(derive_size + 1U)))))
                    / 64ULL
                )
            ),
            1
        );
        const int schedule_chunk = std::max(1, static_cast<int>(set.buckets.size() / static_cast<size_t>(chunk_count + 1)));

#pragma omp parallel for schedule(dynamic, schedule_chunk) num_threads(num_threads)
        for (int64_t bucket_i = 0; bucket_i < static_cast<int64_t>(set.buckets.size()); ++bucket_i) {
            const size_t thread_index = static_cast<size_t>(omp_get_thread_num());
            AdSolveWorkspace<T> &workspace = thread_workspaces[thread_index];
            const EXAD::BucketEntry &bucket = set.buckets[static_cast<size_t>(bucket_i)];
            const uint64_t prefix36 = EXAD::bucket_key_prefix36(bucket.key);
            const uint32_t group = EXAD::lut_group_index(EXAD::bucket_key_semantic_sum(bucket.key));
            const uint32_t valid_count = luts.size_table[group];
            const uint32_t unrank_base = luts.offset_table[group];
            uint32_t ordinal = 0;
            std::array<uint64_t, kEXADScalarBatchSize> scalar_boards{};
            std::array<uint64_t, kEXADScalarBatchSize> scalar_local_rows{};
            uint32_t scalar_count = 0U;
            std::array<uint64_t, kEXADVectorMaxBatchSize> vector_boards{};
            std::array<uint64_t, kEXADVectorMaxBatchSize> vector_local_rows{};
            uint32_t vector_count = 0U;

            auto flush_scalar = [&]() {
                if (scalar_count == 0U) {
                    return;
                }
                recalculate_exad_scalar_batch_single_future(
                    current,
                    slot,
                    scalar_boards.data(),
                    scalar_local_rows.data(),
                    scalar_count,
                    future,
                    luts,
                    tiles_table,
                    param,
                    board_sum_after_spawn,
                    spec,
                    static_cast<int8_t>(key),
                    new_value,
                    contribution_weight,
                    max_scale,
                    zero_val,
                    scalar_workspaces[thread_index]
                );
                scalar_count = 0U;
            };

            auto flush_vector = [&]() {
                if (vector_count == 0U) {
                    return;
                }
                recalculate_exad_vector_batch_single_future(
                    current,
                    slot,
                    vector_boards.data(),
                    vector_local_rows.data(),
                    vector_count,
                    future,
                    luts,
                    tiles_table,
                    permutation_table,
                    param,
                    board_sum_after_spawn,
                    spec,
                    static_cast<int8_t>(key),
                    new_value,
                    contribution_weight,
                    max_scale,
                    zero_val,
                    *match_cache,
                    workspace,
                    static_cast<uint32_t>(derive_size),
                    vector_workspaces[thread_index]
                );
                vector_count = 0U;
            };

            auto process_rank = [&](uint32_t rank) {
                const uint64_t board = (prefix36 << EXAD::kSuffixBits) | luts.unrank_array[unrank_base + rank];
                const uint64_t local_row = static_cast<uint64_t>(bucket.dense_offset) + ordinal;
                ++ordinal;
                T *dst = EXAD::row_ptr(current, slot, local_row);
                if (do_check && is_success_by_shifts(board, target, spec.success_shifts)) {
                    std::fill(dst, dst + derive_size, max_scale);
                    return;
                }
                const uint32_t empty_mask_for_row = empty_cell_mask16(board);
                if (empty_mask_for_row == 0U) {
                    std::fill(dst, dst + derive_size, zero_val);
                    return;
                }
                if (derive_size == 1U) {
                    scalar_boards[scalar_count] = board;
                    scalar_local_rows[scalar_count] = local_row;
                    ++scalar_count;
                    if (scalar_count == kEXADScalarBatchSize) {
                        flush_scalar();
                    }
                    return;
                }
                if (use_vector_batch) {
                    vector_boards[vector_count] = board;
                    vector_local_rows[vector_count] = local_row;
                    ++vector_count;
                    if (vector_count == vector_batch_size) {
                        flush_vector();
                    }
                    return;
                }
            };

            auto process_rank_fallback = [&](uint32_t rank) {
                const uint64_t board = (prefix36 << EXAD::kSuffixBits) | luts.unrank_array[unrank_base + rank];
                const uint64_t local_row = static_cast<uint64_t>(bucket.dense_offset) + ordinal;
                ++ordinal;
                T *dst = EXAD::row_ptr(current, slot, local_row);
                if (do_check && is_success_by_shifts(board, target, spec.success_shifts)) {
                    std::fill(dst, dst + derive_size, max_scale);
                    return;
                }
                auto [rep_t, rep_v] = replace_val(board);
                uint64_t rep_t_rev = FormationAD::reverse(rep_t);
                uint32_t empty_mask = empty_cell_mask16(board);
                const int empty_slots = static_cast<int>(popcount_u32(empty_mask));
                if (empty_slots == 0) {
                    std::fill(dst, dst + derive_size, zero_val);
                    return;
                }
                while (empty_mask != 0U) {
                    const int pos = static_cast<int>(ctz_u32(empty_mask));
                    empty_mask &= empty_mask - 1U;
                    solve_optimal_success_rate_arr_into_exad(
                        board, new_value, pos, rep_t, rep_t_rev, derive_size, spec, rep_v,
                        static_cast<int8_t>(key), future, luts, board_sum_after_spawn,
                        *match_cache, tiles_table, permutation_table, param, zero_val, max_scale,
                        workspace, workspace.optimal_values
                    );
                    const double factor = contribution_weight / static_cast<double>(empty_slots);
                    for (size_t idx = 0; idx < derive_size; ++idx) {
                        dst[idx] = static_cast<T>(
                            static_cast<double>(dst[idx]) +
                            static_cast<double>(workspace.optimal_values[idx]) * factor
                        );
                    }
                }
            };

            auto process_live_rank = [&](uint32_t rank) {
                if (derive_size == 1U || use_vector_batch) {
                    process_rank(rank);
                } else {
                    process_rank_fallback(rank);
                }
            };

            if (valid_count <= set.threshold_bits) {
                const uint32_t bytes = static_cast<uint32_t>(ZMaskFrozen::bytes_for_bits(valid_count));
                for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                    uint8_t value = set.small_bitmap_bytes[bucket.bitmap_offset + byte_idx];
                    while (value != 0U) {
#if defined(__GNUC__) || defined(__clang__)
                        const uint32_t bit = static_cast<uint32_t>(__builtin_ctz(value));
#else
                        uint32_t bit = 0;
                        while (((value >> bit) & 1U) == 0U) {
                            ++bit;
                        }
#endif
                        const uint32_t rank = byte_idx * 8U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        process_live_rank(rank);
                        value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                    }
                }
            } else {
                const uint32_t words = static_cast<uint32_t>(ZMaskFrozen::words_for_bits(valid_count));
                for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
                    uint64_t value = set.large_bitmap_words[bucket.bitmap_offset + word_idx];
                    while (value != 0ULL) {
#if defined(__GNUC__) || defined(__clang__)
                        const uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(value));
#else
                        uint32_t bit = 0;
                        while (((value >> bit) & 1ULL) == 0ULL) {
                            ++bit;
                        }
#endif
                        const uint32_t rank = word_idx * 64U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        process_live_rank(rank);
                        value &= value - 1ULL;
                    }
                }
            }
            flush_scalar();
            flush_vector();
        }
    }
}

template <typename T>
void recalculate_exad_direct(
    EXAD::SolvedLayer<T> &current,
    const EXAD::SolvedLayer<T> &future1,
    const EXAD::SolvedLayer<T> &future2,
    const EXAD::Luts &luts,
    const FormationAD::TilesCombinationTable &tiles_table,
    const FormationAD::PermutationTable &permutation_table,
    const AdvancedMaskParam &param,
    uint32_t original_board_sum,
    const AdvancedPatternSpec &spec,
    T max_scale,
    T zero_val,
    double spawn_rate4,
    bool do_check,
    int target,
    int num_threads,
    std::unordered_map<uint32_t, MatchCache> &match_dict
) {
    for (int key = bucket_key_min(); key <= bucket_key_max(); ++key) {
        const size_t slot = bucket_to_index(key);
        const EXAD::BoardSet &set = current.sets[slot];
        const size_t derive_size = current.row_width[slot];
        if (set.live_board_count == 0 || derive_size == 0) {
            continue;
        }

        MatchCache disabled_match_cache;
        MatchCache *match_cache = nullptr;
        if (derive_size != 1U) {
            match_cache = exad_match_cache_enabled(derive_size)
                ? &get_exad_match_cache<T>(match_dict, derive_size)
                : &disabled_match_cache;
        }
        std::vector<AdSolveWorkspace<T>> thread_workspaces(static_cast<size_t>(num_threads));
        std::vector<EXADScalarBatchWorkspace<T>> scalar_workspaces;
        std::vector<EXADVectorBatchWorkspace<T>> vector_workspaces;
        const bool use_vector_batch = exad_vector_batch_enabled(derive_size);
        const uint32_t vector_batch_size = use_vector_batch ? exad_vector_batch_size(derive_size) : 0U;
        if (derive_size == 1U) {
            scalar_workspaces.resize(static_cast<size_t>(num_threads));
        } else if (use_vector_batch) {
            vector_workspaces.resize(static_cast<size_t>(num_threads));
        }
        const int chunk_count = std::max(
            std::min(
                1024,
                static_cast<int>(
                    (set.buckets.size() * static_cast<size_t>(std::llround(std::log2(static_cast<double>(derive_size + 1U)))))
                    / 64ULL
                )
            ),
            1
        );
        const int schedule_chunk = std::max(1, static_cast<int>(set.buckets.size() / static_cast<size_t>(chunk_count + 1)));

#pragma omp parallel for schedule(dynamic, schedule_chunk) num_threads(num_threads)
        for (int64_t bucket_i = 0; bucket_i < static_cast<int64_t>(set.buckets.size()); ++bucket_i) {
            const size_t thread_index = static_cast<size_t>(omp_get_thread_num());
            AdSolveWorkspace<T> &workspace = thread_workspaces[thread_index];
            const EXAD::BucketEntry &bucket = set.buckets[static_cast<size_t>(bucket_i)];
            const uint64_t prefix36 = EXAD::bucket_key_prefix36(bucket.key);
            const uint32_t group = EXAD::lut_group_index(EXAD::bucket_key_semantic_sum(bucket.key));
            const uint32_t valid_count = luts.size_table[group];
            const uint32_t unrank_base = luts.offset_table[group];
            uint32_t ordinal = 0;
            std::array<uint64_t, kEXADScalarBatchSize> scalar_boards{};
            std::array<uint64_t, kEXADScalarBatchSize> scalar_local_rows{};
            uint32_t scalar_count = 0U;
            std::array<uint64_t, kEXADVectorMaxBatchSize> vector_boards{};
            std::array<uint64_t, kEXADVectorMaxBatchSize> vector_local_rows{};
            uint32_t vector_count = 0U;

            auto flush_scalar = [&]() {
                if (scalar_count == 0U) {
                    return;
                }
                recalculate_exad_scalar_batch(
                    current,
                    slot,
                    scalar_boards.data(),
                    scalar_local_rows.data(),
                    scalar_count,
                    future1,
                    future2,
                    luts,
                    tiles_table,
                    param,
                    original_board_sum,
                    spec,
                    static_cast<int8_t>(key),
                    max_scale,
                    zero_val,
                    spawn_rate4,
                    scalar_workspaces[thread_index]
                );
                scalar_count = 0U;
            };

            auto flush_vector = [&]() {
                if (vector_count == 0U) {
                    return;
                }
                recalculate_exad_vector_batch(
                    current,
                    slot,
                    vector_boards.data(),
                    vector_local_rows.data(),
                    vector_count,
                    future1,
                    future2,
                    luts,
                    tiles_table,
                    permutation_table,
                    param,
                    original_board_sum,
                    spec,
                    static_cast<int8_t>(key),
                    max_scale,
                    zero_val,
                    spawn_rate4,
                    *match_cache,
                    workspace,
                    static_cast<uint32_t>(derive_size),
                    vector_workspaces[thread_index]
                );
                vector_count = 0U;
            };

            auto process_rank = [&](uint32_t rank) {
                const uint64_t board = (prefix36 << EXAD::kSuffixBits) | luts.unrank_array[unrank_base + rank];
                const uint64_t local_row = static_cast<uint64_t>(bucket.dense_offset) + ordinal;
                ++ordinal;
                if (do_check && is_success_by_shifts(board, target, spec.success_shifts)) {
                    T *dst = EXAD::row_ptr(current, slot, local_row);
                    std::fill(dst, dst + derive_size, max_scale);
                    return;
                }
                if (derive_size == 1U) {
                    scalar_boards[scalar_count] = board;
                    scalar_local_rows[scalar_count] = local_row;
                    ++scalar_count;
                    if (scalar_count == kEXADScalarBatchSize) {
                        flush_scalar();
                    }
                    return;
                }
                if (use_vector_batch) {
                    vector_boards[vector_count] = board;
                    vector_local_rows[vector_count] = local_row;
                    ++vector_count;
                    if (vector_count == vector_batch_size) {
                        flush_vector();
                    }
                    return;
                }

                T *dst = EXAD::row_ptr(current, slot, local_row);
                auto [rep_t, rep_v] = replace_val(board);
                uint64_t rep_t_rev = FormationAD::reverse(rep_t);
                auto &success_probability = workspace.success_probability;
                success_probability.assign(derive_size, 0.0);
                uint32_t empty_mask = empty_cell_mask16(board);
                const int empty_slots = static_cast<int>(popcount_u32(empty_mask));
                while (empty_mask != 0U) {
                    const int pos = static_cast<int>(ctz_u32(empty_mask));
                    empty_mask &= empty_mask - 1U;
                    solve_optimal_success_rate_arr_into_exad(
                        board, 1ULL, pos, rep_t, rep_t_rev, derive_size, spec, rep_v,
                        static_cast<int8_t>(key), future1, luts, original_board_sum + 2U,
                        *match_cache, tiles_table, permutation_table, param, zero_val, max_scale,
                        workspace, workspace.optimal_values
                    );
                    for (size_t idx = 0; idx < derive_size; ++idx) {
                        success_probability[idx] += static_cast<double>(workspace.optimal_values[idx]) * (1.0 - spawn_rate4);
                    }
                    solve_optimal_success_rate_arr_into_exad(
                        board, 2ULL, pos, rep_t, rep_t_rev, derive_size, spec, rep_v,
                        static_cast<int8_t>(key), future2, luts, original_board_sum + 4U,
                        *match_cache, tiles_table, permutation_table, param, zero_val, max_scale,
                        workspace, workspace.temp_values
                    );
                    for (size_t idx = 0; idx < derive_size; ++idx) {
                        success_probability[idx] += static_cast<double>(workspace.temp_values[idx]) * spawn_rate4;
                    }
                }
                for (size_t idx = 0; idx < derive_size; ++idx) {
                    dst[idx] = empty_slots == 0
                        ? zero_val
                        : static_cast<T>(success_probability[idx] / static_cast<double>(empty_slots));
                }
            };

            if (valid_count <= set.threshold_bits) {
                const uint32_t bytes = static_cast<uint32_t>(ZMaskFrozen::bytes_for_bits(valid_count));
                for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                    uint8_t value = set.small_bitmap_bytes[bucket.bitmap_offset + byte_idx];
                    while (value != 0U) {
#if defined(__GNUC__) || defined(__clang__)
                        const uint32_t bit = static_cast<uint32_t>(__builtin_ctz(value));
#else
                        uint32_t bit = 0;
                        while (((value >> bit) & 1U) == 0U) {
                            ++bit;
                        }
#endif
                        const uint32_t rank = byte_idx * 8U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        process_rank(rank);
                        value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                    }
                }
            } else {
                const uint32_t words = static_cast<uint32_t>(ZMaskFrozen::words_for_bits(valid_count));
                for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
                    uint64_t value = set.large_bitmap_words[bucket.bitmap_offset + word_idx];
                    while (value != 0ULL) {
#if defined(__GNUC__) || defined(__clang__)
                        const uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(value));
#else
                        uint32_t bit = 0;
                        while (((value >> bit) & 1ULL) == 0ULL) {
                            ++bit;
                        }
#endif
                        const uint32_t rank = word_idx * 64U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        process_rank(rank);
                        value &= value - 1ULL;
                    }
                }
            }
            flush_scalar();
            flush_vector();
        }
    }
}

template <typename T>
EXAD::SolvedLayer<T> empty_future_layer(EXAD::DTypeMode mode) {
    EXAD::SolvedLayer<T> layer;
    layer.dtype_mode = mode;
    return layer;
}

template <typename T>
bool solved_physical_metadata_matches(const EXAD::SolvedLayer<T> &layer, const EXAD::Luts &luts) {
    return layer.physical_transform == luts.physical_transform &&
        layer.inverse_physical_transform == luts.inverse_physical_transform &&
        layer.logical_pattern_signature == luts.logical_pattern_signature &&
        layer.physical_pattern_signature == luts.physical_pattern_signature;
}

void remove_exad_temp_layers(const RunOptions &options) {
    for (int step = 0; step < options.steps; ++step) {
        EXAD::remove_layer_file(EXAD::layer_file_path(options.pathname, step));
    }
}

template <typename T>
void load_future_layer(
    const RunOptions &options,
    int target_step,
    EXAD::DTypeMode mode,
    const EXAD::Luts &luts,
    EXAD::SolvedLayer<T> &layer,
    int &cached_step,
    FileIOUtils::DirectIoConfig io_config,
    double &read_seconds,
    double &index_seconds
) {
    if (cached_step == target_step) {
        return;
    }
    if (target_step >= options.steps - 2) {
        layer = empty_future_layer<T>(mode);
        cached_step = target_step;
        return;
    }
    const std::string path = EXAD::solved_file_path(options.pathname, target_step);
    const std::string compressed_path = exad_compressed_file_path(options, target_step);
    if (!EXAD::solved_file_exists(path) && !NativePath::exists(compressed_path)) {
        throw std::runtime_error("missing EXAD solved future layer: " + path);
    }
    const double read_t0 = wall_time_seconds();
    if (EXAD::solved_file_exists(path)) {
        layer = EXAD::read_solved_layer_file<T>(path, mode, io_config);
    } else {
        layer = EXADCompressedResult::read_exad_compressed_layer<T>(compressed_path, mode, luts);
    }
    read_seconds += wall_time_seconds() - read_t0;
    if (!solved_physical_metadata_matches(layer, luts)) {
        throw std::runtime_error("EXAD solved layer physical metadata does not match LUT: " + path);
    }
    const double index_t0 = wall_time_seconds();
    EXAD::build_direct_indexes(layer, luts);
    index_seconds += wall_time_seconds() - index_t0;
    cached_step = target_step;
}

template <typename T>
void recalculate_process_exad_impl(
    const std::vector<uint64_t> &arr_init,
    const AdvancedPatternSpec &spec,
    const RunOptions &options
) {
    ensure_exad_solve_stats_header(options);
    EXADSolveStatsRecord total_record;
    total_record.stage = "_total";
    RuntimeControls::DeletionThresholdState deletion_threshold_state =
        RuntimeControls::current_deletion_thresholds(options);
    total_record.deletion_threshold = deletion_threshold_state.absolute;
    total_record.relative_deletion_threshold = deletion_threshold_state.relative;
    uint64_t total_bitmap_live = 0;
    uint64_t total_bitmap_bits = 0;

    const int num_threads = effective_num_threads(options);
    const FileIOUtils::DirectIoConfig io_config = FileIOUtils::direct_io_config_from_options(options);
    FormationAD::MaskerContext masker = FormationAD::init_masker(spec);
    const AdvancedMaskParam param = masker.param;
    EXAD::Luts luts = exad_load_or_build_luts(arr_init, spec, options, num_threads, io_config);
    const uint32_t ini_board_sum = arr_init.empty() ? 0U : board_sum(arr_init.front());
    const T max_scale = max_scale_value_for_dtype<T>(options.success_rate_dtype);
    const T zero_val = zero_value_for_dtype<T>(options.success_rate_dtype);
    const EXAD::DTypeMode dtype_mode = EXAD::dtype_mode_from_name(options.success_rate_dtype);

    EXAD::SolvedLayer<T> future1 = empty_future_layer<T>(dtype_mode);
    EXAD::SolvedLayer<T> future2 = empty_future_layer<T>(dtype_mode);
    int cached_future1_step = std::numeric_limits<int>::min();
    int cached_future2_step = std::numeric_limits<int>::min();
    std::unordered_map<uint32_t, MatchCache> match_dict;
    const uint32_t progress_total = build_progress_total(options);
    const EXADSolvePlan solve_plan = make_exad_solve_plan(options);

    for (int step = solve_plan.first_step; step >= 0; --step) {
        FormationProgress::update_build_progress(
            progress_total - static_cast<uint32_t>(step) - 2U,
            progress_total
        );
        const std::string solved_path = EXAD::solved_file_path(options.pathname, step);
        if (EXAD::solved_file_exists(solved_path)) {
            cleanup_exad_chunk_work_files(options, step);
            continue;
        }
        if (exad_compressed_file_exists(options, step)) {
            cleanup_exad_chunk_work_files(options, step);
            continue;
        }
        deletion_threshold_state =
            RuntimeControls::refresh_deletion_thresholds(options, deletion_threshold_state);
        const double layer_deletion_threshold = deletion_threshold_state.absolute;

        const double current_read_t0 = wall_time_seconds();
        const std::string current_temp_path = EXAD::layer_file_path(options.pathname, step);
        if (!EXAD::layer_file_exists(current_temp_path)) {
            ensure_exad_temp_through_cpp(arr_init, spec, options, step);
        }
        if (!EXAD::layer_file_exists(current_temp_path)) {
            throw std::runtime_error(
                "missing EXAD temp layer for local solve resume at step " + std::to_string(step)
            );
        }
        EXAD::Layer generation_layer = EXAD::read_layer_file(current_temp_path, io_config);
        const double current_read_t1 = wall_time_seconds();
        if (!EXAD::physical_metadata_matches(generation_layer, luts)) {
            throw std::runtime_error("EXAD temp layer physical metadata does not match LUT");
        }
        const double build_t0 = wall_time_seconds();
        EXAD::SolvedLayer<T> current = EXAD::make_solved_layer_from_generation<T>(
            std::move(generation_layer),
            param,
            dtype_mode,
            zero_val
        );
        EXAD::fill_success_values(current.success_values, zero_val, num_threads);
        const double build_t1 = wall_time_seconds();

        double future_read_seconds = 0.0;
        double future_index_seconds = 0.0;
        load_future_layer(
            options, step + 1, dtype_mode, luts, future1, cached_future1_step, io_config,
            future_read_seconds, future_index_seconds
        );
        load_future_layer(
            options, step + 2, dtype_mode, luts, future2, cached_future2_step, io_config,
            future_read_seconds, future_index_seconds
        );

        const double recalc_t0 = wall_time_seconds();
        recalculate_exad_direct(
            current,
            future1,
            future2,
            luts,
            masker.tiles_combination_table,
            masker.permutation_table,
            param,
            static_cast<uint32_t>(2 * step) + ini_board_sum,
            spec,
            max_scale,
            zero_val,
            options.spawn_rate4,
            step > options.docheck_step,
            options.target,
            num_threads,
            match_dict
        );
        const double recalc_t1 = wall_time_seconds();

        auto [input_values, max_rate] = EXAD::solved_value_count_and_max(current, zero_val);
        const uint64_t input_rows = current.live_board_count;
        const double normalized_max_rate = static_cast<double>(max_rate - zero_val) /
            static_cast<double>(max_scale - zero_val);

        double future_compact_seconds = 0.0;
        double future_write_seconds = 0.0;
        double compress_seconds = 0.0;
        uint64_t future_threshold_values_before = static_cast<uint64_t>(future2.success_values.size());
        uint64_t future_threshold_values_after = future_threshold_values_before;
        double effective_deletion_threshold = 0.0;
        if (RuntimeControls::deletion_threshold_enabled(deletion_threshold_state) &&
            cached_future2_step == step + 2 && !future2.empty()) {
            bool should_compact_future = deletion_threshold_state.absolute > 0.0;
            T layer_threshold = RuntimeControls::absolute_deletion_threshold(
                zero_val,
                max_scale,
                deletion_threshold_state
            );
            if (deletion_threshold_state.relative > 0.0) {
                const auto [ignored_future_values, future_max] =
                    EXAD::solved_value_count_and_max(future2, zero_val);
                (void)ignored_future_values;
                if (future_max > zero_val) {
                    const T relative_threshold = RuntimeControls::relative_deletion_threshold(
                        future_max,
                        zero_val,
                        deletion_threshold_state
                    );
                    layer_threshold = RuntimeControls::max_deletion_threshold(
                        layer_threshold,
                        relative_threshold
                    );
                    should_compact_future = true;
                }
            }
            if (should_compact_future) {
                effective_deletion_threshold =
                    RuntimeControls::normalized_deletion_threshold(layer_threshold, zero_val, max_scale);
                const double future_compact_t0 = wall_time_seconds();
                EXAD::SolvedLayer<T> compacted_future =
                    EXAD::compact_solved_layer(future2, luts, layer_threshold, num_threads);
                future_compact_seconds = wall_time_seconds() - future_compact_t0;
                future_threshold_values_after = static_cast<uint64_t>(compacted_future.success_values.size());

                if (options.compress) {
                    compress_seconds += compress_exad_solved_layer_from_memory(
                        options,
                        step + 2,
                        compacted_future,
                        luts
                    );
                } else {
                    const double future_write_t0 = wall_time_seconds();
                    EXAD::write_solved_layer_file(
                        EXAD::solved_file_path(options.pathname, step + 2),
                        compacted_future,
                        io_config
                    );
                    future_write_seconds = wall_time_seconds() - future_write_t0;
                }
                if (cached_future2_step == step + 2) {
                    future2 = std::move(compacted_future);
                    EXAD::build_direct_indexes(future2, luts);
                }
            }
        }

        const double compact_t0 = wall_time_seconds();
        current = EXAD::compact_solved_layer(current, luts, zero_val, num_threads);
        const double compact_t1 = wall_time_seconds();
        auto [post_zero_values, ignored_max] = EXAD::solved_value_count_and_max(current, zero_val);
        (void)ignored_max;
        const uint64_t post_zero_rows = current.live_board_count;

        const double current_index_t0 = wall_time_seconds();
        EXAD::build_direct_indexes(current, luts);
        future_index_seconds += wall_time_seconds() - current_index_t0;

        const double write_t0 = wall_time_seconds();
        EXAD::write_solved_layer_file(solved_path, current, io_config);
        const double write_t1 = wall_time_seconds();
        compress_seconds += maybe_compress_exad_solved_file(options, step);
        EXAD::remove_layer_file(EXAD::layer_file_path(options.pathname, step));

        EXADSolveStatsRecord record;
        record.stage = "solve";
        record.step = step;
        record.input_rows = input_rows;
        record.input_values = input_values;
        record.post_zero_rows = post_zero_rows;
        record.post_zero_values = post_zero_values;
        record.deletion_threshold = layer_deletion_threshold;
        record.relative_deletion_threshold = deletion_threshold_state.relative;
        record.effective_deletion_threshold = effective_deletion_threshold;
        record.current_retained_ratio = RuntimeControls::retention_ratio(record.post_zero_values, record.input_values);
        record.future_threshold_values_before = future_threshold_values_before;
        record.future_threshold_values_after = future_threshold_values_after;
        record.future_threshold_retained_ratio =
            RuntimeControls::retention_ratio(future_threshold_values_after, future_threshold_values_before);
        record.max_success = normalized_max_rate;
        record.current_read_seconds = current_read_t1 - current_read_t0;
        record.current_build_seconds = build_t1 - build_t0;
        record.future_read_seconds = future_read_seconds;
        record.future_index_seconds = future_index_seconds;
        record.recalculate_seconds = recalc_t1 - recalc_t0;
        record.zero_compact_seconds = compact_t1 - compact_t0;
        record.current_write_seconds = write_t1 - write_t0;
        record.future_compact_seconds = future_compact_seconds;
        record.future_write_seconds = future_write_seconds;
        record.compress_seconds = compress_seconds;
        record.metadata_bytes = EXAD::metadata_bytes(current);
        record.success_bytes = static_cast<uint64_t>(current.success_values.size()) * sizeof(T);
        record.bitmap_density = EXAD::bitmap_density(current);
        append_exad_solve_stats_record(options, record);

        total_record.input_rows += record.input_rows;
        total_record.input_values += record.input_values;
        total_record.post_zero_rows += record.post_zero_rows;
        total_record.post_zero_values += record.post_zero_values;
        total_record.current_retained_ratio =
            RuntimeControls::retention_ratio(total_record.post_zero_values, total_record.input_values);
        total_record.future_threshold_values_before += record.future_threshold_values_before;
        total_record.future_threshold_values_after += record.future_threshold_values_after;
        total_record.future_threshold_retained_ratio =
            RuntimeControls::retention_ratio(
                total_record.future_threshold_values_after,
                total_record.future_threshold_values_before
            );
        total_record.max_success = std::max(total_record.max_success, record.max_success);
        total_record.current_read_seconds += record.current_read_seconds;
        total_record.current_build_seconds += record.current_build_seconds;
        total_record.future_read_seconds += record.future_read_seconds;
        total_record.future_index_seconds += record.future_index_seconds;
        total_record.recalculate_seconds += record.recalculate_seconds;
        total_record.zero_compact_seconds += record.zero_compact_seconds;
        total_record.current_write_seconds += record.current_write_seconds;
        total_record.future_compact_seconds += record.future_compact_seconds;
        total_record.future_write_seconds += record.future_write_seconds;
        total_record.compress_seconds += record.compress_seconds;
        total_record.metadata_bytes += record.metadata_bytes;
        total_record.success_bytes += record.success_bytes;
        total_bitmap_live += record.post_zero_rows;
        total_bitmap_bits += solved_layer_bitmap_bits(current);
        total_record.bitmap_density = total_bitmap_bits == 0U
            ? 0.0
            : static_cast<double>(total_bitmap_live) / static_cast<double>(total_bitmap_bits);

        future2 = std::move(future1);
        cached_future2_step = cached_future1_step;
        future1 = std::move(current);
        cached_future1_step = step;
    }

    total_record.deletion_threshold = deletion_threshold_state.absolute;
    total_record.relative_deletion_threshold = deletion_threshold_state.relative;
    future1 = empty_future_layer<T>(dtype_mode);
    future2 = empty_future_layer<T>(dtype_mode);
    match_dict.clear();
    total_record.compress_seconds += compress_all_exad_solved_files(options, true);
    remove_exad_temp_layers(options);
    append_exad_solve_stats_record(options, total_record);
}

template <typename T>
void recalculate_process_exad_chunked_impl(
    const std::vector<uint64_t> &arr_init,
    const AdvancedPatternSpec &spec,
    const RunOptions &options
) {
    ensure_exad_solve_stats_header(options);
    EXADSolveStatsRecord total_record;
    total_record.stage = "_total";
    RuntimeControls::DeletionThresholdState deletion_threshold_state =
        RuntimeControls::current_deletion_thresholds(options);
    total_record.deletion_threshold = deletion_threshold_state.absolute;
    total_record.relative_deletion_threshold = deletion_threshold_state.relative;
    uint64_t total_bitmap_live = 0;
    uint64_t total_bitmap_bits = 0;

    const int num_threads = effective_num_threads(options);
    const FileIOUtils::DirectIoConfig io_config = FileIOUtils::direct_io_config_from_options(options);
    FormationAD::MaskerContext masker = FormationAD::init_masker(spec);
    const AdvancedMaskParam param = masker.param;
    EXAD::Luts luts = exad_load_or_build_luts(arr_init, spec, options, num_threads, io_config);
    const uint32_t ini_board_sum = arr_init.empty() ? 0U : board_sum(arr_init.front());
    const T max_scale = max_scale_value_for_dtype<T>(options.success_rate_dtype);
    const T zero_val = zero_value_for_dtype<T>(options.success_rate_dtype);
    const EXAD::DTypeMode dtype_mode = EXAD::dtype_mode_from_name(options.success_rate_dtype);

    std::unordered_map<uint32_t, MatchCache> match_dict;
    const uint32_t progress_total = build_progress_total(options);
    const EXADSolvePlan solve_plan = make_exad_solve_plan(options);
    const int last_solve_step = options.steps - 3;
    for (int solved_step = 0; solved_step <= last_solve_step; ++solved_step) {
        if (exad_solved_output_exists(options, solved_step)) {
            cleanup_exad_chunk_work_files(options, solved_step);
        }
    }

    auto ensure_future_available = [&](int target_step) {
        if (target_step >= options.steps - 2) {
            return;
        }
        if (!exad_solved_output_exists(options, target_step)) {
            throw std::runtime_error(
                "missing EXAD solved future layer for chunked solve at step " +
                std::to_string(target_step)
            );
        }
    };

    for (int step = solve_plan.first_step; step >= 0; --step) {
        FormationProgress::update_build_progress(
            progress_total - static_cast<uint32_t>(step) - 2U,
            progress_total
        );
        const std::string solved_path = EXAD::solved_file_path(options.pathname, step);
        if (EXAD::solved_file_exists(solved_path)) {
            cleanup_exad_chunk_work_files(options, step);
            continue;
        }
        if (exad_compressed_file_exists(options, step)) {
            cleanup_exad_chunk_work_files(options, step);
            continue;
        }
        deletion_threshold_state =
            RuntimeControls::refresh_deletion_thresholds(options, deletion_threshold_state);
        const double layer_deletion_threshold = deletion_threshold_state.absolute;

        ensure_future_available(step + 1);
        ensure_future_available(step + 2);

        const std::string chunk_dir = exad_chunk_dir_path(options, step);
        const std::string writing_path = exad_chunk_writing_path(options, step);
        std::error_code cleanup_ec;
        NativePath::remove_all(chunk_dir, cleanup_ec);
        if (cleanup_ec) {
            throw std::runtime_error("failed to remove stale EXAD chunk directory: " + chunk_dir);
        }
        NativePath::create_directories(chunk_dir, cleanup_ec);
        if (cleanup_ec) {
            throw std::runtime_error("failed to create EXAD chunk directory: " + chunk_dir);
        }
        NativePath::remove(writing_path, cleanup_ec);
        NativePath::remove(FileIOUtils::temp_write_path(writing_path), cleanup_ec);

        double future_read_seconds = 0.0;
        double future_index_seconds = 0.0;
        double current_read_seconds = 0.0;
        double current_build_seconds = 0.0;
        double recalculate_seconds = 0.0;
        double zero_compact_seconds = 0.0;
        double current_write_seconds = 0.0;
        uint64_t input_rows = 0;
        uint64_t input_values = 0;
        T max_rate = zero_val;
        std::array<EXADSlotChunkManifest, bucket_slot_count()> manifests{};
        std::array<std::vector<std::string>, bucket_slot_count()> partial_chunk_paths{};
        EXAD::LayerFileInfo layer_info{};

        const std::string layer_path = EXAD::layer_file_path(options.pathname, step);
        if (!EXAD::layer_file_exists(layer_path)) {
            ensure_exad_temp_through_cpp(arr_init, spec, options, step);
        }
        if (!EXAD::layer_file_exists(layer_path)) {
            throw std::runtime_error(
                "missing EXAD temp layer for local chunked solve resume at step " + std::to_string(step)
            );
        }
        {
            EXAD::SolvedLayer<T> future = empty_future_layer<T>(dtype_mode);
            int cached_future_step = std::numeric_limits<int>::min();
            load_future_layer(
                options, step + 1, dtype_mode, luts, future, cached_future_step, io_config,
                future_read_seconds, future_index_seconds
            );

            const double open_t0 = wall_time_seconds();
            EXAD::LayerSlotReader reader(layer_path, io_config);
            layer_info = reader.info();
            current_read_seconds += wall_time_seconds() - open_t0;

            for (size_t slot = 0; slot < bucket_slot_count(); ++slot) {
                EXAD::BoardSet set;
                const double slot_read_t0 = wall_time_seconds();
                if (!reader.read_next(set)) {
                    throw std::runtime_error("EXAD temp layer ended before all slots were read: " + layer_path);
                }
                current_read_seconds += wall_time_seconds() - slot_read_t0;

                const uint32_t width = static_cast<uint32_t>(
                    EXAD::solved_derive_size_for_bucket(
                        bucket_key_min() + static_cast<int>(slot),
                        param.num_free_32k
                    )
                );
                const uint64_t row_budget = exad_current_chunk_row_budget<T>(width);
                const std::vector<EXADCurrentChunkRange> ranges =
                    exad_current_chunk_ranges(set, row_budget);

                for (size_t part = 0; part < ranges.size(); ++part) {
                    const double build_t0 = wall_time_seconds();
                    EXAD::SolvedLayer<T> current = make_solved_layer_from_slot_range<T>(
                        set,
                        luts,
                        layer_info,
                        slot,
                        param,
                        dtype_mode,
                        zero_val,
                        num_threads,
                        ranges[part]
                    );
                    EXAD::fill_success_values(current.success_values, T{}, num_threads);
                    current_build_seconds += wall_time_seconds() - build_t0;

                    const double recalc_t0 = wall_time_seconds();
                    recalculate_exad_direct_single_future(
                        current,
                        future,
                        1ULL,
                        1.0 - options.spawn_rate4,
                        luts,
                        masker.tiles_combination_table,
                        masker.permutation_table,
                        param,
                        static_cast<uint32_t>(2 * step) + ini_board_sum + 2U,
                        spec,
                        max_scale,
                        zero_val,
                        step > options.docheck_step,
                        options.target,
                        num_threads,
                        match_dict
                    );
                    recalculate_seconds += wall_time_seconds() - recalc_t0;

                    const std::string chunk_path =
                        exad_slot_part_chunk_path(chunk_dir, slot, part, ".partial");
                    const double chunk_write_t0 = wall_time_seconds();
                    write_slot_chunk_file<T>(chunk_path, current, slot, io_config);
                    current_write_seconds += wall_time_seconds() - chunk_write_t0;
                    partial_chunk_paths[slot].push_back(chunk_path);
                }
            }
            reader.close();
            future = empty_future_layer<T>(dtype_mode);
        }
        maybe_throw_exad_chunked_test_stop(
            "EXAD_TEST_STOP_AFTER_CHUNKED_FIRST_PASS_STEP",
            step,
            "after_first_pass"
        );

        double future_compact_seconds = 0.0;
        double future_write_seconds = 0.0;
        double compress_seconds = 0.0;
        uint64_t future_threshold_values_before = 0;
        uint64_t future_threshold_values_after = future_threshold_values_before;
        double effective_deletion_threshold = 0.0;

        {
            EXAD::SolvedLayer<T> future = empty_future_layer<T>(dtype_mode);
            int cached_future_step = std::numeric_limits<int>::min();
            load_future_layer(
                options, step + 2, dtype_mode, luts, future, cached_future_step, io_config,
                future_read_seconds, future_index_seconds
            );

            for (size_t slot = 0; slot < bucket_slot_count(); ++slot) {
                if (partial_chunk_paths[slot].empty()) {
                    throw std::runtime_error("missing EXAD partial slot chunks before second pass");
                }
                std::vector<std::string> final_part_paths;
                final_part_paths.reserve(partial_chunk_paths[slot].size());

                for (size_t part = 0; part < partial_chunk_paths[slot].size(); ++part) {
                    const std::string &partial_path = partial_chunk_paths[slot][part];
                    const double build_t0 = wall_time_seconds();
                    EXAD::SolvedLayer<T> current =
                        read_slot_chunk_layer<T>(partial_path, dtype_mode, slot, io_config);
                    current_build_seconds += wall_time_seconds() - build_t0;

                    const double recalc_t0 = wall_time_seconds();
                    recalculate_exad_direct_single_future(
                        current,
                        future,
                        2ULL,
                        options.spawn_rate4,
                        luts,
                        masker.tiles_combination_table,
                        masker.permutation_table,
                        param,
                        static_cast<uint32_t>(2 * step) + ini_board_sum + 4U,
                        spec,
                        max_scale,
                        zero_val,
                        step > options.docheck_step,
                        options.target,
                        num_threads,
                        match_dict
                    );
                    recalculate_seconds += wall_time_seconds() - recalc_t0;

                    auto [slot_values, slot_max] = EXAD::solved_value_count_and_max(current, zero_val);
                    input_values += slot_values;
                    input_rows += current.live_board_count;
                    if (slot_max > max_rate) {
                        max_rate = slot_max;
                    }

                    const double compact_t0 = wall_time_seconds();
                    current = EXAD::compact_solved_layer(current, luts, zero_val, num_threads);
                    zero_compact_seconds += wall_time_seconds() - compact_t0;

                    const bool single_part = partial_chunk_paths[slot].size() == 1U;
                    const std::string chunk_path = single_part
                        ? exad_slot_chunk_path(chunk_dir, slot)
                        : exad_slot_part_chunk_path(chunk_dir, slot, part, ".final");
                    const double chunk_write_t0 = wall_time_seconds();
                    write_slot_chunk_file<T>(chunk_path, current, slot, io_config);
                    current_write_seconds += wall_time_seconds() - chunk_write_t0;
                    if (single_part) {
                        manifests[slot] = read_slot_chunk_manifest<T>(chunk_path, dtype_mode, slot);
                    } else {
                        final_part_paths.push_back(chunk_path);
                    }
                    NativePath::remove(partial_path, cleanup_ec);
                }

                if (partial_chunk_paths[slot].size() > 1U) {
                    const std::string chunk_path = exad_slot_chunk_path(chunk_dir, slot);
                    const double merge_part_t0 = wall_time_seconds();
                    manifests[slot] = merge_slot_part_chunks_to_slot_chunk<T>(
                        final_part_paths,
                        chunk_path,
                        slot,
                        dtype_mode,
                        luts,
                        io_config
                    );
                    current_write_seconds += wall_time_seconds() - merge_part_t0;
                    for (const std::string &path : final_part_paths) {
                        NativePath::remove(path, cleanup_ec);
                    }
                }
            }

            future_threshold_values_before = static_cast<uint64_t>(future.success_values.size());
            future_threshold_values_after = future_threshold_values_before;
            if (RuntimeControls::deletion_threshold_enabled(deletion_threshold_state) && !future.empty()) {
                bool should_compact_future = deletion_threshold_state.absolute > 0.0;
                T layer_threshold = RuntimeControls::absolute_deletion_threshold(
                    zero_val,
                    max_scale,
                    deletion_threshold_state
                );
                if (deletion_threshold_state.relative > 0.0) {
                    const auto [ignored_future_values, future_max] =
                        EXAD::solved_value_count_and_max(future, zero_val);
                    (void)ignored_future_values;
                    if (future_max > zero_val) {
                        const T relative_threshold = RuntimeControls::relative_deletion_threshold(
                            future_max,
                            zero_val,
                            deletion_threshold_state
                        );
                        layer_threshold = RuntimeControls::max_deletion_threshold(
                            layer_threshold,
                            relative_threshold
                        );
                        should_compact_future = true;
                    }
                }
                if (should_compact_future) {
                    effective_deletion_threshold =
                        RuntimeControls::normalized_deletion_threshold(layer_threshold, zero_val, max_scale);
                    const double future_compact_t0 = wall_time_seconds();
                    EXAD::SolvedLayer<T> compacted_future =
                        EXAD::compact_solved_layer(future, luts, layer_threshold, num_threads);
                    future_compact_seconds = wall_time_seconds() - future_compact_t0;
                    future_threshold_values_after = static_cast<uint64_t>(compacted_future.success_values.size());

                    if (options.compress) {
                        compress_seconds += compress_exad_solved_layer_from_memory(
                            options,
                            step + 2,
                            compacted_future,
                            luts
                        );
                    } else {
                        const double future_write_t0 = wall_time_seconds();
                        EXAD::write_solved_layer_file(
                            EXAD::solved_file_path(options.pathname, step + 2),
                            compacted_future,
                            io_config
                        );
                        future_write_seconds = wall_time_seconds() - future_write_t0;
                    }
                }
            }
            future = empty_future_layer<T>(dtype_mode);
        }
        maybe_throw_exad_chunked_test_stop(
            "EXAD_TEST_STOP_AFTER_CHUNKED_FINAL_CHUNKS_STEP",
            step,
            "after_final_chunks"
        );

        const double merge_t0 = wall_time_seconds();
        const EXADChunkMergeSummary merge_summary = merge_slot_chunks_to_solved_file<T>(
            solved_path,
            writing_path,
            dtype_mode,
            layer_info.original_board_sum,
            layer_info.threshold_bits,
            layer_info.lut_signature,
            layer_info.physical_transform,
            layer_info.inverse_physical_transform,
            layer_info.logical_pattern_signature,
            layer_info.physical_pattern_signature,
            manifests,
            io_config
        );
        current_write_seconds += wall_time_seconds() - merge_t0;
        maybe_throw_exad_chunked_test_stop(
            "EXAD_TEST_STOP_AFTER_CHUNKED_SOLVED_WRITE_STEP",
            step,
            "after_solved_write"
        );

        NativePath::remove_all(chunk_dir, cleanup_ec);
        if (cleanup_ec) {
            throw std::runtime_error("failed to remove EXAD chunk directory: " + chunk_dir);
        }
        EXAD::remove_layer_file(layer_path);

        const double normalized_max_rate = static_cast<double>(max_rate - zero_val) /
            static_cast<double>(max_scale - zero_val);

        EXADSolveStatsRecord record;
        record.stage = "chunked_solve";
        record.step = step;
        record.input_rows = input_rows;
        record.input_values = input_values;
        record.post_zero_rows = merge_summary.post_zero_rows;
        record.post_zero_values = merge_summary.post_zero_values;
        record.deletion_threshold = layer_deletion_threshold;
        record.relative_deletion_threshold = deletion_threshold_state.relative;
        record.effective_deletion_threshold = effective_deletion_threshold;
        record.current_retained_ratio = RuntimeControls::retention_ratio(record.post_zero_values, record.input_values);
        record.future_threshold_values_before = future_threshold_values_before;
        record.future_threshold_values_after = future_threshold_values_after;
        record.future_threshold_retained_ratio =
            RuntimeControls::retention_ratio(future_threshold_values_after, future_threshold_values_before);
        record.max_success = normalized_max_rate;
        record.current_read_seconds = current_read_seconds;
        record.current_build_seconds = current_build_seconds;
        record.future_read_seconds = future_read_seconds;
        record.future_index_seconds = future_index_seconds;
        record.recalculate_seconds = recalculate_seconds;
        record.zero_compact_seconds = zero_compact_seconds;
        record.current_write_seconds = current_write_seconds;
        record.future_compact_seconds = future_compact_seconds;
        record.future_write_seconds = future_write_seconds;
        record.compress_seconds = compress_seconds;
        record.metadata_bytes = merge_summary.metadata_bytes;
        record.success_bytes = merge_summary.success_bytes;
        record.bitmap_density = merge_summary.bitmap_density;
        append_exad_solve_stats_record(options, record);

        total_record.input_rows += record.input_rows;
        total_record.input_values += record.input_values;
        total_record.post_zero_rows += record.post_zero_rows;
        total_record.post_zero_values += record.post_zero_values;
        total_record.current_retained_ratio =
            RuntimeControls::retention_ratio(total_record.post_zero_values, total_record.input_values);
        total_record.future_threshold_values_before += record.future_threshold_values_before;
        total_record.future_threshold_values_after += record.future_threshold_values_after;
        total_record.future_threshold_retained_ratio =
            RuntimeControls::retention_ratio(
                total_record.future_threshold_values_after,
                total_record.future_threshold_values_before
            );
        total_record.max_success = std::max(total_record.max_success, record.max_success);
        total_record.current_read_seconds += record.current_read_seconds;
        total_record.current_build_seconds += record.current_build_seconds;
        total_record.future_read_seconds += record.future_read_seconds;
        total_record.future_index_seconds += record.future_index_seconds;
        total_record.recalculate_seconds += record.recalculate_seconds;
        total_record.zero_compact_seconds += record.zero_compact_seconds;
        total_record.current_write_seconds += record.current_write_seconds;
        total_record.future_compact_seconds += record.future_compact_seconds;
        total_record.future_write_seconds += record.future_write_seconds;
        total_record.compress_seconds += record.compress_seconds;
        total_record.metadata_bytes += record.metadata_bytes;
        total_record.success_bytes += record.success_bytes;
        total_bitmap_live += record.post_zero_rows;
        total_bitmap_bits += merge_summary.bitmap_bits;
        total_record.bitmap_density = total_bitmap_bits == 0U
            ? 0.0
            : static_cast<double>(total_bitmap_live) / static_cast<double>(total_bitmap_bits);
    }

    total_record.deletion_threshold = deletion_threshold_state.absolute;
    total_record.relative_deletion_threshold = deletion_threshold_state.relative;
    match_dict.clear();
    total_record.compress_seconds += compress_all_exad_solved_files(options, true);
    remove_exad_temp_layers(options);
    append_exad_solve_stats_record(options, total_record);
}

} // namespace

void run_pattern_solve_exad_cpp(
    const std::vector<uint64_t> &arr_init,
    const AdvancedPatternSpec &spec,
    const RunOptions &options
) {
    (void)HybridSearch::mode();
    const bool use_chunked_solve = options.chunked_solve || exad_force_standard_chunked_solve();
    switch (success_rate_kind_from_name(options.success_rate_dtype)) {
        case SuccessRateKind::UInt64:
            if (use_chunked_solve) {
                recalculate_process_exad_chunked_impl<uint64_t>(arr_init, spec, options);
            } else {
                recalculate_process_exad_impl<uint64_t>(arr_init, spec, options);
            }
            return;
        case SuccessRateKind::Float32:
            if (use_chunked_solve) {
                recalculate_process_exad_chunked_impl<float>(arr_init, spec, options);
            } else {
                recalculate_process_exad_impl<float>(arr_init, spec, options);
            }
            return;
        case SuccessRateKind::Float64:
            if (use_chunked_solve) {
                recalculate_process_exad_chunked_impl<double>(arr_init, spec, options);
            } else {
                recalculate_process_exad_impl<double>(arr_init, spec, options);
            }
            return;
        case SuccessRateKind::UInt32:
        default:
            if (use_chunked_solve) {
                recalculate_process_exad_chunked_impl<uint32_t>(arr_init, spec, options);
            } else {
                recalculate_process_exad_impl<uint32_t>(arr_init, spec, options);
            }
            return;
    }
}
