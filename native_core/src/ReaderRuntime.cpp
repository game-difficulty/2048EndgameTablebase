#include "ReaderRuntime.h"

#include "BoardCodec.h"
#include "BoardMover.h"
#include "BCCompressedResult.h"
#include "EXADCompressedResult.h"
#include "EXCompressedResult.h"
#include "EXPrefix36Runtime.h"
#include "FileIOUtils.h"
#include "Formation.h"
#include "NativeLzma.h"
#include "PathUtils.h"
#include "SymmetryUtils.h"
#include "TrieCompression.h"
#include "VBoardMover.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kPrecisionDigits = 9;
constexpr std::array<const char *, 4> kDirectionNames = {"up", "right", "down", "left"};
constexpr std::array<const char *, 4> kOrderedResultKeys = {"down", "right", "left", "up"};

#pragma pack(push, 1)
struct TrieNode32 {
    uint8_t key;
    uint32_t next;
};

struct TrieNode16 {
    uint8_t key;
    uint16_t next;
};

struct TrieSegmentEntry {
    uint32_t index;
    uint64_t file_offset;
};
#pragma pack(pop)

#pragma pack(push, 1)
template <typename T>
struct CompactBookEntry {
    uint32_t lower32;
    T success;
};
#pragma pack(pop)

static_assert(sizeof(TrieNode32) == 5, "Unexpected trie node32 layout");
static_assert(sizeof(TrieNode16) == 3, "Unexpected trie node16 layout");
static_assert(sizeof(TrieSegmentEntry) == 12, "Unexpected trie segment entry layout");

struct DTypeInfo {
    SuccessRateKind kind = SuccessRateKind::UInt32;
    double max_scale = 4e9;
    double zero_value = 0.0;
};

struct SearchValue {
    ReaderValueKind kind = ReaderValueKind::NoneValue;
    double number = 0.0;
    std::string text;
};

struct ClassicLookupContext {
    std::string pathname;
    std::string filename;
    std::string success_rate_dtype;
    fs::path book_path;
    fs::path compressed_dir;
    std::string prefix;
    bool book_exists = false;
    bool compressed_exists = false;
    std::vector<TrieNode32> ind;
    std::vector<TrieSegmentEntry> segments;
};

BoardMatrix rotate_left(const BoardMatrix &board) {
    BoardMatrix rotated{};
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 4; ++col) {
            rotated[3U - col][row] = board[row][col];
        }
    }
    return rotated;
}

BoardMatrix rotate_right(const BoardMatrix &board) {
    BoardMatrix rotated{};
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 4; ++col) {
            rotated[col][3U - row] = board[row][col];
        }
    }
    return rotated;
}

BoardMatrix rotate_180(const BoardMatrix &board) {
    BoardMatrix rotated{};
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 4; ++col) {
            rotated[3U - row][3U - col] = board[row][col];
        }
    }
    return rotated;
}

BoardMatrix flip_horizontal(const BoardMatrix &board) {
    BoardMatrix flipped = board;
    for (auto &row : flipped) {
        std::reverse(row.begin(), row.end());
    }
    return flipped;
}

BoardMatrix apply_operation(const BoardMatrix &board, int operation_index) {
    switch (operation_index) {
        case 1:
            return rotate_left(board);
        case 2:
            return rotate_180(board);
        case 3:
            return rotate_right(board);
        case 4:
            return flip_horizontal(board);
        case 5:
            return flip_horizontal(rotate_left(board));
        case 6:
            return flip_horizontal(rotate_180(board));
        case 7:
            return flip_horizontal(rotate_right(board));
        case 0:
        default:
            return board;
    }
}

std::string adjust_direction(int operation_index, const std::string &direction) {
    std::string adjusted = direction;
    if (operation_index >= 4) {
        if (adjusted == "left") {
            adjusted = "right";
        } else if (adjusted == "right") {
            adjusted = "left";
        }
    }

    int direction_index = 0;
    if (adjusted == "right") {
        direction_index = 1;
    } else if (adjusted == "down") {
        direction_index = 2;
    } else if (adjusted == "left") {
        direction_index = 3;
    }
    direction_index = (direction_index + (operation_index % 4)) % 4;
    return std::string(kDirectionNames[static_cast<size_t>(direction_index)]);
}

std::vector<int> operation_sequence(bool is_variant, int last_operation_index) {
    if (is_variant) {
        return {0};
    }
    std::vector<int> operations;
    operations.reserve(9);
    operations.push_back(last_operation_index);
    for (int index = 0; index < 8; ++index) {
        operations.push_back(index);
    }
    return operations;
}

template <typename T>
std::vector<T> read_binary_vector(const fs::path &path) {
    return FileIOUtils::read_binary_vector<T>(path);
}

DTypeInfo dtype_info_for_name(const std::string &name) {
    if (name == "uint64") {
        return {SuccessRateKind::UInt64, 1.6e18, 0.0};
    }
    if (name == "float32") {
        return {SuccessRateKind::Float32, 1.0, 0.0};
    }
    if (name == "float64") {
        return {SuccessRateKind::Float64, 1.0, 0.0};
    }
    if (name == "1-float32") {
        return {SuccessRateKind::Float32, 0.0, -1.0};
    }
    if (name == "1-float64") {
        return {SuccessRateKind::Float64, 0.0, -1.0};
    }
    return {SuccessRateKind::UInt32, 4e9, 0.0};
}

double maybe_round_value(double value, const std::string &dtype_name) {
    if (dtype_name.find("32") == std::string::npos || std::abs(value) <= 1e-7) {
        return value;
    }
    const double scale = std::pow(10.0, kPrecisionDigits);
    return std::round(value * scale) / scale;
}

SearchValue numeric_search_value(double value, const std::string &dtype_name) {
    return {ReaderValueKind::Numeric, maybe_round_value(value, dtype_name), {}};
}

SearchValue none_search_value() {
    return {};
}

SearchValue string_search_value(std::string value) {
    SearchValue result;
    result.kind = ReaderValueKind::String;
    result.text = std::move(value);
    return result;
}

std::vector<OrderedReaderEntry> blank_direction_entries() {
    return {
        {"down", ReaderValueKind::String, 0.0, ""},
        {"right", ReaderValueKind::String, 0.0, ""},
        {"left", ReaderValueKind::String, 0.0, ""},
        {"up", ReaderValueKind::String, 0.0, ""},
    };
}

struct PreferredResultAccumulator {
    bool has_best = false;
    double best_success_rate = 0.0;
    std::vector<OrderedReaderEntry> entries = blank_direction_entries();
    std::string dtype;
    int operation_index = 0;
};

std::vector<OrderedReaderEntry> question_entries() {
    return {{"?", ReaderValueKind::String, 0.0, "?"}};
}

std::vector<OrderedReaderEntry> sort_adjusted_entries(
    const std::vector<std::pair<std::string, SearchValue>> &adjusted_entries
) {
    std::vector<OrderedReaderEntry> numeric_entries;
    std::vector<OrderedReaderEntry> other_entries;
    for (const auto &item : adjusted_entries) {
        OrderedReaderEntry entry{item.first, item.second.kind, item.second.number, item.second.text};
        if (entry.kind == ReaderValueKind::Numeric) {
            numeric_entries.push_back(entry);
        } else {
            other_entries.push_back(entry);
        }
    }
    std::sort(
        numeric_entries.begin(),
        numeric_entries.end(),
        [](const OrderedReaderEntry &lhs, const OrderedReaderEntry &rhs) {
            return lhs.number > rhs.number;
        }
    );
    std::vector<OrderedReaderEntry> result;
    result.reserve(numeric_entries.size() + other_entries.size());
    result.insert(result.end(), numeric_entries.begin(), numeric_entries.end());
    result.insert(result.end(), other_entries.begin(), other_entries.end());
    return result;
}

std::optional<double> first_numeric_value(const std::vector<OrderedReaderEntry> &entries) {
    for (const auto &entry : entries) {
        if (entry.kind == ReaderValueKind::Numeric) {
            return entry.number;
        }
    }
    return std::nullopt;
}

void consider_preferred_result(
    PreferredResultAccumulator &accumulator,
    const std::vector<OrderedReaderEntry> &entries,
    const std::string &dtype,
    int operation_index
) {
    const std::optional<double> raw_value = first_numeric_value(entries);
    if (!raw_value) {
        return;
    }
    const double success_rate = *raw_value - dtype_info_for_name(dtype).zero_value;
    if (success_rate <= 0.0 ||
        (accumulator.has_best && success_rate <= accumulator.best_success_rate)) {
        return;
    }
    accumulator.has_best = true;
    accumulator.best_success_rate = success_rate;
    accumulator.entries = entries;
    accumulator.dtype = dtype;
    accumulator.operation_index = operation_index;
}

template <typename T>
double normalize_raw_value(T value, double max_scale) {
    if constexpr (std::is_floating_point_v<T>) {
        return static_cast<double>(value);
    }
    return max_scale > 1.0 ? static_cast<double>(value) / max_scale : static_cast<double>(value);
}

template <typename T>
bool find_success_entry_in_file(
    const fs::path &path,
    uint64_t search_key,
    const DTypeInfo &dtype_info,
    double &result
) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return false;
    }
    const size_t file_size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    const size_t record_size = sizeof(SuccessEntry<T>);
    const size_t num_records = record_size == 0 ? 0 : file_size / record_size;
    size_t left = 0;
    size_t right = (num_records == 0) ? 0 : (num_records - 1);
    while (left <= right && num_records != 0) {
        const size_t mid = (left + right) / 2U;
        SuccessEntry<T> entry{};
        file.seekg(static_cast<std::streamoff>(mid * record_size), std::ios::beg);
        file.read(reinterpret_cast<char *>(&entry), static_cast<std::streamsize>(record_size));
        if (entry.board == search_key) {
            result = normalize_raw_value(entry.success, dtype_info.max_scale);
            return true;
        }
        if (entry.board < search_key) {
            left = mid + 1U;
        } else {
            if (mid == 0U) {
                break;
            }
            right = mid - 1U;
        }
    }
    result = dtype_info.zero_value;
    return true;
}

std::pair<size_t, size_t> search_trie_range(const std::vector<TrieNode32> &ind, uint64_t board) {
    const uint8_t prefixes[3] = {
        static_cast<uint8_t>((board >> 56U) & 0xFFU),
        static_cast<uint8_t>((board >> 48U) & 0xFFU),
        static_cast<uint8_t>((board >> 40U) & 0xFFU),
    };

    size_t low = 2;
    size_t high = static_cast<size_t>(ind[1].next) - 1ULL;
    for (uint8_t prefix : prefixes) {
        bool found = false;
        size_t mid = low;
        while (low <= high) {
            mid = (low + high) / 2ULL;
            if (ind[mid].key == prefix) {
                found = true;
                break;
            }
            if (ind[mid].key < prefix) {
                low = mid + 1ULL;
            } else {
                if (mid == 0) {
                    break;
                }
                high = mid - 1ULL;
            }
        }
        if (!found) {
            return {0, 0};
        }
        low = static_cast<size_t>(ind[mid - 1].next) + 1ULL;
        high = static_cast<size_t>(ind[mid].next);
    }
    return {low, high};
}

std::pair<uint64_t, uint64_t> get_segment_position(const std::vector<TrieSegmentEntry> &segments, uint32_t pos) {
    size_t low = 0;
    size_t high = segments.size() - 1ULL;
    uint64_t start = 0ULL;
    uint64_t end = 0ULL;
    while (low <= high) {
        const size_t mid = (low + high) / 2ULL;
        if (segments[mid].index < pos) {
            low = mid + 1ULL;
        } else {
            start = segments[mid - 1ULL].file_offset;
            end = segments[mid].file_offset;
            if (mid == 0ULL) {
                break;
            }
            high = mid - 1ULL;
        }
    }
    return {start, end};
}

bool search_ind3_relative_position(const std::vector<TrieNode16> &ind3_seg, uint8_t target_prefix, size_t &pos) {
    if (ind3_seg.size() <= 1ULL) {
        return false;
    }
    size_t low = 1ULL;
    size_t high = ind3_seg.size() - 1ULL;
    while (low <= high) {
        const size_t mid = (low + high) / 2ULL;
        if (ind3_seg[mid].key == target_prefix) {
            pos = mid - 1ULL;
            return true;
        }
        if (ind3_seg[mid].key < target_prefix) {
            low = mid + 1ULL;
        } else {
            if (mid == 0ULL) {
                break;
            }
            high = mid - 1ULL;
        }
    }
    return false;
}

template <typename T>
std::optional<double> trie_search_with_context(
    const std::string &path_prefix,
    uint64_t board,
    const DTypeInfo &dtype_info,
    const std::vector<TrieNode32> &ind,
    const std::vector<TrieSegmentEntry> &segments
) {
    if (ind.size() < 2 || segments.size() < 2) {
        return std::nullopt;
    }

    const auto [low, high] = search_trie_range(ind, board);
    if (low == 0 && high == 0) {
        return dtype_info.zero_value;
    }

    std::ifstream ii_file(NativePath::from_utf8(path_prefix + "ii"), std::ios::binary);
    if (!ii_file) {
        return std::nullopt;
    }
    const size_t count = high - low + 2ULL;
    std::vector<TrieNode16> ind3_seg(count);
    ii_file.seekg(static_cast<std::streamoff>(low * sizeof(TrieNode16) - sizeof(TrieNode16)), std::ios::beg);
    FileIOUtils::read_exact(
        ii_file,
        ind3_seg.data(),
        count * sizeof(TrieNode16),
        path_prefix + "ii"
    );

    const uint8_t target_prefix = static_cast<uint8_t>((board >> 32U) & 0xFFU);
    size_t last_pos = 0ULL;
    if (!search_ind3_relative_position(ind3_seg, target_prefix, last_pos)) {
        return dtype_info.zero_value;
    }

    const auto [start, end] = get_segment_position(segments, static_cast<uint32_t>(last_pos + low));
    std::ifstream z_file(NativePath::from_utf8(path_prefix + "z"), std::ios::binary);
    if (!z_file || end < start) {
        return std::nullopt;
    }
    z_file.seekg(static_cast<std::streamoff>(start), std::ios::beg);
    std::vector<uint8_t> compressed(static_cast<size_t>(end - start));
    if (!compressed.empty()) {
        FileIOUtils::read_exact(z_file, compressed.data(), compressed.size(), path_prefix + "z");
    }
    std::vector<uint8_t> decompressed = decompress_xz_block_native(compressed.data(), compressed.size());
    if (decompressed.size() % sizeof(CompactBookEntry<T>) != 0U) {
        return std::nullopt;
    }
    const auto *block = reinterpret_cast<const CompactBookEntry<T> *>(decompressed.data());
    const size_t block_size = decompressed.size() / sizeof(CompactBookEntry<T>);
    const uint32_t target = static_cast<uint32_t>(board & 0xFFFFFFFFULL);

    size_t sub_low = static_cast<size_t>(ind3_seg[last_pos].next);
    size_t sub_high = static_cast<size_t>(ind3_seg[last_pos + 1].next) + 1ULL;
    if (sub_high == 1ULL) {
        sub_high = block_size;
    }
    if (sub_low != 0ULL) {
        ++sub_low;
    }
    while (sub_low < sub_high) {
        const size_t mid = (sub_low + sub_high) / 2ULL;
        if (block[mid].lower32 < target) {
            sub_low = mid + 1ULL;
        } else {
            sub_high = mid;
        }
    }
    if (sub_low < block_size && block[sub_low].lower32 == target) {
        return normalize_raw_value(block[sub_low].success, dtype_info.max_scale);
    }
    return dtype_info.zero_value;
}

ClassicLookupContext make_classic_lookup_context(
    const std::string &pathname,
    const std::string &filename,
    const std::string &success_rate_dtype
) {
    ClassicLookupContext context;
    context.pathname = pathname;
    context.filename = filename;
    context.success_rate_dtype = success_rate_dtype;
    context.book_path = NativePath::from_utf8(pathname) / filename;
    context.compressed_dir = context.book_path;
    context.compressed_dir.replace_extension(".z");
    context.book_exists = fs::exists(context.book_path);
    context.compressed_exists = fs::exists(context.compressed_dir);
    if (context.compressed_exists) {
        const std::string stem = filename.substr(0, filename.size() - 4U);
        context.prefix = NativePath::to_utf8_string(context.compressed_dir / stem);
        context.ind = read_binary_vector<TrieNode32>(context.prefix + "i");
        context.segments = read_binary_vector<TrieSegmentEntry>(context.prefix + "s");
    }
    return context;
}

std::optional<double> trie_search_dispatch(
    const std::string &path_prefix,
    uint64_t board,
    const DTypeInfo &dtype_info,
    const std::vector<TrieNode32> &ind,
    const std::vector<TrieSegmentEntry> &segments
) {
    switch (dtype_info.kind) {
        case SuccessRateKind::UInt64:
            return trie_search_with_context<uint64_t>(path_prefix, board, dtype_info, ind, segments);
        case SuccessRateKind::Float32:
            return trie_search_with_context<float>(path_prefix, board, dtype_info, ind, segments);
        case SuccessRateKind::Float64:
            return trie_search_with_context<double>(path_prefix, board, dtype_info, ind, segments);
        case SuccessRateKind::UInt32:
        default:
            return trie_search_with_context<uint32_t>(path_prefix, board, dtype_info, ind, segments);
    }
}

SearchValue find_classic_value_with_context(const ClassicLookupContext &context, uint64_t search_key) {
    if (context.book_exists) {
        bool found = false;
        const double value = find_classic_value_native(
            context.pathname,
            context.filename,
            search_key,
            context.success_rate_dtype,
            found
        );
        return found ? numeric_search_value(value, context.success_rate_dtype) : none_search_value();
    }
    if (context.compressed_exists) {
        const DTypeInfo dtype_info = dtype_info_for_name(context.success_rate_dtype);
        const std::optional<double> value = trie_search_dispatch(
            context.prefix,
            search_key,
            dtype_info,
            context.ind,
            context.segments
        );
        return value ? numeric_search_value(*value, context.success_rate_dtype) : none_search_value();
    }
    return none_search_value();
}

struct FileStamp {
    uint64_t size = 0U;
    std::filesystem::file_time_type write_time{};
    bool valid = false;
};

template <typename T>
struct CachedFileEntry {
    FileStamp stamp;
    std::shared_ptr<const T> value;
};

struct BCCompressedReaderEntry {
    FileStamp stamp;
    std::shared_ptr<const BC::BCLut> lut;
    std::shared_ptr<const BCCompressedResult::PointReader> reader;
};

FileStamp file_stamp(const fs::path &path) {
    FileStamp stamp;
    std::error_code ec;
    const uint64_t size = std::filesystem::file_size(path, ec);
    if (ec) {
        return stamp;
    }
    const auto write_time = std::filesystem::last_write_time(path, ec);
    if (ec) {
        return stamp;
    }
    stamp.size = size;
    stamp.write_time = write_time;
    stamp.valid = true;
    return stamp;
}

bool same_stamp(const FileStamp &lhs, const FileStamp &rhs) {
    return lhs.valid && rhs.valid && lhs.size == rhs.size && lhs.write_time == rhs.write_time;
}

std::mutex g_bc_reader_cache_mutex;
std::unordered_map<std::string, BCCompressedReaderEntry> g_bc_compressed_reader_cache;

std::vector<uint8_t> bc_legal_tiles_for_rank(uint32_t target_rank) {
    if (target_rank >= 15U) {
        throw std::invalid_argument("BC target_rank must be < 15");
    }
    std::vector<uint8_t> legal_tiles;
    legal_tiles.reserve(static_cast<size_t>(target_rank) + 2U);
    for (uint32_t tile = 0U; tile <= target_rank; ++tile) {
        legal_tiles.push_back(static_cast<uint8_t>(tile));
    }
    legal_tiles.push_back(15U);
    return legal_tiles;
}

std::shared_ptr<const BCCompressedResult::PointReader> cached_bc_compressed_reader(
    const fs::path &path,
    uint32_t target_rank
) {
    const FileStamp stamp = file_stamp(path);
    const std::string key = NativePath::to_utf8_string(path) + "#" + std::to_string(target_rank);
    {
        std::lock_guard<std::mutex> lock(g_bc_reader_cache_mutex);
        const auto it = g_bc_compressed_reader_cache.find(key);
        if (it != g_bc_compressed_reader_cache.end() &&
            same_stamp(it->second.stamp, stamp) &&
            it->second.reader) {
            return it->second.reader;
        }
    }

    auto lut = std::make_shared<BC::BCLut>(bc_legal_tiles_for_rank(target_rank));
    auto reader = std::make_shared<BCCompressedResult::PointReader>(path, *lut);
    if (stamp.valid) {
        std::lock_guard<std::mutex> lock(g_bc_reader_cache_mutex);
        g_bc_compressed_reader_cache[key] = BCCompressedReaderEntry{stamp, std::move(lut), reader};
    }
    return reader;
}

bool bc_axis_looks_like_modulo_partition(const BC::BCFamilyTable &axis) {
    if (!axis.is_contiguous_range() || axis.axis_base_coord() != 0U ||
        axis.family_count() == 0U) {
        return false;
    }
    for (BC::FamilyId id = 0U; id < axis.family_count(); ++id) {
        if (axis.id_to_coord(id) != id) {
            return false;
        }
    }
    return true;
}

BC::BCBoardEncodedPosition bc_encode_for_point_reader(
    const BC::BCLut &lut,
    const BC::BCFamilyTable &axis,
    uint64_t board
) {
    BC::BCBoardEncodedPosition out = BC::encode_spawned_canonical_board(lut, axis, board);
    if (out.valid || !bc_axis_looks_like_modulo_partition(axis)) {
        return out;
    }

    const BC::BCQuadrantWords q = BC::unpack_board_to_quadrants(board);
    const BC::BCWordDesc &nw_desc = lut.word_desc(q.nw);
    const BC::BCWordDesc &ne_desc = lut.word_desc(q.ne);
    const BC::BCWordDesc &sw_desc = lut.word_desc(q.sw);
    const BC::BCWordDesc &se_desc = lut.word_desc(q.se);
    if (!nw_desc.valid || !ne_desc.valid || !sw_desc.valid || !se_desc.valid) {
        return {};
    }

    const uint64_t total_sum =
        static_cast<uint64_t>(nw_desc.sum) +
        static_cast<uint64_t>(ne_desc.sum) +
        static_cast<uint64_t>(sw_desc.sum) +
        static_cast<uint64_t>(se_desc.sum);
    if (total_sum != axis.layer_sum()) {
        return {};
    }

    BC::FamilyCoord row_coord = 0U;
    BC::FamilyCoord col_coord = 0U;
    if (!BC::bc_min_side_coord_u64(
            static_cast<uint64_t>(nw_desc.sum) + ne_desc.sum,
            static_cast<uint64_t>(sw_desc.sum) + se_desc.sum,
            axis.family_unit(),
            row_coord) ||
        !BC::bc_min_side_coord_u64(
            static_cast<uint64_t>(nw_desc.sum) + sw_desc.sum,
            static_cast<uint64_t>(ne_desc.sum) + se_desc.sum,
            axis.family_unit(),
            col_coord)) {
        return {};
    }

    const BC::BCEncodedKeyRank encoded =
        BC::bc_encode_key_rank_from_descs(lut, q.nw, nw_desc, ne_desc, sw_desc, se_desc);
    if (!encoded.valid) {
        return {};
    }

    const uint32_t family_count = axis.family_count();
    out.row_family = static_cast<BC::FamilyId>(row_coord % family_count);
    out.col_family = static_cast<BC::FamilyId>(col_coord % family_count);
    const uint64_t cid =
        static_cast<uint64_t>(out.row_family) * family_count +
        static_cast<uint32_t>(out.col_family);
    if (cid > std::numeric_limits<BC::CellId>::max()) {
        throw std::overflow_error("BC point reader modulo cid exceeds CellId");
    }
    out.cid = static_cast<BC::CellId>(cid);
    out.key = encoded.key;
    out.rank = encoded.rank;
    out.bitmap_len = encoded.bitmap_len;
    out.count_ne = encoded.count_ne;
    out.count_sw = encoded.count_sw;
    out.count_se = encoded.count_se;
    out.valid = true;
    return out;
}

double bc_numeric_from_raw(BC::BCSuccessDTypeMode mode, uint64_t raw_bits) {
    switch (mode) {
        case BC::BCSuccessDTypeMode::UInt32:
            return static_cast<double>(static_cast<uint32_t>(raw_bits));
        case BC::BCSuccessDTypeMode::UInt64:
            return static_cast<double>(raw_bits);
        case BC::BCSuccessDTypeMode::Float32:
        case BC::BCSuccessDTypeMode::OneMinusFloat32: {
            uint32_t bits = static_cast<uint32_t>(raw_bits);
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            return static_cast<double>(value);
        }
        case BC::BCSuccessDTypeMode::Float64:
        case BC::BCSuccessDTypeMode::OneMinusFloat64: {
            double value = 0.0;
            std::memcpy(&value, &raw_bits, sizeof(value));
            return value;
        }
    }
    return 0.0;
}

BC::BCPositionHeader bc_read_exact_position_header(
    const BC::BCBufferedFileReader &file
) {
    std::vector<uint8_t> bytes(BC::kBCPositionHeaderBytes);
    file.read_at_cached_size(0U, bytes.data(), bytes.size());
    BC::BCPositionHeader header = BC::bc_read_header(bytes);
    if (header.magic != BC::kBCPositionMagic ||
        header.format_version != BC::kBCPositionFormatVersion ||
        header.header_bytes != BC::kBCPositionHeaderBytes) {
        throw std::runtime_error("BC exact point lookup position header mismatch");
    }
    if (header.key_mode != BC::kBCPositionKeyModeQ4NwExactNeSwSeSumMaskPrefix256 ||
        header.rank_prefix_bits != BC::kBCRankPrefixBits ||
        header.rank_prefix_type != BC::kBCPositionRankPrefixTypeUint16 ||
        header.rank_payload_align != 8U) {
        throw std::runtime_error("BC exact point lookup unsupported position encoding");
    }
    if (header.family_unit == 0U ||
        header.family_unit > std::numeric_limits<uint16_t>::max() ||
        header.family_count == 0U ||
        header.family_count > std::numeric_limits<uint16_t>::max()) {
        throw std::runtime_error("BC exact point lookup invalid position axis");
    }
    const uint64_t axis_bytes = BC::bc_axis_coord_table_bytes(header.family_count);
    if (header.axis_coord_table_bytes != axis_bytes) {
        throw std::runtime_error("BC exact point lookup axis byte mismatch");
    }
    return header;
}

BC::BCFamilyTable bc_read_exact_position_axis(
    const BC::BCBufferedFileReader &file,
    const BC::BCPositionHeader &header,
    uint64_t *bytes_read = nullptr
) {
    const uint64_t axis_bytes = BC::bc_axis_coord_table_bytes(header.family_count);
    if (axis_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC exact point lookup axis table too large");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(axis_bytes));
    file.read_at_cached_size(BC::kBCPositionHeaderBytes, bytes.data(), axis_bytes);
    if (bytes_read != nullptr) {
        *bytes_read += axis_bytes;
    }
    std::vector<BC::FamilyCoord> coords;
    coords.reserve(static_cast<size_t>(header.family_count));
    for (uint32_t i = 0U; i < header.family_count; ++i) {
        coords.push_back(static_cast<BC::FamilyCoord>(
            BC::bc_load_u32_le(bytes.data() + static_cast<size_t>(i) * sizeof(uint32_t))));
    }
    return BC::BCFamilyTable(
        header.layer_sum,
        static_cast<uint16_t>(header.family_unit),
        coords);
}

BC::BCPositionCellDescriptor bc_read_exact_cell_descriptor(
    const BC::BCBufferedFileReader &file,
    const BC::BCPositionHeader &header,
    BC::CellId cid,
    uint64_t *bytes_read = nullptr
) {
    if (cid >= header.descriptor_count) {
        throw std::out_of_range("BC exact point lookup descriptor cid out of range");
    }
    const uint64_t descriptor_offset = BC::bc_checked_add_u64(
        header.descriptor_table_offset,
        static_cast<uint64_t>(cid) * BC::kBCPositionCellDescriptorBytes,
        "BC exact point lookup descriptor offset overflow");
    std::array<uint8_t, BC::kBCPositionCellDescriptorBytes> bytes{};
    file.read_at_cached_size(descriptor_offset, bytes.data(), bytes.size());
    if (bytes_read != nullptr) {
        *bytes_read += bytes.size();
    }
    return BC::bc_read_cell_descriptor(bytes.data(), bytes.size());
}

std::vector<BC::BCPositionCellDescriptor> bc_read_exact_position_descriptors(
    const BC::BCBufferedFileReader &file,
    const BC::BCPositionHeader &header,
    uint64_t *bytes_read = nullptr
) {
    if (header.descriptor_count != 0U &&
        header.descriptor_count >
            std::numeric_limits<uint64_t>::max() / BC::kBCPositionCellDescriptorBytes) {
        throw std::overflow_error("BC exact descriptor byte count overflow");
    }
    const uint64_t expected_bytes =
        header.descriptor_count * static_cast<uint64_t>(BC::kBCPositionCellDescriptorBytes);
    if (header.descriptor_table_bytes != expected_bytes) {
        throw std::runtime_error("BC exact descriptor table byte mismatch");
    }
    if (expected_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC exact descriptor table too large");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(expected_bytes));
    file.read_at_cached_size(header.descriptor_table_offset, bytes.data(), expected_bytes);
    if (bytes_read != nullptr) {
        *bytes_read += expected_bytes;
    }
    std::vector<BC::BCPositionCellDescriptor> descriptors;
    descriptors.reserve(static_cast<size_t>(header.descriptor_count));
    for (uint64_t cid = 0U; cid < header.descriptor_count; ++cid) {
        const size_t offset =
            static_cast<size_t>(cid) * BC::kBCPositionCellDescriptorBytes;
        descriptors.push_back(BC::bc_read_cell_descriptor(
            bytes.data() + offset,
            BC::kBCPositionCellDescriptorBytes));
    }
    return descriptors;
}

BC::BCBucketEntry bc_read_exact_bucket_entry(
    const BC::BCBufferedFileReader &file,
    uint64_t bucket_file_offset,
    uint64_t *bytes_read = nullptr
) {
    std::array<uint8_t, BC::kBCPositionBucketEntryBytes> bytes{};
    file.read_at_cached_size(bucket_file_offset, bytes.data(), bytes.size());
    if (bytes_read != nullptr) {
        *bytes_read += bytes.size();
    }
    return BC::bc_read_bucket_entry(bytes.data());
}

uint64_t bc_exact_bucket_base(
    const BC::BCPositionHeader &header,
    const BC::BCPositionCellDescriptor &desc
) {
    const uint64_t bucket_bytes =
        static_cast<uint64_t>(desc.bucket_count) * BC::kBCPositionBucketEntryBytes;
    if (desc.bucket_meta_offset > header.bucket_meta_bytes ||
        bucket_bytes > header.bucket_meta_bytes - desc.bucket_meta_offset) {
        throw std::runtime_error("BC exact bucket range exceeds file metadata");
    }
    return BC::bc_checked_add_u64(
        header.bucket_meta_offset,
        desc.bucket_meta_offset,
        "BC exact bucket base overflow");
}

BC::BCBucketEntry bc_read_exact_bucket_entry_by_index(
    const BC::BCBufferedFileReader &file,
    uint64_t bucket_base,
    uint32_t bucket_index,
    uint64_t *bytes_read = nullptr
) {
    const uint64_t offset = BC::bc_checked_add_u64(
        bucket_base,
        static_cast<uint64_t>(bucket_index) * BC::kBCPositionBucketEntryBytes,
        "BC exact bucket entry offset overflow");
    return bc_read_exact_bucket_entry(file, offset, bytes_read);
}

bool bc_exact_find_bucket_entry(
    const BC::BCBufferedFileReader &file,
    const BC::BCPositionHeader &header,
    const BC::BCPositionCellDescriptor &desc,
    uint64_t key,
    BC::BCBucketEntry &bucket,
    uint64_t *bytes_read = nullptr
) {
    if (desc.bucket_count == 0U) {
        return false;
    }
    const uint64_t bucket_base = bc_exact_bucket_base(header, desc);
    uint32_t lo = 0U;
    uint32_t hi = desc.bucket_count;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2U;
        const BC::BCBucketEntry candidate =
            bc_read_exact_bucket_entry_by_index(file, bucket_base, mid, bytes_read);
        if (candidate.key < key) {
            lo = mid + 1U;
        } else {
            hi = mid;
        }
    }
    if (lo >= desc.bucket_count) {
        return false;
    }
    bucket = bc_read_exact_bucket_entry_by_index(file, bucket_base, lo, bytes_read);
    return bucket.key == key;
}

BC::BCLookupResult bc_exact_lookup_bucket_rank_precise(
    const BC::BCBufferedFileReader &file,
    const BC::BCLut &lut,
    const BC::BCPositionHeader &header,
    const BC::BCPositionCellDescriptor &desc,
    const BC::BCBucketEntry &bucket,
    BC::BucketRank rank,
    uint64_t *bytes_read = nullptr
) {
    const uint32_t bitmap_len = BC::bitmap_len_from_key(lut, bucket.key);
    if (rank >= bitmap_len) {
        return {};
    }
    const uint32_t prefix_count = BC::prefix_count_for_bits(bitmap_len);
    const uint32_t bitmap_word_count = BC::words_for_bits(bitmap_len);
    if (prefix_count == 0U || bitmap_word_count == 0U) {
        return {};
    }
    const uint32_t prefix_offset = bucket.rank_payload_offset;
    const uint32_t bitmap_offset = BC::bc_rank_payload_bitmap_offset(prefix_offset, bitmap_len);
    const uint64_t prefix_end =
        static_cast<uint64_t>(prefix_offset) +
        static_cast<uint64_t>(prefix_count) * sizeof(BC::RankPrefix);
    const uint64_t bitmap_end =
        static_cast<uint64_t>(bitmap_offset) +
        static_cast<uint64_t>(bitmap_word_count) * sizeof(uint64_t);
    if (prefix_end > desc.rank_payload_bytes || bitmap_end > desc.rank_payload_bytes) {
        throw std::runtime_error("BC exact point lookup rank payload range exceeds cell");
    }
    if (desc.rank_payload_offset > header.rank_payload_bytes ||
        desc.rank_payload_bytes > header.rank_payload_bytes - desc.rank_payload_offset) {
        throw std::runtime_error("BC exact point lookup cell rank range exceeds file metadata");
    }
    const uint64_t rank_base = BC::bc_checked_add_u64(
        header.rank_payload_offset,
        desc.rank_payload_offset,
        "BC exact point lookup rank base overflow");
    const uint32_t rank_u32 = static_cast<uint32_t>(rank);
    const uint32_t block = std::min<uint32_t>(
        rank_u32 / BC::kBCRankPrefixBits,
        prefix_count - 1U);
    const uint32_t block_first_word =
        (block * BC::kBCRankPrefixBits) / BC::kBCBitmapWordBits;
    const uint32_t target_word = rank_u32 / BC::kBCBitmapWordBits;
    if (target_word < block_first_word || target_word >= bitmap_word_count) {
        throw std::logic_error("BC exact point lookup target bitmap word out of range");
    }

    std::array<uint8_t, sizeof(BC::RankPrefix)> prefix_bytes{};
    const uint64_t prefix_file_offset = BC::bc_checked_add_u64(
        rank_base,
        static_cast<uint64_t>(prefix_offset) +
            static_cast<uint64_t>(block) * sizeof(BC::RankPrefix),
        "BC exact point lookup prefix file offset overflow");
    file.read_at_cached_size(prefix_file_offset, prefix_bytes.data(), prefix_bytes.size());
    if (bytes_read != nullptr) {
        *bytes_read += prefix_bytes.size();
    }
    uint32_t before = BC::load_u16_le(prefix_bytes.data());

    const uint32_t words_to_read = target_word - block_first_word + 1U;
    std::array<uint8_t, 4U * sizeof(uint64_t)> word_bytes{};
    const uint64_t word_file_offset = BC::bc_checked_add_u64(
        rank_base,
        static_cast<uint64_t>(bitmap_offset) +
            static_cast<uint64_t>(block_first_word) * sizeof(uint64_t),
        "BC exact point lookup bitmap word file offset overflow");
    const uint64_t word_read_bytes = static_cast<uint64_t>(words_to_read) * sizeof(uint64_t);
    file.read_at_cached_size(word_file_offset, word_bytes.data(), word_read_bytes);
    if (bytes_read != nullptr) {
        *bytes_read += word_read_bytes;
    }
    for (uint32_t i = 0U; i + 1U < words_to_read; ++i) {
        before += BC::popcount64(
            BC::load_u64_le(word_bytes.data() + static_cast<size_t>(i) * sizeof(uint64_t)));
    }
    const uint64_t target =
        BC::load_u64_le(word_bytes.data() + static_cast<size_t>(words_to_read - 1U) * sizeof(uint64_t));
    const uint32_t bit = rank_u32 & (BC::kBCBitmapWordBits - 1U);
    if (((target >> bit) & 1ULL) == 0ULL) {
        return {};
    }
    if (bit != 0U) {
        before += BC::popcount64(target & ((1ULL << bit) - 1ULL));
    }
    if (bucket.success_row_offset > std::numeric_limits<uint32_t>::max() - before) {
        throw std::overflow_error("BC exact point lookup local success row overflow");
    }
    return BC::BCLookupResult{true, bucket.success_row_offset + before};
}

BC::BCSuccessHeader bc_read_exact_success_header(
    const BC::BCBufferedFileReader &file
) {
    std::vector<uint8_t> bytes(BC::kBCSuccessHeaderBytes);
    file.read_at_cached_size(0U, bytes.data(), bytes.size());
    BC::BCSuccessHeader header = BC::bc_read_success_header(bytes);
    if (header.magic != BC::kBCSuccessMagic ||
        header.format_version != BC::kBCSuccessFormatVersion ||
        header.header_bytes != BC::kBCSuccessHeaderBytes) {
        throw std::runtime_error("BC exact point lookup success header mismatch");
    }
    if (header.row_width == 0U) {
        throw std::runtime_error("BC exact point lookup success row_width is zero");
    }
    (void)BC::bc_success_dtype_from_u32(header.dtype);
    return header;
}

uint64_t bc_exact_read_success_raw_bits_precise(
    const BC::BCBufferedFileReader &file,
    const BC::BCSuccessHeader &header,
    BC::CellId cid,
    uint32_t local_success_row,
    uint32_t lane,
    uint64_t *bytes_read = nullptr,
    uint64_t *value_index_out = nullptr
) {
    if (lane >= header.row_width) {
        throw std::out_of_range("BC exact point lookup lane out of range");
    }
    if (cid >= header.descriptor_count) {
        throw std::out_of_range("BC exact point lookup success cid out of range");
    }
    std::array<uint8_t, BC::kBCSuccessCellValueOffsetBytes> offset_bytes{};
    const uint64_t offset_file = BC::bc_checked_add_u64(
        header.cell_value_offsets_offset,
        static_cast<uint64_t>(cid) * BC::kBCSuccessCellValueOffsetBytes,
        "BC exact point lookup success offset table overflow");
    file.read_at_cached_size(offset_file, offset_bytes.data(), offset_bytes.size());
    if (bytes_read != nullptr) {
        *bytes_read += offset_bytes.size();
    }
    const uint64_t cell_value_offset = BC::load_u64_le(offset_bytes.data());
    const uint64_t local_value_index = BC::bc_checked_add_u64(
        static_cast<uint64_t>(local_success_row) * header.row_width,
        lane,
        "BC exact point lookup local success value index overflow");
    const uint64_t value_index = BC::bc_checked_add_u64(
        cell_value_offset,
        local_value_index,
        "BC exact point lookup success value index overflow");
    const BC::BCSuccessDTypeMode dtype = BC::bc_success_dtype_from_u32(header.dtype);
    const uint32_t value_size = BC::bc_success_dtype_value_size(dtype);
    if (value_size != 0U &&
        value_index > std::numeric_limits<uint64_t>::max() / value_size) {
        throw std::overflow_error("BC exact point lookup success value byte index overflow");
    }
    const uint64_t value_relative_byte_offset =
        value_index * static_cast<uint64_t>(value_size);
    const uint64_t value_byte_offset = BC::bc_checked_add_u64(
        header.payload_offset,
        value_relative_byte_offset,
        "BC exact point lookup success value byte offset overflow");
    const uint64_t payload_relative = value_byte_offset - header.payload_offset;
    if (payload_relative > header.payload_bytes ||
        value_size > header.payload_bytes - payload_relative) {
        throw std::runtime_error("BC exact point lookup success value exceeds payload");
    }
    std::array<uint8_t, sizeof(uint64_t)> value_bytes{};
    file.read_at_cached_size(value_byte_offset, value_bytes.data(), value_size);
    if (bytes_read != nullptr) {
        *bytes_read += value_size;
    }
    if (value_index_out != nullptr) {
        *value_index_out = value_index;
    }
    switch (dtype) {
        case BC::BCSuccessDTypeMode::UInt32:
        case BC::BCSuccessDTypeMode::Float32:
        case BC::BCSuccessDTypeMode::OneMinusFloat32:
            return static_cast<uint64_t>(BC::bc_load_u32_le(value_bytes.data()));
        case BC::BCSuccessDTypeMode::UInt64:
        case BC::BCSuccessDTypeMode::Float64:
        case BC::BCSuccessDTypeMode::OneMinusFloat64:
            return BC::load_u64_le(value_bytes.data());
    }
    return 0U;
}

bool bc_exact_find_bucket_by_success_row(
    const BC::BCBufferedFileReader &file,
    const BC::BCPositionHeader &header,
    const BC::BCPositionCellDescriptor &desc,
    uint32_t local_success_row,
    BC::BCBucketEntry &bucket,
    uint32_t &bucket_row_count,
    uint64_t *bytes_read = nullptr
) {
    if (desc.empty() || desc.bucket_count == 0U || local_success_row >= desc.success_rows) {
        return false;
    }
    const uint64_t bucket_base = bc_exact_bucket_base(header, desc);
    uint32_t lo = 0U;
    uint32_t hi = desc.bucket_count;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2U;
        const BC::BCBucketEntry candidate =
            bc_read_exact_bucket_entry_by_index(file, bucket_base, mid, bytes_read);
        if (candidate.success_row_offset <= local_success_row) {
            lo = mid + 1U;
        } else {
            hi = mid;
        }
    }
    if (lo == 0U) {
        return false;
    }
    const uint32_t bucket_index = lo - 1U;
    bucket = bc_read_exact_bucket_entry_by_index(file, bucket_base, bucket_index, bytes_read);
    const uint32_t next_success_row = bucket_index + 1U < desc.bucket_count
        ? bc_read_exact_bucket_entry_by_index(file, bucket_base, bucket_index + 1U, bytes_read).success_row_offset
        : desc.success_rows;
    if (next_success_row < bucket.success_row_offset) {
        throw std::runtime_error("BC exact sample bucket success rows are not monotonic");
    }
    if (local_success_row < bucket.success_row_offset ||
        local_success_row >= next_success_row) {
        return false;
    }
    bucket_row_count = next_success_row - bucket.success_row_offset;
    return bucket_row_count != 0U;
}

BC::RankPrefix bc_exact_read_rank_prefix(
    const BC::BCBufferedFileReader &file,
    uint64_t rank_base,
    uint32_t prefix_offset,
    uint32_t prefix_index,
    uint64_t *bytes_read = nullptr
) {
    std::array<uint8_t, sizeof(BC::RankPrefix)> bytes{};
    const uint64_t offset = BC::bc_checked_add_u64(
        rank_base,
        static_cast<uint64_t>(prefix_offset) +
            static_cast<uint64_t>(prefix_index) * sizeof(BC::RankPrefix),
        "BC exact rank prefix offset overflow");
    file.read_at_cached_size(offset, bytes.data(), bytes.size());
    if (bytes_read != nullptr) {
        *bytes_read += bytes.size();
    }
    return static_cast<BC::RankPrefix>(BC::load_u16_le(bytes.data()));
}

bool bc_exact_select_rank_for_bucket_ordinal_precise(
    const BC::BCBufferedFileReader &file,
    const BC::BCLut &lut,
    const BC::BCPositionHeader &header,
    const BC::BCPositionCellDescriptor &desc,
    const BC::BCBucketEntry &bucket,
    uint32_t bucket_row_count,
    uint32_t ordinal,
    BC::BucketRank &rank,
    uint64_t *bytes_read = nullptr
) {
    if (ordinal >= bucket_row_count) {
        return false;
    }
    const uint32_t bitmap_len = BC::bitmap_len_from_key(lut, bucket.key);
    const uint32_t prefix_count = BC::prefix_count_for_bits(bitmap_len);
    const uint32_t bitmap_word_count = BC::words_for_bits(bitmap_len);
    if (prefix_count == 0U || bitmap_word_count == 0U) {
        return false;
    }
    const uint32_t prefix_offset = bucket.rank_payload_offset;
    const uint32_t bitmap_offset = BC::bc_rank_payload_bitmap_offset(prefix_offset, bitmap_len);
    const uint64_t prefix_end =
        static_cast<uint64_t>(prefix_offset) +
        static_cast<uint64_t>(prefix_count) * sizeof(BC::RankPrefix);
    const uint64_t bitmap_end =
        static_cast<uint64_t>(bitmap_offset) +
        static_cast<uint64_t>(bitmap_word_count) * sizeof(uint64_t);
    if (prefix_end > desc.rank_payload_bytes || bitmap_end > desc.rank_payload_bytes) {
        throw std::runtime_error("BC exact sample rank payload range exceeds cell");
    }
    if (desc.rank_payload_offset > header.rank_payload_bytes ||
        desc.rank_payload_bytes > header.rank_payload_bytes - desc.rank_payload_offset) {
        throw std::runtime_error("BC exact sample cell rank range exceeds file metadata");
    }
    const uint64_t rank_base = BC::bc_checked_add_u64(
        header.rank_payload_offset,
        desc.rank_payload_offset,
        "BC exact sample rank base overflow");

    uint32_t lo = 0U;
    uint32_t hi = prefix_count;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2U;
        const uint32_t block_end = mid + 1U < prefix_count
            ? bc_exact_read_rank_prefix(file, rank_base, prefix_offset, mid + 1U, bytes_read)
            : bucket_row_count;
        if (ordinal < block_end) {
            hi = mid;
        } else {
            lo = mid + 1U;
        }
    }
    if (lo >= prefix_count) {
        return false;
    }
    const uint32_t block = lo;
    const uint32_t block_begin = bc_exact_read_rank_prefix(
        file,
        rank_base,
        prefix_offset,
        block,
        bytes_read);
    const uint32_t block_end = block + 1U < prefix_count
        ? bc_exact_read_rank_prefix(file, rank_base, prefix_offset, block + 1U, bytes_read)
        : bucket_row_count;
    if (ordinal < block_begin || ordinal >= block_end) {
        return false;
    }
    uint32_t remaining = ordinal - block_begin;
    constexpr uint32_t kWordsPerPrefixBlock = BC::kBCRankPrefixBits / BC::kBCBitmapWordBits;
    const uint32_t block_first_word = block * kWordsPerPrefixBlock;
    if (block_first_word >= bitmap_word_count) {
        return false;
    }
    const uint32_t words_to_read =
        std::min<uint32_t>(kWordsPerPrefixBlock, bitmap_word_count - block_first_word);
    std::array<uint8_t, kWordsPerPrefixBlock * sizeof(uint64_t)> word_bytes{};
    const uint64_t word_file_offset = BC::bc_checked_add_u64(
        rank_base,
        static_cast<uint64_t>(bitmap_offset) +
            static_cast<uint64_t>(block_first_word) * sizeof(uint64_t),
        "BC exact sample bitmap word offset overflow");
    const uint64_t word_read_bytes = static_cast<uint64_t>(words_to_read) * sizeof(uint64_t);
    file.read_at_cached_size(word_file_offset, word_bytes.data(), word_read_bytes);
    if (bytes_read != nullptr) {
        *bytes_read += word_read_bytes;
    }
    for (uint32_t word_i = 0U; word_i < words_to_read; ++word_i) {
        const uint32_t global_word = block_first_word + word_i;
        uint64_t word = BC::load_u64_le(
            word_bytes.data() + static_cast<size_t>(word_i) * sizeof(uint64_t));
        if (global_word + 1U == bitmap_word_count && (bitmap_len & 63U) != 0U) {
            word &= (1ULL << (bitmap_len & 63U)) - 1ULL;
        }
        const uint32_t live_in_word = BC::popcount64(word);
        if (remaining >= live_in_word) {
            remaining -= live_in_word;
            continue;
        }
        for (uint32_t bit = 0U; bit < BC::kBCBitmapWordBits; ++bit) {
            const uint32_t rank_candidate = global_word * BC::kBCBitmapWordBits + bit;
            if (rank_candidate >= bitmap_len) {
                return false;
            }
            if (((word >> bit) & 1ULL) == 0ULL) {
                continue;
            }
            if (remaining == 0U) {
                rank = static_cast<BC::BucketRank>(rank_candidate);
                return true;
            }
            --remaining;
        }
        return false;
    }
    return false;
}

class BCExactPointReader {
public:
    BCExactPointReader(
        fs::path position_path,
        fs::path success_path,
        uint32_t target_rank
    )
        : lut_(bc_legal_tiles_for_rank(target_rank)),
          position_path_(std::move(position_path)),
          success_path_(std::move(success_path)),
          position_file_(std::make_unique<BC::BCBufferedFileReader>(position_path_)),
          success_file_(std::make_unique<BC::BCBufferedFileReader>(success_path_)) {
        position_header_ = bc_read_exact_position_header(*position_file_);
        uint64_t position_bytes_read = BC::kBCPositionHeaderBytes;
        axis_ = bc_read_exact_position_axis(*position_file_, position_header_, &position_bytes_read);
        descriptors_ = bc_read_exact_position_descriptors(*position_file_, position_header_, nullptr);
        success_header_ = bc_read_exact_success_header(*success_file_);
        validate_success_header();
    }

    BCCompressedResult::ColdLookupResult lookup(uint64_t board, uint32_t lane) const {
        std::lock_guard<std::mutex> lock(io_mutex_);
        BCCompressedResult::ColdLookupResult result;
        result.dtype = success_header_.dtype;
        result.row_width = success_header_.row_width;
        uint64_t position_bytes_read = BC::kBCPositionHeaderBytes +
            BC::bc_axis_coord_table_bytes(position_header_.family_count) +
            position_header_.descriptor_table_bytes;
        uint64_t success_bytes_read = BC::kBCSuccessHeaderBytes;

        const BC::BCBoardEncodedPosition encoded =
            bc_encode_for_point_reader(lut_, axis_, board);
        if (!encoded.valid || encoded.cid >= descriptors_.size()) {
            result.bucket_block_raw_bytes = position_bytes_read;
            return result;
        }
        const BC::BCPositionCellDescriptor &desc = descriptors_[encoded.cid];
        if (desc.empty() || desc.success_rows == 0U) {
            result.bucket_block_raw_bytes = position_bytes_read;
            return result;
        }
        BC::BCBucketEntry bucket;
        if (!bc_exact_find_bucket_entry(
                *position_file_,
                position_header_,
                desc,
                encoded.key,
                bucket,
                &position_bytes_read)) {
            result.bucket_block_raw_bytes = position_bytes_read;
            return result;
        }
        const BC::BCLookupResult row = bc_exact_lookup_bucket_rank_precise(
            *position_file_,
            lut_,
            position_header_,
            desc,
            bucket,
            encoded.rank,
            &position_bytes_read);
        if (!row.found) {
            result.bucket_block_raw_bytes = position_bytes_read;
            return result;
        }
        if (row.local_success_row >= desc.success_rows) {
            throw std::runtime_error("BC exact point lookup local success row exceeds descriptor");
        }

        uint64_t value_index = 0U;
        const uint64_t raw_bits = bc_exact_read_success_raw_bits_precise(
            *success_file_,
            success_header_,
            encoded.cid,
            row.local_success_row,
            lane,
            &success_bytes_read,
            &value_index);
        const BC::BCSuccessDTypeMode dtype = BC::bc_success_dtype_from_u32(success_header_.dtype);
        result.found = true;
        result.dtype = success_header_.dtype;
        result.row_width = success_header_.row_width;
        result.raw_value_bits = raw_bits;
        result.numeric_value = bc_numeric_from_raw(dtype, raw_bits);
        result.cid = encoded.cid;
        result.local_success_row = row.local_success_row;
        result.value_index = value_index;
        result.bucket_block_raw_bytes = position_bytes_read;
        result.value_block_raw_bytes = success_bytes_read;
        return result;
    }

    uint64_t sample_board() const {
        std::lock_guard<std::mutex> lock(io_mutex_);
        uint64_t live_rows = 0U;
        for (const BC::BCPositionCellDescriptor &desc : descriptors_) {
            live_rows += desc.success_rows;
        }
        if (live_rows == 0U) {
            return 0ULL;
        }

        static thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<uint64_t> row_pick(0U, live_rows - 1U);
        constexpr uint32_t kSampleAttempts = 128U;
        uint64_t bytes_read = 0U;
        for (uint32_t attempt = 0U; attempt < kSampleAttempts; ++attempt) {
            uint64_t target_row = row_pick(rng);
            BC::CellId cid = 0U;
            const BC::BCPositionCellDescriptor *desc = nullptr;
            for (; cid < descriptors_.size(); ++cid) {
                const BC::BCPositionCellDescriptor &candidate = descriptors_[cid];
                if (target_row < candidate.success_rows) {
                    desc = &candidate;
                    break;
                }
                target_row -= candidate.success_rows;
            }
            if (desc == nullptr || desc->empty() || desc->success_rows == 0U) {
                continue;
            }
            const uint32_t local_row = static_cast<uint32_t>(target_row);
            BC::BCBucketEntry bucket;
            uint32_t bucket_row_count = 0U;
            if (!bc_exact_find_bucket_by_success_row(
                    *position_file_,
                    position_header_,
                    *desc,
                    local_row,
                    bucket,
                    bucket_row_count,
                    &bytes_read)) {
                continue;
            }
            BC::BucketRank rank = 0U;
            if (!bc_exact_select_rank_for_bucket_ordinal_precise(
                    *position_file_,
                    lut_,
                    position_header_,
                    *desc,
                    bucket,
                    bucket_row_count,
                    local_row - bucket.success_row_offset,
                    rank,
                    &bytes_read)) {
                continue;
            }
            const BC::BCBucketBoardDecoder decoder(lut_, bucket.key);
            return decoder.board(rank);
        }
        return 0ULL;
    }

private:
    void validate_success_header() const {
        if (success_header_.descriptor_count != position_header_.descriptor_count ||
            success_header_.family_count != position_header_.family_count ||
            success_header_.position_key_mode != position_header_.key_mode ||
            success_header_.family_unit != position_header_.family_unit ||
            success_header_.axis_base_coord != position_header_.axis_base_coord ||
            success_header_.layer_sum != position_header_.layer_sum) {
            throw std::runtime_error("BC exact point lookup position/success metadata mismatch");
        }
    }

    BC::BCLut lut_;
    fs::path position_path_;
    fs::path success_path_;
    std::unique_ptr<BC::BCBufferedFileReader> position_file_;
    std::unique_ptr<BC::BCBufferedFileReader> success_file_;
    BC::BCPositionHeader position_header_{};
    BC::BCSuccessHeader success_header_{};
    BC::BCFamilyTable axis_{0U, 1U, std::vector<BC::FamilyCoord>{0U}};
    std::vector<BC::BCPositionCellDescriptor> descriptors_;
    mutable std::mutex io_mutex_;
};

struct BCExactReaderEntry {
    FileStamp position_stamp;
    FileStamp success_stamp;
    std::shared_ptr<const BCExactPointReader> reader;
};

std::unordered_map<std::string, BCExactReaderEntry> g_bc_exact_reader_cache;

std::shared_ptr<const BCExactPointReader> cached_bc_exact_reader(
    const fs::path &position_path,
    const fs::path &success_path,
    uint32_t target_rank
) {
    const FileStamp position_stamp = file_stamp(position_path);
    const FileStamp success_stamp = file_stamp(success_path);
    const std::string key =
        NativePath::to_utf8_string(position_path) + "|" +
        NativePath::to_utf8_string(success_path) + "#" +
        std::to_string(target_rank);
    {
        std::lock_guard<std::mutex> lock(g_bc_reader_cache_mutex);
        const auto it = g_bc_exact_reader_cache.find(key);
        if (it != g_bc_exact_reader_cache.end() &&
            same_stamp(it->second.position_stamp, position_stamp) &&
            same_stamp(it->second.success_stamp, success_stamp) &&
            it->second.reader) {
            return it->second.reader;
        }
    }

    auto reader = std::make_shared<BCExactPointReader>(position_path, success_path, target_rank);
    if (position_stamp.valid && success_stamp.valid) {
        std::lock_guard<std::mutex> lock(g_bc_reader_cache_mutex);
        g_bc_exact_reader_cache[key] = BCExactReaderEntry{position_stamp, success_stamp, reader};
    }
    return reader;
}

std::string bc_dtype_name(uint32_t dtype) {
    switch (BC::bc_success_dtype_from_u32(dtype)) {
        case BC::BCSuccessDTypeMode::UInt32:
            return "uint32";
        case BC::BCSuccessDTypeMode::UInt64:
            return "uint64";
        case BC::BCSuccessDTypeMode::Float32:
            return "float32";
        case BC::BCSuccessDTypeMode::Float64:
            return "float64";
        case BC::BCSuccessDTypeMode::OneMinusFloat32:
            return "1-float32";
        case BC::BCSuccessDTypeMode::OneMinusFloat64:
            return "1-float64";
    }
    return "uint32";
}

double normalize_bc_lookup_value(const BCCompressedResult::ColdLookupResult &lookup) {
    if (!lookup.found) {
        return 0.0;
    }
    const std::string dtype = bc_dtype_name(lookup.dtype);
    const DTypeInfo dtype_info = dtype_info_for_name(dtype);
    switch (BC::bc_success_dtype_from_u32(lookup.dtype)) {
        case BC::BCSuccessDTypeMode::UInt64:
            return dtype_info.max_scale > 1.0
                ? static_cast<double>(lookup.raw_value_bits) / dtype_info.max_scale
                : static_cast<double>(lookup.raw_value_bits);
        case BC::BCSuccessDTypeMode::UInt32:
            return dtype_info.max_scale > 1.0
                ? static_cast<double>(static_cast<uint32_t>(lookup.raw_value_bits)) / dtype_info.max_scale
                : static_cast<double>(static_cast<uint32_t>(lookup.raw_value_bits));
        case BC::BCSuccessDTypeMode::Float32:
        case BC::BCSuccessDTypeMode::OneMinusFloat32:
        case BC::BCSuccessDTypeMode::Float64:
        case BC::BCSuccessDTypeMode::OneMinusFloat64:
            return lookup.numeric_value;
    }
    return lookup.numeric_value;
}

uint64_t encoded_board_value_sum(uint64_t board) {
    uint64_t total = 0U;
    for (uint32_t cell = 0U; cell < 16U; ++cell) {
        const uint32_t tile = static_cast<uint32_t>((board >> (cell * 4U)) & 0xFULL);
        if (tile != 0U) {
            total += uint64_t{1} << tile;
        }
    }
    return total;
}

std::optional<uint32_t> bc_layer_ordinal(uint64_t board, int64_t nums_adjust) {
    const int64_t seed_sum = -nums_adjust;
    if (seed_sum < 0) {
        return std::nullopt;
    }
    const uint64_t board_sum = encoded_board_value_sum(board);
    if (board_sum < static_cast<uint64_t>(seed_sum)) {
        return std::nullopt;
    }
    const uint64_t delta = board_sum - static_cast<uint64_t>(seed_sum);
    if ((delta & 1ULL) != 0ULL || delta / 2ULL > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<uint32_t>(delta / 2ULL);
}

struct BCSearchResult {
    SearchValue value = none_search_value();
    std::string dtype = "uint32";
    bool found = false;
};

std::vector<std::pair<fs::path, fs::path>> bc_exact_path_pairs_for_layer_prefix(
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &prefix
) {
    std::vector<fs::path> position_paths;
    std::vector<fs::path> success_paths;
    for (const auto &path_entry : path_list) {
        if (path_entry.first.empty()) {
            continue;
        }
        const fs::path root = NativePath::from_utf8(path_entry.first);
        if (!fs::exists(root)) {
            continue;
        }
        const fs::path position_candidate = root / (prefix + ".bcpos");
        if (fs::exists(position_candidate)) {
            position_paths.push_back(position_candidate);
        }
        const fs::path success_candidate = root / (prefix + ".bcsuc");
        if (fs::exists(success_candidate)) {
            success_paths.push_back(success_candidate);
        }
    }
    std::vector<std::pair<fs::path, fs::path>> pairs;
    pairs.reserve(position_paths.size() * success_paths.size());
    for (const fs::path &position_path : position_paths) {
        for (const fs::path &success_path : success_paths) {
            pairs.push_back({position_path, success_path});
        }
    }
    return pairs;
}

BCSearchResult find_bc_value(
    const BCBookReader &reader,
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    uint64_t canonical_board,
    int64_t nums_adjust
) {
    const std::optional<uint32_t> ordinal = bc_layer_ordinal(canonical_board, nums_adjust);
    if (!ordinal) {
        return {};
    }
    const std::string prefix = pattern_full + "_" + std::to_string(*ordinal);
    for (const auto &path_entry : path_list) {
        if (path_entry.first.empty()) {
            continue;
        }
        const fs::path root = NativePath::from_utf8(path_entry.first);
        if (!fs::exists(root)) {
            continue;
        }
        const fs::path compressed_path =
            root / (prefix + BCCompressedResult::kCompressedLayerFileExtension);
        try {
            if (fs::exists(compressed_path)) {
                const BCCompressedResult::ColdLookupResult lookup =
                    BCRuntime::lookup_compressed_result_cached(
                        compressed_path,
                        reader.target_rank_,
                        canonical_board,
                        0U);
                if (lookup.found) {
                    const std::string dtype = bc_dtype_name(lookup.dtype);
                    return BCSearchResult{
                        numeric_search_value(normalize_bc_lookup_value(lookup), dtype),
                        dtype,
                        true
                    };
                }
            }
        } catch (...) {
            continue;
        }
    }
    for (const auto &exact_paths : bc_exact_path_pairs_for_layer_prefix(path_list, prefix)) {
        try {
            const BCCompressedResult::ColdLookupResult lookup =
                BCRuntime::lookup_exact_result_cached(
                    exact_paths.first,
                    exact_paths.second,
                    reader.target_rank_,
                    canonical_board,
                    0U);
            if (lookup.found) {
                const std::string dtype = bc_dtype_name(lookup.dtype);
                return BCSearchResult{
                    numeric_search_value(normalize_bc_lookup_value(lookup), dtype),
                    dtype,
                    true
                };
            }
        } catch (...) {
            continue;
        }
    }
    return {};
}

template <typename MoverType>
std::array<uint64_t, 4> move_all_dir_dispatch(uint64_t board) {
    const auto moves = MoverType::move_all_dir(board);
    return {
        std::get<0>(moves),
        std::get<1>(moves),
        std::get<2>(moves),
        std::get<3>(moves),
    };
}

std::array<uint64_t, 4> move_all_dir_for_variant(uint64_t board, bool is_variant) {
    return is_variant ? move_all_dir_dispatch<VBoardMover>(board) : move_all_dir_dispatch<BoardMover>(board);
}

SearchValue find_advanced_value(
    const AdvancedPatternSpec &spec,
    const FormationAD::MaskerContext &masker,
    const std::string &pathname,
    const std::string &filename,
    uint64_t board,
    const std::string &success_rate_dtype
) {
    const DTypeInfo dtype_info = dtype_info_for_name(success_rate_dtype);
    const auto stats = FormationAD::tile_sum_and_32k_count2(board, masker.param);
    const uint32_t total_sum = stats.total_sum;
    int count_32k = stats.count_32k;
    const uint32_t original_board_sum = board_value_sum(decode_board_matrix(board));
    const uint32_t total_32k_sum =
        static_cast<uint32_t>(masker.param.num_free_32k + masker.param.num_fixed_32k) << 15U;
    const uint32_t large_tiles_sum = original_board_sum - total_sum - total_32k_sum;

    const auto tiles_combinations = FormationAD::tiles_combination_view(
        masker.tiles_combination_table,
        static_cast<uint8_t>(large_tiles_sum >> 6U),
        static_cast<uint8_t>(count_32k - masker.param.num_free_32k)
    );

    uint64_t search_key = 0ULL;
    int symm_index = 0;
    if (!tiles_combinations.empty()) {
        if (tiles_combinations.size >= 3U && tiles_combinations[0] == tiles_combinations[2]) {
            count_32k = count_32k - 3 + 16;
            const auto pair = canonical_pair_by_mode(FormationAD::mask_board(board, 7), spec.symm_mode);
            search_key = pair.first;
            symm_index = pair.second;
        } else if (tiles_combinations.size >= 2U && tiles_combinations[0] == tiles_combinations[1]) {
            count_32k = -count_32k;
            const auto pair = canonical_pair_by_mode(FormationAD::mask_board(board, tiles_combinations[0] + 1), spec.symm_mode);
            search_key = pair.first;
            symm_index = pair.second;
        } else {
            const auto pair = canonical_pair_by_mode(FormationAD::mask_board(board, 6), spec.symm_mode);
            search_key = pair.first;
            symm_index = pair.second;
        }
    } else {
        const auto pair = canonical_pair_by_mode(board, spec.symm_mode);
        search_key = pair.first;
        symm_index = pair.second;
    }

    const fs::path root = NativePath::from_utf8(pathname) / filename;
    const fs::path index_path = root / (std::to_string(count_32k) + ".i");
    const fs::path compressed_index_path = root / (std::to_string(count_32k) + ".zi");
    const fs::path segments_path = root / (std::to_string(count_32k) + ".s");

    std::optional<size_t> ind;
    if (fs::exists(index_path)) {
        std::ifstream file(index_path, std::ios::binary | std::ios::ate);
        if (file) {
            const size_t num_records = static_cast<size_t>(file.tellg()) / sizeof(uint64_t);
            file.seekg(0, std::ios::beg);
            size_t left = 0;
            size_t right = (num_records == 0) ? 0 : (num_records - 1);
            while (left <= right && num_records != 0) {
                const size_t mid = (left + right) / 2U;
                uint64_t key = 0ULL;
                file.seekg(static_cast<std::streamoff>(mid * sizeof(uint64_t)), std::ios::beg);
                file.read(reinterpret_cast<char *>(&key), static_cast<std::streamsize>(sizeof(uint64_t)));
                if (key == search_key) {
                    ind = mid;
                    break;
                }
                if (key < search_key) {
                    left = mid + 1U;
                } else {
                    if (mid == 0U) {
                        break;
                    }
                    right = mid - 1U;
                }
            }
        }
    } else if (fs::exists(compressed_index_path) && fs::exists(segments_path)) {
        ind = find_value_uint64_compressed_native(NativePath::to_utf8_string(compressed_index_path), search_key);
    } else {
        return string_search_value("?");
    }

    if (!ind.has_value()) {
        return numeric_search_value(dtype_info.zero_value, success_rate_dtype);
    }

    const fs::path book_path = root / (std::to_string(count_32k) + ".b");
    if (!fs::exists(book_path)) {
        return string_search_value("?");
    }
    std::ifstream book_file(book_path, std::ios::binary);
    if (!book_file) {
        return string_search_value("?");
    }

    auto read_value_at_index = [&](size_t index) -> SearchValue {
        switch (dtype_info.kind) {
            case SuccessRateKind::UInt64: {
                uint64_t raw = 0ULL;
                book_file.seekg(static_cast<std::streamoff>(index * sizeof(uint64_t)), std::ios::beg);
                book_file.read(reinterpret_cast<char *>(&raw), static_cast<std::streamsize>(sizeof(uint64_t)));
                return numeric_search_value(normalize_raw_value(raw, dtype_info.max_scale), success_rate_dtype);
            }
            case SuccessRateKind::Float32: {
                float raw = 0.0f;
                book_file.seekg(static_cast<std::streamoff>(index * sizeof(float)), std::ios::beg);
                book_file.read(reinterpret_cast<char *>(&raw), static_cast<std::streamsize>(sizeof(float)));
                return numeric_search_value(normalize_raw_value(raw, dtype_info.max_scale), success_rate_dtype);
            }
            case SuccessRateKind::Float64: {
                double raw = 0.0;
                book_file.seekg(static_cast<std::streamoff>(index * sizeof(double)), std::ios::beg);
                book_file.read(reinterpret_cast<char *>(&raw), static_cast<std::streamsize>(sizeof(double)));
                return numeric_search_value(normalize_raw_value(raw, dtype_info.max_scale), success_rate_dtype);
            }
            case SuccessRateKind::UInt32:
            default: {
                uint32_t raw = 0U;
                book_file.seekg(static_cast<std::streamoff>(index * sizeof(uint32_t)), std::ios::beg);
                book_file.read(reinterpret_cast<char *>(&raw), static_cast<std::streamsize>(sizeof(uint32_t)));
                return numeric_search_value(normalize_raw_value(raw, dtype_info.max_scale), success_rate_dtype);
            }
        }
    };

    if (!tiles_combinations.empty()) {
        const std::vector<uint64_t> board_derived = FormationAD::unmask_board(
            search_key,
            original_board_sum,
            masker.tiles_combination_table,
            masker.permutation_table,
            masker.param
        );
        const uint64_t symm_board = apply_sym_like(board, symm_index);
        const auto it = std::lower_bound(board_derived.begin(), board_derived.end(), symm_board);
        if (it == board_derived.end() || *it != symm_board) {
            return numeric_search_value(dtype_info.zero_value, success_rate_dtype);
        }
        const size_t ind2 = static_cast<size_t>(std::distance(board_derived.begin(), it));
        return read_value_at_index(ind2 + (*ind) * board_derived.size());
    }

    return read_value_at_index(*ind);
}

struct EXADLookupTarget {
    int ad_key = 0;
    uint64_t board = 0;
    uint32_t column = 0;
    bool valid = false;
    bool zero = false;
};

EXADLookupTarget build_exad_lookup_target(
    const AdvancedPatternSpec &spec,
    const FormationAD::MaskerContext &masker,
    uint64_t board
) {
    EXADLookupTarget target;
    const auto stats = FormationAD::tile_sum_and_32k_count2(board, masker.param);
    const uint32_t total_sum = stats.total_sum;
    int count_32k = stats.count_32k;
    const uint32_t original_board_sum = board_value_sum(decode_board_matrix(board));
    const uint32_t total_32k_sum =
        static_cast<uint32_t>(masker.param.num_free_32k + masker.param.num_fixed_32k) << 15U;
    if (count_32k < static_cast<int>(masker.param.num_free_32k) ||
        original_board_sum < total_sum + total_32k_sum) {
        target.zero = true;
        return target;
    }
    const uint32_t large_tiles_sum = original_board_sum - total_sum - total_32k_sum;

    const auto tiles_combinations = FormationAD::tiles_combination_view(
        masker.tiles_combination_table,
        static_cast<uint8_t>(large_tiles_sum >> 6U),
        static_cast<uint8_t>(count_32k - masker.param.num_free_32k)
    );

    uint64_t search_key = 0ULL;
    int symm_index = 0;
    if (!tiles_combinations.empty()) {
        if (tiles_combinations.size >= 3U && tiles_combinations[0] == tiles_combinations[2]) {
            count_32k = count_32k - 3 + 16;
            const auto pair = canonical_pair_by_mode(FormationAD::mask_board(board, 7), spec.symm_mode);
            search_key = pair.first;
            symm_index = pair.second;
        } else if (tiles_combinations.size >= 2U && tiles_combinations[0] == tiles_combinations[1]) {
            count_32k = -count_32k;
            const auto pair = canonical_pair_by_mode(
                FormationAD::mask_board(board, tiles_combinations[0] + 1),
                spec.symm_mode
            );
            search_key = pair.first;
            symm_index = pair.second;
        } else {
            const auto pair = canonical_pair_by_mode(FormationAD::mask_board(board, 6), spec.symm_mode);
            search_key = pair.first;
            symm_index = pair.second;
        }
    } else {
        const auto pair = canonical_pair_by_mode(board, spec.symm_mode);
        search_key = pair.first;
        symm_index = pair.second;
    }

    uint32_t column = 0U;
    if (!tiles_combinations.empty()) {
        const std::vector<uint64_t> board_derived = FormationAD::unmask_board(
            search_key,
            original_board_sum,
            masker.tiles_combination_table,
            masker.permutation_table,
            masker.param
        );
        const uint64_t symm_board = apply_sym_like(board, symm_index);
        const auto it = std::lower_bound(board_derived.begin(), board_derived.end(), symm_board);
        if (it == board_derived.end() || *it != symm_board) {
            target.zero = true;
            return target;
        }
        column = static_cast<uint32_t>(std::distance(board_derived.begin(), it));
    }

    target.ad_key = count_32k;
    target.board = search_key;
    target.column = column;
    target.valid = true;
    return target;
}

double normalize_ex_lookup_value(const EXCompressedResult::ColdLookupResult &lookup, const DTypeInfo &dtype_info) {
    if (!lookup.found) {
        return dtype_info.zero_value;
    }
    switch (lookup.success_kind) {
        case SuccessRateKind::UInt64:
            return dtype_info.max_scale > 1.0
                ? static_cast<double>(lookup.raw_value_bits) / dtype_info.max_scale
                : static_cast<double>(lookup.raw_value_bits);
        case SuccessRateKind::Float32: {
            uint32_t bits = static_cast<uint32_t>(lookup.raw_value_bits);
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            return static_cast<double>(value);
        }
        case SuccessRateKind::Float64: {
            uint64_t bits = lookup.raw_value_bits;
            double value = 0.0;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }
        case SuccessRateKind::UInt32:
        default:
            return dtype_info.max_scale > 1.0
                ? static_cast<double>(static_cast<uint32_t>(lookup.raw_value_bits)) / dtype_info.max_scale
                : static_cast<double>(static_cast<uint32_t>(lookup.raw_value_bits));
    }
}

double normalize_exad_lookup_value(const EXADCompressedResult::ColdLookupResult &lookup, const DTypeInfo &dtype_info) {
    if (!lookup.found) {
        return dtype_info.zero_value;
    }
    switch (lookup.success_kind) {
        case SuccessRateKind::UInt64:
            return dtype_info.max_scale > 1.0
                ? static_cast<double>(lookup.raw_value_bits) / dtype_info.max_scale
                : static_cast<double>(lookup.raw_value_bits);
        case SuccessRateKind::Float32: {
            uint32_t bits = static_cast<uint32_t>(lookup.raw_value_bits);
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            return static_cast<double>(value);
        }
        case SuccessRateKind::Float64: {
            uint64_t bits = lookup.raw_value_bits;
            double value = 0.0;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }
        case SuccessRateKind::UInt32:
        default:
            return dtype_info.max_scale > 1.0
                ? static_cast<double>(static_cast<uint32_t>(lookup.raw_value_bits)) / dtype_info.max_scale
                : static_cast<double>(static_cast<uint32_t>(lookup.raw_value_bits));
    }
}

std::vector<fs::path> exad_compressed_candidates(const fs::path &exadbook_path) {
    std::vector<fs::path> candidates;
    fs::path replaced = exadbook_path;
    replaced.replace_extension(EXADCompressedResult::kCompressedLayerFileExtension);
    candidates.push_back(replaced);
    candidates.push_back(NativePath::from_utf8(
        NativePath::to_utf8_string(exadbook_path) + EXADCompressedResult::kCompressedLayerFileExtension));
    return candidates;
}

std::optional<fs::path> first_existing_exad_lut_path(
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full
) {
    for (const auto &path_entry : path_list) {
        const fs::path exadlut_path = NativePath::from_utf8(path_entry.first) / (pattern_full + "_.exadlut");
        if (fs::exists(exadlut_path)) {
            return exadlut_path;
        }
    }
    return std::nullopt;
}

SearchValue find_exad_value(
    const AdvancedPatternSpec &spec,
    const FormationAD::MaskerContext &masker,
    const std::string &pathname,
    const std::string &filename,
    const fs::path &exadlut_path,
    const std::string &pattern_full,
    uint64_t board,
    const std::string &success_rate_dtype
) {
    const DTypeInfo dtype_info = dtype_info_for_name(success_rate_dtype);
    const fs::path root = NativePath::from_utf8(pathname);
    if (!fs::exists(exadlut_path)) {
        return string_search_value("?");
    }

    const EXADLookupTarget target = build_exad_lookup_target(spec, masker, board);
    if (target.zero) {
        return numeric_search_value(dtype_info.zero_value, success_rate_dtype);
    }
    if (!target.valid) {
        return string_search_value("?");
    }

    const fs::path exadbook_path = root / filename;
    try {
        if (fs::exists(exadbook_path)) {
            const auto lookup = EXADCompressedResult::lookup_exadbook_cold(
                NativePath::to_utf8_string(exadbook_path),
                NativePath::to_utf8_string(exadlut_path),
                target.ad_key,
                target.board,
                target.column
            );
            return numeric_search_value(normalize_exad_lookup_value(lookup, dtype_info), success_rate_dtype);
        }
        for (const fs::path &candidate : exad_compressed_candidates(exadbook_path)) {
            if (!fs::exists(candidate)) {
                continue;
            }
            const auto lookup = EXADCompressedResult::lookup_exad_cold(
                NativePath::to_utf8_string(candidate),
                NativePath::to_utf8_string(exadlut_path),
                target.ad_key,
                target.board,
                target.column
            );
            return numeric_search_value(normalize_exad_lookup_value(lookup, dtype_info), success_rate_dtype);
        }
        return string_search_value("?");
    } catch (...) {
        return string_search_value("?");
    }
}

SearchValue find_exad_value(
    const AdvancedPatternSpec &spec,
    const FormationAD::MaskerContext &masker,
    const std::string &pathname,
    const std::string &filename,
    const std::string &pattern_full,
    uint64_t board,
    const std::string &success_rate_dtype
) {
    const fs::path exadlut_path = NativePath::from_utf8(pathname) / (pattern_full + "_.exadlut");
    return find_exad_value(
        spec,
        masker,
        pathname,
        filename,
        exadlut_path,
        pattern_full,
        board,
        success_rate_dtype);
}

std::vector<fs::path> ex_compressed_candidates(const fs::path &zbook_path) {
    std::vector<fs::path> candidates;
    fs::path replaced = zbook_path;
    replaced.replace_extension(EXCompressedResult::kCompressedLayerFileExtension);
    candidates.push_back(replaced);
    candidates.push_back(NativePath::from_utf8(
        NativePath::to_utf8_string(zbook_path) + EXCompressedResult::kCompressedLayerFileExtension));
    return candidates;
}

std::optional<fs::path> first_existing_ex_zlut_path(
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full
) {
    for (const auto &path_entry : path_list) {
        const fs::path zlut_path = NativePath::from_utf8(path_entry.first) / (pattern_full + "_.zlut");
        if (fs::exists(zlut_path)) {
            return zlut_path;
        }
    }
    return std::nullopt;
}

SearchValue find_ex_value(
    const std::string &pathname,
    const std::string &filename,
    const fs::path &zlut_path,
    const std::string &pattern_full,
    uint64_t board,
    const std::string &success_rate_dtype
) {
    const DTypeInfo dtype_info = dtype_info_for_name(success_rate_dtype);
    const fs::path root = NativePath::from_utf8(pathname);
    if (!fs::exists(zlut_path)) {
        return string_search_value("?");
    }

    const fs::path zbook_path = root / filename;
    try {
        EXCompressedResult::ColdLookupResult lookup;
        if (fs::exists(zbook_path)) {
            lookup = EXPrefix36Runtime::lookup_zbook_cold(
                NativePath::to_utf8_string(zbook_path), NativePath::to_utf8_string(zlut_path), board);
        } else {
            bool found_compressed = false;
            for (const fs::path &candidate : ex_compressed_candidates(zbook_path)) {
                if (!fs::exists(candidate)) {
                    continue;
                }
                lookup = EXCompressedResult::lookup_cold(
                    NativePath::to_utf8_string(candidate),
                    NativePath::to_utf8_string(zlut_path),
                    board);
                found_compressed = true;
                break;
            }
            if (!found_compressed) {
                return string_search_value("?");
            }
        }
        return numeric_search_value(normalize_ex_lookup_value(lookup, dtype_info), success_rate_dtype);
    } catch (...) {
        return string_search_value("?");
    }
}

SearchValue find_ex_value(
    const std::string &pathname,
    const std::string &filename,
    const std::string &pattern_full,
    uint64_t board,
    const std::string &success_rate_dtype
) {
    const fs::path zlut_path = NativePath::from_utf8(pathname) / (pattern_full + "_.zlut");
    return find_ex_value(
        pathname,
        filename,
        zlut_path,
        pattern_full,
        board,
        success_rate_dtype);
}

bool sample_ex_zbook_state(
    const fs::path &zbook_path,
    const fs::path &zlut_path,
    std::mt19937 &rng,
    uint64_t &state
) {
    (void)rng;
    uint64_t raw = 0;
    double numeric = 0.0;
    try {
        return EXPrefix36Runtime::sample_zbook_state(
            NativePath::to_utf8_string(zbook_path),
            NativePath::to_utf8_string(zlut_path),
            state,
            raw,
            numeric);
    } catch (...) {
        return false;
    }
}

ReaderMoveResult evaluate_classic_result_candidates(
    ClassicBookReader &reader,
    const BoardMatrix &board_matrix,
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    int64_t nums_adjust
);

ReaderMoveResult evaluate_advanced_result_candidates(
    AdvancedBookReader &reader,
    const BoardMatrix &board_matrix,
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    int64_t nums_adjust
);

ReaderMoveResult evaluate_exad_result_candidates(
    EXADBookReader &reader,
    const BoardMatrix &board_matrix,
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    int64_t nums_adjust
);

ReaderMoveResult evaluate_ex_result_candidates(
    EXBookReader &reader,
    const BoardMatrix &board_matrix,
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    int64_t nums_adjust
);

ReaderMoveResult evaluate_bc_result_candidates(
    BCBookReader &reader,
    const BoardMatrix &board_matrix,
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    int64_t nums_adjust
);

uint64_t sample_classic_book_state(
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    double spawn_rate4
);

uint64_t sample_advanced_book_state(
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    double spawn_rate4
);

ReaderMoveResult evaluate_classic_result_candidates(
    ClassicBookReader &reader,
    const BoardMatrix &board_matrix,
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    int64_t nums_adjust
) {
    if (path_list.empty()) {
        return {question_entries(), {}};
    }
    const int64_t nums = static_cast<int64_t>((board_value_sum(board_matrix) + nums_adjust) / 2);
    if (nums < 0) {
        return {blank_direction_entries(), {}};
    }

    PreferredResultAccumulator preferred;
    const std::string filename = pattern_full + "_" + std::to_string(nums) + ".book";
    const std::vector<int> operations = operation_sequence(reader.is_variant_, reader.last_operation_index_);

    for (const auto &path_entry : path_list) {
        if (!NativePath::exists(path_entry.first) || preferred.has_best) {
            continue;
        }
        const ClassicLookupContext lookup = make_classic_lookup_context(path_entry.first, filename, path_entry.second);
        if (!lookup.book_exists && !lookup.compressed_exists) {
            continue;
        }

        for (int operation_index : operations) {
            const BoardMatrix transformed_board = apply_operation(board_matrix, operation_index);
            const uint64_t encoded = encode_board_matrix(transformed_board);
            if (!is_pattern(encoded, reader.spec_.pattern_masks)) {
                continue;
            }

            std::array<SearchValue, 4> result_values = {
                none_search_value(),
                none_search_value(),
                none_search_value(),
                none_search_value(),
            };
            const auto moved_boards = move_all_dir_for_variant(encoded, reader.is_variant_);
            for (size_t index = 0; index < moved_boards.size(); ++index) {
                const uint64_t moved_board = moved_boards[index];
                if (moved_board == encoded || !is_pattern(moved_board, reader.spec_.pattern_masks)) {
                    continue;
                }
                const SearchValue value = find_classic_value_with_context(
                    lookup,
                    canonical_by_mode(moved_board, reader.spec_.symm_mode)
                );
                if (index == 0U) {
                    result_values[2] = value;
                } else if (index == 1U) {
                    result_values[1] = value;
                } else if (index == 2U) {
                    result_values[3] = value;
                } else {
                    result_values[0] = value;
                }
            }

            std::vector<std::pair<std::string, SearchValue>> adjusted_entries;
            adjusted_entries.reserve(4);
            for (size_t ordered_index = 0; ordered_index < kOrderedResultKeys.size(); ++ordered_index) {
                adjusted_entries.push_back({
                    adjust_direction(operation_index, std::string(kOrderedResultKeys[ordered_index])),
                    result_values[ordered_index],
                });
            }

            const std::vector<OrderedReaderEntry> sorted_entries = sort_adjusted_entries(adjusted_entries);
            if (!first_numeric_value(sorted_entries)) {
                continue;
            }

            if (reader.prefer_max_result_) {
                consider_preferred_result(
                    preferred, sorted_entries, path_entry.second, operation_index);
            } else {
                reader.last_operation_index_ = operation_index;
                return {sorted_entries, path_entry.second};
            }
        }
    }

    if (preferred.has_best) {
        reader.last_operation_index_ = preferred.operation_index;
    }
    return {preferred.entries, preferred.dtype};
}

ReaderMoveResult evaluate_advanced_result_candidates(
    AdvancedBookReader &reader,
    const BoardMatrix &board_matrix,
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    int64_t nums_adjust
) {
    if (path_list.empty()) {
        return {question_entries(), {}};
    }
    const int64_t nums = static_cast<int64_t>((board_value_sum(board_matrix) + nums_adjust) / 2);
    if (nums < 0) {
        return {blank_direction_entries(), {}};
    }

    PreferredResultAccumulator preferred;
    const std::string filename = pattern_full + "_" + std::to_string(nums) + "b";
    const std::vector<int> operations = operation_sequence(reader.is_variant_, reader.last_operation_index_);

    for (const auto &path_entry : path_list) {
        if (!NativePath::exists(path_entry.first) || preferred.has_best) {
            continue;
        }

        for (int operation_index : operations) {
            const BoardMatrix transformed_board = apply_operation(board_matrix, operation_index);
            const uint64_t encoded = encode_board_matrix(transformed_board);
            if (!is_pattern(encoded, reader.spec_.pattern_masks)) {
                continue;
            }

            std::array<SearchValue, 4> result_values = {
                none_search_value(),
                none_search_value(),
                none_search_value(),
                none_search_value(),
            };
            const auto moved_boards = move_all_dir_for_variant(encoded, reader.is_variant_);
            for (size_t index = 0; index < moved_boards.size(); ++index) {
                const uint64_t moved_board = moved_boards[index];
                if (moved_board == encoded || !is_pattern(moved_board, reader.spec_.pattern_masks)) {
                    continue;
                }
                const SearchValue value = find_advanced_value(
                    reader.spec_,
                    reader.masker_,
                    path_entry.first,
                    filename,
                    moved_board,
                    path_entry.second
                );
                if (index == 0U) {
                    result_values[2] = value;
                } else if (index == 1U) {
                    result_values[1] = value;
                } else if (index == 2U) {
                    result_values[3] = value;
                } else {
                    result_values[0] = value;
                }
            }

            std::vector<std::pair<std::string, SearchValue>> adjusted_entries;
            adjusted_entries.reserve(4);
            for (size_t ordered_index = 0; ordered_index < kOrderedResultKeys.size(); ++ordered_index) {
                adjusted_entries.push_back({
                    adjust_direction(operation_index, std::string(kOrderedResultKeys[ordered_index])),
                    result_values[ordered_index],
                });
            }

            const std::vector<OrderedReaderEntry> sorted_entries = sort_adjusted_entries(adjusted_entries);
            if (!first_numeric_value(sorted_entries)) {
                continue;
            }

            if (reader.prefer_max_result_) {
                consider_preferred_result(
                    preferred, sorted_entries, path_entry.second, operation_index);
            } else {
                reader.last_operation_index_ = operation_index;
                return {sorted_entries, path_entry.second};
            }
        }
    }

    if (preferred.has_best) {
        reader.last_operation_index_ = preferred.operation_index;
    }
    return {preferred.entries, preferred.dtype};
}

ReaderMoveResult evaluate_exad_result_candidates(
    EXADBookReader &reader,
    const BoardMatrix &board_matrix,
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    int64_t nums_adjust
) {
    if (path_list.empty()) {
        return {question_entries(), {}};
    }
    const int64_t nums = static_cast<int64_t>((board_value_sum(board_matrix) + nums_adjust) / 2);
    if (nums < 0) {
        return {blank_direction_entries(), {}};
    }

    PreferredResultAccumulator preferred;
    const std::string filename = pattern_full + "_" + std::to_string(nums) + ".exadbook";
    const std::optional<fs::path> exadlut_path = first_existing_exad_lut_path(path_list, pattern_full);
    if (!exadlut_path) {
        return {blank_direction_entries(), {}};
    }
    const std::vector<int> operations = operation_sequence(reader.is_variant_, reader.last_operation_index_);

    for (const auto &path_entry : path_list) {
        if (!NativePath::exists(path_entry.first) || preferred.has_best) {
            continue;
        }

        for (int operation_index : operations) {
            const BoardMatrix transformed_board = apply_operation(board_matrix, operation_index);
            const uint64_t encoded = encode_board_matrix(transformed_board);
            const uint64_t physical_encoded =
                apply_sym_like(encoded, static_cast<int>(reader.spec_.physical_transform));
            if (!is_pattern(physical_encoded, reader.spec_.pattern_masks)) {
                continue;
            }

            std::array<SearchValue, 4> result_values = {
                none_search_value(),
                none_search_value(),
                none_search_value(),
                none_search_value(),
            };
            const auto moved_boards = move_all_dir_for_variant(encoded, reader.is_variant_);
            for (size_t index = 0; index < moved_boards.size(); ++index) {
                const uint64_t moved_board = moved_boards[index];
                const uint64_t physical_moved =
                    apply_sym_like(moved_board, static_cast<int>(reader.spec_.physical_transform));
                if (moved_board == encoded || !is_pattern(physical_moved, reader.spec_.pattern_masks)) {
                    continue;
                }
                const SearchValue value = find_exad_value(
                    reader.spec_,
                    reader.masker_,
                    path_entry.first,
                    filename,
                    *exadlut_path,
                    pattern_full,
                    physical_moved,
                    path_entry.second
                );
                if (index == 0U) {
                    result_values[2] = value;
                } else if (index == 1U) {
                    result_values[1] = value;
                } else if (index == 2U) {
                    result_values[3] = value;
                } else {
                    result_values[0] = value;
                }
            }

            std::vector<std::pair<std::string, SearchValue>> adjusted_entries;
            adjusted_entries.reserve(4);
            for (size_t ordered_index = 0; ordered_index < kOrderedResultKeys.size(); ++ordered_index) {
                adjusted_entries.push_back({
                    adjust_direction(operation_index, std::string(kOrderedResultKeys[ordered_index])),
                    result_values[ordered_index],
                });
            }

            const std::vector<OrderedReaderEntry> sorted_entries = sort_adjusted_entries(adjusted_entries);
            if (!first_numeric_value(sorted_entries)) {
                continue;
            }

            if (reader.prefer_max_result_) {
                consider_preferred_result(
                    preferred, sorted_entries, path_entry.second, operation_index);
            } else {
                reader.last_operation_index_ = operation_index;
                return {sorted_entries, path_entry.second};
            }
        }
    }

    if (preferred.has_best) {
        reader.last_operation_index_ = preferred.operation_index;
    }
    return {preferred.entries, preferred.dtype};
}

ReaderMoveResult evaluate_ex_result_candidates(
    EXBookReader &reader,
    const BoardMatrix &board_matrix,
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    int64_t nums_adjust
) {
    if (path_list.empty()) {
        return {question_entries(), {}};
    }
    const int64_t nums = static_cast<int64_t>((board_value_sum(board_matrix) + nums_adjust) / 2);
    if (nums < 0) {
        return {blank_direction_entries(), {}};
    }

    const std::string filename = pattern_full + "_" + std::to_string(nums) + ".zbook";
    const std::optional<fs::path> zlut_path = first_existing_ex_zlut_path(path_list, pattern_full);
    if (!zlut_path) {
        return {blank_direction_entries(), {}};
    }
    const std::vector<int> operations = operation_sequence(reader.is_variant_, reader.last_operation_index_);

    for (const auto &path_entry : path_list) {
        if (!NativePath::exists(path_entry.first)) {
            continue;
        }

        for (int operation_index : operations) {
            const BoardMatrix transformed_board = apply_operation(board_matrix, operation_index);
            const uint64_t encoded = encode_board_matrix(transformed_board);
            const uint64_t physical_encoded =
                apply_sym_like(encoded, static_cast<int>(reader.spec_.physical_transform));
            if (!is_pattern(physical_encoded, reader.spec_.pattern_masks)) {
                continue;
            }

            std::array<SearchValue, 4> result_values = {
                none_search_value(),
                none_search_value(),
                none_search_value(),
                none_search_value(),
            };
            const auto moved_boards = move_all_dir_for_variant(encoded, reader.is_variant_);
            for (size_t index = 0; index < moved_boards.size(); ++index) {
                const uint64_t moved_board = moved_boards[index];
                const uint64_t physical_moved =
                    apply_sym_like(moved_board, static_cast<int>(reader.spec_.physical_transform));
                if (moved_board == encoded || !is_pattern(physical_moved, reader.spec_.pattern_masks)) {
                    continue;
                }
                const SearchValue value = find_ex_value(
                    path_entry.first,
                    filename,
                    *zlut_path,
                    pattern_full,
                    canonical_by_mode(physical_moved, reader.spec_.symm_mode),
                    path_entry.second
                );
                if (index == 0U) {
                    result_values[2] = value;
                } else if (index == 1U) {
                    result_values[1] = value;
                } else if (index == 2U) {
                    result_values[3] = value;
                } else {
                    result_values[0] = value;
                }
            }

            std::vector<std::pair<std::string, SearchValue>> adjusted_entries;
            adjusted_entries.reserve(4);
            for (size_t ordered_index = 0; ordered_index < kOrderedResultKeys.size(); ++ordered_index) {
                adjusted_entries.push_back({
                    adjust_direction(operation_index, std::string(kOrderedResultKeys[ordered_index])),
                    result_values[ordered_index],
                });
            }

            const std::vector<OrderedReaderEntry> sorted_entries = sort_adjusted_entries(adjusted_entries);
            bool has_numeric = false;
            double first_numeric = 0.0;
            for (const auto &entry : sorted_entries) {
                if (entry.kind == ReaderValueKind::Numeric) {
                    has_numeric = true;
                    first_numeric = entry.number;
                    break;
                }
            }
            if (!has_numeric) {
                continue;
            }

            reader.last_operation_index_ = operation_index;
            return {sorted_entries, path_entry.second};
        }
    }

    return {blank_direction_entries(), {}};
}

ReaderMoveResult evaluate_bc_result_candidates(
    BCBookReader &reader,
    const BoardMatrix &board_matrix,
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    int64_t nums_adjust
) {
    if (path_list.empty()) {
        return {question_entries(), "uint32"};
    }

    const std::vector<int> operations = operation_sequence(reader.is_variant_, reader.last_operation_index_);
    for (int operation_index : operations) {
        const BoardMatrix transformed_board = apply_operation(board_matrix, operation_index);
        const uint64_t encoded = encode_board_matrix(transformed_board);
        const uint64_t physical_encoded =
            apply_sym_like(encoded, static_cast<int>(reader.spec_.physical_transform));
        if (!is_pattern(physical_encoded, reader.spec_.pattern_masks)) {
            continue;
        }

        std::array<SearchValue, 4> result_values = {
            none_search_value(),
            none_search_value(),
            none_search_value(),
            none_search_value(),
        };
        std::string dtype_name = "uint32";
        const auto moved_boards = move_all_dir_for_variant(encoded, reader.is_variant_);
        for (size_t index = 0; index < moved_boards.size(); ++index) {
            const uint64_t moved_board = moved_boards[index];
            const uint64_t physical_moved =
                apply_sym_like(moved_board, static_cast<int>(reader.spec_.physical_transform));
            if (moved_board == encoded || !is_pattern(physical_moved, reader.spec_.pattern_masks)) {
                continue;
            }
            const uint64_t canonical_board = canonical_by_mode(physical_moved, reader.spec_.symm_mode);
            const BCSearchResult value = find_bc_value(
                reader,
                path_list,
                pattern_full,
                canonical_board,
                nums_adjust);
            if (value.found) {
                dtype_name = value.dtype;
            }
            if (index == 0U) {
                result_values[2] = value.value;
            } else if (index == 1U) {
                result_values[1] = value.value;
            } else if (index == 2U) {
                result_values[3] = value.value;
            } else {
                result_values[0] = value.value;
            }
        }

        std::vector<std::pair<std::string, SearchValue>> adjusted_entries;
        adjusted_entries.reserve(4);
        for (size_t ordered_index = 0; ordered_index < kOrderedResultKeys.size(); ++ordered_index) {
            adjusted_entries.push_back({
                adjust_direction(operation_index, std::string(kOrderedResultKeys[ordered_index])),
                result_values[ordered_index],
            });
        }

        const std::vector<OrderedReaderEntry> sorted_entries = sort_adjusted_entries(adjusted_entries);
        const bool has_numeric = std::any_of(
            sorted_entries.begin(),
            sorted_entries.end(),
            [](const OrderedReaderEntry &entry) {
                return entry.kind == ReaderValueKind::Numeric;
            });
        if (!has_numeric) {
            continue;
        }

        reader.last_operation_index_ = operation_index;
        return {sorted_entries, dtype_name};
    }

    return {blank_direction_entries(), "uint32"};
}

bool parse_numbered_layer_filename(
    const fs::path &path,
    const std::string &pattern_full,
    const std::string &extension,
    uint32_t &ordinal
) {
    const std::string name = NativePath::to_utf8_string(path.filename());
    const std::string prefix = pattern_full + "_";
    if (name.size() <= prefix.size() + extension.size() ||
        name.compare(0U, prefix.size(), prefix) != 0 ||
        name.compare(name.size() - extension.size(), extension.size(), extension) != 0) {
        return false;
    }
    const std::string ordinal_text =
        name.substr(prefix.size(), name.size() - prefix.size() - extension.size());
    if (ordinal_text.empty()) {
        return false;
    }
    uint64_t parsed = 0U;
    for (char ch : ordinal_text) {
        if (ch < '0' || ch > '9') {
            return false;
        }
        parsed = parsed * 10U + static_cast<uint64_t>(ch - '0');
        if (parsed > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
    }
    ordinal = static_cast<uint32_t>(parsed);
    return true;
}

std::vector<std::pair<uint32_t, fs::path>> bc_compressed_candidates(
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full
) {
    std::vector<std::pair<uint32_t, fs::path>> candidates;
    for (const auto &path_entry : path_list) {
        if (path_entry.first.empty()) {
            continue;
        }
        const fs::path root = NativePath::from_utf8(path_entry.first);
        if (!fs::exists(root) || !fs::is_directory(root)) {
            continue;
        }
        for (const auto &entry : fs::directory_iterator(root)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            uint32_t ordinal = 0U;
            if (parse_numbered_layer_filename(
                    entry.path(),
                    pattern_full,
                    BCCompressedResult::kCompressedLayerFileExtension,
                    ordinal)) {
                candidates.push_back({ordinal, entry.path()});
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.first < rhs.first;
    });
    return candidates;
}

struct BCExactCandidate {
    uint32_t ordinal = 0U;
    fs::path position_path;
    fs::path success_path;
};

uint32_t bc_candidate_ordinal(const std::pair<uint32_t, fs::path> &item) {
    return item.first;
}

uint32_t bc_candidate_ordinal(const BCExactCandidate &item) {
    return item.ordinal;
}

std::vector<BCExactCandidate> bc_exact_candidates(
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full
) {
    struct Partial {
        std::vector<fs::path> position_paths;
        std::vector<fs::path> success_paths;
    };
    std::unordered_map<uint32_t, Partial> partials;
    for (const auto &path_entry : path_list) {
        if (path_entry.first.empty()) {
            continue;
        }
        const fs::path root = NativePath::from_utf8(path_entry.first);
        if (!fs::exists(root) || !fs::is_directory(root)) {
            continue;
        }
        for (const auto &entry : fs::directory_iterator(root)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            uint32_t ordinal = 0U;
            if (parse_numbered_layer_filename(entry.path(), pattern_full, ".bcpos", ordinal)) {
                partials[ordinal].position_paths.push_back(entry.path());
            } else if (parse_numbered_layer_filename(entry.path(), pattern_full, ".bcsuc", ordinal)) {
                partials[ordinal].success_paths.push_back(entry.path());
            }
        }
    }
    std::vector<BCExactCandidate> candidates;
    candidates.reserve(partials.size());
    for (const auto &entry : partials) {
        for (const fs::path &position_path : entry.second.position_paths) {
            for (const fs::path &success_path : entry.second.success_paths) {
                candidates.push_back({
                    entry.first,
                    position_path,
                    success_path,
                });
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.ordinal < rhs.ordinal;
    });
    return candidates;
}

std::vector<uint32_t> exad_layer_ordinals(
    const fs::path &root,
    const std::string &pattern_full
) {
    std::vector<uint32_t> ordinals;
    try {
        if (!fs::exists(root) || !fs::is_directory(root)) {
            return ordinals;
        }
        constexpr std::array<const char *, 3> extensions = {
            ".exadbook",
            EXADCompressedResult::kCompressedLayerFileExtension,
            ".exadbook.exadzbook",
        };
        for (const auto &entry : fs::directory_iterator(root)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            uint32_t ordinal = 0U;
            for (const char *extension : extensions) {
                if (parse_numbered_layer_filename(
                        entry.path(), pattern_full, extension, ordinal)) {
                    ordinals.push_back(ordinal);
                    break;
                }
            }
        }
    } catch (...) {
        return {};
    }
    std::sort(ordinals.begin(), ordinals.end());
    ordinals.erase(std::unique(ordinals.begin(), ordinals.end()), ordinals.end());
    return ordinals;
}

std::optional<uint64_t> playable_exad_sample(
    const EXADCompressedResult::ColdSampleResult &sample,
    const AdvancedPatternSpec &spec,
    const FormationAD::MaskerContext &masker,
    std::mt19937 &rng
) {
    if (!sample.found) {
        return std::nullopt;
    }
    std::vector<uint64_t> boards = FormationAD::unmask_board(
        sample.board,
        sample.original_board_sum,
        masker.tiles_combination_table,
        masker.permutation_table,
        masker.param
    );
    std::shuffle(boards.begin(), boards.end(), rng);
    for (uint64_t physical_board : boards) {
        if (is_pattern(physical_board, spec.pattern_masks)) {
            return apply_sym_like(
                physical_board, static_cast<int>(spec.inverse_physical_transform));
        }
    }
    return std::nullopt;
}

bool bc_result_has_numeric(const ReaderMoveResult &result) {
    return std::any_of(result.entries.begin(), result.entries.end(), [](const OrderedReaderEntry &entry) {
        return entry.kind == ReaderValueKind::Numeric;
    });
}

uint64_t sample_bc_book_state(
    const BCBookReader &reader,
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    int64_t nums_adjust,
    double spawn_rate4
) {
    static thread_local std::mt19937 rng(std::random_device{}());

    auto order_candidates = [](auto candidates) {
        auto split = std::partition(candidates.begin(), candidates.end(), [](const auto &item) {
            return bc_candidate_ordinal(item) < 10U;
        });
        std::shuffle(candidates.begin(), split, rng);
        std::shuffle(split, candidates.end(), rng);
        return candidates;
    };

    for (const auto &item : order_candidates(bc_compressed_candidates(path_list, pattern_full))) {
        for (uint32_t attempt = 0U; attempt < 8U; ++attempt) {
            uint64_t board = 0ULL;
            uint64_t raw_bits = 0ULL;
            double numeric = 0.0;
            try {
                if (!BCRuntime::sample_compressed_result_cached(
                        item.second,
                        reader.target_rank_,
                        board,
                        raw_bits,
                        numeric,
                        0U) ||
                    board == 0ULL) {
                    continue;
                }
            } catch (...) {
                continue;
            }
            const uint64_t logical_board =
                apply_sym_like(board, static_cast<int>(reader.spec_.inverse_physical_transform));
            const uint64_t spawned = gen_new_num(logical_board, static_cast<float>(spawn_rate4)).first;
            if (spawned == 0ULL) {
                continue;
            }
            BCBookReader probe = reader;
            if (bc_result_has_numeric(evaluate_bc_result_candidates(
                    probe,
                    decode_board_matrix(spawned),
                    path_list,
                    pattern_full,
                    nums_adjust))) {
                return spawned;
            }
        }
    }

    for (const auto &item : order_candidates(bc_exact_candidates(path_list, pattern_full))) {
        for (uint32_t attempt = 0U; attempt < 8U; ++attempt) {
            uint64_t board = 0ULL;
            try {
                board = BCRuntime::sample_exact_random_board_cached(
                    item.position_path,
                    item.success_path,
                    reader.target_rank_);
            } catch (...) {
                continue;
            }
            if (board == 0ULL) {
                continue;
            }
            const uint64_t logical_board =
                apply_sym_like(board, static_cast<int>(reader.spec_.inverse_physical_transform));
            const uint64_t spawned = gen_new_num(logical_board, static_cast<float>(spawn_rate4)).first;
            if (spawned == 0ULL) {
                continue;
            }
            BCBookReader probe = reader;
            if (bc_result_has_numeric(evaluate_bc_result_candidates(
                    probe,
                    decode_board_matrix(spawned),
                    path_list,
                    pattern_full,
                    nums_adjust))) {
                return spawned;
            }
        }
    }

    return 0ULL;
}

uint64_t sample_classic_book_state(
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    double spawn_rate4
) {
    static thread_local std::mt19937 rng(std::random_device{}());
    for (const auto &path_entry : path_list) {
        std::vector<int> book_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        const DTypeInfo dtype_info = dtype_info_for_name(path_entry.second);
        size_t record_size = sizeof(SuccessEntry<uint32_t>);
        if (dtype_info.kind == SuccessRateKind::UInt64) {
            record_size = sizeof(SuccessEntry<uint64_t>);
        } else if (dtype_info.kind == SuccessRateKind::Float32) {
            record_size = sizeof(SuccessEntry<float>);
        } else if (dtype_info.kind == SuccessRateKind::Float64) {
            record_size = sizeof(SuccessEntry<double>);
        }

        while (!book_indices.empty()) {
            std::uniform_int_distribution<size_t> pick(0, book_indices.size() - 1U);
            const size_t chosen = pick(rng);
            const int book_id = book_indices[chosen];
            book_indices.erase(book_indices.begin() + static_cast<ptrdiff_t>(chosen));

            const fs::path filepath = NativePath::from_utf8(path_entry.first) / (pattern_full + "_" + std::to_string(book_id) + ".book");
            if (!fs::exists(filepath)) {
                continue;
            }

            std::ifstream file(filepath, std::ios::binary | std::ios::ate);
            if (!file) {
                continue;
            }
            const size_t num_records = static_cast<size_t>(file.tellg()) / record_size;
            if (num_records == 0U) {
                continue;
            }
            std::uniform_int_distribution<size_t> record_pick(0, num_records - 1U);
            const size_t record_index = record_pick(rng);
            file.seekg(static_cast<std::streamoff>(record_index * record_size), std::ios::beg);
            uint64_t state = 0ULL;
            file.read(reinterpret_cast<char *>(&state), static_cast<std::streamsize>(sizeof(uint64_t)));
            if (state != 0ULL) {
                return gen_new_num(state, static_cast<float>(spawn_rate4)).first;
            }
        }
    }
    return 0ULL;
}

uint64_t sample_advanced_book_state(
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    double spawn_rate4
) {
    static thread_local std::mt19937 rng(std::random_device{}());
    for (const auto &path_entry : path_list) {
        std::vector<int> book_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        while (!book_indices.empty()) {
            std::uniform_int_distribution<size_t> pick(0, book_indices.size() - 1U);
            const size_t chosen = pick(rng);
            const int book_id = book_indices[chosen];
            book_indices.erase(book_indices.begin() + static_cast<ptrdiff_t>(chosen));

            const fs::path filepath = NativePath::from_utf8(path_entry.first) / (pattern_full + "_" + std::to_string(book_id) + "b");
            if (!fs::exists(filepath) || !fs::is_directory(filepath)) {
                continue;
            }

            std::vector<fs::path> index_files;
            for (const auto &entry : fs::directory_iterator(filepath)) {
                if (entry.path().extension() == ".i") {
                    index_files.push_back(entry.path());
                }
            }
            if (index_files.empty()) {
                continue;
            }

            std::uniform_int_distribution<size_t> index_pick(0, index_files.size() - 1U);
            const fs::path &index_path = index_files[index_pick(rng)];
            std::ifstream file(index_path, std::ios::binary | std::ios::ate);
            if (!file) {
                continue;
            }

            const size_t num_records = static_cast<size_t>(file.tellg()) / sizeof(uint64_t);
            if (num_records == 0U) {
                continue;
            }
            std::uniform_int_distribution<size_t> record_pick(0, num_records - 1U);
            const size_t record_index = record_pick(rng);
            file.seekg(static_cast<std::streamoff>(record_index * sizeof(uint64_t)), std::ios::beg);
            uint64_t state = 0ULL;
            file.read(reinterpret_cast<char *>(&state), static_cast<std::streamsize>(sizeof(uint64_t)));
            if (state != 0ULL) {
                return gen_new_num(state, static_cast<float>(spawn_rate4)).first;
            }
        }
    }
    return 0ULL;
}

uint64_t sample_exad_book_state(
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    const AdvancedPatternSpec &spec,
    const FormationAD::MaskerContext &masker,
    double spawn_rate4
) {
    static thread_local std::mt19937 rng(std::random_device{}());
    const std::optional<fs::path> exadlut_path = first_existing_exad_lut_path(path_list, pattern_full);
    if (!exadlut_path) {
        return 0ULL;
    }
    for (const auto &path_entry : path_list) {
        const fs::path root = NativePath::from_utf8(path_entry.first);
        std::vector<uint32_t> layer_ordinals = exad_layer_ordinals(root, pattern_full);
        const auto first_above_initial_range = std::upper_bound(
            layer_ordinals.begin(), layer_ordinals.end(), 9U);
        if (first_above_initial_range != layer_ordinals.begin()) {
            layer_ordinals.erase(first_above_initial_range, layer_ordinals.end());
            std::shuffle(layer_ordinals.begin(), layer_ordinals.end(), rng);
        } else if (layer_ordinals.size() > 1U) {
            layer_ordinals.resize(1U);
        }
        for (uint32_t layer_ordinal : layer_ordinals) {
            const std::string layer_name = pattern_full + "_" + std::to_string(layer_ordinal);
            const fs::path exadbook_path = root / (layer_name + ".exadbook");
            EXADCompressedResult::ColdSampleResult sample;

            if (fs::exists(exadbook_path)) {
                sample = EXADCompressedResult::sample_exadbook_cold(
                    NativePath::to_utf8_string(exadbook_path),
                    NativePath::to_utf8_string(*exadlut_path));
            }
            if (!sample.found) {
                for (const fs::path &candidate : exad_compressed_candidates(exadbook_path)) {
                    if (!fs::exists(candidate)) {
                        continue;
                    }
                    sample = EXADCompressedResult::sample_exad_cold(
                        NativePath::to_utf8_string(candidate),
                        NativePath::to_utf8_string(*exadlut_path));
                    if (sample.found) {
                        break;
                    }
                }
            }
            const std::optional<uint64_t> state = playable_exad_sample(
                sample, spec, masker, rng);
            if (state) {
                return gen_new_num(*state, static_cast<float>(spawn_rate4)).first;
            }
        }
    }
    return 0ULL;
}

uint64_t sample_ex_book_state(
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    int inverse_transform,
    double spawn_rate4
) {
    static thread_local std::mt19937 rng(std::random_device{}());
    const std::optional<fs::path> zlut_path = first_existing_ex_zlut_path(path_list, pattern_full);
    if (!zlut_path) {
        return 0ULL;
    }
    for (const auto &path_entry : path_list) {
        std::vector<int> book_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        while (!book_indices.empty()) {
            std::uniform_int_distribution<size_t> pick(0, book_indices.size() - 1U);
            const size_t chosen = pick(rng);
            const int book_id = book_indices[chosen];
            book_indices.erase(book_indices.begin() + static_cast<ptrdiff_t>(chosen));

            const fs::path zbook_path =
                NativePath::from_utf8(path_entry.first) / (pattern_full + "_" + std::to_string(book_id) + ".zbook");
            uint64_t state = 0ULL;
            if (fs::exists(zbook_path) && sample_ex_zbook_state(zbook_path, *zlut_path, rng, state)) {
                return gen_new_num(apply_sym_like(state, inverse_transform), static_cast<float>(spawn_rate4)).first;
            }
            if (!fs::exists(zbook_path)) {
                for (const fs::path &candidate : ex_compressed_candidates(zbook_path)) {
                    if (!fs::exists(candidate)) {
                        continue;
                    }
                    uint64_t raw = 0ULL;
                    double numeric = 0.0;
                    if (EXCompressedResult::sample_cold(
                            NativePath::to_utf8_string(candidate),
                            NativePath::to_utf8_string(*zlut_path),
                            state,
                            raw,
                            numeric)) {
                        return gen_new_num(apply_sym_like(state, inverse_transform), static_cast<float>(spawn_rate4)).first;
                    }
                }
            }
        }
    }
    return 0ULL;
}

} // namespace

namespace BCRuntime {

std::vector<uint8_t> legal_tiles(uint32_t target_rank) {
    return bc_legal_tiles_for_rank(target_rank);
}

std::string dtype_name(uint32_t dtype) {
    return bc_dtype_name(dtype);
}

double normalize_lookup_value(const BCCompressedResult::ColdLookupResult &lookup) {
    return normalize_bc_lookup_value(lookup);
}

BCCompressedResult::ColdLookupResult lookup_compressed_result_cached(
    const std::filesystem::path &compressed_path,
    uint32_t target_rank,
    uint64_t board,
    uint32_t lane
) {
    return cached_bc_compressed_reader(compressed_path, target_rank)->lookup(board, lane);
}

BCCompressedResult::ColdLookupResult lookup_exact_result_cached(
    const std::filesystem::path &position_path,
    const std::filesystem::path &success_path,
    uint32_t target_rank,
    uint64_t board,
    uint32_t lane
) {
    return cached_bc_exact_reader(position_path, success_path, target_rank)->lookup(board, lane);
}

bool sample_compressed_result_cached(
    const std::filesystem::path &compressed_path,
    uint32_t target_rank,
    uint64_t &board,
    uint64_t &raw_value_bits,
    double &numeric_value,
    uint32_t lane
) {
    return cached_bc_compressed_reader(compressed_path, target_rank)->sample(
        board,
        raw_value_bits,
        numeric_value,
        lane);
}

uint64_t sample_exact_random_board_cached(
    const std::filesystem::path &position_path,
    const std::filesystem::path &success_path,
    uint32_t target_rank
) {
    return cached_bc_exact_reader(position_path, success_path, target_rank)->sample_board();
}

uint64_t sample_exact_random_board_cached(
    const std::filesystem::path &position_path,
    uint32_t target_rank
) {
    std::filesystem::path success_path = position_path;
    success_path.replace_extension(".bcsuc");
    return sample_exact_random_board_cached(position_path, success_path, target_rank);
}

} // namespace BCRuntime

ClassicBookReader::ClassicBookReader(PatternSpec spec, bool is_variant)
    : spec_(std::move(spec)),
      is_variant_(is_variant),
      prefer_max_result_(spec_.name == "4442ff" || spec_.name == "4442f" || spec_.name == "4tiler") {}

ReaderMoveResult ClassicBookReader::move_on_dic(
    const std::vector<std::vector<int>> &board,
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    int64_t nums_adjust
) {
    BoardMatrix board_matrix{};
    for (size_t row = 0; row < std::min<size_t>(4U, board.size()); ++row) {
        for (size_t col = 0; col < std::min<size_t>(4U, board[row].size()); ++col) {
            board_matrix[row][col] = static_cast<uint32_t>(board[row][col]);
        }
    }
    return evaluate_classic_result_candidates(*this, board_matrix, path_list, pattern_full, nums_adjust);
}

uint64_t ClassicBookReader::get_random_state(
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    double spawn_rate4
) const {
    return sample_classic_book_state(path_list, pattern_full, spawn_rate4);
}

AdvancedBookReader::AdvancedBookReader(AdvancedPatternSpec spec, bool is_variant)
    : spec_(std::move(spec)),
      masker_(FormationAD::init_masker(spec_)),
      is_variant_(is_variant),
      prefer_max_result_(spec_.name == "4442ff" || spec_.name == "4442f" || spec_.name == "4tiler") {}

ReaderMoveResult AdvancedBookReader::move_on_dic(
    const std::vector<std::vector<int>> &board,
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    int64_t nums_adjust
) {
    BoardMatrix board_matrix{};
    for (size_t row = 0; row < std::min<size_t>(4U, board.size()); ++row) {
        for (size_t col = 0; col < std::min<size_t>(4U, board[row].size()); ++col) {
            board_matrix[row][col] = static_cast<uint32_t>(board[row][col]);
        }
    }
    return evaluate_advanced_result_candidates(*this, board_matrix, path_list, pattern_full, nums_adjust);
}

uint64_t AdvancedBookReader::get_random_state(
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    double spawn_rate4
) const {
    return sample_advanced_book_state(path_list, pattern_full, spawn_rate4);
}

EXADBookReader::EXADBookReader(AdvancedPatternSpec spec, bool is_variant)
    : spec_(std::move(spec)),
      masker_(FormationAD::init_masker(spec_)),
      is_variant_(is_variant),
      prefer_max_result_(spec_.name == "4442ff" || spec_.name == "4442f" || spec_.name == "4tiler") {}

ReaderMoveResult EXADBookReader::move_on_dic(
    const std::vector<std::vector<int>> &board,
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    int64_t nums_adjust
) {
    BoardMatrix board_matrix{};
    for (size_t row = 0; row < std::min<size_t>(4U, board.size()); ++row) {
        for (size_t col = 0; col < std::min<size_t>(4U, board[row].size()); ++col) {
            board_matrix[row][col] = static_cast<uint32_t>(board[row][col]);
        }
    }
    return evaluate_exad_result_candidates(*this, board_matrix, path_list, pattern_full, nums_adjust);
}

uint64_t EXADBookReader::get_random_state(
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    double spawn_rate4
) const {
    return sample_exad_book_state(
        path_list,
        pattern_full,
        spec_,
        masker_,
        spawn_rate4
    );
}

EXBookReader::EXBookReader(PatternSpec spec, bool is_variant)
    : spec_(std::move(spec)),
      is_variant_(is_variant) {}

ReaderMoveResult EXBookReader::move_on_dic(
    const std::vector<std::vector<int>> &board,
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    int64_t nums_adjust
) {
    BoardMatrix board_matrix{};
    for (size_t row = 0; row < std::min<size_t>(4U, board.size()); ++row) {
        for (size_t col = 0; col < std::min<size_t>(4U, board[row].size()); ++col) {
            board_matrix[row][col] = static_cast<uint32_t>(board[row][col]);
        }
    }
    return evaluate_ex_result_candidates(*this, board_matrix, path_list, pattern_full, nums_adjust);
}

uint64_t EXBookReader::get_random_state(
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    double spawn_rate4
) const {
    return sample_ex_book_state(
        path_list,
        pattern_full,
        static_cast<int>(spec_.inverse_physical_transform),
        spawn_rate4
    );
}

BCBookReader::BCBookReader(PatternSpec spec, uint32_t target_rank, bool is_variant)
    : spec_(std::move(spec)),
      target_rank_(target_rank),
      is_variant_(is_variant) {}

ReaderMoveResult BCBookReader::move_on_dic(
    const std::vector<std::vector<int>> &board,
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    int64_t nums_adjust
) {
    BoardMatrix board_matrix{};
    for (size_t row = 0; row < std::min<size_t>(4U, board.size()); ++row) {
        for (size_t col = 0; col < std::min<size_t>(4U, board[row].size()); ++col) {
            board_matrix[row][col] = static_cast<uint32_t>(board[row][col]);
        }
    }
    return evaluate_bc_result_candidates(*this, board_matrix, path_list, pattern_full, nums_adjust);
}

uint64_t BCBookReader::get_random_state(
    const std::vector<std::pair<std::string, std::string>> &path_list,
    const std::string &pattern_full,
    double spawn_rate4,
    int64_t nums_adjust
) const {
    return sample_bc_book_state(*this, path_list, pattern_full, nums_adjust, spawn_rate4);
}

double find_classic_value_native(
    const std::string &pathname,
    const std::string &filename,
    uint64_t search_key,
    const std::string &success_rate_dtype,
    bool &found
) {
    const DTypeInfo dtype_info = dtype_info_for_name(success_rate_dtype);
    const fs::path path = NativePath::from_utf8(pathname) / filename;
    double result = 0.0;
    switch (dtype_info.kind) {
        case SuccessRateKind::UInt64:
            found = find_success_entry_in_file<uint64_t>(path, search_key, dtype_info, result);
            break;
        case SuccessRateKind::Float32:
            found = find_success_entry_in_file<float>(path, search_key, dtype_info, result);
            break;
        case SuccessRateKind::Float64:
            found = find_success_entry_in_file<double>(path, search_key, dtype_info, result);
            break;
        case SuccessRateKind::UInt32:
        default:
            found = find_success_entry_in_file<uint32_t>(path, search_key, dtype_info, result);
            break;
    }
    return result;
}

std::optional<double> trie_decompress_search_cached_native(
    const std::string &path_prefix,
    uint64_t board,
    const std::string &success_rate_dtype
) {
    const DTypeInfo dtype_info = dtype_info_for_name(success_rate_dtype);
    const std::vector<TrieNode32> ind = read_binary_vector<TrieNode32>(path_prefix + "i");
    const std::vector<TrieSegmentEntry> segments = read_binary_vector<TrieSegmentEntry>(path_prefix + "s");
    return trie_search_dispatch(path_prefix, board, dtype_info, ind, segments);
}
