#include "ReaderRuntime.h"

#include "BoardCodec.h"
#include "BoardMover.h"
#include "FileIOUtils.h"
#include "Formation.h"
#include "NativeLzma.h"
#include "SymmetryUtils.h"
#include "TrieCompression.h"
#include "VBoardMover.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
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
    while (low <= high) {
        const size_t mid = (low + high) / 2ULL;
        if (segments[mid].index < pos) {
            low = mid + 1ULL;
        } else {
            return {segments[mid - 1].file_offset, segments[mid].file_offset};
        }
    }
    return {0ULL, 0ULL};
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

    std::ifstream ii_file(path_prefix + "ii", std::ios::binary);
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
    size_t last_pos = 0;
    bool found = false;
    size_t last_low = 0;
    size_t last_high = ind3_seg.size() - 2ULL;
    while (last_low <= last_high) {
        last_pos = (last_low + last_high) / 2ULL + 1ULL;
        if (ind3_seg[last_pos].key == target_prefix) {
            found = true;
            last_pos -= 1ULL;
            break;
        }
        if (ind3_seg[last_pos].key < target_prefix) {
            last_low = (last_pos - 1ULL) + 1ULL;
        } else {
            if (last_pos <= 1ULL) {
                break;
            }
            last_high = (last_pos - 1ULL) - 1ULL;
        }
    }
    if (!found) {
        return dtype_info.zero_value;
    }

    const auto [start, end] = get_segment_position(segments, static_cast<uint32_t>(last_pos + low));
    std::ifstream z_file(path_prefix + "z", std::ios::binary);
    if (!z_file || end < start) {
        return std::nullopt;
    }
    z_file.seekg(static_cast<std::streamoff>(start), std::ios::beg);
    std::vector<uint8_t> compressed(static_cast<size_t>(end - start));
    if (!compressed.empty()) {
        FileIOUtils::read_exact(z_file, compressed.data(), compressed.size(), path_prefix + "z");
    }
    std::vector<uint8_t> decompressed = decompress_xz_block_native(compressed.data(), compressed.size());
    const auto *block = reinterpret_cast<const CompactBookEntry<T> *>(decompressed.data());
    const size_t block_size = decompressed.size() / sizeof(CompactBookEntry<T>);
    const uint32_t target = static_cast<uint32_t>(board & 0xFFFFFFFFULL);

    size_t sub_low = static_cast<size_t>(ind3_seg[last_pos].next);
    size_t sub_high = static_cast<size_t>(ind3_seg[last_pos + 1].next) + 1ULL;
    if (sub_high == 1ULL) {
        if (block_size != 0 && target == block[0].lower32) {
            return normalize_raw_value(block[0].success, dtype_info.max_scale);
        }
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
    context.book_path = fs::path(pathname) / filename;
    context.compressed_dir = context.book_path;
    context.compressed_dir.replace_extension(".z");
    context.book_exists = fs::exists(context.book_path);
    context.compressed_exists = fs::exists(context.compressed_dir);
    if (context.compressed_exists) {
        const std::string stem = filename.substr(0, filename.size() - 4U);
        context.prefix = (context.compressed_dir / stem).string();
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

    const fs::path root = fs::path(pathname) / filename;
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
        ind = find_value_uint64_compressed_native(compressed_index_path.string(), search_key);
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

    std::vector<OrderedReaderEntry> final_results = blank_direction_entries();
    double max_success_rate = 0.0;
    std::string success_rate_dtype;
    const std::string filename = pattern_full + "_" + std::to_string(nums) + ".book";
    const std::vector<int> operations = operation_sequence(reader.is_variant_, reader.last_operation_index_);

    for (const auto &path_entry : path_list) {
        if (!fs::exists(path_entry.first) || max_success_rate > 0.0) {
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
            if (reader.prefer_max_result_) {
                if (first_numeric > max_success_rate) {
                    max_success_rate = first_numeric;
                    final_results = sorted_entries;
                    success_rate_dtype = path_entry.second;
                }
            } else {
                return {sorted_entries, path_entry.second};
            }
        }
    }

    return {final_results, success_rate_dtype};
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

    std::vector<OrderedReaderEntry> final_results = blank_direction_entries();
    double max_success_rate = 0.0;
    std::string success_rate_dtype;
    const std::string filename = pattern_full + "_" + std::to_string(nums) + "b";
    const std::vector<int> operations = operation_sequence(reader.is_variant_, reader.last_operation_index_);

    for (const auto &path_entry : path_list) {
        if (!fs::exists(path_entry.first) || max_success_rate > 0.0) {
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
            if (reader.prefer_max_result_) {
                if (first_numeric > max_success_rate) {
                    max_success_rate = first_numeric;
                    final_results = sorted_entries;
                    success_rate_dtype = path_entry.second;
                }
            } else {
                return {sorted_entries, path_entry.second};
            }
        }
    }

    return {final_results, success_rate_dtype};
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

            const fs::path filepath = fs::path(path_entry.first) / (pattern_full + "_" + std::to_string(book_id) + ".book");
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

            const fs::path filepath = fs::path(path_entry.first) / (pattern_full + "_" + std::to_string(book_id) + "b");
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

} // namespace

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

double find_classic_value_native(
    const std::string &pathname,
    const std::string &filename,
    uint64_t search_key,
    const std::string &success_rate_dtype,
    bool &found
) {
    const DTypeInfo dtype_info = dtype_info_for_name(success_rate_dtype);
    const fs::path path = fs::path(pathname) / filename;
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
