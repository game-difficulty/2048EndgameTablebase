#include "TrieCompression.h"

#include "FormationRuntime.h"
#include "NativeLzma.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <omp.h>

namespace fs = std::filesystem;

namespace {

constexpr size_t kTrieParallelThreshold = 2097152ULL;
constexpr size_t kTrieBlockTarget = 32768ULL;

#pragma pack(push, 1)
struct TrieNode32 {
    uint8_t key;
    uint32_t next;
};

struct TrieNode64 {
    uint8_t key;
    uint64_t next;
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

static_assert(sizeof(TrieNode32) == 5, "Unexpected trie node32 layout");
static_assert(sizeof(TrieNode64) == 9, "Unexpected trie node64 layout");
static_assert(sizeof(TrieNode16) == 3, "Unexpected trie node16 layout");
static_assert(sizeof(TrieSegmentEntry) == 12, "Unexpected trie segment entry layout");

#pragma pack(push, 1)
template <typename T>
struct CompactBookEntry {
    uint32_t lower32;
    T success;
};
#pragma pack(pop)

static_assert(sizeof(CompactBookEntry<uint32_t>) == 8, "Unexpected compact uint32 layout");
static_assert(sizeof(CompactBookEntry<uint64_t>) == 12, "Unexpected compact uint64 layout");
static_assert(sizeof(CompactBookEntry<float>) == 8, "Unexpected compact float32 layout");
static_assert(sizeof(CompactBookEntry<double>) == 12, "Unexpected compact float64 layout");

template <typename T>
struct ValueTraits;

template <>
struct ValueTraits<uint32_t> {
    static constexpr double max_scale = 4e9;
    static constexpr uint32_t zero_value = 0U;
};

template <>
struct ValueTraits<uint64_t> {
    static constexpr double max_scale = 1.6e18;
    static constexpr uint64_t zero_value = 0ULL;
};

template <>
struct ValueTraits<float> {
    static constexpr double max_scale = 1.0;
    static constexpr float zero_value = 0.0f;
};

template <>
struct ValueTraits<double> {
    static constexpr double max_scale = 1.0;
    static constexpr double zero_value = 0.0;
};

template <typename T>
struct TrieIndexData {
    std::vector<TrieNode32> ind0;
    std::vector<TrieNode32> ind1;
    std::vector<TrieNode32> ind2;
    std::vector<TrieNode64> ind3;
    std::vector<CompactBookEntry<T>> book;
};

struct SegmentInfo {
    uint32_t start;
    uint32_t end;
    uint32_t split_index;
};

template <typename T>
void write_binary_vector(const fs::path &path, const std::vector<T> &data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return;
    }
    if (!data.empty()) {
        out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    }
}

template <typename T>
std::vector<T> read_binary_vector(const fs::path &path) {
    std::vector<T> data;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return data;
    }
    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    data.resize(size / sizeof(T));
    if (!data.empty()) {
        file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    }
    return data;
}

template <typename T>
TrieIndexData<T> build_trie_index(const std::vector<SuccessEntry<T>> &rows) {
    TrieIndexData<T> data;
    if (rows.empty()) {
        return data;
    }

    data.ind0.push_back({0, 0});
    data.ind1.push_back({0, 0});
    data.ind2.push_back({0, 0});
    data.ind3.push_back({0, 0});
    data.book.resize(rows.size());

    for (size_t i = 0; i < rows.size(); ++i) {
        data.book[i].lower32 = static_cast<uint32_t>(rows[i].board & 0xFFFFFFFFULL);
        data.book[i].success = rows[i].success;
    }

    auto key_f4 = [](uint64_t board) { return static_cast<uint8_t>((board >> 56U) & 0xFFU); };
    auto key_f3 = [](uint64_t board) { return static_cast<uint8_t>((board >> 48U) & 0xFFU); };
    auto key_f2 = [](uint64_t board) { return static_cast<uint8_t>((board >> 40U) & 0xFFU); };
    auto key_f1 = [](uint64_t board) { return static_cast<uint8_t>((board >> 32U) & 0xFFU); };

    uint8_t d0 = key_f4(rows.front().board);
    uint8_t d1 = key_f3(rows.front().board);
    uint8_t d2 = key_f2(rows.front().board);
    uint8_t d3 = key_f1(rows.front().board);

    size_t last_index = 0;
    for (size_t j = 0; j < rows.size(); ++j) {
        last_index = j;
        const uint64_t board = rows[j].board;
        const uint8_t f4 = key_f4(board);
        const uint8_t f3 = key_f3(board);
        const uint8_t f2 = key_f2(board);
        const uint8_t f1 = key_f1(board);

        if (f4 != d0) {
            data.ind0.push_back({d0, static_cast<uint32_t>(data.ind1.size())});
            d0 = f4;
            data.ind1.push_back({d1, static_cast<uint32_t>(data.ind2.size())});
            d1 = f3;
            data.ind2.push_back({d2, static_cast<uint32_t>(data.ind3.size())});
            d2 = f2;
            data.ind3.push_back({d3, static_cast<uint64_t>(j - 1)});
            d3 = f1;
        } else if (f3 != d1) {
            data.ind1.push_back({d1, static_cast<uint32_t>(data.ind2.size())});
            d1 = f3;
            data.ind2.push_back({d2, static_cast<uint32_t>(data.ind3.size())});
            d2 = f2;
            data.ind3.push_back({d3, static_cast<uint64_t>(j - 1)});
            d3 = f1;
        } else if (f2 != d2) {
            data.ind2.push_back({d2, static_cast<uint32_t>(data.ind3.size())});
            d2 = f2;
            data.ind3.push_back({d3, static_cast<uint64_t>(j - 1)});
            d3 = f1;
        } else if (f1 != d3) {
            data.ind3.push_back({d3, static_cast<uint64_t>(j - 1)});
            d3 = f1;
        }
    }

    data.ind0.push_back({d0, static_cast<uint32_t>(data.ind1.size())});
    data.ind1.push_back({d1, static_cast<uint32_t>(data.ind2.size())});
    data.ind2.push_back({d2, static_cast<uint32_t>(data.ind3.size())});
    data.ind3.push_back({d3, static_cast<uint64_t>(last_index)});
    return data;
}

template <typename T>
std::vector<TrieSegmentEntry> compress_and_save_serial(
    std::vector<TrieNode64> &ind3,
    const std::vector<CompactBookEntry<T>> &book,
    const fs::path &z_path
) {
    std::vector<TrieSegmentEntry> segments;
    segments.reserve(book.size() / 16000ULL + 2ULL);
    segments.push_back({0U, 0ULL});

    uint64_t current_size = 0;
    size_t start = 0;
    std::ofstream out(z_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return {};
    }

    for (size_t i = 1; i + 1 < ind3.size(); ++i) {
        const size_t end = static_cast<size_t>(ind3[i].next) + 1ULL;
        if (end - start > kTrieBlockTarget) {
            const uint8_t *bytes = reinterpret_cast<const uint8_t *>(book.data() + start);
            const size_t byte_count = (end - start) * sizeof(CompactBookEntry<T>);
            std::vector<uint8_t> block = compress_xz_block_native(bytes, byte_count, 1);
            if (!block.empty()) {
                out.write(reinterpret_cast<const char *>(block.data()), static_cast<std::streamsize>(block.size()));
            }
            current_size += static_cast<uint64_t>(block.size());
            ind3[i].next = 0;
            start = end;
            segments.push_back({static_cast<uint32_t>(i), current_size});
        } else {
            ind3[i].next -= static_cast<uint64_t>(start);
        }
    }

    const uint8_t *tail_bytes = reinterpret_cast<const uint8_t *>(book.data() + start);
    const size_t tail_count = (book.size() - start) * sizeof(CompactBookEntry<T>);
    std::vector<uint8_t> block = compress_xz_block_native(tail_bytes, tail_count, 1);
    if (!block.empty()) {
        out.write(reinterpret_cast<const char *>(block.data()), static_cast<std::streamsize>(block.size()));
    }
    current_size += static_cast<uint64_t>(block.size());
    if (!ind3.empty()) {
        ind3.back().next = 0;
    }
    segments.push_back({static_cast<uint32_t>(ind3.size() - 1), current_size});
    return segments;
}

inline std::pair<std::vector<SegmentInfo>, std::vector<TrieNode64>> get_segment_positions(std::vector<TrieNode64> ind3, size_t book_size) {
    std::vector<SegmentInfo> infos;
    infos.reserve(ind3.empty() ? 0ULL : static_cast<size_t>(ind3.back().next / 16384ULL) + 1ULL);

    size_t start = 0;
    for (size_t i = 1; i + 1 < ind3.size(); ++i) {
        const size_t end = static_cast<size_t>(ind3[i].next) + 1ULL;
        if (end - start > kTrieBlockTarget) {
            infos.push_back({static_cast<uint32_t>(start), static_cast<uint32_t>(end), static_cast<uint32_t>(i)});
            start = end;
            ind3[i].next = 0;
        } else {
            ind3[i].next -= static_cast<uint64_t>(start);
        }
    }
    if (!ind3.empty()) {
        ind3.back().next = 0;
        infos.push_back({static_cast<uint32_t>(start), static_cast<uint32_t>(book_size), static_cast<uint32_t>(ind3.size() - 1)});
    }
    return {std::move(infos), std::move(ind3)};
}

template <typename T>
std::vector<TrieSegmentEntry> compress_and_save_parallel(
    std::vector<TrieNode64> &ind3,
    const std::vector<CompactBookEntry<T>> &book,
    const fs::path &z_path
) {
    auto [infos, updated_ind3] = get_segment_positions(ind3, book.size());
    ind3 = std::move(updated_ind3);

    std::vector<std::vector<uint8_t>> blocks(infos.size());
    #pragma omp parallel for schedule(dynamic, 8)
    for (int64_t idx = 0; idx < static_cast<int64_t>(infos.size()); ++idx) {
        const SegmentInfo &info = infos[static_cast<size_t>(idx)];
        const uint8_t *bytes = reinterpret_cast<const uint8_t *>(book.data() + info.start);
        const size_t byte_count = static_cast<size_t>(info.end - info.start) * sizeof(CompactBookEntry<T>);
        blocks[static_cast<size_t>(idx)] = compress_xz_block_native(bytes, byte_count, 1);
    }

    std::vector<TrieSegmentEntry> segments;
    segments.reserve(infos.size() + 1ULL);
    segments.push_back({0U, 0ULL});

    uint64_t current_size = 0;
    std::ofstream out(z_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return {};
    }
    for (size_t idx = 0; idx < infos.size(); ++idx) {
        const auto &block = blocks[idx];
        if (!block.empty()) {
            out.write(reinterpret_cast<const char *>(block.data()), static_cast<std::streamsize>(block.size()));
        }
        current_size += static_cast<uint64_t>(block.size());
        segments.push_back({infos[idx].split_index, current_size});
    }
    return segments;
}

template <typename T>
bool trie_compress_typed(const std::string &book_path) {
    const fs::path input_path(book_path);
    std::ifstream in(input_path, std::ios::binary | std::ios::ate);
    if (!in) {
        return false;
    }
    const size_t size = static_cast<size_t>(in.tellg());
    if (size == 0 || size % sizeof(SuccessEntry<T>) != 0) {
        return false;
    }
    in.seekg(0, std::ios::beg);
    std::vector<SuccessEntry<T>> rows(size / sizeof(SuccessEntry<T>));
    in.read(reinterpret_cast<char *>(rows.data()), static_cast<std::streamsize>(size));

    TrieIndexData<T> data = build_trie_index(rows);
    fs::path final_dir = input_path;
    final_dir.replace_extension(".z");
    if (fs::exists(final_dir)) {
        fs::remove_all(final_dir);
    }
    fs::create_directories(final_dir);

    const std::string filename = input_path.filename().string();
    const std::string stem = filename.substr(0, filename.size() - 4U);
    const fs::path prefix = final_dir / stem;
    std::vector<TrieSegmentEntry> segments =
        (data.book.size() >= kTrieParallelThreshold)
            ? compress_and_save_parallel(data.ind3, data.book, prefix.string() + "z")
            : compress_and_save_serial(data.ind3, data.book, prefix.string() + "z");
    if (segments.empty()) {
        return false;
    }

    for (auto &entry : data.ind0) {
        entry.next += 1U + static_cast<uint32_t>(data.ind0.size());
    }
    for (auto &entry : data.ind1) {
        entry.next += 1U + static_cast<uint32_t>(data.ind0.size()) + static_cast<uint32_t>(data.ind1.size());
    }

    std::vector<TrieNode32> ind;
    ind.reserve(1ULL + data.ind0.size() + data.ind1.size() + data.ind2.size());
    ind.push_back({0U, 1U});
    ind.insert(ind.end(), data.ind0.begin(), data.ind0.end());
    ind.insert(ind.end(), data.ind1.begin(), data.ind1.end());
    ind.insert(ind.end(), data.ind2.begin(), data.ind2.end());

    std::vector<TrieNode16> ind3_short(data.ind3.size());
    for (size_t i = 0; i < data.ind3.size(); ++i) {
        ind3_short[i].key = data.ind3[i].key;
        ind3_short[i].next = static_cast<uint16_t>(data.ind3[i].next);
    }

    write_binary_vector(prefix.string() + "ii", ind3_short);
    write_binary_vector(prefix.string() + "i", ind);
    write_binary_vector(prefix.string() + "s", segments);
    return true;
}

template <typename T>
double normalize_value(T value) {
    if constexpr (std::is_floating_point_v<T>) {
        return static_cast<double>(value);
    } else {
        return static_cast<double>(value) / ValueTraits<T>::max_scale;
    }
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
std::optional<double> trie_decompress_search_typed(const std::string &path_prefix, uint64_t board) {
    const fs::path prefix(path_prefix);
    const std::vector<TrieNode32> ind = read_binary_vector<TrieNode32>(prefix.string() + "i");
    const std::vector<TrieSegmentEntry> segments = read_binary_vector<TrieSegmentEntry>(prefix.string() + "s");
    if (ind.size() < 2 || segments.size() < 2) {
        return std::nullopt;
    }

    const auto [low, high] = search_trie_range(ind, board);
    if (low == 0 && high == 0) {
        return std::nullopt;
    }

    std::ifstream ii_file(prefix.string() + "ii", std::ios::binary);
    if (!ii_file) {
        return std::nullopt;
    }
    const size_t count = high - low + 2ULL;
    std::vector<TrieNode16> ind3_seg(count);
    ii_file.seekg(static_cast<std::streamoff>(low * sizeof(TrieNode16) - sizeof(TrieNode16)), std::ios::beg);
    ii_file.read(reinterpret_cast<char *>(ind3_seg.data()), static_cast<std::streamsize>(count * sizeof(TrieNode16)));

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
        return std::nullopt;
    }

    const auto [start, end] = get_segment_position(segments, static_cast<uint32_t>(last_pos + low));
    std::ifstream z_file(prefix.string() + "z", std::ios::binary | std::ios::ate);
    if (!z_file || end < start) {
        return std::nullopt;
    }
    z_file.seekg(static_cast<std::streamoff>(start), std::ios::beg);
    std::vector<uint8_t> compressed(static_cast<size_t>(end - start));
    if (!compressed.empty()) {
        z_file.read(reinterpret_cast<char *>(compressed.data()), static_cast<std::streamsize>(compressed.size()));
    }
    std::vector<uint8_t> decompressed = decompress_xz_block_native(compressed.data(), compressed.size());
    using BlockEntry = CompactBookEntry<T>;
    const BlockEntry *block = reinterpret_cast<const BlockEntry *>(decompressed.data());
    const size_t block_size = decompressed.size() / sizeof(BlockEntry);
    const uint32_t target = static_cast<uint32_t>(board & 0xFFFFFFFFULL);

    size_t sub_low = static_cast<size_t>(ind3_seg[last_pos].next);
    size_t sub_high = static_cast<size_t>(ind3_seg[last_pos + 1].next) + 1ULL;
    if (sub_high == 1ULL) {
        if (block_size != 0 && target == block[0].lower32) {
            return normalize_value(block[0].success);
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
        return normalize_value(block[sub_low].success);
    }
    return normalize_value(ValueTraits<T>::zero_value);
}

template <typename T>
bool dispatch_trie_compress(const std::string &book_path) {
    return trie_compress_typed<T>(book_path);
}

template <typename T>
std::optional<double> dispatch_trie_search(const std::string &path_prefix, uint64_t board) {
    return trie_decompress_search_typed<T>(path_prefix, board);
}

} // namespace

bool trie_compress_progress_native(const std::string &book_path, const std::string &success_rate_dtype) {
    if (success_rate_dtype == "uint64") {
        return dispatch_trie_compress<uint64_t>(book_path);
    }
    if (success_rate_dtype == "float32") {
        return dispatch_trie_compress<float>(book_path);
    }
    if (success_rate_dtype == "float64") {
        return dispatch_trie_compress<double>(book_path);
    }
    return dispatch_trie_compress<uint32_t>(book_path);
}

std::optional<double> trie_decompress_search_native(
    const std::string &path_prefix,
    uint64_t board,
    const std::string &success_rate_dtype
) {
    if (success_rate_dtype == "uint64") {
        return dispatch_trie_search<uint64_t>(path_prefix, board);
    }
    if (success_rate_dtype == "float32") {
        return dispatch_trie_search<float>(path_prefix, board);
    }
    if (success_rate_dtype == "float64") {
        return dispatch_trie_search<double>(path_prefix, board);
    }
    return dispatch_trie_search<uint32_t>(path_prefix, board);
}
