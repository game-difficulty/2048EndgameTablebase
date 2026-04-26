#include "CompressionBridge.h"

#include "FileIOUtils.h"
#include "NativeLzma.h"
#include "TrieCompression.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool has_suffix(const std::string &value, const std::string &suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

} // namespace

void maybe_compress_with_7z(const std::string &path) {
    if (!fs::exists(path)) {
        return;
    }
    compress_with_7z_or_xz(path, 1);
}

void maybe_decompress_with_7z(const std::string &archive_path) {
    if (!fs::exists(archive_path)) {
        return;
    }
    decompress_with_7z_or_xz(archive_path);
}

void maybe_do_compress_classic(const std::string &book_path, const std::string &success_rate_dtype) {
    if (!fs::exists(book_path)) {
        return;
    }
    if (!has_suffix(book_path, "book")) {
        return;
    }
    if (has_suffix(book_path, "_0.book") || has_suffix(book_path, "_1.book") || has_suffix(book_path, "_2.book")) {
        return;
    }
    if (fs::file_size(book_path) <= 2097152ULL) {
        return;
    }
    if (!trie_compress_progress_native(book_path, success_rate_dtype)) {
        return;
    }
    if (fs::exists(book_path)) {
        fs::remove(book_path);
    }
}

void maybe_do_compress_ad(const std::string &folder_path) {
    if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
        return;
    }
    for (const auto &entry : fs::directory_iterator(folder_path)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".i") {
            continue;
        }
        const fs::path index_path = entry.path();
        const fs::path base_path = index_path.parent_path() / index_path.stem();
        const fs::path zi_path = base_path;
        if (!fs::exists(zi_path.string() + ".zi")) {
            std::vector<uint64_t> data = FileIOUtils::read_binary_vector<uint64_t>(index_path.string());
            compress_uint64_array_native(data, base_path.string(), 1);
        }
        if (fs::exists(index_path) && fs::exists(zi_path.string() + ".zi")) {
            fs::remove(index_path);
        }
    }
}

std::vector<uint64_t> maybe_decompress_uint64_array(const std::string &compressed_path) {
    const fs::path zi_path(compressed_path);
    const fs::path segments_path = zi_path.parent_path() / (zi_path.stem().string() + ".s");
    if (!fs::exists(zi_path) || !fs::exists(segments_path)) {
        return {};
    }
    return decompress_uint64_array_native(compressed_path);
}
