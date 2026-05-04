#include "CompressionBridge.h"

#include "FileIOUtils.h"
#include "NativeLzma.h"
#include "TrieCompression.h"

#include <filesystem>
#include <fstream>
#include <cstring>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool has_suffix(const std::string &value, const std::string &suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string temp_archive_entry_name(const std::string &archive_path) {
    std::string name = fs::path(archive_path).stem().string();
    if (name.empty()) {
        name = "data";
    }
    return name + ".bin";
}

bool is_xz_stream(const std::vector<uint8_t> &bytes) {
    static constexpr uint8_t kMagic[] = {0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00};
    return bytes.size() >= sizeof(kMagic) &&
           std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) == 0;
}

FileIOUtils::DirectIoConfig compression_direct_io_config() {
    return FileIOUtils::normalize_direct_io_config(FileIOUtils::DirectIoConfig{true, 16U, 8U});
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
            std::vector<uint64_t> data =
                FileIOUtils::read_binary_vector_direct<uint64_t>(index_path.string(), compression_direct_io_config());
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

bool write_temp_uint64_archive(const std::string &archive_path, const std::vector<uint64_t> &data, int lvl) {
    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(data.data());
    const size_t byte_size = data.size() * sizeof(uint64_t);
    return compress_bytes_to_7z_archive_streaming(
        bytes,
        byte_size,
        archive_path,
        temp_archive_entry_name(archive_path),
        lvl
    );
}

std::vector<uint64_t> read_temp_uint64_archive(const std::string &archive_path) {
    if (!fs::exists(archive_path)) {
        return {};
    }

    std::vector<uint8_t> header = FileIOUtils::read_binary_bytes_range(archive_path, 0, 6);
    if (is_xz_stream(header)) {
        std::vector<uint8_t> archive_bytes = FileIOUtils::read_binary_bytes(archive_path);
        std::vector<uint8_t> decompressed = decompress_xz_block_native(archive_bytes.data(), archive_bytes.size());
        if (decompressed.empty() && !archive_bytes.empty()) {
            return {};
        }
        if ((decompressed.size() % sizeof(uint64_t)) != 0) {
            return {};
        }
        std::vector<uint64_t> result(decompressed.size() / sizeof(uint64_t));
        if (!result.empty()) {
            std::memcpy(result.data(), decompressed.data(), decompressed.size());
        }
        return result;
    }

    std::vector<uint8_t> decompressed;
    if (!decompress_7z_archive_to_bytes_streaming(archive_path, decompressed)) {
        return {};
    }
    if ((decompressed.size() % sizeof(uint64_t)) != 0) {
        return {};
    }
    std::vector<uint64_t> result(decompressed.size() / sizeof(uint64_t));
    if (!result.empty()) {
        std::memcpy(result.data(), decompressed.data(), decompressed.size());
    }
    return result;
}
