#include "CompressionBridge.h"

#include "FileIOUtils.h"
#include "NativeLzma.h"
#include "PathUtils.h"
#include "TrieCompression.h"

#include <filesystem>
#include <fstream>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool has_suffix(const std::string &value, const std::string &suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string temp_archive_entry_name(const std::string &archive_path) {
    std::string name = NativePath::to_utf8_string(NativePath::from_utf8(archive_path).stem());
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

bool write_temp_byte_payload_archive(
    const std::string &archive_path,
    const uint8_t *data,
    size_t size,
    int lvl
) {
    static constexpr uint8_t kEmptyPayloadByte = 0U;
    if (size != 0U && data == nullptr) {
        return false;
    }
    const uint8_t *payload = size == 0U ? &kEmptyPayloadByte : data;
    if (compress_bytes_to_7z_archive_streaming(
            payload,
            size,
            archive_path,
            temp_archive_entry_name(archive_path),
            lvl)) {
        return true;
    }

    std::vector<uint8_t> compressed = compress_xz_block_native(payload, size, lvl);
    if (compressed.empty() && size != 0U) {
        return false;
    }
    const std::string temp_path = FileIOUtils::temp_write_path(archive_path);
    FileIOUtils::write_binary_bytes(temp_path, compressed);
    try {
        FileIOUtils::finalize_temporary_file(temp_path, archive_path);
        return true;
    } catch (...) {
        std::error_code ec;
        NativePath::remove(temp_path, ec);
        return false;
    }
}

} // namespace

void maybe_do_compress_classic(
    const std::string &book_path,
    const std::string &success_rate_dtype,
    const std::string &output_book_path
) {
    if (!NativePath::exists(book_path)) {
        return;
    }
    if (!has_suffix(book_path, "book")) {
        return;
    }
    if (has_suffix(book_path, "_0.book") || has_suffix(book_path, "_1.book") || has_suffix(book_path, "_2.book")) {
        return;
    }
    if (NativePath::file_size(book_path) <= 2097152ULL) {
        return;
    }
    if (!trie_compress_progress_native(book_path, success_rate_dtype, output_book_path)) {
        return;
    }
    if (NativePath::exists(book_path)) {
        NativePath::remove(book_path);
    }
}

void maybe_do_compress_ad(const std::string &folder_path) {
    std::error_code folder_ec;
    if (!NativePath::exists(folder_path) || !NativePath::is_directory(folder_path, folder_ec)) {
        return;
    }
    for (const auto &entry : fs::directory_iterator(NativePath::from_utf8(folder_path))) {
        if (!entry.is_regular_file() || entry.path().extension() != ".i") {
            continue;
        }
        const fs::path index_path = entry.path();
        const fs::path base_path = index_path.parent_path() / index_path.stem();
        const fs::path zi_path = base_path;
        const std::string zi_path_utf8 = NativePath::to_utf8_string(zi_path);
        if (!NativePath::exists(zi_path_utf8 + ".zi")) {
            std::vector<uint64_t> data =
                FileIOUtils::read_binary_vector_direct<uint64_t>(
                    NativePath::to_utf8_string(index_path),
                    compression_direct_io_config());
            compress_uint64_array_native(data, NativePath::to_utf8_string(base_path), 1);
        }
        if (fs::exists(index_path) && NativePath::exists(zi_path_utf8 + ".zi")) {
            fs::remove(index_path);
        }
    }
}

std::vector<uint64_t> maybe_decompress_uint64_array(const std::string &compressed_path) {
    const fs::path zi_path = NativePath::from_utf8(compressed_path);
    fs::path segments_path = zi_path;
    segments_path.replace_extension(".s");
    if (!fs::exists(zi_path) || !fs::exists(segments_path)) {
        return {};
    }
    return decompress_uint64_array_native(compressed_path);
}

bool write_temp_uint64_archive(const std::string &archive_path, const std::vector<uint64_t> &data, int lvl) {
    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(data.data());
    const size_t byte_size = data.size() * sizeof(uint64_t);
    return write_temp_byte_payload_archive(archive_path, bytes, byte_size, lvl);
}

bool write_temp_byte_archive(const std::string &archive_path, const std::vector<uint8_t> &data, int lvl) {
    return write_temp_byte_payload_archive(archive_path, data.data(), data.size(), lvl);
}

bool write_temp_byte_spans_archive(
    const std::string &archive_path,
    const std::vector<ArchiveByteSpan> &spans,
    int lvl
) {
    if (compress_spans_to_7z_archive_streaming(
            spans,
            archive_path,
            temp_archive_entry_name(archive_path),
            lvl)) {
        return true;
    }

    size_t total_size = 0U;
    for (const ArchiveByteSpan &span : spans) {
        if (span.size > std::numeric_limits<size_t>::max() - total_size) {
            return false;
        }
        total_size += span.size;
    }
    std::vector<uint8_t> bytes;
    bytes.reserve(total_size);
    for (const ArchiveByteSpan &span : spans) {
        if (span.size == 0U) {
            continue;
        }
        if (span.data == nullptr) {
            return false;
        }
        bytes.insert(bytes.end(), span.data, span.data + span.size);
    }
    return write_temp_byte_payload_archive(archive_path, bytes.data(), bytes.size(), lvl);
}

std::vector<uint8_t> read_temp_byte_archive(const std::string &archive_path) {
    if (!NativePath::exists(archive_path)) {
        return {};
    }

    std::vector<uint8_t> header = FileIOUtils::read_binary_bytes_range(archive_path, 0, 6);
    if (is_xz_stream(header)) {
        std::vector<uint8_t> archive_bytes = FileIOUtils::read_binary_bytes(archive_path);
        std::vector<uint8_t> decompressed = decompress_xz_block_native(archive_bytes.data(), archive_bytes.size());
        if (decompressed.empty() && !archive_bytes.empty()) {
            return {};
        }
        return decompressed;
    }

    std::vector<uint8_t> decompressed;
    if (!decompress_7z_archive_to_bytes_streaming(archive_path, decompressed)) {
        return {};
    }
    return decompressed;
}

std::vector<uint64_t> read_temp_uint64_archive(const std::string &archive_path) {
    std::vector<uint8_t> decompressed = read_temp_byte_archive(archive_path);
    if (decompressed.empty()) {
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

bool is_readable_temp_archive(const std::string &archive_path) {
    return is_readable_7z_or_xz_archive(archive_path);
}
