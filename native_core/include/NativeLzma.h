#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct ArchiveByteSpan {
    const uint8_t *data = nullptr;
    size_t size = 0;
};

bool compress_with_7z_or_xz(const std::string &input_path, int lvl = 1);
bool decompress_with_7z_or_xz(const std::string &archive_path);
bool compress_bytes_to_7z_archive_streaming(
    const uint8_t *data,
    size_t size,
    const std::string &archive_path,
    const std::string &entry_name,
    int lvl = 1
);
bool compress_spans_to_7z_archive_streaming(
    const std::vector<ArchiveByteSpan> &spans,
    const std::string &archive_path,
    const std::string &entry_name,
    int lvl = 1
);
bool decompress_7z_archive_to_bytes_streaming(const std::string &archive_path, std::vector<uint8_t> &output);

class SevenZipArchiveWriter {
public:
    SevenZipArchiveWriter();
    SevenZipArchiveWriter(const std::string &archive_path, const std::string &entry_name, int lvl = 1);
    ~SevenZipArchiveWriter();

    SevenZipArchiveWriter(const SevenZipArchiveWriter &) = delete;
    SevenZipArchiveWriter &operator=(const SevenZipArchiveWriter &) = delete;
    SevenZipArchiveWriter(SevenZipArchiveWriter &&) noexcept;
    SevenZipArchiveWriter &operator=(SevenZipArchiveWriter &&) noexcept;

    void open(const std::string &archive_path, const std::string &entry_name, int lvl = 1);
    void append(const void *data, size_t size);
    void close();
    [[nodiscard]] bool is_open() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class SevenZipSequentialReader {
public:
    SevenZipSequentialReader();
    explicit SevenZipSequentialReader(const std::string &archive_path);
    ~SevenZipSequentialReader();

    SevenZipSequentialReader(const SevenZipSequentialReader &) = delete;
    SevenZipSequentialReader &operator=(const SevenZipSequentialReader &) = delete;
    SevenZipSequentialReader(SevenZipSequentialReader &&) noexcept;
    SevenZipSequentialReader &operator=(SevenZipSequentialReader &&) noexcept;

    void open(const std::string &archive_path);
    void read(void *dst, size_t bytes);
    void close();
    [[nodiscard]] bool is_open() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::vector<uint8_t> compress_xz_block_native(const uint8_t *data, size_t size, int lvl = 1);
std::vector<uint8_t> decompress_xz_block_native(const uint8_t *data, size_t size);

bool compress_uint64_array_native(const std::vector<uint64_t> &data, const std::string &output_base, int lvl = 1);
std::vector<uint64_t> decompress_uint64_array_native(const std::string &compressed_path);
std::optional<size_t> find_value_uint64_compressed_native(const std::string &compressed_path, uint64_t value);
