#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "NativeLzma.h"

void maybe_compress_with_7z(const std::string &path);
void maybe_decompress_with_7z(const std::string &archive_path);
void maybe_do_compress_classic(const std::string &book_path, const std::string &success_rate_dtype);
void maybe_do_compress_ad(const std::string &folder_path);
std::vector<uint64_t> maybe_decompress_uint64_array(const std::string &compressed_path);
bool write_temp_uint64_archive(const std::string &archive_path, const std::vector<uint64_t> &data, int lvl = 1);
std::vector<uint64_t> read_temp_uint64_archive(const std::string &archive_path);
bool write_temp_byte_archive(const std::string &archive_path, const std::vector<uint8_t> &data, int lvl = 1);
bool write_temp_byte_spans_archive(
    const std::string &archive_path,
    const std::vector<ArchiveByteSpan> &spans,
    int lvl = 1
);
std::vector<uint8_t> read_temp_byte_archive(const std::string &archive_path);
bool is_readable_temp_archive(const std::string &archive_path);
