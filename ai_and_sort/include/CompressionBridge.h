#pragma once

#include <cstdint>
#include <string>
#include <vector>

void maybe_compress_with_7z(const std::string &path);
void maybe_decompress_with_7z(const std::string &archive_path);
void maybe_do_compress_classic(const std::string &book_path, const std::string &success_rate_dtype);
void maybe_do_compress_ad(const std::string &folder_path);
std::vector<uint64_t> maybe_decompress_uint64_array(const std::string &compressed_path);
