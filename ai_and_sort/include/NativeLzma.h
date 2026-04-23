#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

bool compress_with_7z_or_xz(const std::string &input_path, int lvl = 1);
bool decompress_with_7z_or_xz(const std::string &archive_path);

std::vector<uint8_t> compress_xz_block_native(const uint8_t *data, size_t size, int lvl = 1);
std::vector<uint8_t> decompress_xz_block_native(const uint8_t *data, size_t size);

bool compress_uint64_array_native(const std::vector<uint64_t> &data, const std::string &output_base, int lvl = 1);
std::vector<uint64_t> decompress_uint64_array_native(const std::string &compressed_path);
std::optional<size_t> find_value_uint64_compressed_native(const std::string &compressed_path, uint64_t value);
