#pragma once

#include <cstdint>
#include <optional>
#include <string>

bool trie_compress_progress_native(const std::string &book_path, const std::string &success_rate_dtype);
std::optional<double> trie_decompress_search_native(
    const std::string &path_prefix,
    uint64_t board,
    const std::string &success_rate_dtype
);
