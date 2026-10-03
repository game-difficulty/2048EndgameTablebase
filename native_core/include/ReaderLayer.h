#pragma once
#include <charconv>
#include <cstdint>
#include <optional>
#include <string>

// Logical filenames preserve the old layer-zero sum; internal build steps do not.
inline int reader_free_layer_offset(const std::string &pattern) {
    if (pattern.compare(0, 4, "free") != 0) return 0;
    const size_t end = pattern.find('_', 4);
    const std::string number = pattern.substr(4, end == std::string::npos ? end : end - 4);
    int n = 0;
    const auto parsed = std::from_chars(number.data(), number.data() + number.size(), n);
    return parsed.ec == std::errc{} && parsed.ptr == number.data() + number.size() && n >= 10 && n <= 16 ? 1 - n : 0;
}

inline std::optional<int64_t> parse_reader_layer_number(const std::string &text) {
    int64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) return std::nullopt;
    return value;
}
