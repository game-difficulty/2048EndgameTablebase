#pragma once
#include <cstdint>
#include <string>

inline bool sum_goal_success(uint64_t board, int target) {
    if (target < 4) return false;
    uint32_t sum = 0;
    for (int shift = 0; shift < 64; shift += 4) {
        const unsigned tile = (board >> shift) & 15U;
        if (tile) sum += 1U << tile;
    }
    return sum % 16384U >= static_cast<uint32_t>(target - 2);
}

inline int sum_goal_from_name(const std::string &name) {
    const auto begin = name.rfind("_sum-");
    if (begin == std::string::npos) return 0;
    const auto text = name.substr(begin + 5);
    unsigned value = 0;
    if (text.empty()) return 0;
    for (char c : text) {
        if (c < '0' || c > '9') return 0;
        value = value * 10 + (c - '0');
        if (value >= 16384) return 0;
    }
    return value >= 4 && value % 2 == 0 ? static_cast<int>(value) : 0;
}
