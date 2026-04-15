#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>

enum class SymmMode { Identity, Full, Diagonal, Horizontal, Min33, Min24, Min34 };

// --- 定式配置结构体 ---
struct PatternConfig {
    std::string name;
    std::vector<uint64_t> masks;
    uint64_t success_mask;
    int symm_mode;
};

inline std::unordered_map<std::string, PatternConfig> pattern_configs;

// --- 2. 通用校验函数 ---

inline bool is_pattern(uint64_t board, const std::vector<uint64_t>& masks) {
    if (masks.empty()) return true;
    for (uint64_t m : masks) {
        if ((board & m) == m) return true;
    }
    return false;
}


inline bool is_success(uint64_t board, uint64_t success_mask, uint64_t target_stacked) {
    uint64_t diff = (board & success_mask) ^ target_stacked;
    uint64_t mask_7 = 0x7777777777777777ULL;
    return ~(((diff & mask_7) + mask_7) | diff | mask_7) != 0;
}


inline bool is_success_general(uint64_t board, uint64_t success_mask, int target) {
    uint64_t target_stacked = (uint64_t)target * 0x1111111111111111ULL;
    return is_success(board, success_mask, target_stacked);
}