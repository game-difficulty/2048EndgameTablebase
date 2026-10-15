#pragma once
#include <cstdint>
#include <vector>
#include "FormationRuntime.h"


// --- 定式配置结构体 ---

// --- 2. 通用校验函数 ---

inline bool is_pattern(uint64_t board, const std::vector<uint64_t>& masks) {
    if (masks.empty()) return true;
    for (uint64_t m : masks) {
        if ((board & m) == m) return true;
    }
    return false;
}



inline bool is_success_by_shifts(uint64_t board, int target, const std::vector<uint8_t> &success_shifts) {
    if (success_shifts.empty()) {
        return true;
    }
    uint64_t target_value = static_cast<uint64_t>(target);
    for (uint8_t shift : success_shifts) {
        if (((board >> shift) & 0xFULL) == target_value) {
            return true;
        }
    }
    return false;
}
