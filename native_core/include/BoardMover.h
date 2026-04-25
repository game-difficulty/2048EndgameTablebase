#pragma once
#include <cstdint>
#include <algorithm>
#include "CommonMover.h"

struct StandardMergePolicy {
    static inline std::pair<uint16_t, uint16_t> _internal_merge(uint16_t row, bool reverse_line) {
        // 解码行数据到数组 (log2 转换为真值)
        uint32_t line[4];
        for (int i = 0; i < 4; ++i) {
            uint32_t v = (row >> (4 * (3 - i))) & 0xF;
            line[i] = (v == 0) ? 0 : (1u << v);
        }

        // 如果是向右移动，先反转数组统一按向左处理
        if (reverse_line) std::reverse(line, line + 4);

        // 第一步：提取所有非零方块 (滑动阶段)
        uint32_t non_zero[4] = {0, 0, 0, 0};
        int nz_count = 0;
        for (int i = 0; i < 4; ++i) {
            if (line[i] != 0) non_zero[nz_count++] = line[i];
        }

        // 第二步：合并相邻相同的方块 (合并阶段)
        uint32_t merged[4] = {0, 0, 0, 0};
        uint32_t score = 0;
        int m_count = 0;
        for (int i = 0; i < nz_count; ++i) {
            // 标准逻辑：相邻相等且不是 32768 则合并
            if (i + 1 < nz_count && non_zero[i] == non_zero[i + 1] && non_zero[i] != 32768) {
                uint32_t combined = non_zero[i] * 2;
                merged[m_count++] = combined;
                score += combined;
                ++i; // 跳过下一个已合并的方块
            } else {
                merged[m_count++] = non_zero[i];
            }
        }

        // 如果之前反转过，现在反转回来
        if (reverse_line) std::reverse(merged, merged + 4);

        // 第三步：重新编码为 uint16_t (真值还原回 log2)
        uint16_t res = 0;
        for (int i = 0; i < 4; ++i) {
            if (merged[i] > 0) {
                uint32_t log_v = 0;
                uint32_t tmp = merged[i];
                while (tmp >>= 1) log_v++;
                res |= (log_v & 0xF) << (4 * (3 - i));
            }
        }
        return {res, (uint16_t)std::min(score, 65535u)};
    }
};

// 使用别名代替 namespace
using BoardMover = MoverEngine<StandardMergePolicy>;

// 实例化模板
template struct MoverEngine<StandardMergePolicy>;
