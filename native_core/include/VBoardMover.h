#pragma once
#include <cstdint>
#include <algorithm>
#include "CommonMover.h"

struct WallMergePolicy {
    static inline std::pair<uint16_t, uint16_t> _internal_merge(uint16_t row, bool reverse_line) {
        // 解码行数据到数组
        uint32_t line[4];
        for (int i = 0; i < 4; ++i) {
            uint32_t v = (row >> (4 * (3 - i))) & 0xF;
            line[i] = (v == 0) ? 0 : (1u << v);
        }

        if (reverse_line) std::reverse(line, line + 4);

        uint32_t merged[4] = {0, 0, 0, 0};
        uint32_t score = 0;
        int m_ptr = 0;

        // 分段合并逻辑 (以 32768 为界)
        for (int i = 0; i < 4; ) {
            if (line[i] == 32768) {
                merged[i] = 32768;
                m_ptr = ++i;
                continue;
            }

            // 寻找当前段的终点（直到遇到 32768 或行尾）
            int next_wall = i;
            while (next_wall < 4 && line[next_wall] != 32768) next_wall++;

            // 对 [i, next_wall) 区间进行合并
            int write_idx = i;
            for (int read_idx = i; read_idx < next_wall; ++read_idx) {
                if (line[read_idx] == 0) continue;
                
                if (write_idx > i && merged[write_idx - 1] == line[read_idx] && merged[write_idx - 1] != 32768) {
                    merged[write_idx - 1] *= 2;
                    score += merged[write_idx - 1];
                    // 合并后的位置标记为已处理，写指针不移动（因为是原地修改前一个）
                } else {
                    merged[write_idx++] = line[read_idx];
                }
            }
            i = next_wall;
        }

        if (reverse_line) std::reverse(merged, merged + 4);

        // 重新编码为 uint16_t
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
using VBoardMover = MoverEngine<WallMergePolicy>;

// 实例化模板
template struct MoverEngine<WallMergePolicy>;
