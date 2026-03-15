#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <optional>
#include <tuple>
#include <omp.h>
#include <atomic>
#include <array>
#include <memory>
#include <cstdlib>
#include <cstring>

// #include <iostream>
// #include <chrono>
// #include <thread>

#include "BoardMover.h"


namespace nb = nanobind;

// 将盘面上 >= threshold 的块全部替换为 0xF
inline uint64_t mask(uint64_t board, int threshold) {
    uint64_t evens = board & 0x0F0F0F0F0F0F0F0FULL;
    uint64_t odds  = (board >> 4) & 0x0F0F0F0F0F0F0F0FULL;
    uint64_t addend = (0x80 - threshold) * 0x0101010101010101ULL;
    uint64_t e_cmp = (evens + addend) & 0x8080808080808080ULL;
    uint64_t o_cmp = (odds + addend)  & 0x8080808080808080ULL;
    uint64_t e_mask = (e_cmp >> 3) - (e_cmp >> 7);
    uint64_t o_mask = (o_cmp >> 3) - (o_cmp >> 7);
    return board | e_mask | (o_mask << 4);
}

inline uint64_t extract_0xf_nibbles(uint64_t v) {
    uint64_t temp = v & (v >> 1) & (v >> 2) & (v >> 3);
    return (temp & 0x1111111111111111ULL) * 0xF;
}

// ------------------------------------------------------------------
// 评估函数与预计算表
// ------------------------------------------------------------------
struct alignas(4) ScorePair {
    int16_t d1;
    int16_t d2;
};

// 全局对齐声明，256KB 大小，Cache Line 对齐
alignas(64) ScorePair _diffs_merged[65536];

const int32_t TILE_WEIGHT_MAP[16] = {
    0, 2, 4, 8, 16, 32, 64, 128,
    248, 388, 488, 518, 519, 519, 519, 520
};

int32_t diffs_evaluation_func(const int32_t* line_masked) {
    // dpdf 计算
    int32_t score_dpdf = line_masked[0];
    for (int x = 0; x < 3; ++x) {  
        if (line_masked[x] < line_masked[x + 1]) {
            if (line_masked[x] > 400) {
                score_dpdf += (line_masked[x] << 1) + (line_masked[x + 1] - line_masked[x]) * x;
            } else if (line_masked[x] > 300 && x == 1 && line_masked[0] > line_masked[1]) {
                score_dpdf += (line_masked[x] << 1);
            } else {
                score_dpdf -= (line_masked[x + 1] - line_masked[x]) << 3;
                score_dpdf -= line_masked[x + 1] * 3;
                if (x < 2 && line_masked[x + 2] < line_masked[x + 1] && line_masked[x + 1] > 30) {
                    score_dpdf -= std::max(80, line_masked[x + 1]);
                }
            }
        } else if (x < 2) {  // 正序是越来越小
            score_dpdf += line_masked[x + 1] + line_masked[x];
        } else {
            score_dpdf += (int32_t)((line_masked[x + 1] + line_masked[x]) * 0.5);
        }
    }
    if (line_masked[0] > 400 && line_masked[1] > 300 && line_masked[2] > 200 && line_masked[2] > line_masked[3] && line_masked[3] < 300) {
        score_dpdf += line_masked[3] >> 2;
    }

    // t 计算
    int32_t score_t;
    if (std::min(line_masked[0], line_masked[3]) < 32) {
        score_t = -16384;
    } else if ((line_masked[0] < line_masked[1] && line_masked[0] < 400) || 
               (line_masked[3] < line_masked[2] && line_masked[3] < 400)) {
        score_t = -(std::max(line_masked[1], line_masked[2]) * 10);
    } else {
        score_t = (int32_t)((line_masked[0] * 1.8 + line_masked[3] * 1.8) + 
                  std::max(line_masked[1], line_masked[2]) * 1.5 + std::min(200, std::min(line_masked[1], line_masked[2])) * 2.5);
        if (std::min(line_masked[1], line_masked[2]) < 8) {
            score_t -= 80;
        }
    }

    int zero_count = 0;
    for (int k = 0; k < 4; ++k) if (line_masked[k] == 0) zero_count++;
    
    int32_t sum_123 = line_masked[1] + line_masked[2] + line_masked[3];
    int32_t penalty = 0;
    if (line_masked[0] > 100 && ((zero_count > 1 && sum_123 < 32) || sum_123 < 12)) {
        penalty = 4;
    }

    return std::max(score_dpdf, score_t) / 4 - penalty;
}

void init_evaluate_tables() {
    for (int i = 0; i < 65536; ++i) {
        int32_t line_masked[4];
        int32_t line_masked_rev[4];

        for (int j = 0; j < 4; ++j) {
            int tile_exponent = (i >> (j * 4)) & 0xF;
            int32_t weight = TILE_WEIGHT_MAP[tile_exponent];
            line_masked[j] = weight;
            line_masked_rev[3 - j] = weight;
        }

        _diffs_merged[i].d1 = static_cast<int16_t>(diffs_evaluation_func(line_masked));
        _diffs_merged[i].d2 = static_cast<int16_t>(diffs_evaluation_func(line_masked_rev));
    }
}

struct EvalInitializer {
    EvalInitializer() { init_evaluate_tables(); }
};
static EvalInitializer _eval_init;

int32_t large_tile_count(uint8_t start, uint8_t counts[]) {
    int sum = 0;
    int i = start;
    while (i < 16 && counts[i] != 0) i++;

    for (i++; i < 16; i++) {
        if (counts[i] == 2 && i < 15) return 0; // 发现重复，直接返回0
        sum += counts[i];
    }
    return sum;
}

uint64_t get_23_mask(uint64_t board) {
    static const uint64_t masks[8] = {
        0xff00fff0ULL,0xfff00ff00000000ULL,0xf00ff00ffULL,0xff00ff00f0000000ULL,0xfff0ff0000000000ULL,0xff00ff000f0000ULL,0xf000ff00ff00ULL,0xff0fffULL
    };

    if ((board & masks[0]) == masks[0]) return masks[0];
    if ((board & masks[1]) == masks[1]) return masks[1];
    if ((board & masks[2]) == masks[2]) return masks[2];
    if ((board & masks[3]) == masks[3]) return masks[3];
    if ((board & masks[4]) == masks[4]) return masks[4];
    if ((board & masks[5]) == masks[5]) return masks[5];
    if ((board & masks[6]) == masks[6]) return masks[6];
    if ((board & masks[7]) == masks[7]) return masks[7];

    return 0;
}

uint64_t get_22_mask(uint64_t board) {
    // 定义四个角落的 2x2 掩码
    static const uint64_t masks[4] = {
        0xff00ff0000000000ULL, // 左上角
        0x00ff00ff00000000ULL, // 右上角
        0x00000000ff00ff00ULL, // 左下角
        0x0000000000ff00ffULL  // 右下角
    };

    if ((board & masks[0]) == masks[0]) return masks[0];
    if ((board & masks[1]) == masks[1]) return masks[1];
    if ((board & masks[2]) == masks[2]) return masks[2];
    if ((board & masks[3]) == masks[3]) return masks[3];

    return 0;
}

uint64_t get_21_mask(uint64_t board) {
    // 定义四个角落的 L 掩码
    static const uint64_t masks[4] = {
        0xff00f00000000000ULL, // 左上角
        0x00ff000f00000000ULL, // 右上角
        0x00000000f000ff00ULL, // 左下角
        0x00000000000f00ffULL  // 右下角
    };

    if ((board & masks[0]) == masks[0]) return masks[0];
    if ((board & masks[1]) == masks[1]) return masks[1];
    if ((board & masks[2]) == masks[2]) return masks[2];
    if ((board & masks[3]) == masks[3]) return masks[3];

    return 0;
}

uint64_t get_20_mask(uint64_t board) {
    static const uint64_t masks[8] = {
    0xff00ULL,0xff000000000000ULL,0xf000fULL,0xf000f00000000000ULL,0xff00000000000000ULL,0xf000f00000000ULL,0xf000f000ULL,0xffULL
    };

    if ((board & masks[0]) == masks[0]) return masks[0];
    if ((board & masks[1]) == masks[1]) return masks[1];
    if ((board & masks[2]) == masks[2]) return masks[2];
    if ((board & masks[3]) == masks[3]) return masks[3];
    if ((board & masks[4]) == masks[4]) return masks[4];
    if ((board & masks[5]) == masks[5]) return masks[5];
    if ((board & masks[6]) == masks[6]) return masks[6];
    if ((board & masks[7]) == masks[7]) return masks[7];

    return 0;
}

void update_specific_scores(uint64_t board, bool flag1, bool flag2, bool flag3, bool flag4) {
    // 1. 判断 2x2 角落位运算判断
    uint64_t corner_mask = get_22_mask(board);
    bool apply_bonus = (corner_mask > 0);

    // 定义结构
    struct Target { uint16_t index; uint16_t index2; int16_t bonus; };
    struct DynamicTarget { uint16_t index; uint16_t index2; int16_t bonus_f1; int16_t bonus_f2; };

    // --- 使用 static 缓存基础评分 ---
    static int32_t base_scores_cache[3];    // 存储第一组的基础分
    static int32_t dynamic_scores_cache[5]; // 存储第二组的基础分
    static bool initialized = false;

    // 数据定义设为 static const，避免每次调用都在栈上重建数组
    static const Target base_targets[] = {
        {0x78ff, 0xff87, 150}, {0x7fff, 0xfff7, 300},
        {0x8fff, 0xfff8, 320}, 
    };

    static const DynamicTarget dynamic_targets[] = {
        {0x2fff, 0xfff2, 4, -4}, {0x3fff, 0xfff3, 12, -10},
        {0x4fff, 0xfff4, 18, -16}, {0x5fff, 0xfff5, 24, -20},
        {0x6fff, 0xfff6, 25, -8}, 
    };

    // --- 第一次运行时初始化缓存 ---
    if (!initialized) {
        // 初始化第一组缓存
        for (int i = 0; i < 3; ++i) {
            int32_t line[4];
            for (int j = 0; j < 4; ++j) 
                line[j] = TILE_WEIGHT_MAP[(base_targets[i].index >> (j * 4)) & 0xF];
            base_scores_cache[i] = diffs_evaluation_func(line);
        }
        // 初始化第二组缓存
        for (int i = 0; i < 5; ++i) {
            int32_t line[4];
            for (int j = 0; j < 4; ++j) 
                line[j] = TILE_WEIGHT_MAP[(dynamic_targets[i].index >> (j * 4)) & 0xF];
            dynamic_scores_cache[i] = diffs_evaluation_func(line);
        }
        initialized = true;
    }

    // --- 运行时：仅进行加法和写回操作 ---
    
    // 处理第一组
    for (int i = 0; i < 3; ++i) {
        int32_t score = base_scores_cache[i];
        if (apply_bonus && flag3) score += base_targets[i].bonus;
        
        _diffs_merged[base_targets[i].index].d1  = (int16_t)score;
        _diffs_merged[base_targets[i].index2].d2 = (int16_t)score;
    }

    // 处理第二组
    for (int i = 0; i < 5; ++i) {
        int32_t score = dynamic_scores_cache[i];
        
        if (apply_bonus) {
            if (flag1) score += dynamic_targets[i].bonus_f1;
            else if (flag2) score += dynamic_targets[i].bonus_f2;
            else if (flag4) score += dynamic_targets[i].bonus_f1 >> 1;
        }
        
        _diffs_merged[dynamic_targets[i].index].d1  = (int16_t)score;
        _diffs_merged[dynamic_targets[i].index2].d2 = (int16_t)score;
    }
}

// ------------------------------------------------------------------
// Cache 类
// ------------------------------------------------------------------

class Cache {
private:
    std::atomic<bool> dummy_false{false};
    int32_t dummy_dead_score = 131072;
    
    // 次级哈希生成
    inline uint32_t get_signature(uint64_t board) const {
        return static_cast<uint32_t>(board + (board >> 32));
    }

    // 定义 64 字节对齐的 Bucket，完美契合 L1/L2/L3 Cache Line
    // 使用 memory_order_relaxed 的原子操作，实现极速无锁并发
    struct alignas(64) CacheBucket {
        std::atomic<uint64_t> entries[8];

        CacheBucket() {
            for (int i = 0; i < 8; ++i) {
                entries[i].store(0, std::memory_order_relaxed);
            }
        }
    };

    // Bucket 总数，原 length 为 2097151 (总计 16MB)
    // 现在我们将容量保持在 16MB 左右，转换为 Bucket 数量
    // 16MB / 64B = 262144 个桶 (这是 2 的幂次方，便于位运算)
    static constexpr uint64_t num_buckets = 262144;
    static constexpr uint64_t bucket_mask = num_buckets - 1;

    std::unique_ptr<CacheBucket[]> table;

public:
    std::atomic<bool>* abort_ptr = &dummy_false; 
    int32_t* dead_score_ptr = &dummy_dead_score;

    Cache() {
        // C++17 的对齐内存分配，确保整个大数组在内存中的物理地址 64 字节对齐
#ifdef _WIN32
        table.reset(new (std::align_val_t(64)) CacheBucket[num_buckets]);
#else
        void* ptr = std::aligned_alloc(64, num_buckets * sizeof(CacheBucket));
        table.reset(new (ptr) CacheBucket[num_buckets]);
#endif
    }

    void clear() {
        for (uint64_t i = 0; i < num_buckets; ++i) {
            std::memset(&table[i], 0, sizeof(CacheBucket));
        }
    }

    // 主哈希：只决定落在哪个 Bucket (返回 0 ~ bucket_mask)
    inline uint64_t hash(uint64_t board) const {
        uint64_t hashed = (((board ^ (board >> 27)) * 0x1A85EC53ULL) + board) >> 23;
        return hashed & bucket_mask;
    }

    inline bool lookup(uint64_t bucket_idx, uint64_t board, int32_t depth, int32_t& out_score) {
        uint32_t current_sig = get_signature(board);
        CacheBucket& bucket = table[bucket_idx];

        // 此时这 64 字节已经整体被吸入 L1 Cache
        // 这里的 8 次循环是在 L1 中完成的，耗时几乎为 0
        for (int i = 0; i < 8; ++i) {
            uint64_t entry = bucket.entries[i].load(std::memory_order_relaxed);
            if (entry == 0) continue;

            uint32_t cached_sig = static_cast<uint32_t>(entry >> 32);
            if (cached_sig == current_sig) {
                int32_t cached_depth = static_cast<int32_t>(entry & 0xFF);
                if (cached_depth >= depth) {
                    uint32_t unsigned_score = (entry >> 8) & 0xFFFFFF;
                    out_score = static_cast<int32_t>(unsigned_score) - 8388608;
                    return true;
                }
            }
        }
        return false;
    }

    inline void update(uint64_t bucket_idx, uint64_t board, int32_t depth, int32_t score) {
        uint32_t current_sig = get_signature(board);
        CacheBucket& bucket = table[bucket_idx];

        // 死局或接近死局的得分比较可靠，强制标记为极大深度
        if (score <= -(*dead_score_ptr) + 32) depth = 63; 
        
        uint8_t pack_depth = static_cast<uint8_t>(std::max(0, std::min(255, depth)));
        uint32_t pack_score = static_cast<uint32_t>(score + 8388608) & 0xFFFFFF;

        uint64_t new_entry = (static_cast<uint64_t>(current_sig) << 32) | 
                             (static_cast<uint64_t>(pack_score) << 8) | 
                             static_cast<uint64_t>(pack_depth);

        int min_depth_idx = 0;
        int32_t min_depth_val = 1000; // 初始设为极大值

        // --- 核心逻辑 ---
        for (int i = 0; i < 8; ++i) {
            uint64_t entry = bucket.entries[i].load(std::memory_order_relaxed);
            
            // 1. 发现空槽直接占用
            if (entry == 0) {
                bucket.entries[i].store(new_entry, std::memory_order_relaxed);
                return;
            }

            uint32_t cached_sig = static_cast<uint32_t>(entry >> 32);
            int32_t cached_depth = static_cast<int32_t>(entry & 0xFF);

            // 2. 签名命中：同一个局面，更新或保持，不能让它出现在两个槽位
            if (cached_sig == current_sig) {
                // 如果当前搜索深度更深，进行覆盖
                if (depth > cached_depth) {
                    bucket.entries[i].store(new_entry, std::memory_order_relaxed);
                }
                return;
            }

            // 3. 统计该桶内最弱（深度最小）的槽位
            if (cached_depth < min_depth_val) {
                min_depth_val = cached_depth;
                min_depth_idx = i;
            }
        }

        // 4. 强制替换：运行到这里说明没有空槽且没有签名命中
        // 始终写入最弱槽位以保证时间局部性
        bucket.entries[min_depth_idx].store(new_entry, std::memory_order_relaxed);
    }
};

// ------------------------------------------------------------------
// AIPlayer 类
// ------------------------------------------------------------------

class AIPlayer {
public:
    alignas(64) std::atomic<bool> stop_search{false};
    int32_t max_d;
    int32_t max_layer;
    uint8_t best_operation;
    uint64_t board; 
    uint32_t board_sum;
    Cache cache;
    std::atomic<uint64_t> node; 
    int32_t max_threads;
    float spawn_rate4;
    int32_t spawn_weight_4;
    int32_t spawn_weight_2;
    uint8_t do_check;
    uint8_t masked_count;
    uint8_t min_masked_tile;
    uint64_t fixed_mask;

    // 剪枝开关与初始评估分
    uint8_t prune;
    int32_t initial_eval_score;
    int32_t threshold;
    int32_t top_scores[4];
    int32_t dead_score;

    AIPlayer(uint64_t initial_board) {
        max_d = 3;
        max_layer = 5;
        best_operation = 0;
        board = initial_board;
        board_sum = 0;
        node = 0;
        spawn_rate4 = 0.1f;
        spawn_weight_4 = 6553;  // 默认 0.1 * 65536
        spawn_weight_2 = 58983;
        do_check = 0;
        masked_count = 0;
        min_masked_tile = 0;
        fixed_mask = 0;
        max_threads = 2;
        cache.abort_ptr = &stop_search;
        cache.dead_score_ptr = &dead_score;
        
        prune = 0;                 // 默认关闭剪枝
        threshold = 4000;          // 剪枝阈值
        initial_eval_score = 0;
        dead_score = 131072;
        for (int i = 0; i < 4; ++i) top_scores[i] = -dead_score;
    }

    void clear_cache() {
        cache.clear();
    }

    void reset_board(uint64_t new_board) {
        best_operation = 0;
        board = new_board;
        node = 0;
    }

    void update_spawn_rate(double new_rate) {
        spawn_rate4 = new_rate;
        // 将浮点概率映射到 0~65536 的整数区间
        spawn_weight_4 = static_cast<int32_t>(new_rate * 65536.0);
        spawn_weight_2 = 65536 - spawn_weight_4;
    }

    inline int32_t check_corner(uint64_t board) const {
        if (masked_count == 3) {
            // 减 3000 分的掩码
            static constexpr uint64_t m3_3000[] = {
                0xf00000000f00fULL, 0xf00f00000000000fULL, 
                0xf00000000000f00fULL, 0xf00f00000000f000ULL
            };
            for (uint64_t m : m3_3000) {
                if ((board & m) == m) return 3000;
            }

            // 减 2000 分的掩码
            static constexpr uint64_t m3_2000[] = {
                0xf0000000000f00fULL, 0xf000000000f00fULL, 
                0xf0000000000ff000ULL, 0xf000000f0000f000ULL, 
                0xf00f0000000000f0ULL, 0xf00f000000000f00ULL, 
                0xf0000f000000fULL, 0xff0000000000fULL
            };
            for (uint64_t m : m3_2000) {
                if ((board & m) == m) return 2000;
            }

            // 减 1000 分的掩码
            static constexpr uint64_t m3_1000[] = {
                0xff00fULL, 0xf000f00fULL, 
                0xf00000000000ff00ULL, 0xff0000000000f000ULL, 
                0xf00ff00000000000ULL, 0xf00f000f00000000ULL, 
                0xff00000000000fULL, 0xf0000000000ffULL
            };
            for (uint64_t m : m3_1000) {
                if ((board & m) == m) return 1000;
            }
            
        } else if (masked_count == 4) {
            // 减 3000 分的掩码
            static constexpr uint64_t m4_3000[] = {
                0xff00f00fULL, 0xff00f00000000fULL, 
                0xfff00fULL, 0xff000f000000f000ULL, 
                0xf00000000f00ff00ULL, 0xf00f00ff00000000ULL, 
                0xf00fff0000000000ULL, 0xf000000f000ffULL
            };
            for (uint64_t m : m4_3000) {
                if ((board & m) == m) return 3000;
            }

            // 减 1200 分的掩码
            static constexpr uint64_t m4_1200[] = {
                0xf000f000f00fULL, 0xf000ff00fULL, 
                0xfff000000000f000ULL, 0xf00000000000fff0ULL, 
                0xf00f000f000f0000ULL, 0xf00ff000f0000000ULL, 
                0xf000000000fffULL, 0xfff00000000000fULL
            };
            for (uint64_t m : m4_1200) {
                if ((board & m) == m) return 1200;
            }

            // 减 600 分的掩码
            static constexpr uint64_t m4_600[] = {
                0xf000f0ffULL, 0xff000000f000f000ULL, 
                0xff0f000f00000000ULL, 0xf000f000000ffULL
            };
            for (uint64_t m : m4_600) {
                if ((board & m) == m) return 600;
            }
        }
        
        return 0; // 没有命中任何糟糕的位置
    }

    int32_t evaluate(uint64_t s) {
        // node++;
        uint64_t s_reverse = reverse_board(s);
        int32_t sum_x1 = 0, sum_x2 = 0, sum_y1 = 0, sum_y2 = 0;

        for (int i = 0; i < 4; ++i) {
            uint16_t l1 = (s >> (16 * i)) & 0xFFFF;
            uint16_t l2 = (s_reverse >> (16 * i)) & 0xFFFF;
            ScorePair px = _diffs_merged[l1];
            sum_x1 += px.d1;
            sum_x2 += px.d2;
            
            ScorePair py = _diffs_merged[l2];
            sum_y1 += py.d1;
            sum_y2 += py.d2;
        }

        int32_t result = std::max(sum_x1, sum_x2) + std::max(sum_y1, sum_y2);
        
        if (do_check) {
            result -= check_corner(s);
        }
        
        return result;
    }

    int32_t search0(uint64_t b) {
        int32_t best = -dead_score;
        auto moves = s_move_board_all(b);

        for (int i = 0; i < 4; ++i) {
            if (moves[i].is_valid) {
                int32_t processed_score = process_score(moves[i].score);
                // 考虑到计算和访存开销，这里不更新缓存
                int32_t temp = evaluate(moves[i].board);

                if (temp + processed_score > best) {
                    best = temp + processed_score;
                }
            }
        }
        return best;
    }

    int32_t process_score(uint32_t score) {
        int32_t s = static_cast<int32_t>(score);
        if (s < 200) return std::max(0, (s >> 2) - 10);
        if (s < 500) return (s >> 1) - 12;
        if (s < 1000) return (s >> 1) + 144;
        return static_cast<int32_t>(score << 1);
    }

    inline int32_t search_branch(uint64_t t, int32_t depth, int32_t sum_increment) {
        // node++;
        // 剪枝
        if (depth < max_d - 2) {
            int32_t current_eval = evaluate(t);
            bool mask_unmoved = (t & fixed_mask) == fixed_mask;
            if (!mask_unmoved) {
                return current_eval * 2 - (dead_score >> 1); 

            }
            if (current_eval < initial_eval_score - threshold) {
                return current_eval * 2 - (dead_score); 
            }
        }

        int32_t temp = 0;
        int32_t cached_val = 0;
        
        uint64_t cache_idx = cache.hash(t);
        if (!cache.lookup(cache_idx, t, depth, cached_val)) {
            int empty_slots_count = 0;
            int64_t local_temp = 0; 

            uint64_t x = t | (t >> 1);
            x |= (x >> 2);
            uint64_t empty_mask = (~x) & 0x1111111111111111ULL;

            while (empty_mask != 0) {
                empty_slots_count++;
                
                int bit_pos = __builtin_ctzll(empty_mask);

                uint64_t t4 = t | (2ULL << bit_pos);
                int32_t score_4 = search_ai_player(t4, depth - 1, sum_increment + 2);
                
                uint64_t t2 = t | (1ULL << bit_pos);
                int32_t score_2 = search_ai_player(t2, depth - 1, sum_increment + 1);
                
                int64_t temp_t = (static_cast<int64_t>(score_2) * spawn_weight_2 + 
                                  static_cast<int64_t>(score_4) * spawn_weight_4) >> 16;
                
                local_temp += temp_t;
                empty_mask &= (empty_mask - 1);
            }
            
            temp = static_cast<int32_t>(local_temp / empty_slots_count);
            
            cache.update(cache_idx, t, depth, temp);

        } else {
            temp = cached_val;
        }

        return (int32_t)temp;
    }

    int32_t search_ai_player(uint64_t b, int32_t depth, int32_t sum_increment) {
        if (stop_search.load(std::memory_order_relaxed)) return 0;
        if (depth <= 0) return search0(b);
        if (sum_increment > max_layer) return search0(b);

        int32_t best = -dead_score;

        // 顶层和次顶层启用 Task 任务池并行模式
        if (depth >= max_d - 2) {
            int32_t scores[4] = {-dead_score, -dead_score, -dead_score, -dead_score};
            auto moves = s_move_board_all(b);

            for (int i = 0; i < 4; ++i) {
                if (moves[i].is_valid) {
                    uint64_t t = moves[i].board;
                    int32_t processed_score = process_score(moves[i].score);
                    
                    // 将分支封装为 Task 丢进全局任务池
                    #pragma omp task shared(scores) firstprivate(i, t, processed_score, depth, sum_increment)
                    {
                        scores[i] = search_branch(t, depth, sum_increment) + processed_score + 1;
                    }
                }
            }

            #pragma omp taskwait

            // 串行汇总逻辑
            for (int i = 0; i < 4; ++i) {
                if (scores[i] > best) {
                    best = scores[i];

                    if (depth == max_d) {
                        best_operation = i + 1;
                    }
                }
                
                if (depth == max_d) {
                    top_scores[i] = scores[i];
                }
            }
        } else {
            // 纯串行层
            auto moves = s_move_board_all(b);
            
            for (int i = 0; i < 4; ++i) {
                if (moves[i].is_valid) {
                    uint64_t t = moves[i].board;
                    uint32_t score = moves[i].score;
                    
                    int32_t current_depth = depth; // 防止修改外层 depth
                    if (score > 250 && score < 5000) {
                        int32_t min_depth = __builtin_clz(score) - 16;
                        current_depth = std::min(min_depth, std::max(current_depth, 2));
                    }
                    
                    int32_t temp = search_branch(t, current_depth, sum_increment) + process_score(score);
                    if (temp > best) best = temp;
                }
            }
        }

        return best;
    }

    void start_search(int32_t depth = 3) {
        best_operation = 0;
        max_d = depth;
        max_layer = (uint32_t)(depth * (1.15 + spawn_rate4 * 1.5) + 2.3);
        cache.clear();

        initial_eval_score = evaluate(board);

        node = 0; 
        
        for (int i = 0; i < 4; ++i) top_scores[i] = -dead_score;
        
        // 不修改成员 board
        uint64_t masked_board = apply_dynamic_mask();
        
        int32_t current_max_threads = max_threads;
        if (depth < 6) {
            current_max_threads = std::min(max_threads, 4);
        }
        omp_set_num_threads(current_max_threads);
        
        #pragma omp parallel
        {
            #pragma omp single
            {
                search_ai_player(masked_board, max_d, 0);
            }
        }
    };

    uint64_t apply_dynamic_mask() {
        uint8_t counts[16] = {0};
        board_sum = 0;
        fixed_mask = 0;
        
        uint64_t current_board = board;
        
        // 1. 统计各个数字的个数与盘面总和
        uint64_t temp = current_board;
        for (int i = 0; i < 16; ++i) {
            uint8_t tile = temp & 0xF;
            counts[tile]++;
            if (tile > 0) {
                board_sum += (1 << tile); 
            }
            temp >>= 4;
        }

        // 2. 统计大数特征并更新 dead_score
        int count_gt_128 = 0;      // 记录 > 128 (即 >= 256，索引 >= 8) 的数字个数
        int count_gt_256 = 0;      // 记录 > 256 (即 >= 512，索引 >= 9) 的数字个数
        bool distinct_gt_256 = true; // 记录 > 256 的数字是否互不相同

        // 只需要遍历我们关心的大数索引范围 (8 ~ 15)
        for (int i = 8; i < 16; ++i) {
            count_gt_128 += counts[i];
            
            if (i >= 9) {
                count_gt_256 += counts[i];
                // 如果某个 > 256 的数字出现超过 1 次，说明它们不是互不相同的
                if (counts[i] > 1 && i < 15) {
                    distinct_gt_256 = false;
                }
            }
        }

        // 3. 根据游戏阶段调整 dead_score
        int32_t large_tiles = large_tile_count(8, counts);
        if (large_tiles > 4) threshold = (prune == 0) ? 5600 : 2400;
        if (large_tiles <= 4) threshold = (prune == 0) ? 8400 : 3200;

        if (count_gt_128 <= 4) {
            dead_score = 524288;
        } else if (count_gt_128 == 5) {
            dead_score = 262144;
        } else if (count_gt_256 > 5 && distinct_gt_256) {
            if (count_gt_128 == 8 && (counts[7] == 1 || counts[6] >= 1)) {
                dead_score = 4096;
            } else if (count_gt_256 >= 6 && 960 < board_sum % 1024){
                dead_score = 65536;
            } else {
                dead_score = 96000;
            }
        } else {
            dead_score = 131072;
        }
        
        // 如果 > 256 的数字不互不相同，直接返回未掩码的 current_board
        if (!distinct_gt_256) return current_board; 

        uint32_t rem = board_sum % 1024;
        
        // 4. mask
        // 条件：(余数在40~512之间 且 不存在512) 或 (余数在512~1000之间)
        bool condA_part1 = (rem >= 40 && rem <= 512 && counts[9] == 0);
        bool condA_part2 = (rem >= 512 || rem < 6);
        
        if (condA_part1 || condA_part2) {
            if (rem > 1000) {
                current_board = mask(current_board, 11);
            }
            else {
                current_board = mask(current_board, 9);
            }
        }

        // 5. 统计 0xF 的个数并更新 masked_count
        uint64_t x = current_board & (current_board >> 1);
        x &= (x >> 2);
        x &= 0x1111111111111111ULL;
        masked_count = __builtin_popcountll(x);

        // 计算被 mask 的数中的最小值
        uint8_t min_masked_tile = 0; 
        if (masked_count > 0) {
            int accumulated = 0;
            for (int i = 15; i >= 0; --i) {
                accumulated += counts[i];
                if (accumulated >= masked_count) {
                    min_masked_tile = (uint8_t)i;
                    break;
                }
            }
        }

        // 6. 根据 current_board 更改评分表中特定索引的分数 
        // 统计大于 512 的数 (即 1024, 2048, 4096, ... 对应索引 10 到 15)
        int count_gt_512 = 0;
        for (int i = 10; i < 16; ++i) {
            count_gt_512 += counts[i];
        }
        bool flag1 = count_gt_512 == 5 && (counts[9] == 1 || counts[8] > 0) && board_sum % 512 < 72; // L3-512 628
        bool flag2 = ((count_gt_512 == 6 || (count_gt_512 == 5 && counts[9] == 1)) && board_sum % 1024 < 72) || // L3-1k 63L
                     (count_gt_512 == 5 && counts[9] == 0 && (counts[8] == 1 || counts[7] > 0) && board_sum % 256 < 72); // L3-512 之前的 63L 256
        bool flag3 = count_gt_512 >= 4 && masked_count >= 5 && (board_sum % 256 > 132 && board_sum % 256 < 234); // 处理 63L 128 滑出
        bool flag4 = count_gt_512 == 4 && counts[9] == 1 && counts[10] == 1 && board_sum % 256 < 60; // 其余 L3 256
        update_specific_scores(current_board, flag1, flag2, flag3, flag4);

        // 7. 计算固定大数掩码
        if ((min_masked_tile == 12 && masked_count == 4) ||
            (min_masked_tile == 11 && masked_count == 5) ||
            (min_masked_tile == 10 && masked_count == 6)
        ) {
            fixed_mask = get_23_mask(current_board) | get_22_mask(current_board);
        } else if (masked_count >= 5 && (large_tiles > 5 || (board_sum % 256 > 48 && board_sum % 256 < 234))) {
            fixed_mask = get_23_mask(current_board) | get_22_mask(current_board);
        } else if (do_check || prune == 0 || ((board_sum % 256 > 246 || board_sum % 256 < 24) && large_tiles < 5
            ) || counts[7] > 1 || (counts[6] > 1 && counts[7] == 1)) {
            fixed_mask = 0; // 不启用固定掩码
        } else {
            if (masked_count >= 4) {
                fixed_mask = get_22_mask(current_board) | get_21_mask(current_board);
            } else if (masked_count == 3) {
                fixed_mask = get_21_mask(current_board) | get_20_mask(current_board);
            } else if (masked_count == 2) {
                fixed_mask = get_20_mask(current_board);
            }
        }
        
        return current_board;
    };
};


// ------------------------------------------------------------------
// EvilGen 类
// ------------------------------------------------------------------
class EvilGen {
public:
    int32_t max_d;
    uint8_t hardest_pos;
    uint8_t hardest_num;
    uint64_t board;
    Cache cache;
    uint64_t node; 
    uint32_t dead_score;

    EvilGen(uint64_t initial_board) : cache() {
        max_d = 3;
        hardest_pos = 0;
        hardest_num = 1;
        board = initial_board;
        node = 0;
        dead_score = 131072;
    }

    void reset_board(uint64_t new_board) {
        hardest_pos = 0;
        hardest_num = 1;
        board = new_board;
        node = 0;
    }

    // 🚀 优化 1：与 AI 引擎同步的极速 evaluate
    int32_t evaluate(uint64_t s) {
        node++;
        uint64_t s_reverse = reverse_board(s);
        int32_t sum_x1 = 0, sum_x2 = 0, sum_y1 = 0, sum_y2 = 0;

        for (int i = 0; i < 4; ++i) {
            uint16_t l1 = (uint16_t)(s >> (16 * i));
            uint16_t l2 = (uint16_t)(s_reverse >> (16 * i));
            
            ScorePair px = _diffs_merged[l1];
            sum_x1 += px.d1;
            sum_x2 += px.d2;
            
            ScorePair py = _diffs_merged[l2];
            sum_y1 += py.d1;
            sum_y2 += py.d2;
        }

        return std::max(sum_x1, sum_x2) + std::max(sum_y1, sum_y2);
    }

    // 寻找让玩家得分最小的生成位置
    int32_t search_evil_gen(uint64_t b, int32_t depth) {
        int32_t evil = dead_score; 

        uint64_t x = b | (b >> 1);
        x |= (x >> 2);
        uint64_t empty_mask = (~x) & 0x1111111111111111ULL;

        while (empty_mask != 0) {
            int bit_pos = __builtin_ctzll(empty_mask);
            int pos_idx = bit_pos >> 2;

            for (uint64_t num : {2ULL, 1ULL}) {
                uint64_t t = b | (num << bit_pos);
                int32_t best = -dead_score; 

                auto moves = s_move_board_all(t);

                for (int d = 0; d < 4; ++d) {
                    if (moves[d].is_valid) {
                        uint64_t t1 = moves[d].board;
                        uint32_t score = moves[d].score;

                        int32_t temp = 0;
                        
                        uint64_t cache_idx = cache.hash(t1);
                        if (!cache.lookup(cache_idx, t1, depth, temp)) {
                            temp = (depth > 1) ? search_evil_gen(t1, depth - 1) : evaluate(t1);
                            cache.update(cache_idx, t1, depth, temp);
                        }

                        best = std::max(best, temp + (int32_t)(score << 1));
                        
                        // Alpha-Beta 剪枝
                        if (best >= evil) break; 
                    }
                }

                if (best <= evil) {
                    evil = best;
                    if (max_d == depth) {
                        hardest_pos = static_cast<uint8_t>(pos_idx);
                        hardest_num = static_cast<uint8_t>(num);
                    }
                }
            }
        
            empty_mask &= (empty_mask - 1);
        }
        
        return evil;
    }

    int32_t dispatcher(uint64_t current_board) {
        node = 0;
        return search_evil_gen(current_board, max_d);
    }

    void start_search(int32_t depth = 4) {
        cache.clear();
        max_d = depth;
        dispatcher(board);
    }

    // 返回: (新棋盘, Python矩阵索引位置(15-pos), 生成的数字值)
    std::tuple<uint64_t, uint8_t, uint8_t> gen_new_num(int32_t depth = 4) {
        start_search(depth);
        uint64_t new_board = board | ((uint64_t)hardest_num << (4 * hardest_pos));
        return {new_board, static_cast<uint8_t>(15 - hardest_pos), hardest_num};
    }
};

extern std::array<float, 4> find_best_egtb_move(uint64_t target_board, int table_type);


// int benchmark() {
//     uint64_t initial_board = 0x11212451ea86fc97ULL;
//     AIPlayer cpp_ai(initial_board);
    
//     cpp_ai.max_threads = 4;
//     cpp_ai.do_check = 0;
//     cpp_ai.prune = 0;

//     int32_t test_depth = 12;

//     std::cout << "[*] Starting search..." << std::endl;
//     std::cout << "[*] Board: 0x" << std::hex << initial_board << std::dec << std::endl;
//     std::cout << "[*] Depth: " << test_depth << std::endl;
//     std::cout << "[*] Threads: " << cpp_ai.max_threads << "\n" << std::endl;

//     auto start_time = std::chrono::high_resolution_clock::now();

//     cpp_ai.start_search(test_depth);

//     auto end_time = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> diff = end_time - start_time;

//     std::cout << "-----------------------------------" << std::endl;
//     std::cout << "Search completed in : " << diff.count() << " seconds." << std::endl;
//     std::cout << "Best operation      : " << (int)cpp_ai.best_operation << std::endl;
//     std::cout << "Scores              : " << cpp_ai.top_scores[0] << ", " << cpp_ai.top_scores[1] << ", " << cpp_ai.top_scores[2] << ", " << cpp_ai.top_scores[3] << std::endl;
//     std::cout << "-----------------------------------" << std::endl;

//     return 0;
// }

// ------------------------------------------------------------------
// Pybind11 绑定代码
// ------------------------------------------------------------------
NB_MODULE(ai_core, m) {
    m.def("find_best_egtb_move", &find_best_egtb_move, 
        nb::arg("board"), nb::arg("table_type"));
          
    nb::class_<AIPlayer>(m, "AIPlayer")
        .def(nb::init<uint64_t>())
        .def("reset_board", &AIPlayer::reset_board, nb::arg("new_board"))
        .def("clear_cache", &AIPlayer::clear_cache)
        .def("update_spawn_rate", &AIPlayer::update_spawn_rate, nb::arg("new_rate"))
        .def("start_search", &AIPlayer::start_search, 
             nb::arg("depth") = 3, 
             nb::call_guard<nb::gil_scoped_release>())
        
        // 使用 nanobind 简写：def_rw
        .def_rw("max_d", &AIPlayer::max_d)
        .def_rw("best_operation", &AIPlayer::best_operation)
        .def_rw("board", &AIPlayer::board)
        .def_rw("max_threads", &AIPlayer::max_threads)
        .def_rw("do_check", &AIPlayer::do_check)
        .def_rw("spawn_rate4", &AIPlayer::spawn_rate4)
        .def_rw("prune", &AIPlayer::prune)

        // 使用 nanobind 简写：def_ro
        .def_ro("masked_count", &AIPlayer::masked_count)
        .def_ro("fixed_mask", &AIPlayer::fixed_mask)
        .def_ro("dead_score", &AIPlayer::dead_score)

        // 使用 nanobind 简写：def_prop_ro
        .def_prop_ro("scores", [](const AIPlayer &self) {
            return std::make_tuple(self.top_scores[0], self.top_scores[1], 
                                self.top_scores[2], self.top_scores[3]);
        })

        .def_prop_ro("node", [](const AIPlayer &self) {
            return self.node.load();
        })

        // 使用 nanobind 简写：def_prop_rw
        .def_prop_rw("stop_search", 
            [](const AIPlayer &self) {
                return self.stop_search.load(std::memory_order_relaxed);
            }, 
            [](AIPlayer &self, bool value) {
                self.stop_search.store(value, std::memory_order_relaxed);
        });


    nb::class_<EvilGen>(m, "EvilGen")
        .def(nb::init<uint64_t>(), nb::arg("initial_board"))
        .def("reset_board", &EvilGen::reset_board, nb::arg("new_board"))
        .def("start_search", &EvilGen::start_search, 
             nb::arg("depth") = 4, 
             nb::call_guard<nb::gil_scoped_release>())
        .def("dispatcher", &EvilGen::dispatcher, 
             nb::arg("board"), 
             nb::call_guard<nb::gil_scoped_release>())
        .def("gen_new_num", &EvilGen::gen_new_num, 
             nb::arg("depth") = 4, 
             nb::call_guard<nb::gil_scoped_release>())
        
        .def_rw("max_d", &EvilGen::max_d)
        .def_rw("hardest_pos", &EvilGen::hardest_pos)
        .def_rw("hardest_num", &EvilGen::hardest_num)
        .def_rw("board", &EvilGen::board)
        
        .def_prop_ro("node", [](const EvilGen &self) {
            return self.node;
        });
}
