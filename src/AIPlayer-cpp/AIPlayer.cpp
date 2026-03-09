#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <optional>
#include <tuple>
#include <omp.h>
#include <atomic>


namespace py = pybind11;

// 从 BoardMover.cpp 引入外部函数
extern std::tuple<uint64_t, uint32_t> s_move_board(uint64_t board, int direction);

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

inline uint64_t reverse_board(uint64_t board) {
    board = (board & 0xFF00FF0000FF00FFULL) | ((board & 0x00FF00FF00000000ULL) >> 24) | ((board & 0x00000000FF00FF00ULL) << 24);
    board = (board & 0xF0F00F0FF0F00F0FULL) | ((board & 0x0F0F00000F0F0000ULL) >> 12) | ((board & 0x0000F0F00000F0F0ULL) << 12);
    return board;
}

// ------------------------------------------------------------------
// 评估函数与预计算表
// ------------------------------------------------------------------
int16_t _diffs[65536];
int16_t _diffs2[65536];


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
                score_dpdf -= line_masked[x + 1] << 1;
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
        if (line_masked[3] > 60) {
            score_dpdf += line_masked[3] >> 3;
        }
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

        _diffs[i] = diffs_evaluation_func(line_masked);
        _diffs2[i] = diffs_evaluation_func(line_masked_rev);
    }
}

struct EvalInitializer {
    EvalInitializer() { init_evaluate_tables(); }
};
static EvalInitializer _eval_init;


// ------------------------------------------------------------------
// Cache 类
// ------------------------------------------------------------------
class Cache {
private:
    std::atomic<bool> dummy_false{false};
    
    // 生成次级哈希 (32位)
    inline uint32_t get_signature(uint64_t board) const {
        return static_cast<uint32_t>(board + (board >> 32));
    }

public:
    std::atomic<bool>* abort_ptr = &dummy_false; 
    uint64_t length = 2097151; 
    
    std::vector<uint64_t> cache_board;

    Cache() : cache_board(length + 1, 0) {}

    void clear() {
        std::fill(cache_board.begin(), cache_board.end(), 0);
    }

    // 独立提供给外部调用的主哈希函数
    inline uint64_t hash(uint64_t board) const {
        uint64_t hashed = (((board ^ (board >> 27)) * 0x1A85EC53ULL) + board) >> 23;
        return hashed & length;
    }

    inline bool lookup(uint64_t index, uint64_t board, int32_t depth, int32_t& out_score) {
        uint64_t entry = cache_board[index];
        
        if (entry == 0) return false;

        uint32_t current_sig = get_signature(board);
        uint32_t cached_sig = static_cast<uint32_t>(entry >> 32);

        // 次级哈希碰撞测试
        if (cached_sig == current_sig) {
            int32_t cached_depth = static_cast<int32_t>(entry & 0xFF);
            
            if (cached_depth >= depth) {
                // 提取中段 24 位分数，并减去偏移量还原负数
                uint32_t unsigned_score = (entry >> 8) & 0xFFFFFF;
                out_score = static_cast<int32_t>(unsigned_score) - 8388608;
                return true;
            }
        }
        return false;
    }

    inline void update(uint64_t index, uint64_t board, int32_t depth, int32_t score) {
        if (abort_ptr->load(std::memory_order_relaxed)) return;
        
        uint64_t entry = cache_board[index];

        uint32_t current_sig = get_signature(board);
        uint32_t cached_sig = static_cast<uint32_t>(entry >> 32);
        int32_t cached_depth = static_cast<int32_t>(entry & 0xFF);

        // 仅在数据更优时写入
        if (entry == 0 || cached_sig != current_sig || depth >= cached_depth) {
            
            uint8_t pack_depth = static_cast<uint8_t>(std::max(0, std::min(255, depth)));
            int32_t clamped_score = std::max(-8388608, std::min(8388607, score));
            uint32_t pack_score = static_cast<uint32_t>(clamped_score + 8388608) & 0xFFFFFF;

            uint64_t new_entry = (static_cast<uint64_t>(current_sig) << 32) | 
                                 (static_cast<uint64_t>(pack_score) << 8) | 
                                 static_cast<uint64_t>(pack_depth);

            cache_board[index] = new_entry;
        }
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
    uint8_t do_check;
    uint8_t masked_count;

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
        do_check = 0;
        masked_count = 0;
        max_threads = 2;
        cache.abort_ptr = &stop_search;
        
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

            // 减 200 分的掩码
            static constexpr uint64_t m4_200[] = {
                0xf000f0ffULL, 0xff000000f000f000ULL, 
                0xff0f000f00000000ULL, 0xf000f000000ffULL
            };
            for (uint64_t m : m4_200) {
                if ((board & m) == m) return 200;
            }
        }
        
        return 0; // 没有命中任何糟糕的位置
    }

    int32_t evaluate(uint64_t s) {
        uint64_t s_reverse = reverse_board(s);
        int32_t sum_x1 = 0, sum_x2 = 0, sum_y1 = 0, sum_y2 = 0;

        for (int i = 0; i < 4; ++i) {
            uint16_t l1 = (s >> (16 * i)) & 0xFFFF;
            uint16_t l2 = (s_reverse >> (16 * i)) & 0xFFFF;
            sum_x1 += _diffs[l1];  
            sum_x2 += _diffs2[l1]; 
            sum_y1 += _diffs[l2];  
            sum_y2 += _diffs2[l2]; 
        }

        int32_t result = std::max(sum_x1, sum_x2) + std::max(sum_y1, sum_y2);
        
        if (do_check) {
            result -= check_corner(s);
        }
        
        return result;
    }

    int32_t search0(uint64_t b) {
        int32_t best = -dead_score;
        static constexpr int directions[4] = {1, 2, 3, 4};
        
        for (int d : directions) {
            auto [t, score] = s_move_board(b, d);
            if (t == b) continue;

            int32_t processed_score = processe_score(score);
            int32_t temp = 0;
            
            // 计算一次 Hash 并传入 Cache 查找
            uint64_t cache_idx = cache.hash(t);

            if (!cache.lookup(cache_idx, t, 0, temp)) {
                temp = evaluate(t);
                cache.update(cache_idx, t, 0, temp);
            }

            if (temp + processed_score >= best) {
                best = temp + processed_score;
            }
        }
        return best;
    }

    inline int32_t search_branch(uint64_t t, int32_t depth, int32_t sum_increment) {
        if (stop_search.load(std::memory_order_relaxed)) return 0;

        // 剪枝
        if (depth < max_d - 2) {
            int32_t current_eval = evaluate(t);
            if (current_eval < initial_eval_score - threshold) {
                return current_eval * 2 - (dead_score >> 2); 
            }
        }

        double temp = 0.0;
        int32_t cached_val = 0;
        
        uint64_t cache_idx = cache.hash(t);
        if (!cache.lookup(cache_idx, t, depth, cached_val)) {
            int empty_slots_count = 0;
            double local_temp = 0.0; 
            
            for (int j = 0; j < 16; ++j) {
                if (((t >> (4 * j)) & 0xF) == 0) {
                    empty_slots_count++;

                    uint64_t t4 = t | (2ULL << (4 * j));
                    int32_t score_4 = search_ai_player(t4, depth - 1, sum_increment + 2);
                    uint64_t t2 = t | (1ULL << (4 * j));
                    int32_t score_2 = search_ai_player(t2, depth - 1, sum_increment + 1);
                    double temp_t = score_2 * (1.0 - spawn_rate4) + score_4 * spawn_rate4;
            
                    local_temp += temp_t;
                }
            }
            temp = local_temp / empty_slots_count; // 必不为0，因为移动后至少有一个空位
            
            // 缓存未命中且计算完成后，利用同一个 index 更新缓存
            cache.update(cache_idx, t, depth, (int32_t)temp);
        } else {
            temp = cached_val;
        }

        return (int32_t)temp;
    }

    int32_t processe_score(int32_t score) {
        if (score < 200) {
            return std::max(0, (int32_t)((score >> 2) - 10));
        } else if (score < 800) {
            return (int32_t)((score >> 1) - 20);
        } else {
            return (int32_t)(score << 1);
        }
    }

    int32_t search_ai_player(uint64_t b, int32_t depth, int32_t sum_increment) {
        if (stop_search.load(std::memory_order_relaxed)) return 0;
        if (depth == 0) return search0(b);
        if (sum_increment > max_layer) return search0(b);

        int32_t best = -dead_score;

        // 顶层和次顶层启用 Task 任务池并行模式
        if (depth >= max_d - 2) {
            int32_t scores[4] = {-dead_score, -dead_score, -dead_score, -dead_score};

            for (int i = 0; i < 4; ++i) {
                auto [t, score] = s_move_board(b, i + 1);
                if (t != b) {
                    int32_t processed_score = processe_score(score);
                    // 2. 将分支封装为 Task 丢进全局任务池
                    // firstprivate 捕获当前的上下文变量，shared 暴露局部数组以供写入
                    #pragma omp task shared(scores) firstprivate(i, t, score, depth, sum_increment)
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
            
            // 1. 残局成功剪枝逻辑
            if (board_sum % 1024 < 40 && depth < max_d - 4) {
                uint64_t next_t[4];
                uint32_t next_score[4];
                // 预计算 4 个方向，寻找大合并
                for (int i = 0; i < 4; ++i) {
                    auto [t, score] = s_move_board(b, i + 1);
                    
                    if (score > 500) {
                        depth = std::min(depth, 4); // 大合并成功，减少搜索深度
                    }
                    next_t[i] = t;
                    next_score[i] = processe_score(score);
                }
                
                // 搜索刚才展开的盘面
                for (int i = 0; i < 4; ++i) {
                    if (next_t[i] != b) {
                        int32_t temp = search_branch(next_t[i], depth, sum_increment) + next_score[i];
                        if (temp > best) {
                            best = temp;
                        }
                    }
                }
            } 
            // 2. 常规搜索逻辑 (不满足大合并前置条件)
            else {
                for (int i = 0; i < 4; ++i) {
                    auto [t, score] = s_move_board(b, i + 1);
                    if (t != b) {
                        int32_t temp = search_branch(t, depth, sum_increment) + processe_score(score);
                        if (temp > best) best = temp;
                    }
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
        prune &= ~((uint8_t)((initial_eval_score - 320) >> 31));
        threshold = (prune == 0) ? 5600 : 2100;
        node = 0; 
        
        for (int i = 0; i < 4; ++i) top_scores[i] = -dead_score;
        
        // 不修改成员 board
        uint64_t masked_board = apply_dynamic_mask();

        omp_set_num_threads(max_threads);
        
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
        if (count_gt_128 <= 4) {
            dead_score = 524288;
        } else if (count_gt_128 == 5) {
            dead_score = 131072;
        } else if (count_gt_256 > 5 && distinct_gt_256) {
            if (count_gt_128 == 8 && counts[7] == 1) {
                dead_score = 2048;
            } else if (board_sum % 1024 < 96) {
                dead_score = 16384;
            } else {
                dead_score = 32768;
            }
        } else {
            dead_score = 65536;
        }
        
        // 如果 > 256 的数字不互不相同，直接返回未掩码的 current_board
        if (!distinct_gt_256) return current_board; 

        uint32_t rem = board_sum % 1024;
        
        // 1. 规则一： mask(current_board, 10)
        // 条件：(余数在40~512之间 且 不存在512) 或 (余数在512~984之间)
        bool condA_part1 = (rem >= 40 && rem <= 512 && counts[9] == 0);
        bool condA_part2 = (rem >= 512 && rem <= 984);
        
        // 且 1024 的数量不多于 1
        if ((condA_part1 || condA_part2) && (counts[10] <= 1)) {
            current_board = mask(current_board, 10);
        }
        
        // 2. 规则二： mask(current_board, 9)
        // 统计大于 512 的数 (即 1024, 2048, 4096, ... 对应索引 10 到 15)
        int count_gt_512 = 0;
        for (int i = 10; i < 16; ++i) {
            count_gt_512 += counts[i];
        }
        
        rem = board_sum % 512;
        // 条件：大于512不超过2 且 没有256(索引8) 且 有且仅有一个512(索引9) 且 余数<80
        if (count_gt_512 <= 2 && counts[8] == 0 && counts[9] == 1 && rem < 80) {
            current_board = mask(current_board, 9);
        }

        // 3. 统计 0xF 的个数并更新 masked_count
        uint64_t x = current_board & (current_board >> 1);
        x &= (x >> 2);
        x &= 0x1111111111111111ULL;
        masked_count = __builtin_popcountll(x);
        
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

    int32_t evaluate(uint64_t s) {
        node++;
        uint64_t s_reverse = reverse_board(s);
        int32_t sum_x1 = 0, sum_x2 = 0, sum_y1 = 0, sum_y2 = 0;

        for (int i = 0; i < 4; ++i) {
            uint16_t l1 = (s >> (16 * i)) & 0xFFFF;
            uint16_t l2 = (s_reverse >> (16 * i)) & 0xFFFF;
            sum_x1 += _diffs[l1];
            sum_x2 += _diffs2[l1];
            sum_y1 += _diffs[l2];
            sum_y2 += _diffs2[l2];
        }

        return std::max(sum_x1, sum_x2) + std::max(sum_y1, sum_y2);
    }

    // 寻找让玩家得分最小的生成位置
    int32_t search_evil_gen(uint64_t b, int32_t depth) {
        int32_t evil = dead_score; 
        static constexpr int directions[4] = {1, 2, 3, 4};

        for (int i = 0; i < 16; ++i) {
            if (((b >> (4 * i)) & 0xF) == 0) {
                for (uint64_t num : {2ULL, 1ULL}) {
                    uint64_t t = b | (num << (4 * i));
                    int32_t best = -dead_score; 

                    for (int d : directions) {
                        auto [t1, score] = s_move_board(t, d);
                        if (t1 == t) continue;

                        int32_t temp = 0;
                        
                        uint64_t cache_idx = cache.hash(t1);
                        if (!cache.lookup(cache_idx, t1, depth, temp)) {
                            temp = (depth > 1) ? search_evil_gen(t1, depth - 1) : evaluate(t1);
                            cache.update(cache_idx, t1, depth, temp);
                        }

                        best = std::max(best, temp + (int32_t)(score << 1));
                        
                        // Alpha-Beta 剪枝：如果 AI 在这个应对下能得到比目前已知最坏情况（evil）更高的分数，
                        if (best >= evil) break; 
                    }

                    // 记录最能让 AI 得低分的生成策略
                    if (best <= evil) {
                        evil = best;
                        if (max_d == depth) {
                            hardest_pos = i;
                            hardest_num = num;
                        }
                    }
                }
            }
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
        return {new_board, 15 - hardest_pos, hardest_num};
    }
};


extern std::vector<float> find_best_egtb_move(uint64_t target_board, int table_type);

// ------------------------------------------------------------------
// Pybind11 绑定代码
// ------------------------------------------------------------------
PYBIND11_MODULE(ai_core, m) {
    m.def("find_best_egtb_move", &find_best_egtb_move, 
        py::arg("board"), py::arg("table_type"));
          
    py::class_<AIPlayer>(m, "AIPlayer")
        .def(py::init<uint64_t>())
        .def("reset_board", &AIPlayer::reset_board, py::arg("new_board"))
        .def("clear_cache", &AIPlayer::clear_cache)
        .def("start_search", &AIPlayer::start_search, 
             py::arg("depth") = 3, 
             py::call_guard<py::gil_scoped_release>())
        
        // 普通属性读写
        .def_readwrite("max_d", &AIPlayer::max_d)
        .def_readwrite("best_operation", &AIPlayer::best_operation)
        .def_readwrite("board", &AIPlayer::board)
        .def_readwrite("max_threads", &AIPlayer::max_threads)
        .def_readwrite("do_check", &AIPlayer::do_check)
        .def_readwrite("spawn_rate4", &AIPlayer::spawn_rate4)
        .def_readwrite("prune", &AIPlayer::prune)

        .def_property_readonly("scores", [](const AIPlayer &self) {
            // 将顶层分数拷贝到 vector 返回给 Python
            return std::vector<int32_t>{
                self.top_scores[0], 
                self.top_scores[1], 
                self.top_scores[2], 
                self.top_scores[3]
            };
        })

        // 原子变量只读属性
        .def_property_readonly("node", [](const AIPlayer &self) {
            return self.node.load();
        })

        // 读写 stop_search 属性
        .def_property("stop_search", 
            [](const AIPlayer &self) {
                // getter: 返回当前原子变量的值
                return self.stop_search.load(std::memory_order_relaxed);
            }, 
            [](AIPlayer &self, bool value) {
                // setter: 修改原子变量的值
                self.stop_search.store(value, std::memory_order_relaxed);
        });


    py::class_<EvilGen>(m, "EvilGen")
        .def(py::init<uint64_t>(), py::arg("initial_board"))
        .def("reset_board", &EvilGen::reset_board, py::arg("new_board"))
        .def("start_search", &EvilGen::start_search, 
             py::arg("depth") = 4, 
             py::call_guard<py::gil_scoped_release>())
        .def("dispatcher", &EvilGen::dispatcher, 
             py::arg("board"), 
             py::call_guard<py::gil_scoped_release>())
        .def("gen_new_num", &EvilGen::gen_new_num, 
             py::arg("depth") = 4, 
             py::call_guard<py::gil_scoped_release>())
        
        // 属性读写映射
        .def_readwrite("max_d", &EvilGen::max_d)
        .def_readwrite("hardest_pos", &EvilGen::hardest_pos)
        .def_readwrite("hardest_num", &EvilGen::hardest_num)
        .def_readwrite("board", &EvilGen::board)
        
        .def_property_readonly("node", [](const EvilGen &self) {
            return self.node;
        });
}
