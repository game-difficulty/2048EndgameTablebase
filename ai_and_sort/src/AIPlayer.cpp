#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <omp.h>
#include <tuple>

// #include <iostream>
// #include <chrono>
// #include <thread>

#include "AIPlayer.h"
#include "BoardMover.h"
#include "CommonMover.h"

// 将盘面上 >= threshold 的块全部替换为 0xF
inline uint64_t mask(uint64_t board, int threshold) {
  uint64_t evens = board & 0x0F0F0F0F0F0F0F0FULL;
  uint64_t odds = (board >> 4) & 0x0F0F0F0F0F0F0F0FULL;
  uint64_t addend = (0x80 - threshold) * 0x0101010101010101ULL;
  uint64_t e_cmp = (evens + addend) & 0x8080808080808080ULL;
  uint64_t o_cmp = (odds + addend) & 0x8080808080808080ULL;
  uint64_t e_mask = (e_cmp >> 3) - (e_cmp >> 7);
  uint64_t o_mask = (o_cmp >> 3) - (o_cmp >> 7);
  return board | e_mask | (o_mask << 4);
}

inline uint64_t extract_0xf_nibbles(uint64_t v) {
  uint64_t temp = v & (v >> 1) & (v >> 2) & (v >> 3);
  return (temp & 0x1111111111111111ULL) * 0xF;
}

// 全局对齐声明，256KB 大小，Cache Line 对齐
alignas(64) ScorePair _diffs_merged[65536];

const int32_t TILE_WEIGHT_MAP[16] = {0,   2,   4,   8,   16,  32,  64,  128,
                                     248, 388, 488, 518, 519, 519, 519, 520};

int32_t diffs_evaluation_func(const int32_t *line_masked) {
  // dpdf 计算
  int32_t score_dpdf = line_masked[0];
  for (int x = 0; x < 3; ++x) {
    if (line_masked[x] < line_masked[x + 1]) {
      if (line_masked[x] > 400) {
        score_dpdf +=
            (line_masked[x] << 1) + (line_masked[x + 1] - line_masked[x]) * x;
      } else if (line_masked[x] > 300 && x == 1 &&
                 line_masked[0] > line_masked[1]) {
        score_dpdf += (line_masked[x] << 1);
      } else {
        score_dpdf -= (line_masked[x + 1] - line_masked[x]) << 3;
        score_dpdf -= line_masked[x + 1] * 3;
        if (x < 2 && line_masked[x + 2] < line_masked[x + 1] &&
            line_masked[x + 1] > 30) {
          score_dpdf -= std::max(80, line_masked[x + 1]);
        }
      }
    } else if (x < 2) { // 正序是越来越小
      score_dpdf += line_masked[x + 1] + line_masked[x];
    } else {
      score_dpdf += (int32_t)((line_masked[x + 1] + line_masked[x]) * 0.5);
    }
  }
  if (line_masked[0] > 400 && line_masked[1] > 300 && line_masked[2] > 200 &&
      line_masked[2] > line_masked[3] && line_masked[3] < 300) {
    score_dpdf += line_masked[3] >> 2;
  }

  // t 计算
  int32_t score_t;
  int32_t min_03 = std::min(line_masked[0], line_masked[3]);
  if (min_03 < 32) {
    score_t = -16384;
  } else if ((line_masked[0] < line_masked[1] && line_masked[0] < 400) ||
             (line_masked[3] < line_masked[2] && line_masked[3] < 400)) {
    score_t = -(std::max(line_masked[1], line_masked[2]) * 10);
  } else {
    score_t =
        (int32_t)((line_masked[0] * 1.8 + line_masked[3] * 1.8) +
                  std::max(line_masked[1], line_masked[2]) * 1.5 +
                  std::min(160, std::min(line_masked[1], line_masked[2])) *
                      2.5);
    if (std::min(line_masked[1], line_masked[2]) < 8) {
      score_t -= 60;
    }
  }

  int zero_count = 0;
  for (int k = 0; k < 4; ++k)
    if (line_masked[k] == 0)
      zero_count++;

  int32_t sum_123 = line_masked[1] + line_masked[2] + line_masked[3];
  int32_t penalty = 0;
  if (line_masked[0] > 100 &&
      ((zero_count > 1 && sum_123 < 32) || sum_123 < 12)) {
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

    _diffs_merged[i].d1 =
        static_cast<int16_t>(diffs_evaluation_func(line_masked));
    _diffs_merged[i].d2 =
        static_cast<int16_t>(diffs_evaluation_func(line_masked_rev));
  }
}

struct EvalInitializer {
  EvalInitializer() { init_evaluate_tables(); }
};
static EvalInitializer _eval_init;

int32_t large_tile_count(uint8_t start, uint8_t counts[]) {
  int sum = 0;
  int i = start;
  while (i < 16 && counts[i] != 0)
    i++;

  for (i++; i < 16; i++) {
    if (counts[i] == 2 && i < 15)
      return 0; // 发现重复，直接返回0
    sum += counts[i];
  }
  return sum;
}

bool is_5tiler(uint64_t board_sum, const uint8_t counts[]) {
  bool sum_range_cond = (board_sum > 62000 && board_sum < 65520);
  int sum_11_up = 0;
  for (int i = 11; i < 16; ++i) {
    sum_11_up += counts[i];
  }
  bool counts_cond = (sum_11_up == 4 && counts[10] == 0);
  uint64_t mod_val = board_sum % 1024;
  bool mod_cond = (mod_val < 24 || mod_val > 996);
  return (sum_range_cond || counts_cond) && mod_cond;
}

uint64_t get_23_mask(uint64_t board) {
  static const uint64_t masks[8] = {
      0xff00fff0ULL,         0xfff00ff00000000ULL,
      0xf00ff00ffULL,        0xff00ff00f0000000ULL,
      0xfff0ff0000000000ULL, 0xff00ff000f0000ULL,
      0xf000ff00ff00ULL,     0xff0fffULL};

  if ((board & masks[0]) == masks[0])
    return masks[0];
  if ((board & masks[1]) == masks[1])
    return masks[1];
  if ((board & masks[2]) == masks[2])
    return masks[2];
  if ((board & masks[3]) == masks[3])
    return masks[3];
  if ((board & masks[4]) == masks[4])
    return masks[4];
  if ((board & masks[5]) == masks[5])
    return masks[5];
  if ((board & masks[6]) == masks[6])
    return masks[6];
  if ((board & masks[7]) == masks[7])
    return masks[7];

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

  if ((board & masks[0]) == masks[0])
    return masks[0];
  if ((board & masks[1]) == masks[1])
    return masks[1];
  if ((board & masks[2]) == masks[2])
    return masks[2];
  if ((board & masks[3]) == masks[3])
    return masks[3];

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

  if ((board & masks[0]) == masks[0])
    return masks[0];
  if ((board & masks[1]) == masks[1])
    return masks[1];
  if ((board & masks[2]) == masks[2])
    return masks[2];
  if ((board & masks[3]) == masks[3])
    return masks[3];

  return 0;
}

uint64_t get_20_mask(uint64_t board) {
  static const uint64_t masks[8] = {0xff00ULL,
                                    0xff000000000000ULL,
                                    0xf000fULL,
                                    0xf000f00000000000ULL,
                                    0xff00000000000000ULL,
                                    0xf000f00000000ULL,
                                    0xf000f000ULL,
                                    0xffULL};

  if ((board & masks[0]) == masks[0])
    return masks[0];
  if ((board & masks[1]) == masks[1])
    return masks[1];
  if ((board & masks[2]) == masks[2])
    return masks[2];
  if ((board & masks[3]) == masks[3])
    return masks[3];
  if ((board & masks[4]) == masks[4])
    return masks[4];
  if ((board & masks[5]) == masks[5])
    return masks[5];
  if ((board & masks[6]) == masks[6])
    return masks[6];
  if ((board & masks[7]) == masks[7])
    return masks[7];

  return 0;
}

void update_specific_scores(uint64_t board, bool flag1, bool flag2, bool flag3,
                            bool flag4) {
  // 1. 判断 2x2 角落位运算判断
  uint64_t corner_mask = get_22_mask(board);
  bool apply_bonus = (corner_mask > 0);

  // 定义结构
  struct Target {
    uint16_t index;
    uint16_t index2;
    int16_t bonus;
  };
  struct DynamicTarget {
    uint16_t index;
    uint16_t index2;
    int16_t bonus_f1;
    int16_t bonus_f2;
  };

  // --- 使用 static 缓存基础评分 ---
  static int32_t base_scores_cache[3];    // 存储第一组的基础分
  static int32_t dynamic_scores_cache[5]; // 存储第二组的基础分
  static bool initialized = false;

  static const Target base_targets[] = {
      {0x78ff, 0xff87, 300},
      {0x7fff, 0xfff7, 320},
      {0x8fff, 0xfff8, 360},
  };

  static const DynamicTarget dynamic_targets[] = {
      {0x2fff, 0xfff2, 4, -4},   {0x3fff, 0xfff3, 12, -10},
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
    if (apply_bonus && flag3)
      score += base_targets[i].bonus;

    _diffs_merged[base_targets[i].index].d1 = (int16_t)score;
    _diffs_merged[base_targets[i].index2].d2 = (int16_t)score;
  }

  // 处理第二组
  for (int i = 0; i < 5; ++i) {
    int32_t score = dynamic_scores_cache[i];

    if (apply_bonus) {
      if (flag1)
        score += dynamic_targets[i].bonus_f1;
      else if (flag2)
        score += dynamic_targets[i].bonus_f2;
      else if (flag4)
        score += dynamic_targets[i].bonus_f1 >> 1;
    }

    _diffs_merged[dynamic_targets[i].index].d1 = (int16_t)score;
    _diffs_merged[dynamic_targets[i].index2].d2 = (int16_t)score;
  }
}

/**
 * 检查编码棋盘中 0xF (32768) 的数量。
 * 如果 2 个，则全部替换为 0xE (16384)。
 */
uint64_t resolve_32768_doubles(uint64_t board) {
  uint64_t m1 = board & (board >> 1);
  uint64_t m2 = m1 & (m1 >> 2);
  uint64_t f_mask = m2 & 0x1111111111111111ULL;
  if (__builtin_popcountll(f_mask) == 2) {
    return board & ~f_mask;
  }

  return board;
}

uint32_t get_max_layer(double spawnrate, double depth) {
  double v = std::sqrt(depth * spawnrate * (1.0 - spawnrate));
  double ceil_9999 = std::ceil(depth * (1.0 + spawnrate) + 3.72 * v);
  return static_cast<uint32_t>(std::min(depth * 2.0, ceil_9999));
}

// ------------------------------------------------------------------
// Cache 类
// ------------------------------------------------------------------

Cache::Cache() {
#ifdef _WIN32
  // Windows 下使用对齐内存分配
  table.reset(new (std::align_val_t(64)) CacheBucket[MAX_BUCKETS]);
#else
  // Linux/macOS 下使用 aligned_alloc
  void *ptr = std::aligned_alloc(64, MAX_BUCKETS * sizeof(CacheBucket));
  table.reset(reinterpret_cast<CacheBucket *>(ptr));
#endif
  // 初始状态全清零
  reset(MAX_BUCKETS);
}

void Cache::reset(uint64_t new_num_buckets) {
  current_num_buckets = new_num_buckets;
  current_mask = new_num_buckets - 1;

  // 获取底层指针进行高效清零
  uint64_t *ptr = reinterpret_cast<uint64_t *>(table.get());
  const size_t elements_to_clear =
      current_num_buckets * 8; // 每个 Bucket 有 8 个 entries

  std::fill(ptr, ptr + elements_to_clear, 0);
}

void Cache::clear() { reset(current_num_buckets); }

bool Cache::lookup(uint64_t bucket_idx, uint64_t board, int32_t depth,
                   int32_t &out_score, uint64_t &out_subtree_nodes) {
  uint32_t current_sig = get_signature(board);
  CacheBucket &bucket = table[bucket_idx];

  for (int i = 0; i < 8; ++i) {
    uint64_t entry = bucket.entries[i];
    if (entry == 0)
      continue;

    uint32_t cached_sig = static_cast<uint32_t>(entry >> 32);
    if (cached_sig == current_sig) {
      // 提取 6-bit Depth (位 6-11)
      int32_t cached_depth = static_cast<int32_t>((entry >> 6) & 0x3F);

      if (cached_depth >= depth) {
        // 提取 20-bit Score (位 12-31)，减去偏移量 524288 还原负数
        uint32_t unsigned_score = (entry >> 12) & 0xFFFFF;
        out_score = static_cast<int32_t>(unsigned_score) - 524288;

        // 提取 6-bit Effort (位 0-5)
        uint8_t out_effort = static_cast<uint8_t>(entry & 0x3F);
        out_subtree_nodes = 1ULL << out_effort;
        return true;
      }
    }
  }
  return false;
}

void Cache::update(uint64_t bucket_idx, uint64_t board, int32_t depth,
                   int32_t score, uint64_t subtree_nodes) {
  uint32_t current_sig = get_signature(board);
  CacheBucket &bucket = table[bucket_idx];

  if (score <= -(*dead_score_ptr) + 32) {
    depth = 63;
  }

  uint8_t pack_depth = static_cast<uint8_t>(depth);
  uint8_t pack_effort =
      static_cast<uint8_t>(63 - __builtin_clzll(subtree_nodes));

  // 20-bit 能够表示 -524288 到 +524287
  uint32_t pack_score = static_cast<uint32_t>(score + 524288) & 0xFFFFF;

  // 执行 64-bit 拼装
  uint64_t new_entry = (static_cast<uint64_t>(current_sig) << 32) |
                       (static_cast<uint64_t>(pack_score) << 12) |
                       (static_cast<uint64_t>(pack_depth) << 6) |
                       static_cast<uint64_t>(pack_effort);

  int min_effort_idx = 0;
  uint8_t min_effort_val = 255; // 初始设为极大值

  // --- 核心逻辑 ---
  for (int i = 0; i < 8; ++i) {
    uint64_t entry = bucket.entries[i];

    // 1. 发现空槽 (记录为绝对的最低优先权，等待被占用)
    if (entry == 0) {
      min_effort_val = 0;
      min_effort_idx = i;
      continue; // 遇到空槽继续看，防止并发情况下同一个签名被存两份
    }

    uint32_t cached_sig = static_cast<uint32_t>(entry >> 32);
    int32_t cached_depth = static_cast<int32_t>((entry >> 6) & 0x3F);
    uint8_t cached_effort = static_cast<uint8_t>(entry & 0x3F);

    // 2. 签名命中：同一个局面，更新或保持
    if (cached_sig == current_sig) {
      // 如果当前搜索深度更深
      if (depth > cached_depth) {
        bucket.entries[i] = new_entry;
      }
      return;
    }

    // 3. 统计该桶内最弱（Effort 计算量最小）的槽位
    if (cached_effort < min_effort_val) {
      min_effort_val = cached_effort;
      min_effort_idx = i;
    }
  }

  // 4. 运行到这里说明没有签名命中
  // 始终挤占计算量最小的槽位
  bucket.entries[min_effort_idx] = new_entry;
}

// ------------------------------------------------------------------
// AIPlayer 类
// ------------------------------------------------------------------

AIPlayer::AIPlayer(uint64_t initial_board)
    : stop_search(false), max_d(3), max_layer(5), best_operation(0),
      board(initial_board), board_sum(0), node(0), max_threads(2),
      spawn_rate4(0.1f), spawn_weight_4(6553), // 0.1 * 65536
      spawn_weight_2(58983), do_check(0), masked_count(0), min_masked_tile(0),
      fixed_mask(0), prune(0), initial_eval_score(0), threshold(4000),
      dead_score(131072) {
  // 1. 将 Cache 内部的控制指针指向 AIPlayer 的成员变量
  cache.abort_ptr = &stop_search;
  cache.dead_score_ptr = &dead_score;

  // 2. 初始化评分数组
  for (int i = 0; i < 4; ++i) {
    top_scores[i] = -dead_score;
  }
}

void AIPlayer::clear_cache() { cache.clear(); }

void AIPlayer::reset_board(uint64_t new_board) {
  best_operation = 0;
  board = new_board;
  node = 0;
}

void AIPlayer::update_spawn_rate(double new_rate) {
  spawn_rate4 = new_rate;
  // 将浮点概率映射到 0~65536 的整数区间
  spawn_weight_4 = static_cast<int32_t>(new_rate * 65536.0);
  spawn_weight_2 = 65536 - spawn_weight_4;
}

int32_t AIPlayer::check_corner(uint64_t board) const {
  if (masked_count == 6) {
    // 加 600 分的掩码
    static constexpr uint64_t m6_600_[] = {
        0xf000fff0ffULL,       0xff0fff000f000000ULL, 0xff0fff0000000fULL,
        0xf0000000fff0ff00ULL, 0xf0ff00ff00f00000ULL, 0xf00000fff00ffULL,
        0xff00fff00000f000ULL, 0xf00ff00ff0fULL};
    for (uint64_t m : m6_600_) {
      if ((board & m) == m)
        return -600;
    }

    // 加 500 分的掩码
    static constexpr uint64_t m6_500[] = {
        0xf0000000ff0fff00ULL, 0xfff0ff0000000fULL,   0xf0000000fff0ffULL,
        0xff0fff0000000f00ULL, 0xff00ff0f0000f000ULL, 0xf0ff00ff000000f0ULL,
        0xf000000ff00ff0fULL,  0xf0000f0ff00ffULL};
    for (uint64_t m : m6_500) {
      if ((board & m) == m)
        return -500;
    }

    // 加 1600 分的掩码
    static constexpr uint64_t m6_1600[] = {
        0xf00000fff0ffULL,     0xff0fff00000f0000ULL, 0xff00ff00000f0fULL,
        0xf0f00000ff00ff00ULL, 0xf0ff00fff0000000ULL, 0xf0f000000ff00ffULL,
        0xff00ff000000f0f0ULL, 0xfff00ff0fULL};
    for (uint64_t m : m6_1600) {
      if ((board & m) == m)
        return -1600;
    }

    // 加 2400 分的掩码
    static constexpr uint64_t m6_2400[] = {
        0xff0fff0fULL,         0xf0fff0ff00000000ULL, 0xff000000ff00ffULL,
        0xff00ff000000ff00ULL, 0xff0fff0f00000000ULL, 0xff00ff000000ffULL,
        0xff000000ff00ff00ULL, 0xf0fff0ffULL};
    for (uint64_t m : m6_2400) {
      if ((board & m) == m)
        return -2400;
    }

    // 减 600 分的掩码
    static constexpr uint64_t m6_600[] = {
        0xffff0ffULL,          0xff0ffff000000000ULL, 0xff00ff00f0000fULL,
        0xf0000f00ff00ff00ULL, 0xf0ff0fff00000000ULL, 0xf00f000ff00ffULL,
        0xff00ff000f00f000ULL, 0xfff0ff0fULL};
    for (uint64_t m : m6_600) {
      if ((board & m) == m)
        return 600;
    }
  }
  if (masked_count == 4) {
    // 减 3000 分的掩码
    static constexpr uint64_t m4_3000[] = {
        0xff00f00fULL,         0xff00f00000000fULL,   0xfff00fULL,
        0xff000f000000f000ULL, 0xf00000000f00ff00ULL, 0xf00f00ff00000000ULL,
        0xf00fff0000000000ULL, 0xf000000f000ffULL};
    for (uint64_t m : m4_3000) {
      if ((board & m) == m)
        return 3000;
    }

    // 减 800 分的掩码
    static constexpr uint64_t m4_800[] = {
        0xf000f000f00fULL,     0xf000ff00fULL,        0xfff000000000f000ULL,
        0xf00000000000fff0ULL, 0xf00f000f000f0000ULL, 0xf00ff000f0000000ULL,
        0xf000000000fffULL,    0xfff00000000000fULL};
    for (uint64_t m : m4_800) {
      if ((board & m) == m)
        return 800;
    }

    // 减 1600 分的掩码
    static constexpr uint64_t m4_1600[] = {
        0xff000000f000f000ULL,
        0xf000f000000ffULL,
        0xf000f0ffULL,
        0xff0f000f00000000ULL,
        0xf000f0000000ff00ULL,
        0xf0fff00000000000ULL,
        0xfff0fULL,
        0xff0000000f000fULL,
    };
    for (uint64_t m : m4_1600) {
      if ((board & m) == m)
        return 1600;
    }
  }
  if (masked_count == 3 || masked_count == 4) {
    // 减 3000 分的掩码
    static constexpr uint64_t m3_3000[] = {
        0xf00000000f00fULL, 0xf00f00000000000fULL, 0xf00000000000f00fULL,
        0xf00f00000000f000ULL};
    for (uint64_t m : m3_3000) {
      if ((board & m) == m)
        return 3000;
    }

    // 减 2000 分的掩码
    static constexpr uint64_t m3_2000[] = {
        0xf0000000000f00fULL,  0xf000000000f00fULL,   0xf0000000000ff000ULL,
        0xf000000f0000f000ULL, 0xf00f0000000000f0ULL, 0xf00f000000000f00ULL,
        0xf0000f000000fULL,    0xff0000000000fULL};
    for (uint64_t m : m3_2000) {
      if ((board & m) == m)
        return 2000;
    }
  }
  if (masked_count == 3) {
    // 减 1000 分的掩码
    static constexpr uint64_t m3_1000[] = {0xff00fULL,
                                           0xf000f00fULL,
                                           0xf00000000000ff00ULL,
                                           0xff0000000000f000ULL,
                                           0xf00ff00000000000ULL,
                                           0xf00f000f00000000ULL,
                                           0xff00000000000fULL,
                                           0xf0000000000ffULL};
    for (uint64_t m : m3_1000) {
      if ((board & m) == m)
        return 1000;
    }
  }

  return 0; // 没有命中任何糟糕的位置
}

int32_t AIPlayer::evaluate(uint64_t s) {
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

int32_t AIPlayer::search0(uint64_t b) {
  int32_t best = -dead_score;
  auto moves = BoardMover::s_move_board_all(b);

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

int32_t AIPlayer::process_score(uint32_t score) {
  int32_t s = static_cast<int32_t>(score);
  if (s < 200)
    return std::max(0, (s >> 2) - 10);
  if (s < 500)
    return (s >> 1) - 12;
  if (s < 1000)
    return (s >> 1) + 144;
  if (s < 2000)
    return (s + 600);
  return 3000;
}

int32_t AIPlayer::search_branch(uint64_t t, int32_t depth,
                                int32_t sum_increment, uint64_t &out_nodes) {
  out_nodes = 1; // 当前节点算作 1 个基础计算量
  // 剪枝
  if (depth < max_d - 2) {
    int32_t current_eval = evaluate(t);
    if (sum_increment > max_layer)
      return current_eval;

    bool mask_unmoved = (t & fixed_mask) == fixed_mask;
    if (!mask_unmoved) {
      return -(dead_score);
    }
    if (current_eval < initial_eval_score - threshold) {
      return -(dead_score);
    }
  }

  // 提取空位掩码并计算数量
  uint64_t x = t | (t >> 1);
  x |= (x >> 2);
  uint64_t empty_mask = (~x) & 0x1111111111111111ULL;
  int empty_slots_count = __builtin_popcountll(empty_mask);

  // 动态深度衰减
  int32_t effective_depth = depth;
  if (empty_slots_count > 5 && masked_count < 4) {
    effective_depth = std::min(effective_depth, 3);
  } else if (empty_slots_count > 4 && masked_count < 4) {
    effective_depth = std::min(effective_depth, 4);
  }

  int32_t temp = 0;
  int32_t cached_val = 0;
  uint64_t cached_nodes = 0;
  uint64_t cache_idx = cache.hash(t);

  if (!cache.lookup(cache_idx, t, effective_depth, cached_val, cached_nodes)) {
    int64_t local_temp = 0;
    uint64_t child_nodes_total = 0;

    while (empty_mask != 0) {
      int bit_pos = __builtin_ctzll(empty_mask);
      uint64_t nodes_4 = 0, nodes_2 = 0;

      // 向下递归时使用 effective_depth - 1
      uint64_t t4 = t | (2ULL << bit_pos);
      int32_t score_4 =
          search_ai_player(t4, effective_depth - 1, sum_increment + 2, nodes_4);

      uint64_t t2 = t | (1ULL << bit_pos);
      int32_t score_2 =
          search_ai_player(t2, effective_depth - 1, sum_increment + 1, nodes_2);

      child_nodes_total += (nodes_4 + nodes_2);
      int64_t temp_t = (static_cast<int64_t>(score_2) * spawn_weight_2 +
                        static_cast<int64_t>(score_4) * spawn_weight_4) >>
                       16;

      local_temp += temp_t;
      empty_mask &= (empty_mask - 1);
    }

    temp = static_cast<int32_t>(local_temp / empty_slots_count);
    out_nodes += child_nodes_total;

    cache.update(cache_idx, t, effective_depth, temp, out_nodes);

  } else {
    temp = cached_val;
    out_nodes += 1;
  }

  return (int32_t)temp;
}

int32_t AIPlayer::search_ai_player(uint64_t b, int32_t depth,
                                   int32_t sum_increment, uint64_t &out_nodes) {
  out_nodes = 1;
  if (stop_search.load(std::memory_order_relaxed))
    return 0;
  if (depth <= 0)
    return search0(b);

  int32_t best = -dead_score;

  // 顶层和次顶层启用 Task 任务池并行模式
  if (depth >= max_d - 2) {
    int32_t scores[4] = {-dead_score, -dead_score, -dead_score, -dead_score};
    uint64_t task_nodes[4] = {0, 0, 0, 0};
    auto moves = BoardMover::s_move_board_all(b);

    for (int i = 0; i < 4; ++i) {
      if (moves[i].is_valid) {
        uint64_t t = moves[i].board;
        int32_t processed_score = process_score(moves[i].score);

// 将分支封装为 Task 丢进全局任务池
#pragma omp task shared(scores, task_nodes)                                    \
    firstprivate(i, t, processed_score, depth, sum_increment)
        {
          uint64_t branch_nodes = 0;
          scores[i] = search_branch(t, depth, sum_increment, branch_nodes) +
                      processed_score + 1;
          task_nodes[i] = branch_nodes;
        }
      }
    }

#pragma omp taskwait

    // 串行汇总逻辑
    for (int i = 0; i < 4; ++i) {
      out_nodes += task_nodes[i];
      if (scores[i] > best) {
        best = scores[i];

        if (depth == max_d) {
          best_operation = i + 1;
        }
      }

      if (depth == max_d) {
        top_scores[i] = scores[i];
        node += task_nodes[i];
      }
    }
  } else {
    // 纯串行层
    auto moves = BoardMover::s_move_board_all(b);

    for (int i = 0; i < 4; ++i) {
      if (moves[i].is_valid) {
        uint64_t t = moves[i].board;
        uint32_t score = moves[i].score;

        int32_t current_depth = depth; // 防止修改外层 depth
        if (score > 250 && score < 2000) {
          int32_t min_depth = __builtin_clz(score) - 16;
          current_depth = std::min(min_depth, std::max(current_depth, 2));
        }

        uint64_t branch_nodes = 0;
        int32_t temp =
            search_branch(t, current_depth, sum_increment, branch_nodes) +
            process_score(moves[i].score);
        out_nodes += branch_nodes;

        if (temp > best)
          best = temp;
      }
    }
  }

  return best;
}

void AIPlayer::start_search(int32_t depth) {
  best_operation = 0;
  max_d = depth;
  max_layer = (uint32_t)get_max_layer(spawn_rate4, depth);

  uint64_t num_buckets;
  if (depth < 3)
    num_buckets = 4096;
  else if (depth < 4)
    num_buckets = 16384;
  else if (depth < 5)
    num_buckets = 65536;
  else if (depth < 7)
    num_buckets = 262144;
  else
    num_buckets = 524288;
  cache.reset(num_buckets);

  initial_eval_score = evaluate(board);

  node = 0;

  for (int i = 0; i < 4; ++i)
    top_scores[i] = -dead_score;

  // 不修改成员 board
  uint64_t masked_board = apply_dynamic_mask();

  int32_t current_max_threads = max_threads;
  if (depth < 5) {
    current_max_threads = 1;
  }
  omp_set_num_threads(current_max_threads);

#pragma omp parallel
  {
#pragma omp single
    {
      uint64_t root_nodes = 0;
      search_ai_player(masked_board, max_d, 0, root_nodes);
    }
  }

  int32_t best_score = top_scores[best_operation - 1];
  if (fixed_mask && ((best_operation > 0) &&
                     (best_score < std::min(-2000, -(dead_score >> 2)) ||
                      (masked_count < 5 && best_score < (480 * masked_count) &&
                       board_sum % 512 < 160)))) {
    fixed_mask = 0;
    masked_count = 0;
    threshold = std::max(threshold, 6000);
    cache.clear();
#pragma omp parallel
    {
#pragma omp single
      {
        uint64_t root_nodes = 0;
        search_ai_player(board, max_d, 0, root_nodes);
      }
    }
  }
};

uint64_t AIPlayer::apply_dynamic_mask() {
  uint8_t counts[16] = {0};
  board_sum = 0;
  fixed_mask = 0;
  uint32_t small_tiles_sum = 0;

  uint64_t current_board = board;

  // 1. 统计各个数字的个数与盘面总和
  uint64_t temp = current_board;
  for (int i = 0; i < 16; ++i) {
    uint8_t tile = temp & 0xF;
    counts[tile]++;
    if (tile > 0) {
      board_sum += (1 << tile);
      if (tile < 9)
        small_tiles_sum += (1 << tile);
    }
    temp >>= 4;
  }

  // 2. 统计大数特征并更新 dead_score
  int count_gt_128 = 0;        // 记录 > 128 (即 >= 256，索引 >= 8) 的数字个数
  int count_gt_256 = 0;        // 记录 > 256 (即 >= 512，索引 >= 9) 的数字个数
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
  if (large_tiles > 4)
    threshold = (prune == 0) ? 5600 : 2400;
  if (large_tiles <= 4)
    threshold = (prune == 0) ? 8400 : 3200;

  if (count_gt_128 <= 4) {
    dead_score = 262144;
  } else if (board_sum % 1024 < 8 || board_sum % 1024 > 1008) {
    dead_score = 131072;
  } else if (count_gt_128 == 5) {
    dead_score = 131072;
  } else if (count_gt_256 > 5 && distinct_gt_256) {
    if (count_gt_128 == 8 && (counts[7] == 1 || counts[6] >= 1)) {
      dead_score = 4096;
    } else if (count_gt_256 >= 6 && 960 < board_sum % 1024) {
      dead_score = 32768;
    } else {
      dead_score = 48000;
    }
  } else {
    dead_score = 65536;
  }

  // 如果 > 256 的数字不互不相同，直接返回未掩码的 current_board
  if (!distinct_gt_256)
    return current_board;

  uint32_t rem = board_sum % 1024;

  // 4. mask
  // 条件：(余数在40~512之间 且 不存在512) 或 (余数在512~1000之间)
  bool condA_part1 = ((rem >= 48 || (rem > 12 && small_tiles_sum == rem)) &&
                      rem <= 512 && counts[9] == 0);
  bool condA_part2 = (rem >= 512 || rem < 6);

  if (condA_part1 || condA_part2) {
    if (rem > 1000) {
      current_board = mask(current_board, 11);
    } else {
      current_board = mask(current_board, 9);
    }
  } else {
    current_board = mask(current_board, 12);
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
  bool flag1 = count_gt_512 == 5 && (counts[9] == 1 || counts[8] > 0) &&
               board_sum % 512 < 72; // L3-512 628
  bool flag2 = ((count_gt_512 == 6 || (count_gt_512 == 5 && counts[9] == 1)) &&
                board_sum % 1024 < 72) || // L3-1k 63L
               (count_gt_512 == 5 && counts[9] == 0 &&
                (counts[8] == 1 || counts[7] > 0) &&
                board_sum % 256 < 72); // L3-512 之前的 63L 256
  bool flag3 =
      count_gt_512 >= 4 && masked_count >= 5 &&
      (board_sum % 256 > 132 && board_sum % 256 < 234); // 处理 63L 128 滑出
  bool flag4 = count_gt_512 == 4 && counts[9] == 1 && counts[10] == 1 &&
               board_sum % 256 < 60; // 其余 L3 256
  update_specific_scores(current_board, flag1, flag2, flag3, flag4);

  // 7. 计算固定大数掩码
  if (masked_count >= 5 && counts[0] >= 3) {
    fixed_mask = get_22_mask(current_board) | get_21_mask(current_board);
  } else if ((min_masked_tile == 12 && masked_count == 4) ||
             (min_masked_tile == 11 && masked_count == 5) ||
             (min_masked_tile == 10 && masked_count == 6)) {
    fixed_mask = get_23_mask(current_board) | get_22_mask(current_board);
  } else if (masked_count >= 5 &&
             (large_tiles > 5 ||
              (board_sum % 256 > 48 && board_sum % 256 < 224) ||
              is_5tiler(board_sum, counts))) {
    fixed_mask = get_23_mask(current_board) | get_22_mask(current_board);
  } else if (do_check ||
             (prune == 0 && (large_tiles > 2 || large_tiles == 0)) ||
             ((board_sum % 256 > 246 || board_sum % 256 < 24) &&
              ((large_tiles < 5 && large_tiles > 2) || large_tiles == 0)) ||
             counts[7] > 1 || (counts[6] > 1 && counts[7] == 1)) {
    fixed_mask = 0;
  } else {
    if (masked_count >= 4) {
      fixed_mask = get_22_mask(current_board) | get_21_mask(current_board);
    } else if (masked_count == 3 && count_gt_512 == 3 &&
               board_sum % 1024 > 24) {
      fixed_mask = get_21_mask(current_board) | get_20_mask(current_board);
    } else if (masked_count >= 2 && count_gt_512 == 2 && board_sum % 512 > 16) {
      fixed_mask = get_20_mask(current_board);
    }
  }

  return current_board;
};

// ------------------------------------------------------------------
// EvilGen 类
// ------------------------------------------------------------------
EvilGen::EvilGen(uint64_t initial_board)
    : cache(), max_d(3), hardest_pos(0), hardest_num(1), board(initial_board),
      node(0), dead_score(131072) {}

void EvilGen::reset_board(uint64_t new_board) {
  hardest_pos = 0;
  hardest_num = 1;
  board = new_board;
  node = 0;
}

int32_t EvilGen::evaluate(uint64_t s) {
  node++;
  uint64_t s_reverse =
      reverse_board(s); // 调用 CommonMover.h 中的全局 inline 函数
  int32_t sum_x1 = 0, sum_x2 = 0, sum_y1 = 0, sum_y2 = 0;

  for (int i = 0; i < 4; ++i) {
    uint16_t l1 = static_cast<uint16_t>(s >> (16 * i));
    uint16_t l2 = static_cast<uint16_t>(s_reverse >> (16 * i));

    ScorePair px = _diffs_merged[l1];
    sum_x1 += px.d1;
    sum_x2 += px.d2;

    ScorePair py = _diffs_merged[l2];
    sum_y1 += py.d1;
    sum_y2 += py.d2;
  }

  return std::max(sum_x1, sum_x2) + std::max(sum_y1, sum_y2);
}

int32_t EvilGen::search_evil_gen(uint64_t b, int32_t depth,
                                 uint64_t &out_nodes) {
  int32_t evil = static_cast<int32_t>(dead_score);
  out_nodes = 1;

  // 提取空位掩码
  uint64_t x = b | (b >> 1);
  x |= (x >> 2);
  uint64_t empty_mask = (~x) & 0x1111111111111111ULL;

  while (empty_mask != 0) {
    int bit_pos = __builtin_ctzll(empty_mask);
    int pos_idx = bit_pos >> 2;

    for (uint64_t num : {2ULL, 1ULL}) {
      uint64_t t = b | (num << bit_pos);
      int32_t best = -static_cast<int32_t>(dead_score);

      // 调用重构后的 BoardMover 模板引擎
      auto moves = BoardMover::s_move_board_all(t);

      for (int d = 0; d < 4; ++d) {
        if (moves[d].is_valid) {
          uint64_t t1 = moves[d].board;
          uint32_t score = moves[d].score;

          int32_t temp = 0;
          uint64_t child_nodes = 0;
          uint64_t cache_idx = cache.hash(t1);

          if (!cache.lookup(cache_idx, t1, depth, temp, child_nodes)) {
            if (depth > 1) {
              temp = search_evil_gen(t1, depth - 1, child_nodes);
            } else {
              temp = evaluate(t1);
              child_nodes = 1;
            }
            cache.update(cache_idx, t1, depth, temp, child_nodes);
          }

          out_nodes += child_nodes;

          // 修正收窄转换问题
          int32_t current_val = temp + static_cast<int32_t>(score << 1);
          best = std::max(best, current_val);

          if (best >= evil)
            break;
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

int32_t EvilGen::dispatcher(uint64_t current_board) {
  node = 0;
  uint64_t total_nodes = 0;
  return search_evil_gen(current_board, max_d, total_nodes);
}

void EvilGen::start_search(int32_t depth) {
  cache.clear();
  max_d = depth;
  dispatcher(board);
}

std::tuple<uint64_t, uint8_t, uint8_t> EvilGen::gen_new_num(int32_t depth) {
  start_search(depth);
  uint64_t new_board =
      board | (static_cast<uint64_t>(hardest_num) << (4 * hardest_pos));
  return {new_board, static_cast<uint8_t>(15 - hardest_pos), hardest_num};
}

extern std::array<float, 4> find_best_egtb_move(uint64_t target_board,
                                                int table_type);

// int benchmark() {
//     uint64_t initial_board = 0x11212451ea86fc97ULL;
//     AIPlayer cpp_ai(initial_board);

//     cpp_ai.max_threads = 4;
//     cpp_ai.do_check = 0;
//     cpp_ai.prune = 0;

//     int32_t test_depth = 12;

//     std::cout << "[*] Starting search..." << std::endl;
//     std::cout << "[*] Board: 0x" << std::hex << initial_board << std::dec <<
//     std::endl; std::cout << "[*] Depth: " << test_depth << std::endl;
//     std::cout << "[*] Threads: " << cpp_ai.max_threads << "\n" << std::endl;

//     auto start_time = std::chrono::high_resolution_clock::now();

//     cpp_ai.start_search(test_depth);

//     auto end_time = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> diff = end_time - start_time;

//     std::cout << "-----------------------------------" << std::endl;
//     std::cout << "Search completed in : " << diff.count() << " seconds." <<
//     std::endl; std::cout << "Best operation      : " <<
//     (int)cpp_ai.best_operation << std::endl; std::cout << "Scores : " <<
//     cpp_ai.top_scores[0] << ", " << cpp_ai.top_scores[1] << ", " <<
//     cpp_ai.top_scores[2] << ", " << cpp_ai.top_scores[3] << std::endl;
//     std::cout << "-----------------------------------" << std::endl;

//     return 0;
// }