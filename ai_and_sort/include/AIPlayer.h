#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <tuple>

// ------------------------------------------------------------------
// 评分与评估相关数据结构
// ------------------------------------------------------------------
struct alignas(4) ScorePair {
  int16_t d1;
  int16_t d2;
};

// 声明全局评分表 (实现在 AIPlayer.cpp)
extern ScorePair _diffs_merged[65536];

// ------------------------------------------------------------------
// 基础位运算工具函数
// ------------------------------------------------------------------
inline uint64_t mask(uint64_t board, int threshold);
inline uint64_t extract_0xf_nibbles(uint64_t v);
uint64_t resolve_32768_doubles(uint64_t board);

// ------------------------------------------------------------------
// Cache 类：用于存储搜索中间结果
// ------------------------------------------------------------------
class Cache {
private:
  struct alignas(64) CacheBucket {
    uint64_t entries[8];
  };

  static constexpr uint64_t MAX_BUCKETS = 524288;
  uint64_t current_num_buckets = 524288;
  uint64_t current_mask = current_num_buckets - 1;
  std::unique_ptr<CacheBucket[]> table;

  std::atomic<bool> dummy_false{false};
  int32_t dummy_dead_score = 131072;

  inline uint32_t get_signature(uint64_t board) const {
    return static_cast<uint32_t>(((board ^ (board >> 31)) * 0x1a7daf1bULL) +
                                 board) >>
           21;
  }

public:
  std::atomic<bool> *abort_ptr = &dummy_false;
  int32_t *dead_score_ptr = &dummy_dead_score;

  Cache();
  void reset(uint64_t new_num_buckets);
  void clear();

  inline uint64_t hash(uint64_t board) const {
    uint64_t hashed = (((board ^ (board >> 27)) * 0x1A85EC53ULL) + board) >> 23;
    return hashed & current_mask;
  }

  bool lookup(uint64_t bucket_idx, uint64_t board, int32_t depth,
              int32_t &out_score, uint64_t &out_subtree_nodes);
  void update(uint64_t bucket_idx, uint64_t board, int32_t depth, int32_t score,
              uint64_t subtree_nodes);
};

// ------------------------------------------------------------------
// AIPlayer 类：核心搜索与评估引擎
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

  uint8_t prune;
  int32_t initial_eval_score;
  int32_t threshold;
  int32_t top_scores[4];
  int32_t dead_score;

  AIPlayer(uint64_t initial_board);

  void clear_cache();
  void reset_board(uint64_t new_board);
  void update_spawn_rate(double new_rate);
  int32_t evaluate(uint64_t s);
  void start_search(int32_t depth = 3);

  // 内部搜索函数
  int32_t search0(uint64_t b);
  int32_t process_score(uint32_t score);
  int32_t check_corner(uint64_t board) const;
  int32_t search_branch(uint64_t t, int32_t depth, int32_t sum_increment,
                        uint64_t &out_nodes);
  int32_t search_ai_player(uint64_t b, int32_t depth, int32_t sum_increment,
                           uint64_t &out_nodes);
  uint64_t apply_dynamic_mask();
};

// ------------------------------------------------------------------
// EvilGen 类：生成“最难”方块的 AI
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

  EvilGen(uint64_t initial_board);
  void reset_board(uint64_t new_board);
  int32_t evaluate(uint64_t s);
  int32_t search_evil_gen(uint64_t b, int32_t depth, uint64_t &out_nodes);
  int32_t dispatcher(uint64_t current_board);
  void start_search(int32_t depth = 4);
  std::tuple<uint64_t, uint8_t, uint8_t> gen_new_num(int32_t depth = 4);
};

// ------------------------------------------------------------------
// 其他全局函数声明
// ------------------------------------------------------------------
std::array<float, 4> find_best_egtb_move(uint64_t target_board, int table_type);
