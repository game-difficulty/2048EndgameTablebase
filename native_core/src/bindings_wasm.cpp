#include <emscripten/bind.h>
#include "AIPlayer.h"
#include "CommonMover.h"
#include "BoardMover.h"
#include <thread>
#include <chrono>
#include <atomic>

using namespace emscripten;

// Helper array wrapper for returned scores
val get_scores(const AIPlayer& self) {
    val arr = val::array();
    arr.set(0, self.top_scores[0]);
    arr.set(1, self.top_scores[1]);
    arr.set(2, self.top_scores[2]);
    arr.set(3, self.top_scores[3]);
    return arr;
}

// Wrapper for stop_search
bool get_stop_search(const AIPlayer& self) {
    return self.stop_search.load(std::memory_order_relaxed);
}

void set_stop_search(AIPlayer& self, bool value) {
    self.stop_search.store(value, std::memory_order_relaxed);
}

// Wrapper for node
uint64_t get_node(const AIPlayer& self) {
    return self.node.load();
}

// Wrapper for find_best_egtb_move
val wrap_find_best_egtb_move(uint64_t target_board, int table_type) {
    auto res = find_best_egtb_move(target_board, table_type);
    val arr = val::array();
    arr.set(0, res[0]);
    arr.set(1, res[1]);
    arr.set(2, res[2]);
    arr.set(3, res[3]);
    return arr;
}

// 为 JS 封装的 move_board 包装器
uint64_t wrap_move_board(uint64_t board, int direction) {
    return BoardMover::move_board(board, direction);
}

void start_search_with_timeout(AIPlayer& self, int depth, int timeout_ms) {
    self.start_search(depth, timeout_ms);
}

EMSCRIPTEN_BINDINGS(ai_core) {
    // Basic functions
    function("find_best_egtb_move", &wrap_find_best_egtb_move);
    function("resolve_32768_doubles", &resolve_32768_doubles);

    function("move_board", &wrap_move_board);

    // AIPlayer class
    class_<AIPlayer>("AIPlayer")
        .constructor<uint64_t>()
        .function("reset_board", &AIPlayer::reset_board)
        .function("clear_cache", &AIPlayer::clear_cache)
        .function("update_spawn_rate", &AIPlayer::update_spawn_rate)
        .function("evaluate", &AIPlayer::evaluate)
        .function("start_search", &start_search_with_timeout)
        
        .property("max_d", &AIPlayer::max_d)
        .property("best_operation", &AIPlayer::best_operation)
        .property("board", &AIPlayer::board)
        .property("max_threads", &AIPlayer::max_threads)
        .property("do_check", &AIPlayer::do_check)
        .property("spawn_rate4", &AIPlayer::spawn_rate4)
        .property("prune", &AIPlayer::prune)

        .property("masked_count", &AIPlayer::masked_count)
        .property("fixed_mask", &AIPlayer::fixed_mask)
        .property("dead_score", &AIPlayer::dead_score)

        // For properties with read-only lambda and custom setter
        .property("scores", &get_scores)
        .property("node", &get_node)
        .property("stop_search", &get_stop_search, &set_stop_search);
}
