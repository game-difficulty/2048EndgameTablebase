#include "BoardMover.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
struct Result {
  double success = 0, death = 0, steps = 0, upper = 1;
  bool complete = true;
};
struct Entry {
  uint64_t board = 0;
  Result value;
  int depth = -1;
};
struct Options {
  uint64_t board = 0x330043598671da10ULL, nodes = 20000000;
  int depth = 6, target = 2048, cache_mib = 32;
  double seconds = 10, p4 = .1;
  bool reverse = false;
};

int exponent(int target) {
  if (target < 8 || target > 32768 || (target & (target - 1)))
    throw std::runtime_error("target must be a power of two from 8 to 32768");
  int n = 0;
  while (target >>= 1) ++n;
  return n;
}
bool contains(uint64_t board, int tile) {
  for (int i = 0; i < 16; ++i)
    if (((board >> (i * 4)) & 15) == unsigned(tile)) return true;
  return false;
}
bool better(const Result& a, const Result& b) {
  // Exact horizon-success objective; death and successful path length break ties.
  if (a.success != b.success) return a.success > b.success;
  if (a.death != b.death) return a.death < b.death;
  return a.steps < b.steps;
}

class Search {
  const Options& o;
  int target;
  std::vector<Entry> cache;
  Clock::time_point deadline;
  size_t slot(uint64_t b, int d) const {
    b ^= uint64_t(d) * 0x9e3779b97f4a7c15ULL;
    b = (b ^ (b >> 30)) * 0xbf58476d1ce4e5b9ULL;
    b = (b ^ (b >> 27)) * 0x94d049bb133111ebULL;
    return (b ^ (b >> 31)) % cache.size();
  }
  bool exhausted() {
    if (nodes >= o.nodes || ((nodes & 1023) == 0 && Clock::now() >= deadline))
      budget_exhausted = true;
    return budget_exhausted;
  }
public:
  uint64_t nodes = 0, hits = 0, goals = 0, deaths = 0, horizons = 0;
  bool budget_exhausted = false;
  explicit Search(const Options& options) : o(options), target(exponent(o.target)),
    cache(size_t(o.cache_mib) * 1024 * 1024 / sizeof(Entry)) {}
  size_t cache_bytes() const { return cache.size() * sizeof(Entry); }
  void begin() {
    deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(o.seconds));
  }
  Result after_move(uint64_t board, int depth) {
    // Goal is the merge itself, not table handoff or survival after that goal.
    if (contains(board, target)) { ++goals; return {1, 0, 1, 1, true}; }
    int empty[16], count = 0;
    for (int i = 0; i < 16; ++i)
      if (((board >> (4 * i)) & 15) == 0) empty[count++] = i;
    if (!count) throw std::runtime_error("valid move must leave an empty cell");
    Result total{0, 0, 0, 0, true};
    for (int j = 0; j < count; ++j) {
      for (int tile = 1; tile <= 2; ++tile) {
        double weight = (tile == 1 ? 1 - o.p4 : o.p4) / count;
        if (weight == 0) continue;
        Result r = player(board | (uint64_t(tile) << (4 * empty[j])), depth - 1);
        total.success += weight * r.success;
        total.death += weight * r.death;
        total.steps += weight * (r.steps + r.success);
        total.upper += weight * r.upper;
        total.complete = total.complete && r.complete;
      }
    }
    return total;
  }
  Result player(uint64_t board, int depth) {
    if (exhausted()) return {0, 0, 0, 1, false};
    ++nodes;
    size_t index = cache.empty() ? 0 : slot(board, depth);
    if (!cache.empty() && cache[index].depth == depth && cache[index].board == board) {
      ++hits;
      return cache[index].value;
    }
    auto moves = BoardMover::s_move_board_all(board);
    bool legal = false;
    for (auto& move : moves) legal = legal || move.is_valid;
    Result best;
    if (!legal) { ++deaths; best = {0, 1, 0, 0, true}; }
    else if (depth == 0) { ++horizons; }
    else {
      best.success = -1;
      double upper = 0;
      bool complete = true;
      // Root order can be reversed, but internal ordering remains fixed.
      for (const auto& move : moves) {
        if (!move.is_valid) continue;
        Result r = after_move(move.board, depth);
        upper = std::max(upper, r.upper);
        complete = complete && r.complete;
        if (better(r, best)) best = r;
      }
      best.upper = upper;
      best.complete = complete;
    }
    // Never reuse a budget-truncated value as a complete horizon result.
    if (!cache.empty() && best.complete) cache[index] = {board, best, depth};
    return best;
  }
};

int main(int argc, char** argv) {
  try {
    Options o;
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (key == "--reverse") { o.reverse = true; continue; }
      if (++i == argc) throw std::runtime_error("missing option value");
      std::string value = argv[i];
      if (key == "--board") {
        if (value.size() != 16 || value.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos)
          throw std::runtime_error("board must contain exactly 16 hex digits");
        o.board = std::stoull(value, nullptr, 16);
      } else if (key == "--target") o.target = std::stoi(value);
      else if (key == "--depth") o.depth = std::stoi(value);
      else if (key == "--nodes") o.nodes = std::stoull(value);
      else if (key == "--seconds") o.seconds = std::stod(value);
      else if (key == "--cache-mib") o.cache_mib = std::stoi(value);
      else if (key == "--p4") o.p4 = std::stod(value);
      else throw std::runtime_error("unknown option: " + key);
    }
    int tile = exponent(o.target);
    if (contains(o.board, tile)) throw std::runtime_error("prototype requires target absent from initial board");
    if (o.depth < 1 || o.depth > 32 || o.cache_mib < 0 || o.cache_mib > 512 ||
        !std::isfinite(o.seconds) || o.seconds <= 0 || !std::isfinite(o.p4) || o.p4 < 0 || o.p4 > 1)
      throw std::runtime_error("invalid depth, budget, cache size or spawn probability");
    BoardMover::init_tables();
    const char* names[] = {"left", "right", "up", "down"};
    auto moves = BoardMover::s_move_board_all(o.board);
    Result results[4];
    bool first = true;
    int best = -1;
    std::cout << std::setprecision(17) << "{\"target\":" << o.target << ",\"depth\":" << o.depth << ",\"directions\":[";
    for (int j = 0; j < 4; ++j) {
      int i = o.reverse ? 3 - j : j;
      if (!moves[i].is_valid) continue;
      Search search(o);
      search.begin();
      auto start = Clock::now();
      Result r = search.after_move(moves[i].board, o.depth);
      double seconds = std::chrono::duration<double>(Clock::now() - start).count();
      results[i] = r;
      if (best < 0 || better(r, results[best]) || (!better(results[best], r) && i < best)) best = i;
      if (!first) std::cout << ',';
      first = false;
      std::cout << "{\"direction\":\"" << names[i] << "\",\"success_lower\":" << r.success
        << ",\"success_upper\":" << r.upper << ",\"policy_death\":" << r.death
        << ",\"policy_unresolved\":" << std::max(0.0, 1 - r.success - r.death)
        << ",\"success_steps\":" << (r.success > 0 ? std::to_string(r.steps / r.success) : "null")
        << ",\"complete\":" << (r.complete ? "true" : "false")
        << ",\"seconds\":" << seconds << ",\"nodes\":" << search.nodes << ",\"hits\":" << search.hits
        << ",\"goal_leaves\":" << search.goals << ",\"dead_leaves\":" << search.deaths
        << ",\"horizon_leaves\":" << search.horizons << ",\"cache_bytes\":" << search.cache_bytes() << '}';
      std::cout.flush();
    }
    bool certified = best >= 0;
    for (int i = 0; i < 4; ++i)
      if (i != best && moves[i].is_valid && results[best].success <= results[i].upper + 1e-12) certified = false;
    std::cout << "],\"best_observed\":" << (best < 0 ? "null" : std::string("\"") + names[best] + "\"")
      << ",\"dominance_certified\":" << (certified ? "true" : "false") << "}\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
