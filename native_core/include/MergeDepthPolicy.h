#pragma once

#include <algorithm>
#include <cstdint>

// A search-wide goal; completion depends on the board, not the path taken.
struct MergeDepthPolicy {
  uint8_t target = 0;
  uint32_t initial_mass = 0;

  static uint8_t find_target(uint64_t board) {
    uint8_t counts[16] = {};
    for (int i = 0; i < 16; ++i, board >>= 4)
      ++counts[board & 15];
    uint8_t goal = 0;
    // Protect complete chains to 256+; do not assume any future spawn.
    for (int level = 1; level < 15; ++level) {
      if (counts[level] >= 2) {
        counts[level + 1] += counts[level] / 2;
        goal = static_cast<uint8_t>(level + 1);
      }
    }
    return goal >= 8 ? goal : 0;
  }

  uint32_t mass(uint64_t board) const {
    uint32_t result = 0;
    for (int i = 0; i < 16; ++i, board >>= 4) {
      const uint8_t tile = board & 15;
      if (tile >= target)
        result += uint32_t{1} << tile;
    }
    return result;
  }

  void reset(uint64_t board, bool enabled) {
    target = enabled ? find_target(board) : 0;
    rebase(board);
  }

  void rebase(uint64_t board) {
    initial_mass = target ? mass(board) : 0;
  }

  bool pending(uint64_t board) const {
    return target && mass(board) <= initial_mass;
  }

  int mask_threshold(int threshold) const {
    return std::max(threshold, static_cast<int>(target));
  }
};
