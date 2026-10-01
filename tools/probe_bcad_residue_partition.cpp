// Standalone research probe; no production algorithm or file-format changes.
// g++ -std=c++17 -O2 -I native_core/include tools/probe_bcad_residue_partition.cpp
//     native_core/src/BoardMoverAD.cpp -o tmp/probe_bcad_residue_partition.exe
#include "BoardMoverAD.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>

static void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
static unsigned value(unsigned rank) { return rank ? 1U << rank : 0U; }
static unsigned residue(uint64_t board) {
    unsigned sum = 0;
    for (unsigned i = 0; i < 16; ++i) sum += value((board >> (4*i)) & 15);
    return (sum / 2) & 31;
}
static unsigned fold(unsigned r, unsigned total) {
    return std::min(r & 31, (total - r) & 31);
}
static std::array<unsigned, 2> families(uint64_t board, unsigned total) {
    unsigned top = 0, left = 0;
    for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) {
        const unsigned v = value((board >> (4*(4*y+x))) & 15);
        if (y < 2) top += v;
        if (x < 2) left += v;
    }
    return {fold(top / 2, total), fold(left / 2, total)};
}
static uint64_t transform(uint64_t board, unsigned code) {
    uint64_t out = 0;
    for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) {
        unsigned xx = x, yy = y;
        if (code & 1) std::swap(xx, yy);
        if (code & 2) xx = 3 - xx;
        if (code & 4) yy = 3 - yy;
        out |= ((board >> (4*(4*y+x))) & 15ULL) << (4*(4*yy+xx));
    }
    return out;
}
int main() {
    uint64_t row_checks = 0, replacement_checks = 0, algebra_checks = 0, board_checks = 0;
    std::map<std::pair<unsigned,unsigned>, unsigned> raw_groups, residue_groups;
    for (unsigned row = 0; row < 65536; ++row) {
        unsigned raw_sum = 0, empty_mask = 0;
        for (unsigned p = 0; p < 4; ++p) {
            const unsigned rank = (row >> (4*p)) & 15;
            raw_sum += value(rank);
            if (!rank) empty_mask |= 1U << p;
            if (rank >= 6) {
                const uint64_t masked = row | (15ULL << (4*p));
                require(residue(masked) == residue(row), "large-tile replacement changed residue");
                ++replacement_checks;
            }
        }
        ++raw_groups[{raw_sum, empty_mask}];
        ++residue_groups[{(raw_sum / 2) & 31, empty_mask}];
        for (bool right : {false, true}) {
            bool mask_new = false;
            const auto moved = FormationAD::merge_line(static_cast<uint16_t>(row), right, mask_new);
            require(residue(moved) == residue(row), "AD row move changed residue");
            ++row_checks;
        }
    }
    unsigned min_families = 32, max_families = 0;
    for (unsigned t = 0; t < 32; ++t) {
        std::array<bool,32> used{};
        for (unsigned r = 0; r < 32; ++r) {
            const unsigned f = fold(r,t);
            used[f] = true;
            for (unsigned d : {1U,2U}) {
                const unsigned next_t = (t+d)&31;
                const std::array<unsigned,2> targets{fold(f,next_t),fold(f+d,next_t)};
                // Either representative side and either spawn side.
                for (unsigned side : {r,(t-r)&31}) for (unsigned increment : {0U,d}) {
                    const unsigned actual = fold(side+increment,next_t);
                    require(actual == targets[0] || actual == targets[1], "fanout2 algebra failed");
                    ++algebra_checks;
                }
            }
        }
        const unsigned count = std::count(used.begin(),used.end(),true);
        min_families = std::min(min_families,count);
        max_families = std::max(max_families,count);
    }
    std::mt19937_64 rng(20260923);
    for (unsigned trial = 0; trial < 20000; ++trial) {
        uint64_t board = 0;
        for (unsigned p = 0; p < 16; ++p) board |= (rng() & 15ULL) << (4*p);
        const unsigned p = rng() & 15;
        board &= ~(15ULL << (4*p));
        const unsigned t = residue(board);
        const auto source = families(board,t);
        for (unsigned rank : {1U,2U}) {
            const unsigned d = rank == 1 ? 1 : 2;
            const unsigned next_t = (t+d)&31;
            const uint64_t spawned = board | (static_cast<uint64_t>(rank) << (4*p));
            const auto moves = FormationAD::m_move_all_dir(spawned);
            for (unsigned dir = 0; dir < 4; ++dir) {
                const unsigned f = source[dir < 2 ? 0 : 1];
                const unsigned a = fold(f,next_t), b = fold(f+d,next_t);
                for (unsigned sym = 0; sym < 8; ++sym) {
                    const uint64_t moved = transform(moves[dir].board,sym);
                    require(residue(moved) == next_t, "board total residue failed");
                    const auto target = families(moved,next_t);
                    require(target[0] == a || target[0] == b || target[1] == a || target[1] == b,
                            "future family cross coverage failed");
                    ++board_checks;
                }
            }
        }
    }
    unsigned raw_max = 0, residue_max = 0;
    for (const auto &entry : raw_groups) raw_max = std::max(raw_max,entry.second);
    for (const auto &entry : residue_groups) residue_max = std::max(residue_max,entry.second);
    require(raw_max <= 36, "raw bucket group bound failed");
    require(residue_max > 36, "expected residue-only bucket overflow absent");
    // Same small tiles / masked positions; exchange the hidden 64 and 128.
    const unsigned old_a = (std::min(64U+2,128U+4)/2)&31;
    const unsigned old_b = (std::min(128U+2,64U+4)/2)&31;
    require(old_a != old_b, "expected old normalized partition counterexample absent");
    require(fold(33,3) == fold(65,3), "new family should ignore hidden permutation");
    std::cout << "row_move_checks=" << row_checks
              << "\nlarge_tile_replacement_checks=" << replacement_checks
              << "\nfanout_algebra_checks=" << algebra_checks
              << "\nrandom_board_direction_symmetry_checks=" << board_checks
              << "\nfamily_count_range=" << min_families << ".." << max_families
              << "\nraw_sum_empty_mask_max_group=" << raw_max
              << "\nresidue_empty_mask_max_group=" << residue_max
              << "\nold_partition_hidden_permutation_counterexample=" << old_a << "," << old_b
              << "\nPASS (partition probe only; no complete AD solver or GPU validation)\n";
}
