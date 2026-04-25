#pragma once

#include <cstdint>
#include <utility>

#include "Calculator.h"
#include "FormationRuntime.h"

inline uint64_t apply_sym_like(uint64_t board, int symm_index) {
    switch (symm_index) {
        case 1:
            return Calculator::ReverseLR(board);
        case 2:
            return Calculator::ReverseUD(board);
        case 3:
            return Calculator::ReverseUL(board);
        case 4:
            return Calculator::ReverseUR(board);
        case 5:
            return Calculator::Rotate180(board);
        case 6:
            return Calculator::RotateL(board);
        case 7:
            return Calculator::RotateR(board);
        case 0:
        default:
            return board;
    }
}

inline uint64_t canonical_by_mode(uint64_t board, int symm_mode) {
    switch (static_cast<SymmMode>(symm_mode)) {
        case SymmMode::Full:
            return Calculator::canonical_full(board);
        case SymmMode::Diagonal:
            return Calculator::canonical_diagonal(board);
        case SymmMode::Horizontal:
            return Calculator::canonical_horizontal(board);
        case SymmMode::Min33:
            return Calculator::canonical_min33(board);
        case SymmMode::Min24:
            return Calculator::canonical_min24(board);
        case SymmMode::Min34:
            return Calculator::canonical_min34(board);
        case SymmMode::Identity:
        default:
            return Calculator::canonical_identity(board);
    }
}

inline std::pair<uint64_t, int> canonical_pair_by_mode(uint64_t board, int symm_mode) {
    switch (static_cast<SymmMode>(symm_mode)) {
        case SymmMode::Full:
            return Calculator::canonical_full_pair(board);
        case SymmMode::Diagonal:
            return Calculator::canonical_diagonal_pair(board);
        case SymmMode::Horizontal:
            return Calculator::canonical_horizontal_pair(board);
        case SymmMode::Identity:
        case SymmMode::Min33:
        case SymmMode::Min24:
        case SymmMode::Min34:
        default:
            return Calculator::canonical_identity_pair(board);
    }
}

inline std::pair<uint64_t, int> apply_sym_pair(uint64_t board, int symm_mode) {
    return canonical_pair_by_mode(board, symm_mode);
}
