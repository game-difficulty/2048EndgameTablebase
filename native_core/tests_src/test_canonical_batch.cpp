#include "Calculator.h"
#include "CanonicalBatch.h"
#include "FormationRuntime.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

uint64_t scalar_by_mode(uint64_t board, int symm_mode) {
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
        case SymmMode::Min34Top:
            return Calculator::canonical_min34_top(board);
        case SymmMode::Identity:
        default:
            return Calculator::canonical_identity(board);
    }
}

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

std::vector<uint64_t> make_boards(size_t count, uint64_t seed) {
    std::vector<uint64_t> boards;
    boards.reserve(count);
    uint64_t state = seed;
    for (size_t i = 0; i < count; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        boards.push_back(state);
    }
    return boards;
}

void verify_batch(const std::vector<uint64_t> &boards, int mode) {
    std::vector<uint64_t> expected(boards.size(), 0ULL);
    for (size_t i = 0; i < boards.size(); ++i) {
        expected[i] = scalar_by_mode(boards[i], mode);
    }

    std::vector<uint64_t> padded_src(boards.size() + 3U, 0xaaaaaaaaaaaaaaaaULL);
    std::copy(boards.begin(), boards.end(), padded_src.begin() + 1U);
    std::vector<uint64_t> padded_dst(boards.size() + 7U, 0x5555555555555555ULL);
    CanonicalBatch::canonicalize_by_mode(
        padded_src.data() + 1U,
        padded_dst.data() + 3U,
        boards.size(),
        mode);
    for (size_t i = 0; i < boards.size(); ++i) {
        if (padded_dst[i + 3U] != expected[i]) {
            fail("canonical out-of-place mismatch mode=" + std::to_string(mode) +
                 " index=" + std::to_string(i));
        }
    }
    if (padded_dst[0] != 0x5555555555555555ULL ||
        padded_dst[2] != 0x5555555555555555ULL ||
        padded_dst[boards.size() + 3U] != 0x5555555555555555ULL) {
        fail("canonical out-of-place wrote outside requested range");
    }

    std::vector<uint64_t> inplace = boards;
    CanonicalBatch::canonicalize_inplace(inplace.data(), inplace.size(), mode);
    if (inplace != expected) {
        fail("canonical inplace mismatch mode=" + std::to_string(mode));
    }
}

} // namespace

int main() {
    try {
        const std::vector<uint64_t> fixed = {
            0ULL,
            0x0123456789abcdefULL,
            0xfedcba9876543210ULL,
            0x1111222233334444ULL,
            0x000f00f000ff0fffULL,
            0xffff000000000000ULL,
        };
        const int modes[] = {
            static_cast<int>(SymmMode::Identity),
            static_cast<int>(SymmMode::Full),
            static_cast<int>(SymmMode::Diagonal),
            static_cast<int>(SymmMode::Horizontal),
            static_cast<int>(SymmMode::Min33),
            static_cast<int>(SymmMode::Min24),
            static_cast<int>(SymmMode::Min34),
            static_cast<int>(SymmMode::Min34Top),
        };
        const size_t counts[] = {
            0U, 1U, 2U, 3U, 4U, 5U, 7U, 8U, 9U, 15U, 16U,
            31U, 32U, 63U, 64U, 255U, 256U, 4099U, 8192U,
        };
        for (const int mode : modes) {
            verify_batch(fixed, mode);
            for (const size_t count : counts) {
                verify_batch(make_boards(count, 0x9e3779b97f4a7c15ULL + count), mode);
            }
        }

        const std::vector<uint64_t> stress_boards =
            make_boards(16384U, 0xd1b54a32d192ed03ULL);
        const unsigned thread_count = std::max(2U, std::min(16U, std::thread::hardware_concurrency()));
        std::vector<std::thread> threads;
        threads.reserve(thread_count);
        for (unsigned t = 0; t < thread_count; ++t) {
            threads.emplace_back([&, t]() {
                for (uint32_t iter = 0; iter < 64U; ++iter) {
                    const int mode = modes[(iter + t) % (sizeof(modes) / sizeof(modes[0]))];
                    verify_batch(stress_boards, mode);
                }
            });
        }
        for (std::thread &thread : threads) {
            thread.join();
        }
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    std::cout << "canonical_batch_backend=" << CanonicalBatch::backend_name() << "\n";
    std::cout << "canonical_batch_ok=1\n";
    return 0;
}
