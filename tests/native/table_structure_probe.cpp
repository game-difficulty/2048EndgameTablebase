#include "Formation.h"
#include <iostream>

int main() {
    size_t count;
    while (std::cin >> std::dec >> count) {
        std::vector<uint64_t> masks(count);
        for (auto &mask : masks) std::cin >> std::hex >> mask;
        bool matched = false;
        for (int operation = 0; operation < 8; ++operation) {
            uint64_t board;
            std::cin >> std::hex >> board;
            matched = is_pattern(board, masks) || matched;
        }
        if (!std::cin) return 1;
        std::cout << (matched ? "match" : "mismatch") << '\n';
    }
}
