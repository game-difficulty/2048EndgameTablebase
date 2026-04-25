#include <iostream>
#include <tuple>
#include <cstdint>
#include <algorithm>
#include <array>

// 引入两个残局表的数据源
#include "egtb_data_512.h" 
#include "egtb_data_256.h" 
#include "egtb_data_1256.h" 
#include "BoardMover.h" 


namespace {
    const uint64_t BIT_OFFSETS[10] = {12, 28, 32, 36, 40, 44, 48, 52, 56, 60};
    const uint32_t POWERS_OF_9[10] = {1, 9, 81, 729, 6561, 59049, 531441, 4782969, 43046721, 387420489};

    inline uint32_t mix32(uint32_t x) {
        x ^= x >> 16;
        x *= 0x85ebca77U; 
        x ^= x >> 13;
        x *= 0xc2b2ae3dU; 
        x ^= x >> 16;
        return x;
    }

    inline uint32_t compress_board(uint64_t board) {
        uint64_t compressed_val = 0;
        for (int i = 0; i < 10; ++i) {
            uint64_t digit = (board >> BIT_OFFSETS[i]) & 0xF;
            compressed_val += digit * POWERS_OF_9[i];
        }
        return static_cast<uint32_t>(compressed_val);
    }

    inline int get_board_sum(uint64_t b) {
        int s = 0;
        for (int i = 0; i < 16; ++i) {
            uint64_t val = (b >> (4 * i)) & 0xF;
            if (val > 0) s += (1 << val);
        }
        return s;
    }

    // 新增参数: table_type
    inline int32_t query_perfect_hash(uint32_t b, int layer, int table_type) {
        if (table_type == 512 && (layer < 0 || layer >= 248)) return -1;
        if (table_type == 256 && (layer < 0 || layer >= 72)) return -1;
        if (table_type == 1256 && (layer < 0 || layer >= 72)) return -1;

        // 声明指向底层数组的指针
        uint32_t B, L;
        const uint16_t* seeds;
        const uint8_t* sigs;
        const uint16_t* rates;

        // 根据 table_type 动态绑定数据源，零运行时开销
        if (table_type == 512) {
            B = EGTB512_B[layer];
            L = EGTB512_L[layer];
            if (B == 0 || L == 0) return -1;
            seeds = EGTB512_SEEDS[layer];
            sigs = EGTB512_SIGS[layer];
            rates = EGTB512_RATES[layer];
        } 
        else if (table_type == 256) {
            B = EGTB256_B[layer];
            L = EGTB256_L[layer];
            if (B == 0 || L == 0) return -1;
            seeds = EGTB256_SEEDS[layer];
            sigs = EGTB256_SIGS[layer];
            rates = EGTB256_RATES[layer];
        } 
        else if (table_type == 1256) {
            B = EGTB1256_B[layer];
            L = EGTB1256_L[layer];
            if (B == 0 || L == 0) return -1;
            seeds = EGTB1256_SEEDS[layer];
            sigs = EGTB1256_SIGS[layer];
            rates = EGTB1256_RATES[layer];
        } 
        else {
            return -1; // 不支持的表类型
        }
        
        uint32_t b_idx = mix32(b) % B;
        uint32_t seed = seeds[b_idx]; 
        uint32_t hash_idx = mix32(b ^ seed) % L;
        uint8_t expected_sig = static_cast<uint8_t>((b + (b >> 8) + (b >> 16) + (b >> 24)) & 0xFFU);
        
        if (sigs[hash_idx] == expected_sig) {
            return rates[hash_idx];
        }
        return -1;
    }
}

// 暴露给外部调用的独立函数，返回 4 个方向的胜率
// 索引 0, 1, 2, 3 分别对应方向 1(左), 2(右), 3(上), 4(下)
std::array<float, 4> find_best_egtb_move(uint64_t target_board, int table_type) {
    // 1. 栈上分配，零开销初始化
    std::array<float, 4> rates = {0.0f, 0.0f, 0.0f, 0.0f};

    // 特判
    if (target_board == 0x012412567fff8fffULL) {
        rates[0] = 0.9111f;
        return rates;
    }
    
    for (int d = 1; d <= 4; ++d) {
        auto [post_board, score] = BoardMover::s_move_board(target_board, d);
        
        if (post_board == target_board) continue;
        
        int board_sum = get_board_sum(post_board) % 32768;
        int initial_sum;
        if (table_type == 1256) initial_sum = 18;
        else initial_sum = 8;
        int layer = (board_sum - initial_sum) / 2;

        if (table_type == 1256) post_board = canonical_diagonal(post_board);
        
        uint32_t compressed_b = compress_board(post_board);
        if (compressed_b == 0) continue;
        
        int32_t rate_uint16 = query_perfect_hash(compressed_b, layer, table_type);
        
        if (rate_uint16 != -1) {
            float decoded_rate = 0.0f;
            // 使用 LaTeX 表达解码逻辑
            // $$decoded\_rate = \frac{rate\_uint16}{65535} \times (Upper - Lower) + Lower$$
            if (table_type == 512) {
                decoded_rate = (static_cast<float>(rate_uint16) / 65535.0f) * (0.96f - 0.4f) + 0.4f;
            } else if (table_type == 256) {
                decoded_rate = (static_cast<float>(rate_uint16) / 65535.0f) * (0.99999f - 0.75f) + 0.75f;
            } else if (table_type == 1256) {
                decoded_rate = (static_cast<float>(rate_uint16) / 65535.0f) * (0.28f - 0.05f) + 0.05f;
            }
            rates[d - 1] = decoded_rate;
        }
    }
    
    return rates;
}
