#pragma once

#include "BCTypes.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace BC {

using BucketRank = uint16_t;
using BucketBitmapLen = uint16_t;
using RankPrefix = uint16_t;

inline constexpr uint32_t kBCQuadrantWordCount = 1U << 16U;
inline constexpr uint32_t kBCMaxSumId = (1U << 12U) - 1U;
inline constexpr uint32_t kBCMaxWordsPerSumMask = 36U;
inline constexpr uint32_t kBCMaxBucketBitmapLen =
    kBCMaxWordsPerSumMask * kBCMaxWordsPerSumMask * kBCMaxWordsPerSumMask;
inline constexpr uint32_t kBCRankPrefixBits = 256U;
inline constexpr uint32_t kBCBitmapWordBits = 64U;

struct BCWordDesc {
    uint32_t sum = 0U;
    uint16_t sum_id = 0U;
    uint16_t packed_sum_mask = 0U;
    uint16_t group_count = 0U;
    uint8_t empty_mask = 0U;
    BucketRank rank = 0U;
    bool valid = false;
};

struct BCWordGroupView {
    const uint16_t *words = nullptr;
    uint16_t count = 0U;
};

[[nodiscard]] inline std::array<uint32_t, 16U> default_2048_tile_sum_values() {
    std::array<uint32_t, 16U> values{};
    values[0] = 0U;
    for (uint32_t tile = 1U; tile < values.size(); ++tile) {
        values[tile] = 1U << tile;
    }
    return values;
}

[[nodiscard]] inline uint8_t word_tile(uint16_t word, uint32_t index) {
    return static_cast<uint8_t>((word >> (index * 4U)) & 0xFU);
}

[[nodiscard]] inline uint8_t word_empty_mask(uint16_t word) {
    uint8_t mask = 0U;
    for (uint32_t i = 0; i < 4U; ++i) {
        if (word_tile(word, i) == 0U) {
            mask |= static_cast<uint8_t>(1U << i);
        }
    }
    return mask;
}

[[nodiscard]] inline uint16_t pack_sum_mask(uint16_t sum_id, uint8_t empty_mask) {
    if (sum_id > kBCMaxSumId) {
        throw std::out_of_range("BC sum_id does not fit in 12 bits");
    }
    if ((empty_mask & 0xF0U) != 0U) {
        throw std::out_of_range("BC empty_mask does not fit in 4 bits");
    }
    return static_cast<uint16_t>((sum_id << 4U) | static_cast<uint16_t>(empty_mask));
}

[[nodiscard]] inline uint16_t packed_sum_id(uint16_t packed) {
    return static_cast<uint16_t>(packed >> 4U);
}

[[nodiscard]] inline uint8_t packed_empty_mask(uint16_t packed) {
    return static_cast<uint8_t>(packed & 0xFU);
}

class BCLut {
public:
    explicit BCLut(const std::vector<uint8_t> &legal_tiles)
        : BCLut(legal_tiles, default_2048_tile_sum_values()) {}

    BCLut(
        const std::vector<uint8_t> &legal_tiles,
        const std::array<uint32_t, 16U> &tile_sum_values
    ) {
        build(legal_tiles, tile_sum_values);
    }

    [[nodiscard]] const BCWordDesc &word_desc(uint16_t word) const {
        return word_descs_[word];
    }

    [[nodiscard]] uint16_t count4(uint16_t sum_id, uint8_t empty_mask) const {
        if (sum_id >= count4_.size()) {
            throw std::out_of_range("BC count4 sum_id out of range");
        }
        if ((empty_mask & 0xF0U) != 0U) {
            throw std::out_of_range("BC count4 empty_mask out of range");
        }
        return count4_[sum_id][empty_mask & 0xFU];
    }

    // Hot path helper for keys produced by BC encode_key_and_rank. This skips
    // validation because packed sum/mask fields are already trusted there.
    [[nodiscard]] uint16_t count4_packed_trusted(uint16_t packed_sum_mask) const noexcept {
        return count4_[packed_sum_mask >> 4U][packed_sum_mask & 0xFU];
    }

    [[nodiscard]] BCWordGroupView word_group(uint16_t sum_id, uint8_t empty_mask) const {
        if (sum_id >= offset4_.size()) {
            throw std::out_of_range("BC word_group sum_id out of range");
        }
        if ((empty_mask & 0xF0U) != 0U) {
            throw std::out_of_range("BC word_group empty_mask out of range");
        }
        const uint8_t mask = static_cast<uint8_t>(empty_mask & 0xFU);
        const uint16_t count = count4_[sum_id][mask];
        const uint32_t offset = offset4_[sum_id][mask];
        if (count == 0U) {
            return BCWordGroupView{nullptr, 0U};
        }
        if (static_cast<uint64_t>(offset) + count > sum4_words_.size()) {
            throw std::logic_error("BC word_group range exceeds LUT word storage");
        }
        return BCWordGroupView{sum4_words_.data() + offset, count};
    }

    [[nodiscard]] uint32_t sum4_value(uint16_t sum_id) const {
        if (sum_id >= sum4_values_.size()) {
            throw std::out_of_range("BC sum4_value sum_id out of range");
        }
        return sum4_values_[sum_id];
    }

    [[nodiscard]] size_t sum_count() const {
        return sum4_values_.size();
    }

    [[nodiscard]] bool is_legal_tile(uint8_t tile) const {
        return tile < legal_tiles_.size() && legal_tiles_[tile];
    }

    [[nodiscard]] uint32_t tile_sum_value(uint8_t tile) const {
        if (tile >= tile_sum_values_.size()) {
            throw std::out_of_range("BC tile rank exceeds 4-bit alphabet");
        }
        return tile_sum_values_[tile];
    }

    [[nodiscard]] bool is_legal_word(uint16_t word) const {
        return word_desc(word).valid;
    }

    [[nodiscard]] uint16_t unrank_word(uint16_t sum_id, uint8_t empty_mask, BucketRank rank) const {
        if (sum_id >= offset4_.size()) {
            throw std::out_of_range("BC unrank sum_id out of range");
        }
        if ((empty_mask & 0xF0U) != 0U) {
            throw std::out_of_range("BC unrank empty_mask out of range");
        }
        const uint16_t count = count4(sum_id, empty_mask);
        if (rank >= count) {
            throw std::out_of_range("BC unrank rank out of range");
        }
        return sum4_words_[offset4_[sum_id][empty_mask & 0xFU] + rank];
    }

    [[nodiscard]] uint16_t pack_sum_mask_for_word(uint16_t word) const {
        const BCWordDesc &desc = word_desc(word);
        if (!desc.valid) {
            throw std::invalid_argument("BC cannot pack invalid quadrant word");
        }
        return desc.packed_sum_mask;
    }

private:
    void build(
        const std::vector<uint8_t> &legal_tiles,
        const std::array<uint32_t, 16U> &tile_sum_values
    ) {
        legal_tiles_.fill(false);
        if (legal_tiles.empty()) {
            throw std::invalid_argument("BC legal tile alphabet must not be empty");
        }
        for (uint8_t tile : legal_tiles) {
            if (tile >= legal_tiles_.size()) {
                throw std::invalid_argument("BC legal tile exceeds 4-bit alphabet");
            }
            legal_tiles_[tile] = true;
        }
        tile_sum_values_ = tile_sum_values;
        word_descs_.assign(kBCQuadrantWordCount, BCWordDesc{});

        std::vector<uint32_t> sums;
        sums.reserve(kBCQuadrantWordCount);
        for (uint32_t word = 0; word < kBCQuadrantWordCount; ++word) {
            bool valid = true;
            uint32_t sum = 0U;
            for (uint32_t i = 0; i < 4U; ++i) {
                const uint8_t tile = word_tile(static_cast<uint16_t>(word), i);
                if (!legal_tiles_[tile]) {
                    valid = false;
                    break;
                }
                sum += tile_sum_values_[tile];
            }
            if (valid) {
                sums.push_back(sum);
            }
        }
        std::sort(sums.begin(), sums.end());
        sums.erase(std::unique(sums.begin(), sums.end()), sums.end());
        if (sums.empty()) {
            throw std::invalid_argument("BC legal tile alphabet produced no valid words");
        }
        if (sums.size() > kBCMaxSumId + 1U) {
            throw std::invalid_argument("BC compact sum_id exceeds 12-bit key field");
        }
        sum4_values_ = sums;

        const uint32_t max_sum = sums.back();
        std::vector<uint16_t> sum_to_id(static_cast<size_t>(max_sum) + 1U, invalid_sum_id());
        for (uint32_t id = 0; id < sums.size(); ++id) {
            sum_to_id[sums[id]] = static_cast<uint16_t>(id);
        }

        std::vector<std::array<std::vector<uint16_t>, 16U>> groups(sums.size());
        for (uint32_t word = 0; word < kBCQuadrantWordCount; ++word) {
            bool valid = true;
            uint32_t sum = 0U;
            for (uint32_t i = 0; i < 4U; ++i) {
                const uint8_t tile = word_tile(static_cast<uint16_t>(word), i);
                if (!legal_tiles_[tile]) {
                    valid = false;
                    break;
                }
                sum += tile_sum_values_[tile];
            }
            if (!valid) {
                continue;
            }
            const uint16_t sum_id = sum_to_id[sum];
            if (sum_id == invalid_sum_id()) {
                throw std::logic_error("BC sum id lookup failed during LUT build");
            }
            const uint8_t empty_mask = word_empty_mask(static_cast<uint16_t>(word));
            std::vector<uint16_t> &group = groups[sum_id][empty_mask];
            if (group.size() >= kBCMaxWordsPerSumMask) {
                throw std::invalid_argument("BC fixed sum+empty_mask word count exceeds 36");
            }
            word_descs_[word] = BCWordDesc{
                sum,
                sum_id,
                pack_sum_mask(sum_id, empty_mask),
                0U,
                empty_mask,
                static_cast<BucketRank>(group.size()),
                true
            };
            group.push_back(static_cast<uint16_t>(word));
        }

        count4_.assign(sums.size(), {});
        offset4_.assign(sums.size(), {});
        uint32_t cursor = 0U;
        for (uint32_t sum_id = 0; sum_id < groups.size(); ++sum_id) {
            for (uint32_t mask = 0; mask < 16U; ++mask) {
                const std::vector<uint16_t> &group = groups[sum_id][mask];
                count4_[sum_id][mask] = static_cast<uint16_t>(group.size());
                offset4_[sum_id][mask] = cursor;
                cursor += static_cast<uint32_t>(group.size());
                sum4_words_.insert(sum4_words_.end(), group.begin(), group.end());
            }
        }
        for (BCWordDesc &desc : word_descs_) {
            if (!desc.valid) {
                continue;
            }
            desc.group_count = count4_[desc.sum_id][desc.empty_mask & 0xFU];
        }
    }

    [[nodiscard]] static constexpr uint16_t invalid_sum_id() {
        return std::numeric_limits<uint16_t>::max();
    }

    std::array<bool, 16U> legal_tiles_{};
    std::array<uint32_t, 16U> tile_sum_values_{};
    std::vector<BCWordDesc> word_descs_;
    std::vector<uint32_t> sum4_values_;
    std::vector<std::array<uint16_t, 16U>> count4_;
    std::vector<std::array<uint32_t, 16U>> offset4_;
    std::vector<uint16_t> sum4_words_;
};

} // namespace BC
