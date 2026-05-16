#include "EXPrefix40Layer.h"

#include "EXFrozenLayer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace Prefix40Baseline {

namespace {

inline int effective_threads(int requested) {
#if defined(_OPENMP)
    if (requested > 0) {
        return requested;
    }
    return std::max(1, omp_get_max_threads());
#else
    return requested > 0 ? requested : 1;
#endif
}

} // namespace

uint32_t tile_value(uint32_t tile) {
    return tile == 0U ? 0U : (1U << tile);
}

uint32_t sum_index(uint32_t sum) {
    if ((sum & 1U) != 0U) {
        throw std::runtime_error("encountered odd sum in prefix40 baseline layer");
    }
    return sum >> 1U;
}

uint64_t bytes_for_bits(uint32_t bits) {
    return static_cast<uint64_t>((bits + 7U) >> 3U);
}

uint64_t words_for_bits(uint32_t bits) {
    return static_cast<uint64_t>((bits + 63U) >> 6U);
}

uint32_t board_layer_sum(uint64_t board, const Luts &luts) {
    return luts.row16_sum[static_cast<uint16_t>(board >> 48U)]
         + luts.row16_sum[static_cast<uint16_t>(board >> 32U)]
         + luts.row16_sum[static_cast<uint16_t>(board >> 16U)]
         + luts.row16_sum[static_cast<uint16_t>(board)];
}

uint32_t prefix40_sum(uint64_t prefix40, const Luts &luts) {
    const uint16_t row0 = static_cast<uint16_t>(prefix40 >> 24U);
    const uint16_t row1 = static_cast<uint16_t>(prefix40 >> 8U);
    const uint16_t tail = static_cast<uint16_t>((prefix40 & 0xFFU) << 8U);
    return luts.row16_sum[row0] + luts.row16_sum[row1] + luts.row16_sum[tail];
}

uint32_t prefix40_remaining_sum(uint64_t prefix40, uint32_t layer_sum, const Luts &luts) {
    const uint32_t used = prefix40_sum(prefix40, luts);
    if (used > layer_sum) {
        throw std::runtime_error("prefix40 sum exceeds layer sum");
    }
    return layer_sum - used;
}

uint64_t pack_bucket_key(uint64_t prefix40, uint32_t remaining_sum) {
    return (prefix40 << 24U) | static_cast<uint64_t>(remaining_sum);
}

uint64_t bucket_key_prefix40(uint64_t key) {
    return key >> 24U;
}

uint32_t bucket_key_remaining_sum(uint64_t key) {
    return static_cast<uint32_t>(key & 0xFFFFFFULL);
}

Luts build_luts(const ZMaskFrozen::TileLimitConfig &config, int num_threads) {
    const int thread_count = effective_threads(num_threads);
    const uint32_t group_count = (kMaxSum >> 1U) + 1U;
    std::vector<uint32_t> counts(group_count, 0U);
    std::vector<uint32_t> sums(kSuffixStateCount, 0U);
    std::vector<uint8_t> valid(kSuffixStateCount, 0U);

#pragma omp parallel
    {
        std::vector<uint32_t> local_counts(group_count, 0U);
#pragma omp for schedule(static) nowait
        for (int64_t state = 0; state < static_cast<int64_t>(kSuffixStateCount); ++state) {
            uint32_t sum = 0U;
            if (!ZMaskFrozen::decode_suffix24_if_valid(static_cast<uint32_t>(state), sum, config)) {
                continue;
            }
            valid[static_cast<size_t>(state)] = 1U;
            sums[static_cast<size_t>(state)] = sum;
            ++local_counts[sum_index(sum)];
        }
#pragma omp critical
        {
            for (uint32_t i = 0; i < group_count; ++i) {
                counts[i] += local_counts[i];
            }
        }
    }

    Luts luts;
    luts.rank_table.assign(kSuffixStateCount, kInvalidRank);
    luts.size_table.resize(group_count);
    luts.offset_table.resize(group_count);

    uint64_t total_states = 0U;
    for (uint32_t i = 0; i < group_count; ++i) {
        if (counts[i] > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("prefix40 baseline LUT size_table exceeds uint16 range");
        }
        luts.size_table[i] = static_cast<uint16_t>(counts[i]);
        luts.offset_table[i] = static_cast<uint32_t>(total_states);
        total_states += counts[i];
    }

    luts.unrank_array.resize(total_states);
    luts.row16_sum.resize(1U << 16U);
    for (uint32_t row = 0; row < (1U << 16U); ++row) {
        luts.row16_sum[row] = tile_value((row >> 0U) & 0xFU)
                            + tile_value((row >> 4U) & 0xFU)
                            + tile_value((row >> 8U) & 0xFU)
                            + tile_value((row >> 12U) & 0xFU);
    }

    std::vector<uint32_t> cursor(group_count, 0U);
    for (uint32_t state = 0; state < kSuffixStateCount; ++state) {
        if (valid[state] == 0U) {
            continue;
        }
        const uint32_t idx = sum_index(sums[state]);
        const uint32_t rank = cursor[idx]++;
        if (rank > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("prefix40 baseline LUT rank exceeds uint16 range");
        }
        luts.rank_table[state] = static_cast<uint16_t>(rank);
        luts.unrank_array[luts.offset_table[idx] + rank] = state;
    }

    (void)thread_count;
    return luts;
}

void set_small_bit(std::vector<uint8_t> &bitmap, uint32_t offset, uint32_t rank) {
    const uint32_t byte_idx = offset + (rank >> 3U);
    const uint32_t bit = rank & 7U;
    bitmap[byte_idx] = static_cast<uint8_t>(bitmap[byte_idx] | static_cast<uint8_t>(1U << bit));
}

void set_large_bit(std::vector<uint64_t> &bitmap, uint32_t offset, uint32_t rank) {
    const uint32_t word_idx = offset + (rank >> 6U);
    const uint32_t bit = rank & 63U;
    bitmap[word_idx] |= (1ULL << bit);
}

Layer build_layer_from_sorted_boards(
    const std::vector<uint64_t> &boards,
    const Luts &luts,
    int num_threads
) {
    Layer layer;
    if (boards.empty()) {
        return layer;
    }

    const int thread_count = effective_threads(num_threads);
    layer.layer_sum = board_layer_sum(boards.front(), luts);
    layer.threshold_bits = kDefaultThresholdBits;
    layer.live_board_count = boards.size();

    std::vector<uint64_t> prefixes;
    std::vector<size_t> range_begin;
    std::vector<size_t> range_count;
    prefixes.reserve(boards.size() / 8U + 1U);
    range_begin.reserve(prefixes.capacity());
    range_count.reserve(prefixes.capacity());

    size_t begin = 0U;
    while (begin < boards.size()) {
        const uint64_t prefix40 = boards[begin] >> 24U;
        size_t end = begin + 1U;
        while (end < boards.size() && (boards[end] >> 24U) == prefix40) {
            ++end;
        }
        prefixes.push_back(prefix40);
        range_begin.push_back(begin);
        range_count.push_back(end - begin);
        begin = end;
    }

    const size_t bucket_count = prefixes.size();
    layer.bucket_keys.resize(bucket_count);
    layer.bitmap_offsets.resize(bucket_count);
    layer.dense_offsets.resize(bucket_count);

    uint64_t small_bytes_total = 0U;
    uint64_t large_words_total = 0U;
    uint64_t exact_bits = 0U;
    uint64_t aligned_bits = 0U;
    uint64_t dense_cursor = 0U;

    for (size_t idx = 0; idx < bucket_count; ++idx) {
        const uint32_t remaining_sum = prefix40_remaining_sum(prefixes[idx], layer.layer_sum, luts);
        const uint32_t valid_count = luts.size_table[sum_index(remaining_sum)];
        layer.bucket_keys[idx] = pack_bucket_key(prefixes[idx], remaining_sum);
        layer.dense_offsets[idx] = static_cast<uint32_t>(dense_cursor);
        dense_cursor += range_count[idx];
        exact_bits += valid_count;
        if (valid_count <= layer.threshold_bits) {
            layer.bitmap_offsets[idx] = static_cast<uint32_t>(small_bytes_total);
            small_bytes_total += bytes_for_bits(valid_count);
            aligned_bits += bytes_for_bits(valid_count) * 8ULL;
        } else {
            layer.bitmap_offsets[idx] = static_cast<uint32_t>(large_words_total);
            large_words_total += words_for_bits(valid_count);
            aligned_bits += words_for_bits(valid_count) * 64ULL;
        }
    }

    layer.exact_bitmap_bits = exact_bits;
    layer.aligned_bitmap_bits = aligned_bits;
    layer.small_bitmap_bytes.assign(small_bytes_total, 0U);
    layer.large_bitmap_words.assign(large_words_total, 0ULL);

#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int64_t idx = 0; idx < static_cast<int64_t>(bucket_count); ++idx) {
        const uint64_t prefix40 = prefixes[static_cast<size_t>(idx)];
        const uint32_t remaining_sum = prefix40_remaining_sum(prefix40, layer.layer_sum, luts);
        const uint32_t valid_count = luts.size_table[sum_index(remaining_sum)];
        const uint32_t bitmap_offset = layer.bitmap_offsets[static_cast<size_t>(idx)];
        const bool is_small = valid_count <= layer.threshold_bits;
        const size_t start = range_begin[static_cast<size_t>(idx)];
        const size_t end = start + range_count[static_cast<size_t>(idx)];
        for (size_t pos = start; pos < end; ++pos) {
            const uint32_t suffix24 = static_cast<uint32_t>(boards[pos] & 0xFFFFFFULL);
            const uint16_t rank = luts.rank_table[suffix24];
            if (rank == kInvalidRank || rank >= valid_count) {
                throw std::runtime_error("invalid prefix40 baseline rank during layer build");
            }
            if (is_small) {
                set_small_bit(layer.small_bitmap_bytes, bitmap_offset, rank);
            } else {
                set_large_bit(layer.large_bitmap_words, bitmap_offset, rank);
            }
        }
    }

    return layer;
}

} // namespace Prefix40Baseline
