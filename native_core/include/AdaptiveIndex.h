#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace AdaptiveIndex {

constexpr uint32_t kL1Bits = 1U << 20U;
constexpr uint32_t kL1Size = kL1Bits + 1U;
constexpr uint32_t kL1WordCount = kL1Bits / 64U;
constexpr uint32_t kSubBits = 1U << 12U;
constexpr uint32_t kSubSize = kSubBits + 1U;
constexpr uint32_t kSubWordCount = kSubBits / 64U;

struct Config {
    uint32_t l2_split_threshold = 64U;
    uint32_t l3_split_threshold = 64U;
    int num_threads = 0;
};

struct Range {
    uint64_t begin = 0;
    uint64_t end = 0;

    [[nodiscard]] bool empty() const {
        return begin >= end;
    }

    [[nodiscard]] uint64_t size() const {
        return end - begin;
    }
};

struct BitRank4096 {
    std::array<uint64_t, kSubWordCount> words{};
    std::array<uint16_t, kSubWordCount + 1U> ranks{};
};

struct L3Node {
    BitRank4096 h3_map{};
    uint64_t starts_offset = 0;
};

struct L2Node {
    uint64_t parent_begin = 0;
    BitRank4096 h2_map{};
    BitRank4096 l3_map{};
    uint64_t starts_offset = 0;
    uint64_t l3_nodes_offset = 0;
};

struct Index {
    std::vector<uint64_t> l1_offsets;
    std::array<uint64_t, kL1WordCount> l2_root_bitmap{};
    std::array<uint32_t, kL1WordCount + 1U> l2_root_rank{};
    std::vector<L2Node> l2_nodes;
    std::vector<L3Node> l3_nodes;
    std::vector<uint32_t> l2_relative_starts;
    std::vector<uint32_t> l3_relative_starts;

    [[nodiscard]] bool empty() const;
    [[nodiscard]] Range locate(uint64_t key) const;
};

Index build(const uint64_t *keys, uint64_t size, const Config &config = {});
Index build(const std::vector<uint64_t> &keys, const Config &config = {});
Index build_experimental(const uint64_t *keys, uint64_t size, const Config &config = {});
Index build_experimental(const std::vector<uint64_t> &keys, const Config &config = {});
Index build_experimental_parallel_l1(const uint64_t *keys, uint64_t size, const Config &config = {});
Index build_experimental_parallel_l1(const std::vector<uint64_t> &keys, const Config &config = {});
Index build_experimental_hybrid(const uint64_t *keys, uint64_t size, const Config &config = {});
Index build_experimental_hybrid(const std::vector<uint64_t> &keys, const Config &config = {});
Range locate(const Index &index, uint64_t key);

} // namespace AdaptiveIndex
