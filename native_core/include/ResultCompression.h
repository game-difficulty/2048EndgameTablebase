#pragma once

#include "NativeLzma.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace ResultCompression {

struct ValuePolicy {
    bool valid = false;
    bool stored = false;
    int64_t sampled_ordinal = 0;
    double absolute_threshold = 0;
    double relative_threshold = 0;
    uint32_t dtype = 0;
    int level = 0;
    std::vector<uint32_t> row_widths;
};

struct Context {
    ValuePolicy *policy = nullptr;
    int64_t ordinal = 0;
    double absolute_threshold = 0;
    double relative_threshold = 0;
};

// Owned by one solve invocation; never inherited across restarts or tables.
template <typename Sample>
bool with_layer_policy(const Context &context, uint32_t dtype, int level,
                       const std::vector<uint32_t> &row_widths, Sample sample) {
    auto *p = context.policy;
    if (p && p->valid && p->absolute_threshold == context.absolute_threshold &&
        p->relative_threshold == context.relative_threshold && p->dtype == dtype &&
        p->level == level && p->row_widths == row_widths &&
        std::abs(context.ordinal - p->sampled_ordinal) < 10) {
        std::clog << "RESULT_COMPRESSION ordinal=" << context.ordinal
                  << " mode=" << (p->stored ? "stored" : "xz")
                  << " reason=inherited sampled_ordinal=" << p->sampled_ordinal << '\n';
        return p->stored;
    }
    const bool stored = sample();
    if (p) {
        *p = ValuePolicy{true, stored, context.ordinal, context.absolute_threshold,
                         context.relative_threshold, dtype, level, row_widths};
    }
    return stored;
}

// Match BC's 5% minimum saving, with a bounded, evenly spread sample per layer.
template <typename Read>
bool use_stored_values(uint64_t bytes, size_t alignment, int level,
                       const std::string &label, Read read) {
    constexpr uint64_t sample_limit = 256U * 1024U;
    if (alignment == 0 || bytes % alignment != 0) {
        throw std::invalid_argument("invalid result compression sample alignment");
    }
    uint64_t sampled = 0;
    uint64_t encoded = 0;
    if (bytes >= sample_limit) {
        const size_t take = static_cast<size_t>(std::min(sample_limit, bytes / 4) / alignment * alignment);
        if (take == 0) throw std::invalid_argument("result compression sample alignment too large");
        for (uint64_t i = 0; i < 4; ++i) {
            const uint64_t offset = ((bytes - take) / 3 * i) / alignment * alignment;
            const auto sample = read(offset, take);
            if (sample.size() != take) throw std::runtime_error("short result compression sample");
            const auto compressed = compress_xz_block_native(sample.data(), sample.size(), level);
            if (compressed.empty()) throw std::runtime_error("result compression sampling failed");
            sampled += sample.size();
            encoded += compressed.size();
        }
    }
    const bool stored = sampled != 0 && encoded * 100 >= sampled * 95;
    std::clog << "RESULT_COMPRESSION path=" << label << " mode=" << (stored ? "stored" : "xz")
              << " reason=" << (sampled ? "sampled" : "insufficient_sample")
              << " sample_raw_bytes=" << sampled << " sample_stored_bytes=" << encoded << '\n';
    return stored;
}

inline bool use_stored_values(const uint8_t *data, size_t bytes, size_t alignment,
                              int level, const std::string &label) {
    if (bytes != 0 && data == nullptr) throw std::invalid_argument("null result compression data");
    return use_stored_values(bytes, alignment, level, label, [&](uint64_t offset, size_t take) {
        return std::vector<uint8_t>(data + offset, data + offset + take);
    });
}

inline std::vector<uint8_t> encode(const uint8_t *data, size_t bytes, int level, bool stored) {
    auto result = stored ? store_xz_block_native(data, bytes) : compress_xz_block_native(data, bytes, level);
    if (result.empty()) throw std::runtime_error("result block encoding failed");
    return result;
}

} // namespace ResultCompression
