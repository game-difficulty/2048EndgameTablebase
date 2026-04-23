#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace HybridSearch {

enum class Mode : uint8_t {
    Scalar = 0,
    AVX2 = 1,
    AVX512 = 2,
};

constexpr size_t kNotFound = std::numeric_limits<size_t>::max();

[[nodiscard]] Mode mode();
[[nodiscard]] size_t exact_search(const uint64_t *arr, size_t length, uint64_t target);
[[nodiscard]] size_t lower_bound(const uint64_t *arr, size_t length, uint64_t target);

} // namespace HybridSearch
