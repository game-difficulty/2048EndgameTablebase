#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <atomic>
#include <array>
#include <type_traits>
#include <variant>
#include <vector>

enum class SymmMode { Identity, Full, Diagonal, Horizontal, Min33, Min24, Min34 };

enum class SuccessRateKind { UInt32, UInt64, Float32, Float64 };

struct PatternSpec {
    std::string name;
    std::vector<uint64_t> pattern_masks;
    std::vector<uint8_t> success_shifts;
    int symm_mode = static_cast<int>(SymmMode::Identity);
};

struct RunOptions {
    int target = 0;
    int steps = 0;
    int docheck_step = 0;
    std::string pathname;
    bool is_free = false;
    bool is_variant = false;
    double spawn_rate4 = 0.1;
    std::string success_rate_dtype = "uint32";
    double deletion_threshold = 0.0;
    bool compress = false;
    bool compress_temp_files = false;
    bool optimal_branch_only = false;
    bool chunked_solve = false;
    int num_threads = 0;
};

[[nodiscard]] inline uint32_t build_progress_total(const RunOptions &options) {
    return options.steps > 0 ? static_cast<uint32_t>(options.steps * 2) : 0U;
}

[[nodiscard]] inline uint32_t classic_build_progress_total(const RunOptions &options) {
    uint32_t total = build_progress_total(options);
    if (options.optimal_branch_only && options.steps > 0) {
        total += static_cast<uint32_t>(options.steps);
    }
    return total;
}

struct BuildProgressSnapshot {
    uint32_t current = 0;
    uint32_t total = 0;
};

namespace FormationProgress {

inline std::atomic<uint32_t> current{0U};
inline std::atomic<uint32_t> total{0U};

inline void reset_build_progress(uint32_t next_total = 0U) {
    current.store(0U, std::memory_order_relaxed);
    total.store(next_total, std::memory_order_relaxed);
}

inline void update_build_progress(uint32_t next_current, uint32_t next_total) {
    total.store(next_total, std::memory_order_relaxed);
    current.store(next_current, std::memory_order_relaxed);
}

[[nodiscard]] inline BuildProgressSnapshot get_build_progress() {
    BuildProgressSnapshot snapshot;
    snapshot.total = total.load(std::memory_order_relaxed);
    snapshot.current = current.load(std::memory_order_relaxed);
    if (snapshot.current > snapshot.total) {
        snapshot.total = snapshot.current;
    }
    return snapshot;
}

} // namespace FormationProgress

struct AdvancedPatternSpec {
    std::string name;
    std::vector<uint64_t> pattern_masks;
    int symm_mode = static_cast<int>(SymmMode::Identity);
    uint8_t num_free_32k = 0;
    std::vector<uint8_t> fixed_32k_shifts;
    uint32_t small_tile_sum_limit = 96;
    uint8_t target = 0;
};

struct AdvancedMaskParam {
    uint32_t small_tile_sum_limit = 96;
    uint8_t target = 0;
    uint64_t pos_fixed_32k_mask = 0;
    uint8_t num_free_32k = 0;
    uint8_t num_fixed_32k = 0;
};

template <typename T> struct ArrayView {
    T *data = nullptr;
    size_t size = 0;

    [[nodiscard]] bool empty() const {
        return data == nullptr || size == 0;
    }

    [[nodiscard]] T &operator[](size_t index) const {
        return data[index];
    }

    [[nodiscard]] T *begin() const {
        return data;
    }

    [[nodiscard]] T *end() const {
        return data + size;
    }
};

template <typename T> struct MatrixView {
    T *data = nullptr;
    size_t rows = 0;
    size_t cols = 0;

    [[nodiscard]] bool empty() const {
        return data == nullptr || rows == 0 || cols == 0;
    }

    [[nodiscard]] T *row(size_t index) const {
        return data + index * cols;
    }

    [[nodiscard]] T &at(size_t row_index, size_t col_index) const {
        return data[row_index * cols + col_index];
    }
};

constexpr int bucket_key_min() {
    return -16;
}

constexpr int bucket_key_max() {
    return 31;
}

constexpr size_t bucket_slot_count() {
    return static_cast<size_t>(bucket_key_max() - bucket_key_min() + 1);
}

constexpr size_t bucket_to_index(int key) {
    return static_cast<size_t>(key - bucket_key_min());
}

inline void validate_bucket_key(int key) {
    if (key < bucket_key_min() || key > bucket_key_max()) {
        throw std::out_of_range("advanced bucket key out of range");
    }
}

template <typename T> struct BucketStore {
    std::array<T, bucket_slot_count()> values{};
    std::array<bool, bucket_slot_count()> present{};

    T &at(int key) {
        validate_bucket_key(key);
        present[bucket_to_index(key)] = true;
        return values[bucket_to_index(key)];
    }

    const T &at(int key) const {
        validate_bucket_key(key);
        return values[bucket_to_index(key)];
    }

    [[nodiscard]] bool contains(int key) const {
        if (key < bucket_key_min() || key > bucket_key_max()) {
            return false;
        }
        return present[bucket_to_index(key)];
    }
};

#pragma pack(push, 1)
template <typename T> struct SuccessEntry {
    uint64_t board;
    T success;
};
#pragma pack(pop)

static_assert(sizeof(SuccessEntry<uint32_t>) == 12, "Packed uint32 entry layout changed");
static_assert(sizeof(SuccessEntry<uint64_t>) == 16, "Packed uint64 entry layout changed");
static_assert(sizeof(SuccessEntry<float>) == 12, "Packed float entry layout changed");
static_assert(sizeof(SuccessEntry<double>) == 16, "Packed double entry layout changed");

struct PatternLayer {
    using Storage = std::variant<
        std::vector<SuccessEntry<uint32_t>>,
        std::vector<SuccessEntry<uint64_t>>,
        std::vector<SuccessEntry<float>>,
        std::vector<SuccessEntry<double>>
    >;

    SuccessRateKind kind = SuccessRateKind::UInt32;
    Storage storage = std::vector<SuccessEntry<uint32_t>>{};

    [[nodiscard]] size_t size() const {
        return std::visit([](const auto &items) { return items.size(); }, storage);
    }

    [[nodiscard]] bool empty() const {
        return size() == 0;
    }

    [[nodiscard]] std::string dtype_name() const {
        switch (kind) {
            case SuccessRateKind::UInt32:
                return "uint32";
            case SuccessRateKind::UInt64:
                return "uint64";
            case SuccessRateKind::Float32:
                return "float32";
            case SuccessRateKind::Float64:
                return "float64";
        }
        return "uint32";
    }
};

template <typename T> struct SuccessRateKindMap;
template <> struct SuccessRateKindMap<uint32_t> { static constexpr SuccessRateKind value = SuccessRateKind::UInt32; };
template <> struct SuccessRateKindMap<uint64_t> { static constexpr SuccessRateKind value = SuccessRateKind::UInt64; };
template <> struct SuccessRateKindMap<float> { static constexpr SuccessRateKind value = SuccessRateKind::Float32; };
template <> struct SuccessRateKindMap<double> { static constexpr SuccessRateKind value = SuccessRateKind::Float64; };

template <typename T> inline PatternLayer make_pattern_layer(std::vector<SuccessEntry<T>> items) {
    PatternLayer layer;
    layer.kind = SuccessRateKindMap<T>::value;
    layer.storage = std::move(items);
    return layer;
}

template <typename T> inline std::vector<SuccessEntry<T>> &layer_as(PatternLayer &layer) {
    return std::get<std::vector<SuccessEntry<T>>>(layer.storage);
}

template <typename T> inline const std::vector<SuccessEntry<T>> &layer_as(const PatternLayer &layer) {
    return std::get<std::vector<SuccessEntry<T>>>(layer.storage);
}

inline bool is_one_minus_success_rate_dtype(const std::string &name) {
    return name == "1-float32" || name == "1-float64";
}

inline SuccessRateKind success_rate_kind_from_name(const std::string &name) {
    if (name == "uint64") {
        return SuccessRateKind::UInt64;
    }
    if (name == "float32" || name == "1-float32") {
        return SuccessRateKind::Float32;
    }
    if (name == "float64" || name == "1-float64") {
        return SuccessRateKind::Float64;
    }
    return SuccessRateKind::UInt32;
}

template <typename T> constexpr T zero_value() {
    return static_cast<T>(0);
}

template <typename T> constexpr T max_scale_value();

template <> constexpr uint32_t max_scale_value<uint32_t>() {
    return 4000000000u;
}

template <> constexpr uint64_t max_scale_value<uint64_t>() {
    return 1600000000000000000ULL;
}

template <> constexpr float max_scale_value<float>() {
    return 1.0f;
}

template <> constexpr double max_scale_value<double>() {
    return 1.0;
}

template <typename T> inline T zero_value_for_dtype(const std::string &name) {
    if constexpr (std::is_floating_point_v<T>) {
        if (is_one_minus_success_rate_dtype(name)) {
            return static_cast<T>(-1);
        }
    }
    return zero_value<T>();
}

template <typename T> inline T max_scale_value_for_dtype(const std::string &name) {
    if constexpr (std::is_floating_point_v<T>) {
        if (is_one_minus_success_rate_dtype(name)) {
            return static_cast<T>(0);
        }
    }
    return max_scale_value<T>();
}
