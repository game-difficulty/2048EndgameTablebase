#pragma once

#include "PathUtils.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <atomic>
#include <array>
#include <cmath>
#include <fstream>
#include <type_traits>
#include <variant>
#include <vector>

enum class SymmMode { Identity, Full, Diagonal, Horizontal, Min33, Min24, Min34, Min34Top };

enum class SuccessRateKind { UInt32, UInt64, Float32, Float64 };

struct PatternSpec {
    std::string name;
    std::vector<uint64_t> pattern_masks;
    std::vector<uint8_t> success_shifts;
    int symm_mode = static_cast<int>(SymmMode::Identity);
    uint8_t physical_transform = 0;
    uint8_t inverse_physical_transform = 0;
    uint64_t logical_pattern_signature = 0;
    uint64_t physical_pattern_signature = 0;
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
    double relative_deletion_threshold = 0.0;
    std::string deletion_threshold_signal_path;
    bool compress = false;
    bool compress_temp_files = false;
    bool optimal_branch_only = false;
    bool chunked_solve = false;
    int num_threads = 0;
    bool direct_io = false;
    int direct_io_queue_depth = 16;
    int direct_io_chunk_mib = 8;
};

namespace RuntimeControls {

struct DeletionThresholdState {
    double absolute = 0.0;
    double relative = 0.0;
};

inline double clamp_deletion_threshold(double value) {
    if (!std::isfinite(value)) {
        return 0.0;
    }
    if (value < 0.0) {
        return 0.0;
    }
    constexpr double kMaxDeletionThreshold = 0.999999;
    return value > kMaxDeletionThreshold ? kMaxDeletionThreshold : value;
}

inline DeletionThresholdState clamp_deletion_thresholds(DeletionThresholdState state) {
    state.absolute = clamp_deletion_threshold(state.absolute);
    state.relative = clamp_deletion_threshold(state.relative);
    return state;
}

inline DeletionThresholdState current_deletion_thresholds(const RunOptions &options) {
    DeletionThresholdState state{
        options.deletion_threshold,
        options.relative_deletion_threshold
    };
    if (!options.deletion_threshold_signal_path.empty()) {
        std::ifstream in(NativePath::from_utf8(options.deletion_threshold_signal_path));
        double signaled_absolute = 0.0;
        if (in >> signaled_absolute) {
            state.absolute = signaled_absolute;
            double signaled_relative = 0.0;
            if (in >> signaled_relative) {
                state.relative = signaled_relative;
            } else {
                state.relative = 0.0;
            }
        }
    }
    return clamp_deletion_thresholds(state);
}

inline DeletionThresholdState refresh_deletion_thresholds(
    const RunOptions &options,
    DeletionThresholdState current_state
) {
    if (options.deletion_threshold_signal_path.empty()) {
        return clamp_deletion_thresholds(current_state);
    }
    std::ifstream in(NativePath::from_utf8(options.deletion_threshold_signal_path));
    double signaled_absolute = 0.0;
    if (in >> signaled_absolute) {
        current_state.absolute = signaled_absolute;
        double signaled_relative = 0.0;
        if (in >> signaled_relative) {
            current_state.relative = signaled_relative;
        } else {
            current_state.relative = 0.0;
        }
    }
    return clamp_deletion_thresholds(current_state);
}

inline double current_deletion_threshold(const RunOptions &options) {
    return current_deletion_thresholds(options).absolute;
}

inline double refresh_deletion_threshold(const RunOptions &options, double current_value) {
    DeletionThresholdState state{current_value, options.relative_deletion_threshold};
    return refresh_deletion_thresholds(options, state).absolute;
}

inline bool deletion_threshold_enabled(DeletionThresholdState state) {
    state = clamp_deletion_thresholds(state);
    return state.absolute > 0.0 || state.relative > 0.0;
}

template <typename T>
inline double normalized_success_value(T value, T zero_value, T max_scale_value) {
    const long double zero = static_cast<long double>(zero_value);
    const long double scale = static_cast<long double>(max_scale_value) - zero;
    if (scale == 0.0L) {
        return 0.0;
    }
    return static_cast<double>((static_cast<long double>(value) - zero) / scale);
}

template <typename T>
inline T scaled_deletion_threshold(
    double ratio,
    T zero_value,
    T max_scale_value
) {
    const long double zero = static_cast<long double>(zero_value);
    const long double scale = static_cast<long double>(max_scale_value) - zero;
    const long double threshold = zero + scale * static_cast<long double>(clamp_deletion_threshold(ratio));
    if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(threshold < zero ? zero : threshold);
    } else {
        return static_cast<T>(threshold);
    }
}

template <typename T>
inline T absolute_deletion_threshold(
    T zero_value,
    T max_scale_value,
    DeletionThresholdState state
) {
    return scaled_deletion_threshold(state.absolute, zero_value, max_scale_value);
}

template <typename T>
inline T relative_deletion_threshold(
    T layer_max_value,
    T zero_value,
    DeletionThresholdState state
) {
    state = clamp_deletion_thresholds(state);
    const long double zero = static_cast<long double>(zero_value);
    const long double threshold = zero
        + (static_cast<long double>(layer_max_value) - zero) * static_cast<long double>(state.relative);
    if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(threshold < zero ? zero : threshold);
    } else {
        return static_cast<T>(threshold);
    }
}

template <typename T>
inline T max_deletion_threshold(T lhs, T rhs) {
    return lhs > rhs ? lhs : rhs;
}

template <typename T>
inline T effective_deletion_threshold(
    T layer_max_value,
    T zero_value,
    T max_scale_value,
    DeletionThresholdState state
) {
    return max_deletion_threshold(
        absolute_deletion_threshold(zero_value, max_scale_value, state),
        relative_deletion_threshold(layer_max_value, zero_value, state)
    );
}

template <typename T>
inline double normalized_deletion_threshold(
    T threshold,
    T zero_value,
    T max_scale_value
) {
    return normalized_success_value(threshold, zero_value, max_scale_value);
}

inline double retention_ratio(uint64_t after, uint64_t before) {
    return before == 0U ? 0.0 : static_cast<double>(after) / static_cast<double>(before);
}

} // namespace RuntimeControls

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
    std::vector<uint8_t> success_shifts;
    int symm_mode = static_cast<int>(SymmMode::Identity);
    uint8_t physical_transform = 0;
    uint8_t inverse_physical_transform = 0;
    uint64_t logical_pattern_signature = 0;
    uint64_t physical_pattern_signature = 0;
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
