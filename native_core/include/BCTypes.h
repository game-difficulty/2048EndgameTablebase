#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace BC {

using FamilyCoord = uint32_t;
using FamilyId = uint16_t;
using CellId = uint32_t;
using LayerSum = uint64_t;
using SpawnDeltaCoord = uint16_t;

template <typename T, std::size_t Capacity>
class SmallVector {
public:
    using iterator = typename std::array<T, Capacity>::iterator;
    using const_iterator = typename std::array<T, Capacity>::const_iterator;

    [[nodiscard]] std::size_t size() const {
        return size_;
    }

    [[nodiscard]] bool empty() const {
        return size_ == 0U;
    }

    void push_back(const T &value) {
        if (size_ >= Capacity) {
            throw std::length_error("BC::SmallVector capacity exceeded");
        }
        values_[size_++] = value;
    }

    [[nodiscard]] const T &operator[](std::size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("BC::SmallVector index out of range");
        }
        return values_[index];
    }

    [[nodiscard]] T &operator[](std::size_t index) {
        if (index >= size_) {
            throw std::out_of_range("BC::SmallVector index out of range");
        }
        return values_[index];
    }

    [[nodiscard]] const T *begin() const {
        return values_.data();
    }

    [[nodiscard]] const T *end() const {
        return values_.data() + size_;
    }

    [[nodiscard]] T *begin() {
        return values_.data();
    }

    [[nodiscard]] T *end() {
        return values_.data() + size_;
    }

private:
    std::array<T, Capacity> values_{};
    std::size_t size_ = 0U;
};

using FamilyIdList2 = SmallVector<FamilyId, 2U>;
using FamilyIdList3 = SmallVector<FamilyId, 3U>;

} // namespace BC
