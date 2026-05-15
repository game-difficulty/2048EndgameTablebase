#pragma once

#include "EXADLayer.h"
#include "FileIOUtils.h"

#include <cstddef>
#include <memory>
#include <string>

namespace EXAD {

constexpr const char *kLayerFileExtension = ".exadtmp";
constexpr const char *kLutFileExtension = ".exadlut";

std::string layer_file_path(const std::string &pathname, int step);
std::string lut_file_path(const std::string &pathname);
bool layer_file_exists(const std::string &path);
void remove_layer_file(const std::string &path);

struct LayerFileInfo {
    uint32_t original_board_sum = 0;
    uint32_t threshold_bits = kDefaultThresholdBits;
    uint64_t lut_signature = 0;
    uint8_t physical_transform = 0;
    uint8_t inverse_physical_transform = 0;
    uint64_t logical_pattern_signature = 0;
    uint64_t physical_pattern_signature = 0;
    uint64_t live_board_count = 0;
};

class LayerSlotReader {
public:
    LayerSlotReader(const std::string &path, FileIOUtils::DirectIoConfig config = {});
    ~LayerSlotReader();

    LayerSlotReader(const LayerSlotReader &) = delete;
    LayerSlotReader &operator=(const LayerSlotReader &) = delete;
    LayerSlotReader(LayerSlotReader &&) noexcept;
    LayerSlotReader &operator=(LayerSlotReader &&) noexcept;

    [[nodiscard]] const LayerFileInfo &info() const;
    [[nodiscard]] size_t next_slot() const;
    bool read_next(BoardSet &set);
    void close();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

void write_lut_file(
    const std::string &path,
    const Luts &luts,
    FileIOUtils::DirectIoConfig config = {}
);

Luts read_lut_file(
    const std::string &path,
    FileIOUtils::DirectIoConfig config = {}
);

void write_layer_file(
    const std::string &path,
    const Layer &layer,
    FileIOUtils::DirectIoConfig config = {},
    bool compressed_archive = false
);

Layer read_layer_file(
    const std::string &path,
    FileIOUtils::DirectIoConfig config = {}
);

} // namespace EXAD
