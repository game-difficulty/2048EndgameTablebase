#include "BCPositionFile.h"
#include "BCPositionScanner.h"
#include "BCSuccessIO.h"
#include "FormationRuntime.h"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

struct Args {
    std::filesystem::path position_path;
    std::filesystem::path success_path;
    uint32_t target_rank = 8U;
};

[[nodiscard]] std::string require_value(int argc, char **argv, int &i, const char *flag) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string(flag) + " requires a value");
    }
    ++i;
    return argv[i];
}

[[nodiscard]] Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--position") {
            args.position_path = require_value(argc, argv, i, "--position");
        } else if (key == "--success") {
            args.success_path = require_value(argc, argv, i, "--success");
        } else if (key == "--target-rank") {
            args.target_rank = static_cast<uint32_t>(
                std::stoul(require_value(argc, argv, i, "--target-rank"))
            );
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.position_path.empty()) {
        throw std::invalid_argument("--position is required");
    }
    if (args.success_path.empty()) {
        throw std::invalid_argument("--success is required");
    }
    return args;
}

[[nodiscard]] std::vector<uint8_t> make_free_legal_tiles(uint32_t target_rank) {
    std::vector<uint8_t> legal_tiles;
    legal_tiles.reserve(static_cast<size_t>(target_rank) + 2U);
    for (uint32_t tile = 0U; tile <= target_rank; ++tile) {
        legal_tiles.push_back(static_cast<uint8_t>(tile));
    }
    legal_tiles.push_back(15U);
    return legal_tiles;
}

template <typename T>
[[nodiscard]] std::string value_to_string(T value) {
    std::ostringstream out;
    if constexpr (std::is_floating_point_v<T>) {
        out << std::setprecision(std::numeric_limits<T>::max_digits10) << value;
    } else {
        out << value;
    }
    return out.str();
}

template <typename StorageT>
void scan_typed(
    const BC::BCPositionLayerReader &position,
    const BC::BCSuccessLayerReader &success
) {
    const StorageT zero = BC::bc_success_zero_value_for_dtype<StorageT>(success.dtype_mode());
    const StorageT terminal =
        BC::bc_success_terminal_value_for_dtype<StorageT>(success.dtype_mode());
    StorageT max_value = zero;
    bool have_value = false;
    uint64_t rows = 0U;
    uint64_t max_count = 0U;
    BC::CellId first_max_cell = 0U;
    uint32_t first_max_row = 0U;
    uint64_t first_max_board = 0U;

    for (BC::CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const BC::BCPositionCellDescriptor &desc = position.descriptor(cid);
        if (desc.empty() || desc.success_rows == 0U) {
            continue;
        }
        for (uint32_t row = 0U; row < desc.success_rows; ++row) {
            const StorageT value = success.read_value_typed<StorageT>(cid, row);
            ++rows;
            if (!have_value || value > max_value) {
                have_value = true;
                max_value = value;
                max_count = 1U;
                first_max_cell = cid;
                first_max_row = row;
            } else if (value == max_value) {
                ++max_count;
            }
        }
    }

    if (have_value) {
        bool found_board = false;
        BC::BCPositionCellScanner scanner(position, first_max_cell);
        scanner.for_each_board([&](const BC::BCScannedBoardEntry &entry) {
            if (!found_board && entry.local_success_row == first_max_row) {
                first_max_board = entry.board;
                found_board = true;
            }
        });
        if (!found_board) {
            throw std::runtime_error("failed to locate first max board");
        }
    }

    std::cout << std::setprecision(15)
              << "rows=" << rows
              << " max_value=" << value_to_string(max_value)
              << " max_rate="
              << RuntimeControls::normalized_success_value(max_value, zero, terminal)
              << " max_count=" << max_count
              << " first_max_cell=" << first_max_cell
              << " first_max_row=" << first_max_row
              << " first_max_board=0x" << std::hex << first_max_board << std::dec
              << '\n';
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Args args = parse_args(argc, argv);
        const BC::BCLut lut(make_free_legal_tiles(args.target_rank));
        const BC::BCPositionFileReader position =
            BC::BCPositionFileReader::open_buffered(args.position_path, lut);
        BC::BCBufferedFileReader success_file(args.success_path);
        std::vector<uint8_t> header_bytes(BC::kBCSuccessHeaderBytes, 0U);
        success_file.read_at(0U, header_bytes.data(), header_bytes.size());
        const BC::BCSuccessHeader header = BC::bc_read_success_header(header_bytes);
        const uint64_t logical_size = BC::bc_success_logical_size(header);
        if (logical_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC success logical size exceeds addressable memory");
        }
        std::vector<uint8_t> success_bytes(static_cast<size_t>(logical_size), 0U);
        if (!success_bytes.empty()) {
            success_file.read_at(0U, success_bytes.data(), logical_size);
        }
        const BC::BCSuccessLayerReader success(success_bytes, position.layer(), 1U);
        switch (success.dtype_mode()) {
            case BC::BCSuccessDTypeMode::UInt32:
                scan_typed<uint32_t>(position.layer(), success);
                break;
            case BC::BCSuccessDTypeMode::UInt64:
                scan_typed<uint64_t>(position.layer(), success);
                break;
            case BC::BCSuccessDTypeMode::Float32:
            case BC::BCSuccessDTypeMode::OneMinusFloat32:
                scan_typed<float>(position.layer(), success);
                break;
            case BC::BCSuccessDTypeMode::Float64:
            case BC::BCSuccessDTypeMode::OneMinusFloat64:
                scan_typed<double>(position.layer(), success);
                break;
        }
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
