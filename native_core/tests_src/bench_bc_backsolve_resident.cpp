#include "BCBacksolve.h"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::filesystem::path current_pos;
    std::filesystem::path future2_pos;
    std::filesystem::path future2_suc;
    std::filesystem::path future4_pos;
    std::filesystem::path future4_suc;
    std::filesystem::path output_suc;
    std::filesystem::path stats_csv;
    uint32_t target_rank = 11U;
    int success_target_rank = -1;
    int canonical_symm_mode = static_cast<int>(SymmMode::Full);
    double spawn_rate4 = 0.1;
    int num_threads = 0;
    uint32_t canonical_batch_size = 8192U;
};

[[nodiscard]] std::string require_value(
    int argc,
    char **argv,
    int &i,
    const char *flag
) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string(flag) + " requires a value");
    }
    ++i;
    return argv[i];
}

[[nodiscard]] int parse_symm_mode(const std::string &value) {
    if (value == "identity") {
        return static_cast<int>(SymmMode::Identity);
    }
    if (value == "full") {
        return static_cast<int>(SymmMode::Full);
    }
    if (value == "diagonal") {
        return static_cast<int>(SymmMode::Diagonal);
    }
    if (value == "horizontal") {
        return static_cast<int>(SymmMode::Horizontal);
    }
    if (value == "min33") {
        return static_cast<int>(SymmMode::Min33);
    }
    if (value == "min24") {
        return static_cast<int>(SymmMode::Min24);
    }
    if (value == "min34") {
        return static_cast<int>(SymmMode::Min34);
    }
    if (value == "min34top") {
        return static_cast<int>(SymmMode::Min34Top);
    }
    return std::stoi(value);
}

[[nodiscard]] std::vector<uint8_t> make_free_legal_tiles(uint32_t target_rank) {
    if (target_rank >= 15U) {
        throw std::invalid_argument("--target-rank must be < 15 for free BC LUT");
    }
    std::vector<uint8_t> legal_tiles;
    legal_tiles.reserve(static_cast<size_t>(target_rank) + 2U);
    for (uint32_t tile = 0U; tile <= target_rank; ++tile) {
        legal_tiles.push_back(static_cast<uint8_t>(tile));
    }
    legal_tiles.push_back(15U);
    return legal_tiles;
}

[[nodiscard]] std::vector<uint8_t> all_board_success_shifts() {
    std::vector<uint8_t> shifts;
    shifts.reserve(16U);
    for (uint8_t cell = 0U; cell < 16U; ++cell) {
        shifts.push_back(static_cast<uint8_t>(cell * 4U));
    }
    return shifts;
}

[[nodiscard]] Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--current-pos") {
            args.current_pos = require_value(argc, argv, i, "--current-pos");
        } else if (key == "--future2-pos") {
            args.future2_pos = require_value(argc, argv, i, "--future2-pos");
        } else if (key == "--future2-suc") {
            args.future2_suc = require_value(argc, argv, i, "--future2-suc");
        } else if (key == "--future4-pos") {
            args.future4_pos = require_value(argc, argv, i, "--future4-pos");
        } else if (key == "--future4-suc") {
            args.future4_suc = require_value(argc, argv, i, "--future4-suc");
        } else if (key == "--output-suc") {
            args.output_suc = require_value(argc, argv, i, "--output-suc");
        } else if (key == "--stats-csv") {
            args.stats_csv = require_value(argc, argv, i, "--stats-csv");
        } else if (key == "--target-rank") {
            args.target_rank = static_cast<uint32_t>(
                std::stoul(require_value(argc, argv, i, "--target-rank"))
            );
        } else if (key == "--success-target-rank") {
            args.success_target_rank = std::stoi(
                require_value(argc, argv, i, "--success-target-rank")
            );
        } else if (key == "--canonical-symm-mode") {
            args.canonical_symm_mode = parse_symm_mode(
                require_value(argc, argv, i, "--canonical-symm-mode")
            );
        } else if (key == "--spawn-rate4") {
            args.spawn_rate4 = std::stod(require_value(argc, argv, i, "--spawn-rate4"));
        } else if (key == "--threads") {
            args.num_threads = std::stoi(require_value(argc, argv, i, "--threads"));
        } else if (key == "--batch-size") {
            args.canonical_batch_size = static_cast<uint32_t>(
                std::stoul(require_value(argc, argv, i, "--batch-size"))
            );
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.current_pos.empty()) {
        throw std::invalid_argument("--current-pos is required");
    }
    if (args.future2_pos.empty()) {
        throw std::invalid_argument("--future2-pos is required");
    }
    if (args.future2_suc.empty()) {
        throw std::invalid_argument("--future2-suc is required");
    }
    if (args.future4_pos.empty()) {
        throw std::invalid_argument("--future4-pos is required");
    }
    if (args.future4_suc.empty()) {
        throw std::invalid_argument("--future4-suc is required");
    }
    if (args.output_suc.empty()) {
        throw std::invalid_argument("--output-suc is required");
    }
    if (args.success_target_rank < 0) {
        args.success_target_rank = static_cast<int>(args.target_rank);
    }
    return args;
}

void write_stats_csv(
    const std::filesystem::path &path,
    const BC::BCPositionLayerReader &current,
    const BC::BCPositionLayerReader &future2,
    const BC::BCPositionLayerReader &future4,
    uint64_t output_bytes,
    const BC::BCBacksolveStats &stats
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open --stats-csv: " + path.string());
    }
    out
        << "route,current_layer_sum,future2_layer_sum,future4_layer_sum,"
        << "current_rows,queries2,queries4,found2,found4,terminal_success_rows,"
        << "future_index_seconds,recalc_seconds,write_seconds,recalc_mbps,output_bytes\n";
    out
        << "resident,"
        << current.header().layer_sum << ','
        << future2.header().layer_sum << ','
        << future4.header().layer_sum << ','
        << stats.current_rows << ','
        << stats.queries2 << ','
        << stats.queries4 << ','
        << stats.found2 << ','
        << stats.found4 << ','
        << stats.terminal_success_rows << ','
        << stats.future_index_seconds << ','
        << stats.recalc_seconds << ','
        << stats.write_seconds << ','
        << stats.recalc_mbps() << ','
        << output_bytes << '\n';
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Args args = parse_args(argc, argv);
        const BC::BCLut lut(make_free_legal_tiles(args.target_rank));
        const std::vector<uint8_t> success_shifts = all_board_success_shifts();

        BC::BCPositionFileReader current =
            BC::BCPositionFileReader::open_buffered(args.current_pos, lut);
        BC::BCPositionFileReader future2 =
            BC::BCPositionFileReader::open_buffered(args.future2_pos, lut);
        BC::BCPositionFileReader future4 =
            BC::BCPositionFileReader::open_buffered(args.future4_pos, lut);
        const BC::BCSuccessFileReader future2_success =
            BC::BCSuccessFileReader::open_buffered(args.future2_suc, future2.layer(), 1U);
        const BC::BCSuccessFileReader future4_success =
            BC::BCSuccessFileReader::open_buffered(args.future4_suc, future4.layer(), 1U);

        BC::BCBacksolveOptions options;
        options.num_threads = args.num_threads;
        options.canonical_batch_size = args.canonical_batch_size;
        options.canonical_symm_mode = args.canonical_symm_mode;
        options.spawn_rate4 = args.spawn_rate4;
        options.success_target_rank = args.success_target_rank;
        options.success_shifts = &success_shifts;

        const BC::BCBacksolveStats stats = BC::backsolve_resident_layer_to_file(
            args.output_suc,
            lut,
            current.layer(),
            future2.layer(),
            future2_success.reader(),
            future4.layer(),
            future4_success.reader(),
            options
        );

        std::error_code ec;
        const uint64_t output_bytes = std::filesystem::file_size(args.output_suc, ec);
        if (ec) {
            throw std::runtime_error("failed to stat output success file: " + ec.message());
        }
        if (!args.stats_csv.empty()) {
            write_stats_csv(
                args.stats_csv,
                current.layer(),
                future2.layer(),
                future4.layer(),
                output_bytes,
                stats
            );
        }

        std::cout
            << "route=resident"
            << " current_rows=" << stats.current_rows
            << " recalc_seconds=" << stats.recalc_seconds
            << " recalc_mbps=" << stats.recalc_mbps()
            << " future_index_seconds=" << stats.future_index_seconds
            << " write_seconds=" << stats.write_seconds
            << " output_bytes=" << output_bytes
            << '\n';
    } catch (const std::exception &ex) {
        std::cerr << "bc_backsolve_resident_bench failed: " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
