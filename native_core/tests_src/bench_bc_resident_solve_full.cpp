#include "BCPositionScanner.h"
#include "BCDirectFileIO.h"
#include "BCResidentSolve.h"
#include "BCSuccessIO.h"
#include "FormationRuntime.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

struct Args {
    std::filesystem::path generated_position_dir = "tmp/free9_256_resident_generated";
    std::filesystem::path solved_output_dir = "tmp/free9_256_resident_solve_optimized_outputs";
    std::filesystem::path stats_csv = "tmp/free9_256_resident_solve_optimized_outputs_stats.csv";
    std::filesystem::path summary_csv = "tmp/free9_256_resident_solve_optimized_outputs_summary.csv";
    std::string prefix = "free9_256_";
    uint32_t target_rank = 8U;
    int success_target_rank = -1;
    int canonical_symm_mode = static_cast<int>(SymmMode::Full);
    double spawn_rate4 = 0.1;
    int num_threads = 0;
    uint32_t canonical_batch_size = 8192U;
    uint32_t inspect_layer_ordinal = 1U;
    bool direct_io = false;
    uint32_t direct_queue_depth = 16U;
    bool keep_direct_padding = false;
    bool buffered_read = false;
    bool compact_zero = true;
    double deletion_threshold = 0.0;
    double relative_deletion_threshold = 0.0;
    std::string deletion_threshold_signal_path;
    BC::BCSuccessDTypeMode success_dtype = BC::BCSuccessDTypeMode::UInt32;
};

struct LayerFile {
    uint32_t ordinal = 0U;
    std::filesystem::path path;
};

struct LayerMetric {
    uint32_t ordinal = 0U;
    uint64_t layer_sum = 0U;
    uint64_t current_rows = 0U;
    uint64_t live_rows = 0U;
    uint64_t zero_pruned_rows = 0U;
    uint64_t position_bytes = 0U;
    uint64_t success_bytes = 0U;
    uint64_t queries2 = 0U;
    uint64_t queries4 = 0U;
    uint64_t found2 = 0U;
    uint64_t found4 = 0U;
    uint64_t terminal_success_rows = 0U;
    uint64_t archive_live_rows = 0U;
    uint64_t threshold_pruned_rows = 0U;
    double read_seconds = 0.0;
    double index_seconds = 0.0;
    double recalc_seconds = 0.0;
    double compact_seconds = 0.0;
    double archive_compact_seconds = 0.0;
    double position_write_seconds = 0.0;
    double success_write_seconds = 0.0;
    double total_seconds = 0.0;
};

struct InspectResult {
    uint32_t ordinal = 0U;
    uint64_t layer_sum = 0U;
    uint64_t rows = 0U;
    std::string max_value = "0";
    double max_rate = 0.0;
    uint64_t max_count = 0U;
    uint64_t first_max_board = 0U;
    BC::CellId first_max_cell = 0U;
    uint32_t first_max_row = 0U;
};

struct LayerWriteResult {
    uint32_t ordinal = 0U;
    uint64_t position_bytes = 0U;
    uint64_t success_bytes = 0U;
    double position_write_seconds = 0.0;
    double success_write_seconds = 0.0;
};

[[nodiscard]] double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

[[nodiscard]] std::string require_value(int argc, char **argv, int &i, const char *flag) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string(flag) + " requires a value");
    }
    ++i;
    return argv[i];
}

[[nodiscard]] BC::BCSuccessDTypeMode parse_success_dtype(const std::string &value) {
    if (value == "uint32") {
        return BC::BCSuccessDTypeMode::UInt32;
    }
    if (value == "uint64") {
        return BC::BCSuccessDTypeMode::UInt64;
    }
    if (value == "float32") {
        return BC::BCSuccessDTypeMode::Float32;
    }
    if (value == "float64") {
        return BC::BCSuccessDTypeMode::Float64;
    }
    if (value == "one-minus-float32" || value == "1-float32") {
        return BC::BCSuccessDTypeMode::OneMinusFloat32;
    }
    if (value == "one-minus-float64" || value == "1-float64") {
        return BC::BCSuccessDTypeMode::OneMinusFloat64;
    }
    throw std::invalid_argument("unsupported success dtype: " + value);
}

[[nodiscard]] std::string success_dtype_name(BC::BCSuccessDTypeMode dtype) {
    switch (dtype) {
        case BC::BCSuccessDTypeMode::UInt32:
            return "uint32";
        case BC::BCSuccessDTypeMode::UInt64:
            return "uint64";
        case BC::BCSuccessDTypeMode::Float32:
            return "float32";
        case BC::BCSuccessDTypeMode::Float64:
            return "float64";
        case BC::BCSuccessDTypeMode::OneMinusFloat32:
            return "one-minus-float32";
        case BC::BCSuccessDTypeMode::OneMinusFloat64:
            return "one-minus-float64";
    }
    throw std::invalid_argument("unsupported success dtype");
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

[[nodiscard]] Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--position-dir" || key == "--generated-position-dir") {
            args.generated_position_dir = require_value(argc, argv, i, key.c_str());
        } else if (key == "--output-dir" || key == "--solved-output-dir") {
            args.solved_output_dir = require_value(argc, argv, i, key.c_str());
        } else if (key == "--stats-csv") {
            args.stats_csv = require_value(argc, argv, i, "--stats-csv");
        } else if (key == "--summary-csv") {
            args.summary_csv = require_value(argc, argv, i, "--summary-csv");
        } else if (key == "--prefix") {
            args.prefix = require_value(argc, argv, i, "--prefix");
        } else if (key == "--target-rank") {
            args.target_rank = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, "--target-rank")));
        } else if (key == "--success-target-rank") {
            args.success_target_rank = std::stoi(require_value(argc, argv, i, "--success-target-rank"));
        } else if (key == "--success-dtype" || key == "--dtype") {
            args.success_dtype = parse_success_dtype(require_value(argc, argv, i, key.c_str()));
        } else if (key == "--canonical-symm-mode") {
            args.canonical_symm_mode = parse_symm_mode(require_value(argc, argv, i, "--canonical-symm-mode"));
        } else if (key == "--spawn-rate4") {
            args.spawn_rate4 = std::stod(require_value(argc, argv, i, "--spawn-rate4"));
        } else if (key == "--threads") {
            args.num_threads = std::stoi(require_value(argc, argv, i, "--threads"));
        } else if (key == "--batch-size") {
            args.canonical_batch_size = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, "--batch-size")));
        } else if (key == "--inspect-layer-ordinal") {
            args.inspect_layer_ordinal = static_cast<uint32_t>(
                std::stoul(require_value(argc, argv, i, "--inspect-layer-ordinal"))
            );
        } else if (key == "--direct-io") {
            args.direct_io = true;
        } else if (key == "--direct-queue-depth") {
            args.direct_queue_depth = static_cast<uint32_t>(
                std::stoul(require_value(argc, argv, i, "--direct-queue-depth"))
            );
        } else if (key == "--keep-direct-padding") {
            args.keep_direct_padding = true;
        } else if (key == "--buffered-read") {
            args.buffered_read = true;
        } else if (key == "--compact-zero") {
            args.compact_zero = true;
        } else if (key == "--no-compact-zero") {
            args.compact_zero = false;
        } else if (key == "--deletion-threshold") {
            args.deletion_threshold = std::stod(require_value(argc, argv, i, "--deletion-threshold"));
        } else if (key == "--relative-deletion-threshold") {
            args.relative_deletion_threshold =
                std::stod(require_value(argc, argv, i, "--relative-deletion-threshold"));
        } else if (key == "--deletion-threshold-signal") {
            args.deletion_threshold_signal_path =
                require_value(argc, argv, i, "--deletion-threshold-signal");
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.success_target_rank < 0) {
        args.success_target_rank = static_cast<int>(args.target_rank);
    }
    if (args.direct_queue_depth == 0U) {
        throw std::invalid_argument("--direct-queue-depth must be non-zero");
    }
    if (!args.compact_zero) {
        throw std::invalid_argument("optimized resident bench requires --compact-zero in this version");
    }
    return args;
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

[[nodiscard]] bool board_has_target_rank(
    uint64_t board,
    int target_rank,
    const std::vector<uint8_t> &success_shifts
) {
    const uint64_t target = static_cast<uint64_t>(target_rank);
    for (uint8_t shift : success_shifts) {
        if (((board >> shift) & 0xFULL) == target) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::map<uint32_t, LayerFile> discover_layers(const Args &args) {
    if (!std::filesystem::is_directory(args.generated_position_dir)) {
        throw std::runtime_error("position dir does not exist: " + args.generated_position_dir.string());
    }
    const std::regex pattern("^" + args.prefix + "([0-9]+)\\.bcpos$");
    std::map<uint32_t, LayerFile> layers;
    for (const std::filesystem::directory_entry &entry :
         std::filesystem::directory_iterator(args.generated_position_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        std::smatch match;
        if (!std::regex_match(name, match, pattern)) {
            continue;
        }
        const uint32_t ordinal = static_cast<uint32_t>(std::stoul(match[1].str()));
        layers.emplace(ordinal, LayerFile{ordinal, entry.path()});
    }
    if (layers.empty()) {
        throw std::runtime_error("no position files found with prefix: " + args.prefix);
    }
    uint32_t expected = layers.begin()->first;
    for (const auto &[ordinal, layer] : layers) {
        (void)layer;
        if (ordinal != expected) {
            throw std::runtime_error("position layer ordinals are not contiguous at " + std::to_string(expected));
        }
        ++expected;
    }
    return layers;
}

[[nodiscard]] std::filesystem::path position_path_for(const Args &args, uint32_t ordinal) {
    return args.solved_output_dir / (args.prefix + std::to_string(ordinal) + ".bcpos");
}

[[nodiscard]] std::filesystem::path success_path_for(const Args &args, uint32_t ordinal) {
    return args.solved_output_dir / (args.prefix + std::to_string(ordinal) + ".bcsuc");
}

[[nodiscard]] BC::BCSuccessFileReader open_success_for_inspect(
    const Args &args,
    const std::filesystem::path &path,
    const BC::BCPositionLayerReader &position
) {
    if (args.direct_io && args.keep_direct_padding) {
        BC::BCBufferedFileReader probe(path);
        std::vector<uint8_t> header_bytes(BC::kBCSuccessHeaderBytes);
        probe.read_at(0U, header_bytes.data(), header_bytes.size());
        const BC::BCSuccessHeader header = BC::bc_read_success_header(header_bytes);
        const uint64_t logical_size = BC::bc_checked_add_u64(
            header.payload_offset,
            header.payload_bytes,
            "BC success inspect logical size overflow"
        );
        BC::BCDirectFileIOOptions options;
        options.queue_depth = args.direct_queue_depth;
        options.overlapped = args.direct_queue_depth > 1U;
        options.logical_size = logical_size;
        return BC::BCSuccessFileReader(
            std::make_unique<BC::BCDirectFileReader>(path, options),
            position,
            1U
        );
    }
    return BC::BCSuccessFileReader::open_buffered(path, position, 1U);
}

[[nodiscard]] std::vector<uint8_t> make_terminal_success_bytes(
    const BC::BCPositionLayerReader &position,
    const std::vector<uint8_t> &success_shifts,
    int target_rank
) {
    BC::BCSuccessLayerWriter writer;
    writer.begin_layer(position, 1U, BC::BCSuccessDTypeMode::UInt32);
    for (BC::CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const BC::BCPositionCellDescriptor &desc = position.descriptor(cid);
        if (desc.empty() || desc.success_rows == 0U) {
            writer.mark_empty_cell(cid);
            continue;
        }
        std::vector<uint32_t> values(desc.success_rows, 0U);
        BC::BCPositionCellScanner(position, cid).for_each_board(
            [&](const BC::BCScannedBoardEntry &entry) {
                if (board_has_target_rank(entry.board, target_rank, success_shifts)) {
                    values[entry.local_success_row] = max_scale_value<uint32_t>();
                }
            }
        );
        writer.write_cell(cid, values);
    }
    return writer.finish_layer();
}

template <typename StorageT>
[[nodiscard]] BC::BCResidentSolvedLayer<StorageT> make_terminal_solved_layer(
    const BC::BCPositionLayerReader &position,
    const std::vector<uint8_t> &success_shifts,
    int target_rank,
    BC::BCSuccessDTypeMode dtype,
    int num_threads
) {
    const std::vector<uint64_t> offsets = BC::bc_resident_cell_value_offsets(position);
    if (offsets.back() > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("terminal BC layer row count exceeds size_t");
    }
    const StorageT zero = BC::bc_success_zero_value_for_dtype<StorageT>(dtype);
    const StorageT terminal = BC::bc_success_terminal_value_for_dtype<StorageT>(dtype);
    std::vector<StorageT> values(static_cast<size_t>(offsets.back()), zero);
    for (BC::CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const BC::BCPositionCellDescriptor &desc = position.descriptor(cid);
        if (desc.empty() || desc.success_rows == 0U) {
            continue;
        }
        const uint64_t cell_base = offsets[static_cast<size_t>(cid)];
        BC::BCPositionCellScanner(position, cid).for_each_board(
            [&](const BC::BCScannedBoardEntry &entry) {
                if (board_has_target_rank(entry.board, target_rank, success_shifts)) {
                    values[static_cast<size_t>(cell_base + entry.local_success_row)] =
                        terminal;
                }
            }
        );
    }
    BC::BCResidentRawSolveResult<StorageT> raw;
    raw.values = std::move(values);
    raw.cell_value_offsets = offsets;
    return BC::bc_resident_compact_zero_in_place<StorageT>(
        position,
        raw,
        position.lut(),
        1U,
        dtype,
        zero,
        num_threads
    );
}

[[nodiscard]] BC::BCResidentUInt32SolvedLayer make_terminal_raw_solved_layer(
    BC::BCPositionLayerReader position,
    const std::vector<uint8_t> &success_shifts,
    int target_rank
) {
    const std::vector<uint64_t> offsets = BC::bc_resident_cell_value_offsets(position);
    if (offsets.back() > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("terminal BC layer row count exceeds size_t");
    }
    std::vector<uint32_t> values(static_cast<size_t>(offsets.back()), 0U);
    for (BC::CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const BC::BCPositionCellDescriptor &desc = position.descriptor(cid);
        if (desc.empty() || desc.success_rows == 0U) {
            continue;
        }
        const uint64_t cell_base = offsets[static_cast<size_t>(cid)];
        BC::BCPositionCellScanner(position, cid).for_each_board(
            [&](const BC::BCScannedBoardEntry &entry) {
                if (board_has_target_rank(entry.board, target_rank, success_shifts)) {
                    values[static_cast<size_t>(cell_base + entry.local_success_row)] =
                        max_scale_value<uint32_t>();
                }
            }
        );
    }

    BC::BCResidentUInt32SolvedLayer out;
    out.open(
        std::move(position),
        std::move(values),
        1U
    );
    return out;
}

[[nodiscard]] std::vector<uint8_t> make_empty_position_bytes(uint64_t layer_sum, uint32_t family_unit) {
    const BC::BCFamilyTable axis(
        layer_sum,
        static_cast<uint16_t>(family_unit),
        std::vector<BC::FamilyCoord>{0U}
    );
    BC::BCPositionLayerWriter writer;
    writer.begin_layer(axis);
    writer.mark_empty_cell(0U);
    return writer.finish_layer();
}

[[nodiscard]] BC::BCPositionFileReader open_position_layer(
    const Args &args,
    const LayerFile &layer,
    const BC::BCLut &lut
) {
    if (args.direct_io && !args.buffered_read) {
        return BC::BCPositionFileReader::open_direct_auto(
            layer.path,
            lut,
            args.direct_queue_depth,
            args.direct_queue_depth > 1U
        );
    }
    return BC::BCPositionFileReader::open_buffered(layer.path, lut);
}

[[nodiscard]] std::unique_ptr<BC::BCWritableFile> make_output_writer(
    const Args &args,
    const std::filesystem::path &path,
    uint64_t logical_size
) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    if (args.direct_io) {
        BC::BCDirectFileIOOptions options;
        options.queue_depth = args.direct_queue_depth;
        options.overlapped = args.direct_queue_depth > 1U;
        options.logical_size = logical_size;
        return std::make_unique<BC::BCDirectFileWriter>(path, options);
    }
    (void)logical_size;
    return std::make_unique<BC::BCBufferedFileWriter>(path);
}

template <typename StorageT>
uint64_t write_position_layer_file(
    const Args &args,
    uint32_t ordinal,
    const BC::BCResidentSolvedLayer<StorageT> &layer
) {
    const std::vector<uint8_t> &bytes = layer.position.bytes();
    const uint64_t logical_size = static_cast<uint64_t>(bytes.size());
    const std::filesystem::path path = position_path_for(args, ordinal);
    std::unique_ptr<BC::BCWritableFile> file =
        make_output_writer(args, path, logical_size);
    file->prepare_full_overwrite(logical_size);
    BC::BCSequentialSuccessWriteStager stager(*file);
    stager.append(bytes.data(), logical_size);
    stager.finish();
    file.reset();
    if (args.direct_io && !args.keep_direct_padding) {
        std::filesystem::resize_file(path, logical_size);
    }
    return logical_size;
}

template <typename StorageT>
uint64_t write_success_layer_file(
    const Args &args,
    uint32_t ordinal,
    const BC::BCResidentSolvedLayer<StorageT> &layer
) {
    const uint64_t logical_size =
        BC::kBCSuccessHeaderBytes +
        static_cast<uint64_t>(layer.success_values.size()) *
            BC::bc_success_dtype_value_size(layer.dtype);
    const std::filesystem::path path = success_path_for(args, ordinal);
    std::unique_ptr<BC::BCWritableFile> file =
        make_output_writer(args, path, logical_size);
    const uint64_t written = BC::write_success_values_to_file(
        *file,
        layer.position,
        1U,
        layer.dtype,
        layer.success_values
    );
    file.reset();
    if (args.direct_io && !args.keep_direct_padding) {
        std::filesystem::resize_file(path, written);
    }
    return written;
}

template <typename StorageT>
LayerWriteResult write_solved_layer_files(
    const Args &args,
    uint32_t ordinal,
    const BC::BCResidentSolvedLayer<StorageT> &layer
) {
    LayerWriteResult result;
    result.ordinal = ordinal;
    const double position_t0 = now_seconds();
    result.position_bytes = write_position_layer_file(args, ordinal, layer);
    result.position_write_seconds = now_seconds() - position_t0;
    const double success_t0 = now_seconds();
    result.success_bytes = write_success_layer_file(args, ordinal, layer);
    result.success_write_seconds = now_seconds() - success_t0;
    return result;
}

void write_checkpoint_manifest(
    const Args &args,
    int64_t next_ordinal,
    int64_t exact_future2_ordinal,
    int64_t exact_future4_ordinal
) {
    const std::filesystem::path path =
        args.solved_output_dir / (args.prefix + "resident_checkpoint.csv");
    const std::filesystem::path tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp);
        if (!out) {
            throw std::runtime_error("failed to open checkpoint manifest: " + tmp.string());
        }
        out << "next_ordinal,exact_future2_ordinal,exact_future4_ordinal\n"
            << next_ordinal << ','
            << exact_future2_ordinal << ','
            << exact_future4_ordinal << '\n';
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        ec.clear();
        std::filesystem::rename(tmp, path, ec);
        if (ec) {
            throw std::runtime_error("failed to publish checkpoint manifest: " + ec.message());
        }
    }
}

void write_stats_header(std::ofstream &out) {
    out
        << "ordinal,layer_sum,current_rows,live_rows,zero_pruned_rows,position_bytes,success_bytes,"
        << "queries2,queries4,found2,found4,terminal_success_rows,archive_live_rows,"
        << "threshold_pruned_rows,read_seconds,index_seconds,recalc_seconds,compact_seconds,"
        << "archive_compact_seconds,position_write_seconds,success_write_seconds,total_seconds,"
        << "recalc_mrows_per_sec,total_mrows_per_sec\n";
}

void write_metric_row(std::ofstream &out, const LayerMetric &m) {
    const double recalc_mrows = m.recalc_seconds > 0.0
        ? static_cast<double>(m.current_rows) / m.recalc_seconds / 1.0e6
        : 0.0;
    const double total_mrows = m.total_seconds > 0.0
        ? static_cast<double>(m.current_rows) / m.total_seconds / 1.0e6
        : 0.0;
    out
        << m.ordinal << ','
        << m.layer_sum << ','
        << m.current_rows << ','
        << m.live_rows << ','
        << m.zero_pruned_rows << ','
        << m.position_bytes << ','
        << m.success_bytes << ','
        << m.queries2 << ','
        << m.queries4 << ','
        << m.found2 << ','
        << m.found4 << ','
        << m.terminal_success_rows << ','
        << m.archive_live_rows << ','
        << m.threshold_pruned_rows << ','
        << m.read_seconds << ','
        << m.index_seconds << ','
        << m.recalc_seconds << ','
        << m.compact_seconds << ','
        << m.archive_compact_seconds << ','
        << m.position_write_seconds << ','
        << m.success_write_seconds << ','
        << m.total_seconds << ','
        << recalc_mrows << ','
        << total_mrows << '\n';
}

template <typename StorageT>
[[nodiscard]] InspectResult inspect_layer_max(
    const Args &args,
    const BC::BCLut &lut,
    uint32_t ordinal
) {
    if (!BC::bc_success_dtype_matches_type<StorageT>(args.success_dtype)) {
        throw std::invalid_argument("inspect storage type does not match success dtype");
    }
    const std::filesystem::path pos_path = position_path_for(args, ordinal);
    const std::filesystem::path suc_path = success_path_for(args, ordinal);
    BC::BCPositionFileReader position = BC::BCPositionFileReader::open_buffered(pos_path, lut);
    const BC::BCSuccessFileReader success =
        open_success_for_inspect(args, suc_path, position.layer());
    if (success.reader().dtype_mode() != args.success_dtype) {
        throw std::runtime_error("inspect success dtype mismatch");
    }
    InspectResult result;
    result.ordinal = ordinal;
    result.layer_sum = position.layer().header().layer_sum;
    const StorageT zero = BC::bc_success_zero_value_for_dtype<StorageT>(args.success_dtype);
    const StorageT terminal =
        BC::bc_success_terminal_value_for_dtype<StorageT>(args.success_dtype);
    StorageT max_value = zero;
    bool have_value = false;
    for (BC::CellId cid = 0U; cid < position.layer().cell_count(); ++cid) {
        const BC::BCPositionCellDescriptor &desc = position.layer().descriptor(cid);
        if (desc.empty() || desc.success_rows == 0U) {
            continue;
        }
        BC::BCPositionCellScanner(position.layer(), cid).for_each_board(
            [&](const BC::BCScannedBoardEntry &entry) {
                ++result.rows;
                const StorageT value =
                    success.reader().read_value_typed<StorageT>(cid, entry.local_success_row);
                if (!have_value || value > max_value) {
                    have_value = true;
                    max_value = value;
                    result.max_count = 1U;
                    result.first_max_board = entry.board;
                    result.first_max_cell = cid;
                    result.first_max_row = entry.local_success_row;
                } else if (value == max_value) {
                    ++result.max_count;
                }
            }
        );
    }
    result.max_value = value_to_string(max_value);
    result.max_rate = RuntimeControls::normalized_success_value(max_value, zero, terminal);
    return result;
}

template <typename StorageT>
int run_typed(const Args &args) {
        if (!BC::bc_success_dtype_matches_type<StorageT>(args.success_dtype)) {
            throw std::invalid_argument("run storage type does not match success dtype");
        }
        const BC::BCLut lut(make_free_legal_tiles(args.target_rank));
        const std::vector<uint8_t> success_shifts = all_board_success_shifts();
        const std::map<uint32_t, LayerFile> layers = discover_layers(args);
        const uint32_t min_ordinal = layers.begin()->first;
        const uint32_t max_ordinal = layers.rbegin()->first;
        if (min_ordinal != 0U || max_ordinal < 2U) {
            throw std::runtime_error("expected free9 position ordinals to start at 0 and include at least 3 layers");
        }
        if (args.inspect_layer_ordinal > max_ordinal) {
            throw std::runtime_error("--inspect-layer-ordinal exceeds discovered layer range");
        }

        std::filesystem::create_directories(args.solved_output_dir);
        if (!args.stats_csv.parent_path().empty()) {
            std::filesystem::create_directories(args.stats_csv.parent_path());
        }
        if (!args.summary_csv.parent_path().empty()) {
            std::filesystem::create_directories(args.summary_csv.parent_path());
        }

        std::ofstream stats_out(args.stats_csv);
        if (!stats_out) {
            throw std::runtime_error("failed to open stats csv: " + args.stats_csv.string());
        }
        stats_out << std::setprecision(9);
        write_stats_header(stats_out);

        std::vector<LayerMetric> metrics;
        metrics.reserve(max_ordinal + 1U);
        std::map<uint32_t, size_t> metric_index_by_ordinal;
        auto record_metric = [&](const LayerMetric &metric) {
            metric_index_by_ordinal[metric.ordinal] = metrics.size();
            metrics.push_back(metric);
        };

        RunOptions deletion_options;
        deletion_options.deletion_threshold = args.deletion_threshold;
        deletion_options.relative_deletion_threshold = args.relative_deletion_threshold;
        deletion_options.deletion_threshold_signal_path = args.deletion_threshold_signal_path;
        RuntimeControls::DeletionThresholdState deletion_state =
            RuntimeControls::current_deletion_thresholds(deletion_options);

        auto record_sync_write = [&](uint32_t ordinal, const BC::BCResidentSolvedLayer<StorageT> &layer) {
            LayerWriteResult write = write_solved_layer_files(args, ordinal, layer);
            const auto it = metric_index_by_ordinal.find(write.ordinal);
            if (it == metric_index_by_ordinal.end()) {
                throw std::runtime_error("BC optimized bench write completed for unknown ordinal");
            }
            LayerMetric &metric = metrics[it->second];
            metric.position_bytes = write.position_bytes;
            metric.success_bytes = write.success_bytes;
            metric.position_write_seconds += write.position_write_seconds;
            metric.success_write_seconds += write.success_write_seconds;
            metric.total_seconds += write.position_write_seconds + write.success_write_seconds;
        };

        auto archive_retired_layer_prune_and_write_if_needed =
            [&](int64_t ordinal_signed, BC::BCResidentSolvedLayer<StorageT> &layer) {
            if (ordinal_signed < static_cast<int64_t>(min_ordinal) ||
                ordinal_signed > static_cast<int64_t>(max_ordinal)) {
                return;
            }
            const uint32_t ordinal = static_cast<uint32_t>(ordinal_signed);
            const auto it = metric_index_by_ordinal.find(ordinal);
            if (it == metric_index_by_ordinal.end()) {
                throw std::runtime_error("BC optimized bench archive prune saw unknown ordinal");
            }
            LayerMetric &metric = metrics[it->second];
            deletion_state = RuntimeControls::refresh_deletion_thresholds(
                deletion_options,
                deletion_state
            );
            if (!RuntimeControls::deletion_threshold_enabled(deletion_state)) {
                return;
            }
            const StorageT zero_value =
                BC::bc_success_zero_value_for_dtype<StorageT>(args.success_dtype);
            const StorageT terminal_value =
                BC::bc_success_terminal_value_for_dtype<StorageT>(args.success_dtype);
            const StorageT layer_max = layer.success_values.empty()
                ? zero_value
                : *std::max_element(layer.success_values.begin(), layer.success_values.end());
            const StorageT threshold = RuntimeControls::effective_deletion_threshold<StorageT>(
                layer_max,
                zero_value,
                terminal_value,
                deletion_state
            );
            const BC::BCResidentArchivePruneResult archive_prune =
                BC::bc_resident_archive_prune_if_threshold_enabled_in_place<StorageT>(
                    layer,
                    threshold,
                    args.num_threads
                );
            if (!archive_prune.pruned) {
                return;
            }
            const BC::BCResidentCompactStats &archive_stats = archive_prune.stats;
            metric.archive_live_rows = archive_stats.live_rows;
            metric.threshold_pruned_rows = archive_stats.zero_pruned_rows;
            metric.archive_compact_seconds += archive_stats.compact_seconds;
            metric.total_seconds += archive_stats.compact_seconds;
            record_sync_write(ordinal, layer);
        };

        const double solve_begin = now_seconds();

        BC::BCResidentSolvedLayer<StorageT> top_layer;
        uint64_t virtual_layer_sum = 0U;
        uint32_t virtual_family_unit = 0U;
        {
            const double top_begin = now_seconds();
            const double top_read_begin = now_seconds();
            BC::BCPositionFileReader top_position = open_position_layer(args, layers.at(max_ordinal), lut);
            const double top_read_seconds = now_seconds() - top_read_begin;
            const uint64_t top_layer_sum = top_position.layer().header().layer_sum;
            const uint32_t top_family_unit = top_position.layer().header().family_unit;
            top_layer =
                make_terminal_solved_layer<StorageT>(
                    top_position.layer(),
                    success_shifts,
                    args.success_target_rank,
                    args.success_dtype,
                    args.num_threads
                );

            LayerMetric top_metric;
            top_metric.ordinal = max_ordinal;
            top_metric.layer_sum = top_layer_sum;
            top_metric.read_seconds = top_read_seconds;
            top_metric.live_rows = top_layer.compact_stats.live_rows;
            top_metric.zero_pruned_rows = top_layer.compact_stats.zero_pruned_rows;
            top_metric.position_bytes = top_layer.position.bytes().size();
            top_metric.success_bytes =
                BC::kBCSuccessHeaderBytes +
                static_cast<uint64_t>(top_layer.success_values.size()) *
                    BC::bc_success_dtype_value_size(args.success_dtype);
            top_metric.compact_seconds = top_layer.compact_stats.compact_seconds;
            top_metric.total_seconds = now_seconds() - top_begin;
            record_metric(top_metric);
            record_sync_write(max_ordinal, top_layer);
            write_checkpoint_manifest(
                args,
                static_cast<int64_t>(max_ordinal) - 1,
                static_cast<int64_t>(max_ordinal),
                static_cast<int64_t>(max_ordinal) + 1
            );

            virtual_layer_sum = top_layer_sum + 2U;
            virtual_family_unit = top_family_unit;
        }

        std::vector<uint8_t> virtual_empty_position_bytes =
            make_empty_position_bytes(virtual_layer_sum, virtual_family_unit);
        BC::BCResidentSolvedLayer<StorageT> future4;
        future4.open(std::move(virtual_empty_position_bytes), {}, lut, 1U, args.success_dtype);
        BC::BCResidentSolvedLayer<StorageT> future2 = std::move(top_layer);
        int64_t future4_ordinal = static_cast<int64_t>(max_ordinal) + 1;
        int64_t future2_ordinal = static_cast<int64_t>(max_ordinal);

        for (int64_t ordinal_signed = static_cast<int64_t>(max_ordinal) - 1;
             ordinal_signed >= static_cast<int64_t>(min_ordinal);
             --ordinal_signed) {
            const uint32_t ordinal = static_cast<uint32_t>(ordinal_signed);
            const double layer_begin = now_seconds();
            BC::BCResidentSolveOptions<StorageT> options;
            options.num_threads = args.num_threads;
            options.row_width = 1U;
            options.set_dtype(args.success_dtype);
            options.edge_options.canonical_batch_size = args.canonical_batch_size;
            options.edge_options.canonical_symm_mode = args.canonical_symm_mode;
            options.edge_options.spawn_rate4 = args.spawn_rate4;
            options.edge_options.success_target_rank = args.success_target_rank;
            options.edge_options.success_shifts = &success_shifts;
            options.edge_options.success_check_all_cells = true;

            double read_seconds = 0.0;
            uint64_t current_layer_sum = 0U;
            BC::BCResidentLayerResult<StorageT> solve_result = [&]() {
                const double read_begin = now_seconds();
                BC::BCPositionFileReader current = open_position_layer(args, layers.at(ordinal), lut);
                read_seconds = now_seconds() - read_begin;
                current_layer_sum = current.layer().header().layer_sum;
                options.edge_options.future_cell_modulus = current.layer().header().family_count;
                return BC::bc_resident_solve_compacted_layer<StorageT>(
                    current.layer(),
                    future2,
                    future4,
                    options
                );
            }();
            BC::BCResidentSolvedLayer<StorageT> solved_layer = std::move(solve_result.layer);
            const BC::BCResidentCompactStats compact_stats = solved_layer.compact_stats;
            const uint64_t compact_position_bytes = solved_layer.position.bytes().size();
            const uint64_t compact_success_bytes =
                BC::kBCSuccessHeaderBytes +
                static_cast<uint64_t>(solved_layer.success_values.size()) *
                    BC::bc_success_dtype_value_size(args.success_dtype);
            const double layer_total = now_seconds() - layer_begin;

            LayerMetric metric;
            metric.ordinal = ordinal;
            metric.layer_sum = current_layer_sum;
            metric.current_rows = solve_result.solve_stats.current_rows;
            metric.live_rows = compact_stats.live_rows;
            metric.zero_pruned_rows = compact_stats.zero_pruned_rows;
            metric.position_bytes = compact_position_bytes;
            metric.success_bytes = compact_success_bytes;
            metric.queries2 = solve_result.solve_stats.queries2;
            metric.queries4 = solve_result.solve_stats.queries4;
            metric.found2 = solve_result.solve_stats.found2;
            metric.found4 = solve_result.solve_stats.found4;
            metric.terminal_success_rows = solve_result.solve_stats.terminal_success_rows;
            metric.read_seconds = read_seconds;
            metric.index_seconds = solve_result.solve_stats.future_index_seconds;
            metric.recalc_seconds = solve_result.solve_stats.recalc_seconds;
            metric.compact_seconds = compact_stats.compact_seconds;
            metric.total_seconds = layer_total;
            record_metric(metric);
            // This exact zero-compacted layer is the only form allowed to enter the future chain.
            record_sync_write(ordinal, solved_layer);

            std::cout
                << "ordinal=" << ordinal
                << " layer_sum=" << metric.layer_sum
                << " rows=" << metric.current_rows
                << " live_rows=" << metric.live_rows
                << " total_seconds=" << metric.total_seconds
                << " recalc_seconds=" << metric.recalc_seconds
                << " compact_seconds=" << metric.compact_seconds
                << " position_bytes=" << metric.position_bytes
                << " success_bytes=" << metric.success_bytes
                << '\n';

            if (future4_ordinal != ordinal_signed + 2) {
                throw std::runtime_error(
                    "BC optimized bench invariant failed: archive prune must target retired future4"
                );
            }
            archive_retired_layer_prune_and_write_if_needed(future4_ordinal, future4);
            write_checkpoint_manifest(
                args,
                ordinal_signed - 1,
                ordinal_signed,
                future2_ordinal
            );
            future4 = std::move(future2);
            future4_ordinal = future2_ordinal;
            future2 = std::move(solved_layer);
            future2_ordinal = ordinal_signed;
        }
        archive_retired_layer_prune_and_write_if_needed(future4_ordinal, future4);
        archive_retired_layer_prune_and_write_if_needed(future2_ordinal, future2);
        write_checkpoint_manifest(args, -1, -1, -1);
        const double solve_seconds = now_seconds() - solve_begin;
        for (const LayerMetric &metric : metrics) {
            write_metric_row(stats_out, metric);
        }
        stats_out.flush();

        uint64_t total_rows = 0U;
        uint64_t total_live_rows = 0U;
        uint64_t total_zero_pruned_rows = 0U;
        uint64_t total_archive_live_rows = 0U;
        uint64_t total_threshold_pruned_rows = 0U;
        uint64_t total_position_bytes = 0U;
        uint64_t total_success_bytes = 0U;
        uint64_t total_queries2 = 0U;
        uint64_t total_queries4 = 0U;
        uint64_t total_found2 = 0U;
        uint64_t total_found4 = 0U;
        uint64_t total_terminal = 0U;
        double total_read = 0.0;
        double total_index = 0.0;
        double total_recalc = 0.0;
        double total_compact = 0.0;
        double total_archive_compact = 0.0;
        double total_position_write = 0.0;
        double total_success_write = 0.0;
        for (const LayerMetric &m : metrics) {
            total_rows += m.current_rows;
            total_live_rows += m.live_rows;
            total_zero_pruned_rows += m.zero_pruned_rows;
            total_archive_live_rows += m.archive_live_rows;
            total_threshold_pruned_rows += m.threshold_pruned_rows;
            total_position_bytes += m.position_bytes;
            total_success_bytes += m.success_bytes;
            total_queries2 += m.queries2;
            total_queries4 += m.queries4;
            total_found2 += m.found2;
            total_found4 += m.found4;
            total_terminal += m.terminal_success_rows;
            total_read += m.read_seconds;
            total_index += m.index_seconds;
            total_recalc += m.recalc_seconds;
            total_compact += m.compact_seconds;
            total_archive_compact += m.archive_compact_seconds;
            total_position_write += m.position_write_seconds;
            total_success_write += m.success_write_seconds;
        }

        const InspectResult inspect =
            inspect_layer_max<StorageT>(args, lut, args.inspect_layer_ordinal);
        std::ofstream summary(args.summary_csv);
        if (!summary) {
            throw std::runtime_error("failed to open summary csv: " + args.summary_csv.string());
        }
        summary << std::setprecision(12);
        summary
            << "generated_position_dir,solved_output_dir,min_ordinal,max_ordinal,layers_solved,total_rows,"
            << "total_live_rows,total_zero_pruned_rows,total_archive_live_rows,"
            << "total_threshold_pruned_rows,total_position_bytes,total_success_bytes,"
            << "total_queries2,total_queries4,total_found2,total_found4,total_terminal_success_rows,"
            << "total_read_seconds,total_index_seconds,total_recalc_seconds,total_compact_seconds,"
            << "total_archive_compact_seconds,total_position_write_seconds,total_success_write_seconds,solve_wall_seconds,"
            << "wall_mrows_per_sec,recalc_mrows_per_sec,inspect_ordinal,inspect_layer_sum,"
            << "inspect_rows,inspect_max_value,inspect_max_rate,inspect_max_count,"
            << "inspect_first_max_board_hex,inspect_first_max_cell,inspect_first_max_row\n";
        const double wall_mrows = solve_seconds > 0.0
            ? static_cast<double>(total_rows) / solve_seconds / 1.0e6
            : 0.0;
        const double recalc_mrows = total_recalc > 0.0
            ? static_cast<double>(total_rows) / total_recalc / 1.0e6
            : 0.0;
        summary
            << args.generated_position_dir.string() << ','
            << args.solved_output_dir.string() << ','
            << min_ordinal << ','
            << max_ordinal << ','
            << (max_ordinal - min_ordinal) << ','
            << total_rows << ','
            << total_live_rows << ','
            << total_zero_pruned_rows << ','
            << total_archive_live_rows << ','
            << total_threshold_pruned_rows << ','
            << total_position_bytes << ','
            << total_success_bytes << ','
            << total_queries2 << ','
            << total_queries4 << ','
            << total_found2 << ','
            << total_found4 << ','
            << total_terminal << ','
            << total_read << ','
            << total_index << ','
            << total_recalc << ','
            << total_compact << ','
            << total_archive_compact << ','
            << total_position_write << ','
            << total_success_write << ','
            << solve_seconds << ','
            << wall_mrows << ','
            << recalc_mrows << ','
            << inspect.ordinal << ','
            << inspect.layer_sum << ','
            << inspect.rows << ','
            << inspect.max_value << ','
            << inspect.max_rate << ','
            << inspect.max_count << ",0x"
            << std::hex << inspect.first_max_board << std::dec << ','
            << inspect.first_max_cell << ','
            << inspect.first_max_row << '\n';

        std::cout << std::setprecision(12)
            << "summary"
            << " layers_solved=" << (max_ordinal - min_ordinal)
            << " total_rows=" << total_rows
            << " total_live_rows=" << total_live_rows
            << " total_threshold_pruned_rows=" << total_threshold_pruned_rows
            << " total_recalc_seconds=" << total_recalc
            << " total_compact_seconds=" << total_compact
            << " total_archive_compact_seconds=" << total_archive_compact
            << " total_position_write_seconds=" << total_position_write
            << " total_success_write_seconds=" << total_success_write
            << " solve_wall_seconds=" << solve_seconds
            << " wall_mrows_per_sec=" << wall_mrows
            << " recalc_mrows_per_sec=" << recalc_mrows
            << " inspect_layer=" << inspect.ordinal
            << " inspect_rows=" << inspect.rows
            << " inspect_max_value=" << inspect.max_value
            << " inspect_max_rate=" << inspect.max_rate
            << '\n';
        return 0;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Args args = parse_args(argc, argv);
        switch (args.success_dtype) {
            case BC::BCSuccessDTypeMode::UInt32:
                return run_typed<uint32_t>(args);
            case BC::BCSuccessDTypeMode::UInt64:
                return run_typed<uint64_t>(args);
            case BC::BCSuccessDTypeMode::Float32:
            case BC::BCSuccessDTypeMode::OneMinusFloat32:
                return run_typed<float>(args);
            case BC::BCSuccessDTypeMode::Float64:
            case BC::BCSuccessDTypeMode::OneMinusFloat64:
                return run_typed<double>(args);
        }
    } catch (const std::exception &ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
    std::cerr << "error: unsupported success dtype\n";
    return 1;
}
