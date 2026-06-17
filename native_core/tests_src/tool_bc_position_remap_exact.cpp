#include "BCBoardOps.h"
#include "BCDirectFileIO.h"
#include "BCFileIO.h"
#include "BCPositionFamilyRemapReader.h"
#include "BCPositionFile.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

struct Args {
    std::filesystem::path input_dir;
    std::filesystem::path output_dir;
    std::string input_prefix = "bc_layer_";
    std::string output_prefix = "free9_256_";
    uint32_t target_rank = 8U;
    uint32_t target_modulus = 0U;
    bool direct_io = false;
    uint32_t direct_queue_depth = 16U;
    int num_threads = 0;
    std::filesystem::path stats_csv;
};

struct InputLayer {
    uint32_t layer_sum = 0U;
    std::filesystem::path path;
};

double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

std::string require_value(int argc, char **argv, int &i, const char *flag) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string(flag) + " requires a value");
    }
    ++i;
    return argv[i];
}

Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--input-dir") {
            args.input_dir = require_value(argc, argv, i, "--input-dir");
        } else if (key == "--output-dir") {
            args.output_dir = require_value(argc, argv, i, "--output-dir");
        } else if (key == "--input-prefix") {
            args.input_prefix = require_value(argc, argv, i, "--input-prefix");
        } else if (key == "--output-prefix") {
            args.output_prefix = require_value(argc, argv, i, "--output-prefix");
        } else if (key == "--target-rank") {
            args.target_rank = static_cast<uint32_t>(
                std::stoul(require_value(argc, argv, i, "--target-rank"))
            );
        } else if (key == "--target-modulus") {
            args.target_modulus = static_cast<uint32_t>(
                std::stoul(require_value(argc, argv, i, "--target-modulus"))
            );
        } else if (key == "--direct-io") {
            args.direct_io = true;
        } else if (key == "--direct-queue-depth") {
            args.direct_queue_depth = static_cast<uint32_t>(
                std::stoul(require_value(argc, argv, i, "--direct-queue-depth"))
            );
        } else if (key == "--threads") {
            args.num_threads = std::stoi(require_value(argc, argv, i, "--threads"));
        } else if (key == "--stats-csv") {
            args.stats_csv = require_value(argc, argv, i, "--stats-csv");
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.input_dir.empty()) {
        throw std::invalid_argument("--input-dir is required");
    }
    if (args.output_dir.empty()) {
        throw std::invalid_argument("--output-dir is required");
    }
    if (args.target_rank >= 15U) {
        throw std::invalid_argument("--target-rank must be < 15");
    }
    if (args.target_modulus > std::numeric_limits<BC::FamilyId>::max()) {
        throw std::invalid_argument("--target-modulus exceeds FamilyId range");
    }
    if (args.direct_queue_depth == 0U) {
        throw std::invalid_argument("--direct-queue-depth must be non-zero");
    }
    return args;
}

std::vector<uint8_t> make_free_legal_tiles(uint32_t target_rank) {
    std::vector<uint8_t> legal_tiles;
    legal_tiles.reserve(static_cast<size_t>(target_rank) + 2U);
    for (uint32_t tile = 0U; tile <= target_rank; ++tile) {
        legal_tiles.push_back(static_cast<uint8_t>(tile));
    }
    legal_tiles.push_back(15U);
    return legal_tiles;
}

BC::BCLut make_free_lut(uint32_t target_rank) {
    return BC::BCLut(make_free_legal_tiles(target_rank));
}

std::vector<InputLayer> discover_inputs(const Args &args) {
    if (!std::filesystem::is_directory(args.input_dir)) {
        throw std::runtime_error("input dir does not exist: " + args.input_dir.string());
    }
    const std::regex pattern("^" + args.input_prefix + "([0-9]+)\\.bcpos$");
    std::vector<InputLayer> layers;
    for (const std::filesystem::directory_entry &entry :
         std::filesystem::directory_iterator(args.input_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        std::smatch match;
        if (!std::regex_match(name, match, pattern)) {
            continue;
        }
        layers.push_back(InputLayer{
            static_cast<uint32_t>(std::stoul(match[1].str())),
            entry.path()
        });
    }
    std::sort(
        layers.begin(),
        layers.end(),
        [](const InputLayer &lhs, const InputLayer &rhs) {
            return lhs.layer_sum < rhs.layer_sum;
        }
    );
    if (layers.empty()) {
        throw std::runtime_error("no input bcpos files found");
    }
    return layers;
}

std::unique_ptr<BC::BCWritableFile> open_writer(const Args &args, const std::filesystem::path &path) {
    if (args.direct_io) {
        BC::BCDirectFileIOOptions options;
        options.queue_depth = args.direct_queue_depth;
        options.overlapped = args.direct_queue_depth > 1U;
        return std::make_unique<BC::BCDirectFileWriter>(path, options);
    }
    return std::make_unique<BC::BCBufferedFileWriter>(path);
}

BC::BCPositionStreamingReader open_reader(
    const Args &args,
    const std::filesystem::path &path,
    const BC::BCLut &lut
) {
    if (args.direct_io) {
        return BC::BCPositionStreamingReader::open_direct_auto(
            path,
            lut,
            args.direct_queue_depth,
            args.direct_queue_depth > 1U
        );
    }
    return BC::BCPositionStreamingReader::open_buffered(path, lut);
}

uint64_t cell_rows(const std::vector<BC::FinalizedCellPayload> &cells) {
    uint64_t rows = 0U;
    for (const BC::FinalizedCellPayload &cell : cells) {
        rows += cell.success_rows;
    }
    return rows;
}

uint64_t cell_bucket_count(const std::vector<BC::FinalizedCellPayload> &cells) {
    uint64_t buckets = 0U;
    for (const BC::FinalizedCellPayload &cell : cells) {
        buckets += cell.buckets.size();
    }
    return buckets;
}

uint64_t cell_rank_bytes(const std::vector<BC::FinalizedCellPayload> &cells) {
    uint64_t bytes = 0U;
    for (const BC::FinalizedCellPayload &cell : cells) {
        bytes += cell.rank_payload.size();
    }
    return bytes;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (args.num_threads > 0) {
#if defined(_OPENMP)
            omp_set_num_threads(args.num_threads);
#endif
        }
        std::filesystem::create_directories(args.output_dir);
        const std::array<uint32_t, 16U> tile_sums = BC::default_2048_tile_sum_values();
        const std::vector<BC::LayerSum> possible_8tile_sums =
            BC::build_possible_8tile_sums(make_free_legal_tiles(args.target_rank), tile_sums);
        const BC::BCLut lut = make_free_lut(args.target_rank);
        const std::vector<InputLayer> layers = discover_inputs(args);

        std::ofstream stats_file;
        std::ostream *stats = &std::cout;
        if (!args.stats_csv.empty()) {
            stats_file.open(args.stats_csv, std::ios::binary | std::ios::trunc);
            if (!stats_file) {
                throw std::runtime_error("failed to open stats csv: " + args.stats_csv.string());
            }
            stats = &stats_file;
        }
        *stats << std::setprecision(9)
               << "ordinal,layer_sum,source_family_count,target_family_count,target_modulus,rows,buckets,"
               << "rank_payload_bytes,logical_bytes,read_remap_seconds,write_seconds,total_seconds,path\n";

        uint64_t total_rows = 0U;
        uint64_t total_logical_bytes = 0U;
        double total_read_remap = 0.0;
        double total_write = 0.0;
        const double all_begin = now_seconds();
        for (size_t ordinal = 0U; ordinal < layers.size(); ++ordinal) {
            const InputLayer &input = layers[ordinal];
            const double begin = now_seconds();
            const double read_begin = now_seconds();
            BC::BCPositionStreamingReader source = open_reader(args, input.path, lut);
            const BC::BCFamilyTable logical_axis =
                args.target_modulus == 0U
                    ? BC::build_family_axis_for_layer(input.layer_sum, 2U, possible_8tile_sums)
                    : BC::build_family_partition_axis_for_layer(
                          input.layer_sum,
                          2U,
                          possible_8tile_sums,
                          BC::BCFamilyPartitionPolicy::modulo(args.target_modulus)
                      );
            BC::BCPositionFamilyRemapReader remap(source, logical_axis, possible_8tile_sums);
            const BC::BCCellMatrix matrix(logical_axis);
            std::vector<BC::CellId> cids(matrix.cell_count());
            for (BC::CellId cid = 0U; cid < matrix.cell_count(); ++cid) {
                cids[static_cast<size_t>(cid)] = cid;
            }
            std::vector<BC::BCLoadedCell> loaded;
            remap.load_cells_into(cids, loaded, nullptr);
            std::vector<BC::FinalizedCellPayload> payloads(loaded.size());
            for (size_t i = 0U; i < loaded.size(); ++i) {
                payloads[i].buckets = std::move(loaded[i].buckets);
                payloads[i].rank_payload = std::move(loaded[i].rank_payload);
                payloads[i].success_rows = loaded[i].success_rows;
            }
            loaded.clear();
            loaded.shrink_to_fit();
            const double read_remap_seconds = now_seconds() - read_begin;

            const std::filesystem::path output_path =
                args.output_dir /
                (args.output_prefix + std::to_string(ordinal) + ".bcpos");
            const double write_begin = now_seconds();
            std::unique_ptr<BC::BCWritableFile> writer = open_writer(args, output_path);
            const uint64_t logical_bytes =
                BC::write_position_payloads_to_file(*writer, logical_axis, payloads, nullptr);
            writer.reset();
            const double write_seconds = now_seconds() - write_begin;
            const double total_seconds = now_seconds() - begin;

            const uint64_t rows = cell_rows(payloads);
            const uint64_t buckets = cell_bucket_count(payloads);
            const uint64_t rank_bytes = cell_rank_bytes(payloads);
            total_rows += rows;
            total_logical_bytes += logical_bytes;
            total_read_remap += read_remap_seconds;
            total_write += write_seconds;
            *stats << ordinal << ','
                   << input.layer_sum << ','
                   << source.axis().family_count() << ','
                   << logical_axis.family_count() << ','
                   << args.target_modulus << ','
                   << rows << ','
                   << buckets << ','
                   << rank_bytes << ','
                   << logical_bytes << ','
                   << read_remap_seconds << ','
                   << write_seconds << ','
                   << total_seconds << ','
                   << output_path.string() << '\n';
        }
        const double wall = now_seconds() - all_begin;
        *stats << "total,,,,"
               << ',' << total_rows
               << ",,,"
               << total_logical_bytes << ','
               << total_read_remap << ','
               << total_write << ','
               << wall << ",\n";
        std::cout << std::setprecision(9)
                  << "summary layers=" << layers.size()
                  << " rows=" << total_rows
                  << " logical_bytes=" << total_logical_bytes
                  << " read_remap_seconds=" << total_read_remap
                  << " write_seconds=" << total_write
                  << " wall_seconds=" << wall
                  << '\n';
    } catch (const std::exception &ex) {
        std::cerr << "bc_position_remap_exact failed: " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
