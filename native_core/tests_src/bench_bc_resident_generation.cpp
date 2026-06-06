#include "BCCellBuilder.h"
#include "BCCellMatrix.h"
#include "BCBoardOps.h"
#include "BCPositionScanner.h"
#include "BCResidentGeneration.h"
#include "Calculator.h"
#include "CanonicalBatch.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using BC::BCCellBuilder;
using BC::BCCellMatrix;
using BC::BCFamilyTable;
using BC::BCLut;
using BC::BCPositionLayerReader;
using BC::BCPositionLayerWriter;
using BC::BCResidentGenerationOptions;
using BC::BCResidentGenerationResult;
using BC::BCResidentGenerationSource;
using BC::CellId;

struct Args {
    std::string pattern = "free9";
    uint32_t target_rank = 8U;
    int num_threads = 0;
    uint32_t batch_size = 4096U;
};

struct ResidentLayer {
    uint32_t layer_sum = 0U;
    std::vector<uint8_t> bytes;
    std::unique_ptr<BCPositionLayerReader> reader;
    uint64_t rows = 0U;
};

void check(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto require_value = [&](const char *label) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + label);
            }
            return argv[++i];
        };
        if (key == "--pattern") {
            args.pattern = require_value("--pattern");
        } else if (key == "--target-rank") {
            args.target_rank = static_cast<uint32_t>(std::stoul(require_value("--target-rank")));
        } else if (key == "--num-threads") {
            args.num_threads = std::stoi(require_value("--num-threads"));
        } else if (key == "--batch-size") {
            args.batch_size = static_cast<uint32_t>(std::stoul(require_value("--batch-size")));
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.pattern != "free9") {
        throw std::invalid_argument("bc_resident_generation_bench currently supports --pattern free9 only");
    }
    if (args.target_rank > 12U) {
        throw std::invalid_argument("--target-rank is unexpectedly large for this memory benchmark");
    }
    return args;
}

std::filesystem::path find_patterns_config() {
    std::filesystem::path current = std::filesystem::current_path();
    for (uint32_t i = 0; i < 8U; ++i) {
        const std::filesystem::path candidate =
            current / "docs_and_configs" / "patterns_config.json";
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
        if (!current.has_parent_path() || current.parent_path() == current) {
            break;
        }
        current = current.parent_path();
    }
    throw std::runtime_error("failed to locate docs_and_configs/patterns_config.json");
}

uint64_t parse_hex_u64(const std::string &text) {
    size_t parsed = 0U;
    const uint64_t value = std::stoull(text, &parsed, 0);
    if (parsed != text.size()) {
        throw std::runtime_error("failed to parse full hex seed board");
    }
    return value;
}

uint64_t load_free9_seed_board() {
    const std::filesystem::path config_path = find_patterns_config();
    std::ifstream file(config_path);
    if (!file) {
        throw std::runtime_error("failed to open patterns_config.json");
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    const std::string text = ss.str();
    const size_t free9 = text.find("\"free9\"");
    if (free9 == std::string::npos) {
        throw std::runtime_error("patterns_config.json does not contain free9");
    }
    const size_t seed_boards = text.find("\"seed boards\"", free9);
    if (seed_boards == std::string::npos) {
        throw std::runtime_error("free9 does not contain seed boards");
    }
    const size_t hex_begin = text.find("0x", seed_boards);
    if (hex_begin == std::string::npos) {
        throw std::runtime_error("free9 seed board has no hex value");
    }
    size_t hex_end = hex_begin + 2U;
    while (hex_end < text.size() &&
           std::isxdigit(static_cast<unsigned char>(text[hex_end])) != 0) {
        ++hex_end;
    }
    return parse_hex_u64(text.substr(hex_begin, hex_end - hex_begin));
}

std::array<uint32_t, 16U> free9_semantic_tile_sums() {
    std::array<uint32_t, 16U> values = BC::default_2048_tile_sum_values();
    values[15] = 0U;
    return values;
}

BCLut make_free9_lut() {
    return BCLut({0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 15U});
}

uint32_t board_semantic_sum(uint64_t board, const std::array<uint32_t, 16U> &tile_sums) {
    uint64_t sum = 0U;
    for (uint32_t cell = 0U; cell < 16U; ++cell) {
        const uint8_t tile = static_cast<uint8_t>((board >> (4U * cell)) & 0xFU);
        sum += tile_sums[tile];
    }
    if (sum > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("semantic layer sum exceeds uint32");
    }
    return static_cast<uint32_t>(sum);
}

BCFamilyTable make_axis(uint32_t layer_sum) {
    if ((layer_sum & 1U) != 0U) {
        throw std::invalid_argument("free9 BC benchmark requires even layer sums");
    }
    const uint32_t total_coord = layer_sum / 2U;
    const uint32_t max_coord = total_coord / 2U;
    if (max_coord > std::numeric_limits<BC::FamilyCoord>::max()) {
        throw std::overflow_error("free9 BC benchmark axis exceeds FamilyCoord");
    }
    return BCFamilyTable::from_range(
        layer_sum,
        2U,
        0U,
        static_cast<BC::FamilyCoord>(max_coord)
    );
}

uint64_t descriptor_success_rows_sum(const BCPositionLayerReader &reader) {
    uint64_t rows = 0U;
    for (CellId cid = 0U; cid < reader.cell_count(); ++cid) {
        rows += reader.descriptor(cid).success_rows;
    }
    return rows;
}

ResidentLayer write_seed_layer(
    const BCLut &lut,
    const BCFamilyTable &axis,
    uint64_t seed_board,
    const std::array<uint32_t, 16U> &family_tile_sums
) {
    const uint64_t canonical = Calculator::canonical_full(seed_board);
    const auto encoded = BC::encode_canonical_quadrants_position(
        lut,
        axis,
        BC::unpack_board_to_quadrants(canonical),
        &family_tile_sums
    );
    check(encoded.valid, "free9 seed board should encode into seed axis");

    const BCCellMatrix matrix(axis);
    std::vector<std::unique_ptr<BCCellBuilder>> builders(matrix.cell_count());
    builders[encoded.cid] = std::make_unique<BCCellBuilder>(lut);
    BC::BCEncodedKeyRank key_rank;
    key_rank.key = encoded.key;
    key_rank.rank = encoded.rank;
    key_rank.bitmap_len = encoded.bitmap_len;
    key_rank.count_ne = encoded.count_ne;
    key_rank.count_sw = encoded.count_sw;
    key_rank.count_se = encoded.count_se;
    key_rank.valid = true;
    (void)builders[encoded.cid]->insert_encoded_and_report(key_rank);

    BCPositionLayerWriter writer;
    writer.begin_layer(axis);
    for (CellId cid = 0U; cid < matrix.cell_count(); ++cid) {
        if (!builders[cid]) {
            writer.mark_empty_cell(cid);
            continue;
        }
        writer.write_cell(cid, builders[cid]->finalize());
    }

    ResidentLayer layer;
    layer.layer_sum = axis.layer_sum();
    layer.bytes = writer.finish_layer();
    layer.reader = std::make_unique<BCPositionLayerReader>(layer.bytes, lut);
    layer.rows = descriptor_success_rows_sum(*layer.reader);
    return layer;
}

ResidentLayer make_generated_layer(
    uint32_t layer_sum,
    const BCLut &lut,
    const BCFamilyTable &target_axis,
    BCResidentGenerationResult &&result
) {
    ResidentLayer layer;
    layer.layer_sum = layer_sum;
    layer.bytes = std::move(result.position_bytes);
    layer.reader = std::make_unique<BCPositionLayerReader>(layer.bytes, lut);
    layer.rows = descriptor_success_rows_sum(*layer.reader);
    return layer;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Args args = parse_args(argc, argv);
        const std::array<uint32_t, 16U> tile_sums = free9_semantic_tile_sums();
        const BCLut lut = make_free9_lut();
        const uint64_t seed_board = load_free9_seed_board();
        const uint32_t seed_sum = board_semantic_sum(seed_board, tile_sums);
        const uint32_t extra_sum = 1U << args.target_rank;
        const uint32_t final_sum = seed_sum + extra_sum;

        BCResidentGenerationOptions options;
        options.num_threads = args.num_threads;
        options.canonical_batch_size = args.batch_size;
        options.family_tile_sum_values = &tile_sums;

        std::map<uint32_t, ResidentLayer> layers;
        layers.emplace(seed_sum, write_seed_layer(lut, make_axis(seed_sum), seed_board, tile_sums));

        uint64_t aggregate_moved = 0U;
        uint64_t aggregate_source = 0U;
        uint64_t aggregate_output = 0U;
        uint64_t aggregate_ex_live = 0U;
        double aggregate_compute = 0.0;
        double aggregate_total = 0.0;
        uint64_t warm_moved = 0U;
        uint64_t warm_ex_live = 0U;
        double warm_compute = 0.0;
        double warm_total = 0.0;
        uint32_t warm_layers = 0U;

        std::cout
            << "canonical_backend=" << CanonicalBatch::backend_name()
            << " pattern=" << args.pattern
            << " seed=0x" << std::hex << seed_board << std::dec
            << " seed_sum=" << seed_sum
            << " target_sum=" << final_sum
            << " num_threads=" << args.num_threads
            << " batch_size=" << args.batch_size
            << " semantic_f_sum=0\n";
        std::cout
            << "layer_sum,input_live,primary_live,throughput_live,source_rows,output_rows,"
            << "source_boards,spawned,move_candidates,moved_candidates,encoded,duplicates,"
            << "total_seconds,throughput_mbps,compute_seconds,compute_throughput_mbps,"
            << "moved_candidate_mbps,source_board_mbps,output_board_mbps\n";

        for (uint32_t layer_sum = seed_sum + 2U; layer_sum <= final_sum; layer_sum += 2U) {
            const BCFamilyTable target_axis = make_axis(layer_sum);
            std::vector<BCResidentGenerationSource> sources;
            uint64_t source_rows = 0U;
            const auto source4_it = layer_sum >= 4U ? layers.find(layer_sum - 4U) : layers.end();
            if (source4_it != layers.end()) {
                sources.push_back(BCResidentGenerationSource{source4_it->second.reader.get(), 2U, 2U});
                source_rows += source4_it->second.rows;
            }
            const auto source2_it = layers.find(layer_sum - 2U);
            if (source2_it != layers.end()) {
                sources.push_back(BCResidentGenerationSource{source2_it->second.reader.get(), 1U, 1U});
                source_rows += source2_it->second.rows;
            }
            if (sources.empty()) {
                throw std::runtime_error("free9 rolling benchmark lost required resident sources");
            }

            BCResidentGenerationResult result =
                BC::generate_resident_position_layer(lut, target_axis, sources, options);
            const uint64_t output_rows = result.output_success_rows;
            const uint64_t throughput_live = result.ex_generate_throughput_live();
            std::cout
                << layer_sum << ','
                << result.source_boards_scanned << ','
                << result.output_success_rows << ','
                << throughput_live << ','
                << source_rows << ','
                << output_rows << ','
                << result.source_boards_scanned << ','
                << result.spawned_boards << ','
                << result.move_candidates << ','
                << result.moved_candidates << ','
                << result.encoded_candidates << ','
                << result.duplicate_candidates << ','
                << result.total_seconds << ','
                << result.throughput_mbps() << ','
                << result.compute_seconds << ','
                << result.compute_throughput_mbps() << ','
                << result.moved_candidate_mbps() << ','
                << result.source_board_mbps() << ','
                << result.output_board_mbps()
                << "\n";

            aggregate_moved += result.moved_candidates;
            aggregate_source += result.source_boards_scanned;
            aggregate_output += output_rows;
            aggregate_ex_live += throughput_live;
            aggregate_compute += result.compute_seconds;
            aggregate_total += result.total_seconds;
            if (layer_sum >= seed_sum + 16U) {
                ++warm_layers;
                warm_moved += result.moved_candidates;
                warm_ex_live += throughput_live;
                warm_compute += result.compute_seconds;
                warm_total += result.total_seconds;
            }

            layers.emplace(
                layer_sum,
                make_generated_layer(layer_sum, lut, target_axis, std::move(result))
            );
            while (!layers.empty() && layers.begin()->first + 4U < layer_sum) {
                layers.erase(layers.begin());
            }
        }

        const double aggregate_mbps =
            aggregate_compute > 0.0
                ? static_cast<double>(aggregate_moved) / aggregate_compute / 1.0e6
                : 0.0;
        const double aggregate_compute_throughput_mbps =
            aggregate_compute > 0.0
                ? static_cast<double>(aggregate_ex_live) / aggregate_compute / 1.0e6
                : 0.0;
        const double aggregate_throughput_mbps =
            aggregate_total > 0.0
                ? static_cast<double>(aggregate_ex_live) / aggregate_total / 1.0e6
                : 0.0;
        const bool has_warm_layers = warm_layers != 0U;
        const double warm_mbps =
            has_warm_layers && warm_compute > 0.0
                ? static_cast<double>(warm_moved) / warm_compute / 1.0e6
                : aggregate_mbps;
        const uint64_t reported_warm_ex_live =
            has_warm_layers ? warm_ex_live : aggregate_ex_live;
        const double warm_compute_throughput_mbps =
            has_warm_layers && warm_compute > 0.0
                ? static_cast<double>(warm_ex_live) / warm_compute / 1.0e6
                : aggregate_compute_throughput_mbps;
        const double warm_throughput_mbps =
            has_warm_layers && warm_total > 0.0
                ? static_cast<double>(warm_ex_live) / warm_total / 1.0e6
                : aggregate_throughput_mbps;
        std::cout
            << "aggregate_moved=" << aggregate_moved
            << " aggregate_source=" << aggregate_source
            << " aggregate_output=" << aggregate_output
            << " aggregate_throughput_live=" << aggregate_ex_live
            << " aggregate_total_seconds=" << aggregate_total
            << " aggregate_compute_seconds=" << aggregate_compute
            << " aggregate_throughput_mbps=" << aggregate_throughput_mbps
            << " aggregate_compute_throughput_mbps=" << aggregate_compute_throughput_mbps
            << " aggregate_moved_candidate_mbps=" << aggregate_mbps
            << " warm_layers=" << warm_layers
            << " warm_throughput_live=" << reported_warm_ex_live
            << " warm_throughput_mbps=" << warm_throughput_mbps
            << " warm_compute_throughput_mbps=" << warm_compute_throughput_mbps
            << " warm_moved_candidate_mbps=" << warm_mbps
            << "\n";

        const char *perf_assert = std::getenv("BC_PERF_ASSERT");
        if (perf_assert != nullptr && std::string(perf_assert) == "1" && warm_mbps < 100.0) {
            std::cerr << "BC_PERF_ASSERT failed: warm moved_candidate_mbps < 100\n";
            return 2;
        }
    } catch (const std::exception &ex) {
        std::cerr << "bc_resident_generation_bench failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
