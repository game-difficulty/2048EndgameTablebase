#include "BCDirectFileIO.h"
#include "BCResidentSolve.h"
#include "BCSuccessIO.h"
#include "FormationRuntime.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::filesystem::path generated_position_dir = "tmp/free9_256_layer100_fixture/generated";
    std::filesystem::path future_solved_dir = "tmp/free9_256_layer100_fixture/solved";
    std::filesystem::path expected_solved_dir = "tmp/free9_256_layer100_fixture/solved";
    std::filesystem::path output_dir = "tmp/free9_256_layer100_bench_output";
    std::filesystem::path stats_csv;
    std::string prefix = "free9_256_";
    uint32_t ordinal = 100U;
    uint32_t target_rank = 8U;
    int success_target_rank = 8;
    int canonical_symm_mode = static_cast<int>(SymmMode::Full);
    double spawn_rate4 = 0.1;
    int num_threads = 0;
    uint32_t canonical_batch_size = 8192U;
    bool direct_io = false;
    uint32_t direct_queue_depth = 16U;
    bool write_output = true;
    bool verify_expected = true;
    uint32_t repeats = 1U;
};

struct IterationMetric {
    uint32_t iteration = 0U;
    uint64_t current_rows = 0U;
    uint64_t live_rows = 0U;
    uint64_t zero_pruned_rows = 0U;
    uint64_t position_bytes = 0U;
    uint64_t success_bytes = 0U;
    uint64_t queries2 = 0U;
    uint64_t queries4 = 0U;
    uint64_t found2 = 0U;
    uint64_t found4 = 0U;
    uint64_t empty_slots = 0U;
    uint64_t spawned_boards = 0U;
    uint64_t unchanged_moves = 0U;
    uint64_t canonicalized_candidates = 0U;
    uint64_t encoded_queries = 0U;
    uint64_t encode_rejects = 0U;
    uint64_t future_lookup_misses = 0U;
    uint64_t batch_flushes = 0U;
    uint64_t batch_tail_flushes = 0U;
    uint64_t batch_source_boards = 0U;
    uint64_t canonical_flushes = 0U;
    uint32_t max_value = 0U;
    double current_read_seconds = 0.0;
    double recalc_seconds = 0.0;
    double compact_seconds = 0.0;
    double position_write_seconds = 0.0;
    double success_write_seconds = 0.0;
    double verify_seconds = 0.0;
    double total_seconds = 0.0;
    bool verified = false;
};

double accounted_seconds(const IterationMetric &m) {
    return m.current_read_seconds +
        m.recalc_seconds +
        m.compact_seconds +
        m.position_write_seconds +
        m.success_write_seconds +
        m.verify_seconds;
}

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

void print_usage(const char *exe) {
    std::cerr
        << "Usage: " << exe << " [options]\n"
        << "  --generated-position-dir DIR   current .bcpos directory\n"
        << "  --future-solved-dir DIR        solved future .bcpos/.bcsuc directory\n"
        << "  --expected-solved-dir DIR      solved oracle directory\n"
        << "  --output-dir DIR               write computed layer here\n"
        << "  --stats-csv PATH               write per-repeat stats CSV\n"
        << "  --ordinal N                    current layer ordinal, default 100\n"
        << "  --target-rank N                free BC LUT target rank, default 8\n"
        << "  --prefix P                     file prefix, default free9_256_\n"
        << "  --repeats N                    repeat current read+solve+compact\n"
        << "  --threads N                    OpenMP thread count\n"
        << "  --direct-io                    use direct IO for position read/write\n"
        << "  --direct-queue-depth N         direct IO queue depth\n"
        << "  --no-write                     skip output writes\n"
        << "  --no-verify                    skip expected solved comparison\n";
}

[[nodiscard]] Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--generated-position-dir") {
            args.generated_position_dir = require_value(argc, argv, i, flag.c_str());
        } else if (flag == "--future-solved-dir") {
            args.future_solved_dir = require_value(argc, argv, i, flag.c_str());
        } else if (flag == "--expected-solved-dir") {
            args.expected_solved_dir = require_value(argc, argv, i, flag.c_str());
        } else if (flag == "--output-dir") {
            args.output_dir = require_value(argc, argv, i, flag.c_str());
        } else if (flag == "--stats-csv") {
            args.stats_csv = require_value(argc, argv, i, flag.c_str());
        } else if (flag == "--ordinal") {
            args.ordinal = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, flag.c_str())));
        } else if (flag == "--target-rank") {
            args.target_rank = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, flag.c_str())));
        } else if (flag == "--prefix") {
            args.prefix = require_value(argc, argv, i, flag.c_str());
        } else if (flag == "--success-target-rank") {
            args.success_target_rank = std::stoi(require_value(argc, argv, i, flag.c_str()));
        } else if (flag == "--canonical-symm-mode") {
            args.canonical_symm_mode = parse_symm_mode(require_value(argc, argv, i, flag.c_str()));
        } else if (flag == "--spawn-rate4") {
            args.spawn_rate4 = std::stod(require_value(argc, argv, i, flag.c_str()));
        } else if (flag == "--threads") {
            args.num_threads = std::stoi(require_value(argc, argv, i, flag.c_str()));
        } else if (flag == "--canonical-batch-size") {
            args.canonical_batch_size = static_cast<uint32_t>(
                std::stoul(require_value(argc, argv, i, flag.c_str())));
        } else if (flag == "--direct-io") {
            args.direct_io = true;
        } else if (flag == "--direct-queue-depth") {
            args.direct_queue_depth = static_cast<uint32_t>(
                std::stoul(require_value(argc, argv, i, flag.c_str())));
        } else if (flag == "--no-write") {
            args.write_output = false;
        } else if (flag == "--no-verify") {
            args.verify_expected = false;
        } else if (flag == "--repeats") {
            args.repeats = static_cast<uint32_t>(std::stoul(require_value(argc, argv, i, flag.c_str())));
        } else if (flag == "--help" || flag == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + flag);
        }
    }
    if (args.repeats == 0U) {
        throw std::invalid_argument("--repeats must be >= 1");
    }
    if (args.direct_queue_depth == 0U) {
        throw std::invalid_argument("--direct-queue-depth must be >= 1");
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

[[nodiscard]] std::filesystem::path position_path_for(
    const std::filesystem::path &dir,
    const std::string &prefix,
    uint32_t ordinal
) {
    return dir / (prefix + std::to_string(ordinal) + ".bcpos");
}

[[nodiscard]] std::filesystem::path success_path_for(
    const std::filesystem::path &dir,
    const std::string &prefix,
    uint32_t ordinal
) {
    return dir / (prefix + std::to_string(ordinal) + ".bcsuc");
}

[[nodiscard]] BC::BCPositionFileReader open_position_layer(
    const std::filesystem::path &path,
    const BC::BCLut &lut,
    const Args &args
) {
    if (args.direct_io) {
        return BC::BCPositionFileReader::open_direct_auto(
            path,
            lut,
            args.direct_queue_depth,
            args.direct_queue_depth > 1U
        );
    }
    return BC::BCPositionFileReader::open_buffered(path, lut);
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

[[nodiscard]] std::vector<uint32_t> read_uint32_success_values(
    const BC::BCPositionLayerReader &position,
    const std::filesystem::path &path
) {
    std::vector<uint8_t> bytes = BC::read_success_layer_from_file(path);
    BC::BCSuccessLayerReader reader(bytes, position, 1U);
    if (reader.dtype_mode() != BC::BCSuccessDTypeMode::UInt32) {
        throw std::runtime_error("single-layer bench currently requires UInt32 success files");
    }
    const uint64_t value_count = BC::bc_success_total_values_for(position, 1U);
    if (value_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("single-layer bench success value count exceeds size_t");
    }
    std::vector<uint32_t> values(static_cast<size_t>(value_count));
    const uint64_t payload_bytes = value_count * sizeof(uint32_t);
    std::memcpy(values.data(), bytes.data() + BC::kBCSuccessHeaderBytes, static_cast<size_t>(payload_bytes));
    return values;
}

[[nodiscard]] BC::BCResidentUInt32SolvedLayer load_solved_layer(
    const Args &args,
    const BC::BCLut &lut,
    uint32_t ordinal,
    double *position_read_seconds = nullptr,
    double *success_read_seconds = nullptr
) {
    const double pos_t0 = now_seconds();
    BC::BCPositionFileReader position_file = open_position_layer(
        position_path_for(args.future_solved_dir, args.prefix, ordinal),
        lut,
        args
    );
    if (position_read_seconds != nullptr) {
        *position_read_seconds = now_seconds() - pos_t0;
    }
    const double suc_t0 = now_seconds();
    std::vector<uint32_t> values = read_uint32_success_values(
        position_file.layer(),
        success_path_for(args.future_solved_dir, args.prefix, ordinal)
    );
    if (success_read_seconds != nullptr) {
        *success_read_seconds = now_seconds() - suc_t0;
    }
    BC::BCResidentUInt32SolvedLayer out;
    out.open(position_file.take_layer(), std::move(values), 1U);
    return out;
}

[[nodiscard]] BC::BCResidentUInt32SolvedLayer load_expected_layer(
    const Args &args,
    const BC::BCLut &lut,
    uint32_t ordinal
) {
    Args local = args;
    local.future_solved_dir = args.expected_solved_dir;
    return load_solved_layer(local, lut, ordinal);
}

uint64_t write_position_layer_file(
    const Args &args,
    uint32_t ordinal,
    const BC::BCResidentUInt32SolvedLayer &layer
) {
    const std::vector<uint8_t> &bytes = layer.position.bytes();
    const uint64_t logical_size = static_cast<uint64_t>(bytes.size());
    const std::filesystem::path path = position_path_for(args.output_dir, args.prefix, ordinal);
    std::unique_ptr<BC::BCWritableFile> file = make_output_writer(args, path, logical_size);
    file->prepare_full_overwrite(logical_size);
    BC::BCSequentialSuccessWriteStager stager(*file);
    stager.append(bytes.data(), logical_size);
    stager.finish();
    file.reset();
    if (args.direct_io) {
        std::filesystem::resize_file(path, logical_size);
    }
    return logical_size;
}

uint64_t write_success_layer_file(
    const Args &args,
    uint32_t ordinal,
    const BC::BCResidentUInt32SolvedLayer &layer
) {
    const uint64_t logical_size =
        BC::kBCSuccessHeaderBytes +
        static_cast<uint64_t>(layer.success_values.size()) * sizeof(uint32_t);
    const std::filesystem::path path = success_path_for(args.output_dir, args.prefix, ordinal);
    std::unique_ptr<BC::BCWritableFile> file = make_output_writer(args, path, logical_size);
    const uint64_t written = BC::write_success_values_to_file(
        *file,
        layer.position,
        1U,
        BC::BCSuccessDTypeMode::UInt32,
        layer.success_values
    );
    file.reset();
    if (args.direct_io) {
        std::filesystem::resize_file(path, written);
    }
    return written;
}

void verify_layer_matches(
    const BC::BCResidentUInt32SolvedLayer &actual,
    const BC::BCResidentUInt32SolvedLayer &expected
) {
    if (actual.position.bytes() != expected.position.bytes()) {
        throw std::runtime_error("computed position layer differs from expected solved position");
    }
    if (actual.success_values != expected.success_values) {
        throw std::runtime_error("computed success values differ from expected solved success");
    }
}

void write_stats_header(std::ofstream &out) {
    out
        << "iteration,current_rows,live_rows,zero_pruned_rows,position_bytes,success_bytes,"
        << "queries2,queries4,found2,found4,empty_slots,spawned_boards,unchanged_moves,"
        << "canonicalized_candidates,encoded_queries,encode_rejects,future_lookup_misses,"
        << "batch_flushes,batch_tail_flushes,batch_source_boards,canonical_flushes,"
        << "max_value,current_read_seconds,recalc_seconds,"
        << "compact_seconds,position_write_seconds,success_write_seconds,verify_seconds,total_seconds,"
        << "recalc_mrows_per_sec,total_mrows_per_sec,verified\n";
}

void write_metric_row(std::ofstream &out, const IterationMetric &m) {
    const double recalc_mrows = m.recalc_seconds > 0.0
        ? static_cast<double>(m.current_rows) / m.recalc_seconds / 1.0e6
        : 0.0;
    const double total_mrows = m.total_seconds > 0.0
        ? static_cast<double>(m.current_rows) / m.total_seconds / 1.0e6
        : 0.0;
    out
        << m.iteration << ','
        << m.current_rows << ','
        << m.live_rows << ','
        << m.zero_pruned_rows << ','
        << m.position_bytes << ','
        << m.success_bytes << ','
        << m.queries2 << ','
        << m.queries4 << ','
        << m.found2 << ','
        << m.found4 << ','
        << m.empty_slots << ','
        << m.spawned_boards << ','
        << m.unchanged_moves << ','
        << m.canonicalized_candidates << ','
        << m.encoded_queries << ','
        << m.encode_rejects << ','
        << m.future_lookup_misses << ','
        << m.batch_flushes << ','
        << m.batch_tail_flushes << ','
        << m.batch_source_boards << ','
        << m.canonical_flushes << ','
        << m.max_value << ','
        << m.current_read_seconds << ','
        << m.recalc_seconds << ','
        << m.compact_seconds << ','
        << m.position_write_seconds << ','
        << m.success_write_seconds << ','
        << m.verify_seconds << ','
        << m.total_seconds << ','
        << recalc_mrows << ','
        << total_mrows << ','
        << (m.verified ? 1 : 0)
        << '\n';
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Args args = parse_args(argc, argv);
        const BC::BCLut lut(make_free_legal_tiles(args.target_rank));
        if (args.ordinal > std::numeric_limits<uint32_t>::max() - 2U) {
            throw std::invalid_argument("ordinal is too large for +2 ordinal future");
        }

        double future2_pos_read = 0.0;
        double future2_suc_read = 0.0;
        double future4_pos_read = 0.0;
        double future4_suc_read = 0.0;
        BC::BCResidentUInt32SolvedLayer future2 =
            load_solved_layer(args, lut, args.ordinal + 1U, &future2_pos_read, &future2_suc_read);
        BC::BCResidentUInt32SolvedLayer future4 =
            load_solved_layer(args, lut, args.ordinal + 2U, &future4_pos_read, &future4_suc_read);

        BC::BCResidentUInt32SolvedLayer expected;
        if (args.verify_expected) {
            expected = load_expected_layer(args, lut, args.ordinal);
        }

        std::ofstream stats_out;
        if (!args.stats_csv.empty()) {
            if (!args.stats_csv.parent_path().empty()) {
                std::filesystem::create_directories(args.stats_csv.parent_path());
            }
            stats_out.open(args.stats_csv);
            if (!stats_out) {
                throw std::runtime_error("failed to open stats csv: " + args.stats_csv.string());
            }
            stats_out << std::setprecision(9);
            write_stats_header(stats_out);
        }

        std::vector<IterationMetric> metrics;
        metrics.reserve(args.repeats);
        uint64_t total_rows = 0U;
        double total_current_read = 0.0;
        double total_recalc = 0.0;
        double total_compact = 0.0;
        double total_position_write = 0.0;
        double total_success_write = 0.0;
        double total_verify = 0.0;
        double total_wall = 0.0;

        for (uint32_t iteration = 0U; iteration < args.repeats; ++iteration) {
            const double iter_t0 = now_seconds();
            const double read_t0 = now_seconds();
            BC::BCPositionFileReader current = open_position_layer(
                position_path_for(args.generated_position_dir, args.prefix, args.ordinal),
                lut,
                args
            );
            const double current_read_seconds = now_seconds() - read_t0;
            if (current.layer().header().layer_sum + 2U != future2.position.header().layer_sum ||
                current.layer().header().layer_sum + 4U != future4.position.header().layer_sum) {
                throw std::runtime_error("current/future layer sums do not form n/+2/+4 chain");
            }

            BC::BCResidentSolveOptions<uint32_t> options;
            options.num_threads = args.num_threads;
            options.row_width = 1U;
            options.set_dtype(BC::BCSuccessDTypeMode::UInt32);
            options.edge_options.canonical_batch_size = args.canonical_batch_size;
            options.edge_options.canonical_symm_mode = args.canonical_symm_mode;
            options.edge_options.spawn_rate4 = args.spawn_rate4;
            options.edge_options.success_target_rank = args.success_target_rank;
            options.edge_options.success_check_all_cells = true;
            options.edge_options.future_cell_modulus = current.layer().header().family_count;

            BC::BCResidentLayerResult<uint32_t> solve_result =
                BC::bc_resident_solve_compacted_layer<uint32_t>(
                    current.layer(),
                    future2,
                    future4,
                    options
                );
            BC::BCResidentUInt32SolvedLayer solved_layer = std::move(solve_result.layer);

            IterationMetric metric;
            metric.iteration = iteration;
            metric.current_rows = solve_result.solve_stats.current_rows;
            metric.live_rows = solved_layer.compact_stats.live_rows;
            metric.zero_pruned_rows = solved_layer.compact_stats.zero_pruned_rows;
            metric.position_bytes = solved_layer.position.bytes().size();
            metric.success_bytes =
                BC::kBCSuccessHeaderBytes +
                static_cast<uint64_t>(solved_layer.success_values.size()) * sizeof(uint32_t);
            metric.queries2 = solve_result.solve_stats.queries2;
            metric.queries4 = solve_result.solve_stats.queries4;
            metric.found2 = solve_result.solve_stats.found2;
            metric.found4 = solve_result.solve_stats.found4;
            metric.empty_slots = solve_result.solve_stats.edge.empty_slots;
            metric.spawned_boards = solve_result.solve_stats.edge.spawned_boards;
            metric.unchanged_moves = solve_result.solve_stats.edge.unchanged_moves;
            metric.canonicalized_candidates =
                solve_result.solve_stats.edge.canonicalized_candidates;
            metric.encoded_queries = solve_result.solve_stats.edge.encoded_queries;
            metric.encode_rejects = solve_result.solve_stats.edge.encode_rejects;
            metric.future_lookup_misses = solve_result.solve_stats.edge.future_lookup_misses;
            metric.batch_flushes = solve_result.solve_stats.edge.batch_flushes;
            metric.batch_tail_flushes = solve_result.solve_stats.edge.batch_tail_flushes;
            metric.batch_source_boards = solve_result.solve_stats.edge.batch_source_boards;
            metric.canonical_flushes = solve_result.solve_stats.edge.canonical_flushes;
            metric.current_read_seconds = current_read_seconds;
            metric.recalc_seconds = solve_result.solve_stats.recalc_seconds;
            metric.compact_seconds = solved_layer.compact_stats.compact_seconds;
            metric.max_value = solved_layer.success_values.empty()
                ? 0U
                : *std::max_element(solved_layer.success_values.begin(), solved_layer.success_values.end());

            if (args.verify_expected) {
                const double verify_t0 = now_seconds();
                verify_layer_matches(solved_layer, expected);
                metric.verify_seconds = now_seconds() - verify_t0;
                metric.verified = true;
            }

            if (args.write_output) {
                const double position_t0 = now_seconds();
                write_position_layer_file(args, args.ordinal, solved_layer);
                metric.position_write_seconds = now_seconds() - position_t0;
                const double success_t0 = now_seconds();
                write_success_layer_file(args, args.ordinal, solved_layer);
                metric.success_write_seconds = now_seconds() - success_t0;
            }
            metric.total_seconds = now_seconds() - iter_t0;
            if (stats_out) {
                write_metric_row(stats_out, metric);
            }
            metrics.push_back(metric);
            total_rows += metric.current_rows;
            total_current_read += metric.current_read_seconds;
            total_recalc += metric.recalc_seconds;
            total_compact += metric.compact_seconds;
            total_position_write += metric.position_write_seconds;
            total_success_write += metric.success_write_seconds;
            total_verify += metric.verify_seconds;
            total_wall += metric.total_seconds;
            const double accounted = accounted_seconds(metric);
            std::cout << std::setprecision(9)
                << "iteration=" << iteration
                << " rows=" << metric.current_rows
                << " live_rows=" << metric.live_rows
                << " recalc_seconds=" << metric.recalc_seconds
                << " compact_seconds=" << metric.compact_seconds
                << " position_write_seconds=" << metric.position_write_seconds
                << " success_write_seconds=" << metric.success_write_seconds
                << " accounted_seconds=" << accounted
                << " untracked_seconds=" << (metric.total_seconds - accounted)
                << " solve_window_seconds=" << metric.total_seconds
                << " total_seconds=" << metric.total_seconds
                << " total_mrows_per_sec="
                << (metric.total_seconds > 0.0
                        ? static_cast<double>(metric.current_rows) / metric.total_seconds / 1.0e6
                        : 0.0)
                << " verified=" << (metric.verified ? 1 : 0)
                << '\n';
        }

        if (stats_out) {
            stats_out.flush();
        }
        const double setup_read =
            future2_pos_read + future2_suc_read + future4_pos_read + future4_suc_read;
        const double wall_mrows = total_wall > 0.0
            ? static_cast<double>(total_rows) / total_wall / 1.0e6
            : 0.0;
        const double cold_wall = setup_read + total_wall;
        const double cold_mrows = cold_wall > 0.0
            ? static_cast<double>(total_rows) / cold_wall / 1.0e6
            : 0.0;
        const double accounted =
            total_current_read +
            total_recalc +
            total_compact +
            total_position_write +
            total_success_write +
            total_verify;
        const double recalc_mrows = total_recalc > 0.0
            ? static_cast<double>(total_rows) / total_recalc / 1.0e6
            : 0.0;
        std::cout << std::setprecision(12)
            << "summary"
            << " ordinal=" << args.ordinal
            << " repeats=" << args.repeats
            << " setup_future_read_seconds=" << setup_read
            << " total_rows=" << total_rows
            << " total_current_read_seconds=" << total_current_read
            << " total_recalc_seconds=" << total_recalc
            << " total_compact_seconds=" << total_compact
            << " total_position_write_seconds=" << total_position_write
            << " total_success_write_seconds=" << total_success_write
            << " total_verify_seconds=" << total_verify
            << " total_accounted_seconds=" << accounted
            << " total_untracked_seconds=" << (total_wall - accounted)
            << " solve_window_seconds=" << total_wall
            << " total_wall_seconds=" << total_wall
            << " cold_wall_seconds=" << cold_wall
            << " wall_mrows_per_sec=" << wall_mrows
            << " cold_wall_mrows_per_sec=" << cold_mrows
            << " recalc_mrows_per_sec=" << recalc_mrows
            << '\n';
    } catch (const std::exception &ex) {
        std::cerr << "bc_resident_solve_layer_bench failed: " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
