#include "BookSolver.h"

#include "BoardMaskerAD.h"
#include "EXADBuilder.h"
#include "EXADIO.h"
#include "FileIOUtils.h"
#include "FormationRuntime.h"
#include "NativeDiagnostics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace fs = std::filesystem;

namespace {

double wall_time_seconds() {
    using clock = std::chrono::steady_clock;
    static const auto epoch = clock::now();
    return std::chrono::duration<double>(clock::now() - epoch).count();
}

double throughput_mbps_for(uint64_t count, double seconds) {
    return seconds > 0.0 ? static_cast<double>(count) / seconds / 1e6 : 0.0;
}

std::string now_string() {
    std::time_t now = std::time(nullptr);
    std::tm local_time{};
#ifdef _WIN32
    localtime_s(&local_time, &now);
#else
    localtime_r(&now, &local_time);
#endif
    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

std::string stats_file_path(const RunOptions &options) {
    return options.pathname + "exad_generate_stats.csv";
}

std::string stats_header() {
    return "stage,step,input_live,arr1_live,arr2_live,output_live,derive_candidate_count,derived_output_count,validate_removed_current,validate_removed_next,prepare_seconds,prepare_estimate_seconds,prepare_arr1_seconds,prepare_arr2_seconds,hashmap_seconds,worklist_seconds,loop_seconds,count_finalize_seconds,generate_seconds,insert_seconds,finalize_seconds,validate_seconds,write_seconds,retry_count,compute_seconds,total_seconds,compute_throughput_mbps,time";
}

void ensure_stats_header(const RunOptions &options) {
    const std::string path = stats_file_path(options);
    if (fs::exists(path)) {
        std::ifstream in(path);
        std::string first;
        if (std::getline(in, first) && first == stats_header()) {
            return;
        }
        in.close();
        std::error_code ec;
        fs::remove(path, ec);
    }
    std::ofstream out(path, std::ios::app);
    out << stats_header() << "\n";
}

struct StatsRecord {
    std::string stage;
    int step = -1;
    uint64_t input_live = 0;
    uint64_t arr1_live = 0;
    uint64_t arr2_live = 0;
    uint64_t output_live = 0;
    uint64_t derive_candidate_count = 0;
    uint64_t derived_output_count = 0;
    uint64_t validate_removed_current = 0;
    uint64_t validate_removed_next = 0;
    double prepare_seconds = 0.0;
    double prepare_estimate_seconds = 0.0;
    double prepare_arr1_seconds = 0.0;
    double prepare_arr2_seconds = 0.0;
    double hashmap_seconds = 0.0;
    double worklist_seconds = 0.0;
    double loop_seconds = 0.0;
    double count_finalize_seconds = 0.0;
    double generate_seconds = 0.0;
    double insert_seconds = 0.0;
    double finalize_seconds = 0.0;
    double validate_seconds = 0.0;
    double write_seconds = 0.0;
    uint32_t retry_count = 0;
};

void append_stats(const RunOptions &options, const StatsRecord &record) {
    ensure_stats_header(options);
    const double compute_seconds =
        record.prepare_seconds + record.generate_seconds + record.finalize_seconds + record.validate_seconds;
    const double total_seconds = compute_seconds + record.write_seconds;
    std::ofstream out(stats_file_path(options), std::ios::app);
    out << record.stage << ","
        << record.step << ","
        << record.input_live << ","
        << record.arr1_live << ","
        << record.arr2_live << ","
        << record.output_live << ","
        << record.derive_candidate_count << ","
        << record.derived_output_count << ","
        << record.validate_removed_current << ","
        << record.validate_removed_next << ","
        << std::fixed << std::setprecision(6)
        << record.prepare_seconds << ","
        << record.prepare_estimate_seconds << ","
        << record.prepare_arr1_seconds << ","
        << record.prepare_arr2_seconds << ","
        << record.hashmap_seconds << ","
        << record.worklist_seconds << ","
        << record.loop_seconds << ","
        << record.count_finalize_seconds << ","
        << record.generate_seconds << ","
        << record.insert_seconds << ","
        << record.finalize_seconds << ","
        << record.validate_seconds << ","
        << record.write_seconds << ","
        << record.retry_count << ","
        << compute_seconds << ","
        << total_seconds << ","
        << throughput_mbps_for(record.input_live, compute_seconds) << ","
        << now_string() << "\n";
}

uint32_t board_sum_from_seed(const std::vector<uint64_t> &boards) {
    if (boards.empty()) {
        return 0U;
    }
    uint32_t sum = 0U;
    const uint64_t board = boards.front();
    for (uint32_t cell = 0; cell < 16U; ++cell) {
        const uint32_t tile = static_cast<uint32_t>((board >> (cell * 4U)) & 0xFULL);
        if (tile != 0U) {
            sum += (1U << tile);
        }
    }
    return sum;
}

PatternSpec make_base_pattern_spec(const AdvancedPatternSpec &spec) {
    PatternSpec base;
    base.name = spec.name;
    base.pattern_masks = spec.pattern_masks;
    base.success_shifts = spec.success_shifts;
    base.symm_mode = spec.symm_mode;
    base.physical_transform = spec.physical_transform;
    base.inverse_physical_transform = spec.inverse_physical_transform;
    base.logical_pattern_signature = spec.logical_pattern_signature;
    base.physical_pattern_signature = spec.physical_pattern_signature;
    return base;
}

EXAD::Luts load_or_build_luts(
    const std::vector<uint64_t> &seed_boards,
    const AdvancedPatternSpec &spec,
    const RunOptions &options,
    int num_threads,
    FileIOUtils::DirectIoConfig io_config
) {
    const std::string path = EXAD::lut_file_path(options.pathname);
    const PatternSpec base = make_base_pattern_spec(spec);
    const ZMaskFrozen::TileLimitConfig config = EXAD::make_exad_lut_tile_limit_config(
        options.target,
        seed_boards,
        base,
        options.is_free,
        options.is_variant
    );
    if (fs::exists(path)) {
        EXAD::Luts luts = EXAD::read_lut_file(path, io_config);
        if (ZMaskFrozen::tile_limit_configs_equal(luts.config, config) &&
            luts.physical_transform == spec.physical_transform &&
            luts.inverse_physical_transform == spec.inverse_physical_transform &&
            luts.logical_pattern_signature == spec.logical_pattern_signature &&
            luts.physical_pattern_signature == spec.physical_pattern_signature) {
            EXAD::initialize_runtime_tables(luts);
            return luts;
        }
    }
    EXAD::Luts luts = EXAD::build_luts(config, num_threads);
    luts.physical_transform = spec.physical_transform;
    luts.inverse_physical_transform = spec.inverse_physical_transform;
    luts.logical_pattern_signature = spec.logical_pattern_signature;
    luts.physical_pattern_signature = spec.physical_pattern_signature;
    EXAD::write_lut_file(path, luts, io_config);
    return luts;
}

bool validate_step_trigger(int step, uint32_t ini_board_sum, const AdvancedMaskParam &param) {
    return ((step + static_cast<int>(ini_board_sum % 64U / 2U)) % 32) ==
        ((static_cast<int>(param.small_tile_sum_limit / 2U)) % 32) + 1;
}

struct ReserveFootprint {
    std::array<uint64_t, bucket_slot_count()> buckets{};
    std::array<uint64_t, bucket_slot_count()> small_bytes{};
    std::array<uint64_t, bucket_slot_count()> large_words{};
};

struct ReserveBaseTracker {
    ReserveFootprint floor;
    bool initialized = false;
};

struct PostValidateReserveFloor {
    ReserveFootprint floor;
    int remaining_layers = 0;
};

struct DerivedOutputHistory {
    std::deque<uint64_t> values;
};

ReserveFootprint reserve_footprint_from_layer(const EXAD::Layer &layer) {
    ReserveFootprint footprint;
    for (size_t slot = 0; slot < layer.sets.size(); ++slot) {
        const EXAD::BoardSet &set = layer.sets[slot];
        footprint.buckets[slot] = static_cast<uint64_t>(set.buckets.size());
        footprint.small_bytes[slot] = static_cast<uint64_t>(set.small_bitmap_bytes.size());
        footprint.large_words[slot] = static_cast<uint64_t>(set.large_bitmap_words.size());
    }
    return footprint;
}

void push_reserve_history(std::deque<ReserveFootprint> &history, const EXAD::Layer &layer) {
    history.push_back(reserve_footprint_from_layer(layer));
    while (history.size() > 5U) {
        history.pop_front();
    }
}

void reset_reserve_history(
    std::deque<ReserveFootprint> &history,
    const EXAD::Layer &current,
    const EXAD::Layer *next = nullptr
) {
    history.clear();
    push_reserve_history(history, current);
    if (next != nullptr) {
        push_reserve_history(history, *next);
    }
}

void apply_reserve_history_floors(EXAD::ReserveFactors &factors, const std::deque<ReserveFootprint> &history) {
    if (history.empty()) {
        return;
    }
    constexpr double kHistoryBaseMargin = 1.08;
    for (size_t slot = 0; slot < bucket_slot_count(); ++slot) {
        uint64_t bucket_floor = 0;
        uint64_t small_floor = 0;
        uint64_t large_floor = 0;
        for (const ReserveFootprint &footprint : history) {
            bucket_floor = std::max(bucket_floor, footprint.buckets[slot]);
            small_floor = std::max(small_floor, footprint.small_bytes[slot]);
            large_floor = std::max(large_floor, footprint.large_words[slot]);
        }
        factors.bucket_floor[slot] = std::max<uint64_t>(
            factors.bucket_floor[slot],
            static_cast<uint64_t>(std::ceil(static_cast<double>(bucket_floor) * kHistoryBaseMargin))
        );
        factors.small_floor[slot] = std::max<uint64_t>(
            factors.small_floor[slot],
            static_cast<uint64_t>(std::ceil(static_cast<double>(small_floor) * kHistoryBaseMargin))
        );
        factors.large_floor[slot] = std::max<uint64_t>(
            factors.large_floor[slot],
            static_cast<uint64_t>(std::ceil(static_cast<double>(large_floor) * kHistoryBaseMargin))
        );
    }
}

uint64_t decayed_reserve_base(uint64_t value) {
    return (value * 95U + 99U) / 100U;
}

void update_reserve_base(ReserveBaseTracker &tracker, const EXAD::Layer &layer) {
    ReserveFootprint footprint = reserve_footprint_from_layer(layer);
    if (!tracker.initialized) {
        tracker.floor = std::move(footprint);
        tracker.initialized = true;
        return;
    }
    for (size_t slot = 0; slot < bucket_slot_count(); ++slot) {
        tracker.floor.buckets[slot] = std::max(footprint.buckets[slot], decayed_reserve_base(tracker.floor.buckets[slot]));
        tracker.floor.small_bytes[slot] = std::max(footprint.small_bytes[slot], decayed_reserve_base(tracker.floor.small_bytes[slot]));
        tracker.floor.large_words[slot] = std::max(footprint.large_words[slot], decayed_reserve_base(tracker.floor.large_words[slot]));
    }
}

void apply_reserve_base_floors(EXAD::ReserveFactors &factors, const ReserveBaseTracker &tracker) {
    if (!tracker.initialized) {
        return;
    }
    constexpr double kSlowBaseMargin = 1.05;
    uint64_t total_buckets = 0;
    uint64_t total_small = 0;
    uint64_t total_large = 0;
    for (size_t slot = 0; slot < bucket_slot_count(); ++slot) {
        total_buckets += tracker.floor.buckets[slot];
        total_small += tracker.floor.small_bytes[slot];
        total_large += tracker.floor.large_words[slot];
    }
    const uint64_t avg_bucket_floor = total_buckets / bucket_slot_count() + 1U;
    const uint64_t avg_small_floor = total_small / bucket_slot_count() + 1U;
    const uint64_t avg_large_floor = total_large / bucket_slot_count() + 1U;
    for (size_t slot = 0; slot < bucket_slot_count(); ++slot) {
        const uint64_t bucket_base = std::max(tracker.floor.buckets[slot], avg_bucket_floor);
        const uint64_t small_base = std::max(tracker.floor.small_bytes[slot], avg_small_floor);
        const uint64_t large_base = std::max(tracker.floor.large_words[slot], avg_large_floor);
        factors.bucket_floor[slot] = std::max<uint64_t>(
            factors.bucket_floor[slot],
            static_cast<uint64_t>(std::ceil(static_cast<double>(bucket_base) * kSlowBaseMargin))
        );
        factors.small_floor[slot] = std::max<uint64_t>(
            factors.small_floor[slot],
            static_cast<uint64_t>(std::ceil(static_cast<double>(small_base) * kSlowBaseMargin))
        );
        factors.large_floor[slot] = std::max<uint64_t>(
            factors.large_floor[slot],
            static_cast<uint64_t>(std::ceil(static_cast<double>(large_base) * kSlowBaseMargin))
        );
    }
}

void apply_post_validate_floor(EXAD::ReserveFactors &factors, const PostValidateReserveFloor &tracker) {
    if (tracker.remaining_layers <= 0) {
        return;
    }
    constexpr double kPostValidateFloorMargin = 1.20;
    for (size_t slot = 0; slot < bucket_slot_count(); ++slot) {
        factors.bucket_floor[slot] = std::max<uint64_t>(
            factors.bucket_floor[slot],
            static_cast<uint64_t>(std::ceil(static_cast<double>(tracker.floor.buckets[slot]) * kPostValidateFloorMargin))
        );
        factors.small_floor[slot] = std::max<uint64_t>(
            factors.small_floor[slot],
            static_cast<uint64_t>(std::ceil(static_cast<double>(tracker.floor.small_bytes[slot]) * kPostValidateFloorMargin))
        );
        factors.large_floor[slot] = std::max<uint64_t>(
            factors.large_floor[slot],
            static_cast<uint64_t>(std::ceil(static_cast<double>(tracker.floor.large_words[slot]) * kPostValidateFloorMargin))
        );
    }
}

void multiply_reserve_prediction(EXAD::ReserveFactors &factors, double multiplier) {
    if (!(multiplier > 1.0)) {
        return;
    }
    factors.bucket *= multiplier;
    factors.small *= multiplier;
    factors.large *= multiplier;
    for (size_t slot = 0; slot < bucket_slot_count(); ++slot) {
        factors.bucket_floor[slot] = static_cast<uint64_t>(
            std::ceil(static_cast<double>(factors.bucket_floor[slot]) * multiplier)
        );
        factors.small_floor[slot] = static_cast<uint64_t>(
            std::ceil(static_cast<double>(factors.small_floor[slot]) * multiplier)
        );
        factors.large_floor[slot] = static_cast<uint64_t>(
            std::ceil(static_cast<double>(factors.large_floor[slot]) * multiplier)
        );
    }
}

void push_derived_output_history(DerivedOutputHistory &history, uint64_t derived_output_count) {
    history.values.push_back(derived_output_count);
    while (history.values.size() > 6U) {
        history.values.pop_front();
    }
}

bool recent_derived_output_onset(const DerivedOutputHistory &history) {
    const auto &values = history.values;
    if (values.size() < 2U) {
        return false;
    }
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] == 0U) {
            continue;
        }
        size_t zero_run = 0U;
        size_t cursor = i;
        while (cursor > 0U && values[cursor - 1U] == 0U) {
            ++zero_run;
            --cursor;
        }
        if (zero_run != 0U) {
            return true;
        }
    }
    return false;
}

bool derived_burst_guard_active(
    int step,
    uint32_t ini_board_sum,
    int target,
    const DerivedOutputHistory &history
) {
    if (!recent_derived_output_onset(history) || target <= 1 || target >= 31) {
        return false;
    }
    const uint64_t target_tile = 1ULL << static_cast<uint32_t>(target);
    const uint64_t half_target = target_tile >> 1U;
    if (half_target == 0U) {
        return false;
    }
    const uint64_t true_sum = static_cast<uint64_t>(ini_board_sum) + static_cast<uint64_t>(step) * 2ULL;
    const uint64_t phase = true_sum % half_target;
    return phase >= 8ULL && phase <= 16ULL;
}

std::vector<std::string> split_csv_simple(const std::string &line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(std::move(field));
    }
    return fields;
}

DerivedOutputHistory load_recent_derived_output_history(const RunOptions &options, int start_step) {
    DerivedOutputHistory history;
    std::ifstream file(stats_file_path(options));
    if (!file) {
        return history;
    }
    std::string line;
    std::getline(file, line);
    std::map<int, uint64_t> by_step;
    while (std::getline(file, line)) {
        const std::vector<std::string> fields = split_csv_simple(line);
        if (fields.size() <= 7U || fields[0] != "forward") {
            continue;
        }
        try {
            const int step = std::stoi(fields[1]);
            if (step >= start_step) {
                continue;
            }
            by_step[step] = static_cast<uint64_t>(std::stoull(fields[7]));
        } catch (...) {
        }
    }
    std::vector<uint64_t> recent;
    recent.reserve(6U);
    for (auto it = by_step.rbegin(); it != by_step.rend() && recent.size() < 6U; ++it) {
        recent.push_back(it->second);
    }
    for (auto it = recent.rbegin(); it != recent.rend(); ++it) {
        push_derived_output_history(history, *it);
    }
    return history;
}

EXAD::ReserveFactors reserve_factors_for_layer(
    int step,
    int total_steps,
    uint64_t live_board_count,
    const std::deque<ReserveFootprint> &history,
    const ReserveBaseTracker &base_tracker,
    const PostValidateReserveFloor &post_validate_floor,
    uint32_t ini_board_sum,
    int target,
    const DerivedOutputHistory &derived_history
) {
    (void)total_steps;
    EXAD::ReserveFactors factors;
    apply_reserve_history_floors(factors, history);
    apply_reserve_base_floors(factors, base_tracker);
    apply_post_validate_floor(factors, post_validate_floor);

    // Resume/cold-start may not have enough footprint history yet. Keep a
    // conservative low-live fallback only in that case; normal sequential runs
    // should be driven by the history floors above.
    if (history.size() < 3U && live_board_count < 5'000'000ULL) {
        factors.bucket = std::max(factors.bucket, 64.0);
        factors.small = std::max(factors.small, 64.0);
        factors.large = std::max(factors.large, 64.0);
    }
    if (derived_burst_guard_active(step, ini_board_sum, target, derived_history)) {
        multiply_reserve_prediction(factors, 2.5);
    }
    return factors;
}

struct ResumeState {
    int start_step = 1;
    EXAD::Layer current;
    EXAD::CarryLayer next_seed;
};

ResumeState initialize_or_resume(
    const std::vector<uint64_t> &masked_seed,
    uint32_t ini_board_sum,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    const EXAD::Luts &luts,
    const RunOptions &options,
    FileIOUtils::DirectIoConfig io_config,
    int num_threads
) {
    const std::string layer0_path = EXAD::layer_file_path(options.pathname, 0);
    if (!EXAD::layer_file_exists(layer0_path)) {
        const double t0 = wall_time_seconds();
        EXAD::Layer seed_layer = EXAD::build_layer_from_boards(
            masked_seed,
            ini_board_sum,
            tiles_table,
            param,
            luts,
            num_threads
        );
        const double t1 = wall_time_seconds();
        EXAD::write_layer_file(layer0_path, seed_layer, io_config, options.compress_temp_files);
        StatsRecord init;
        init.stage = "init";
        init.step = 0;
        init.input_live = masked_seed.size();
        init.output_live = seed_layer.live_board_count;
        init.finalize_seconds = t1 - t0;
        append_stats(options, init);
    }

    int first_missing = 1;
    while (first_missing < options.steps - 1 &&
           EXAD::layer_file_exists(EXAD::layer_file_path(options.pathname, first_missing))) {
        ++first_missing;
    }
    ResumeState state;
    if (first_missing <= 1) {
        state.start_step = 1;
        state.current = EXAD::read_layer_file(layer0_path, io_config);
        if (!EXAD::physical_metadata_matches(state.current, luts)) {
            throw std::runtime_error("EXAD temp layer physical metadata does not match LUT");
        }
        return state;
    }

    state.start_step = first_missing - 1;
    state.current = EXAD::read_layer_file(EXAD::layer_file_path(options.pathname, state.start_step - 1), io_config);
    if (!EXAD::physical_metadata_matches(state.current, luts)) {
        throw std::runtime_error("EXAD temp layer physical metadata does not match LUT");
    }
    EXAD::Layer seed_layer = EXAD::read_layer_file(EXAD::layer_file_path(options.pathname, state.start_step), io_config);
    if (!EXAD::physical_metadata_matches(seed_layer, luts)) {
        throw std::runtime_error("EXAD temp layer physical metadata does not match LUT");
    }
    state.next_seed = EXAD::carry_from_layer(seed_layer, luts, num_threads);
    return state;
}

} // namespace

void run_pattern_build_exad_cpp(
    const std::vector<uint64_t> &arr_init,
    const AdvancedPatternSpec &spec,
    const RunOptions &options
) {
    NDIAG_START_RUN("run_pattern_build_exad_cpp", spec, options, static_cast<uint64_t>(arr_init.size()));
    NDIAG_CHECKPOINT("exad_build_entry", 0, static_cast<uint64_t>(arr_init.size()), 0, 0);
    FormationProgress::reset_build_progress(build_progress_total(options));
    ensure_stats_header(options);
    const int num_threads = options.num_threads > 0 ? options.num_threads :
#if defined(_OPENMP)
        std::max(4, std::min(32, omp_get_max_threads()));
#else
        4;
#endif
    const FileIOUtils::DirectIoConfig io_config = FileIOUtils::direct_io_config_from_options(options);
    NDIAG_CHECKPOINT("exad_init_masker_begin", 0, static_cast<uint64_t>(num_threads), 0, 0);
    FormationAD::MaskerContext masker = FormationAD::init_masker(spec);
    NDIAG_CHECKPOINT("exad_lut_load_or_build_begin", 0, static_cast<uint64_t>(arr_init.size()), 0, 0);
    EXAD::Luts luts = load_or_build_luts(arr_init, spec, options, num_threads, io_config);
    NDIAG_CHECKPOINT("exad_lut_load_or_build_end", 0, static_cast<uint64_t>(luts.size_table.size()), 0, 0);

    const uint32_t ini_board_sum = board_sum_from_seed(arr_init);
    std::vector<uint64_t> masked_seed = arr_init;
    for (uint64_t &board : masked_seed) {
        board = FormationAD::mask_board(board);
    }

    ResumeState resume = initialize_or_resume(
        masked_seed,
        ini_board_sum,
        masker.tiles_combination_table,
        masker.param,
        luts,
        options,
        io_config,
        num_threads
    );
    NDIAG_CHECKPOINT(
        "exad_resume_ready",
        resume.start_step,
        resume.current.live_board_count,
        EXAD::carry_bucket_count(resume.next_seed),
        0
    );

    EXAD::Layer current = std::move(resume.current);
    EXAD::CarryLayer next_seed = std::move(resume.next_seed);
    EXAD::DeriveHashState derive_hash_state;
    std::deque<ReserveFootprint> reserve_history;
    ReserveBaseTracker reserve_base;
    PostValidateReserveFloor post_validate_floor;
    DerivedOutputHistory derived_history = load_recent_derived_output_history(options, resume.start_step);
    const int history_begin = std::max(0, resume.start_step - 5);
    for (int history_step = history_begin; history_step < resume.start_step; ++history_step) {
        const std::string path = EXAD::layer_file_path(options.pathname, history_step);
        if (!EXAD::layer_file_exists(path)) {
            continue;
        }
        EXAD::Layer history_layer = EXAD::read_layer_file(path, io_config);
        if (!EXAD::physical_metadata_matches(history_layer, luts)) {
            throw std::runtime_error("EXAD temp layer physical metadata does not match LUT");
        }
        push_reserve_history(reserve_history, history_layer);
        update_reserve_base(reserve_base, history_layer);
    }
    push_reserve_history(reserve_history, current);
    update_reserve_base(reserve_base, current);
    StatsRecord total;
    total.stage = "_total";
    const uint32_t progress_total = build_progress_total(options);

    for (int step = resume.start_step; step < options.steps - 1; ++step) {
        NDIAG_CHECKPOINT("exad_forward_step_begin", step, current.live_board_count, 0, 0);
        FormationProgress::update_build_progress(static_cast<uint32_t>(step), progress_total);
        const EXAD::ReserveFactors reserve_factors =
            reserve_factors_for_layer(
                step,
                options.steps,
                current.live_board_count,
                reserve_history,
                reserve_base,
                post_validate_floor,
                ini_board_sum,
                options.target,
                derived_history
            );
        NDIAG_CHECKPOINT(
            "exad_forward_reserve_ready",
            step,
            static_cast<uint64_t>(reserve_factors.bucket * 1000.0),
            static_cast<uint64_t>(reserve_factors.small * 1000.0),
            static_cast<uint64_t>(reserve_factors.large * 1000.0)
        );
        if (post_validate_floor.remaining_layers > 0) {
            --post_validate_floor.remaining_layers;
        }
        NDIAG_CHECKPOINT(
            "exad_generate_two_layers_begin",
            step,
            current.live_board_count,
            EXAD::carry_bucket_count(next_seed),
            0
        );
        EXAD::GeneratePairResult generated = EXAD::generate_two_layers_carry(
            current,
            spec,
            options,
            masker.tiles_combination_table,
            masker.param,
            luts,
            num_threads,
            std::move(next_seed),
            reserve_factors,
            &derive_hash_state
        );
        NDIAG_CHECKPOINT(
            "exad_generate_two_layers_end",
            step,
            generated.stats.input_live,
            generated.stats.derive_candidate_count,
            generated.stats.derived_output_count
        );
        current = EXAD::Layer{};
        const double finalize0 = wall_time_seconds();
        NDIAG_CHECKPOINT("exad_finalize_arr1_begin", step, EXAD::carry_bucket_count(generated.arr1), 0, 0);
        EXAD::Layer merged = EXAD::finalize_carry_layer(generated.arr1, luts, num_threads);
        const double finalize1 = wall_time_seconds();
        NDIAG_CHECKPOINT("exad_finalize_arr1_end", step, merged.live_board_count, 0, 0);
        next_seed = std::move(generated.arr2);

        StatsRecord record;
        record.stage = "forward";
        record.step = step;
        record.input_live = generated.stats.input_live;
        record.arr1_live = merged.live_board_count;
        record.arr2_live = EXAD::carry_bucket_count(next_seed);
        record.output_live = merged.live_board_count;
        record.derive_candidate_count = generated.stats.derive_candidate_count;
        record.derived_output_count = generated.stats.derived_output_count;
        record.prepare_seconds = generated.stats.prepare_seconds;
        record.prepare_estimate_seconds = generated.stats.prepare_estimate_seconds;
        record.prepare_arr1_seconds = generated.stats.prepare_arr1_seconds;
        record.prepare_arr2_seconds = generated.stats.prepare_arr2_seconds;
        record.hashmap_seconds = generated.stats.hashmap_seconds;
        record.worklist_seconds = generated.stats.worklist_seconds;
        record.loop_seconds = generated.stats.loop_seconds;
        record.count_finalize_seconds = generated.stats.count_finalize_seconds;
        record.generate_seconds = generated.stats.generate_seconds;
        record.insert_seconds = generated.stats.insert_seconds;
        record.finalize_seconds = finalize1 - finalize0;
        record.retry_count = generated.stats.retry_count;

        const double validate0 = wall_time_seconds();
        if (validate_step_trigger(step, ini_board_sum, masker.param)) {
            NDIAG_CHECKPOINT("exad_validate_begin", step, merged.live_board_count, EXAD::carry_bucket_count(next_seed), 0);
            post_validate_floor.floor = reserve_footprint_from_layer(merged);
            post_validate_floor.remaining_layers = 3;

            const uint64_t before_current = merged.live_board_count;
            merged = EXAD::validate_layer_streaming(
                merged,
                static_cast<uint32_t>(2 * step + ini_board_sum),
                masker.tiles_combination_table,
                masker.param,
                luts,
                num_threads
            );
            record.validate_removed_current = before_current - merged.live_board_count;

            EXAD::Layer next_layer = EXAD::finalize_carry_layer(next_seed, luts, num_threads);
            const uint64_t before_next = next_layer.live_board_count;
            next_layer = EXAD::validate_layer_streaming(
                next_layer,
                static_cast<uint32_t>(2 * step + ini_board_sum + 2U),
                masker.tiles_combination_table,
                masker.param,
                luts,
                num_threads
            );
            record.validate_removed_next = before_next - next_layer.live_board_count;
            next_seed = EXAD::carry_from_layer(next_layer, luts, num_threads);
            NDIAG_CHECKPOINT("exad_validate_end", step, merged.live_board_count, EXAD::carry_bucket_count(next_seed), 0);
        }
        record.validate_seconds = wall_time_seconds() - validate0;

        const double write0 = wall_time_seconds();
        NDIAG_CHECKPOINT("exad_write_layer_begin", step, merged.live_board_count, 0, 0);
        EXAD::write_layer_file(
            EXAD::layer_file_path(options.pathname, step),
            merged,
            io_config,
            options.compress_temp_files
        );
        record.write_seconds = wall_time_seconds() - write0;
        NDIAG_CHECKPOINT("exad_write_layer_end", step, merged.live_board_count, 0, 0);
        append_stats(options, record);

        total.input_live += record.input_live;
        total.arr1_live += record.arr1_live;
        total.arr2_live += record.arr2_live;
        total.output_live += record.output_live;
        total.derive_candidate_count += record.derive_candidate_count;
        total.derived_output_count += record.derived_output_count;
        total.validate_removed_current += record.validate_removed_current;
        total.validate_removed_next += record.validate_removed_next;
        total.prepare_seconds += record.prepare_seconds;
        total.prepare_estimate_seconds += record.prepare_estimate_seconds;
        total.prepare_arr1_seconds += record.prepare_arr1_seconds;
        total.prepare_arr2_seconds += record.prepare_arr2_seconds;
        total.hashmap_seconds += record.hashmap_seconds;
        total.worklist_seconds += record.worklist_seconds;
        total.loop_seconds += record.loop_seconds;
        total.count_finalize_seconds += record.count_finalize_seconds;
        total.generate_seconds += record.generate_seconds;
        total.insert_seconds += record.insert_seconds;
        total.finalize_seconds += record.finalize_seconds;
        total.validate_seconds += record.validate_seconds;
        total.write_seconds += record.write_seconds;
        total.retry_count += record.retry_count;

        push_derived_output_history(derived_history, record.derived_output_count);
        push_reserve_history(reserve_history, merged);
        update_reserve_base(reserve_base, merged);
        current = std::move(merged);
    }

    append_stats(options, total);
    NDIAG_CHECKPOINT("exad_build_forward_complete", options.steps, total.input_live, total.output_live, 0);
    run_pattern_solve_exad_cpp(arr_init, spec, options);
}
