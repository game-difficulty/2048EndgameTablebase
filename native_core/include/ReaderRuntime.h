#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "BCCompressedResult.h"
#include "BoardMaskerAD.h"
#include "FormationRuntime.h"

enum class ReaderValueKind {
    NoneValue,
    Numeric,
    String,
};

struct OrderedReaderEntry {
    std::string key;
    ReaderValueKind kind = ReaderValueKind::NoneValue;
    double number = 0.0;
    std::string text;
};

struct ReaderMoveResult {
    std::vector<OrderedReaderEntry> entries;
    std::string success_rate_dtype;
};

class ClassicBookReader {
public:
    ClassicBookReader(PatternSpec spec, bool is_variant = false);

    ReaderMoveResult move_on_dic(
        const std::vector<std::vector<int>> &board,
        const std::vector<std::pair<std::string, std::string>> &path_list,
        const std::string &pattern_full,
        int64_t nums_adjust
    );

    uint64_t get_random_state(
        const std::vector<std::pair<std::string, std::string>> &path_list,
        const std::string &pattern_full,
        double spawn_rate4
    ) const;

public:
    PatternSpec spec_;
    bool is_variant_ = false;
    bool prefer_max_result_ = false;
    int last_operation_index_ = 0;
};

class AdvancedBookReader {
public:
    AdvancedBookReader(AdvancedPatternSpec spec, bool is_variant = false);

    ReaderMoveResult move_on_dic(
        const std::vector<std::vector<int>> &board,
        const std::vector<std::pair<std::string, std::string>> &path_list,
        const std::string &pattern_full,
        int64_t nums_adjust
    );

    uint64_t get_random_state(
        const std::vector<std::pair<std::string, std::string>> &path_list,
        const std::string &pattern_full,
        double spawn_rate4
    ) const;

public:
    AdvancedPatternSpec spec_;
    FormationAD::MaskerContext masker_;
    bool is_variant_ = false;
    bool prefer_max_result_ = false;
    int last_operation_index_ = 0;
};

class EXADBookReader {
public:
    EXADBookReader(AdvancedPatternSpec spec, bool is_variant = false);

    ReaderMoveResult move_on_dic(
        const std::vector<std::vector<int>> &board,
        const std::vector<std::pair<std::string, std::string>> &path_list,
        const std::string &pattern_full,
        int64_t nums_adjust
    );

    uint64_t get_random_state(
        const std::vector<std::pair<std::string, std::string>> &path_list,
        const std::string &pattern_full,
        double spawn_rate4
    ) const;

public:
    AdvancedPatternSpec spec_;
    FormationAD::MaskerContext masker_;
    bool is_variant_ = false;
    bool prefer_max_result_ = false;
    int last_operation_index_ = 0;
};

class EXBookReader {
public:
    EXBookReader(PatternSpec spec, bool is_variant = false);

    ReaderMoveResult move_on_dic(
        const std::vector<std::vector<int>> &board,
        const std::vector<std::pair<std::string, std::string>> &path_list,
        const std::string &pattern_full,
        int64_t nums_adjust
    );

    uint64_t get_random_state(
        const std::vector<std::pair<std::string, std::string>> &path_list,
        const std::string &pattern_full,
        double spawn_rate4
    ) const;

public:
    PatternSpec spec_;
    bool is_variant_ = false;
    int last_operation_index_ = 0;
};

class BCBookReader {
public:
    BCBookReader(PatternSpec spec, uint32_t target_rank, bool is_variant = false);

    ReaderMoveResult move_on_dic(
        const std::vector<std::vector<int>> &board,
        const std::vector<std::pair<std::string, std::string>> &path_list,
        const std::string &pattern_full,
        int64_t nums_adjust
    );

    uint64_t get_random_state(
        const std::vector<std::pair<std::string, std::string>> &path_list,
        const std::string &pattern_full,
        double spawn_rate4,
        int64_t nums_adjust = 0
    ) const;

public:
    PatternSpec spec_;
    uint32_t target_rank_ = 8U;
    bool is_variant_ = false;
    int last_operation_index_ = 0;
};

namespace BCRuntime {

std::vector<uint8_t> legal_tiles(uint32_t target_rank);

std::string dtype_name(uint32_t dtype);

double normalize_lookup_value(const BCCompressedResult::ColdLookupResult &lookup);

BCCompressedResult::ColdLookupResult lookup_compressed_result_cached(
    const std::filesystem::path &compressed_path,
    uint32_t target_rank,
    uint64_t board,
    uint32_t lane = 0U
);

BCCompressedResult::ColdLookupResult lookup_exact_result_cached(
    const std::filesystem::path &position_path,
    const std::filesystem::path &success_path,
    uint32_t target_rank,
    uint64_t board,
    uint32_t lane = 0U
);

bool sample_compressed_result_cached(
    const std::filesystem::path &compressed_path,
    uint32_t target_rank,
    uint64_t &board,
    uint64_t &raw_value_bits,
    double &numeric_value,
    uint32_t lane = 0U
);

uint64_t sample_exact_random_board_cached(
    const std::filesystem::path &position_path,
    uint32_t target_rank
);

} // namespace BCRuntime

double find_classic_value_native(
    const std::string &pathname,
    const std::string &filename,
    uint64_t search_key,
    const std::string &success_rate_dtype,
    bool &found
);

std::optional<double> trie_decompress_search_cached_native(
    const std::string &path_prefix,
    uint64_t board,
    const std::string &success_rate_dtype
);
