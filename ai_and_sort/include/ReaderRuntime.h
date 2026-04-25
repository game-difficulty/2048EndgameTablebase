#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
