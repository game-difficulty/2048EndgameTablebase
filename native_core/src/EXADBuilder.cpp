#include "EXADBuilder.h"

#include "BookGenerator.h"
#include "BookGeneratorUtils.h"
#include "BoardMoverAD.h"
#include "Calculator.h"
#include "CanonicalBatch.h"
#include "Formation.h"
#include "NativeDiagnostics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace EXAD {

namespace {

constexpr uint64_t kEmptyKey = std::numeric_limits<uint64_t>::max();
constexpr uint32_t kPendingOffset = std::numeric_limits<uint32_t>::max();
constexpr double kHashLoadUpper = 0.60;
constexpr uint32_t kInsertBufferSize = 128;
constexpr uint32_t kSlotInsertBufferSize = 256;
constexpr uint32_t kCandidateBufferSize = 4096;
constexpr uint32_t kDerivedCanonicalBufferSize = 4096;
constexpr uint32_t kPrefetchDistance = 32;
constexpr uint32_t kDeriveHashPrefetchDistance = 16;
constexpr uint32_t kDynamicSmallChunkBytes = 4U * 1024U;
constexpr uint32_t kDynamicLargeChunkWords = 512U;
constexpr uint32_t kAdBucketLookupMaxCount = 16U;
constexpr uint32_t kAdBucketLookupMaxTotalSum = 16U * 32U;
constexpr uint32_t kAdBucketLookupStride = kAdBucketLookupMaxTotalSum + 1U;

constexpr uint32_t kOverflowSmallArena = 1U;
constexpr uint32_t kOverflowLargeArena = 2U;
constexpr uint32_t kOverflowHashFull = 3U;
constexpr uint8_t kBufferedFromDeriveFlag = 0x80U;
constexpr uint8_t kBufferedSlotMask = 0x3FU;
constexpr uint8_t kBufferedUnknownSlot = kBufferedSlotMask;

constexpr std::array<uint64_t, 60> kPrimeCapacities = {
    100003ULL, 120011ULL, 144031ULL, 172849ULL, 207433ULL, 248971ULL,
    298777ULL, 358541ULL, 430259ULL, 516319ULL, 619583ULL, 743507ULL,
    892219ULL, 1070681ULL, 1284823ULL, 1541791ULL, 1850159ULL, 2220193ULL,
    2664241ULL, 3197101ULL, 3836587ULL, 4603913ULL, 5524697ULL, 6629647ULL,
    7955579ULL, 9546697ULL, 11456041ULL, 13747277ULL, 16496791ULL, 19796173ULL,
    23755411ULL, 28506497ULL, 34207807ULL, 41049383ULL, 49259281ULL, 59111149ULL,
    70933397ULL, 85120081ULL, 102144103ULL, 122572943ULL, 147087533ULL, 176505047ULL,
    211806059ULL, 254167301ULL, 305000767ULL, 366000923ULL, 439201109ULL, 527041337ULL,
    632449619ULL, 758939549ULL, 910727479ULL, 1092872981ULL, 1311447581ULL, 1573737133ULL,
    1888484569ULL, 2266181501ULL, 2719417817ULL, 3263301383ULL, 3915961681ULL, 4294967291ULL
};

struct PendingInsert {
    uint64_t packed_key = 0;
    uint32_t valid_count = 0;
    uint32_t rank = 0;
    uint32_t home_slot = 0;
    uint8_t ad_slot = 0;
};
static_assert(sizeof(PendingInsert) == 24, "PendingInsert should stay cache-compact");

struct SlotPendingInsert {
    uint64_t packed_key;
    uint32_t valid_count;
    uint32_t rank;
    uint32_t home_slot;
};
static_assert(sizeof(SlotPendingInsert) == 24, "SlotPendingInsert should stay cache-compact");

struct ResolvedInsert {
    uint8_t ad_slot = 0;
    uint32_t bitmap_offset = 0;
    uint32_t valid_count = 0;
    uint32_t rank = 0;
};

struct ThreadChunkState {
    uint32_t small_next = 0;
    uint32_t small_end = 0;
    uint32_t large_next = 0;
    uint32_t large_end = 0;
};

struct Estimate {
    uint64_t buckets = 0;
    uint64_t small_bytes = 0;
    uint64_t large_words = 0;
};

struct EstimateSet {
    std::array<Estimate, bucket_slot_count()> slots{};
};

struct DeriveResult {
    bool is_valid;
    bool is_derived;
    uint8_t count;
    std::array<uint64_t, 120> boards;

    DeriveResult() noexcept : is_valid(false), is_derived(false), count(0) {}
};

struct AdBucketStats {
    uint32_t total_sum = 0;
    uint8_t count_32k = 0;
};

struct AdBucketLookup {
    std::array<int8_t, (kAdBucketLookupMaxCount + 1U) * kAdBucketLookupStride> key{};
};

inline size_t ad_bucket_lookup_index(uint32_t count_32k, uint32_t total_sum) {
    return static_cast<size_t>(count_32k * kAdBucketLookupStride + total_sum);
}

const std::array<uint16_t, 65536> &ad_bucket_row_stats_table() {
    static const std::array<uint16_t, 65536> table = []() {
        std::array<uint16_t, 65536> out{};
        const auto &source = FormationAD::info_table1();
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = static_cast<uint16_t>(
                (static_cast<uint16_t>(source[i].count_32k) << 9U) |
                static_cast<uint16_t>(source[i].total_sum)
            );
        }
        return out;
    }();
    return table;
}

inline AdBucketStats ad_bucket_stats(uint64_t board, const AdvancedMaskParam &param) {
    const uint64_t non_fixed_board = board & ~param.pos_fixed_32k_mask;
    const auto &table = ad_bucket_row_stats_table();
    const uint16_t r0 = table[static_cast<uint16_t>(non_fixed_board & 0xFFFFULL)];
    const uint16_t r1 = table[static_cast<uint16_t>((non_fixed_board >> 16U) & 0xFFFFULL)];
    const uint16_t r2 = table[static_cast<uint16_t>((non_fixed_board >> 32U) & 0xFFFFULL)];
    const uint16_t r3 = table[static_cast<uint16_t>((non_fixed_board >> 48U) & 0xFFFFULL)];
    return AdBucketStats{
        static_cast<uint32_t>((r0 & 0x1FFU) + (r1 & 0x1FFU) + (r2 & 0x1FFU) + (r3 & 0x1FFU)),
        static_cast<uint8_t>((r0 >> 9U) + (r1 >> 9U) + (r2 >> 9U) + (r3 >> 9U))
    };
}

AdBucketLookup make_ad_bucket_lookup(
    uint32_t original_board_sum,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param
) {
    AdBucketLookup lookup;
    lookup.key.fill(-1);
    for (uint32_t count_32k = 0; count_32k <= kAdBucketLookupMaxCount; ++count_32k) {
        for (uint32_t total_sum = 0; total_sum <= kAdBucketLookupMaxTotalSum; ++total_sum) {
            int8_t ad_key = -1;
            if (count_32k == param.num_free_32k) {
                ad_key = static_cast<int8_t>(count_32k);
            } else {
                const int32_t remaining_count = static_cast<int32_t>(count_32k) - static_cast<int32_t>(param.num_free_32k);
                if (remaining_count >= 0 && remaining_count <= 9) {
                    const int64_t large_tiles_sum64 =
                        static_cast<int64_t>(original_board_sum) -
                        static_cast<int64_t>(total_sum) -
                        (static_cast<int64_t>(param.num_free_32k + param.num_fixed_32k) << 15);
                    if (large_tiles_sum64 >= 0 && (large_tiles_sum64 & 63LL) == 0LL) {
                        const uint64_t large_tiles_sum = static_cast<uint64_t>(large_tiles_sum64);
                        if ((large_tiles_sum >> 6U) <= 255ULL) {
                            auto tiles = FormationAD::tiles_combination_view(
                                tiles_table,
                                static_cast<uint8_t>(large_tiles_sum >> 6U),
                                static_cast<uint8_t>(remaining_count)
                            );
                            if (!tiles.empty()) {
                                if (remaining_count == 1) {
                                    ad_key = static_cast<int8_t>(count_32k);
                                } else if (tiles.size > 1 && tiles[0] == tiles[1]) {
                                    if (tiles.size > 2 && tiles[0] == tiles[2]) {
                                        ad_key = static_cast<int8_t>(static_cast<int32_t>(count_32k) - 3 + 16);
                                    } else {
                                        ad_key = static_cast<int8_t>(-static_cast<int32_t>(count_32k));
                                    }
                                } else {
                                    ad_key = static_cast<int8_t>(count_32k);
                                }
                            }
                        }
                    }
                }
            }
            lookup.key[ad_bucket_lookup_index(count_32k, total_sum)] = ad_key;
        }
    }
    return lookup;
}

inline int8_t ad_bucket_key_for_board_fast(
    uint64_t board,
    const AdvancedMaskParam &param,
    const AdBucketLookup &lookup
) {
    const AdBucketStats stats = ad_bucket_stats(board, param);
    if (stats.count_32k > kAdBucketLookupMaxCount || stats.total_sum > kAdBucketLookupMaxTotalSum) {
        return -1;
    }
    return lookup.key[ad_bucket_lookup_index(stats.count_32k, stats.total_sum)];
}

inline int effective_threads(int requested) {
#if defined(_OPENMP)
    return requested > 0 ? requested : std::max(1, omp_get_max_threads());
#else
    return requested > 0 ? requested : 1;
#endif
}

inline double now_seconds() {
#if defined(_OPENMP)
    return omp_get_wtime();
#else
    return 0.0;
#endif
}

size_t derive_input_hash_length(uint64_t current_live) {
    constexpr uint64_t kDefaultDivisor = 8ULL;
    constexpr uint64_t kMinSlots = 262144ULL;
    const uint64_t target = std::max<uint64_t>(current_live / kDefaultDivisor + 1ULL, kMinSlots);
    return static_cast<size_t>(BookGeneratorUtils::largest_power_of_2(target));
}

std::unique_ptr<uint64_t[]> make_zeroed_hashmap(size_t length, int thread_count) {
    std::unique_ptr<uint64_t[]> data(new uint64_t[length]);
#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int64_t i = 0; i < static_cast<int64_t>(length); ++i) {
        data[static_cast<size_t>(i)] = 0ULL;
    }
    return data;
}

void zero_hashmap_vector(std::vector<uint64_t> &hashmap, int thread_count) {
#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int64_t i = 0; i < static_cast<int64_t>(hashmap.size()); ++i) {
        hashmap[static_cast<size_t>(i)] = 0ULL;
    }
}

void grow_inherited_hashmap(std::vector<uint64_t> &hashmap, size_t length, int thread_count) {
    if (hashmap.size() >= length) {
        return;
    }
    if (hashmap.empty()) {
        hashmap.assign(length, 0ULL);
        return;
    }
    const size_t old_size = hashmap.size();
    size_t new_size = old_size;
    while (new_size < length) {
        new_size <<= 1U;
    }
    std::vector<uint64_t> grown(new_size);
#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int64_t i = 0; i < static_cast<int64_t>(new_size); ++i) {
        grown[static_cast<size_t>(i)] = hashmap[static_cast<size_t>(i) & (old_size - 1U)];
    }
    hashmap = std::move(grown);
}

void prepare_fresh_hashmap(std::vector<uint64_t> &hashmap, size_t length, int thread_count) {
    if (hashmap.size() < length) {
        hashmap.resize(length);
    }
    zero_hashmap_vector(hashmap, thread_count);
}

size_t prepare_derive_hashmaps(DeriveHashState &state, size_t desired_length, int thread_count) {
    grow_inherited_hashmap(state.next1, desired_length, thread_count);
    const size_t length = state.next1.size();
    // The +4 target is new for this step; only the +2 target can inherit the previous +4 table.
    prepare_fresh_hashmap(state.next2, length, thread_count);
    return length;
}

void reset_derive_hashmaps(DeriveHashState &state, int thread_count) {
    zero_hashmap_vector(state.next1, thread_count);
    zero_hashmap_vector(state.next2, thread_count);
}

void rotate_derive_hashmaps(DeriveHashState &state) {
    std::swap(state.next1, state.next2);
}

inline bool derive_input_seen_or_record(uint64_t *hashmap, uint64_t hashmask, uint64_t board) {
    const size_t hash_index = static_cast<size_t>(BookGeneratorUtils::hash_board(board) & hashmask);
    if (hashmap[hash_index] == board) {
        return true;
    }
    hashmap[hash_index] = board;
    return false;
}

inline bool derive_input_seen_or_record_at(uint64_t *hashmap, size_t hash_index, uint64_t board) {
    if (hashmap[hash_index] == board) {
        return true;
    }
    hashmap[hash_index] = board;
    return false;
}

inline uint32_t popcount64(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_popcountll(value));
#else
    uint32_t count = 0;
    while (value != 0ULL) {
        value &= (value - 1ULL);
        ++count;
    }
    return count;
#endif
}

inline uint32_t countr_zero64(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_ctzll(value));
#else
    uint32_t count = 0;
    while (((value >> count) & 1ULL) == 0ULL) {
        ++count;
    }
    return count;
#endif
}

inline uint32_t choose_hash_capacity(uint64_t bucket_estimate) {
    const uint64_t required = static_cast<uint64_t>(std::ceil(static_cast<double>(std::max<uint64_t>(bucket_estimate, 1U)) / kHashLoadUpper));
    for (uint64_t candidate : kPrimeCapacities) {
        if (candidate >= required) {
            return static_cast<uint32_t>(candidate);
        }
    }
    throw std::runtime_error("EXAD builder hash capacity exceeds supported range");
}

inline uint32_t hash_slot_for_key(uint64_t key, uint32_t capacity) {
    const uint64_t prefix36 = key >> 18U;
    const uint64_t mixed = prefix36 * 11400714819323198485ULL;
#if defined(__SIZEOF_INT128__)
    return static_cast<uint32_t>((static_cast<unsigned __int128>(mixed) * capacity) >> 64U);
#else
    return static_cast<uint32_t>(mixed % capacity);
#endif
}

uint64_t zero_nibble_lsb_mask(uint64_t board) {
    constexpr uint64_t kNibbleLowBits = 0x1111111111111111ULL;
    const uint64_t occupied_low_bits =
        (board | (board >> 1U) | (board >> 2U) | (board >> 3U)) & kNibbleLowBits;
    return (~occupied_low_bits) & kNibbleLowBits;
}

const std::array<uint32_t, 16> &reverse_spawn_shifts() {
    static const std::array<uint32_t, 16> table = []() {
        std::array<uint32_t, 16> out{};
        for (uint32_t cell = 0; cell < 16U; ++cell) {
            const uint64_t reversed = FormationAD::reverse(1ULL << (cell * 4U));
            out[cell] = countr_zero64(reversed);
        }
        return out;
    }();
    return table;
}

uint64_t apply_canonical(uint64_t board, int symm_mode) {
    switch (static_cast<SymmMode>(symm_mode)) {
        case SymmMode::Full:
            return Calculator::canonical_full(board);
        case SymmMode::Diagonal:
            return Calculator::canonical_diagonal(board);
        case SymmMode::Horizontal:
            return Calculator::canonical_horizontal(board);
        case SymmMode::Min33:
            return Calculator::canonical_min33(board);
        case SymmMode::Min24:
            return Calculator::canonical_min24(board);
        case SymmMode::Min34:
            return Calculator::canonical_min34(board);
        case SymmMode::Min34Top:
            return Calculator::canonical_min34_top(board);
        case SymmMode::Identity:
        default:
            return Calculator::canonical_identity(board);
    }
}

uint8_t derive_3x64(uint64_t masked, const std::array<uint64_t, 16> &pos_32k, int8_t count_32k, std::array<uint64_t, 120> &out) {
    uint8_t count = 0;
    for (int idx = 0; idx < count_32k; ++idx) {
        const uint64_t pos = pos_32k[static_cast<size_t>(idx)];
        const uint64_t tile_value = (masked >> pos) & 0xFULL;
        if (tile_value == 6ULL) {
            continue;
        }
        const uint64_t delta = ((tile_value ^ 6ULL) << pos);
        out[static_cast<size_t>(count++)] = masked ^ delta;
    }
    return count;
}

FormationAD::TileCount3Result tile_sum_and_32k_count3_fast(uint64_t board, const AdvancedMaskParam &param) {
    FormationAD::TileCount3Result result;
    result.masked_board = board;

    const uint64_t non_fixed_board = board & ~param.pos_fixed_32k_mask;
    const auto &table = FormationAD::info_table1();
    const FormationAD::InfoEntry &r0 = table[static_cast<uint16_t>(non_fixed_board & 0xFFFFULL)];
    const FormationAD::InfoEntry &r1 = table[static_cast<uint16_t>((non_fixed_board >> 16U) & 0xFFFFULL)];
    const FormationAD::InfoEntry &r2 = table[static_cast<uint16_t>((non_fixed_board >> 32U) & 0xFFFFULL)];
    const FormationAD::InfoEntry &r3 = table[static_cast<uint16_t>((non_fixed_board >> 48U) & 0xFFFFULL)];

    result.total_sum = static_cast<uint32_t>(r0.total_sum + r1.total_sum + r2.total_sum + r3.total_sum);
    result.count_32k = static_cast<int8_t>(r0.count_32k + r1.count_32k + r2.count_32k + r3.count_32k);

    uint64_t pos_bitmap = static_cast<uint64_t>(r0.pos_bitmap)
        | (static_cast<uint64_t>(r1.pos_bitmap) << 16U)
        | (static_cast<uint64_t>(r2.pos_bitmap) << 32U)
        | (static_cast<uint64_t>(r3.pos_bitmap) << 48U);

    int tile64_count = 0;
    uint64_t tile64_pos = 0;
    int idx = 0;
    while (pos_bitmap != 0ULL) {
#if defined(__GNUC__) || defined(__clang__)
        const unsigned msb = 63U - static_cast<unsigned>(__builtin_clzll(pos_bitmap));
        const uint64_t shift = static_cast<uint64_t>(msb & ~3U);
#else
        int shift_int = 60;
        while (((pos_bitmap >> static_cast<uint64_t>(shift_int)) & 0xFULL) == 0ULL) {
            shift_int -= 4;
        }
        const uint64_t shift = static_cast<uint64_t>(shift_int);
#endif
        result.pos_32k[static_cast<size_t>(idx++)] = shift;
        if (((board >> shift) & 0xFULL) == 6ULL) {
            ++tile64_count;
            tile64_pos = shift;
        }
        pos_bitmap &= ~(0xFULL << shift);
    }
    result.tile64_count = tile64_count;
    if (tile64_count == 1) {
        result.masked_board &= ~(0xFULL << tile64_pos);
    }
    return result;
}

DeriveResult derive(
    uint64_t board,
    uint32_t original_board_sum,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param
) {
    FormationAD::TileCount3Result stats = tile_sum_and_32k_count3_fast(board, param);
    if (stats.total_sum >= param.small_tile_sum_limit + 64U) {
        return {};
    }
    const uint32_t large_tiles_sum = original_board_sum - stats.total_sum
        - (static_cast<uint32_t>(param.num_free_32k + param.num_fixed_32k) << 15U);
    auto tiles_combinations = FormationAD::tiles_combination_view(
        tiles_table,
        static_cast<uint8_t>(large_tiles_sum >> 6U),
        static_cast<uint8_t>(stats.count_32k - static_cast<int8_t>(param.num_free_32k))
    );
    if (tiles_combinations.empty()) {
        return {};
    }
    if (tiles_combinations[tiles_combinations.size - 1] == param.target) {
        return {};
    }
    if (stats.count_32k - static_cast<int8_t>(param.num_free_32k) < 2) {
        DeriveResult result;
        result.is_valid = true;
        return result;
    }

    if (tiles_combinations[0] == tiles_combinations[1]) {
        if (stats.total_sum >= param.small_tile_sum_limit) {
            return {};
        }
        if (tiles_combinations.size > 2 && tiles_combinations[0] == tiles_combinations[2]) {
            if (stats.total_sum >= param.small_tile_sum_limit - 64U) {
                return {};
            }
            DeriveResult result;
            result.is_valid = true;
            result.is_derived = true;
            if (stats.tile64_count != 0) {
                result.count = derive_3x64(board, stats.pos_32k, stats.count_32k, result.boards);
            }
            return result;
        }
        DeriveResult result;
        result.is_valid = true;
        result.is_derived = true;
        const uint64_t target_value = static_cast<uint64_t>(tiles_combinations[0]);
        std::array<uint64_t, 16> deltas{};
        for (int idx = 0; idx < stats.count_32k; ++idx) {
            const uint64_t pos = stats.pos_32k[static_cast<size_t>(idx)];
            const uint64_t tile_value = (board >> pos) & 0xFULL;
            deltas[static_cast<size_t>(idx)] = ((tile_value ^ target_value) << pos);
        }
        for (int pos1 = 0; pos1 < stats.count_32k - 1; ++pos1) {
            const uint64_t base = board ^ deltas[static_cast<size_t>(pos1)];
            for (int pos2 = pos1 + 1; pos2 < stats.count_32k; ++pos2) {
                result.boards[static_cast<size_t>(result.count++)] = base ^ deltas[static_cast<size_t>(pos2)];
            }
        }
        return result;
    }

    if (stats.masked_board != board) {
        DeriveResult result;
        result.is_valid = true;
        result.is_derived = true;
        result.count = 1;
        result.boards[0] = stats.masked_board;
        return result;
    }
    DeriveResult result;
    result.is_valid = true;
    return result;
}

CarryState make_state(uint32_t original_board_sum, uint64_t bucket_estimate, uint64_t small_bytes, uint64_t large_words) {
    static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t), "EXAD atomic uint64 must not be padded");
    static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t), "EXAD atomic uint32 must not be padded");
    CarryState state;
    state.original_board_sum = original_board_sum;
    state.threshold_bits = kDefaultThresholdBits;
    state.hash_capacity = choose_hash_capacity(bucket_estimate);
    state.reserved_small_bytes = std::max<uint64_t>(small_bytes, kDynamicSmallChunkBytes);
    state.reserved_large_words = std::max<uint64_t>(large_words, kDynamicLargeChunkWords);
    state.key_array.reset(new std::atomic<uint64_t>[state.hash_capacity]);
    state.offset_array.reset(new std::atomic<uint32_t>[state.hash_capacity]);
    state.overflow_reason.store(0U, std::memory_order_relaxed);
    std::memset(
        state.key_array.get(),
        0xFF,
        static_cast<size_t>(state.hash_capacity) * sizeof(std::atomic<uint64_t>)
    );
    std::memset(
        state.offset_array.get(),
        0xFF,
        static_cast<size_t>(state.hash_capacity) * sizeof(std::atomic<uint32_t>)
    );
    state.small_arena.reset(new std::atomic<uint8_t>[static_cast<size_t>(state.reserved_small_bytes)]);
    state.large_arena.reset(new std::atomic<uint64_t>[static_cast<size_t>(state.reserved_large_words)]);
    return state;
}

EstimateSet estimate_from_layer(const Layer &layer, ReserveFactors factors, int thread_count) {
    EstimateSet estimates{};
    uint64_t total_buckets = 0;
    uint64_t total_small = 0;
    uint64_t total_large = 0;
    for (const BoardSet &set : layer.sets) {
        total_buckets += set.buckets.size();
        total_small += set.small_bitmap_bytes.size();
        total_large += set.large_bitmap_words.size();
    }
    const uint64_t avg_bucket_floor = std::max<uint64_t>(static_cast<uint64_t>(thread_count * 128), total_buckets / bucket_slot_count() + 1U);
    const uint64_t avg_small_floor = std::max<uint64_t>(kDynamicSmallChunkBytes, total_small / bucket_slot_count() + kDynamicSmallChunkBytes);
    const uint64_t avg_large_floor = std::max<uint64_t>(kDynamicLargeChunkWords, total_large / bucket_slot_count() + kDynamicLargeChunkWords);
    for (size_t slot = 0; slot < bucket_slot_count(); ++slot) {
        const BoardSet &set = layer.sets[slot];
        estimates.slots[slot].buckets = std::max<uint64_t>(
            static_cast<uint64_t>(std::ceil(static_cast<double>(set.buckets.size()) * factors.bucket)),
            static_cast<uint64_t>(std::ceil(static_cast<double>(avg_bucket_floor) * factors.bucket))
        );
        estimates.slots[slot].small_bytes = std::max<uint64_t>(
            static_cast<uint64_t>(std::ceil(static_cast<double>(set.small_bitmap_bytes.size()) * factors.small)) + kDynamicSmallChunkBytes,
            static_cast<uint64_t>(std::ceil(static_cast<double>(avg_small_floor) * factors.small))
        );
        estimates.slots[slot].large_words = std::max<uint64_t>(
            static_cast<uint64_t>(std::ceil(static_cast<double>(set.large_bitmap_words.size()) * factors.large)) + kDynamicLargeChunkWords,
            static_cast<uint64_t>(std::ceil(static_cast<double>(avg_large_floor) * factors.large))
        );
        estimates.slots[slot].buckets = std::max(estimates.slots[slot].buckets, factors.bucket_floor[slot]);
        estimates.slots[slot].small_bytes = std::max(
            estimates.slots[slot].small_bytes,
            factors.small_floor[slot] == 0U
                ? 0U
                : factors.small_floor[slot] + static_cast<uint64_t>(thread_count + 2) * kDynamicSmallChunkBytes
        );
        estimates.slots[slot].large_words = std::max(
            estimates.slots[slot].large_words,
            factors.large_floor[slot] == 0U
                ? 0U
                : factors.large_floor[slot] + static_cast<uint64_t>(thread_count + 2) * kDynamicLargeChunkWords
        );
        estimates.slots[slot].small_bytes += static_cast<uint64_t>(thread_count + 2) * kDynamicSmallChunkBytes;
        estimates.slots[slot].large_words += static_cast<uint64_t>(thread_count + 2) * kDynamicLargeChunkWords;
    }
    return estimates;
}

EstimateSet estimate_from_board_count(uint64_t boards, ReserveFactors factors, int thread_count) {
    EstimateSet estimates{};
    const uint64_t per_slot = boards / bucket_slot_count() + 1U;
    for (Estimate &estimate : estimates.slots) {
        estimate.buckets = std::max<uint64_t>(static_cast<uint64_t>(thread_count * 128), static_cast<uint64_t>(std::ceil(static_cast<double>(per_slot) * factors.bucket)));
        estimate.small_bytes = std::max<uint64_t>(kDynamicSmallChunkBytes, static_cast<uint64_t>(std::ceil(static_cast<double>(per_slot) * factors.small)));
        estimate.large_words = std::max<uint64_t>(kDynamicLargeChunkWords, static_cast<uint64_t>(std::ceil(static_cast<double>(per_slot / 8U + 1U) * factors.large)));
    }
    return estimates;
}

EstimateSet estimate_from_boards(
    const std::vector<uint64_t> &boards,
    uint32_t original_board_sum,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    const Luts &luts,
    ReserveFactors factors,
    int thread_count
) {
    const AdBucketLookup ad_lookup = make_ad_bucket_lookup(original_board_sum, tiles_table, param);
    std::array<std::vector<uint64_t>, bucket_slot_count()> keys_by_slot;
    const size_t reserve_each = boards.size() / bucket_slot_count() + 1U;
    for (auto &keys : keys_by_slot) {
        keys.reserve(reserve_each);
    }
    for (uint64_t board : boards) {
        const int8_t ad_key = ad_bucket_key_for_board_fast(board, param, ad_lookup);
        if (ad_key < bucket_key_min() || ad_key > bucket_key_max()) {
            continue;
        }
        const uint32_t suffix28 = static_cast<uint32_t>(board & kSuffixMask);
        uint32_t group = 0U;
        uint32_t rank = 0U;
        uint32_t semantic_sum = 0U;
        if (!suffix28_rank_group_sum(luts, suffix28, group, rank, semantic_sum)) {
            continue;
        }
        const uint32_t valid_count = luts.size_table[group];
        if (valid_count == 0U || rank >= valid_count) {
            continue;
        }
        keys_by_slot[bucket_to_index(ad_key)].push_back(pack_bucket_key(board >> kSuffixBits, semantic_sum));
    }

    EstimateSet estimates{};
    for (size_t slot = 0; slot < bucket_slot_count(); ++slot) {
        std::vector<uint64_t> &keys = keys_by_slot[slot];
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        uint64_t small_bytes = 0U;
        uint64_t large_words = 0U;
        for (uint64_t key : keys) {
            const uint32_t valid_count = luts.size_table[lut_group_index(bucket_key_semantic_sum(key))];
            if (valid_count <= kDefaultThresholdBits) {
                small_bytes += ZMaskFrozen::bytes_for_bits(valid_count);
            } else {
                large_words += ZMaskFrozen::words_for_bits(valid_count);
            }
        }
        estimates.slots[slot].buckets = std::max<uint64_t>(
            static_cast<uint64_t>(std::ceil(static_cast<double>(keys.size()) * factors.bucket)),
            static_cast<uint64_t>(thread_count * 128)
        );
        estimates.slots[slot].small_bytes = std::max<uint64_t>(
            static_cast<uint64_t>(std::ceil(static_cast<double>(small_bytes) * factors.small)) + kDynamicSmallChunkBytes,
            static_cast<uint64_t>(thread_count) * kDynamicSmallChunkBytes
        );
        estimates.slots[slot].large_words = std::max<uint64_t>(
            static_cast<uint64_t>(std::ceil(static_cast<double>(large_words) * factors.large)) + kDynamicLargeChunkWords,
            static_cast<uint64_t>(thread_count) * kDynamicLargeChunkWords
        );
    }
    return estimates;
}

CarryLayer make_carry(uint32_t original_board_sum, const EstimateSet &estimates, int thread_count) {
    CarryLayer carry;
    carry.original_board_sum = original_board_sum;
#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
    for (int64_t slot_i = 0; slot_i < static_cast<int64_t>(bucket_slot_count()); ++slot_i) {
        const size_t slot = static_cast<size_t>(slot_i);
        carry.states[slot] = make_state(
            original_board_sum,
            estimates.slots[slot].buckets,
            estimates.slots[slot].small_bytes,
            estimates.slots[slot].large_words
        );
    }
    return carry;
}

void mark_overflow(CarryState &state, uint32_t reason) {
    uint32_t expected = 0U;
    state.overflow_reason.compare_exchange_strong(
        expected,
        reason,
        std::memory_order_acq_rel,
        std::memory_order_relaxed
    );
    state.overflowed.store(true, std::memory_order_release);
}

uint32_t acquire_small_bytes(CarryState &state, ThreadChunkState &chunks, uint32_t bytes) {
    if (chunks.small_next + bytes <= chunks.small_end) {
        const uint32_t offset = chunks.small_next;
        chunks.small_next += bytes;
        return offset;
    }
    const uint32_t chunk_bytes = std::max<uint32_t>(kDynamicSmallChunkBytes, bytes);
    const uint32_t chunk_begin = state.small_cursor_bytes.fetch_add(chunk_bytes, std::memory_order_acq_rel);
    if (static_cast<uint64_t>(chunk_begin) + chunk_bytes > state.reserved_small_bytes) {
        mark_overflow(state, kOverflowSmallArena);
        return kPendingOffset;
    }
    for (uint32_t i = 0; i < chunk_bytes; ++i) {
        state.small_arena[chunk_begin + i].store(0U, std::memory_order_relaxed);
    }
    chunks.small_next = chunk_begin + bytes;
    chunks.small_end = chunk_begin + chunk_bytes;
    return chunk_begin;
}

uint32_t acquire_large_words(CarryState &state, ThreadChunkState &chunks, uint32_t words) {
    if (chunks.large_next + words <= chunks.large_end) {
        const uint32_t offset = chunks.large_next;
        chunks.large_next += words;
        return offset;
    }
    const uint32_t chunk_words = std::max<uint32_t>(kDynamicLargeChunkWords, words);
    const uint32_t chunk_begin = state.large_cursor_words.fetch_add(chunk_words, std::memory_order_acq_rel);
    if (static_cast<uint64_t>(chunk_begin) + chunk_words > state.reserved_large_words) {
        mark_overflow(state, kOverflowLargeArena);
        return kPendingOffset;
    }
    for (uint32_t i = 0; i < chunk_words; ++i) {
        state.large_arena[chunk_begin + i].store(0ULL, std::memory_order_relaxed);
    }
    chunks.large_next = chunk_begin + words;
    chunks.large_end = chunk_begin + chunk_words;
    return chunk_begin;
}

uint32_t probe_or_insert_slot(
    CarryState &state,
    uint64_t packed_key,
    uint32_t home_slot,
    uint32_t valid_count,
    ThreadChunkState &chunks,
    uint32_t &local_new_count
) {
    uint32_t slot = home_slot;
    uint32_t probes = 0U;
    for (;;) {
        const uint64_t current = state.key_array[slot].load(std::memory_order_acquire);
        if (current == packed_key) {
            uint32_t offset = state.offset_array[slot].load(std::memory_order_acquire);
            while (offset == kPendingOffset) {
                if (state.overflowed.load(std::memory_order_acquire)) {
                    return kPendingOffset;
                }
                offset = state.offset_array[slot].load(std::memory_order_acquire);
            }
            return offset;
        }
        if (current == kEmptyKey) {
            uint64_t expected = kEmptyKey;
            if (state.key_array[slot].compare_exchange_strong(expected, packed_key, std::memory_order_acq_rel, std::memory_order_acquire)) {
                ++local_new_count;
                const uint32_t offset = valid_count <= state.threshold_bits
                    ? acquire_small_bytes(state, chunks, static_cast<uint32_t>(ZMaskFrozen::bytes_for_bits(valid_count)))
                    : acquire_large_words(state, chunks, static_cast<uint32_t>(ZMaskFrozen::words_for_bits(valid_count)));
                if (offset != kPendingOffset) {
                    state.offset_array[slot].store(offset, std::memory_order_release);
                }
                return offset;
            }
            continue;
        }
        slot += 1U;
        if (slot == state.hash_capacity) {
            slot = 0U;
        }
        ++probes;
        if (probes >= state.hash_capacity) {
            mark_overflow(state, kOverflowHashFull);
            return kPendingOffset;
        }
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(static_cast<const void *>(&state.key_array[slot]), 0, 1);
#endif
    }
}

void prefetch_hash_slot(const CarryState &state, uint32_t slot) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(static_cast<const void *>(&state.key_array[slot]), 0, 1);
    __builtin_prefetch(static_cast<const void *>(&state.offset_array[slot]), 0, 1);
#else
    (void)state;
    (void)slot;
#endif
}

void prefetch_bitmap_target(CarryState &state, uint32_t offset, uint32_t rank, uint32_t valid_count) {
#if defined(__GNUC__) || defined(__clang__)
    if (valid_count <= state.threshold_bits) {
        __builtin_prefetch(static_cast<const void *>(&state.small_arena[offset + (rank >> 3U)]), 0, 1);
    } else {
        __builtin_prefetch(static_cast<const void *>(&state.large_arena[offset + (rank >> 6U)]), 0, 1);
    }
#else
    (void)state;
    (void)offset;
    (void)rank;
    (void)valid_count;
#endif
}

inline void set_bitmap_bit(CarryState &state, uint32_t bitmap_offset, uint32_t rank, uint32_t valid_count) {
    if (valid_count <= state.threshold_bits) {
        const uint32_t byte_offset = bitmap_offset + (rank >> 3U);
        const uint8_t mask = static_cast<uint8_t>(1U << (rank & 7U));
        if ((state.small_arena[byte_offset].load(std::memory_order_relaxed) & mask) == 0U) {
            state.small_arena[byte_offset].fetch_or(mask, std::memory_order_relaxed);
        }
    } else {
        const uint32_t word_offset = bitmap_offset + (rank >> 6U);
        const uint64_t mask = 1ULL << (rank & 63U);
        if ((state.large_arena[word_offset].load(std::memory_order_relaxed) & mask) == 0ULL) {
            state.large_arena[word_offset].fetch_or(mask, std::memory_order_relaxed);
        }
    }
}

void flush_pending(
    const PendingInsert *pending,
    size_t count,
    CarryLayer &layer,
    std::array<ThreadChunkState, bucket_slot_count()> &chunks,
    std::array<uint32_t, bucket_slot_count()> &local_new_counts
) {
    if (count == 0U) {
        return;
    }
    std::array<ResolvedInsert, kInsertBufferSize> resolved{};
    const size_t prefetch_count = std::min<size_t>(count, kPrefetchDistance);
    for (size_t i = 0; i < prefetch_count; ++i) {
        prefetch_hash_slot(layer.states[pending[i].ad_slot], pending[i].home_slot);
    }
    for (size_t i = 0; i < count; ++i) {
        if (i + kPrefetchDistance < count) {
            prefetch_hash_slot(layer.states[pending[i + kPrefetchDistance].ad_slot], pending[i + kPrefetchDistance].home_slot);
        }
        const PendingInsert &entry = pending[i];
        CarryState &state = layer.states[entry.ad_slot];
        if (state.overflowed.load(std::memory_order_acquire)) {
            return;
        }
        const uint32_t bitmap_offset = probe_or_insert_slot(
            state,
            entry.packed_key,
            entry.home_slot,
            entry.valid_count,
            chunks[entry.ad_slot],
            local_new_counts[entry.ad_slot]
        );
        if (bitmap_offset == kPendingOffset) {
            return;
        }
        resolved[i] = ResolvedInsert{entry.ad_slot, bitmap_offset, entry.valid_count, entry.rank};
        prefetch_bitmap_target(state, bitmap_offset, entry.rank, entry.valid_count);
    }
    for (size_t i = 0; i < count; ++i) {
        const ResolvedInsert &entry = resolved[i];
        CarryState &state = layer.states[entry.ad_slot];
        set_bitmap_bit(state, entry.bitmap_offset, entry.rank, entry.valid_count);
    }
}

void flush_pending_slot(
    uint8_t ad_slot,
    const SlotPendingInsert *pending,
    size_t count,
    CarryLayer &layer,
    ThreadChunkState &chunks,
    uint32_t &local_new_count
) {
    if (count == 0U) {
        return;
    }
    CarryState &state = layer.states[ad_slot];
    if (state.overflowed.load(std::memory_order_acquire)) {
        return;
    }
    const size_t prefetch_count = std::min<size_t>(count, kPrefetchDistance);
    for (size_t i = 0; i < prefetch_count; ++i) {
        prefetch_hash_slot(state, pending[i].home_slot);
    }
    for (size_t i = 0; i < count; ++i) {
        if (i + kPrefetchDistance < count) {
            prefetch_hash_slot(state, pending[i + kPrefetchDistance].home_slot);
        }
        const SlotPendingInsert &entry = pending[i];
        const uint32_t bitmap_offset = probe_or_insert_slot(
            state,
            entry.packed_key,
            entry.home_slot,
            entry.valid_count,
            chunks,
            local_new_count
        );
        if (bitmap_offset == kPendingOffset) {
            return;
        }
        set_bitmap_bit(state, bitmap_offset, entry.rank, entry.valid_count);
    }
}

bool make_pending_insert(
    uint64_t board,
    const AdvancedMaskParam &param,
    const AdBucketLookup &ad_lookup,
    const Luts &luts,
    const CarryLayer &target,
    PendingInsert &out
) {
    const int8_t ad_key = ad_bucket_key_for_board_fast(board, param, ad_lookup);
    if (ad_key < bucket_key_min() || ad_key > bucket_key_max()) {
        return false;
    }
    const uint32_t suffix28 = static_cast<uint32_t>(board & kSuffixMask);
    uint32_t group = 0U;
    uint32_t rank = 0U;
    uint32_t semantic_sum = 0U;
    if (!suffix28_rank_group_sum(luts, suffix28, group, rank, semantic_sum)) {
        return false;
    }
    const uint32_t valid_count = luts.size_table[group];
    if (valid_count == 0U || rank >= valid_count) {
        return false;
    }
    const uint8_t slot = static_cast<uint8_t>(bucket_to_index(ad_key));
    const uint64_t key = pack_bucket_key(board >> kSuffixBits, semantic_sum);
    const CarryState &state = target.states[slot];
    out = PendingInsert{
        key,
        valid_count,
        rank,
        hash_slot_for_key(key, state.hash_capacity),
        slot
    };
    return true;
}

bool make_pending_insert_for_slot(
    uint64_t board,
    uint8_t slot,
    const Luts &luts,
    const CarryLayer &target,
    SlotPendingInsert &out
) {
    const uint32_t suffix28 = static_cast<uint32_t>(board & kSuffixMask);
    uint32_t group = 0U;
    uint32_t rank = 0U;
    uint32_t semantic_sum = 0U;
    if (!suffix28_rank_group_sum(luts, suffix28, group, rank, semantic_sum)) {
        return false;
    }
    const uint32_t valid_count = luts.size_table[group];
    if (valid_count == 0U || rank >= valid_count) {
        return false;
    }
    const uint64_t key = pack_bucket_key(board >> kSuffixBits, semantic_sum);
    const CarryState &state = target.states[slot];
    out = SlotPendingInsert{
        key,
        valid_count,
        rank,
        hash_slot_for_key(key, state.hash_capacity)
    };
    return true;
}

void insert_board_immediate(
    CarryLayer &target,
    uint64_t board,
    uint32_t original_board_sum,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    const Luts &luts,
    std::array<ThreadChunkState, bucket_slot_count()> &chunks,
    std::array<uint32_t, bucket_slot_count()> &local_new_counts
) {
    const AdBucketLookup ad_lookup = make_ad_bucket_lookup(original_board_sum, tiles_table, param);
    PendingInsert insert{};
    if (!make_pending_insert(board, param, ad_lookup, luts, target, insert)) {
        return;
    }
    flush_pending(&insert, 1U, target, chunks, local_new_counts);
}

void finalize_counts(CarryLayer &layer, const std::vector<std::array<uint32_t, bucket_slot_count()>> &thread_new_counts) {
    std::array<uint64_t, bucket_slot_count()> totals{};
    for (const auto &counts : thread_new_counts) {
        for (size_t slot = 0; slot < bucket_slot_count(); ++slot) {
            totals[slot] += counts[slot];
        }
    }
    for (size_t slot = 0; slot < bucket_slot_count(); ++slot) {
        const uint64_t total = static_cast<uint64_t>(layer.states[slot].bucket_count.load(std::memory_order_relaxed)) + totals[slot];
        if (total > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("EXAD bucket count exceeds uint32 range");
        }
        layer.states[slot].bucket_count.store(static_cast<uint32_t>(total), std::memory_order_release);
    }
}

void insert_existing_bucket(CarryState &state, uint64_t key, uint32_t offset) {
    uint32_t slot = hash_slot_for_key(key, state.hash_capacity);
    for (;;) {
        const uint64_t current = state.key_array[slot].load(std::memory_order_relaxed);
        if (current == key) {
            state.offset_array[slot].store(offset, std::memory_order_relaxed);
            return;
        }
        if (current == kEmptyKey) {
            state.key_array[slot].store(key, std::memory_order_relaxed);
            state.offset_array[slot].store(offset, std::memory_order_relaxed);
            state.bucket_count.fetch_add(1U, std::memory_order_relaxed);
            return;
        }
        slot += 1U;
        if (slot == state.hash_capacity) {
            slot = 0U;
        }
    }
}

CarryState grow_state(CarryState &&seed, const Estimate &estimate) {
    CarryState state = make_state(seed.original_board_sum, estimate.buckets, estimate.small_bytes, estimate.large_words);
    const uint32_t used_small = std::min<uint32_t>(
        seed.small_cursor_bytes.load(std::memory_order_relaxed),
        static_cast<uint32_t>(std::min<uint64_t>(seed.reserved_small_bytes, std::numeric_limits<uint32_t>::max()))
    );
    const uint32_t used_large = std::min<uint32_t>(
        seed.large_cursor_words.load(std::memory_order_relaxed),
        static_cast<uint32_t>(std::min<uint64_t>(seed.reserved_large_words, std::numeric_limits<uint32_t>::max()))
    );
    if (used_small > state.reserved_small_bytes || used_large > state.reserved_large_words) {
        throw std::runtime_error("EXAD grown carry is smaller than seed usage");
    }
    for (uint32_t i = 0; i < used_small; ++i) {
        state.small_arena[i].store(seed.small_arena[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    for (uint32_t i = 0; i < used_large; ++i) {
        state.large_arena[i].store(seed.large_arena[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    state.small_cursor_bytes.store(used_small, std::memory_order_relaxed);
    state.large_cursor_words.store(used_large, std::memory_order_relaxed);
    for (uint32_t slot = 0; slot < seed.hash_capacity; ++slot) {
        const uint64_t key = seed.key_array[slot].load(std::memory_order_relaxed);
        if (key == kEmptyKey) {
            continue;
        }
        const uint32_t offset = seed.offset_array[slot].load(std::memory_order_relaxed);
        if (offset == kPendingOffset) {
            continue;
        }
        insert_existing_bucket(state, key, offset);
    }
    state.overflowed.store(false, std::memory_order_relaxed);
    state.overflow_reason.store(0U, std::memory_order_relaxed);
    return state;
}

bool state_has_capacity_for(const CarryState &state, const Estimate &additional) {
    const uint64_t used_buckets = state.bucket_count.load(std::memory_order_relaxed);
    const uint64_t used_small = state.small_cursor_bytes.load(std::memory_order_relaxed);
    const uint64_t used_large = state.large_cursor_words.load(std::memory_order_relaxed);
    return state.hash_capacity >= choose_hash_capacity(used_buckets + additional.buckets) &&
        state.reserved_small_bytes >= used_small + additional.small_bytes &&
        state.reserved_large_words >= used_large + additional.large_words;
}

CarryLayer prepare_arr1(CarryLayer seed, uint32_t target_sum, const EstimateSet &additional, int thread_count) {
    if (seed.empty()) {
        return make_carry(target_sum, additional, thread_count);
    }
    if (seed.original_board_sum != target_sum) {
        throw std::runtime_error("EXAD cannot reuse carry with different original_board_sum");
    }
#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
    for (int64_t slot_i = 0; slot_i < static_cast<int64_t>(bucket_slot_count()); ++slot_i) {
        const size_t slot = static_cast<size_t>(slot_i);
        CarryState &state = seed.states[slot];
        state.overflowed.store(false, std::memory_order_relaxed);
        state.overflow_reason.store(0U, std::memory_order_relaxed);
        const uint64_t used_buckets = state.bucket_count.load(std::memory_order_relaxed);
        const uint64_t used_small = state.small_cursor_bytes.load(std::memory_order_relaxed);
        const uint64_t used_large = state.large_cursor_words.load(std::memory_order_relaxed);
        const Estimate required_additional = additional.slots[slot];
        if (state_has_capacity_for(state, required_additional)) {
            continue;
        }
        Estimate target{};
        target.buckets = used_buckets + required_additional.buckets;
        target.small_bytes = used_small + required_additional.small_bytes;
        target.large_words = used_large + required_additional.large_words;
        state = grow_state(std::move(state), target);
    }
    return seed;
}

ReserveFactors doubled(ReserveFactors factors) {
    factors.bucket *= 2.0;
    factors.small *= 2.0;
    factors.large *= 2.0;
    return factors;
}

BoardSet finalize_state(const CarryState &state, const Luts &luts, int thread_count) {
    const size_t bucket_count = static_cast<size_t>(state.bucket_count.load(std::memory_order_relaxed));
    std::vector<uint64_t> keys;
    std::vector<uint32_t> old_offsets;
    keys.reserve(bucket_count);
    old_offsets.reserve(bucket_count);

    for (uint32_t slot = 0; slot < state.hash_capacity; ++slot) {
        const uint64_t key = state.key_array[slot].load(std::memory_order_relaxed);
        if (key == kEmptyKey) {
            continue;
        }
        const uint32_t offset = state.offset_array[slot].load(std::memory_order_relaxed);
        if (offset == kPendingOffset) {
            throw std::runtime_error("EXAD finalize saw pending offset");
        }
        keys.push_back(key);
        old_offsets.push_back(offset);
    }
    BookGeneratorUtils::sort_keyvalue_uint64_uint32(keys.data(), old_offsets.data(), keys.size(), false);

    BoardSet set;
    set.threshold_bits = state.threshold_bits;
    set.buckets.resize(keys.size());

    struct CopyPlan {
        uint64_t key = 0;
        uint32_t old_offset = 0;
        uint32_t new_offset = 0;
        uint32_t valid_count = 0;
        bool small = false;
    };
    std::vector<CopyPlan> plans(keys.size());
    uint64_t small_bytes = 0;
    uint64_t large_words = 0;
    for (size_t dst = 0; dst < keys.size(); ++dst) {
        const uint64_t key = keys[dst];
        const uint32_t group = lut_group_index(bucket_key_semantic_sum(key));
        const uint32_t valid_count = luts.size_table[group];
        const bool small = valid_count <= state.threshold_bits;
        set.buckets[dst].key = key;
        plans[dst] = CopyPlan{key, old_offsets[dst], 0U, valid_count, small};
        set.exact_bitmap_bits += valid_count;
        if (small) {
            plans[dst].new_offset = static_cast<uint32_t>(small_bytes);
            set.buckets[dst].bitmap_offset = static_cast<uint32_t>(small_bytes);
            small_bytes += ZMaskFrozen::bytes_for_bits(valid_count);
            set.aligned_bitmap_bits += ZMaskFrozen::bytes_for_bits(valid_count) * 8ULL;
        } else {
            plans[dst].new_offset = static_cast<uint32_t>(large_words);
            set.buckets[dst].bitmap_offset = static_cast<uint32_t>(large_words);
            large_words += ZMaskFrozen::words_for_bits(valid_count);
            set.aligned_bitmap_bits += ZMaskFrozen::words_for_bits(valid_count) * 64ULL;
        }
    }
    set.small_bitmap_bytes.assign(small_bytes, 0U);
    set.large_bitmap_words.assign(large_words, 0ULL);
    std::vector<uint32_t> live_counts(plans.size(), 0U);

#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int64_t i = 0; i < static_cast<int64_t>(plans.size()); ++i) {
        const CopyPlan &plan = plans[static_cast<size_t>(i)];
        uint32_t live = 0;
        if (plan.small) {
            const uint32_t bytes = static_cast<uint32_t>(ZMaskFrozen::bytes_for_bits(plan.valid_count));
            for (uint32_t j = 0; j < bytes; ++j) {
                const uint8_t value = state.small_arena[plan.old_offset + j].load(std::memory_order_relaxed);
                set.small_bitmap_bytes[plan.new_offset + j] = value;
                live += popcount64(value);
            }
        } else {
            const uint32_t words = static_cast<uint32_t>(ZMaskFrozen::words_for_bits(plan.valid_count));
            for (uint32_t j = 0; j < words; ++j) {
                const uint64_t value = state.large_arena[plan.old_offset + j].load(std::memory_order_relaxed);
                set.large_bitmap_words[plan.new_offset + j] = value;
                live += popcount64(value);
            }
        }
        live_counts[static_cast<size_t>(i)] = live;
    }

    uint64_t dense = 0;
    for (size_t i = 0; i < live_counts.size(); ++i) {
        set.buckets[i].dense_offset = static_cast<uint32_t>(dense);
        dense += live_counts[i];
    }
    set.live_board_count = dense;
    return set;
}

void append_pending_or_flush(
    PendingInsert insert,
    CarryLayer &target,
    std::array<PendingInsert, kInsertBufferSize> &pending,
    size_t &pending_count,
    std::array<ThreadChunkState, bucket_slot_count()> &chunks,
    std::array<uint32_t, bucket_slot_count()> &local_new_counts
) {
    pending[pending_count++] = insert;
    if (pending_count == pending.size()) {
        flush_pending(pending.data(), pending_count, target, chunks, local_new_counts);
        pending_count = 0;
    }
}

void generate_into_carries_once(
    const Layer &current,
    const AdvancedPatternSpec &spec,
    const RunOptions &options,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    const Luts &luts,
    int thread_count,
    CarryLayer &arr1,
    CarryLayer &arr2,
    DeriveHashState *derive_hash_state,
    GenerateStats &stats
) {
    const double hashmap0 = now_seconds();
    size_t derive_hash_length = derive_input_hash_length(current.live_board_count);
    std::unique_ptr<uint64_t[]> local_derive_hash1;
    std::unique_ptr<uint64_t[]> local_derive_hash2;
    uint64_t *derive_hash1 = nullptr;
    uint64_t *derive_hash2 = nullptr;
    if (derive_hash_state != nullptr) {
        derive_hash_length = prepare_derive_hashmaps(*derive_hash_state, derive_hash_length, thread_count);
        derive_hash1 = derive_hash_state->next1.data();
        derive_hash2 = derive_hash_state->next2.data();
    } else {
        local_derive_hash1 = make_zeroed_hashmap(derive_hash_length, thread_count);
        local_derive_hash2 = make_zeroed_hashmap(derive_hash_length, thread_count);
        derive_hash1 = local_derive_hash1.get();
        derive_hash2 = local_derive_hash2.get();
    }
    const uint64_t derive_hashmask = static_cast<uint64_t>(derive_hash_length - 1U);
    const double hashmap1_done = now_seconds();

    std::vector<std::array<uint32_t, bucket_slot_count()>> thread_new_counts1(static_cast<size_t>(thread_count));
    std::vector<std::array<uint32_t, bucket_slot_count()>> thread_new_counts2(static_cast<size_t>(thread_count));
    std::vector<uint64_t> thread_derive(static_cast<size_t>(thread_count), 0U);
    std::vector<uint64_t> thread_derived_out(static_cast<size_t>(thread_count), 0U);
    const AdBucketLookup ad_lookup1 = make_ad_bucket_lookup(current.original_board_sum + 2U, tiles_table, param);
    const AdBucketLookup ad_lookup2 = make_ad_bucket_lookup(current.original_board_sum + 4U, tiles_table, param);
    const bool skip_pattern_check = spec.pattern_masks.empty();

    struct WorkRef {
        uint8_t ad_slot = 0;
        uint32_t bucket = 0;
    };
    std::vector<WorkRef> work;
    size_t work_count = 0U;
    for (const BoardSet &set : current.sets) {
        work_count += set.buckets.size();
    }
    work.reserve(work_count);
    for (size_t slot = 0; slot < current.sets.size(); ++slot) {
        const BoardSet &set = current.sets[slot];
        for (uint32_t bucket = 0; bucket < static_cast<uint32_t>(set.buckets.size()); ++bucket) {
            work.push_back(WorkRef{static_cast<uint8_t>(slot), bucket});
        }
    }
    const double work_done = now_seconds();

#pragma omp parallel num_threads(thread_count)
    {
        const int tid =
#if defined(_OPENMP)
            omp_get_thread_num();
#else
            0;
#endif
        std::array<std::array<SlotPendingInsert, kSlotInsertBufferSize>, bucket_slot_count()> pending1;
        std::array<std::array<SlotPendingInsert, kSlotInsertBufferSize>, bucket_slot_count()> pending2;
        std::array<size_t, bucket_slot_count()> pending1_count{};
        std::array<size_t, bucket_slot_count()> pending2_count{};
        std::array<ThreadChunkState, bucket_slot_count()> chunks1{};
        std::array<ThreadChunkState, bucket_slot_count()> chunks2{};
        std::array<uint32_t, bucket_slot_count()> local_new1{};
        std::array<uint32_t, bucket_slot_count()> local_new2{};
        std::array<uint64_t, kCandidateBufferSize> final1;
        std::array<uint8_t, kCandidateBufferSize> final1_kinds;
        std::array<uint64_t, kCandidateBufferSize> final2;
        std::array<uint8_t, kCandidateBufferSize> final2_kinds;
        std::array<uint32_t, kCandidateBufferSize> derive_hash_indices;
        std::array<uint64_t, kDerivedCanonicalBufferSize> derived1_buffer;
        std::array<uint64_t, kDerivedCanonicalBufferSize> derived2_buffer;
        size_t final1_count = 0;
        size_t final2_count = 0;
        size_t derived1_count = 0;
        size_t derived2_count = 0;
        uint64_t local_derive = 0;
        uint64_t local_derived_out = 0;
        const bool inherit_plain_slot = param.pos_fixed_32k_mask == 0ULL;

        auto append_slot_pending = [&](
            uint8_t slot,
            SlotPendingInsert insert,
            CarryLayer &target,
            std::array<std::array<SlotPendingInsert, kSlotInsertBufferSize>, bucket_slot_count()> &pending_by_slot,
            std::array<size_t, bucket_slot_count()> &pending_counts,
            std::array<ThreadChunkState, bucket_slot_count()> &chunks,
            std::array<uint32_t, bucket_slot_count()> &local_new_counts
        ) {
            size_t &count = pending_counts[slot];
            auto &buffer = pending_by_slot[slot];
            buffer[count++] = insert;
            if (count == kSlotInsertBufferSize) {
                flush_pending_slot(slot, buffer.data(), count, target, chunks[slot], local_new_counts[slot]);
                count = 0U;
            }
        };

        auto insert_canonical1 = [&](uint64_t board, bool from_derive) {
            PendingInsert insert{};
            if (make_pending_insert(board, param, ad_lookup1, luts, arr1, insert)) {
                if (from_derive) {
                    ++local_derived_out;
                }
                append_slot_pending(
                    insert.ad_slot,
                    SlotPendingInsert{insert.packed_key, insert.valid_count, insert.rank, insert.home_slot},
                    arr1,
                    pending1,
                    pending1_count,
                    chunks1,
                    local_new1
                );
            }
        };

        auto insert_canonical1_for_slot = [&](uint64_t board, uint8_t slot, bool from_derive) {
            SlotPendingInsert insert{};
            if (make_pending_insert_for_slot(board, slot, luts, arr1, insert)) {
                if (from_derive) {
                    ++local_derived_out;
                }
                append_slot_pending(slot, insert, arr1, pending1, pending1_count, chunks1, local_new1);
            }
        };

        auto insert_canonical2 = [&](uint64_t board, bool from_derive) {
            PendingInsert insert{};
            if (make_pending_insert(board, param, ad_lookup2, luts, arr2, insert)) {
                if (from_derive) {
                    ++local_derived_out;
                }
                append_slot_pending(
                    insert.ad_slot,
                    SlotPendingInsert{insert.packed_key, insert.valid_count, insert.rank, insert.home_slot},
                    arr2,
                    pending2,
                    pending2_count,
                    chunks2,
                    local_new2
                );
            }
        };

        auto insert_canonical2_for_slot = [&](uint64_t board, uint8_t slot, bool from_derive) {
            SlotPendingInsert insert{};
            if (make_pending_insert_for_slot(board, slot, luts, arr2, insert)) {
                if (from_derive) {
                    ++local_derived_out;
                }
                append_slot_pending(slot, insert, arr2, pending2, pending2_count, chunks2, local_new2);
            }
        };

        auto flush_final1 = [&]() {
            if (final1_count == 0U) {
                return;
            }
            CanonicalBatch::canonicalize_inplace(final1.data(), final1_count, spec.symm_mode);
            for (size_t i = 0; i < final1_count; ++i) {
                const uint8_t meta = final1_kinds[i];
                const bool from_derive = (meta & kBufferedFromDeriveFlag) != 0U;
                const uint8_t slot = meta & kBufferedSlotMask;
                if (inherit_plain_slot && slot != kBufferedUnknownSlot) {
                    insert_canonical1_for_slot(final1[i], slot, from_derive);
                } else {
                    insert_canonical1(final1[i], from_derive);
                }
            }
            final1_count = 0U;
        };

        auto flush_final2 = [&]() {
            if (final2_count == 0U) {
                return;
            }
            CanonicalBatch::canonicalize_inplace(final2.data(), final2_count, spec.symm_mode);
            for (size_t i = 0; i < final2_count; ++i) {
                const uint8_t meta = final2_kinds[i];
                const bool from_derive = (meta & kBufferedFromDeriveFlag) != 0U;
                const uint8_t slot = meta & kBufferedSlotMask;
                if (inherit_plain_slot && slot != kBufferedUnknownSlot) {
                    insert_canonical2_for_slot(final2[i], slot, from_derive);
                } else {
                    insert_canonical2(final2[i], from_derive);
                }
            }
            final2_count = 0U;
        };

        auto push_final1 = [&](uint64_t board, uint8_t source_slot, bool from_derive) {
            final1[final1_count] = board;
            final1_kinds[final1_count] =
                static_cast<uint8_t>((from_derive ? kBufferedFromDeriveFlag : 0U) | (source_slot & kBufferedSlotMask));
            ++final1_count;
            if (final1_count == final1.size()) {
                flush_final1();
            }
        };

        auto push_final2 = [&](uint64_t board, uint8_t source_slot, bool from_derive) {
            final2[final2_count] = board;
            final2_kinds[final2_count] =
                static_cast<uint8_t>((from_derive ? kBufferedFromDeriveFlag : 0U) | (source_slot & kBufferedSlotMask));
            ++final2_count;
            if (final2_count == final2.size()) {
                flush_final2();
            }
        };

        auto flush_derive1 = [&]() {
            if (derived1_count == 0U) {
                return;
            }
            CanonicalBatch::canonicalize_inplace(derived1_buffer.data(), derived1_count, spec.symm_mode);
            for (size_t i = 0; i < derived1_count; ++i) {
                const uint32_t hash_index = static_cast<uint32_t>(BookGeneratorUtils::hash_board(derived1_buffer[i]) & derive_hashmask);
                derive_hash_indices[i] = hash_index;
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(derive_hash1 + hash_index, 1, 1);
#endif
            }
            for (size_t i = 0; i < derived1_count; ++i) {
                const uint64_t canon = derived1_buffer[i];
                if (i + kDeriveHashPrefetchDistance < derived1_count) {
#if defined(__GNUC__) || defined(__clang__)
                    __builtin_prefetch(derive_hash1 + derive_hash_indices[i + kDeriveHashPrefetchDistance], 1, 1);
#endif
                }
                const bool seen = derive_input_seen_or_record_at(derive_hash1, derive_hash_indices[i], canon);
                if (seen) {
                    continue;
                }
                ++local_derive;
                DeriveResult derived;
                derived = derive(canon, current.original_board_sum + 2U, tiles_table, param);
                if (!derived.is_valid) {
                    continue;
                }
                if (!derived.is_derived) {
                    push_final1(canon, kBufferedUnknownSlot, false);
                    continue;
                }
                for (uint8_t j = 0; j < derived.count; ++j) {
                    const uint64_t board = derived.boards[j];
                    if (!skip_pattern_check && !is_pattern(board, spec.pattern_masks)) {
                        continue;
                    }
                    push_final1(board, kBufferedUnknownSlot, true);
                }
            }
            derived1_count = 0U;
        };

        auto flush_derive2 = [&]() {
            if (derived2_count == 0U) {
                return;
            }
            CanonicalBatch::canonicalize_inplace(derived2_buffer.data(), derived2_count, spec.symm_mode);
            for (size_t i = 0; i < derived2_count; ++i) {
                const uint32_t hash_index = static_cast<uint32_t>(BookGeneratorUtils::hash_board(derived2_buffer[i]) & derive_hashmask);
                derive_hash_indices[i] = hash_index;
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(derive_hash2 + hash_index, 1, 1);
#endif
            }
            for (size_t i = 0; i < derived2_count; ++i) {
                const uint64_t canon = derived2_buffer[i];
                if (i + kDeriveHashPrefetchDistance < derived2_count) {
#if defined(__GNUC__) || defined(__clang__)
                    __builtin_prefetch(derive_hash2 + derive_hash_indices[i + kDeriveHashPrefetchDistance], 1, 1);
#endif
                }
                const bool seen = derive_input_seen_or_record_at(derive_hash2, derive_hash_indices[i], canon);
                if (seen) {
                    continue;
                }
                ++local_derive;
                DeriveResult derived;
                derived = derive(canon, current.original_board_sum + 4U, tiles_table, param);
                if (!derived.is_valid) {
                    continue;
                }
                if (!derived.is_derived) {
                    push_final2(canon, kBufferedUnknownSlot, false);
                    continue;
                }
                for (uint8_t j = 0; j < derived.count; ++j) {
                    const uint64_t board = derived.boards[j];
                    if (!skip_pattern_check && !is_pattern(board, spec.pattern_masks)) {
                        continue;
                    }
                    push_final2(board, kBufferedUnknownSlot, true);
                }
            }
            derived2_count = 0U;
        };

        auto push_derive1 = [&](uint64_t board) {
            derived1_buffer[derived1_count++] = board;
            if (derived1_count == derived1_buffer.size()) {
                flush_derive1();
            }
        };

        auto push_derive2 = [&](uint64_t board) {
            derived2_buffer[derived2_count++] = board;
            if (derived2_count == derived2_buffer.size()) {
                flush_derive2();
            }
        };

        auto handle_moved1 = [&](uint64_t moved, bool needs_derive, uint8_t source_slot) {
            if (!skip_pattern_check && !is_pattern(moved, spec.pattern_masks)) {
                return;
            }
            if (needs_derive) {
                push_derive1(moved);
            } else {
                push_final1(moved, source_slot, false);
            }
        };

        auto handle_moved2 = [&](uint64_t moved, bool needs_derive, uint8_t source_slot) {
            if (!skip_pattern_check && !is_pattern(moved, spec.pattern_masks)) {
                return;
            }
            if (needs_derive) {
                push_derive2(moved);
            } else {
                push_final2(moved, source_slot, false);
            }
        };

        const auto &spawn_reverse_shifts = reverse_spawn_shifts();
        auto process_board = [&](uint64_t board, uint8_t source_slot) {
            const uint64_t board_rev = FormationAD::reverse(board);
            uint64_t zero_mask = zero_nibble_lsb_mask(board);
            while (zero_mask != 0ULL) {
                const uint32_t shift = countr_zero64(zero_mask);
                const uint32_t reverse_shift = spawn_reverse_shifts[shift >> 2U];
                uint64_t spawned = board | (1ULL << shift);
                uint64_t rev = board_rev | (1ULL << reverse_shift);
                const uint64_t md2 = FormationAD::m_move_down(spawned, rev);
                const uint64_t mr2 = FormationAD::m_move_right(spawned);
                const auto [ml2, mnt_h2] = FormationAD::m_move_left2(spawned);
                const auto [mu2, mnt_v2] = FormationAD::m_move_up2(spawned, rev);
                if (ml2 != spawned) {
                    handle_moved1(ml2, mnt_h2, source_slot);
                }
                if (mr2 != spawned) {
                    handle_moved1(mr2, mnt_h2, source_slot);
                }
                if (mu2 != spawned) {
                    handle_moved1(mu2, mnt_v2, source_slot);
                }
                if (md2 != spawned) {
                    handle_moved1(md2, mnt_v2, source_slot);
                }

                spawned = board | (2ULL << shift);
                rev = board_rev | (2ULL << reverse_shift);
                const uint64_t md4 = FormationAD::m_move_down(spawned, rev);
                const uint64_t mr4 = FormationAD::m_move_right(spawned);
                const auto [ml4, mnt_h4] = FormationAD::m_move_left2(spawned);
                const auto [mu4, mnt_v4] = FormationAD::m_move_up2(spawned, rev);
                if (ml4 != spawned) {
                    handle_moved2(ml4, mnt_h4, source_slot);
                }
                if (mr4 != spawned) {
                    handle_moved2(mr4, mnt_h4, source_slot);
                }
                if (mu4 != spawned) {
                    handle_moved2(mu4, mnt_v4, source_slot);
                }
                if (md4 != spawned) {
                    handle_moved2(md4, mnt_v4, source_slot);
                }
                zero_mask &= (zero_mask - 1ULL);
            }
        };

#pragma omp for schedule(dynamic, 1024)
        for (int64_t work_idx = 0; work_idx < static_cast<int64_t>(work.size()); ++work_idx) {
            if (arr1.states[0].overflowed.load(std::memory_order_acquire) || arr2.states[0].overflowed.load(std::memory_order_acquire)) {
                continue;
            }
            const WorkRef ref = work[static_cast<size_t>(work_idx)];
            const BoardSet &set = current.sets[ref.ad_slot];
            const BucketEntry &bucket_entry = set.buckets[ref.bucket];
            const uint64_t key = bucket_entry.key;
            const uint64_t prefix36 = bucket_key_prefix36(key);
            const uint32_t semantic_sum = bucket_key_semantic_sum(key);
            const uint32_t group = lut_group_index(semantic_sum);
            const uint32_t valid_count = luts.size_table[group];
            const uint32_t unrank_base = luts.offset_table[group];
            const uint32_t bitmap_offset = bucket_entry.bitmap_offset;

            if (valid_count <= set.threshold_bits) {
                const uint32_t bytes = static_cast<uint32_t>(ZMaskFrozen::bytes_for_bits(valid_count));
                for (uint32_t byte = 0; byte < bytes; ++byte) {
                    uint8_t value = set.small_bitmap_bytes[bitmap_offset + byte];
                    while (value != 0U) {
#if defined(__GNUC__) || defined(__clang__)
                        const uint32_t bit = static_cast<uint32_t>(__builtin_ctz(value));
#else
                        uint32_t bit = 0;
                        while (((value >> bit) & 1U) == 0U) {
                            ++bit;
                        }
#endif
                        const uint32_t rank = byte * 8U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        process_board((prefix36 << kSuffixBits) | luts.unrank_array[unrank_base + rank], ref.ad_slot);
                        value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                    }
                }
            } else {
                const uint32_t words = static_cast<uint32_t>(ZMaskFrozen::words_for_bits(valid_count));
                for (uint32_t word = 0; word < words; ++word) {
                    uint64_t value = set.large_bitmap_words[bitmap_offset + word];
                    while (value != 0ULL) {
                        const uint32_t bit = countr_zero64(value);
                        const uint32_t rank = word * 64U + bit;
                        if (rank >= valid_count) {
                            break;
                        }
                        process_board((prefix36 << kSuffixBits) | luts.unrank_array[unrank_base + rank], ref.ad_slot);
                        value &= (value - 1ULL);
                    }
                }
            }
        }

        flush_derive1();
        flush_derive2();
        flush_final1();
        flush_final2();
        for (size_t slot = 0; slot < bucket_slot_count(); ++slot) {
            flush_pending_slot(static_cast<uint8_t>(slot), pending1[slot].data(), pending1_count[slot], arr1, chunks1[slot], local_new1[slot]);
            flush_pending_slot(static_cast<uint8_t>(slot), pending2[slot].data(), pending2_count[slot], arr2, chunks2[slot], local_new2[slot]);
        }
        thread_new_counts1[static_cast<size_t>(tid)] = local_new1;
        thread_new_counts2[static_cast<size_t>(tid)] = local_new2;
        thread_derive[static_cast<size_t>(tid)] = local_derive;
        thread_derived_out[static_cast<size_t>(tid)] = local_derived_out;
    }
    const double loop_done = now_seconds();

    finalize_counts(arr1, thread_new_counts1);
    finalize_counts(arr2, thread_new_counts2);
    const double counts_done = now_seconds();
    stats.hashmap_seconds += hashmap1_done - hashmap0;
    stats.worklist_seconds += work_done - hashmap1_done;
    stats.loop_seconds += loop_done - work_done;
    stats.count_finalize_seconds += counts_done - loop_done;
    stats.derive_candidate_count += std::accumulate(thread_derive.begin(), thread_derive.end(), 0ULL);
    stats.derived_output_count += std::accumulate(thread_derived_out.begin(), thread_derived_out.end(), 0ULL);
    (void)options;
}

bool any_overflow(const CarryLayer &layer) {
    for (const CarryState &state : layer.states) {
        if (state.overflowed.load(std::memory_order_acquire)) {
            return true;
        }
    }
    return false;
}

const char *overflow_reason_name(uint32_t reason) {
    switch (reason) {
        case kOverflowSmallArena:
            return "small_arena";
        case kOverflowLargeArena:
            return "large_arena";
        case kOverflowHashFull:
            return "hash_full";
        default:
            return "unknown";
    }
}

std::string overflow_summary(const CarryLayer &layer, const char *label) {
    std::ostringstream oss;
    oss << label << " overflow";
    for (size_t slot = 0; slot < layer.states.size(); ++slot) {
        const CarryState &state = layer.states[slot];
        if (!state.overflowed.load(std::memory_order_acquire)) {
            continue;
        }
        const uint32_t reason = state.overflow_reason.load(std::memory_order_acquire);
        oss << " slot=" << static_cast<int>(bucket_key_min() + static_cast<int>(slot))
            << " reason=" << overflow_reason_name(reason)
            << " buckets=" << state.bucket_count.load(std::memory_order_relaxed)
            << " hash_capacity=" << state.hash_capacity
            << " small_used=" << state.small_cursor_bytes.load(std::memory_order_relaxed)
            << "/" << state.reserved_small_bytes
            << " large_used=" << state.large_cursor_words.load(std::memory_order_relaxed)
            << "/" << state.reserved_large_words;
        break;
    }
    return oss.str();
}

void insert_boards_into_carry(
    CarryLayer &carry,
    const std::vector<uint64_t> &boards,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    const Luts &luts,
    int thread_count
) {
    const AdBucketLookup ad_lookup = make_ad_bucket_lookup(carry.original_board_sum, tiles_table, param);
    std::vector<std::array<uint32_t, bucket_slot_count()>> thread_new_counts(static_cast<size_t>(thread_count));
#pragma omp parallel num_threads(thread_count)
    {
        const int tid =
#if defined(_OPENMP)
            omp_get_thread_num();
#else
            0;
#endif
        std::array<PendingInsert, kInsertBufferSize> pending{};
        size_t pending_count = 0;
        std::array<ThreadChunkState, bucket_slot_count()> chunks{};
        std::array<uint32_t, bucket_slot_count()> local_new{};
#pragma omp for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(boards.size()); ++i) {
            PendingInsert insert{};
            if (make_pending_insert(boards[static_cast<size_t>(i)], param, ad_lookup, luts, carry, insert)) {
                append_pending_or_flush(insert, carry, pending, pending_count, chunks, local_new);
            }
        }
        flush_pending(pending.data(), pending_count, carry, chunks, local_new);
        thread_new_counts[static_cast<size_t>(tid)] = local_new;
    }
    finalize_counts(carry, thread_new_counts);
}

} // namespace

CarryState::CarryState(CarryState &&other) noexcept
    : original_board_sum(other.original_board_sum),
      threshold_bits(other.threshold_bits),
      hash_capacity(other.hash_capacity),
      reserved_small_bytes(other.reserved_small_bytes),
      reserved_large_words(other.reserved_large_words),
      key_array(std::move(other.key_array)),
      offset_array(std::move(other.offset_array)),
      small_arena(std::move(other.small_arena)),
      large_arena(std::move(other.large_arena)),
      small_cursor_bytes(other.small_cursor_bytes.load(std::memory_order_relaxed)),
      large_cursor_words(other.large_cursor_words.load(std::memory_order_relaxed)),
      bucket_count(other.bucket_count.load(std::memory_order_relaxed)),
      overflowed(other.overflowed.load(std::memory_order_relaxed)),
      overflow_reason(other.overflow_reason.load(std::memory_order_relaxed)) {}

CarryState &CarryState::operator=(CarryState &&other) noexcept {
    if (this != &other) {
        original_board_sum = other.original_board_sum;
        threshold_bits = other.threshold_bits;
        hash_capacity = other.hash_capacity;
        reserved_small_bytes = other.reserved_small_bytes;
        reserved_large_words = other.reserved_large_words;
        key_array = std::move(other.key_array);
        offset_array = std::move(other.offset_array);
        small_arena = std::move(other.small_arena);
        large_arena = std::move(other.large_arena);
        small_cursor_bytes.store(other.small_cursor_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
        large_cursor_words.store(other.large_cursor_words.load(std::memory_order_relaxed), std::memory_order_relaxed);
        bucket_count.store(other.bucket_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
        overflowed.store(other.overflowed.load(std::memory_order_relaxed), std::memory_order_relaxed);
        overflow_reason.store(other.overflow_reason.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    return *this;
}

Layer build_layer_from_boards(
    const std::vector<uint64_t> &masked_boards,
    uint32_t original_board_sum,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    const Luts &luts,
    int num_threads,
    ReserveFactors factors
) {
    const int thread_count = effective_threads(num_threads);
    for (uint32_t retry = 0; retry < 7U; ++retry) {
        NativeDiagnostics::mark(
            "EXADBuilder.build_layer_from_boards retry=" + std::to_string(retry) +
            " boards=" + std::to_string(masked_boards.size())
        );
        CarryLayer carry = make_carry(
            original_board_sum,
            estimate_from_boards(masked_boards, original_board_sum, tiles_table, param, luts, factors, thread_count),
            thread_count
        );
        insert_boards_into_carry(carry, masked_boards, tiles_table, param, luts, thread_count);
        if (!any_overflow(carry)) {
            NativeDiagnostics::mark("EXADBuilder.build_layer_from_boards finalize begin");
            return finalize_carry_layer(carry, luts, thread_count);
        }
        factors = doubled(factors);
    }
    throw std::runtime_error("EXAD seed build exceeded retry limit");
}

CarryLayer carry_from_layer(
    const Layer &layer,
    const Luts &luts,
    int num_threads,
    ReserveFactors factors
) {
    const int thread_count = effective_threads(num_threads);
    CarryLayer carry = make_carry(layer.original_board_sum, estimate_from_layer(layer, factors, thread_count), thread_count);
    for (size_t slot = 0; slot < layer.sets.size(); ++slot) {
        const BoardSet &set = layer.sets[slot];
        CarryState &state = carry.states[slot];
        for (uint32_t bucket = 0; bucket < static_cast<uint32_t>(set.buckets.size()); ++bucket) {
            const BucketEntry &entry = set.buckets[bucket];
            const uint64_t key = entry.key;
            const uint32_t group = lut_group_index(bucket_key_semantic_sum(key));
            const uint32_t valid_count = luts.size_table[group];
            const bool small = valid_count <= set.threshold_bits;
            const uint32_t old_offset = entry.bitmap_offset;
            uint32_t new_offset = 0;
            if (small) {
                const uint32_t bytes = static_cast<uint32_t>(ZMaskFrozen::bytes_for_bits(valid_count));
                new_offset = state.small_cursor_bytes.fetch_add(bytes, std::memory_order_relaxed);
                if (static_cast<uint64_t>(new_offset) + bytes > state.reserved_small_bytes) {
                    throw std::runtime_error("EXAD carry_from_layer small arena under-reserved");
                }
                for (uint32_t i = 0; i < bytes; ++i) {
                    state.small_arena[new_offset + i].store(set.small_bitmap_bytes[old_offset + i], std::memory_order_relaxed);
                }
            } else {
                const uint32_t words = static_cast<uint32_t>(ZMaskFrozen::words_for_bits(valid_count));
                new_offset = state.large_cursor_words.fetch_add(words, std::memory_order_relaxed);
                if (static_cast<uint64_t>(new_offset) + words > state.reserved_large_words) {
                    throw std::runtime_error("EXAD carry_from_layer large arena under-reserved");
                }
                for (uint32_t i = 0; i < words; ++i) {
                    state.large_arena[new_offset + i].store(set.large_bitmap_words[old_offset + i], std::memory_order_relaxed);
                }
            }
            insert_existing_bucket(state, key, new_offset);
        }
    }
    return carry;
}

Layer finalize_carry_layer(
    const CarryLayer &carry,
    const Luts &luts,
    int num_threads
) {
    NativeDiagnostics::Scope scope("EXADBuilder.finalize_carry_layer");
    const int thread_count = effective_threads(num_threads);
    Layer layer;
    layer.original_board_sum = carry.original_board_sum;
    layer.threshold_bits = carry.threshold_bits;
    layer.lut_signature = luts.config_signature;
    layer.physical_transform = luts.physical_transform;
    layer.inverse_physical_transform = luts.inverse_physical_transform;
    layer.logical_pattern_signature = luts.logical_pattern_signature;
    layer.physical_pattern_signature = luts.physical_pattern_signature;
    std::array<uint64_t, bucket_slot_count()> live_counts{};
#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
    for (int64_t slot_i = 0; slot_i < static_cast<int64_t>(bucket_slot_count()); ++slot_i) {
        const size_t slot = static_cast<size_t>(slot_i);
        layer.sets[slot] = finalize_state(carry.states[slot], luts, 1);
        live_counts[slot] = layer.sets[slot].live_board_count;
    }
    for (uint64_t live : live_counts) {
        layer.live_board_count += live;
    }
    return layer;
}

GeneratePairResult generate_two_layers_carry(
    const Layer &current,
    const AdvancedPatternSpec &spec,
    const RunOptions &options,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    const Luts &luts,
    int num_threads,
    CarryLayer arr1_seed,
    ReserveFactors factors,
    DeriveHashState *derive_hash_state
) {
    const int thread_count = effective_threads(num_threads);
    GenerateStats stats;
    stats.input_live = current.live_board_count;
    for (;;) {
        NativeDiagnostics::mark(
            "EXADBuilder.generate_two_layers_carry attempt retry=" + std::to_string(stats.retry_count) +
            " input_live=" + std::to_string(current.live_board_count)
        );
        const double prepare0 =
#if defined(_OPENMP)
            omp_get_wtime();
#else
            0.0;
#endif
        const EstimateSet estimate = estimate_from_layer(current, factors, thread_count);
        uint64_t estimated_buckets = 0;
        uint64_t estimated_small = 0;
        uint64_t estimated_large = 0;
        for (const Estimate &slot_estimate : estimate.slots) {
            estimated_buckets += slot_estimate.buckets;
            estimated_small += slot_estimate.small_bytes;
            estimated_large += slot_estimate.large_words;
        }
        NativeDiagnostics::mark(
            "EXADBuilder.generate_two_layers_carry estimate done buckets=" +
            std::to_string(estimated_buckets) +
            " small=" + std::to_string(estimated_small) +
            " large=" + std::to_string(estimated_large)
        );
        const double estimate_done =
#if defined(_OPENMP)
            omp_get_wtime();
#else
            0.0;
#endif
        CarryLayer arr1 = prepare_arr1(std::move(arr1_seed), current.original_board_sum + 2U, estimate, thread_count);
        NativeDiagnostics::mark("EXADBuilder.generate_two_layers_carry prepare_arr1 done");
        const double arr1_done =
#if defined(_OPENMP)
            omp_get_wtime();
#else
            0.0;
#endif
        CarryLayer arr2 = make_carry(current.original_board_sum + 4U, estimate, thread_count);
        NativeDiagnostics::mark("EXADBuilder.generate_two_layers_carry make_arr2 done");
        const double prepare1 =
#if defined(_OPENMP)
            omp_get_wtime();
#else
            0.0;
#endif
        const double gen0 = prepare1;
        generate_into_carries_once(
            current,
            spec,
            options,
            tiles_table,
            param,
            luts,
            thread_count,
            arr1,
            arr2,
            derive_hash_state,
            stats
        );
        NativeDiagnostics::mark("EXADBuilder.generate_two_layers_carry hot loop done");
        const double gen1 =
#if defined(_OPENMP)
            omp_get_wtime();
#else
            0.0;
#endif
        stats.prepare_seconds += prepare1 - prepare0;
        stats.prepare_estimate_seconds += estimate_done - prepare0;
        stats.prepare_arr1_seconds += arr1_done - estimate_done;
        stats.prepare_arr2_seconds += prepare1 - arr1_done;
        stats.generate_seconds += gen1 - gen0;
        stats.insert_seconds = stats.generate_seconds;
        const bool arr1_overflow = any_overflow(arr1);
        const bool arr2_overflow = any_overflow(arr2);
        if (!arr1_overflow && !arr2_overflow) {
            if (derive_hash_state != nullptr) {
                // This step's arr2 target becomes the next step's arr1 target.
                rotate_derive_hashmaps(*derive_hash_state);
            }
            return {std::move(arr1), std::move(arr2), stats};
        }
        if (derive_hash_state != nullptr) {
            reset_derive_hashmaps(*derive_hash_state, thread_count);
        }
        factors = doubled(factors);
        arr1_seed = std::move(arr1);
        ++stats.retry_count;
        if (stats.retry_count > 6U) {
            throw std::runtime_error("EXAD generation exceeded retry limit");
        }
    }
}

Layer validate_layer_streaming(
    Layer layer,
    uint32_t original_board_sum,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    const Luts &luts,
    int num_threads
) {
    const int thread_count = effective_threads(num_threads);
    struct WorkRef {
        uint8_t ad_slot = 0;
        uint32_t bucket = 0;
    };
    std::vector<WorkRef> work;
    size_t work_count = 0U;
    for (const BoardSet &set : layer.sets) {
        work_count += set.buckets.size();
    }
    work.reserve(work_count);
    for (size_t slot = 0; slot < layer.sets.size(); ++slot) {
        const BoardSet &set = layer.sets[slot];
        for (uint32_t bucket = 0; bucket < static_cast<uint32_t>(set.buckets.size()); ++bucket) {
            work.push_back(WorkRef{static_cast<uint8_t>(slot), bucket});
        }
    }

    std::array<std::vector<uint32_t>, bucket_slot_count()> live_counts_by_slot;
    for (size_t slot = 0; slot < layer.sets.size(); ++slot) {
        live_counts_by_slot[slot].resize(layer.sets[slot].buckets.size(), 0U);
    }

#pragma omp parallel for schedule(dynamic, 32) num_threads(thread_count)
    for (int64_t work_idx = 0; work_idx < static_cast<int64_t>(work.size()); ++work_idx) {
        const WorkRef ref = work[static_cast<size_t>(work_idx)];
        BoardSet &set = layer.sets[ref.ad_slot];
        const BucketEntry &bucket = set.buckets[ref.bucket];
        const uint64_t prefix36 = bucket_key_prefix36(bucket.key);
        const uint32_t semantic_sum = bucket_key_semantic_sum(bucket.key);
        const uint32_t group = lut_group_index(semantic_sum);
        const uint32_t valid_count = luts.size_table[group];
        const uint32_t unrank_base = luts.offset_table[group];
        const uint32_t bitmap_offset = bucket.bitmap_offset;
        uint32_t kept = 0U;
        if (valid_count <= set.threshold_bits) {
            const uint32_t bytes = static_cast<uint32_t>(ZMaskFrozen::bytes_for_bits(valid_count));
            for (uint32_t byte_idx = 0; byte_idx < bytes; ++byte_idx) {
                uint8_t value = set.small_bitmap_bytes[bitmap_offset + byte_idx];
                uint8_t kept_value = value;
                while (value != 0U) {
#if defined(__GNUC__) || defined(__clang__)
                    const uint32_t bit_idx = static_cast<uint32_t>(__builtin_ctz(value));
#else
                    uint32_t bit_idx = 0;
                    while (((value >> bit_idx) & 1U) == 0U) {
                        ++bit_idx;
                    }
#endif
                    const uint32_t rank = byte_idx * 8U + bit_idx;
                    if (rank >= valid_count) {
                        break;
                    }
                    const uint64_t board = (prefix36 << kSuffixBits) | luts.unrank_array[unrank_base + rank];
                    if (FormationAD::validate(board, original_board_sum, tiles_table, param)) {
                        ++kept;
                    } else {
                        kept_value = static_cast<uint8_t>(kept_value & static_cast<uint8_t>(~(1U << (rank & 7U))));
                    }
                    value = static_cast<uint8_t>(value & static_cast<uint8_t>(value - 1U));
                }
                set.small_bitmap_bytes[bitmap_offset + byte_idx] = kept_value;
            }
        } else {
            const uint32_t words = static_cast<uint32_t>(ZMaskFrozen::words_for_bits(valid_count));
            for (uint32_t word_idx = 0; word_idx < words; ++word_idx) {
                uint64_t value = set.large_bitmap_words[bitmap_offset + word_idx];
                uint64_t kept_value = value;
                while (value != 0ULL) {
                    const uint32_t bit_idx = countr_zero64(value);
                    const uint32_t rank = word_idx * 64U + bit_idx;
                    if (rank >= valid_count) {
                        break;
                    }
                    const uint64_t board = (prefix36 << kSuffixBits) | luts.unrank_array[unrank_base + rank];
                    if (FormationAD::validate(board, original_board_sum, tiles_table, param)) {
                        ++kept;
                    } else {
                        kept_value &= ~(1ULL << (rank & 63U));
                    }
                    value &= (value - 1ULL);
                }
                set.large_bitmap_words[bitmap_offset + word_idx] = kept_value;
            }
        }
        live_counts_by_slot[ref.ad_slot][ref.bucket] = kept;
    }

    Layer out;
    out.original_board_sum = layer.original_board_sum;
    out.threshold_bits = layer.threshold_bits;
    out.lut_signature = layer.lut_signature;
    out.physical_transform = layer.physical_transform;
    out.inverse_physical_transform = layer.inverse_physical_transform;
    out.logical_pattern_signature = layer.logical_pattern_signature;
    out.physical_pattern_signature = layer.physical_pattern_signature;

    std::array<uint64_t, bucket_slot_count()> output_live{};
#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
    for (int64_t slot_i = 0; slot_i < static_cast<int64_t>(layer.sets.size()); ++slot_i) {
        const size_t slot = static_cast<size_t>(slot_i);
        const BoardSet &src = layer.sets[slot];
        BoardSet &dst = out.sets[slot];
        dst.threshold_bits = src.threshold_bits;
        uint64_t small_bytes = 0U;
        uint64_t large_words = 0U;
        uint64_t dense = 0U;
        for (uint32_t bucket_idx = 0; bucket_idx < static_cast<uint32_t>(src.buckets.size()); ++bucket_idx) {
            const uint32_t kept_count = live_counts_by_slot[slot][bucket_idx];
            if (kept_count == 0U) {
                continue;
            }
            const BucketEntry &old_entry = src.buckets[bucket_idx];
            const uint32_t group = lut_group_index(bucket_key_semantic_sum(old_entry.key));
            const uint32_t valid_count = luts.size_table[group];
            const bool small = valid_count <= src.threshold_bits;
            BucketEntry entry = old_entry;
            entry.bitmap_offset = small ? static_cast<uint32_t>(small_bytes) : static_cast<uint32_t>(large_words);
            entry.dense_offset = static_cast<uint32_t>(dense);
            dst.buckets.push_back(entry);
            dense += kept_count;
            dst.exact_bitmap_bits += valid_count;
            if (small) {
                const uint64_t bytes = ZMaskFrozen::bytes_for_bits(valid_count);
                const uint32_t old_offset = old_entry.bitmap_offset;
                dst.small_bitmap_bytes.insert(
                    dst.small_bitmap_bytes.end(),
                    src.small_bitmap_bytes.begin() + old_offset,
                    src.small_bitmap_bytes.begin() + old_offset + static_cast<size_t>(bytes)
                );
                small_bytes += bytes;
                dst.aligned_bitmap_bits += bytes * 8ULL;
            } else {
                const uint64_t words = ZMaskFrozen::words_for_bits(valid_count);
                const uint32_t old_offset = old_entry.bitmap_offset;
                dst.large_bitmap_words.insert(
                    dst.large_bitmap_words.end(),
                    src.large_bitmap_words.begin() + old_offset,
                    src.large_bitmap_words.begin() + old_offset + static_cast<size_t>(words)
                );
                large_words += words;
                dst.aligned_bitmap_bits += words * 64ULL;
            }
        }
        dst.live_board_count = dense;
        output_live[slot] = dense;
    }
    for (uint64_t live : output_live) {
        out.live_board_count += live;
    }
    return out;
}

CarryLayer validate_carry_streaming(
    const CarryLayer &carry,
    uint32_t original_board_sum,
    const FormationAD::TilesCombinationTable &tiles_table,
    const AdvancedMaskParam &param,
    const Luts &luts,
    int num_threads
) {
    Layer layer = finalize_carry_layer(carry, luts, num_threads);
    layer = validate_layer_streaming(layer, original_board_sum, tiles_table, param, luts, num_threads);
    return carry_from_layer(layer, luts, num_threads);
}

uint64_t carry_bucket_count(const CarryLayer &carry) {
    uint64_t total = 0;
    for (const CarryState &state : carry.states) {
        total += state.bucket_count.load(std::memory_order_relaxed);
    }
    return total;
}

} // namespace EXAD
