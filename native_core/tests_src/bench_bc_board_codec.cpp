#include "BCBoardCodec.h"
#include "BCCellBuilder.h"
#include "BCCellMatrix.h"
#include "BCFamilyTable.h"
#include "BCPositionFile.h"
#include "BCPositionScanner.h"
#include "BoardMover.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <chrono>
#endif

namespace {

volatile uint64_t g_sink64 = 0U;

uint32_t xorshift32(uint32_t &state) {
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return state;
}

[[nodiscard]] uint64_t rotl64(uint64_t value, uint32_t bits) {
    return (value << bits) | (value >> (64U - bits));
}

struct MultiAccumulator {
    static constexpr uint32_t kLaneCount = 8U;

    std::array<uint64_t, kLaneCount> lanes{
        0x243F6A8885A308D3ULL,
        0x13198A2E03707344ULL,
        0xA4093822299F31D0ULL,
        0x082EFA98EC4E6C89ULL,
        0x452821E638D01377ULL,
        0xBE5466CF34E90C6CULL,
        0xC0AC29B7C97C50DDULL,
        0x3F84D5B5B5470917ULL,
    };

    void mix(size_t index, uint64_t value) {
        uint64_t &lane = lanes[index & (kLaneCount - 1U)];
        lane += value + 0x9E3779B97F4A7C15ULL + rotl64(lane, 17U);
    }

    [[nodiscard]] uint64_t finish() const {
        uint64_t out = 0xD1B54A32D192ED03ULL;
        for (uint64_t lane : lanes) {
            out ^= lane + 0x9E3779B97F4A7C15ULL + rotl64(out, 11U);
        }
        return out;
    }
};

struct LightAccumulator {
    uint64_t sum = 0U;

    void mix(size_t, uint64_t value) {
        sum += value;
    }

    [[nodiscard]] uint64_t finish() const {
        return sum;
    }
};

std::vector<uint8_t> test_alphabet() {
    return {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 15U};
}

std::vector<BC::BCQuadrantWords> make_quadrants(size_t count) {
    std::vector<BC::BCQuadrantWords> values;
    values.reserve(count);
    uint32_t state = 0x12345678U;
    for (size_t i = 0; i < count; ++i) {
        values.push_back(BC::BCQuadrantWords{
            static_cast<uint16_t>(xorshift32(state)),
            static_cast<uint16_t>(xorshift32(state)),
            static_cast<uint16_t>(xorshift32(state)),
            static_cast<uint16_t>(xorshift32(state))
        });
    }
    return values;
}

std::vector<uint64_t> make_boards(const std::vector<BC::BCQuadrantWords> &quadrants) {
    std::vector<uint64_t> boards;
    boards.reserve(quadrants.size());
    for (const BC::BCQuadrantWords &q : quadrants) {
        boards.push_back(BC::pack_quadrants_to_board(q));
    }
    return boards;
}

std::vector<uint16_t> collect_valid_words(const BC::BCLut &lut) {
    std::vector<uint16_t> words;
    for (uint32_t word = 0; word < BC::kBCQuadrantWordCount; ++word) {
        if (lut.word_desc(static_cast<uint16_t>(word)).valid) {
            words.push_back(static_cast<uint16_t>(word));
        }
    }
    return words;
}

struct GroupChoice {
    uint16_t sum_id = 0U;
    uint8_t empty_mask = 0U;
    uint16_t count = 0U;
};

GroupChoice find_largest_group(const BC::BCLut &lut) {
    GroupChoice best;
    for (uint32_t sum_id = 0; sum_id < lut.sum_count(); ++sum_id) {
        for (uint32_t mask = 0; mask < 16U; ++mask) {
            const uint16_t count =
                lut.count4(static_cast<uint16_t>(sum_id), static_cast<uint8_t>(mask));
            if (count > best.count) {
                best = GroupChoice{static_cast<uint16_t>(sum_id), static_cast<uint8_t>(mask), count};
            }
        }
    }
    if (best.count == 0U) {
        throw std::runtime_error("BC board codec bench failed to find a non-empty LUT group");
    }
    return best;
}

struct PackTables {
    std::array<uint64_t, BC::kBCQuadrantWordCount> nw{};
    std::array<uint64_t, BC::kBCQuadrantWordCount> ne{};
    std::array<uint64_t, BC::kBCQuadrantWordCount> sw{};
    std::array<uint64_t, BC::kBCQuadrantWordCount> se{};

    PackTables() {
        for (uint32_t word = 0; word < BC::kBCQuadrantWordCount; ++word) {
            const uint16_t w = static_cast<uint16_t>(word);
            nw[word] = BC::pack_quadrants_to_board(BC::BCQuadrantWords{w, 0U, 0U, 0U});
            ne[word] = BC::pack_quadrants_to_board(BC::BCQuadrantWords{0U, w, 0U, 0U});
            sw[word] = BC::pack_quadrants_to_board(BC::BCQuadrantWords{0U, 0U, w, 0U});
            se[word] = BC::pack_quadrants_to_board(BC::BCQuadrantWords{0U, 0U, 0U, w});
        }
    }

    [[nodiscard]] uint64_t pack(const BC::BCQuadrantWords &q) const {
        return nw[q.nw] | ne[q.ne] | sw[q.sw] | se[q.se];
    }
};

struct ScannerFixture {
    std::vector<uint8_t> bytes;
    BC::CellId cid = 0U;
    uint32_t rows = 0U;
    uint32_t bucket_count = 0U;
    uint32_t bitmap_len = 0U;
};

ScannerFixture make_scanner_fixture(const BC::BCLut &lut) {
    const BC::BCFamilyTable axis = BC::BCFamilyTable::from_range(6U, 1U, 0U, 0U);
    const BC::BCCellMatrix matrix(axis);
    const BC::CellId cid = matrix.cid(0U, 0U);
    const std::vector<uint16_t> valid_words = collect_valid_words(lut);
    const GroupChoice group = find_largest_group(lut);
    const uint32_t bitmap_len =
        static_cast<uint32_t>(group.count) *
        static_cast<uint32_t>(group.count) *
        static_cast<uint32_t>(group.count);
    if (bitmap_len == 0U || bitmap_len > BC::kBCMaxBucketBitmapLen) {
        throw std::runtime_error("BC board codec bench computed invalid scanner bitmap_len");
    }

    constexpr uint32_t kTargetBuckets = 8U;
    const uint32_t bucket_count =
        std::min<uint32_t>(kTargetBuckets, static_cast<uint32_t>(valid_words.size()));
    BC::BCCellBuilder builder(lut);
    for (uint32_t bucket = 0; bucket < bucket_count; ++bucket) {
        const uint16_t nw = valid_words[bucket];
        const uint16_t ne = lut.unrank_word(group.sum_id, group.empty_mask, 0U);
        const uint16_t sw = ne;
        const uint16_t se = ne;
        const BC::BCEncodedKeyRank encoded = BC::encode_key_and_rank(lut, nw, ne, sw, se);
        if (!encoded.valid || encoded.bitmap_len != bitmap_len) {
            throw std::runtime_error("BC board codec bench failed to create scanner bucket");
        }
        for (uint32_t rank = 0; rank < bitmap_len; ++rank) {
            builder.insert(encoded.key, static_cast<BC::BucketRank>(rank));
        }
    }

    const BC::FinalizedCellPayload payload = builder.finalize();
    BC::BCPositionLayerWriter writer;
    writer.begin_layer(axis);
    writer.write_cell(cid, payload);

    return ScannerFixture{
        writer.finish_layer(),
        cid,
        payload.success_rows,
        bucket_count,
        bitmap_len
    };
}

double now_seconds() {
#ifdef _WIN32
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return static_cast<double>(counter.QuadPart) / static_cast<double>(frequency.QuadPart);
#else
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
#endif
}

template <class Fn>
double median_ns_per_op(size_t count, uint32_t repeats, Fn &&fn, uint64_t &checksum_out) {
    std::vector<double> samples;
    samples.reserve(repeats);
    uint64_t final_checksum = 0U;
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        const double begin = now_seconds();
        const uint64_t checksum = fn();
        const double end = now_seconds();
        final_checksum = checksum;
        const double ns = (end - begin) * 1000000000.0;
        samples.push_back(ns / static_cast<double>(count));
    }
    std::sort(samples.begin(), samples.end());
    checksum_out = final_checksum;
    return samples[samples.size() / 2U];
}

struct DualBenchResult {
    double light_ns_per_op = 0.0;
    uint64_t light_checksum = 0U;
    double multi_ns_per_op = 0.0;
    uint64_t multi_checksum = 0U;
};

void print_result(const char *name, const DualBenchResult &result) {
    std::printf(
        "%-22s light=%7.3f ns/op  multi=%7.3f ns/op  "
        "light_sum=0x%llx multi_sum=0x%llx\n",
        name,
        result.light_ns_per_op,
        result.multi_ns_per_op,
        static_cast<unsigned long long>(result.light_checksum),
        static_cast<unsigned long long>(result.multi_checksum)
    );
}

template <class Accumulator, class Fn>
uint64_t run_with_accumulator(Fn &&fn) {
    Accumulator acc;
    fn(acc);
    g_sink64 = acc.finish();
    return g_sink64;
}

template <class Fn>
DualBenchResult run_dual_result(size_t count, uint32_t repeats, Fn &&fn) {
    DualBenchResult result;
    result.light_ns_per_op = median_ns_per_op(
        count,
        repeats,
        [&]() {
            return run_with_accumulator<LightAccumulator>(fn);
        },
        result.light_checksum
    );
    result.multi_ns_per_op = median_ns_per_op(
        count,
        repeats,
        [&]() {
            return run_with_accumulator<MultiAccumulator>(fn);
        },
        result.multi_checksum
    );
    return result;
}

template <class Fn>
void run_result(const char *name, size_t count, uint32_t repeats, Fn &&fn) {
    print_result(name, run_dual_result(count, repeats, fn));
}

} // namespace

int main() {
    constexpr size_t kCount = 1U << 21U;
    constexpr uint32_t kRepeats = 7U;

    const std::vector<BC::BCQuadrantWords> quadrants = make_quadrants(kCount);
    const std::vector<uint64_t> boards = make_boards(quadrants);
    const PackTables *tables = new PackTables();

    std::printf(
        "board_codec route=shift count=%llu repeats=%u checksum_modes=light_sum,multi%u\n",
        static_cast<unsigned long long>(kCount),
        kRepeats,
        MultiAccumulator::kLaneCount
    );

    run_result(
        "pack",
        kCount,
        kRepeats,
        [&](auto &acc) {
            size_t index = 0U;
            for (const BC::BCQuadrantWords &q : quadrants) {
                acc.mix(index++, BC::pack_quadrants_to_board(q));
            }
        }
    );

    run_result(
        "pack_table",
        kCount,
        kRepeats,
        [&](auto &acc) {
            size_t index = 0U;
            for (const BC::BCQuadrantWords &q : quadrants) {
                acc.mix(index++, tables->pack(q));
            }
        }
    );

    run_result(
        "unpack",
        kCount,
        kRepeats,
        [&](auto &acc) {
            size_t index = 0U;
            for (uint64_t board : boards) {
                const BC::BCQuadrantWords q = BC::unpack_board_to_quadrants(board);
                acc.mix(
                    index++,
                    static_cast<uint64_t>(q.nw) |
                    (static_cast<uint64_t>(q.ne) << 16U) |
                    (static_cast<uint64_t>(q.sw) << 32U) |
                    (static_cast<uint64_t>(q.se) << 48U)
                );
            }
        }
    );

    delete tables;

    const BC::BCLut lut(test_alphabet());
    const ScannerFixture fixture = make_scanner_fixture(lut);
    const BC::BCPositionLayerReader reader(fixture.bytes, lut);
    const BC::BCPositionCellScanner scanner(reader, fixture.cid);

    std::printf(
        "scanner_pipeline rows=%u buckets=%u bitmap_len=%u repeats=%u\n",
        fixture.rows,
        fixture.bucket_count,
        fixture.bitmap_len,
        kRepeats
    );

    run_result(
        "scan_quadrants",
        fixture.rows,
        kRepeats,
        [&](auto &acc) {
            size_t index = 0U;
            scanner.for_each([&](const BC::BCScannedPositionEntry &entry) {
                acc.mix(
                    index++,
                    entry.key ^
                    (static_cast<uint64_t>(entry.rank) << 32U) ^
                    (static_cast<uint64_t>(entry.local_success_row) << 16U) ^
                    static_cast<uint64_t>(entry.nw) ^
                    (static_cast<uint64_t>(entry.ne) << 16U) ^
                    (static_cast<uint64_t>(entry.sw) << 32U) ^
                    (static_cast<uint64_t>(entry.se) << 48U)
                );
            });
        }
    );

    run_result(
        "scan_for_each_pack",
        fixture.rows,
        kRepeats,
        [&](auto &acc) {
            size_t index = 0U;
            scanner.for_each([&](const BC::BCScannedPositionEntry &entry) {
                const uint64_t board = BC::pack_quadrants_to_board(
                    BC::BCQuadrantWords{entry.nw, entry.ne, entry.sw, entry.se}
                );
                acc.mix(
                    index++,
                    board ^
                    entry.key ^
                    (static_cast<uint64_t>(entry.rank) << 32U) ^
                    static_cast<uint64_t>(entry.local_success_row)
                );
            });
        }
    );

    run_result(
        "scan_for_each_board",
        fixture.rows,
        kRepeats,
        [&](auto &acc) {
            size_t index = 0U;
            scanner.for_each_board([&](const BC::BCScannedBoardEntry &entry) {
                acc.mix(
                    index++,
                    entry.board ^
                    entry.key ^
                    (static_cast<uint64_t>(entry.rank) << 32U) ^
                    static_cast<uint64_t>(entry.local_success_row)
                );
            });
        }
    );

    run_result(
        "scan_board_move4",
        fixture.rows,
        kRepeats,
        [&](auto &acc) {
            size_t index = 0U;
            scanner.for_each_board([&](const BC::BCScannedBoardEntry &entry) {
                const uint64_t left = BoardMover::move_left(entry.board);
                const uint64_t right = BoardMover::move_right(entry.board);
                const uint64_t up = BoardMover::move_up(entry.board);
                const uint64_t down = BoardMover::move_down(entry.board);
                acc.mix(
                    index++,
                    left ^
                    rotl64(right, 13U) ^
                    rotl64(up, 29U) ^
                    rotl64(down, 47U) ^
                    entry.key ^
                    (static_cast<uint64_t>(entry.rank) << 32U) ^
                    static_cast<uint64_t>(entry.local_success_row)
                );
            });
        }
    );

    return static_cast<int>(g_sink64 & 0U);
}
