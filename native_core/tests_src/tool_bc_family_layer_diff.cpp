#include "BCBoardOps.h"
#include "BCFamilyPartitionPolicy.h"
#include "BCPositionFile.h"
#include "BCPositionScanner.h"
#include "Calculator.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::filesystem::path a;
    std::filesystem::path b;
    uint32_t target_rank = 8U;
    uint32_t sample = 32U;
};

[[nodiscard]] uint64_t board_sum(uint64_t board, const std::array<uint32_t, 16U> &tile_values) {
    uint64_t sum = 0U;
    for (uint32_t cell = 0U; cell < BC::kBCBoardCellCount; ++cell) {
        sum += tile_values[BC::board_tile_unchecked(board, cell)];
    }
    return sum;
}

[[nodiscard]] uint32_t tile_count(uint64_t board, uint8_t rank) {
    uint32_t count = 0U;
    for (uint32_t cell = 0U; cell < BC::kBCBoardCellCount; ++cell) {
        count += BC::board_tile_unchecked(board, cell) == rank ? 1U : 0U;
    }
    return count;
}

struct BoardFamilyInfo {
    uint32_t nw_sum = 0U;
    uint32_t ne_sum = 0U;
    uint32_t sw_sum = 0U;
    uint32_t se_sum = 0U;
    uint64_t top_sum = 0U;
    uint64_t bottom_sum = 0U;
    uint64_t left_sum = 0U;
    uint64_t right_sum = 0U;
    uint64_t row_min_sum = 0U;
    uint64_t col_min_sum = 0U;
    uint64_t row_coord = 0U;
    uint64_t col_coord = 0U;
    uint32_t row_mod = 0U;
    uint32_t col_mod = 0U;
    bool valid = false;
};

[[nodiscard]] BoardFamilyInfo board_family_info(
    const BC::BCLut &lut,
    const BC::BCPositionLayerReader &reader,
    const BC::BCQuadrantWords &q
) {
    BoardFamilyInfo info;
    const BC::BCWordDesc &nw = lut.word_desc(q.nw);
    const BC::BCWordDesc &ne = lut.word_desc(q.ne);
    const BC::BCWordDesc &sw = lut.word_desc(q.sw);
    const BC::BCWordDesc &se = lut.word_desc(q.se);
    if (!nw.valid || !ne.valid || !sw.valid || !se.valid) {
        return info;
    }
    info.nw_sum = nw.sum;
    info.ne_sum = ne.sum;
    info.sw_sum = sw.sum;
    info.se_sum = se.sum;
    info.top_sum = static_cast<uint64_t>(nw.sum) + ne.sum;
    info.bottom_sum = static_cast<uint64_t>(sw.sum) + se.sum;
    info.left_sum = static_cast<uint64_t>(nw.sum) + sw.sum;
    info.right_sum = static_cast<uint64_t>(ne.sum) + se.sum;
    info.row_min_sum = std::min(info.top_sum, info.bottom_sum);
    info.col_min_sum = std::min(info.left_sum, info.right_sum);
    const uint32_t unit = reader.axis().family_unit();
    if (unit == 0U ||
        (info.row_min_sum % unit) != 0U ||
        (info.col_min_sum % unit) != 0U) {
        return info;
    }
    info.row_coord = info.row_min_sum / unit;
    info.col_coord = info.col_min_sum / unit;
    const uint32_t family_count = reader.axis().family_count();
    info.row_mod = family_count == 0U ? 0U : static_cast<uint32_t>(info.row_coord % family_count);
    info.col_mod = family_count == 0U ? 0U : static_cast<uint32_t>(info.col_coord % family_count);
    info.valid = true;
    return info;
}

[[nodiscard]] std::vector<uint8_t> make_free_legal_tiles(uint32_t target_rank) {
    if (target_rank >= 15U) {
        throw std::invalid_argument("target rank must be < 15");
    }
    std::vector<uint8_t> legal;
    legal.reserve(target_rank + 2U);
    for (uint32_t tile = 0U; tile <= target_rank; ++tile) {
        legal.push_back(static_cast<uint8_t>(tile));
    }
    legal.push_back(15U);
    return legal;
}

[[nodiscard]] BC::BCLut make_lut(uint32_t target_rank) {
    return BC::BCLut(make_free_legal_tiles(target_rank));
}

[[nodiscard]] uint64_t descriptor_rows(const BC::BCPositionLayerReader &reader) {
    uint64_t rows = 0U;
    for (BC::CellId cid = 0U; cid < reader.cell_count(); ++cid) {
        rows += reader.descriptor(cid).success_rows;
    }
    return rows;
}

[[nodiscard]] std::vector<uint64_t> load_boards(
    const std::filesystem::path &path,
    const BC::BCLut &lut
) {
    BC::BCPositionFileReader file = BC::BCPositionFileReader::open_buffered(path, lut);
    const BC::BCPositionLayerReader &reader = file.layer();
    const uint64_t rows = descriptor_rows(reader);
    if (rows > static_cast<uint64_t>(std::vector<uint64_t>{}.max_size())) {
        throw std::length_error("too many boards for vector");
    }
    std::vector<uint64_t> boards;
    boards.reserve(static_cast<size_t>(rows));
    for (BC::CellId cid = 0U; cid < reader.cell_count(); ++cid) {
        BC::BCPositionCellScanner(reader, cid).for_each_board(
            [&boards](const BC::BCScannedBoardEntry &entry) {
                boards.push_back(entry.board);
            }
        );
    }
    std::sort(boards.begin(), boards.end());
    const auto unique_end = std::unique(boards.begin(), boards.end());
    if (unique_end != boards.end()) {
        std::cerr << "warning: duplicate boards collapsed: "
                  << (boards.end() - unique_end) << '\n';
        boards.erase(unique_end, boards.end());
    }
    return boards;
}

void print_board_grid(uint64_t board) {
    for (int row = 3; row >= 0; --row) {
        std::cout << "    ";
        for (int col = 3; col >= 0; --col) {
            const uint32_t cell = static_cast<uint32_t>(row * 4 + col);
            std::cout << std::setw(2)
                      << static_cast<uint32_t>(BC::board_tile_unchecked(board, cell));
            if (col != 0) {
                std::cout << ' ';
            }
        }
        std::cout << '\n';
    }
}

void print_board_info(
    const char *prefix,
    uint64_t board,
    const BC::BCLut &lut,
    const BC::BCPositionLayerReader &reader,
    const std::array<uint32_t, 16U> &tile_values
) {
    const BC::BCBoardEncodedPosition encoded =
        BC::encode_canonical_board_position(lut, reader.axis(), board);
    const BC::BCQuadrantWords q = BC::unpack_board_to_quadrants(board);
    const BoardFamilyInfo family = board_family_info(lut, reader, q);
    const bool canonical = Calculator::canonical_full(board) == board;
    std::cout << prefix
              << " board=0x" << std::hex << std::setw(16) << std::setfill('0') << board
              << std::setfill(' ') << std::dec
              << " sum=" << board_sum(board, tile_values)
              << " rank15=" << tile_count(board, 15U)
              << " canonical=" << (canonical ? 1 : 0)
              << " q=[" << q.nw << ',' << q.ne << ',' << q.sw << ',' << q.se << ']';
    if (family.valid) {
        std::cout
            << " qsum=[" << family.nw_sum << ',' << family.ne_sum << ','
            << family.sw_sum << ',' << family.se_sum << ']'
            << " side_sum=[top:" << family.top_sum
            << " bottom:" << family.bottom_sum
            << " left:" << family.left_sum
            << " right:" << family.right_sum << ']'
            << " coord=[row:" << family.row_coord
            << " col:" << family.col_coord << ']'
            << " mod_family=[row:" << family.row_mod
            << " col:" << family.col_mod << ']';
    } else {
        std::cout << " family_invalid";
    }
    if (encoded.valid) {
        const BC::BCCellMatrix matrix(reader.axis());
        std::cout
            << " cid=" << encoded.cid
            << " row_id=" << encoded.row_family
            << " row_coord=" << reader.axis().id_to_coord(encoded.row_family)
            << " col_id=" << encoded.col_family
            << " col_coord=" << reader.axis().id_to_coord(encoded.col_family)
            << " key=0x" << std::hex << std::setw(16) << std::setfill('0') << encoded.key
            << std::setfill(' ') << std::dec
            << " rank=" << encoded.rank
            << " bitmap_len=" << encoded.bitmap_len
            << " cell_rows=" << reader.descriptor(encoded.cid).success_rows;
        (void)matrix;
    } else {
        std::cout << " encoded_invalid";
    }
    std::cout << '\n';
    print_board_grid(board);
}

void print_diff_samples(
    const char *label,
    const std::vector<uint64_t> &diff,
    const BC::BCLut &lut,
    const BC::BCPositionLayerReader &reader,
    const std::array<uint32_t, 16U> &tile_values,
    uint32_t sample
) {
    std::cout << label << "_count=" << diff.size() << '\n';
    const uint32_t limit = std::min<uint32_t>(sample, static_cast<uint32_t>(diff.size()));
    for (uint32_t i = 0U; i < limit; ++i) {
        print_board_info(label, diff[i], lut, reader, tile_values);
    }
}

void print_diff_groups(
    const char *label,
    const std::vector<uint64_t> &diff,
    const BC::BCLut &lut,
    const BC::BCPositionLayerReader &reader,
    uint32_t target_family_count
) {
    struct GroupKey {
        uint64_t row_coord = 0U;
        uint64_t col_coord = 0U;
        uint32_t rank15 = 0U;

        [[nodiscard]] bool operator<(const GroupKey &other) const {
            if (row_coord != other.row_coord) {
                return row_coord < other.row_coord;
            }
            if (col_coord != other.col_coord) {
                return col_coord < other.col_coord;
            }
            return rank15 < other.rank15;
        }
    };
    std::map<GroupKey, uint64_t> groups;
    for (uint64_t board : diff) {
        const BC::BCQuadrantWords q = BC::unpack_board_to_quadrants(board);
        const BoardFamilyInfo info = board_family_info(lut, reader, q);
        if (!info.valid) {
            continue;
        }
        ++groups[GroupKey{info.row_coord, info.col_coord, tile_count(board, 15U)}];
    }
    std::cout << label << "_groups=" << groups.size() << '\n';
    uint32_t printed = 0U;
    for (const auto &entry : groups) {
        const GroupKey &key = entry.first;
        const uint32_t row_mod = target_family_count == 0U
            ? 0U
            : static_cast<uint32_t>(key.row_coord % target_family_count);
        const uint32_t col_mod = target_family_count == 0U
            ? 0U
            : static_cast<uint32_t>(key.col_coord % target_family_count);
        std::cout
            << label
            << "_group row_coord=" << key.row_coord
            << " col_coord=" << key.col_coord
            << " target_mod=[" << row_mod << ',' << col_mod << ']'
            << " rank15=" << key.rank15
            << " count=" << entry.second
            << '\n';
        if (++printed >= 32U) {
            break;
        }
    }
}

void print_diff_quadrant_sum_groups(
    const char *label,
    const std::vector<uint64_t> &diff,
    const BC::BCLut &lut,
    const BC::BCPositionLayerReader &reader
) {
    struct Key {
        uint64_t row_coord = 0U;
        uint64_t col_coord = 0U;
        uint32_t nw = 0U;
        uint32_t ne = 0U;
        uint32_t sw = 0U;
        uint32_t se = 0U;

        [[nodiscard]] bool operator<(const Key &other) const {
            if (row_coord != other.row_coord) {
                return row_coord < other.row_coord;
            }
            if (col_coord != other.col_coord) {
                return col_coord < other.col_coord;
            }
            if (nw != other.nw) {
                return nw < other.nw;
            }
            if (ne != other.ne) {
                return ne < other.ne;
            }
            if (sw != other.sw) {
                return sw < other.sw;
            }
            return se < other.se;
        }
    };
    std::map<Key, uint64_t> groups;
    for (uint64_t board : diff) {
        const BoardFamilyInfo info =
            board_family_info(lut, reader, BC::unpack_board_to_quadrants(board));
        if (info.valid) {
            ++groups[Key{
                info.row_coord,
                info.col_coord,
                info.nw_sum,
                info.ne_sum,
                info.sw_sum,
                info.se_sum
            }];
        }
    }
    std::cout << label << "_qsum_groups=" << groups.size() << '\n';
    uint32_t printed = 0U;
    for (const auto &entry : groups) {
        const Key &key = entry.first;
        std::cout
            << label
            << "_qsum row_coord=" << key.row_coord
            << " col_coord=" << key.col_coord
            << " qsum=[" << key.nw << ',' << key.ne << ',' << key.sw << ',' << key.se << ']'
            << " count=" << entry.second
            << '\n';
        if (++printed >= 32U) {
            break;
        }
    }
}

void print_diff_key_groups(
    const char *label,
    const std::vector<uint64_t> &diff,
    const BC::BCLut &lut
) {
    struct Value {
        uint64_t count = 0U;
        uint32_t min_rank = std::numeric_limits<uint32_t>::max();
        uint32_t max_rank = 0U;
        std::vector<uint32_t> ranks;
    };
    std::map<uint64_t, Value> groups;
    for (uint64_t board : diff) {
        const BC::BCQuadrantWords q = BC::unpack_board_to_quadrants(board);
        const BC::BCEncodedKeyRank encoded =
            BC::encode_key_and_rank(lut, q.nw, q.ne, q.sw, q.se);
        if (!encoded.valid) {
            continue;
        }
        Value &value = groups[encoded.key];
        ++value.count;
        value.min_rank = std::min<uint32_t>(value.min_rank, encoded.rank);
        value.max_rank = std::max<uint32_t>(value.max_rank, encoded.rank);
        if (value.ranks.size() < 128U) {
            value.ranks.push_back(encoded.rank);
        }
    }
    std::cout << label << "_key_groups=" << groups.size() << '\n';
    uint32_t printed = 0U;
    for (const auto &entry : groups) {
        const BC::BCBucketRankDecoder decoder(lut, entry.first);
        std::cout
            << label
            << "_key key=0x" << std::hex << std::setw(16) << std::setfill('0') << entry.first
            << std::setfill(' ') << std::dec
            << " count=" << entry.second.count
            << " min_rank=" << entry.second.min_rank
            << " max_rank=" << entry.second.max_rank
            << " dims=[" << decoder.count_ne << ',' << decoder.count_sw << ','
            << decoder.count_se << ']';
        if (entry.second.count <= 128U) {
            std::vector<uint32_t> ranks = entry.second.ranks;
            std::sort(ranks.begin(), ranks.end());
            std::cout << " ranks=";
            for (uint32_t rank : ranks) {
                std::cout << rank << ';';
            }
            std::cout << " parts=";
            for (uint32_t rank : ranks) {
                const uint32_t rank_se = rank % decoder.count_se;
                const uint32_t tmp = rank / decoder.count_se;
                const uint32_t rank_sw = tmp % decoder.count_sw;
                const uint32_t rank_ne = tmp / decoder.count_sw;
                std::cout << '(' << rank_ne << ',' << rank_sw << ',' << rank_se << ");";
            }
        }
        std::cout << '\n';
        if (++printed >= 32U) {
            break;
        }
    }
}

[[nodiscard]] uint64_t count_exact_coord_group(
    const std::vector<uint64_t> &boards,
    const BC::BCLut &lut,
    const BC::BCPositionLayerReader &reader,
    uint64_t row_coord,
    uint64_t col_coord
) {
    uint64_t count = 0U;
    for (uint64_t board : boards) {
        const BoardFamilyInfo info =
            board_family_info(lut, reader, BC::unpack_board_to_quadrants(board));
        if (info.valid && info.row_coord == row_coord && info.col_coord == col_coord) {
            ++count;
        }
    }
    return count;
}

[[nodiscard]] uint64_t count_exact_qsum_group(
    const std::vector<uint64_t> &boards,
    const BC::BCLut &lut,
    const BC::BCPositionLayerReader &reader,
    const BoardFamilyInfo &needle
) {
    uint64_t count = 0U;
    for (uint64_t board : boards) {
        const BoardFamilyInfo info =
            board_family_info(lut, reader, BC::unpack_board_to_quadrants(board));
        if (info.valid &&
            info.row_coord == needle.row_coord &&
            info.col_coord == needle.col_coord &&
            info.nw_sum == needle.nw_sum &&
            info.ne_sum == needle.ne_sum &&
            info.sw_sum == needle.sw_sum &&
            info.se_sum == needle.se_sum) {
            ++count;
        }
    }
    return count;
}

void print_diff_group_coverage(
    const char *label,
    const std::vector<uint64_t> &diff,
    const std::vector<uint64_t> &boards_a,
    const std::vector<uint64_t> &boards_b,
    const BC::BCLut &lut,
    const BC::BCPositionLayerReader &reader_a,
    const BC::BCPositionLayerReader &reader_b,
    uint32_t target_family_count
) {
    struct CoordKey {
        uint64_t row_coord = 0U;
        uint64_t col_coord = 0U;

        [[nodiscard]] bool operator<(const CoordKey &other) const {
            return row_coord != other.row_coord
                ? row_coord < other.row_coord
                : col_coord < other.col_coord;
        }
    };
    std::map<CoordKey, uint64_t> groups;
    for (uint64_t board : diff) {
        const BoardFamilyInfo info =
            board_family_info(lut, reader_a, BC::unpack_board_to_quadrants(board));
        if (info.valid) {
            ++groups[CoordKey{info.row_coord, info.col_coord}];
        }
    }
    uint32_t printed = 0U;
    for (const auto &entry : groups) {
        const CoordKey &key = entry.first;
        const uint32_t row_mod = target_family_count == 0U
            ? 0U
            : static_cast<uint32_t>(key.row_coord % target_family_count);
        const uint32_t col_mod = target_family_count == 0U
            ? 0U
            : static_cast<uint32_t>(key.col_coord % target_family_count);
        std::cout
            << label
            << "_coverage row_coord=" << key.row_coord
            << " col_coord=" << key.col_coord
            << " target_mod=[" << row_mod << ',' << col_mod << ']'
            << " diff_count=" << entry.second
            << " a_group_count=" << count_exact_coord_group(boards_a, lut, reader_a, key.row_coord, key.col_coord)
            << " b_group_count=" << count_exact_coord_group(boards_b, lut, reader_b, key.row_coord, key.col_coord)
            << '\n';
        if (++printed >= 16U) {
            break;
        }
    }
}

void print_diff_qsum_coverage(
    const char *label,
    const std::vector<uint64_t> &diff,
    const std::vector<uint64_t> &boards_a,
    const std::vector<uint64_t> &boards_b,
    const BC::BCLut &lut,
    const BC::BCPositionLayerReader &reader_a,
    const BC::BCPositionLayerReader &reader_b
) {
    struct Key {
        uint64_t row_coord = 0U;
        uint64_t col_coord = 0U;
        uint32_t nw = 0U;
        uint32_t ne = 0U;
        uint32_t sw = 0U;
        uint32_t se = 0U;

        [[nodiscard]] bool operator<(const Key &other) const {
            if (row_coord != other.row_coord) {
                return row_coord < other.row_coord;
            }
            if (col_coord != other.col_coord) {
                return col_coord < other.col_coord;
            }
            if (nw != other.nw) {
                return nw < other.nw;
            }
            if (ne != other.ne) {
                return ne < other.ne;
            }
            if (sw != other.sw) {
                return sw < other.sw;
            }
            return se < other.se;
        }
    };
    std::map<Key, uint64_t> groups;
    for (uint64_t board : diff) {
        const BoardFamilyInfo info =
            board_family_info(lut, reader_a, BC::unpack_board_to_quadrants(board));
        if (info.valid) {
            ++groups[Key{
                info.row_coord,
                info.col_coord,
                info.nw_sum,
                info.ne_sum,
                info.sw_sum,
                info.se_sum
            }];
        }
    }
    uint32_t printed = 0U;
    for (const auto &entry : groups) {
        const Key &key = entry.first;
        BoardFamilyInfo needle;
        needle.valid = true;
        needle.row_coord = key.row_coord;
        needle.col_coord = key.col_coord;
        needle.nw_sum = key.nw;
        needle.ne_sum = key.ne;
        needle.sw_sum = key.sw;
        needle.se_sum = key.se;
        std::cout
            << label
            << "_qsum_coverage row_coord=" << key.row_coord
            << " col_coord=" << key.col_coord
            << " qsum=[" << key.nw << ',' << key.ne << ',' << key.sw << ',' << key.se << ']'
            << " diff_count=" << entry.second
            << " a_qsum_count=" << count_exact_qsum_group(boards_a, lut, reader_a, needle)
            << " b_qsum_count=" << count_exact_qsum_group(boards_b, lut, reader_b, needle)
            << '\n';
        if (++printed >= 16U) {
            break;
        }
    }
}

[[nodiscard]] Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto require_value = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(name) + " requires a value");
            }
            return argv[++i];
        };
        if (key == "--a") {
            args.a = require_value("--a");
        } else if (key == "--b") {
            args.b = require_value("--b");
        } else if (key == "--target-rank") {
            args.target_rank = static_cast<uint32_t>(std::stoul(require_value("--target-rank")));
        } else if (key == "--sample") {
            args.sample = static_cast<uint32_t>(std::stoul(require_value("--sample")));
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.a.empty() || args.b.empty()) {
        throw std::invalid_argument("--a and --b are required");
    }
    return args;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Args args = parse_args(argc, argv);
        const BC::BCLut lut = make_lut(args.target_rank);
        const std::array<uint32_t, 16U> tile_values = BC::default_2048_tile_sum_values();

        BC::BCPositionFileReader file_a = BC::BCPositionFileReader::open_buffered(args.a, lut);
        BC::BCPositionFileReader file_b = BC::BCPositionFileReader::open_buffered(args.b, lut);
        const BC::BCPositionLayerReader &reader_a = file_a.layer();
        const BC::BCPositionLayerReader &reader_b = file_b.layer();
        std::cout
            << "a_layer=" << reader_a.axis().layer_sum()
            << " a_families=" << reader_a.axis().family_count()
            << " a_rows=" << descriptor_rows(reader_a) << '\n'
            << "b_layer=" << reader_b.axis().layer_sum()
            << " b_families=" << reader_b.axis().family_count()
            << " b_rows=" << descriptor_rows(reader_b) << '\n';
        if (reader_a.axis().layer_sum() != reader_b.axis().layer_sum()) {
            throw std::runtime_error("layer sums differ");
        }

        std::vector<uint64_t> boards_a = load_boards(args.a, lut);
        std::vector<uint64_t> boards_b = load_boards(args.b, lut);
        std::cout << "a_unique=" << boards_a.size() << " b_unique=" << boards_b.size() << '\n';

        std::vector<uint64_t> a_only;
        std::vector<uint64_t> b_only;
        std::set_difference(
            boards_a.begin(),
            boards_a.end(),
            boards_b.begin(),
            boards_b.end(),
            std::back_inserter(a_only)
        );
        std::set_difference(
            boards_b.begin(),
            boards_b.end(),
            boards_a.begin(),
            boards_a.end(),
            std::back_inserter(b_only)
        );

        print_diff_groups("a_only", a_only, lut, reader_a, reader_b.axis().family_count());
        print_diff_groups("b_only", b_only, lut, reader_b, reader_a.axis().family_count());
        print_diff_quadrant_sum_groups("a_only", a_only, lut, reader_a);
        print_diff_quadrant_sum_groups("b_only", b_only, lut, reader_b);
        print_diff_key_groups("a_only", a_only, lut);
        print_diff_key_groups("b_only", b_only, lut);
        print_diff_group_coverage(
            "a_only",
            a_only,
            boards_a,
            boards_b,
            lut,
            reader_a,
            reader_b,
            reader_b.axis().family_count()
        );
        print_diff_qsum_coverage(
            "a_only",
            a_only,
            boards_a,
            boards_b,
            lut,
            reader_a,
            reader_b
        );
        print_diff_group_coverage(
            "b_only",
            b_only,
            boards_b,
            boards_a,
            lut,
            reader_b,
            reader_a,
            reader_a.axis().family_count()
        );
        print_diff_qsum_coverage(
            "b_only",
            b_only,
            boards_b,
            boards_a,
            lut,
            reader_b,
            reader_a
        );
        print_diff_samples("a_only", a_only, lut, reader_a, tile_values, args.sample);
        print_diff_samples("b_only", b_only, lut, reader_b, tile_values, args.sample);
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "bc_family_layer_diff_tool failed: " << ex.what() << '\n';
        return 1;
    }
}
