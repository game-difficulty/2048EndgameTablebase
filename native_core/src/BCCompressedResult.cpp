#include "BCCompressedResult.h"

#include "NativeLzma.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>

namespace BCCompressedResult {
namespace {

constexpr char kMagic[8] = {'B', 'C', 'C', 'M', 'P', '1', '\0', '\0'};
constexpr uint32_t kFormatVersion = 1U;

template <typename T>
[[nodiscard]] T load_pod(const uint8_t *data) {
    T out{};
    std::memcpy(&out, data, sizeof(T));
    return out;
}

template <typename T>
void append_pod(std::vector<uint8_t> &out, const T &value) {
    const size_t offset = out.size();
    out.resize(offset + sizeof(T));
    std::memcpy(out.data() + offset, &value, sizeof(T));
}

template <typename T>
void append_pod_array(std::vector<uint8_t> &out, const std::vector<T> &values) {
    if (values.empty()) {
        return;
    }
    const size_t offset = out.size();
    const size_t bytes = values.size() * sizeof(T);
    out.resize(offset + bytes);
    std::memcpy(out.data() + offset, values.data(), bytes);
}

[[nodiscard]] double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

void read_exact(std::ifstream &in, uint64_t offset, void *data, uint64_t size, const char *what) {
    if (size == 0U) {
        return;
    }
    if (size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::overflow_error(std::string("BC compressed read too large: ") + what);
    }
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in) {
        throw std::runtime_error(std::string("BC compressed seek failed: ") + what);
    }
    in.read(static_cast<char *>(data), static_cast<std::streamsize>(size));
    if (!in) {
        throw std::runtime_error(std::string("BC compressed read failed: ") + what);
    }
}

[[nodiscard]] std::vector<uint8_t> read_range(
    const std::filesystem::path &path,
    uint64_t offset,
    uint64_t size,
    const char *what
) {
    if (size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error(std::string("BC compressed range too large: ") + what);
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open BC compressed file: " + path.string());
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    read_exact(in, offset, bytes.data(), size, what);
    return bytes;
}

template <typename T>
[[nodiscard]] T read_pod_at(
    const std::filesystem::path &path,
    uint64_t offset,
    const char *what
) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open BC compressed file: " + path.string());
    }
    std::array<uint8_t, sizeof(T)> bytes{};
    read_exact(in, offset, bytes.data(), bytes.size(), what);
    return load_pod<T>(bytes.data());
}

void write_at(std::fstream &out, uint64_t offset, const void *data, uint64_t size, const char *what) {
    if (size == 0U) {
        return;
    }
    if (size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::overflow_error(std::string("BC compressed write too large: ") + what);
    }
    out.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!out) {
        throw std::runtime_error(std::string("BC compressed seek for write failed: ") + what);
    }
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    if (!out) {
        throw std::runtime_error(std::string("BC compressed write failed: ") + what);
    }
}

void write_bytes(std::fstream &out, const void *data, uint64_t size, const char *what) {
    if (size == 0U) {
        return;
    }
    if (size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::overflow_error(std::string("BC compressed append too large: ") + what);
    }
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    if (!out) {
        throw std::runtime_error(std::string("BC compressed append failed: ") + what);
    }
}

[[nodiscard]] uint64_t file_offset(std::fstream &out) {
    const std::streampos pos = out.tellp();
    if (pos < 0) {
        throw std::runtime_error("BC compressed tellp failed");
    }
    return static_cast<uint64_t>(pos);
}

struct Header {
    char magic[8]{};
    uint32_t version = kFormatVersion;
    uint32_t header_bytes = sizeof(Header);
    uint32_t dtype = BC::kBCSuccessDTypeUint32;
    uint32_t value_size = sizeof(uint32_t);
    uint32_t row_width = 0U;
    uint32_t family_count = 0U;
    uint32_t family_unit = 0U;
    uint32_t axis_base_coord = 0U;
    uint32_t position_key_mode = BC::kBCPositionKeyModeQ4NwExactNeSwSeSumMaskPrefix256;
    uint32_t rank_prefix_bits = BC::kBCRankPrefixBits;
    uint32_t rank_prefix_type = BC::kBCPositionRankPrefixTypeUint16;
    uint32_t rank_payload_align = 8U;
    uint32_t bucket_block_raw_target_bytes = 0U;
    uint32_t bucket_block_raw_hard_cap_bytes = 0U;
    uint32_t value_block_raw_target_bytes = 0U;
    uint32_t value_block_raw_hard_cap_bytes = 0U;
    uint64_t layer_sum = 0U;
    uint64_t cell_count = 0U;
    uint64_t success_value_count = 0U;
    uint64_t live_rows = 0U;
    uint64_t data_offset = 0U;
    uint64_t data_bytes = 0U;
    uint64_t axis_offset = 0U;
    uint64_t axis_bytes = 0U;
    uint64_t cell_dir_offset = 0U;
    uint64_t cell_dir_count = 0U;
    uint64_t bucket_dir_offset = 0U;
    uint64_t bucket_block_count = 0U;
    uint64_t value_dir_offset = 0U;
    uint64_t value_block_count = 0U;
    uint64_t original_position_bytes = 0U;
    uint64_t original_success_bytes = 0U;
    uint64_t position_metadata_fingerprint = 0U;
    uint64_t reserved0 = 0U;
};

struct CellDirEntry {
    uint64_t value_base = 0U;
    uint32_t bucket_block_begin = 0U;
    uint32_t bucket_block_count = 0U;
    uint32_t success_rows = 0U;
    uint32_t flags = 0U;
};

struct BucketBlockDirEntry {
    uint32_t cid = 0U;
    uint32_t first_bucket_index = 0U;
    uint32_t bucket_count = 0U;
    uint32_t first_success_row = 0U;
    uint64_t first_key = 0U;
    uint64_t last_key = 0U;
    uint64_t compressed_offset = 0U;
    uint64_t compressed_size = 0U;
    uint64_t raw_size = 0U;
};

struct ValueBlockDirEntry {
    uint64_t first_value_index = 0U;
    uint32_t value_count = 0U;
    uint32_t value_size = 0U;
    uint64_t compressed_offset = 0U;
    uint64_t compressed_size = 0U;
    uint64_t raw_size = 0U;
};

struct BucketBlockRawHeader {
    uint32_t bucket_count = 0U;
    uint32_t rank_payload_bytes = 0U;
    uint32_t local_rank_payload_offsets = 0U;
    uint32_t reserved = 0U;
};

static_assert(sizeof(CellDirEntry) == 24U);
static_assert(sizeof(BucketBlockRawHeader) == 16U);

enum class BlockKind : uint8_t {
    Bucket,
    Value,
};

struct PendingBlock {
    BlockKind kind = BlockKind::Bucket;
    size_t dir_index = 0U;
    uint64_t raw_size = 0U;
    std::future<std::vector<uint8_t>> compressed;
};

struct BuilderState {
    Header header{};
    CompressOptions options;
    CompressStats stats;
    std::vector<BC::FamilyCoord> axis_coords;
    std::vector<CellDirEntry> cell_dirs;
    std::vector<BucketBlockDirEntry> bucket_dirs;
    std::vector<ValueBlockDirEntry> value_dirs;
    uint64_t value_cursor = 0U;
    uint64_t data_begin = sizeof(Header);
};

[[nodiscard]] uint32_t normalized_worker_count(const CompressOptions &options) {
    if (options.worker_count != 0U) {
        return std::max<uint32_t>(1U, options.worker_count);
    }
    const uint32_t hw = std::thread::hardware_concurrency();
    return std::max<uint32_t>(1U, hw == 0U ? 1U : hw);
}

class BlockCompressor {
public:
    BlockCompressor(std::fstream &out, BuilderState &state)
        : out_(out), state_(state), max_pending_(normalized_worker_count(state.options)) {}

    void submit(BlockKind kind, size_t dir_index, std::vector<uint8_t> raw) {
        const uint64_t raw_size = static_cast<uint64_t>(raw.size());
        const int level = static_cast<int>(state_.options.compression_level);
        pending_.push_back(PendingBlock{
            kind,
            dir_index,
            raw_size,
            std::async(
                std::launch::async,
                [bytes = std::move(raw), level]() {
                    return compress_xz_block_native(bytes.data(), bytes.size(), level);
                })
        });
        if (pending_.size() >= max_pending_) {
            flush_one();
        }
    }

    void finish() {
        while (!pending_.empty()) {
            flush_one();
        }
    }

private:
    void flush_one() {
        PendingBlock block = std::move(pending_.front());
        pending_.erase(pending_.begin());
        std::vector<uint8_t> compressed = block.compressed.get();
        const uint64_t offset = file_offset(out_);
        write_bytes(out_, compressed.data(), static_cast<uint64_t>(compressed.size()), "BC compressed data block");
        if (block.kind == BlockKind::Bucket) {
            BucketBlockDirEntry &entry = state_.bucket_dirs.at(block.dir_index);
            entry.compressed_offset = offset;
            entry.compressed_size = static_cast<uint64_t>(compressed.size());
            entry.raw_size = block.raw_size;
            state_.stats.bucket_raw_bytes += block.raw_size;
            state_.stats.bucket_compressed_bytes += static_cast<uint64_t>(compressed.size());
        } else {
            ValueBlockDirEntry &entry = state_.value_dirs.at(block.dir_index);
            entry.compressed_offset = offset;
            entry.compressed_size = static_cast<uint64_t>(compressed.size());
            entry.raw_size = block.raw_size;
            state_.stats.value_raw_bytes += block.raw_size;
            state_.stats.value_compressed_bytes += static_cast<uint64_t>(compressed.size());
        }
    }

    std::fstream &out_;
    BuilderState &state_;
    size_t max_pending_ = 1U;
    std::vector<PendingBlock> pending_;
};

[[nodiscard]] uint32_t bucket_rank_payload_end(
    const BC::BCLut &lut,
    const BC::BCBucketEntry &bucket
) {
    const uint32_t bitmap_len = BC::bitmap_len_from_key(lut, bucket.key);
    const uint32_t bitmap_offset = BC::bc_rank_payload_bitmap_offset(
        bucket.rank_payload_offset,
        bitmap_len);
    const uint32_t bitmap_bytes =
        BC::words_for_bits(bitmap_len) * static_cast<uint32_t>(sizeof(uint64_t));
    return BC::checked_u32_add(bitmap_offset, bitmap_bytes, "BC compressed bucket payload end overflow");
}

void append_bucket_block_raw(
    const BC::BCLut &lut,
    const std::vector<BC::BCBucketEntry> &buckets,
    const std::vector<uint8_t> &rank_payload,
    uint32_t begin,
    uint32_t end,
    std::vector<uint8_t> &raw
) {
    raw.clear();
    const uint32_t count = end - begin;
    std::vector<uint64_t> keys;
    std::vector<uint32_t> success_offsets;
    std::vector<uint32_t> local_offsets;
    keys.reserve(count);
    success_offsets.reserve(count);
    local_offsets.reserve(static_cast<size_t>(count) + 1U);
    std::vector<uint8_t> payload;

    uint32_t payload_cursor = 0U;
    local_offsets.push_back(0U);
    for (uint32_t i = begin; i < end; ++i) {
        const BC::BCBucketEntry &bucket = buckets.at(i);
        const uint32_t slice_begin = bucket.rank_payload_offset;
        const uint32_t slice_end = bucket_rank_payload_end(lut, bucket);
        if (slice_end < slice_begin || slice_end > rank_payload.size()) {
            throw std::runtime_error("BC compressed bucket rank payload slice exceeds cell payload");
        }
        keys.push_back(bucket.key);
        success_offsets.push_back(bucket.success_row_offset);
        payload.insert(
            payload.end(),
            rank_payload.begin() + static_cast<std::ptrdiff_t>(slice_begin),
            rank_payload.begin() + static_cast<std::ptrdiff_t>(slice_end));
        payload_cursor = BC::checked_u32_add(
            payload_cursor,
            slice_end - slice_begin,
            "BC compressed raw bucket payload cursor overflow");
        local_offsets.push_back(payload_cursor);
    }

    BucketBlockRawHeader header;
    header.bucket_count = count;
    header.rank_payload_bytes = payload_cursor;
    header.local_rank_payload_offsets = count + 1U;
    append_pod(raw, header);
    append_pod_array(raw, keys);
    append_pod_array(raw, success_offsets);
    append_pod_array(raw, local_offsets);
    raw.insert(raw.end(), payload.begin(), payload.end());
}

[[nodiscard]] uint64_t bucket_block_raw_estimate(
    const BC::BCLut &lut,
    const std::vector<BC::BCBucketEntry> &buckets,
    uint32_t begin,
    uint32_t end
) {
    uint64_t payload = 0U;
    for (uint32_t i = begin; i < end; ++i) {
        const BC::BCBucketEntry &bucket = buckets.at(i);
        const uint32_t slice_end = bucket_rank_payload_end(lut, bucket);
        if (slice_end < bucket.rank_payload_offset) {
            throw std::runtime_error("BC compressed bucket rank payload invalid range");
        }
        payload += slice_end - bucket.rank_payload_offset;
    }
    const uint64_t count = end - begin;
    return sizeof(BucketBlockRawHeader) +
        count * sizeof(uint64_t) +
        count * sizeof(uint32_t) +
        (count + 1U) * sizeof(uint32_t) +
        payload;
}

void emit_bucket_blocks_for_cell(
    BuilderState &state,
    BlockCompressor &compressor,
    const BC::BCLut &lut,
    BC::CellId cid,
    const std::vector<BC::BCBucketEntry> &buckets,
    const std::vector<uint8_t> &rank_payload
) {
    CellDirEntry &cell = state.cell_dirs.at(cid);
    cell.bucket_block_begin = BC::checked_u32_size(
        state.bucket_dirs.size(),
        "BC compressed bucket dir count exceeds uint32");
    uint32_t begin = 0U;
    while (begin < buckets.size()) {
        uint32_t end = begin + 1U;
        while (end < buckets.size()) {
            const uint64_t estimate = bucket_block_raw_estimate(lut, buckets, begin, end + 1U);
            if (estimate > state.options.bucket_block_raw_target_bytes && end > begin) {
                break;
            }
            if (estimate > state.options.bucket_block_raw_hard_cap_bytes && end > begin) {
                break;
            }
            ++end;
        }

        std::vector<uint8_t> raw;
        append_bucket_block_raw(lut, buckets, rank_payload, begin, end, raw);
        if (raw.size() > state.options.bucket_block_raw_hard_cap_bytes) {
            throw std::runtime_error("BC compressed bucket block exceeds hard cap");
        }

        BucketBlockDirEntry dir;
        dir.cid = cid;
        dir.first_bucket_index = begin;
        dir.bucket_count = end - begin;
        dir.first_success_row = buckets.at(begin).success_row_offset;
        dir.first_key = buckets.at(begin).key;
        dir.last_key = buckets.at(end - 1U).key;
        const size_t dir_index = state.bucket_dirs.size();
        state.bucket_dirs.push_back(dir);
        compressor.submit(BlockKind::Bucket, dir_index, std::move(raw));
        begin = end;
    }
    cell.bucket_block_count = BC::checked_u32_size(
        state.bucket_dirs.size() - cell.bucket_block_begin,
        "BC compressed cell bucket block count exceeds uint32");
}

void emit_value_blocks_for_cell(
    BuilderState &state,
    BlockCompressor &compressor,
    BC::CellId cid,
    const uint8_t *bytes,
    uint64_t byte_count
) {
    const uint32_t value_size = state.header.value_size;
    if ((byte_count % value_size) != 0U) {
        throw std::runtime_error("BC compressed success cell bytes are not value-aligned");
    }
    const uint64_t value_count = byte_count / value_size;
    CellDirEntry &cell = state.cell_dirs.at(cid);
    cell.value_base = state.value_cursor;
    const uint64_t values_per_block = std::max<uint64_t>(
        1U,
        state.options.value_block_raw_target_bytes / value_size);
    uint64_t cursor = 0U;
    while (cursor < value_count) {
        const uint64_t take = std::min<uint64_t>(values_per_block, value_count - cursor);
        const uint64_t raw_bytes = take * value_size;
        if (raw_bytes > state.options.value_block_raw_hard_cap_bytes) {
            throw std::runtime_error("BC compressed value block exceeds hard cap");
        }
        std::vector<uint8_t> raw(
            bytes + static_cast<size_t>(cursor * value_size),
            bytes + static_cast<size_t>((cursor + take) * value_size));
        ValueBlockDirEntry dir;
        dir.first_value_index = state.value_cursor + cursor;
        dir.value_count = BC::checked_u32_size(take, "BC compressed value block count exceeds uint32");
        dir.value_size = value_size;
        const size_t dir_index = state.value_dirs.size();
        state.value_dirs.push_back(dir);
        compressor.submit(BlockKind::Value, dir_index, std::move(raw));
        cursor += take;
    }
    state.value_cursor += value_count;
    state.stats.success_values += value_count;
}

[[nodiscard]] const uint8_t *success_cell_bytes(
    const BC::BCLoadedSuccessCell &cell,
    std::vector<uint8_t> &scratch
) {
    scratch.clear();
    if (cell.external_value_data != nullptr) {
        return cell.external_value_data;
    }
    if (!cell.raw_bytes.empty()) {
        return cell.raw_bytes.data();
    }
    if (!cell.values.empty()) {
#if defined(_WIN32) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
        return reinterpret_cast<const uint8_t *>(cell.values.data());
#else
        scratch.reserve(cell.values.size() * sizeof(uint32_t));
        for (uint32_t value : cell.values) {
            BC::bc_append_u32_le(scratch, value);
        }
        return scratch.data();
#endif
    }
    return nullptr;
}

void initialize_state_from_position(
    BuilderState &state,
    const BC::BCPositionStreamingReader &position,
    const BC::BCSuccessHeader &success_header,
    uint64_t position_bytes,
    uint64_t success_bytes
) {
    const BC::BCPositionHeader &p = position.header();
    std::memcpy(state.header.magic, kMagic, sizeof(kMagic));
    state.header.version = kFormatVersion;
    state.header.header_bytes = sizeof(Header);
    state.header.dtype = success_header.dtype;
    state.header.value_size = BC::bc_success_dtype_value_size(BC::bc_success_dtype_from_u32(success_header.dtype));
    state.header.row_width = success_header.row_width;
    state.header.family_count = p.family_count;
    state.header.family_unit = p.family_unit;
    state.header.axis_base_coord = p.axis_base_coord;
    state.header.position_key_mode = p.key_mode;
    state.header.rank_prefix_bits = p.rank_prefix_bits;
    state.header.rank_prefix_type = p.rank_prefix_type;
    state.header.rank_payload_align = p.rank_payload_align;
    state.header.bucket_block_raw_target_bytes = state.options.bucket_block_raw_target_bytes;
    state.header.bucket_block_raw_hard_cap_bytes = state.options.bucket_block_raw_hard_cap_bytes;
    state.header.value_block_raw_target_bytes = state.options.value_block_raw_target_bytes;
    state.header.value_block_raw_hard_cap_bytes = state.options.value_block_raw_hard_cap_bytes;
    state.header.layer_sum = p.layer_sum;
    state.header.cell_count = position.cell_count();
    state.header.data_offset = sizeof(Header);
    state.header.original_position_bytes = position_bytes;
    state.header.original_success_bytes = success_bytes;
    state.header.position_metadata_fingerprint = BC::bc_success_position_fingerprint_for(position);
    state.axis_coords = position.axis().coords();
    state.cell_dirs.assign(position.cell_count(), CellDirEntry{});
    state.stats.cells = position.cell_count();
}

void finalize_file(std::fstream &out, BuilderState &state, const std::filesystem::path &tmp_path) {
    state.header.data_bytes = file_offset(out) - state.header.data_offset;
    state.header.axis_offset = file_offset(out);
    state.header.axis_bytes = static_cast<uint64_t>(state.axis_coords.size()) * sizeof(uint32_t);
    for (BC::FamilyCoord coord : state.axis_coords) {
        const uint32_t value = coord;
        write_bytes(out, &value, sizeof(value), "BC compressed axis");
    }

    state.header.cell_dir_offset = file_offset(out);
    state.header.cell_dir_count = state.cell_dirs.size();
    write_bytes(
        out,
        state.cell_dirs.data(),
        static_cast<uint64_t>(state.cell_dirs.size()) * sizeof(CellDirEntry),
        "BC compressed cell dir");

    state.header.bucket_dir_offset = file_offset(out);
    state.header.bucket_block_count = state.bucket_dirs.size();
    write_bytes(
        out,
        state.bucket_dirs.data(),
        static_cast<uint64_t>(state.bucket_dirs.size()) * sizeof(BucketBlockDirEntry),
        "BC compressed bucket dir");

    state.header.value_dir_offset = file_offset(out);
    state.header.value_block_count = state.value_dirs.size();
    write_bytes(
        out,
        state.value_dirs.data(),
        static_cast<uint64_t>(state.value_dirs.size()) * sizeof(ValueBlockDirEntry),
        "BC compressed value dir");

    state.header.success_value_count = state.value_cursor;
    state.header.live_rows = state.stats.live_rows;
    write_at(out, 0U, &state.header, sizeof(state.header), "BC compressed header");
    out.flush();
    if (!out) {
        throw std::runtime_error("BC compressed flush failed: " + tmp_path.string());
    }
    state.stats.bucket_blocks = state.bucket_dirs.size();
    state.stats.value_blocks = state.value_dirs.size();
    state.stats.output_bytes = file_offset(out);
}

[[nodiscard]] BC::BCSuccessHeader read_success_header_from_path(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open BC success file: " + path.string());
    }
    std::vector<uint8_t> bytes(BC::kBCSuccessHeaderBytes);
    read_exact(in, 0U, bytes.data(), bytes.size(), "BC success header");
    return BC::bc_read_success_header(bytes);
}

void publish_file(const std::filesystem::path &tmp, const std::filesystem::path &target) {
    std::error_code ec;
    std::filesystem::rename(tmp, target, ec);
    if (!ec) {
        return;
    }
    std::filesystem::remove(target, ec);
    ec.clear();
    std::filesystem::rename(tmp, target, ec);
    if (ec) {
        throw std::runtime_error("failed to publish BC compressed file: " + ec.message());
    }
}

void validate_options(const CompressOptions &options) {
    if (options.bucket_block_raw_target_bytes == 0U ||
        options.bucket_block_raw_hard_cap_bytes < options.bucket_block_raw_target_bytes ||
        options.value_block_raw_target_bytes == 0U ||
        options.value_block_raw_hard_cap_bytes < options.value_block_raw_target_bytes) {
        throw std::invalid_argument("BC compressed invalid block size options");
    }
}

[[nodiscard]] CompressStats compress_streaming_readers(
    BC::BCPositionStreamingReader &position,
    BC::BCSuccessStreamingReader &success,
    const std::filesystem::path &position_path,
    const std::filesystem::path &success_path,
    const std::filesystem::path &output_path,
    const CompressOptions &options
) {
    validate_options(options);
    const double t0 = now_seconds();
    BuilderState state;
    state.options = options;
    initialize_state_from_position(
        state,
        position,
        success.header(),
        position.file_size(),
        success.file_size());

    const std::filesystem::path tmp_path = output_path.string() + ".tmp";
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::error_code ec;
    std::filesystem::remove(tmp_path, ec);
    std::fstream out(tmp_path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open BC compressed output: " + tmp_path.string());
    }
    write_bytes(out, &state.header, sizeof(state.header), "BC compressed provisional header");
    BlockCompressor compressor(out, state);

    for (BC::CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const BC::BCPositionCellDescriptor &desc = position.descriptor(cid);
        CellDirEntry &cell_dir = state.cell_dirs.at(cid);
        cell_dir.value_base = state.value_cursor;
        cell_dir.success_rows = desc.success_rows;
        if (desc.empty() || desc.success_rows == 0U) {
            continue;
        }
        ++state.stats.non_empty_cells;
        state.stats.live_rows += desc.success_rows;
        BC::BCLoadedCell cell = position.load_cell(cid);
        emit_bucket_blocks_for_cell(
            state,
            compressor,
            position.lut(),
            cid,
            cell.buckets,
            cell.rank_payload);

        std::vector<BC::BCLoadedSuccessCell> success_cells =
            success.load_cells(std::vector<BC::CellId>{cid}, nullptr, true);
        if (success_cells.size() != 1U) {
            throw std::logic_error("BC compressed success load_cell size mismatch");
        }
        BC::BCLoadedSuccessCell &success_cell = success_cells.front();
        std::vector<uint8_t> scratch;
        const uint8_t *success_data = success_cell_bytes(success_cell, scratch);
        const uint64_t success_bytes =
            static_cast<uint64_t>(desc.success_rows) *
            success.row_width() *
            success.value_size();
        if (success_bytes != 0U && success_data == nullptr) {
            throw std::runtime_error("BC compressed success cell payload is null");
        }
        emit_value_blocks_for_cell(state, compressor, cid, success_data, success_bytes);
    }
    (void)position_path;
    (void)success_path;
    compressor.finish();
    finalize_file(out, state, tmp_path);
    out.close();
    publish_file(tmp_path, output_path);
    state.stats.total_seconds = now_seconds() - t0;
    return state.stats;
}

struct InMemoryPositionAdapter {
    const BC::BCPositionLayerReader &position;

    [[nodiscard]] const BC::BCPositionHeader &header() const { return position.header(); }
    [[nodiscard]] const BC::BCFamilyTable &axis() const { return position.axis(); }
    [[nodiscard]] uint32_t cell_count() const { return position.cell_count(); }
    [[nodiscard]] const BC::BCLut &lut() const { return position.lut(); }
    [[nodiscard]] const BC::BCPositionCellDescriptor &descriptor(BC::CellId cid) const {
        return position.descriptor(cid);
    }
    [[nodiscard]] BC::BCLoadedCell load_cell(BC::CellId cid) const {
        BC::BCLoadedCell cell;
        cell.cid = cid;
        cell.success_rows = position.descriptor(cid).success_rows;
        const BC::BCBucketEntryView buckets = position.bucket_entries_for_cell(cid);
        cell.buckets.assign(buckets.data, buckets.data + buckets.size);
        const BC::BCRankPayloadView payload = position.rank_payload_for_cell(cid);
        cell.rank_payload.assign(payload.data, payload.data + payload.size);
        return cell;
    }
};

[[nodiscard]] CompressStats compress_in_memory_impl(
    const BC::BCPositionLayerReader &position,
    const BC::BCSuccessLayerReader &success,
    const std::filesystem::path &output_path,
    const CompressOptions &options
) {
    validate_options(options);
    const double t0 = now_seconds();
    BuilderState state;
    state.options = options;

    BC::BCSuccessHeader success_header = success.header();
    const BC::BCPositionHeader &p = position.header();
    std::memcpy(state.header.magic, kMagic, sizeof(kMagic));
    state.header.version = kFormatVersion;
    state.header.header_bytes = sizeof(Header);
    state.header.dtype = success_header.dtype;
    state.header.value_size = BC::bc_success_dtype_value_size(BC::bc_success_dtype_from_u32(success_header.dtype));
    state.header.row_width = success_header.row_width;
    state.header.family_count = p.family_count;
    state.header.family_unit = p.family_unit;
    state.header.axis_base_coord = p.axis_base_coord;
    state.header.position_key_mode = p.key_mode;
    state.header.rank_prefix_bits = p.rank_prefix_bits;
    state.header.rank_prefix_type = p.rank_prefix_type;
    state.header.rank_payload_align = p.rank_payload_align;
    state.header.bucket_block_raw_target_bytes = options.bucket_block_raw_target_bytes;
    state.header.bucket_block_raw_hard_cap_bytes = options.bucket_block_raw_hard_cap_bytes;
    state.header.value_block_raw_target_bytes = options.value_block_raw_target_bytes;
    state.header.value_block_raw_hard_cap_bytes = options.value_block_raw_hard_cap_bytes;
    state.header.layer_sum = p.layer_sum;
    state.header.cell_count = position.cell_count();
    state.header.data_offset = sizeof(Header);
    state.header.original_position_bytes = position.bytes().size();
    state.header.original_success_bytes = BC::bc_success_logical_size(success.header());
    state.header.position_metadata_fingerprint = BC::bc_success_position_fingerprint(position);
    state.axis_coords = position.axis().coords();
    state.cell_dirs.assign(position.cell_count(), CellDirEntry{});
    state.stats.cells = position.cell_count();

    const std::filesystem::path tmp_path = output_path.string() + ".tmp";
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::error_code ec;
    std::filesystem::remove(tmp_path, ec);
    std::fstream out(tmp_path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open BC compressed output: " + tmp_path.string());
    }
    write_bytes(out, &state.header, sizeof(state.header), "BC compressed provisional header");
    BlockCompressor compressor(out, state);

    InMemoryPositionAdapter adapter{position};
    for (BC::CellId cid = 0U; cid < position.cell_count(); ++cid) {
        const BC::BCPositionCellDescriptor &desc = position.descriptor(cid);
        CellDirEntry &cell_dir = state.cell_dirs.at(cid);
        cell_dir.value_base = state.value_cursor;
        cell_dir.success_rows = desc.success_rows;
        if (desc.empty() || desc.success_rows == 0U) {
            continue;
        }
        ++state.stats.non_empty_cells;
        state.stats.live_rows += desc.success_rows;
        BC::BCLoadedCell cell = adapter.load_cell(cid);
        emit_bucket_blocks_for_cell(
            state,
            compressor,
            position.lut(),
            cid,
            cell.buckets,
            cell.rank_payload);
        std::vector<uint8_t> success_bytes = success.read_cell_raw(cid);
        emit_value_blocks_for_cell(
            state,
            compressor,
            cid,
            success_bytes.data(),
            static_cast<uint64_t>(success_bytes.size()));
    }
    compressor.finish();
    finalize_file(out, state, tmp_path);
    out.close();
    publish_file(tmp_path, output_path);
    state.stats.total_seconds = now_seconds() - t0;
    return state.stats;
}

void validate_header(const Header &header) {
    if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("BC compressed magic mismatch");
    }
    if (header.version != kFormatVersion || header.header_bytes != sizeof(Header)) {
        throw std::runtime_error("BC compressed version/header size mismatch");
    }
    (void)BC::bc_success_dtype_from_u32(header.dtype);
    if (header.value_size != BC::bc_success_dtype_value_size(BC::bc_success_dtype_from_u32(header.dtype))) {
        throw std::runtime_error("BC compressed dtype value size mismatch");
    }
    if (header.row_width == 0U || header.family_count == 0U || header.family_unit == 0U) {
        throw std::runtime_error("BC compressed invalid layer metadata");
    }
}

[[nodiscard]] double numeric_from_raw(uint32_t dtype, uint64_t raw_bits) {
    const BC::BCSuccessDTypeMode mode = BC::bc_success_dtype_from_u32(dtype);
    switch (mode) {
        case BC::BCSuccessDTypeMode::UInt32:
            return static_cast<double>(static_cast<uint32_t>(raw_bits));
        case BC::BCSuccessDTypeMode::UInt64:
            return static_cast<double>(raw_bits);
        case BC::BCSuccessDTypeMode::Float32: {
            uint32_t bits = static_cast<uint32_t>(raw_bits);
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            return static_cast<double>(value);
        }
        case BC::BCSuccessDTypeMode::OneMinusFloat32: {
            uint32_t bits = static_cast<uint32_t>(raw_bits);
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            return static_cast<double>(value);
        }
        case BC::BCSuccessDTypeMode::Float64:
        case BC::BCSuccessDTypeMode::OneMinusFloat64: {
            double value = 0.0;
            std::memcpy(&value, &raw_bits, sizeof(value));
            return value;
        }
    }
    return 0.0;
}

[[nodiscard]] uint64_t raw_bits_from_value(const uint8_t *data, uint32_t dtype) {
    const BC::BCSuccessDTypeMode mode = BC::bc_success_dtype_from_u32(dtype);
    switch (mode) {
        case BC::BCSuccessDTypeMode::UInt32:
            return BC::bc_load_u32_le(data);
        case BC::BCSuccessDTypeMode::UInt64:
            return BC::load_u64_le(data);
        case BC::BCSuccessDTypeMode::Float32:
        case BC::BCSuccessDTypeMode::OneMinusFloat32:
            return BC::bc_load_u32_le(data);
        case BC::BCSuccessDTypeMode::Float64:
        case BC::BCSuccessDTypeMode::OneMinusFloat64:
            return BC::load_u64_le(data);
    }
    return 0U;
}

[[nodiscard]] bool axis_looks_like_modulo_partition(const BC::BCFamilyTable &axis) {
    if (!axis.is_contiguous_range() || axis.axis_base_coord() != 0U ||
        axis.family_count() == 0U) {
        return false;
    }
    for (BC::FamilyId id = 0U; id < axis.family_count(); ++id) {
        if (axis.id_to_coord(id) != id) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] BC::BCBoardEncodedPosition encode_for_compressed_reader(
    const BC::BCLut &lut,
    const BC::BCFamilyTable &axis,
    uint64_t board
) {
    BC::BCBoardEncodedPosition out =
        BC::encode_spawned_canonical_board(lut, axis, board);
    if (out.valid || !axis_looks_like_modulo_partition(axis)) {
        return out;
    }

    const BC::BCQuadrantWords q = BC::unpack_board_to_quadrants(board);
    const BC::BCWordDesc &nw_desc = lut.word_desc(q.nw);
    const BC::BCWordDesc &ne_desc = lut.word_desc(q.ne);
    const BC::BCWordDesc &sw_desc = lut.word_desc(q.sw);
    const BC::BCWordDesc &se_desc = lut.word_desc(q.se);
    if (!nw_desc.valid || !ne_desc.valid || !sw_desc.valid || !se_desc.valid) {
        return {};
    }

    const uint64_t total_sum =
        static_cast<uint64_t>(nw_desc.sum) +
        static_cast<uint64_t>(ne_desc.sum) +
        static_cast<uint64_t>(sw_desc.sum) +
        static_cast<uint64_t>(se_desc.sum);
    if (total_sum != axis.layer_sum()) {
        return {};
    }

    BC::FamilyCoord row_coord = 0U;
    BC::FamilyCoord col_coord = 0U;
    if (!BC::bc_min_side_coord_u64(
            static_cast<uint64_t>(nw_desc.sum) + ne_desc.sum,
            static_cast<uint64_t>(sw_desc.sum) + se_desc.sum,
            axis.family_unit(),
            row_coord) ||
        !BC::bc_min_side_coord_u64(
            static_cast<uint64_t>(nw_desc.sum) + sw_desc.sum,
            static_cast<uint64_t>(ne_desc.sum) + se_desc.sum,
            axis.family_unit(),
            col_coord)) {
        return {};
    }

    const BC::BCEncodedKeyRank encoded =
        BC::bc_encode_key_rank_from_descs(lut, q.nw, nw_desc, ne_desc, sw_desc, se_desc);
    if (!encoded.valid) {
        return {};
    }

    const uint32_t family_count = axis.family_count();
    out.row_family = static_cast<BC::FamilyId>(row_coord % family_count);
    out.col_family = static_cast<BC::FamilyId>(col_coord % family_count);
    const uint64_t cid =
        static_cast<uint64_t>(out.row_family) * family_count +
        static_cast<uint32_t>(out.col_family);
    if (cid > std::numeric_limits<BC::CellId>::max()) {
        throw std::overflow_error("BC compressed modulo cid exceeds CellId");
    }
    out.cid = static_cast<BC::CellId>(cid);
    out.key = encoded.key;
    out.rank = encoded.rank;
    out.bitmap_len = encoded.bitmap_len;
    out.count_ne = encoded.count_ne;
    out.count_sw = encoded.count_sw;
    out.count_se = encoded.count_se;
    out.valid = true;
    return out;
}

[[nodiscard]] bool rank_for_bucket_payload_ordinal(
    const BC::BCLut &lut,
    uint64_t key,
    const uint8_t *payload,
    uint32_t payload_size,
    uint64_t ordinal,
    BC::BucketRank &rank_out
) {
    if (payload == nullptr && payload_size != 0U) {
        throw std::invalid_argument("BC compressed sample bucket payload is null");
    }
    const uint32_t bitmap_len = BC::bitmap_len_from_key(lut, key);
    const uint32_t bitmap_word_count = BC::words_for_bits(bitmap_len);
    const uint32_t bitmap_offset = BC::bc_rank_payload_bitmap_offset(0U, bitmap_len);
    const uint64_t bitmap_bytes =
        static_cast<uint64_t>(bitmap_word_count) * sizeof(uint64_t);
    if (bitmap_offset > payload_size || bitmap_bytes > payload_size - bitmap_offset) {
        throw std::runtime_error("BC compressed sample bucket payload is truncated");
    }
    const uint8_t *bitmap = payload + bitmap_offset;
    uint64_t remaining = ordinal;
    for (uint32_t word_i = 0U; word_i < bitmap_word_count; ++word_i) {
        uint64_t word = BC::load_u64_le(bitmap + static_cast<size_t>(word_i) * sizeof(uint64_t));
        const uint32_t first_bit = word_i * BC::kBCBitmapWordBits;
        if (first_bit + BC::kBCBitmapWordBits > bitmap_len) {
            const uint32_t valid_bits = bitmap_len - first_bit;
            if (valid_bits < 64U) {
                word &= (uint64_t{1} << valid_bits) - 1ULL;
            }
        }
        const uint32_t live = BC::popcount64(word);
        if (remaining >= live) {
            remaining -= live;
            continue;
        }
        for (uint32_t bit = 0U; bit < BC::kBCBitmapWordBits; ++bit) {
            const uint32_t rank_u32 = first_bit + bit;
            if (rank_u32 >= bitmap_len) {
                break;
            }
            if (((word >> bit) & 1ULL) == 0ULL) {
                continue;
            }
            if (remaining == 0U) {
                rank_out = static_cast<BC::BucketRank>(rank_u32);
                return true;
            }
            --remaining;
        }
        throw std::logic_error("BC compressed sample ordinal scan mismatch");
    }
    return false;
}

} // namespace

CompressStats compress_exact_layer_to_result(
    const std::filesystem::path &position_path,
    const std::filesystem::path &success_path,
    const BC::BCLut &lut,
    const std::filesystem::path &output_path,
    const CompressOptions &options
) {
    BC::BCPositionStreamingReader position =
        BC::BCPositionStreamingReader::open_buffered(position_path, lut);
    const BC::BCSuccessHeader success_header = read_success_header_from_path(success_path);
    BC::BCSuccessStreamingReader success =
        BC::BCSuccessStreamingReader::open_buffered(
            success_path,
            position,
            success_header.row_width);
    return compress_streaming_readers(
        position,
        success,
        position_path,
        success_path,
        output_path,
        options);
}

CompressStats compress_in_memory_layer_to_result(
    const BC::BCPositionLayerReader &position,
    const BC::BCSuccessLayerReader &success,
    const std::filesystem::path &output_path,
    const CompressOptions &options
) {
    return compress_in_memory_impl(position, success, output_path, options);
}

struct PointReader::Impl {
    std::filesystem::path path;
    const BC::BCLut *lut = nullptr;
    Header header{};
    BC::BCFamilyTable axis;
    std::vector<CellDirEntry> cell_dirs;

    Impl() : axis(0U, 1U, std::vector<BC::FamilyCoord>{0U}) {}

    [[nodiscard]] BucketBlockDirEntry read_bucket_dir(uint64_t index) const {
        if (index >= header.bucket_block_count) {
            throw std::out_of_range("BC compressed bucket dir index out of range");
        }
        return read_pod_at<BucketBlockDirEntry>(
            path,
            header.bucket_dir_offset + index * sizeof(BucketBlockDirEntry),
            "BC compressed bucket dir entry");
    }

    [[nodiscard]] ValueBlockDirEntry read_value_dir(uint64_t index) const {
        if (index >= header.value_block_count) {
            throw std::out_of_range("BC compressed value dir index out of range");
        }
        return read_pod_at<ValueBlockDirEntry>(
            path,
            header.value_dir_offset + index * sizeof(ValueBlockDirEntry),
            "BC compressed value dir entry");
    }
};

PointReader::PointReader(
    const std::filesystem::path &compressed_path,
    const BC::BCLut &lut
) {
    open(compressed_path, lut);
}

void PointReader::open(
    const std::filesystem::path &compressed_path,
    const BC::BCLut &lut
) {
    auto impl = std::make_shared<Impl>();
    impl->path = compressed_path;
    impl->lut = &lut;
    impl->header = read_pod_at<Header>(compressed_path, 0U, "BC compressed header");
    validate_header(impl->header);

    std::vector<uint8_t> axis_bytes = read_range(
        compressed_path,
        impl->header.axis_offset,
        impl->header.axis_bytes,
        "BC compressed axis");
    if ((axis_bytes.size() % sizeof(uint32_t)) != 0U ||
        axis_bytes.size() / sizeof(uint32_t) != impl->header.family_count) {
        throw std::runtime_error("BC compressed axis byte size mismatch");
    }
    std::vector<BC::FamilyCoord> coords;
    coords.reserve(static_cast<size_t>(impl->header.family_count));
    for (uint32_t i = 0U; i < impl->header.family_count; ++i) {
        coords.push_back(static_cast<BC::FamilyCoord>(
            BC::bc_load_u32_le(axis_bytes.data() + static_cast<size_t>(i) * sizeof(uint32_t))));
    }
    impl->axis = BC::BCFamilyTable(
        static_cast<BC::LayerSum>(impl->header.layer_sum),
        static_cast<uint16_t>(impl->header.family_unit),
        std::move(coords));

    if (impl->header.cell_dir_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC compressed cell dir too large");
    }
    std::vector<uint8_t> cell_bytes = read_range(
        compressed_path,
        impl->header.cell_dir_offset,
        impl->header.cell_dir_count * sizeof(CellDirEntry),
        "BC compressed cell dir");
    impl->cell_dirs.resize(static_cast<size_t>(impl->header.cell_dir_count));
    if (!impl->cell_dirs.empty()) {
        std::memcpy(
            impl->cell_dirs.data(),
            cell_bytes.data(),
            impl->cell_dirs.size() * sizeof(CellDirEntry));
    }
    impl_ = std::move(impl);
}

uint32_t PointReader::row_width() const {
    if (!impl_) {
        throw std::logic_error("BC compressed point reader is not open");
    }
    return impl_->header.row_width;
}

BC::BCSuccessDTypeMode PointReader::dtype_mode() const {
    if (!impl_) {
        throw std::logic_error("BC compressed point reader is not open");
    }
    return BC::bc_success_dtype_from_u32(impl_->header.dtype);
}

uint64_t PointReader::layer_sum() const {
    if (!impl_) {
        throw std::logic_error("BC compressed point reader is not open");
    }
    return impl_->header.layer_sum;
}

ColdLookupResult PointReader::lookup(uint64_t board, uint32_t lane) const {
    if (!impl_) {
        throw std::logic_error("BC compressed point reader is not open");
    }
    ColdLookupResult miss;
    miss.dtype = impl_->header.dtype;
    miss.row_width = impl_->header.row_width;
    if (lane >= impl_->header.row_width) {
        throw std::out_of_range("BC compressed lookup lane out of range");
    }

    const BC::BCBoardEncodedPosition encoded =
        encode_for_compressed_reader(*impl_->lut, impl_->axis, board);
    if (!encoded.valid || encoded.cid >= impl_->cell_dirs.size()) {
        return miss;
    }
    const CellDirEntry &cell = impl_->cell_dirs[encoded.cid];
    if (cell.success_rows == 0U || cell.bucket_block_count == 0U) {
        return miss;
    }

    uint64_t lo = cell.bucket_block_begin;
    uint64_t hi = lo + cell.bucket_block_count;
    BucketBlockDirEntry bucket_dir{};
    bool bucket_block_found = false;
    while (lo < hi) {
        const uint64_t mid = lo + (hi - lo) / 2U;
        const BucketBlockDirEntry candidate = impl_->read_bucket_dir(mid);
        if (encoded.key < candidate.first_key) {
            hi = mid;
        } else if (encoded.key > candidate.last_key) {
            lo = mid + 1U;
        } else {
            bucket_dir = candidate;
            bucket_block_found = true;
            break;
        }
    }
    if (!bucket_block_found) {
        return miss;
    }

    std::vector<uint8_t> compressed_bucket = read_range(
        impl_->path,
        bucket_dir.compressed_offset,
        bucket_dir.compressed_size,
        "BC compressed bucket block");
    std::vector<uint8_t> bucket_raw =
        decompress_xz_block_native(compressed_bucket.data(), compressed_bucket.size());
    if (bucket_raw.size() != bucket_dir.raw_size ||
        bucket_raw.size() < sizeof(BucketBlockRawHeader)) {
        throw std::runtime_error("BC compressed bucket block raw size mismatch");
    }
    const BucketBlockRawHeader raw_header = load_pod<BucketBlockRawHeader>(bucket_raw.data());
    if (raw_header.bucket_count != bucket_dir.bucket_count ||
        raw_header.local_rank_payload_offsets != bucket_dir.bucket_count + 1U) {
        throw std::runtime_error("BC compressed bucket block header mismatch");
    }
    const size_t count = raw_header.bucket_count;
    const size_t keys_off = sizeof(BucketBlockRawHeader);
    const size_t success_off = keys_off + count * sizeof(uint64_t);
    const size_t local_offsets_off = success_off + count * sizeof(uint32_t);
    const size_t payload_off = local_offsets_off + (count + 1U) * sizeof(uint32_t);
    const size_t expected = payload_off + raw_header.rank_payload_bytes;
    if (expected != bucket_raw.size()) {
        throw std::runtime_error("BC compressed bucket block layout mismatch");
    }
    const uint8_t *keys = bucket_raw.data() + keys_off;
    uint32_t local_bucket = std::numeric_limits<uint32_t>::max();
    for (uint32_t i = 0U; i < raw_header.bucket_count; ++i) {
        const uint64_t key = BC::load_u64_le(keys + static_cast<size_t>(i) * sizeof(uint64_t));
        if (key == encoded.key) {
            local_bucket = i;
            break;
        }
        if (key > encoded.key) {
            break;
        }
    }
    if (local_bucket == std::numeric_limits<uint32_t>::max()) {
        return miss;
    }
    const uint8_t *success_offsets = bucket_raw.data() + success_off;
    const uint8_t *local_offsets = bucket_raw.data() + local_offsets_off;
    const uint32_t success_row_offset =
        BC::bc_load_u32_le(success_offsets + static_cast<size_t>(local_bucket) * sizeof(uint32_t));
    const uint32_t payload_begin =
        BC::bc_load_u32_le(local_offsets + static_cast<size_t>(local_bucket) * sizeof(uint32_t));
    const uint32_t payload_end =
        BC::bc_load_u32_le(local_offsets + static_cast<size_t>(local_bucket + 1U) * sizeof(uint32_t));
    if (payload_end < payload_begin || payload_end > raw_header.rank_payload_bytes) {
        throw std::runtime_error("BC compressed bucket payload offset mismatch");
    }
    BC::BCBucketEntry lookup_bucket;
    lookup_bucket.key = encoded.key;
    lookup_bucket.rank_payload_offset = 0U;
    lookup_bucket.success_row_offset = success_row_offset;
    const BC::BCLookupResult row = BC::lookup_finalized_cell(
        *impl_->lut,
        BC::BCBucketEntryView{&lookup_bucket, 1U},
        BC::BCRankPayloadView{
            bucket_raw.data() + payload_off + payload_begin,
            payload_end - payload_begin},
        encoded.key,
        encoded.rank);
    if (!row.found) {
        return miss;
    }
    const uint64_t value_index =
        cell.value_base +
        static_cast<uint64_t>(row.local_success_row) * impl_->header.row_width +
        lane;
    if (value_index >= impl_->header.success_value_count) {
        throw std::runtime_error("BC compressed lookup value index out of range");
    }

    uint64_t vlo = 0U;
    uint64_t vhi = impl_->header.value_block_count;
    ValueBlockDirEntry value_dir{};
    bool value_block_found = false;
    while (vlo < vhi) {
        const uint64_t mid = vlo + (vhi - vlo) / 2U;
        const ValueBlockDirEntry candidate = impl_->read_value_dir(mid);
        const uint64_t end = candidate.first_value_index + candidate.value_count;
        if (value_index < candidate.first_value_index) {
            vhi = mid;
        } else if (value_index >= end) {
            vlo = mid + 1U;
        } else {
            value_dir = candidate;
            value_block_found = true;
            break;
        }
    }
    if (!value_block_found) {
        throw std::runtime_error("BC compressed value block not found");
    }
    std::vector<uint8_t> compressed_value = read_range(
        impl_->path,
        value_dir.compressed_offset,
        value_dir.compressed_size,
        "BC compressed value block");
    std::vector<uint8_t> value_raw =
        decompress_xz_block_native(compressed_value.data(), compressed_value.size());
    if (value_raw.size() != value_dir.raw_size ||
        value_dir.value_size != impl_->header.value_size) {
        throw std::runtime_error("BC compressed value block raw size mismatch");
    }
    const uint64_t local_value = value_index - value_dir.first_value_index;
    const uint64_t byte_offset = local_value * value_dir.value_size;
    if (byte_offset + value_dir.value_size > value_raw.size()) {
        throw std::runtime_error("BC compressed value offset mismatch");
    }

    ColdLookupResult result;
    result.found = true;
    result.dtype = impl_->header.dtype;
    result.row_width = impl_->header.row_width;
    result.cid = encoded.cid;
    result.local_success_row = row.local_success_row;
    result.value_index = value_index;
    result.bucket_block_raw_bytes = bucket_dir.raw_size;
    result.bucket_block_compressed_bytes = bucket_dir.compressed_size;
    result.value_block_raw_bytes = value_dir.raw_size;
    result.value_block_compressed_bytes = value_dir.compressed_size;
    result.raw_value_bits =
        raw_bits_from_value(value_raw.data() + static_cast<size_t>(byte_offset), impl_->header.dtype);
    result.numeric_value = numeric_from_raw(impl_->header.dtype, result.raw_value_bits);
    return result;
}

bool PointReader::sample(
    uint64_t &board,
    uint64_t &raw_value_bits,
    double &numeric_value,
    uint32_t lane
) const {
    if (!impl_) {
        throw std::logic_error("BC compressed point reader is not open");
    }
    if (lane >= impl_->header.row_width) {
        throw std::out_of_range("BC compressed sample lane out of range");
    }
    if (impl_->header.live_rows == 0U || impl_->header.bucket_block_count == 0U) {
        return false;
    }

    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> row_pick(0U, impl_->header.live_rows - 1U);
    constexpr uint32_t kSampleAttempts = 128U;
    for (uint32_t attempt = 0U; attempt < kSampleAttempts; ++attempt) {
        const uint64_t target_row = row_pick(rng);
        const CellDirEntry *cell = nullptr;
        BC::CellId cid = 0U;
        uint64_t row_base = 0U;
        for (size_t i = 0U; i < impl_->cell_dirs.size(); ++i) {
            const CellDirEntry &candidate = impl_->cell_dirs[i];
            if (candidate.success_rows == 0U) {
                continue;
            }
            const uint64_t candidate_row_base =
                candidate.value_base / impl_->header.row_width;
            const uint64_t candidate_row_end =
                candidate_row_base + candidate.success_rows;
            if (target_row >= candidate_row_base && target_row < candidate_row_end) {
                cell = &candidate;
                cid = static_cast<BC::CellId>(i);
                row_base = candidate_row_base;
                break;
            }
        }
        if (cell == nullptr || cell->bucket_block_count == 0U) {
            continue;
        }
        const uint32_t local_row = static_cast<uint32_t>(target_row - row_base);

        uint64_t lo = cell->bucket_block_begin;
        uint64_t hi = lo + cell->bucket_block_count;
        BucketBlockDirEntry bucket_dir{};
        uint64_t bucket_dir_index = 0U;
        bool bucket_block_found = false;
        while (lo < hi) {
            const uint64_t mid = lo + (hi - lo) / 2U;
            const BucketBlockDirEntry current = impl_->read_bucket_dir(mid);
            uint32_t next_first = cell->success_rows;
            if (mid + 1U < cell->bucket_block_begin + cell->bucket_block_count) {
                next_first = impl_->read_bucket_dir(mid + 1U).first_success_row;
            }
            if (local_row < current.first_success_row) {
                hi = mid;
            } else if (local_row >= next_first) {
                lo = mid + 1U;
            } else {
                bucket_dir = current;
                bucket_dir_index = mid;
                bucket_block_found = true;
                break;
            }
        }
        if (!bucket_block_found) {
            continue;
        }

        std::vector<uint8_t> compressed_bucket = read_range(
            impl_->path,
            bucket_dir.compressed_offset,
            bucket_dir.compressed_size,
            "BC compressed sample bucket block");
        std::vector<uint8_t> bucket_raw =
            decompress_xz_block_native(compressed_bucket.data(), compressed_bucket.size());
        if (bucket_raw.size() != bucket_dir.raw_size ||
            bucket_raw.size() < sizeof(BucketBlockRawHeader)) {
            throw std::runtime_error("BC compressed sample bucket block raw size mismatch");
        }
        const BucketBlockRawHeader raw_header = load_pod<BucketBlockRawHeader>(bucket_raw.data());
        if (raw_header.bucket_count != bucket_dir.bucket_count) {
            throw std::runtime_error("BC compressed sample bucket block count mismatch");
        }
        const size_t count = raw_header.bucket_count;
        const size_t keys_off = sizeof(BucketBlockRawHeader);
        const size_t success_off = keys_off + count * sizeof(uint64_t);
        const size_t local_offsets_off = success_off + count * sizeof(uint32_t);
        const size_t payload_off = local_offsets_off + (count + 1U) * sizeof(uint32_t);
        if (payload_off + raw_header.rank_payload_bytes != bucket_raw.size()) {
            throw std::runtime_error("BC compressed sample bucket layout mismatch");
        }
        const uint8_t *success_offsets = bucket_raw.data() + success_off;
        const uint8_t *local_offsets = bucket_raw.data() + local_offsets_off;
        uint32_t local_bucket = std::numeric_limits<uint32_t>::max();
        for (uint32_t i = 0U; i < raw_header.bucket_count; ++i) {
            const uint32_t row_begin =
                BC::bc_load_u32_le(success_offsets + static_cast<size_t>(i) * sizeof(uint32_t));
            uint32_t row_end = cell->success_rows;
            if (i + 1U < raw_header.bucket_count) {
                row_end = BC::bc_load_u32_le(
                    success_offsets + static_cast<size_t>(i + 1U) * sizeof(uint32_t));
            } else if (bucket_dir_index + 1U <
                       static_cast<uint64_t>(cell->bucket_block_begin) + cell->bucket_block_count) {
                row_end = impl_->read_bucket_dir(bucket_dir_index + 1U).first_success_row;
            }
            if (local_row >= row_begin && local_row < row_end) {
                local_bucket = i;
                break;
            }
        }
        if (local_bucket == std::numeric_limits<uint32_t>::max()) {
            continue;
        }

        const uint8_t *keys = bucket_raw.data() + keys_off;
        const uint64_t key =
            BC::load_u64_le(keys + static_cast<size_t>(local_bucket) * sizeof(uint64_t));
        const uint32_t row_begin =
            BC::bc_load_u32_le(success_offsets + static_cast<size_t>(local_bucket) * sizeof(uint32_t));
        const uint64_t ordinal = static_cast<uint64_t>(local_row - row_begin);
        const uint32_t payload_begin =
            BC::bc_load_u32_le(local_offsets + static_cast<size_t>(local_bucket) * sizeof(uint32_t));
        const uint32_t payload_end =
            BC::bc_load_u32_le(local_offsets + static_cast<size_t>(local_bucket + 1U) * sizeof(uint32_t));
        if (payload_end < payload_begin || payload_end > raw_header.rank_payload_bytes) {
            throw std::runtime_error("BC compressed sample payload offset mismatch");
        }
        BC::BucketRank rank = 0U;
        if (!rank_for_bucket_payload_ordinal(
                *impl_->lut,
                key,
                bucket_raw.data() + payload_off + payload_begin,
                payload_end - payload_begin,
                ordinal,
                rank)) {
            continue;
        }
        const BC::BCBucketBoardDecoder decoder(*impl_->lut, key);
        board = decoder.board(rank);
        const ColdLookupResult lookup = this->lookup(board, lane);
        if (!lookup.found || lookup.cid != cid || lookup.local_success_row != local_row) {
            continue;
        }
        raw_value_bits = lookup.raw_value_bits;
        numeric_value = lookup.numeric_value;
        return true;
    }
    return false;
}

ColdLookupResult lookup_cold(
    const std::filesystem::path &compressed_path,
    const BC::BCLut &lut,
    uint64_t board,
    uint32_t lane
) {
    PointReader reader(compressed_path, lut);
    return reader.lookup(board, lane);
}

bool sample_cold(
    const std::filesystem::path &compressed_path,
    const BC::BCLut &lut,
    uint64_t &board,
    uint64_t &raw_value_bits,
    double &numeric_value,
    uint32_t lane
) {
    try {
        PointReader reader(compressed_path, lut);
        return reader.sample(board, raw_value_bits, numeric_value, lane);
    } catch (...) {
        return false;
    }
}

} // namespace BCCompressedResult
