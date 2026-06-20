#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace BC {

enum class BCFileIOMode {
    Buffered,
    Direct,
};

struct BCFileExtent {
    uint64_t offset = 0U;
    uint64_t bytes = 0U;
};

struct BCFileReadRequest {
    uint64_t offset = 0U;
    void *data = nullptr;
    uint64_t bytes = 0U;
};

struct BCFileWriteRequest {
    uint64_t offset = 0U;
    const void *data = nullptr;
    uint64_t bytes = 0U;
};

struct BCFileIOStats {
    uint64_t request_count = 0U;
    uint64_t requested_bytes = 0U;
    uint64_t backend_io_count = 0U;
    uint64_t backend_bytes = 0U;
    double backend_seconds = 0.0;
};

inline void bc_fileio_accumulate_request(
    BCFileIOStats *stats,
    uint64_t bytes
) {
    if (stats == nullptr) {
        return;
    }
    if (stats->request_count == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("BC file IO request count overflow");
    }
    ++stats->request_count;
    if (stats->backend_io_count == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("BC file IO backend IO count overflow");
    }
    ++stats->backend_io_count;
    if (stats->requested_bytes > std::numeric_limits<uint64_t>::max() - bytes ||
        stats->backend_bytes > std::numeric_limits<uint64_t>::max() - bytes) {
        throw std::overflow_error("BC file IO byte stats overflow");
    }
    stats->requested_bytes += bytes;
    stats->backend_bytes += bytes;
}

class BCWritableFile {
public:
    virtual ~BCWritableFile() = default;
    [[nodiscard]] virtual BCFileIOMode mode() const {
        return BCFileIOMode::Buffered;
    }
    [[nodiscard]] virtual uint32_t preferred_write_alignment() const {
        return 1U;
    }
    virtual void write_at(uint64_t offset, const void *data, uint64_t bytes) = 0;
    virtual void write_many(
        const std::vector<BCFileWriteRequest> &requests,
        BCFileIOStats *stats = nullptr
    ) {
        if (stats != nullptr) {
            *stats = {};
        }
        for (const BCFileWriteRequest &request : requests) {
            write_at(request.offset, request.data, request.bytes);
            bc_fileio_accumulate_request(stats, request.bytes);
        }
    }
    virtual void resize(uint64_t bytes) = 0;
    // Prepare a newly created file for a planned full overwrite. Backends that
    // require explicit logical/physical sizing can delegate to resize(); plain
    // buffered streams opened with truncation can avoid a close/reopen resize
    // because the subsequent sequential writes extend the file to logical size.
    virtual void prepare_full_overwrite(uint64_t bytes) {
        resize(bytes);
    }
    virtual void flush() = 0;
};

// Implementations of BCReadableFile are not guaranteed to be thread-safe unless
// a concrete backend explicitly documents that guarantee. Current buffered and
// direct backends should be used from a single IO stage, or opened once per
// worker thread if concurrent reads are required.
class BCReadableFile {
public:
    virtual ~BCReadableFile() = default;
    [[nodiscard]] virtual BCFileIOMode mode() const {
        return BCFileIOMode::Buffered;
    }
    [[nodiscard]] virtual uint32_t preferred_read_alignment() const {
        return 1U;
    }
    virtual void read_at(uint64_t offset, void *data, uint64_t bytes) const = 0;
    virtual void read_many(
        const std::vector<BCFileReadRequest> &requests,
        BCFileIOStats *stats = nullptr
    ) const {
        if (stats != nullptr) {
            *stats = {};
        }
        for (const BCFileReadRequest &request : requests) {
            read_at(request.offset, request.data, request.bytes);
            bc_fileio_accumulate_request(stats, request.bytes);
        }
    }
    virtual bool try_read_physical_prefix_aligned(
        void *data,
        uint64_t bytes,
        BCFileIOStats *stats = nullptr
    ) const {
        (void)data;
        (void)bytes;
        if (stats != nullptr) {
            *stats = {};
        }
        return false;
    }
    virtual uint64_t size() const = 0;
};

class BCMemoryReadableFile final : public BCReadableFile {
public:
    explicit BCMemoryReadableFile(std::vector<uint8_t> bytes)
        : bytes_(std::move(bytes)) {}

    void read_at(uint64_t offset, void *data, uint64_t bytes) const override {
        if (bytes == 0U) {
            return;
        }
        if (data == nullptr) {
            throw std::invalid_argument("BC memory file reader data pointer is null");
        }
        if (offset > bytes_.size() || bytes > static_cast<uint64_t>(bytes_.size()) - offset) {
            throw std::out_of_range("BC memory file reader read exceeds file size");
        }
        std::memcpy(
            data,
            bytes_.data() + static_cast<size_t>(offset),
            static_cast<size_t>(bytes));
    }

    [[nodiscard]] uint64_t size() const override {
        return static_cast<uint64_t>(bytes_.size());
    }

private:
    std::vector<uint8_t> bytes_;
};

// Buffered correctness backend. It intentionally uses ordinary OS-buffered file
// IO. A later direct backend should implement the same interface with Windows
// FILE_FLAG_NO_BUFFERING, FILE_FLAG_OVERLAPPED, aligned staging buffers,
// aligned offsets/sizes, high queue depth, and coalesced positioned IO.
class BCBufferedFileWriter final : public BCWritableFile {
public:
    explicit BCBufferedFileWriter(const std::filesystem::path &path)
        : path_(path),
          file_(path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc) {
        if (!file_) {
            throw std::runtime_error("BC buffered file writer failed to open: " + path_.string());
        }
    }

    void write_at(uint64_t offset, const void *data, uint64_t bytes) override {
        if (bytes == 0U) {
            return;
        }
        if (data == nullptr) {
            throw std::invalid_argument("BC buffered file writer data pointer is null");
        }
        file_.clear();
        file_.seekp(checked_stream_offset(offset, "BC buffered write offset exceeds stream range"));
        if (!file_) {
            throw std::runtime_error("BC buffered file writer seekp failed: " + path_.string());
        }
        static_cast<void>(write_contiguous(data, bytes));
    }

    void write_many(
        const std::vector<BCFileWriteRequest> &requests,
        BCFileIOStats *stats = nullptr
    ) override {
        if (stats != nullptr) {
            *stats = {};
        }
        bool positioned = false;
        uint64_t cursor = 0U;
        for (const BCFileWriteRequest &request : requests) {
            if (request.bytes == 0U) {
                continue;
            }
            if (request.data == nullptr) {
                throw std::invalid_argument("BC buffered file writer data pointer is null");
            }
            if (!positioned || request.offset != cursor) {
                file_.clear();
                file_.seekp(checked_stream_offset(request.offset, "BC buffered write offset exceeds stream range"));
                if (!file_) {
                    throw std::runtime_error("BC buffered file writer seekp failed: " + path_.string());
                }
                positioned = true;
            }
            const uint64_t backend_chunks = write_contiguous(request.data, request.bytes);
            cursor = request.offset + request.bytes;
            if (stats != nullptr) {
                if (stats->request_count == std::numeric_limits<uint64_t>::max() ||
                    stats->requested_bytes > std::numeric_limits<uint64_t>::max() - request.bytes ||
                    stats->backend_io_count > std::numeric_limits<uint64_t>::max() - backend_chunks ||
                    stats->backend_bytes > std::numeric_limits<uint64_t>::max() - request.bytes) {
                    throw std::overflow_error("BC buffered file writer stats overflow");
                }
                ++stats->request_count;
                stats->requested_bytes += request.bytes;
                stats->backend_io_count += backend_chunks;
                stats->backend_bytes += request.bytes;
            }
        }
    }

    void resize(uint64_t bytes) override {
        file_.flush();
        if (!file_) {
            throw std::runtime_error("BC buffered file writer flush before resize failed: " + path_.string());
        }
        file_.close();
        std::error_code ec;
        std::filesystem::resize_file(path_, bytes, ec);
        if (ec) {
            throw std::runtime_error("BC buffered file writer resize failed: " + ec.message());
        }
        file_.open(path_, std::ios::binary | std::ios::in | std::ios::out);
        if (!file_) {
            throw std::runtime_error("BC buffered file writer reopen after resize failed: " + path_.string());
        }
    }

    void prepare_full_overwrite(uint64_t bytes) override {
        (void)bytes;
        file_.clear();
        file_.seekp(0);
        if (!file_) {
            throw std::runtime_error("BC buffered file writer seekp before full overwrite failed: " + path_.string());
        }
    }

    void flush() override {
        file_.flush();
        if (!file_) {
            throw std::runtime_error("BC buffered file writer flush failed: " + path_.string());
        }
    }

private:
    [[nodiscard]] uint64_t write_contiguous(const void *data, uint64_t bytes) {
        const uint8_t *cursor = static_cast<const uint8_t *>(data);
        uint64_t remaining = bytes;
        uint64_t chunks = 0U;
        while (remaining != 0U) {
            const uint64_t take_u64 =
                remaining > kMaxBufferedWriteChunkBytes
                    ? kMaxBufferedWriteChunkBytes
                    : remaining;
            const std::streamsize take = checked_stream_size(
                take_u64,
                "BC buffered write size exceeds stream range"
            );
            file_.write(reinterpret_cast<const char *>(cursor), take);
            if (!file_) {
                throw std::runtime_error("BC buffered file writer write failed: " + path_.string());
            }
            cursor += take_u64;
            remaining -= take_u64;
            ++chunks;
        }
        return chunks;
    }

    [[nodiscard]] static std::streamoff checked_stream_offset(uint64_t value, const char *label) {
        if (value > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
            throw std::overflow_error(label);
        }
        return static_cast<std::streamoff>(value);
    }

    [[nodiscard]] static std::streamsize checked_stream_size(uint64_t value, const char *label) {
        if (value > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::overflow_error(label);
        }
        return static_cast<std::streamsize>(value);
    }

    static constexpr uint64_t kMaxBufferedWriteChunkBytes = 64ULL * 1024ULL * 1024ULL;

    std::filesystem::path path_;
    std::fstream file_;
};

class BCBufferedFileReader final : public BCReadableFile {
public:
    explicit BCBufferedFileReader(const std::filesystem::path &path)
        : path_(path),
          file_(path, std::ios::binary | std::ios::in) {
        if (!file_) {
            throw std::runtime_error("BC buffered file reader failed to open: " + path_.string());
        }
        std::error_code ec;
        size_ = std::filesystem::file_size(path_, ec);
        if (ec) {
            throw std::runtime_error("BC buffered file reader file_size failed: " + ec.message());
        }
    }

    void read_at(uint64_t offset, void *data, uint64_t bytes) const override {
        if (bytes == 0U) {
            return;
        }
        if (data == nullptr) {
            throw std::invalid_argument("BC buffered file reader data pointer is null");
        }
        refresh_size();
        if (offset > size_ || bytes > size_ - offset) {
            throw std::out_of_range("BC buffered file reader read exceeds file size");
        }
        file_.clear();
        file_.seekg(checked_stream_offset(offset, "BC buffered read offset exceeds stream range"));
        if (!file_) {
            throw std::runtime_error("BC buffered file reader seekg failed: " + path_.string());
        }
        read_contiguous(data, bytes);
    }

    void read_at_cached_size(uint64_t offset, void *data, uint64_t bytes) const {
        if (bytes == 0U) {
            return;
        }
        if (data == nullptr) {
            throw std::invalid_argument("BC buffered file reader data pointer is null");
        }
        if (offset > size_ || bytes > size_ - offset) {
            throw std::out_of_range("BC buffered file reader read exceeds cached file size");
        }
        file_.clear();
        file_.seekg(checked_stream_offset(offset, "BC buffered read offset exceeds stream range"));
        if (!file_) {
            throw std::runtime_error("BC buffered file reader seekg failed: " + path_.string());
        }
        read_contiguous(data, bytes);
    }

    void read_many(
        const std::vector<BCFileReadRequest> &requests,
        BCFileIOStats *stats = nullptr
    ) const override {
        if (stats != nullptr) {
            *stats = {};
        }
        refresh_size();
        bool positioned = false;
        uint64_t cursor = 0U;
        for (const BCFileReadRequest &request : requests) {
            if (request.bytes == 0U) {
                continue;
            }
            if (request.data == nullptr) {
                throw std::invalid_argument("BC buffered file reader data pointer is null");
            }
            if (request.offset > size_ || request.bytes > size_ - request.offset) {
                throw std::out_of_range("BC buffered file reader read exceeds file size");
            }
            if (!positioned || request.offset != cursor) {
                file_.clear();
                file_.seekg(checked_stream_offset(request.offset, "BC buffered read offset exceeds stream range"));
                if (!file_) {
                    throw std::runtime_error("BC buffered file reader seekg failed: " + path_.string());
                }
                positioned = true;
            }
            read_contiguous(request.data, request.bytes);
            cursor = request.offset + request.bytes;
            bc_fileio_accumulate_request(stats, request.bytes);
        }
    }

    [[nodiscard]] uint64_t size() const override {
        refresh_size();
        return size_;
    }

    [[nodiscard]] uint64_t cached_size() const {
        return size_;
    }

private:
    void read_contiguous(void *data, uint64_t bytes) const {
        uint8_t *cursor = static_cast<uint8_t *>(data);
        uint64_t remaining = bytes;
        while (remaining != 0U) {
            const uint64_t take_u64 =
                remaining > kMaxBufferedReadChunkBytes
                    ? kMaxBufferedReadChunkBytes
                    : remaining;
            const std::streamsize take = checked_stream_size(
                take_u64,
                "BC buffered read size exceeds stream range"
            );
            file_.read(reinterpret_cast<char *>(cursor), take);
            if (!file_ || static_cast<uint64_t>(file_.gcount()) != take_u64) {
                throw std::runtime_error("BC buffered file reader short read: " + path_.string());
            }
            cursor += take_u64;
            remaining -= take_u64;
        }
    }

    void refresh_size() const {
        std::error_code ec;
        const uint64_t current = std::filesystem::file_size(path_, ec);
        if (ec) {
            throw std::runtime_error("BC buffered file reader file_size failed: " + ec.message());
        }
        size_ = current;
    }

    [[nodiscard]] static std::streamoff checked_stream_offset(uint64_t value, const char *label) {
        if (value > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
            throw std::overflow_error(label);
        }
        return static_cast<std::streamoff>(value);
    }

    [[nodiscard]] static std::streamsize checked_stream_size(uint64_t value, const char *label) {
        if (value > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::overflow_error(label);
        }
        return static_cast<std::streamsize>(value);
    }

    static constexpr uint64_t kMaxBufferedReadChunkBytes = 64ULL * 1024ULL * 1024ULL;

    std::filesystem::path path_;
    mutable std::ifstream file_;
    mutable uint64_t size_ = 0U;
};

inline void write_bytes_to_buffered_file(
    const std::filesystem::path &path,
    const std::vector<uint8_t> &bytes
) {
    BCBufferedFileWriter writer(path);
    writer.resize(static_cast<uint64_t>(bytes.size()));
    if (!bytes.empty()) {
        writer.write_at(0U, bytes.data(), static_cast<uint64_t>(bytes.size()));
    }
    writer.flush();
}

[[nodiscard]] inline std::vector<uint8_t> read_bytes_from_buffered_file(
    const std::filesystem::path &path
) {
    BCBufferedFileReader reader(path);
    const uint64_t byte_count = reader.size();
    if (byte_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("BC buffered file exceeds addressable memory vector size");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(byte_count));
    if (!bytes.empty()) {
        reader.read_at(0U, bytes.data(), byte_count);
    }
    return bytes;
}

} // namespace BC
