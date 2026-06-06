#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace BC {

enum class BCFileIOMode {
    Buffered,
    Direct,
};

class BCWritableFile {
public:
    virtual ~BCWritableFile() = default;
    virtual void write_at(uint64_t offset, const void *data, uint64_t bytes) = 0;
    virtual void resize(uint64_t bytes) = 0;
    virtual void flush() = 0;
};

class BCReadableFile {
public:
    virtual ~BCReadableFile() = default;
    virtual void read_at(uint64_t offset, void *data, uint64_t bytes) const = 0;
    virtual uint64_t size() const = 0;
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
        file_.write(
            static_cast<const char *>(data),
            checked_stream_size(bytes, "BC buffered write size exceeds stream range")
        );
        if (!file_) {
            throw std::runtime_error("BC buffered file writer write failed: " + path_.string());
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

    void flush() override {
        file_.flush();
        if (!file_) {
            throw std::runtime_error("BC buffered file writer flush failed: " + path_.string());
        }
    }

private:
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
        if (offset > size_ || bytes > size_ - offset) {
            throw std::out_of_range("BC buffered file reader read exceeds file size");
        }
        file_.clear();
        file_.seekg(checked_stream_offset(offset, "BC buffered read offset exceeds stream range"));
        if (!file_) {
            throw std::runtime_error("BC buffered file reader seekg failed: " + path_.string());
        }
        file_.read(
            static_cast<char *>(data),
            checked_stream_size(bytes, "BC buffered read size exceeds stream range")
        );
        if (!file_ || static_cast<uint64_t>(file_.gcount()) != bytes) {
            throw std::runtime_error("BC buffered file reader short read: " + path_.string());
        }
    }

    [[nodiscard]] uint64_t size() const override {
        return size_;
    }

private:
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

    std::filesystem::path path_;
    mutable std::ifstream file_;
    uint64_t size_ = 0U;
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
