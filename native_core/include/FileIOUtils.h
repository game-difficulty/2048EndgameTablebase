#pragma once

#include "FormationRuntime.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace FileIOUtils {

inline constexpr size_t kChunkBytes = 64ULL * 1024ULL * 1024ULL;
inline constexpr uint64_t kDirectIoAlignment = 4096ULL;

struct DirectIoConfig {
    bool enabled = false;
    uint32_t queue_depth = 16U;
    uint32_t chunk_mib = 8U;
};

template <typename PathLike>
std::string path_string(const PathLike &path_like) {
    return std::filesystem::path(path_like).string();
}

inline uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
    return alignment == 0ULL ? value : ((value + alignment - 1ULL) / alignment) * alignment;
}

inline uint64_t align_down_u64(uint64_t value, uint64_t alignment) {
    return alignment == 0ULL ? value : (value / alignment) * alignment;
}

inline size_t checked_tellg(std::ifstream &file, const std::string &path) {
    const std::ifstream::pos_type end_pos = file.tellg();
    if (end_pos < 0) {
        throw std::runtime_error("failed to determine file size: " + path);
    }
    return static_cast<size_t>(end_pos);
}

template <typename Stream>
void read_exact(Stream &stream, void *dst, size_t bytes, const std::string &context) {
    auto *out = static_cast<char *>(dst);
    size_t offset = 0U;
    while (offset < bytes) {
        const size_t chunk = std::min(kChunkBytes, bytes - offset);
        stream.read(out + offset, static_cast<std::streamsize>(chunk));
        if (!stream) {
            throw std::runtime_error("failed to read: " + context);
        }
        offset += chunk;
    }
}

template <typename Stream>
void write_exact(Stream &stream, const void *src, size_t bytes, const std::string &context) {
    const auto *in = static_cast<const char *>(src);
    size_t offset = 0U;
    while (offset < bytes) {
        const size_t chunk = std::min(kChunkBytes, bytes - offset);
        stream.write(in + offset, static_cast<std::streamsize>(chunk));
        if (!stream) {
            throw std::runtime_error("failed to write: " + context);
        }
        offset += chunk;
    }
}

DirectIoConfig normalize_direct_io_config(DirectIoConfig config);
DirectIoConfig direct_io_config_from_options(const RunOptions &options);
std::string temp_write_path(const std::string &final_path);
void finalize_temporary_file(const std::string &temp_path, const std::string &final_path);

class DirectAppendWriter {
public:
    DirectAppendWriter();
    DirectAppendWriter(const std::string &path, uint64_t logical_bytes, DirectIoConfig config = {});
    ~DirectAppendWriter();

    DirectAppendWriter(DirectAppendWriter &&other) noexcept;
    DirectAppendWriter &operator=(DirectAppendWriter &&other) noexcept;

    DirectAppendWriter(const DirectAppendWriter &) = delete;
    DirectAppendWriter &operator=(const DirectAppendWriter &) = delete;

    void append(const void *src, size_t bytes);
    void close();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class DirectSequentialReader {
public:
    DirectSequentialReader();
    DirectSequentialReader(const std::string &path, uint64_t logical_bytes, DirectIoConfig config = {});
    ~DirectSequentialReader();

    DirectSequentialReader(DirectSequentialReader &&other) noexcept;
    DirectSequentialReader &operator=(DirectSequentialReader &&other) noexcept;

    DirectSequentialReader(const DirectSequentialReader &) = delete;
    DirectSequentialReader &operator=(const DirectSequentialReader &) = delete;

    void read(void *dst, size_t bytes);
    void close();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

template <typename PathLike>
std::vector<uint8_t> read_binary_bytes(const PathLike &path_like) {
    const std::string path = path_string(path_like);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }
    const size_t size = checked_tellg(file, path);
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(size);
    if (!data.empty()) {
        read_exact(file, data.data(), data.size(), path);
    }
    return data;
}

template <typename PathLike>
std::vector<uint8_t> read_binary_bytes_range(
    const PathLike &path_like,
    uint64_t begin,
    uint64_t end
) {
    const std::string path = path_string(path_like);
    std::ifstream file(path, std::ios::binary);
    if (!file || end < begin) {
        return {};
    }
    const size_t size = static_cast<size_t>(end - begin);
    std::vector<uint8_t> data(size);
    file.seekg(static_cast<std::streamoff>(begin), std::ios::beg);
    if (!data.empty()) {
        read_exact(file, data.data(), data.size(), path);
    }
    return data;
}

template <typename PathLike>
void write_binary_bytes(const PathLike &path_like, const std::vector<uint8_t> &data) {
    const std::string path = path_string(path_like);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open for write: " + path);
    }
    if (!data.empty()) {
        write_exact(out, data.data(), data.size(), path);
    }
}

template <typename T, typename PathLike>
std::vector<T> read_binary_vector(const PathLike &path_like) {
    static_assert(std::is_trivially_copyable_v<T>, "binary vector I/O requires trivially copyable types");
    const std::string path = path_string(path_like);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }
    const size_t size = checked_tellg(file, path);
    if (size % sizeof(T) != 0U) {
        throw std::runtime_error("misaligned binary file size: " + path);
    }
    file.seekg(0, std::ios::beg);
    std::vector<T> data(size / sizeof(T));
    if (!data.empty()) {
        read_exact(file, data.data(), size, path);
    }
    return data;
}

template <typename T, typename PathLike>
std::vector<T> read_binary_vector_direct(
    const PathLike &path_like,
    DirectIoConfig config = {}
) {
    static_assert(std::is_trivially_copyable_v<T>, "binary vector I/O requires trivially copyable types");
    const DirectIoConfig normalized = normalize_direct_io_config(config);
    if (!normalized.enabled) {
        return read_binary_vector<T>(path_like);
    }
    const std::string path = path_string(path_like);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }
    const size_t size = checked_tellg(file, path);
    if (size % sizeof(T) != 0U) {
        throw std::runtime_error("misaligned binary file size: " + path);
    }
    std::vector<T> data(size / sizeof(T));
    if (!data.empty()) {
        DirectSequentialReader reader(path, static_cast<uint64_t>(size), normalized);
        reader.read(data.data(), size);
        reader.close();
    }
    return data;
}

template <typename T, typename PathLike>
void write_binary_span(const PathLike &path_like, const T *data, size_t count) {
    static_assert(std::is_trivially_copyable_v<T>, "binary span I/O requires trivially copyable types");
    const std::string path = path_string(path_like);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open for write: " + path);
    }
    if (count != 0U) {
        write_exact(out, data, count * sizeof(T), path);
    }
}

template <typename T, typename PathLike>
void write_binary_vector(const PathLike &path_like, const std::vector<T> &data) {
    write_binary_span(path_like, data.data(), data.size());
}

template <typename PathLike>
void write_binary_bytes_direct(
    const PathLike &path_like,
    const std::vector<uint8_t> &data,
    DirectIoConfig config = {}
) {
    if (!normalize_direct_io_config(config).enabled) {
        write_binary_bytes(path_like, data);
        return;
    }
    DirectAppendWriter out(path_string(path_like), static_cast<uint64_t>(data.size()), config);
    if (!data.empty()) {
        out.append(data.data(), data.size());
    }
    out.close();
}

template <typename T, typename PathLike>
void write_binary_span_direct(
    const PathLike &path_like,
    const T *data,
    size_t count,
    DirectIoConfig config = {}
) {
    static_assert(std::is_trivially_copyable_v<T>, "binary span I/O requires trivially copyable types");
    if (!normalize_direct_io_config(config).enabled) {
        write_binary_span(path_like, data, count);
        return;
    }
    DirectAppendWriter out(
        path_string(path_like),
        static_cast<uint64_t>(count) * static_cast<uint64_t>(sizeof(T)),
        config
    );
    if (count != 0U) {
        out.append(data, count * sizeof(T));
    }
    out.close();
}

template <typename T, typename PathLike>
void write_binary_vector_direct(
    const PathLike &path_like,
    const std::vector<T> &data,
    DirectIoConfig config = {}
) {
    write_binary_span_direct(path_like, data.data(), data.size(), config);
}

} // namespace FileIOUtils
