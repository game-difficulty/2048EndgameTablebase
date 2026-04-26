#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace FileIOUtils {

inline constexpr size_t kChunkBytes = 64ULL * 1024ULL * 1024ULL;

template <typename PathLike>
std::string path_string(const PathLike &path_like) {
    return std::filesystem::path(path_like).string();
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

} // namespace FileIOUtils
