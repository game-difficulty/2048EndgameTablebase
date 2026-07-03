#pragma once

#include <filesystem>
#include <string>
#include <system_error>

namespace NativePath {

inline std::filesystem::path from_utf8(const std::string &path) {
#if defined(_WIN32)
    return std::filesystem::u8path(path);
#else
    return std::filesystem::path(path);
#endif
}

inline std::filesystem::path from_utf8(const char *path) {
    return from_utf8(std::string(path == nullptr ? "" : path));
}

inline std::string to_utf8_string(const std::filesystem::path &path) {
#if defined(_WIN32)
    const auto u8 = path.u8string();
    return std::string(reinterpret_cast<const char *>(u8.data()), u8.size());
#else
    return path.string();
#endif
}

inline bool exists(const std::string &path) {
    return std::filesystem::exists(from_utf8(path));
}

inline bool exists(const std::string &path, std::error_code &ec) {
    return std::filesystem::exists(from_utf8(path), ec);
}

inline bool is_regular_file(const std::string &path, std::error_code &ec) {
    return std::filesystem::is_regular_file(from_utf8(path), ec);
}

inline bool is_directory(const std::string &path, std::error_code &ec) {
    return std::filesystem::is_directory(from_utf8(path), ec);
}

inline uintmax_t file_size(const std::string &path) {
    return std::filesystem::file_size(from_utf8(path));
}

inline uintmax_t file_size(const std::string &path, std::error_code &ec) {
    return std::filesystem::file_size(from_utf8(path), ec);
}

inline bool remove(const std::string &path) {
    return std::filesystem::remove(from_utf8(path));
}

inline bool remove(const std::string &path, std::error_code &ec) {
    return std::filesystem::remove(from_utf8(path), ec);
}

inline uintmax_t remove_all(const std::string &path, std::error_code &ec) {
    return std::filesystem::remove_all(from_utf8(path), ec);
}

inline void rename(const std::string &from, const std::string &to, std::error_code &ec) {
    std::filesystem::rename(from_utf8(from), from_utf8(to), ec);
}

inline bool create_directories(const std::string &path, std::error_code &ec) {
    return std::filesystem::create_directories(from_utf8(path), ec);
}

inline std::string replace_extension_utf8(const std::string &path, const std::string &replacement = std::string()) {
    auto native = from_utf8(path);
    native.replace_extension(replacement);
    return to_utf8_string(native);
}

} // namespace NativePath
