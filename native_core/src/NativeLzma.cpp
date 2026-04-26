#include "NativeLzma.h"

#include "FileIOUtils.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <lzma.h>
#include <omp.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr size_t kBlockSize = 32768ULL;
constexpr size_t kParallelThreshold = 4194304ULL;

#pragma pack(push, 1)
struct U64SegmentEntry {
    uint64_t first_value;
    uint64_t file_offset;
};
#pragma pack(pop)

static_assert(sizeof(U64SegmentEntry) == 16, "Unexpected uint64 segment entry layout");

template <typename T>
std::vector<T> read_binary_vector(const std::string &path) {
    return FileIOUtils::read_binary_vector<T>(path);
}

template <typename T>
void write_binary_vector(const std::string &path, const std::vector<T> &data) {
    FileIOUtils::write_binary_vector(path, data);
}

struct LzmaApi {
#ifdef _WIN32
    HMODULE module = nullptr;
#else
    void *module = nullptr;
#endif
    size_t (*stream_buffer_bound)(size_t) = nullptr;
    lzma_ret (*easy_buffer_encode)(
        uint32_t,
        lzma_check,
        const lzma_allocator *,
        const uint8_t *,
        size_t,
        uint8_t *,
        size_t *,
        size_t) = nullptr;
    lzma_ret (*stream_buffer_decode)(
        uint64_t *,
        uint32_t,
        const lzma_allocator *,
        const uint8_t *,
        size_t *,
        size_t,
        uint8_t *,
        size_t *,
        size_t) = nullptr;
    bool loaded = false;
};

LzmaApi &lzma_api() {
    static LzmaApi api;
    static std::once_flag once;
    std::call_once(once, []() {
        auto load_symbol = [&](const char *name) -> void * {
#ifdef _WIN32
            return api.module ? reinterpret_cast<void *>(GetProcAddress(api.module, name)) : nullptr;
#else
            return api.module ? dlsym(api.module, name) : nullptr;
#endif
        };

#ifdef _WIN32
        std::vector<std::string> candidates = {"liblzma.dll", "liblzma-5.dll"};
        char exe_path[MAX_PATH] = {};
        if (GetModuleFileNameA(nullptr, exe_path, MAX_PATH) > 0) {
            fs::path root = fs::path(exe_path).parent_path();
            candidates.push_back((root / "Library" / "bin" / "liblzma.dll").string());
            candidates.push_back((root / "Library" / "bin" / "liblzma-5.dll").string());
        }
        for (const auto &candidate : candidates) {
            api.module = LoadLibraryA(candidate.c_str());
            if (api.module) {
                break;
            }
        }
#else
        for (const char *candidate : {"liblzma.so", "liblzma.so.5"}) {
            api.module = dlopen(candidate, RTLD_LAZY);
            if (api.module) {
                break;
            }
        }
#endif
        if (!api.module) {
            return;
        }

        api.stream_buffer_bound =
            reinterpret_cast<size_t (*)(size_t)>(load_symbol("lzma_stream_buffer_bound"));
        api.easy_buffer_encode =
            reinterpret_cast<lzma_ret (*)(uint32_t, lzma_check, const lzma_allocator *, const uint8_t *, size_t, uint8_t *, size_t *, size_t)>(
                load_symbol("lzma_easy_buffer_encode"));
        api.stream_buffer_decode =
            reinterpret_cast<lzma_ret (*)(uint64_t *, uint32_t, const lzma_allocator *, const uint8_t *, size_t *, size_t, uint8_t *, size_t *, size_t)>(
                load_symbol("lzma_stream_buffer_decode"));
        api.loaded = api.stream_buffer_bound && api.easy_buffer_encode && api.stream_buffer_decode;
    });
    return api;
}

bool is_lzma_available() {
    return lzma_api().loaded;
}

std::vector<uint8_t> xz_compress_bytes(const uint8_t *data, size_t size, uint32_t preset) {
    const auto &api = lzma_api();
    if (!api.loaded) {
        return {};
    }
    size_t out_capacity = api.stream_buffer_bound(size);
    std::vector<uint8_t> output(out_capacity);
    size_t out_pos = 0;
    const lzma_ret ret = api.easy_buffer_encode(
        preset,
        LZMA_CHECK_CRC64,
        nullptr,
        data,
        size,
        output.data(),
        &out_pos,
        output.size());
    if (ret != LZMA_OK) {
        return {};
    }
    output.resize(out_pos);
    return output;
}

std::vector<uint8_t> xz_decompress_bytes(const uint8_t *data, size_t size) {
    const auto &api = lzma_api();
    if (!api.loaded) {
        return {};
    }
    uint64_t memlimit = std::numeric_limits<uint64_t>::max();
    size_t out_capacity = std::max<size_t>(size * 8, 4096);
    for (int attempt = 0; attempt < 8; ++attempt) {
        std::vector<uint8_t> output(out_capacity);
        size_t in_pos = 0;
        size_t out_pos = 0;
        const lzma_ret ret = api.stream_buffer_decode(
            &memlimit,
            0,
            nullptr,
            data,
            &in_pos,
            size,
            output.data(),
            &out_pos,
            output.size());
        if (ret == LZMA_OK) {
            output.resize(out_pos);
            return output;
        }
        if (ret != LZMA_BUF_ERROR) {
            return {};
        }
        out_capacity *= 2;
    }
    return {};
}

std::optional<std::string> resolve_7z_executable() {
    std::vector<fs::path> candidates = {
        fs::path("7z.exe"),
        fs::path("_internal") / "7z.exe",
        fs::path("7zip") / "7z.exe",
    };

#ifdef _WIN32
    char exe_path[MAX_PATH] = {};
    if (GetModuleFileNameA(nullptr, exe_path, MAX_PATH) > 0) {
        fs::path root = fs::path(exe_path).parent_path();
        candidates.push_back(root / "7z.exe");
        candidates.push_back(root / "_internal" / "7z.exe");
        candidates.push_back(root / "7zip" / "7z.exe");
    }
#endif

    for (const auto &candidate : candidates) {
        if (fs::exists(candidate)) {
            return candidate.string();
        }
    }
    const char *path = std::getenv("PATH");
    if (path && std::strlen(path) > 0) {
        return std::string("7z");
    }
    return std::nullopt;
}

bool run_command(const std::string &command_line) {
#ifdef _WIN32
    STARTUPINFOA startup_info{};
    PROCESS_INFORMATION process_info{};
    startup_info.cb = sizeof(startup_info);
    startup_info.dwFlags |= STARTF_USESHOWWINDOW;
    startup_info.wShowWindow = SW_HIDE;
    std::vector<char> command(command_line.begin(), command_line.end());
    command.push_back('\0');
    const BOOL ok = CreateProcessA(
        nullptr,
        command.data(),
        nullptr,
        nullptr,
        FALSE,
        CREATE_NO_WINDOW,
        nullptr,
        nullptr,
        &startup_info,
        &process_info);
    if (!ok) {
        return false;
    }
    WaitForSingleObject(process_info.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(process_info.hProcess, &exit_code);
    CloseHandle(process_info.hThread);
    CloseHandle(process_info.hProcess);
    return exit_code == 0;
#else
    const int status = std::system(command_line.c_str());
    return status == 0;
#endif
}

std::string quote_arg(const std::string &value) {
    return "\"" + value + "\"";
}

bool compress_file_xz(const std::string &input_path, const std::string &output_path, int lvl) {
    std::vector<uint8_t> bytes = FileIOUtils::read_binary_bytes(input_path);
    if (bytes.empty() && !fs::exists(input_path)) {
        return false;
    }
    std::vector<uint8_t> compressed = xz_compress_bytes(bytes.data(), bytes.size(), static_cast<uint32_t>(lvl));
    if (compressed.empty() && !bytes.empty()) {
        return false;
    }
    FileIOUtils::write_binary_bytes(output_path, compressed);
    fs::remove(input_path);
    return true;
}

bool decompress_file_xz(const std::string &input_path, const std::string &output_path) {
    std::vector<uint8_t> bytes = FileIOUtils::read_binary_bytes(input_path);
    if (bytes.empty() && !fs::exists(input_path)) {
        return false;
    }
    std::vector<uint8_t> decompressed = xz_decompress_bytes(bytes.data(), bytes.size());
    if (decompressed.empty() && !bytes.empty()) {
        return false;
    }
    FileIOUtils::write_binary_bytes(output_path, decompressed);
    fs::remove(input_path);
    return true;
}

std::vector<U64SegmentEntry> read_u64_segments(const fs::path &segments_path) {
    return read_binary_vector<U64SegmentEntry>(segments_path.string());
}

std::vector<uint8_t> read_file_bytes_range(const std::string &path, uint64_t begin, uint64_t end) {
    return FileIOUtils::read_binary_bytes_range(path, begin, end);
}

} // namespace

std::vector<uint8_t> compress_xz_block_native(const uint8_t *data, size_t size, int lvl) {
    return xz_compress_bytes(data, size, static_cast<uint32_t>(lvl));
}

std::vector<uint8_t> decompress_xz_block_native(const uint8_t *data, size_t size) {
    return xz_decompress_bytes(data, size);
}

bool compress_with_7z_or_xz(const std::string &input_path, int lvl) {
    if (!fs::exists(input_path)) {
        return false;
    }
    const std::string output_path = input_path + ".7z";
    if (auto exe = resolve_7z_executable()) {
        const int max_threads = std::max(2, omp_get_max_threads());
        std::string cmd = quote_arg(*exe) + " a -t7z -m0=lzma2 -mx=" + std::to_string(lvl) +
                          " -mmt=" + std::to_string(max_threads) + " -sdel " +
                          quote_arg(output_path) + " " + quote_arg(input_path);
        if (run_command(cmd)) {
            return true;
        }
    }
    return compress_file_xz(input_path, output_path, lvl);
}

bool decompress_with_7z_or_xz(const std::string &archive_path) {
    if (!fs::exists(archive_path)) {
        return false;
    }
    const std::string output_path = fs::path(archive_path).replace_extension().string();
    if (auto exe = resolve_7z_executable()) {
        const fs::path output_dir = fs::path(output_path).parent_path();
        std::string cmd = quote_arg(*exe) + " x " + quote_arg(archive_path) + " -o" +
                          quote_arg(output_dir.string()) + " -y";
        if (run_command(cmd)) {
            fs::remove(archive_path);
            return true;
        }
    }
    return decompress_file_xz(archive_path, output_path);
}

bool compress_uint64_array_native(const std::vector<uint64_t> &data, const std::string &output_base, int lvl) {
    if (data.size() < 65536ULL) {
        return false;
    }

    const size_t total_segments = (data.size() + kBlockSize - 1ULL) / kBlockSize;
    std::vector<U64SegmentEntry> segments(total_segments);
    std::vector<std::vector<uint8_t>> compressed_blocks(total_segments);
    if (data.size() > kParallelThreshold) {
        #pragma omp parallel for schedule(dynamic, 8)
        for (int64_t seg = 0; seg < static_cast<int64_t>(total_segments); ++seg) {
            const size_t start = static_cast<size_t>(seg) * kBlockSize;
            const size_t end = std::min(start + kBlockSize, data.size());
            const uint8_t *begin_ptr = reinterpret_cast<const uint8_t *>(data.data() + start);
            const size_t byte_size = (end - start) * sizeof(uint64_t);
            compressed_blocks[static_cast<size_t>(seg)] =
                xz_compress_bytes(begin_ptr, byte_size, static_cast<uint32_t>(lvl));
            segments[static_cast<size_t>(seg)] = {data[start], 0ULL};
        }
    } else {
        for (size_t seg = 0; seg < total_segments; ++seg) {
            const size_t start = seg * kBlockSize;
            const size_t end = std::min(start + kBlockSize, data.size());
            const uint8_t *begin_ptr = reinterpret_cast<const uint8_t *>(data.data() + start);
            const size_t byte_size = (end - start) * sizeof(uint64_t);
            compressed_blocks[seg] = xz_compress_bytes(begin_ptr, byte_size, static_cast<uint32_t>(lvl));
            segments[seg] = {data[start], 0ULL};
        }
    }

    uint64_t current_offset = 0;
    std::ofstream out(output_base + ".zi", std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    for (size_t seg = 0; seg < total_segments; ++seg) {
        segments[seg].file_offset = current_offset;
        const auto &block = compressed_blocks[seg];
        if (!block.empty()) {
            FileIOUtils::write_exact(out, block.data(), block.size(), output_base + ".zi");
        }
        current_offset += static_cast<uint64_t>(block.size());
    }
    write_binary_vector<U64SegmentEntry>(output_base + ".s", segments);
    return true;
}

std::vector<uint64_t> decompress_uint64_array_native(const std::string &compressed_path) {
    const fs::path zi_path(compressed_path);
    const fs::path segments_path = zi_path.parent_path() / (zi_path.stem().string() + ".s");
    std::vector<U64SegmentEntry> segments = read_u64_segments(segments_path);
    if (segments.empty()) {
        return {};
    }

    std::vector<uint64_t> result;
    for (size_t seg = 0; seg < segments.size(); ++seg) {
        const uint64_t begin = segments[seg].file_offset;
        uint64_t end = 0;
        if (seg + 1 < segments.size()) {
            end = segments[seg + 1].file_offset;
        } else {
            end = fs::file_size(zi_path);
        }
        std::vector<uint8_t> compressed = read_file_bytes_range(zi_path.string(), begin, end);
        std::vector<uint8_t> decompressed = decompress_xz_block_native(compressed.data(), compressed.size());
        const size_t item_count = decompressed.size() / sizeof(uint64_t);
        const size_t old_size = result.size();
        result.resize(old_size + item_count);
        if (item_count != 0) {
            std::memcpy(result.data() + old_size, decompressed.data(), item_count * sizeof(uint64_t));
        }
    }
    return result;
}

std::optional<size_t> find_value_uint64_compressed_native(const std::string &compressed_path, uint64_t value) {
    const fs::path zi_path(compressed_path);
    const fs::path segments_path = zi_path.parent_path() / (zi_path.stem().string() + ".s");
    std::vector<U64SegmentEntry> segments = read_u64_segments(segments_path);
    if (segments.empty() || value < segments.front().first_value) {
        return std::nullopt;
    }

    auto it = std::lower_bound(
        segments.begin(),
        segments.end(),
        value,
        [](const U64SegmentEntry &entry, uint64_t target) {
            return entry.first_value < target;
        });
    size_t seg_idx = static_cast<size_t>(std::distance(segments.begin(), it));
    if (it != segments.end() && it->first_value == value) {
        return seg_idx * kBlockSize;
    }
    if (seg_idx == 0) {
        return std::nullopt;
    }
    --seg_idx;

    const uint64_t begin = segments[seg_idx].file_offset;
    const uint64_t end =
        (seg_idx + 1 < segments.size()) ? segments[seg_idx + 1].file_offset : fs::file_size(zi_path);
    std::vector<uint8_t> compressed = read_file_bytes_range(zi_path.string(), begin, end);
    std::vector<uint8_t> decompressed = decompress_xz_block_native(compressed.data(), compressed.size());
    const uint64_t *data = reinterpret_cast<const uint64_t *>(decompressed.data());
    const size_t length = decompressed.size() / sizeof(uint64_t);
    auto found = std::lower_bound(data, data + length, value);
    if (found == data + length || *found != value) {
        return std::nullopt;
    }
    return seg_idx * kBlockSize + static_cast<size_t>(found - data);
}
