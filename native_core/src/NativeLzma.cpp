#include "NativeLzma.h"

#include "FileIOUtils.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
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
#include <fcntl.h>
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
        fs::path("7za.exe"),
        fs::path("7zz.exe"),
        fs::path("_internal") / "7z.exe",
        fs::path("_internal") / "7za.exe",
        fs::path("_internal") / "7zz.exe",
        fs::path("7zip") / "7z.exe",
        fs::path("7zip") / "7za.exe",
        fs::path("7zip") / "7zz.exe",
        fs::path("7z"),
        fs::path("7za"),
        fs::path("7zz"),
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

std::string build_command_line(const std::vector<std::string> &args) {
    std::string command_line;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i != 0) {
            command_line.push_back(' ');
        }
        command_line += quote_arg(args[i]);
    }
    return command_line;
}

#ifdef _WIN32
bool wait_process_success(PROCESS_INFORMATION &process_info) {
    WaitForSingleObject(process_info.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(process_info.hProcess, &exit_code);
    CloseHandle(process_info.hThread);
    CloseHandle(process_info.hProcess);
    process_info.hThread = nullptr;
    process_info.hProcess = nullptr;
    return exit_code == 0;
}

bool write_all_handle(HANDLE handle, const uint8_t *data, size_t size) {
    size_t offset = 0;
    while (offset < size) {
        const DWORD chunk = static_cast<DWORD>(std::min<size_t>(size - offset, 1U << 20));
        DWORD written = 0;
        if (!WriteFile(handle, data + offset, chunk, &written, nullptr)) {
            return false;
        }
        offset += static_cast<size_t>(written);
    }
    return true;
}

std::vector<uint8_t> read_all_handle(HANDLE handle, bool &ok) {
    ok = true;
    std::vector<uint8_t> output;
    std::array<uint8_t, 1U << 20> buffer{};
    for (;;) {
        DWORD read_bytes = 0;
        const BOOL read_ok = ReadFile(handle, buffer.data(), static_cast<DWORD>(buffer.size()), &read_bytes, nullptr);
        if (!read_ok || read_bytes == 0) {
            if (!read_ok && GetLastError() != ERROR_BROKEN_PIPE) {
                ok = false;
            }
            break;
        }
        output.insert(output.end(), buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(read_bytes));
    }
    return output;
}

bool spawn_process_with_redirects(
    const std::vector<std::string> &args,
    HANDLE child_stdin,
    HANDLE child_stdout,
    HANDLE child_stderr,
    PROCESS_INFORMATION &process_info
) {
    STARTUPINFOA startup_info{};
    startup_info.cb = sizeof(startup_info);
    startup_info.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES;
    startup_info.wShowWindow = SW_HIDE;
    startup_info.hStdInput = child_stdin;
    startup_info.hStdOutput = child_stdout;
    startup_info.hStdError = child_stderr;
    std::string command_line = build_command_line(args);
    std::vector<char> command(command_line.begin(), command_line.end());
    command.push_back('\0');
    return CreateProcessA(
        nullptr,
        command.data(),
        nullptr,
        nullptr,
        TRUE,
        CREATE_NO_WINDOW,
        nullptr,
        nullptr,
        &startup_info,
        &process_info) != FALSE;
}
#else
bool write_all_fd(int fd, const uint8_t *data, size_t size) {
    size_t offset = 0;
    while (offset < size) {
        ssize_t written = write(fd, data + offset, size - offset);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        offset += static_cast<size_t>(written);
    }
    return true;
}

std::vector<uint8_t> read_all_fd(int fd, bool &ok) {
    ok = true;
    std::vector<uint8_t> output;
    std::array<uint8_t, 1U << 20> buffer{};
    for (;;) {
        ssize_t read_bytes = read(fd, buffer.data(), buffer.size());
        if (read_bytes < 0) {
            if (errno == EINTR) {
                continue;
            }
            ok = false;
            break;
        }
        if (read_bytes == 0) {
            break;
        }
        output.insert(output.end(), buffer.begin(), buffer.begin() + read_bytes);
    }
    return output;
}

bool wait_pid_success(pid_t pid) {
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            return false;
        }
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}
#endif

bool compress_bytes_to_7z_archive_streaming_impl(
    const uint8_t *data,
    size_t size,
    const std::string &archive_path,
    const std::string &entry_name,
    int lvl
) {
    auto exe = resolve_7z_executable();
    if (!exe) {
        return false;
    }
    const int max_threads = std::max(2, omp_get_max_threads());
    std::error_code ec;
    fs::remove(archive_path, ec);
    const std::vector<std::string> args = {
        *exe,
        "a",
        "-t7z",
        "-m0=lzma2",
        "-mx=" + std::to_string(lvl),
        "-mmt=" + std::to_string(max_threads),
        "-bd",
        "-y",
        archive_path,
        "-si" + entry_name
    };

#ifdef _WIN32
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE stdin_read = nullptr;
    HANDLE stdin_write = nullptr;
    if (!CreatePipe(&stdin_read, &stdin_write, &sa, 0)) {
        return false;
    }
    SetHandleInformation(stdin_write, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul_out = CreateFileA("NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (nul_out == INVALID_HANDLE_VALUE) {
        CloseHandle(stdin_read);
        CloseHandle(stdin_write);
        return false;
    }
    PROCESS_INFORMATION process_info{};
    const bool spawned = spawn_process_with_redirects(args, stdin_read, nul_out, nul_out, process_info);
    CloseHandle(stdin_read);
    CloseHandle(nul_out);
    if (!spawned) {
        CloseHandle(stdin_write);
        return false;
    }
    const bool write_ok = write_all_handle(stdin_write, data, size);
    CloseHandle(stdin_write);
    const bool exit_ok = wait_process_success(process_info);
    if (!write_ok || !exit_ok) {
        fs::remove(archive_path, ec);
        return false;
    }
    return true;
#else
    int stdin_pipe[2];
    if (pipe(stdin_pipe) != 0) {
        return false;
    }
    pid_t pid = fork();
    if (pid < 0) {
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        return false;
    }
    if (pid == 0) {
        int nul_out = open("/dev/null", O_WRONLY);
        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(nul_out, STDOUT_FILENO);
        dup2(nul_out, STDERR_FILENO);
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        close(nul_out);
        std::vector<char *> argv;
        argv.reserve(args.size() + 1);
        for (const auto &arg : args) {
            argv.push_back(const_cast<char *>(arg.c_str()));
        }
        argv.push_back(nullptr);
        execvp(argv[0], argv.data());
        _exit(127);
    }
    close(stdin_pipe[0]);
    const bool write_ok = write_all_fd(stdin_pipe[1], data, size);
    close(stdin_pipe[1]);
    const bool exit_ok = wait_pid_success(pid);
    if (!write_ok || !exit_ok) {
        fs::remove(archive_path, ec);
        return false;
    }
    return true;
#endif
}

bool decompress_7z_archive_to_bytes_streaming_impl(const std::string &archive_path, std::vector<uint8_t> &output) {
    auto exe = resolve_7z_executable();
    if (!exe || !fs::exists(archive_path)) {
        return false;
    }
    const std::vector<std::string> args = {
        *exe,
        "x",
        "-so",
        "-bd",
        "-y",
        archive_path
    };

#ifdef _WIN32
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE stdout_read = nullptr;
    HANDLE stdout_write = nullptr;
    if (!CreatePipe(&stdout_read, &stdout_write, &sa, 0)) {
        return false;
    }
    SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul_in = CreateFileA("NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    HANDLE nul_err = CreateFileA("NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (nul_in == INVALID_HANDLE_VALUE || nul_err == INVALID_HANDLE_VALUE) {
        if (nul_in != INVALID_HANDLE_VALUE) CloseHandle(nul_in);
        if (nul_err != INVALID_HANDLE_VALUE) CloseHandle(nul_err);
        CloseHandle(stdout_read);
        CloseHandle(stdout_write);
        return false;
    }
    PROCESS_INFORMATION process_info{};
    const bool spawned = spawn_process_with_redirects(args, nul_in, stdout_write, nul_err, process_info);
    CloseHandle(nul_in);
    CloseHandle(nul_err);
    CloseHandle(stdout_write);
    if (!spawned) {
        CloseHandle(stdout_read);
        return false;
    }
    bool read_ok = false;
    output = read_all_handle(stdout_read, read_ok);
    CloseHandle(stdout_read);
    const bool exit_ok = wait_process_success(process_info);
    return read_ok && exit_ok;
#else
    int stdout_pipe[2];
    if (pipe(stdout_pipe) != 0) {
        return false;
    }
    pid_t pid = fork();
    if (pid < 0) {
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        return false;
    }
    if (pid == 0) {
        int nul_in = open("/dev/null", O_RDONLY);
        int nul_err = open("/dev/null", O_WRONLY);
        dup2(nul_in, STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(nul_err, STDERR_FILENO);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        close(nul_in);
        close(nul_err);
        std::vector<char *> argv;
        argv.reserve(args.size() + 1);
        for (const auto &arg : args) {
            argv.push_back(const_cast<char *>(arg.c_str()));
        }
        argv.push_back(nullptr);
        execvp(argv[0], argv.data());
        _exit(127);
    }
    close(stdout_pipe[1]);
    bool read_ok = false;
    output = read_all_fd(stdout_pipe[0], read_ok);
    close(stdout_pipe[0]);
    const bool exit_ok = wait_pid_success(pid);
    return read_ok && exit_ok;
#endif
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

struct SevenZipArchiveWriter::Impl {
    std::string archive_path;
#ifdef _WIN32
    HANDLE stdin_write = nullptr;
    PROCESS_INFORMATION process_info{};
#else
    int stdin_fd = -1;
    pid_t pid = -1;
#endif
    bool opened = false;

    void open_process(const std::string &path, const std::string &entry_name, int lvl) {
        if (opened) {
            throw std::runtime_error("7z archive writer is already open");
        }
        auto exe = resolve_7z_executable();
        if (!exe) {
            throw std::runtime_error("7z executable not found");
        }
        archive_path = path;
        const int max_threads = std::max(2, omp_get_max_threads());
        std::error_code ec;
        fs::remove(archive_path, ec);
        const std::vector<std::string> args = {
            *exe,
            "a",
            "-t7z",
            "-m0=lzma2",
            "-mx=" + std::to_string(lvl),
            "-mmt=" + std::to_string(max_threads),
            "-bd",
            "-y",
            archive_path,
            "-si" + entry_name
        };

#ifdef _WIN32
        SECURITY_ATTRIBUTES sa{};
        sa.nLength = sizeof(sa);
        sa.bInheritHandle = TRUE;
        HANDLE stdin_read = nullptr;
        if (!CreatePipe(&stdin_read, &stdin_write, &sa, 0)) {
            throw std::runtime_error("failed to create 7z stdin pipe");
        }
        SetHandleInformation(stdin_write, HANDLE_FLAG_INHERIT, 0);
        HANDLE nul_out = CreateFileA(
            "NUL",
            GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            &sa,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);
        if (nul_out == INVALID_HANDLE_VALUE) {
            CloseHandle(stdin_read);
            CloseHandle(stdin_write);
            stdin_write = nullptr;
            throw std::runtime_error("failed to open NUL for 7z output");
        }
        const bool spawned = spawn_process_with_redirects(args, stdin_read, nul_out, nul_out, process_info);
        CloseHandle(stdin_read);
        CloseHandle(nul_out);
        if (!spawned) {
            CloseHandle(stdin_write);
            stdin_write = nullptr;
            throw std::runtime_error("failed to spawn 7z archive writer");
        }
#else
        int stdin_pipe[2];
        if (pipe(stdin_pipe) != 0) {
            throw std::runtime_error("failed to create 7z stdin pipe");
        }
        pid = fork();
        if (pid < 0) {
            close(stdin_pipe[0]);
            close(stdin_pipe[1]);
            pid = -1;
            throw std::runtime_error("failed to fork 7z archive writer");
        }
        if (pid == 0) {
            int nul_out = open("/dev/null", O_WRONLY);
            dup2(stdin_pipe[0], STDIN_FILENO);
            dup2(nul_out, STDOUT_FILENO);
            dup2(nul_out, STDERR_FILENO);
            close(stdin_pipe[0]);
            close(stdin_pipe[1]);
            close(nul_out);
            std::vector<char *> argv;
            argv.reserve(args.size() + 1);
            for (const auto &arg : args) {
                argv.push_back(const_cast<char *>(arg.c_str()));
            }
            argv.push_back(nullptr);
            execvp(argv[0], argv.data());
            _exit(127);
        }
        close(stdin_pipe[0]);
        stdin_fd = stdin_pipe[1];
#endif
        opened = true;
    }

    void append_bytes(const void *data, size_t size) {
        if (size == 0U) {
            return;
        }
        if (!opened) {
            throw std::runtime_error("7z archive writer is not open");
        }
        if (data == nullptr) {
            throw std::runtime_error("attempted to append null data to 7z archive");
        }
        const uint8_t *bytes = static_cast<const uint8_t *>(data);
#ifdef _WIN32
        if (!write_all_handle(stdin_write, bytes, size)) {
            throw std::runtime_error("failed while writing 7z archive stream");
        }
#else
        if (!write_all_fd(stdin_fd, bytes, size)) {
            throw std::runtime_error("failed while writing 7z archive stream");
        }
#endif
    }

    bool finish(bool throw_on_error) {
        if (!opened) {
            return true;
        }
        opened = false;
        bool ok = true;
#ifdef _WIN32
        if (stdin_write != nullptr) {
            CloseHandle(stdin_write);
            stdin_write = nullptr;
        }
        ok = wait_process_success(process_info);
#else
        if (stdin_fd >= 0) {
            close(stdin_fd);
            stdin_fd = -1;
        }
        if (pid >= 0) {
            ok = wait_pid_success(pid);
            pid = -1;
        }
#endif
        if (!ok) {
            std::error_code ec;
            fs::remove(archive_path, ec);
            if (throw_on_error) {
                throw std::runtime_error("7z archive writer failed: " + archive_path);
            }
        }
        return ok;
    }

    ~Impl() {
        try {
            finish(false);
        } catch (...) {
        }
    }
};

struct SevenZipSequentialReader::Impl {
    std::string archive_path;
#ifdef _WIN32
    HANDLE stdout_read = nullptr;
    PROCESS_INFORMATION process_info{};
#else
    int stdout_fd = -1;
    pid_t pid = -1;
#endif
    bool opened = false;

    void open_process(const std::string &path) {
        if (opened) {
            throw std::runtime_error("7z archive reader is already open");
        }
        auto exe = resolve_7z_executable();
        if (!exe || !fs::exists(path)) {
            throw std::runtime_error("7z archive not available: " + path);
        }
        archive_path = path;
        const std::vector<std::string> args = {
            *exe,
            "x",
            "-so",
            "-bd",
            "-y",
            archive_path
        };

#ifdef _WIN32
        SECURITY_ATTRIBUTES sa{};
        sa.nLength = sizeof(sa);
        sa.bInheritHandle = TRUE;
        HANDLE stdout_write = nullptr;
        if (!CreatePipe(&stdout_read, &stdout_write, &sa, 0)) {
            throw std::runtime_error("failed to create 7z stdout pipe");
        }
        SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0);
        HANDLE nul_in = CreateFileA(
            "NUL",
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            &sa,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);
        HANDLE nul_err = CreateFileA(
            "NUL",
            GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            &sa,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);
        if (nul_in == INVALID_HANDLE_VALUE || nul_err == INVALID_HANDLE_VALUE) {
            if (nul_in != INVALID_HANDLE_VALUE) CloseHandle(nul_in);
            if (nul_err != INVALID_HANDLE_VALUE) CloseHandle(nul_err);
            CloseHandle(stdout_read);
            CloseHandle(stdout_write);
            stdout_read = nullptr;
            throw std::runtime_error("failed to open NUL for 7z reader");
        }
        const bool spawned = spawn_process_with_redirects(args, nul_in, stdout_write, nul_err, process_info);
        CloseHandle(nul_in);
        CloseHandle(nul_err);
        CloseHandle(stdout_write);
        if (!spawned) {
            CloseHandle(stdout_read);
            stdout_read = nullptr;
            throw std::runtime_error("failed to spawn 7z archive reader");
        }
#else
        int stdout_pipe[2];
        if (pipe(stdout_pipe) != 0) {
            throw std::runtime_error("failed to create 7z stdout pipe");
        }
        pid = fork();
        if (pid < 0) {
            close(stdout_pipe[0]);
            close(stdout_pipe[1]);
            pid = -1;
            throw std::runtime_error("failed to fork 7z archive reader");
        }
        if (pid == 0) {
            int nul_in = open("/dev/null", O_RDONLY);
            int nul_err = open("/dev/null", O_WRONLY);
            dup2(nul_in, STDIN_FILENO);
            dup2(stdout_pipe[1], STDOUT_FILENO);
            dup2(nul_err, STDERR_FILENO);
            close(stdout_pipe[0]);
            close(stdout_pipe[1]);
            close(nul_in);
            close(nul_err);
            std::vector<char *> argv;
            argv.reserve(args.size() + 1);
            for (const auto &arg : args) {
                argv.push_back(const_cast<char *>(arg.c_str()));
            }
            argv.push_back(nullptr);
            execvp(argv[0], argv.data());
            _exit(127);
        }
        close(stdout_pipe[1]);
        stdout_fd = stdout_pipe[0];
#endif
        opened = true;
    }

    void read_exact(void *dst, size_t bytes) {
        if (bytes == 0U) {
            return;
        }
        if (!opened) {
            throw std::runtime_error("7z archive reader is not open");
        }
        if (dst == nullptr) {
            throw std::runtime_error("attempted to read 7z archive into null buffer");
        }
        uint8_t *out = static_cast<uint8_t *>(dst);
        size_t offset = 0;
        while (offset < bytes) {
#ifdef _WIN32
            DWORD read_bytes = 0;
            const DWORD chunk = static_cast<DWORD>(std::min<size_t>(bytes - offset, 1U << 20));
            const BOOL ok = ReadFile(stdout_read, out + offset, chunk, &read_bytes, nullptr);
            if (!ok || read_bytes == 0) {
                throw std::runtime_error("truncated 7z archive stream: " + archive_path);
            }
            offset += static_cast<size_t>(read_bytes);
#else
            const size_t chunk = std::min<size_t>(bytes - offset, 1U << 20);
            ssize_t read_bytes = read(stdout_fd, out + offset, chunk);
            if (read_bytes < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error("failed while reading 7z archive stream: " + archive_path);
            }
            if (read_bytes == 0) {
                throw std::runtime_error("truncated 7z archive stream: " + archive_path);
            }
            offset += static_cast<size_t>(read_bytes);
#endif
        }
    }

    bool finish(bool throw_on_error) {
        if (!opened) {
            return true;
        }
        opened = false;
        bool ok = true;
#ifdef _WIN32
        if (stdout_read != nullptr) {
            CloseHandle(stdout_read);
            stdout_read = nullptr;
        }
        ok = wait_process_success(process_info);
#else
        if (stdout_fd >= 0) {
            close(stdout_fd);
            stdout_fd = -1;
        }
        if (pid >= 0) {
            ok = wait_pid_success(pid);
            pid = -1;
        }
#endif
        if (!ok && throw_on_error) {
            throw std::runtime_error("7z archive reader failed: " + archive_path);
        }
        return ok;
    }

    ~Impl() {
        try {
            finish(false);
        } catch (...) {
        }
    }
};

SevenZipArchiveWriter::SevenZipArchiveWriter() = default;

SevenZipArchiveWriter::SevenZipArchiveWriter(
    const std::string &archive_path,
    const std::string &entry_name,
    int lvl
) {
    open(archive_path, entry_name, lvl);
}

SevenZipArchiveWriter::~SevenZipArchiveWriter() = default;
SevenZipArchiveWriter::SevenZipArchiveWriter(SevenZipArchiveWriter &&) noexcept = default;
SevenZipArchiveWriter &SevenZipArchiveWriter::operator=(SevenZipArchiveWriter &&) noexcept = default;

void SevenZipArchiveWriter::open(const std::string &archive_path, const std::string &entry_name, int lvl) {
    impl_ = std::make_unique<Impl>();
    impl_->open_process(archive_path, entry_name, lvl);
}

void SevenZipArchiveWriter::append(const void *data, size_t size) {
    if (!impl_) {
        throw std::runtime_error("7z archive writer is not open");
    }
    impl_->append_bytes(data, size);
}

void SevenZipArchiveWriter::close() {
    if (impl_) {
        impl_->finish(true);
        impl_.reset();
    }
}

bool SevenZipArchiveWriter::is_open() const {
    return impl_ != nullptr && impl_->opened;
}

SevenZipSequentialReader::SevenZipSequentialReader() = default;

SevenZipSequentialReader::SevenZipSequentialReader(const std::string &archive_path) {
    open(archive_path);
}

SevenZipSequentialReader::~SevenZipSequentialReader() = default;
SevenZipSequentialReader::SevenZipSequentialReader(SevenZipSequentialReader &&) noexcept = default;
SevenZipSequentialReader &SevenZipSequentialReader::operator=(SevenZipSequentialReader &&) noexcept = default;

void SevenZipSequentialReader::open(const std::string &archive_path) {
    impl_ = std::make_unique<Impl>();
    impl_->open_process(archive_path);
}

void SevenZipSequentialReader::read(void *dst, size_t bytes) {
    if (!impl_) {
        throw std::runtime_error("7z archive reader is not open");
    }
    impl_->read_exact(dst, bytes);
}

void SevenZipSequentialReader::close() {
    if (impl_) {
        impl_->finish(true);
        impl_.reset();
    }
}

bool SevenZipSequentialReader::is_open() const {
    return impl_ != nullptr && impl_->opened;
}

bool compress_bytes_to_7z_archive_streaming(
    const uint8_t *data,
    size_t size,
    const std::string &archive_path,
    const std::string &entry_name,
    int lvl
) {
    return compress_bytes_to_7z_archive_streaming_impl(data, size, archive_path, entry_name, lvl);
}

bool compress_spans_to_7z_archive_streaming(
    const std::vector<ArchiveByteSpan> &spans,
    const std::string &archive_path,
    const std::string &entry_name,
    int lvl
) {
    try {
        SevenZipArchiveWriter writer(archive_path, entry_name, lvl);
        for (const ArchiveByteSpan &span : spans) {
            writer.append(span.data, span.size);
        }
        writer.close();
        return true;
    } catch (...) {
        std::error_code ec;
        fs::remove(archive_path, ec);
        return false;
    }
}

bool decompress_7z_archive_to_bytes_streaming(const std::string &archive_path, std::vector<uint8_t> &output) {
    output.clear();
    return decompress_7z_archive_to_bytes_streaming_impl(archive_path, output);
}

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
