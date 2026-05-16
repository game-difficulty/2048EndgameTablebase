#pragma once

#include "FormationRuntime.h"
#include "UniqueUtils.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#if defined(NATIVE_DIAGNOSTIC_BUILD) && defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
#endif

namespace NativeDiagnostics {

#if defined(NATIVE_DIAGNOSTIC_BUILD)

inline std::mutex &log_mutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::atomic<bool> &installed() {
    static std::atomic<bool> value{false};
    return value;
}

inline std::atomic<bool> &crash_logged() {
    static std::atomic<bool> value{false};
    return value;
}

inline std::atomic<int> &current_step() {
    static std::atomic<int> value{-999999};
    return value;
}

inline std::atomic<uint64_t> &heartbeat_sequence() {
    static std::atomic<uint64_t> value{0};
    return value;
}

inline char *log_path_buffer() {
    static char buffer[1024] = {};
    return buffer;
}

inline char *stage_buffer() {
    static char buffer[256] = {};
    return buffer;
}

inline uint64_t heartbeat_interval() {
    static const uint64_t interval = []() {
        const char *value = std::getenv("EGTB_DIAG_HEARTBEAT_INTERVAL");
        if (value == nullptr || value[0] == '\0') {
            return 100000ULL;
        }
        char *end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        return parsed == 0ULL ? 100000ULL : static_cast<uint64_t>(parsed);
    }();
    return interval;
}

inline std::string timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &now_time);
#else
    localtime_r(&now_time, &tm);
#endif
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count() % 1000;
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%d %H:%M:%S")
        << "." << std::setw(3) << std::setfill('0') << millis;
    return out.str();
}

inline std::string current_log_path() {
    const char *path = log_path_buffer();
    if (path[0] != '\0') {
        return std::string(path);
    }
    return "native_diagnostic.log";
}

inline std::string fallback_log_path() {
#if defined(_WIN32)
    char temp_path[MAX_PATH] = {};
    const DWORD len = GetTempPathA(MAX_PATH, temp_path);
    if (len != 0 && len < MAX_PATH) {
        std::filesystem::path path(temp_path);
        path /= "2048_native_diagnostic.log";
        return path.string();
    }
#endif
    return "2048_native_diagnostic.log";
}

inline void set_log_path(const std::string &path) {
    std::lock_guard<std::mutex> lock(log_mutex());
    char *buffer = log_path_buffer();
    std::snprintf(buffer, 1024, "%s", path.c_str());
}

inline void set_stage(const char *stage, int step) {
    if (step != -999999) {
        current_step().store(step, std::memory_order_relaxed);
    }
    char *buffer = stage_buffer();
    std::snprintf(buffer, 256, "%s", stage == nullptr ? "" : stage);
}

inline void write_line_unlocked(const std::string &line) {
    std::ostringstream full;
    full << timestamp()
        << " pid=";
#if defined(_WIN32)
    full << static_cast<unsigned long>(GetCurrentProcessId());
#else
    full << 0;
#endif
    full << " tid=" << std::hash<std::thread::id>{}(std::this_thread::get_id())
        << " " << line << "\n";
    const std::string full_line = full.str();
    const std::string primary = current_log_path();
    const std::string fallback = fallback_log_path();
    bool primary_ok = false;
    {
        std::ofstream out(primary, std::ios::app);
        if (out) {
            out << full_line;
            out.flush();
            primary_ok = true;
        }
    }
    if (!primary_ok || primary != fallback) {
        std::ofstream out(fallback, std::ios::app);
        if (out) {
            out << full_line;
            out.flush();
        }
    }
}

inline void write_line(const std::string &line) {
    std::lock_guard<std::mutex> lock(log_mutex());
    write_line_unlocked(line);
}

#if defined(_WIN32)
inline void write_crash_line(const std::string &line) {
    auto write_one = [&](const std::string &path) {
    HANDLE file = CreateFileA(
        path.c_str(),
        FILE_APPEND_DATA,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr,
        OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    if (file == INVALID_HANDLE_VALUE) {
        return;
    }
    std::string full = timestamp() + " pid=" + std::to_string(GetCurrentProcessId()) +
        " crash " + line + "\r\n";
    DWORD written = 0;
    (void)WriteFile(file, full.data(), static_cast<DWORD>(full.size()), &written, nullptr);
    FlushFileBuffers(file);
    CloseHandle(file);
    };
    const std::string primary = current_log_path();
    const std::string fallback = fallback_log_path();
    write_one(primary);
    if (primary != fallback) {
        write_one(fallback);
    }
}

inline std::string module_from_address(void *address) {
    HMODULE module = nullptr;
    if (GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(address),
            &module
        ) == 0 || module == nullptr) {
        return "";
    }
    char path[MAX_PATH] = {};
    if (GetModuleFileNameA(module, path, MAX_PATH) == 0) {
        return "";
    }
    return std::string(path);
}

inline std::string dump_path_from_log_path() {
    std::filesystem::path path(current_log_path());
    path.replace_extension(".dmp");
    return path.string();
}

inline void write_minidump(EXCEPTION_POINTERS *exception_info) {
    HMODULE dbghelp = LoadLibraryA("dbghelp.dll");
    if (dbghelp == nullptr) {
        write_crash_line("minidump skipped: dbghelp.dll not available");
        return;
    }
    using MiniDumpWriteDumpFn = BOOL (WINAPI *)(
        HANDLE,
        DWORD,
        HANDLE,
        MINIDUMP_TYPE,
        PMINIDUMP_EXCEPTION_INFORMATION,
        PMINIDUMP_USER_STREAM_INFORMATION,
        PMINIDUMP_CALLBACK_INFORMATION
    );
    auto fn = reinterpret_cast<MiniDumpWriteDumpFn>(GetProcAddress(dbghelp, "MiniDumpWriteDump"));
    if (fn == nullptr) {
        write_crash_line("minidump skipped: MiniDumpWriteDump not available");
        FreeLibrary(dbghelp);
        return;
    }
    const std::string dump_path = dump_path_from_log_path();
    HANDLE file = CreateFileA(
        dump_path.c_str(),
        GENERIC_WRITE,
        0,
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    if (file == INVALID_HANDLE_VALUE) {
        write_crash_line("minidump create failed path=" + dump_path);
        FreeLibrary(dbghelp);
        return;
    }
    MINIDUMP_EXCEPTION_INFORMATION mei{};
    mei.ThreadId = GetCurrentThreadId();
    mei.ExceptionPointers = exception_info;
    mei.ClientPointers = FALSE;
    const BOOL ok = fn(
        GetCurrentProcess(),
        GetCurrentProcessId(),
        file,
        MiniDumpWithIndirectlyReferencedMemory,
        exception_info == nullptr ? nullptr : &mei,
        nullptr,
        nullptr
    );
    CloseHandle(file);
    FreeLibrary(dbghelp);
    write_crash_line(std::string("minidump ") + (ok ? "written path=" : "failed path=") + dump_path);
}

inline LONG WINAPI unhandled_exception_filter(EXCEPTION_POINTERS *exception_info) {
    if (crash_logged().exchange(true, std::memory_order_acq_rel)) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    DWORD code = 0;
    void *address = nullptr;
    if (exception_info != nullptr && exception_info->ExceptionRecord != nullptr) {
        code = exception_info->ExceptionRecord->ExceptionCode;
        address = exception_info->ExceptionRecord->ExceptionAddress;
    }
    std::ostringstream out;
    out << "exception_code=0x" << std::hex << std::uppercase << code
        << " address=0x" << reinterpret_cast<uintptr_t>(address)
        << std::dec
        << " module=\"" << module_from_address(address) << "\""
        << " stage=\"" << stage_buffer() << "\""
        << " step=" << current_step().load(std::memory_order_relaxed)
        << " heartbeat_seq=" << heartbeat_sequence().load(std::memory_order_relaxed);
    write_crash_line(out.str());
    write_minidump(exception_info);
    return EXCEPTION_CONTINUE_SEARCH;
}

inline LONG CALLBACK vectored_exception_handler(EXCEPTION_POINTERS *exception_info) {
    if (exception_info == nullptr || exception_info->ExceptionRecord == nullptr) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const DWORD code = exception_info->ExceptionRecord->ExceptionCode;
    if (code == EXCEPTION_ACCESS_VIOLATION ||
        code == EXCEPTION_ILLEGAL_INSTRUCTION ||
        code == EXCEPTION_STACK_OVERFLOW ||
        code == EXCEPTION_ARRAY_BOUNDS_EXCEEDED ||
        code == EXCEPTION_DATATYPE_MISALIGNMENT) {
        return unhandled_exception_filter(exception_info);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

inline std::string cpu_brand_string() {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386))
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    if (__get_cpuid_max(0x80000000U, nullptr) < 0x80000004U) {
        return "unknown";
    }
    char brand[49] = {};
    unsigned int *as_ints = reinterpret_cast<unsigned int *>(brand);
    __get_cpuid(0x80000002U, as_ints + 0, as_ints + 1, as_ints + 2, as_ints + 3);
    __get_cpuid(0x80000003U, as_ints + 4, as_ints + 5, as_ints + 6, as_ints + 7);
    __get_cpuid(0x80000004U, as_ints + 8, as_ints + 9, as_ints + 10, as_ints + 11);
    return std::string(brand);
#else
    return "unknown";
#endif
}

inline void write_environment_snapshot() {
    std::ostringstream out;
    out << "environment cpu=\"" << cpu_brand_string() << "\""
        << " hw_threads=" << std::thread::hardware_concurrency()
        << " avx2=" << (UniqueUtils::cpu_has_avx2() ? 1 : 0)
        << " avx512=" << (UniqueUtils::cpu_has_avx512() ? 1 : 0)
        << " avx512_dq_bw_vl=" << (UniqueUtils::cpu_has_avx512_dq_bw_vl() ? 1 : 0)
        << " avx512_disabled_env=" << (UniqueUtils::avx512_disabled() ? 1 : 0);
#if defined(_WIN32)
    MEMORYSTATUSEX mem{};
    mem.dwLength = sizeof(mem);
    if (GlobalMemoryStatusEx(&mem)) {
        out << " memory_load=" << mem.dwMemoryLoad
            << " total_phys=" << static_cast<uint64_t>(mem.ullTotalPhys)
            << " avail_phys=" << static_cast<uint64_t>(mem.ullAvailPhys)
            << " total_pagefile=" << static_cast<uint64_t>(mem.ullTotalPageFile)
            << " avail_pagefile=" << static_cast<uint64_t>(mem.ullAvailPageFile);
    }
#endif
    write_line(out.str());
}

inline void install_crash_handler() {
    if (installed().exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    if (log_path_buffer()[0] == '\0') {
        const char *env = std::getenv("EGTB_DIAG_LOG");
        if (env != nullptr && env[0] != '\0') {
            set_log_path(env);
        }
    }
#if defined(_WIN32)
    SetUnhandledExceptionFilter(unhandled_exception_filter);
    AddVectoredExceptionHandler(1, vectored_exception_handler);
#endif
    write_line("diagnostic handler installed build=ReleaseDiagnostic");
    write_environment_snapshot();
}

inline std::filesystem::path diagnostic_dir_from_pathname(const std::string &pathname) {
    std::filesystem::path path(pathname);
    std::filesystem::path dir = path.parent_path();
    if (dir.empty()) {
        dir = std::filesystem::current_path();
    }
    return dir;
}

inline void configure_log_path(const RunOptions &options) {
    const char *env = std::getenv("EGTB_DIAG_LOG");
    if (env != nullptr && env[0] != '\0') {
        set_log_path(env);
        return;
    }
    const std::filesystem::path dir = diagnostic_dir_from_pathname(options.pathname);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    set_log_path((dir / "native_diagnostic.log").string());
}

template <typename Spec>
inline void start_run(const char *api, const Spec &spec, const RunOptions &options, uint64_t input_count = 0) {
    configure_log_path(options);
    install_crash_handler();
    set_stage(api, -999999);
    std::ostringstream out;
    out << "run_start api=" << api
        << " pattern=\"" << spec.name << "\""
        << " target=" << options.target
        << " steps=" << options.steps
        << " input_count=" << input_count
        << " pathname=\"" << options.pathname << "\""
        << " dtype=\"" << options.success_rate_dtype << "\""
        << " symm_mode=" << spec.symm_mode
        << " physical_transform=" << static_cast<uint32_t>(spec.physical_transform)
        << " logical_signature=" << spec.logical_pattern_signature
        << " physical_signature=" << spec.physical_pattern_signature
        << " compress=" << (options.compress ? 1 : 0)
        << " compress_temp_files=" << (options.compress_temp_files ? 1 : 0)
        << " optimal_branch_only=" << (options.optimal_branch_only ? 1 : 0)
        << " chunked_solve=" << (options.chunked_solve ? 1 : 0)
        << " direct_io=" << (options.direct_io ? 1 : 0)
        << " num_threads=" << options.num_threads;
    write_line(out.str());
}

inline void checkpoint(
    const char *stage,
    int step = -999999,
    uint64_t a = 0,
    uint64_t b = 0,
    uint64_t c = 0
) {
    set_stage(stage, step);
    std::ostringstream out;
    out << "checkpoint stage=\"" << stage << "\""
        << " step=" << step
        << " a=" << a
        << " b=" << b
        << " c=" << c;
    write_line(out.str());
}

inline void heartbeat(
    const char *stage,
    uint64_t index,
    uint64_t total = 0,
    uint64_t aux = 0
) {
    const uint64_t interval = heartbeat_interval();
    if (index != 0 && (index % interval) != 0) {
        return;
    }
    heartbeat_sequence().fetch_add(1, std::memory_order_relaxed);
    std::ostringstream out;
    out << "heartbeat stage=\"" << stage << "\""
        << " step=" << current_step().load(std::memory_order_relaxed)
        << " index=" << index
        << " total=" << total
        << " aux=" << aux;
    write_line(out.str());
}

inline void record_exception(const char *api, const char *what) {
    std::ostringstream out;
    out << "cpp_exception api=" << api
        << " stage=\"" << stage_buffer() << "\""
        << " step=" << current_step().load(std::memory_order_relaxed)
        << " what=\"" << (what == nullptr ? "" : what) << "\"";
    write_line(out.str());
}

#else

inline void install_crash_handler() {}
template <typename Spec>
inline void start_run(const char *, const Spec &, const RunOptions &, uint64_t = 0) {}
inline void checkpoint(const char *, int = -999999, uint64_t = 0, uint64_t = 0, uint64_t = 0) {}
inline void heartbeat(const char *, uint64_t, uint64_t = 0, uint64_t = 0) {}
inline void record_exception(const char *, const char *) {}

#endif

} // namespace NativeDiagnostics

#define NDIAG_INSTALL() ::NativeDiagnostics::install_crash_handler()
#define NDIAG_START_RUN(api, spec, options, input_count) ::NativeDiagnostics::start_run((api), (spec), (options), (input_count))
#define NDIAG_CHECKPOINT(stage, step, a, b, c) ::NativeDiagnostics::checkpoint((stage), (step), (a), (b), (c))
#define NDIAG_HEARTBEAT(stage, index, total, aux) ::NativeDiagnostics::heartbeat((stage), (index), (total), (aux))
#define NDIAG_EXCEPTION(api, what) ::NativeDiagnostics::record_exception((api), (what))
