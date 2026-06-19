#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>

#include "PathUtils.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace NativeDiagnostics {

inline bool enabled() {
    const char *value = std::getenv("TABLEBASE_NATIVE_DIAG");
    return value != nullptr &&
           value[0] != '\0' &&
           std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0;
}

inline std::atomic<uint64_t> &sequence() {
    static std::atomic<uint64_t> value{0};
    return value;
}

inline thread_local char last_stage[512] = "startup";

#if defined(_WIN32)

inline void append_file(const char *path, const char *text) {
    if (path == nullptr || path[0] == '\0' || text == nullptr) {
        return;
    }
    const std::filesystem::path native_path = NativePath::from_utf8(path);
    HANDLE file = CreateFileW(
        native_path.wstring().c_str(),
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
    DWORD written = 0;
    WriteFile(file, text, static_cast<DWORD>(std::strlen(text)), &written, nullptr);
    CloseHandle(file);
}

inline void append_file_w(const std::filesystem::path &path, const char *text) {
    if (text == nullptr) {
        return;
    }
    HANDLE file = CreateFileW(
        path.wstring().c_str(),
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
    DWORD written = 0;
    WriteFile(file, text, static_cast<DWORD>(std::strlen(text)), &written, nullptr);
    CloseHandle(file);
}

inline void append_line(const char *line) {
    if (!enabled()) {
        return;
    }

    const char *override_path = std::getenv("TABLEBASE_NATIVE_DIAG_FILE");
    if (override_path != nullptr && override_path[0] != '\0') {
        append_file(override_path, line);
        return;
    }

    wchar_t exe_path[MAX_PATH] = {};
    const DWORD exe_len = GetModuleFileNameW(nullptr, exe_path, MAX_PATH);
    if (exe_len > 0 && exe_len < MAX_PATH) {
        const std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();
        append_file_w(exe_dir / L"native_diagnostics.log", line);
    }

    wchar_t temp_path[MAX_PATH] = {};
    const DWORD temp_len = GetTempPathW(MAX_PATH, temp_path);
    if (temp_len > 0 && temp_len < MAX_PATH) {
        append_file_w(std::filesystem::path(temp_path) / L"2048_native_diagnostics.log", line);
    }
}

inline const char *logger_file_path() {
    const char *override_path = std::getenv("TABLEBASE_NATIVE_LOG_FILE");
    if (override_path != nullptr && override_path[0] != '\0') {
        return override_path;
    }
    return "logger.txt";
}

inline void append_logger_line(const char *line) {
    if (line == nullptr || line[0] == '\0') {
        return;
    }
    append_file(logger_file_path(), line);
}

inline void mark(const std::string &stage) {
    std::snprintf(last_stage, sizeof(last_stage), "%s", stage.c_str());
    if (!enabled()) {
        return;
    }
    SYSTEMTIME st;
    GetLocalTime(&st);
    char line[1024] = {};
    std::snprintf(
        line,
        sizeof(line),
        "%04u-%02u-%02u %02u:%02u:%02u.%03u pid=%lu tid=%lu seq=%llu %s\n",
        static_cast<unsigned>(st.wYear),
        static_cast<unsigned>(st.wMonth),
        static_cast<unsigned>(st.wDay),
        static_cast<unsigned>(st.wHour),
        static_cast<unsigned>(st.wMinute),
        static_cast<unsigned>(st.wSecond),
        static_cast<unsigned>(st.wMilliseconds),
        static_cast<unsigned long>(GetCurrentProcessId()),
        static_cast<unsigned long>(GetCurrentThreadId()),
        static_cast<unsigned long long>(sequence().fetch_add(1, std::memory_order_relaxed)),
        last_stage
    );
    append_line(line);
}

inline void format_crash_line(EXCEPTION_POINTERS *info, char *line, size_t line_size) {
    if (line == nullptr || line_size == 0U) {
        return;
    }
    if (info != nullptr && info->ExceptionRecord != nullptr) {
        std::snprintf(
            line,
            line_size,
            "NATIVE_EXCEPTION code=0x%08lx address=%p last_stage=%s\n",
            static_cast<unsigned long>(info->ExceptionRecord->ExceptionCode),
            info->ExceptionRecord->ExceptionAddress,
            last_stage
        );
    } else {
        std::snprintf(line, line_size, "NATIVE_EXCEPTION code=unknown address=unknown last_stage=%s\n", last_stage);
    }
}

inline LONG WINAPI diagnostic_crash_filter(EXCEPTION_POINTERS *info) {
    if (enabled()) {
        char line[1024] = {};
        format_crash_line(info, line, sizeof(line));
        append_line(line);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

inline LONG WINAPI unhandled_crash_filter(EXCEPTION_POINTERS *info) {
    char crash_line[1024] = {};
    format_crash_line(info, crash_line, sizeof(crash_line));

    SYSTEMTIME st;
    GetLocalTime(&st);
    char logger_line[1400] = {};
    std::snprintf(
        logger_line,
        sizeof(logger_line),
        "%04u-%02u-%02u %02u:%02u:%02u,%03u - ERROR - Native crash in formation_core: %s",
        static_cast<unsigned>(st.wYear),
        static_cast<unsigned>(st.wMonth),
        static_cast<unsigned>(st.wDay),
        static_cast<unsigned>(st.wHour),
        static_cast<unsigned>(st.wMinute),
        static_cast<unsigned>(st.wSecond),
        static_cast<unsigned>(st.wMilliseconds),
        crash_line
    );
    append_logger_line(logger_line);
    if (enabled()) {
        append_line(crash_line);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

inline void install_crash_handler(const char *module_name) {
    static std::atomic<bool> installed{false};
    bool expected = false;
    if (!installed.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return;
    }
    AddVectoredExceptionHandler(1, diagnostic_crash_filter);
    SetUnhandledExceptionFilter(unhandled_crash_filter);
    mark(std::string("diagnostics installed module=") + (module_name == nullptr ? "unknown" : module_name));
}

#else

inline void append_line(const char *) {}
inline void mark(const std::string &stage) {
    std::snprintf(last_stage, sizeof(last_stage), "%s", stage.c_str());
}
inline void install_crash_handler(const char *) {}

#endif

class Scope {
public:
    explicit Scope(std::string stage) : stage_(std::move(stage)) {
        mark("BEGIN " + stage_);
    }

    ~Scope() {
        mark("END " + stage_);
    }

    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

private:
    std::string stage_;
};

} // namespace NativeDiagnostics
