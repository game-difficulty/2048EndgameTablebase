#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

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
    HANDLE file = CreateFileA(
        path,
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

    char exe_path[MAX_PATH] = {};
    const DWORD exe_len = GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
    if (exe_len > 0 && exe_len < MAX_PATH) {
        for (DWORD i = exe_len; i > 0; --i) {
            if (exe_path[i - 1] == '\\' || exe_path[i - 1] == '/') {
                exe_path[i] = '\0';
                break;
            }
        }
        char path[MAX_PATH] = {};
        std::snprintf(path, sizeof(path), "%snative_diagnostics.log", exe_path);
        append_file(path, line);
    }

    char temp_path[MAX_PATH] = {};
    const DWORD temp_len = GetTempPathA(MAX_PATH, temp_path);
    if (temp_len > 0 && temp_len < MAX_PATH) {
        char path[MAX_PATH] = {};
        std::snprintf(path, sizeof(path), "%s2048_native_diagnostics.log", temp_path);
        append_file(path, line);
    }
}

inline void mark(const std::string &stage) {
    if (!enabled()) {
        return;
    }
    std::snprintf(last_stage, sizeof(last_stage), "%s", stage.c_str());
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

inline LONG WINAPI crash_filter(EXCEPTION_POINTERS *info) {
    if (info != nullptr && info->ExceptionRecord != nullptr) {
        char line[1024] = {};
        std::snprintf(
            line,
            sizeof(line),
            "NATIVE_EXCEPTION code=0x%08lx address=%p last_stage=%s\n",
            static_cast<unsigned long>(info->ExceptionRecord->ExceptionCode),
            info->ExceptionRecord->ExceptionAddress,
            last_stage
        );
        append_line(line);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

inline void install_crash_handler(const char *module_name) {
    static std::atomic<bool> installed{false};
    bool expected = false;
    if (!installed.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return;
    }
    AddVectoredExceptionHandler(1, crash_filter);
    SetUnhandledExceptionFilter(crash_filter);
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
