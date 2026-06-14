#pragma once

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace NativeSortPolicy {

inline bool env_truthy(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "OFF") != 0;
}

inline bool native_sort_disabled() {
    return env_truthy("TABLEBASE_DISABLE_NATIVE_SORT") ||
           env_truthy("TABLEBASE_FORCE_STD_SORT");
}

inline bool sort_avx512_enabled() {
    return env_truthy("TABLEBASE_SORT_AVX512") ||
           env_truthy("TABLEBASE_ENABLE_SORT_AVX512");
}

inline bool sort_avx512_probe_disabled() {
    return env_truthy("TABLEBASE_DISABLE_SORT_AVX512_PROBE");
}

#if defined(_WIN32)

inline std::mutex &sort_probe_mutex() {
    static std::mutex value;
    return value;
}

inline int &sort_probe_state() {
    // 0 unknown, 1 AVX512 probe passed, 2 AVX512 probe failed.
    static int value = 0;
    return value;
}

inline std::string quote_arg(const std::string &arg) {
    std::string result;
    result.reserve(arg.size() + 2U);
    result.push_back('"');
    result.append(arg);
    result.push_back('"');
    return result;
}

inline bool run_avx512_sort_probe(const std::filesystem::path &bookgen_native_path) {
    char system_dir[MAX_PATH] = {};
    const UINT system_dir_len = GetSystemDirectoryA(system_dir, MAX_PATH);
    if (system_dir_len == 0U || system_dir_len >= MAX_PATH) {
        return false;
    }

    std::filesystem::path dll_path;
    try {
        dll_path = std::filesystem::absolute(bookgen_native_path);
    } catch (...) {
        dll_path = bookgen_native_path;
    }

    const std::string rundll32 =
        (std::filesystem::path(system_dir) / "rundll32.exe").string();
    const std::string command =
        quote_arg(rundll32) + " " +
        quote_arg(dll_path.string()) + ",native_sort_probe_rundll32";

    std::vector<char> command_buffer(command.begin(), command.end());
    command_buffer.push_back('\0');

    STARTUPINFOA startup_info = {};
    startup_info.cb = sizeof(startup_info);
    PROCESS_INFORMATION process_info = {};
    if (!CreateProcessA(
            nullptr,
            command_buffer.data(),
            nullptr,
            nullptr,
            FALSE,
            CREATE_NO_WINDOW,
            nullptr,
            nullptr,
            &startup_info,
            &process_info
        )) {
        return false;
    }

    bool passed = false;
    const DWORD wait_result = WaitForSingleObject(process_info.hProcess, 15000U);
    if (wait_result == WAIT_OBJECT_0) {
        DWORD exit_code = 1U;
        if (GetExitCodeProcess(process_info.hProcess, &exit_code) && exit_code == 0U) {
            passed = true;
        }
    } else {
        TerminateProcess(process_info.hProcess, 0x100U);
        WaitForSingleObject(process_info.hProcess, 1000U);
    }
    CloseHandle(process_info.hThread);
    CloseHandle(process_info.hProcess);
    return passed;
}

#endif

inline void prepare_bookgen_native_load(const std::filesystem::path &bookgen_native_path) {
#if defined(_WIN32)
    if (sort_avx512_enabled()) {
        SetEnvironmentVariableA("XSS_DISABLE_AVX512", nullptr);
        return;
    }

    if (std::getenv("XSS_DISABLE_AVX512") != nullptr) {
        return;
    }

    if (sort_avx512_probe_disabled()) {
        SetEnvironmentVariableA("XSS_DISABLE_AVX512", "1");
        return;
    }

    std::lock_guard<std::mutex> lock(sort_probe_mutex());
    if (std::getenv("XSS_DISABLE_AVX512") != nullptr) {
        return;
    }

    int &state = sort_probe_state();
    if (state == 0) {
        state = run_avx512_sort_probe(bookgen_native_path) ? 1 : 2;
    }

    if (state != 1) {
        SetEnvironmentVariableA("XSS_DISABLE_AVX512", "1");
    }
#endif
}

} // namespace NativeSortPolicy
