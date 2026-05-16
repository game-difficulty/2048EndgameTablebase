#include "NativeDiagnostics.h"

#if defined(NATIVE_DIAGNOSTIC_BUILD) && defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <string>

namespace {

void append_loader_log(const std::string &path, const std::string &line) {
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
    DWORD written = 0;
    (void)WriteFile(file, line.data(), static_cast<DWORD>(line.size()), &written, nullptr);
    FlushFileBuffers(file);
    CloseHandle(file);
}

std::string temp_loader_log_path() {
    char temp_path[MAX_PATH] = {};
    const DWORD len = GetTempPathA(MAX_PATH, temp_path);
    if (len == 0 || len >= MAX_PATH) {
        return "2048_formation_core_loader.log";
    }
    return std::string(temp_path) + "2048_formation_core_loader.log";
}

std::string module_loader_log_path(HMODULE module) {
    char path[MAX_PATH] = {};
    if (GetModuleFileNameA(module, path, MAX_PATH) == 0) {
        return "formation_core_loader_diagnostic.log";
    }
    std::string result(path);
    const size_t slash = result.find_last_of("\\/");
    if (slash == std::string::npos) {
        return "formation_core_loader_diagnostic.log";
    }
    result.resize(slash + 1U);
    result += "formation_core_loader_diagnostic.log";
    return result;
}

void log_loader_event(HMODULE module, const char *event) {
    const DWORD pid = GetCurrentProcessId();
    const DWORD tid = GetCurrentThreadId();
    char module_path[MAX_PATH] = {};
    (void)GetModuleFileNameA(module, module_path, MAX_PATH);
    std::string line = "pid=" + std::to_string(pid) +
        " tid=" + std::to_string(tid) +
        " event=" + event +
        " module=\"" + module_path + "\"\r\n";
    append_loader_log(temp_loader_log_path(), line);
    append_loader_log(module_loader_log_path(module), line);
}

} // namespace

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(instance);
        log_loader_event(instance, "DLL_PROCESS_ATTACH");
    } else if (reason == DLL_PROCESS_DETACH) {
        log_loader_event(instance, "DLL_PROCESS_DETACH");
    }
    return TRUE;
}

#else

int native_diagnostics_dllmain_stub = 0;

#endif
