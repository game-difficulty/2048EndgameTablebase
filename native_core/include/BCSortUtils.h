#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "NativeSortPolicy.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace BC {

using BCSortKeyValueUint64Uint32Fn = void (*)(uint64_t *, uint32_t *, size_t, bool);

namespace detail {

inline std::vector<std::filesystem::path> bc_bookgen_native_candidates() {
    std::vector<std::filesystem::path> candidates;
    auto append_unique = [&candidates](const std::filesystem::path &candidate) {
        if (!candidate.empty() &&
            std::find(candidates.begin(), candidates.end(), candidate) == candidates.end()) {
            candidates.push_back(candidate);
        }
    };

#ifdef _WIN32
    HMODULE module = nullptr;
    if (GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&bc_bookgen_native_candidates),
            &module) &&
        module != nullptr) {
        wchar_t module_path[MAX_PATH];
        const DWORD module_len = GetModuleFileNameW(module, module_path, MAX_PATH);
        if (module_len > 0U) {
            const std::filesystem::path module_dir =
                std::filesystem::path(module_path).parent_path();
            append_unique(module_dir / "bookgen_native.dll");
            append_unique(module_dir / "native_core" / "bookgen_native.dll");
        }
    }

    wchar_t exe_path[MAX_PATH];
    const DWORD exe_len = GetModuleFileNameW(nullptr, exe_path, MAX_PATH);
    if (exe_len > 0U) {
        const std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();
        append_unique(exe_dir / "bookgen_native.dll");
        append_unique(exe_dir / "native_core" / "bookgen_native.dll");
    }
    append_unique(std::filesystem::path("native_core") / "bookgen_native.dll");
    append_unique("bookgen_native.dll");
#endif
    return candidates;
}

inline BCSortKeyValueUint64Uint32Fn bc_resolve_keyvalue_sort_uint64_uint32() {
    static BCSortKeyValueUint64Uint32Fn fn = []() -> BCSortKeyValueUint64Uint32Fn {
        if (NativeSortPolicy::native_sort_disabled()) {
            return nullptr;
        }
        try {
#ifdef _WIN32
            for (const std::filesystem::path &candidate : bc_bookgen_native_candidates()) {
                if (!std::filesystem::exists(candidate)) {
                    continue;
                }
                NativeSortPolicy::prepare_bookgen_native_load(candidate);
                HMODULE lib = LoadLibraryW(candidate.wstring().c_str());
                if (lib == nullptr) {
                    continue;
                }
                auto proc = reinterpret_cast<BCSortKeyValueUint64Uint32Fn>(
                    GetProcAddress(lib, "keyvalue_sort_uint64_uint32")
                );
                if (proc != nullptr) {
                    return proc;
                }
            }
            return nullptr;
#else
            void *lib = dlopen("bookgen_native.so", RTLD_LAZY);
            if (lib == nullptr) {
                lib = dlopen("native_core/bookgen_native.so", RTLD_LAZY);
            }
            return lib != nullptr
                ? reinterpret_cast<BCSortKeyValueUint64Uint32Fn>(
                      dlsym(lib, "keyvalue_sort_uint64_uint32")
                  )
                : nullptr;
#endif
        } catch (...) {
            return nullptr;
        }
    }();
    return fn;
}

inline void bc_sort_keyvalue_uint64_uint32_by_order(
    uint64_t *keys,
    uint32_t *values,
    size_t count,
    const std::vector<size_t> &order
) {
    std::vector<uint64_t> sorted_keys(count);
    std::vector<uint32_t> sorted_values(count);
    for (size_t i = 0U; i < count; ++i) {
        sorted_keys[i] = keys[order[i]];
        sorted_values[i] = values[order[i]];
    }
    std::copy(sorted_keys.begin(), sorted_keys.end(), keys);
    std::copy(sorted_values.begin(), sorted_values.end(), values);
}

inline void bc_sort_keyvalue_uint64_uint32_fallback(
    uint64_t *keys,
    uint32_t *values,
    size_t count,
    bool descending
) {
    std::vector<size_t> order(count);
    for (size_t i = 0U; i < count; ++i) {
        order[i] = i;
    }
    if (descending) {
        std::sort(order.begin(), order.end(), [keys](size_t lhs, size_t rhs) {
            return keys[lhs] > keys[rhs];
        });
    } else {
        std::sort(order.begin(), order.end(), [keys](size_t lhs, size_t rhs) {
            return keys[lhs] < keys[rhs];
        });
    }
    bc_sort_keyvalue_uint64_uint32_by_order(keys, values, count, order);
}

} // namespace detail

inline void sort_keyvalue_uint64_uint32(
    uint64_t *keys,
    uint32_t *values,
    size_t count,
    bool descending
) {
    if (keys == nullptr || values == nullptr || count < 2U) {
        return;
    }
    if (count >= 10000U) {
        if (auto fn = detail::bc_resolve_keyvalue_sort_uint64_uint32(); fn != nullptr) {
            fn(keys, values, count, descending);
            return;
        }
    }

    detail::bc_sort_keyvalue_uint64_uint32_fallback(keys, values, count, descending);
}

} // namespace BC
