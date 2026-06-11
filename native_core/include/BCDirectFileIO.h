#pragma once

#include "BCFileIO.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <malloc.h>
#include <windows.h>
#endif

#if !defined(_WIN32) && defined(__linux__)
#include <cstdlib>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#ifndef O_DIRECT
#define O_DIRECT 040000
#endif
#endif

namespace BC {

struct BCDirectFileIOOptions {
    uint32_t alignment = 4096U;
    uint32_t queue_depth = 1U;
    // Enables the reusable high-QD positioned IO path for read_many/write_many.
    // The direct writer only uses this path for planned writes that do not need
    // read-modify-write preservation of untouched bytes inside aligned blocks.
    bool overlapped = false;

    // If present, size() reports the logical bytes while the physical file may
    // be padded to alignment. This is required when reading BC position/success
    // files through direct IO without changing their logical format.
    std::optional<uint64_t> logical_size = std::nullopt;

    // Preserve bytes in aligned physical blocks that are not covered by a
    // write request. This is slower but required for append-style streams that
    // may issue non-4KB-aligned writes into an existing tail block.
    bool preserve_unwritten_bytes = false;
};

[[nodiscard]] inline uint64_t bc_direct_align_down(uint64_t value, uint64_t alignment) {
    if (alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
        throw std::invalid_argument("BC direct IO alignment must be a power of two");
    }
    return value & ~(alignment - 1U);
}

[[nodiscard]] inline uint64_t bc_direct_align_up(uint64_t value, uint64_t alignment) {
    if (alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
        throw std::invalid_argument("BC direct IO alignment must be a power of two");
    }
    if (value > std::numeric_limits<uint64_t>::max() - (alignment - 1U)) {
        throw std::overflow_error("BC direct IO align_up overflow");
    }
    return (value + alignment - 1U) & ~(alignment - 1U);
}

#ifdef _WIN32

namespace detail {

[[nodiscard]] inline std::wstring bc_direct_wide_path(const std::filesystem::path &path) {
    return path.wstring();
}

[[nodiscard]] inline std::string bc_direct_win_error(const char *label, DWORD error = GetLastError()) {
    LPSTR message = nullptr;
    const DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS;
    const DWORD len = FormatMessageA(
        flags,
        nullptr,
        error,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&message),
        0,
        nullptr
    );
    std::string out(label);
    out += " (";
    out += std::to_string(error);
    out += ")";
    if (len != 0U && message != nullptr) {
        out += ": ";
        out += message;
    }
    if (message != nullptr) {
        LocalFree(message);
    }
    return out;
}

class BCWinHandle {
public:
    BCWinHandle() = default;
    explicit BCWinHandle(HANDLE handle) : handle_(handle) {}
    ~BCWinHandle() {
        close();
    }

    BCWinHandle(const BCWinHandle &) = delete;
    BCWinHandle &operator=(const BCWinHandle &) = delete;

    BCWinHandle(BCWinHandle &&other) noexcept : handle_(other.handle_) {
        other.handle_ = INVALID_HANDLE_VALUE;
    }

    BCWinHandle &operator=(BCWinHandle &&other) noexcept {
        if (this != &other) {
            close();
            handle_ = other.handle_;
            other.handle_ = INVALID_HANDLE_VALUE;
        }
        return *this;
    }

    [[nodiscard]] HANDLE get() const {
        return handle_;
    }

    [[nodiscard]] bool valid() const {
        return handle_ != INVALID_HANDLE_VALUE && handle_ != nullptr;
    }

    void close() {
        if (valid()) {
            CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
        }
    }

private:
    HANDLE handle_ = INVALID_HANDLE_VALUE;
};

class BCAlignedBuffer {
public:
    BCAlignedBuffer() = default;
    BCAlignedBuffer(uint64_t bytes, uint32_t alignment) {
        reset(bytes, alignment);
    }
    ~BCAlignedBuffer() {
        reset();
    }

    BCAlignedBuffer(const BCAlignedBuffer &) = delete;
    BCAlignedBuffer &operator=(const BCAlignedBuffer &) = delete;

    BCAlignedBuffer(BCAlignedBuffer &&other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0U;
    }

    BCAlignedBuffer &operator=(BCAlignedBuffer &&other) noexcept {
        if (this != &other) {
            reset();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0U;
        }
        return *this;
    }

    void reset(uint64_t bytes = 0U, uint32_t alignment = 4096U) {
        if (data_ != nullptr) {
            _aligned_free(data_);
            data_ = nullptr;
            size_ = 0U;
        }
        if (bytes == 0U) {
            return;
        }
        if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC direct IO aligned buffer exceeds size_t");
        }
        data_ = static_cast<uint8_t *>(_aligned_malloc(static_cast<size_t>(bytes), alignment));
        if (data_ == nullptr) {
            throw std::bad_alloc();
        }
        size_ = static_cast<size_t>(bytes);
    }

    void ensure_at_least(uint64_t bytes, uint32_t alignment = 4096U) {
        if (bytes <= size_) {
            return;
        }
        reset(bytes, alignment);
    }

    [[nodiscard]] uint8_t *data() {
        return data_;
    }

    [[nodiscard]] const uint8_t *data() const {
        return data_;
    }

    [[nodiscard]] size_t size() const {
        return size_;
    }

private:
    uint8_t *data_ = nullptr;
    size_t size_ = 0U;
};

[[nodiscard]] inline BCDirectFileIOOptions normalize_direct_options(const BCDirectFileIOOptions &input) {
    BCDirectFileIOOptions options;
    options.alignment = input.alignment;
    options.queue_depth = input.queue_depth;
    options.overlapped = input.overlapped;
    options.logical_size = input.logical_size;
    options.preserve_unwritten_bytes = input.preserve_unwritten_bytes;
    if (options.alignment == 0U || (options.alignment & (options.alignment - 1U)) != 0U) {
        throw std::invalid_argument("BC direct IO alignment must be a non-zero power of two");
    }
    if (options.queue_depth == 0U) {
        options.queue_depth = 1U;
    }
    if (options.queue_depth > MAXIMUM_WAIT_OBJECTS) {
        options.queue_depth = MAXIMUM_WAIT_OBJECTS;
    }
    return options;
}

[[nodiscard]] inline BCWinHandle open_direct_handle(
    const std::filesystem::path &path,
    bool write,
    bool overlapped
) {
    const DWORD access = write ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ;
    const DWORD creation = write ? CREATE_ALWAYS : OPEN_EXISTING;
    DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING;
    if (overlapped) {
        flags |= FILE_FLAG_OVERLAPPED;
    }
    const std::wstring wpath = bc_direct_wide_path(path);
    HANDLE handle = CreateFileW(
        wpath.c_str(),
        access,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr,
        creation,
        flags,
        nullptr
    );
    if (handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error(bc_direct_win_error("BC direct IO CreateFileW failed"));
    }
    return BCWinHandle(handle);
}

inline void set_direct_file_size(HANDLE handle, uint64_t bytes) {
    LARGE_INTEGER pos;
    pos.QuadPart = static_cast<LONGLONG>(bytes);
    if (!SetFilePointerEx(handle, pos, nullptr, FILE_BEGIN)) {
        throw std::runtime_error(bc_direct_win_error("BC direct IO SetFilePointerEx resize failed"));
    }
    if (!SetEndOfFile(handle)) {
        throw std::runtime_error(bc_direct_win_error("BC direct IO SetEndOfFile failed"));
    }
}

inline void direct_read_sync(HANDLE handle, uint64_t offset, void *data, uint64_t bytes) {
    if (bytes > static_cast<uint64_t>(std::numeric_limits<DWORD>::max())) {
        throw std::overflow_error("BC direct IO sync read exceeds DWORD bytes");
    }
    LARGE_INTEGER pos;
    pos.QuadPart = static_cast<LONGLONG>(offset);
    if (!SetFilePointerEx(handle, pos, nullptr, FILE_BEGIN)) {
        throw std::runtime_error(bc_direct_win_error("BC direct IO SetFilePointerEx read failed"));
    }
    DWORD read = 0U;
    if (!ReadFile(handle, data, static_cast<DWORD>(bytes), &read, nullptr) ||
        read != static_cast<DWORD>(bytes)) {
        throw std::runtime_error(bc_direct_win_error("BC direct IO ReadFile failed"));
    }
}

inline void direct_write_sync(HANDLE handle, uint64_t offset, const void *data, uint64_t bytes) {
    if (bytes > static_cast<uint64_t>(std::numeric_limits<DWORD>::max())) {
        throw std::overflow_error("BC direct IO sync write exceeds DWORD bytes");
    }
    LARGE_INTEGER pos;
    pos.QuadPart = static_cast<LONGLONG>(offset);
    if (!SetFilePointerEx(handle, pos, nullptr, FILE_BEGIN)) {
        throw std::runtime_error(bc_direct_win_error("BC direct IO SetFilePointerEx write failed"));
    }
    DWORD written = 0U;
    if (!WriteFile(handle, data, static_cast<DWORD>(bytes), &written, nullptr) ||
        written != static_cast<DWORD>(bytes)) {
        throw std::runtime_error(bc_direct_win_error("BC direct IO WriteFile failed"));
    }
}

struct BCPhysicalRange {
    uint64_t offset = 0U;
    uint64_t bytes = 0U;
    BCAlignedBuffer buffer;

    [[nodiscard]] uint64_t end() const {
        return offset + bytes;
    }
};

inline void add_backend_stats(BCFileIOStats *stats, uint64_t io_count, uint64_t bytes) {
    if (stats == nullptr) {
        return;
    }
    if (stats->backend_io_count > std::numeric_limits<uint64_t>::max() - io_count ||
        stats->backend_bytes > std::numeric_limits<uint64_t>::max() - bytes) {
        throw std::overflow_error("BC direct IO backend stats overflow");
    }
    stats->backend_io_count += io_count;
    stats->backend_bytes += bytes;
}

} // namespace detail

class BCDirectFileReader final : public BCReadableFile {
public:
    explicit BCDirectFileReader(
        const std::filesystem::path &path,
        const BCDirectFileIOOptions &options = BCDirectFileIOOptions{}
    )
        : path_(path), options_(detail::normalize_direct_options(options)) {
        handle_ = detail::open_direct_handle(path_, false, options_.overlapped);
        std::error_code ec;
        physical_size_ = std::filesystem::file_size(path_, ec);
        if (ec) {
            throw std::runtime_error("BC direct reader file_size failed: " + ec.message());
        }
        logical_size_ = options_.logical_size.value_or(physical_size_);
        if (logical_size_ > physical_size_) {
            throw std::runtime_error("BC direct reader logical size exceeds physical file size");
        }
        const uint64_t required_physical = bc_direct_align_up(logical_size_, options_.alignment);
        if (required_physical > physical_size_) {
            throw std::runtime_error("BC direct reader requires padded physical file size");
        }
    }

    void read_at(uint64_t offset, void *data, uint64_t bytes) const override {
        read_many(std::vector<BCFileReadRequest>{BCFileReadRequest{offset, data, bytes}});
    }

    [[nodiscard]] BCFileIOMode mode() const override {
        return BCFileIOMode::Direct;
    }

    void read_many(
        const std::vector<BCFileReadRequest> &requests,
        BCFileIOStats *stats = nullptr
    ) const override {
        if (stats != nullptr) {
            *stats = {};
        }
        refresh_size_if_growing();
        if (requests.empty()) {
            return;
        }

        if (options_.overlapped && options_.queue_depth > 1U) {
            read_many_overlapped_pipeline(requests, stats);
            return;
        }
        if (try_read_many_direct_aligned_sync(requests, stats)) {
            return;
        }

        std::vector<detail::BCPhysicalRange> ranges;
        ranges.reserve(requests.size());
        for (const BCFileReadRequest &request : requests) {
            if (stats != nullptr) {
                ++stats->request_count;
                if (stats->requested_bytes > std::numeric_limits<uint64_t>::max() - request.bytes) {
                    throw std::overflow_error("BC direct read requested byte stats overflow");
                }
                stats->requested_bytes += request.bytes;
            }
            if (request.bytes == 0U) {
                continue;
            }
            if (request.data == nullptr) {
                throw std::invalid_argument("BC direct read request data pointer is null");
            }
            if (request.offset > logical_size_ || request.bytes > logical_size_ - request.offset) {
                throw std::out_of_range("BC direct read request exceeds logical file size");
            }
            const uint64_t physical_offset = bc_direct_align_down(request.offset, options_.alignment);
            const uint64_t physical_end = bc_direct_align_up(
                request.offset + request.bytes,
                options_.alignment
            );
            if (physical_end > physical_size_) {
                throw std::out_of_range("BC direct read request exceeds physical file size");
            }
            ranges.push_back(detail::BCPhysicalRange{physical_offset, physical_end - physical_offset, {}});
        }
        if (ranges.empty()) {
            return;
        }

        std::sort(ranges.begin(), ranges.end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.offset != rhs.offset) {
                return lhs.offset < rhs.offset;
            }
            return lhs.bytes < rhs.bytes;
        });
        std::vector<detail::BCPhysicalRange> merged;
        for (detail::BCPhysicalRange &range : ranges) {
            if (merged.empty() || range.offset > merged.back().end()) {
                merged.push_back(detail::BCPhysicalRange{range.offset, range.bytes, {}});
                continue;
            }
            detail::BCPhysicalRange &last = merged.back();
            const uint64_t end = std::max(last.end(), range.end());
            last.bytes = end - last.offset;
        }

        for (detail::BCPhysicalRange &range : merged) {
            range.buffer.reset(range.bytes, options_.alignment);
        }
        read_physical_ranges(merged, stats);
        copy_requests_from_ranges(requests, merged);
    }

    [[nodiscard]] uint64_t size() const override {
        refresh_size_if_growing();
        return logical_size_;
    }

private:
    void refresh_size_if_growing() const {
        if (options_.logical_size.has_value()) {
            return;
        }
        std::error_code ec;
        const uint64_t current = std::filesystem::file_size(path_, ec);
        if (ec) {
            throw std::runtime_error("BC direct reader file_size refresh failed: " + ec.message());
        }
        physical_size_ = current;
        logical_size_ = current;
    }

    void read_physical_ranges(
        std::vector<detail::BCPhysicalRange> &ranges,
        BCFileIOStats *stats
    ) const {
        if (ranges.empty()) {
            return;
        }
        const uint32_t worker_count = static_cast<uint32_t>(
            std::min<uint64_t>(ranges.size(), std::max<uint32_t>(1U, options_.queue_depth))
        );
        uint64_t bytes_sum = 0U;
        for (const detail::BCPhysicalRange &range : ranges) {
            if (bytes_sum > std::numeric_limits<uint64_t>::max() - range.bytes) {
                throw std::overflow_error("BC direct read backend byte stats overflow");
            }
            bytes_sum += range.bytes;
        }
        if (!options_.overlapped || options_.queue_depth <= 1U) {
            for (detail::BCPhysicalRange &range : ranges) {
                detail::direct_read_sync(handle_.get(), range.offset, range.buffer.data(), range.bytes);
            }
            detail::add_backend_stats(stats, ranges.size(), bytes_sum);
            return;
        }
        read_physical_ranges_overlapped(ranges, worker_count);
        detail::add_backend_stats(stats, ranges.size(), bytes_sum);
    }

    [[nodiscard]] bool try_read_many_direct_aligned_sync(
        const std::vector<BCFileReadRequest> &requests,
        BCFileIOStats *stats
    ) const {
        uint64_t requested_bytes = 0U;
        size_t non_empty = 0U;
        for (const BCFileReadRequest &request : requests) {
            if (request.bytes == 0U) {
                continue;
            }
            if (request.data == nullptr) {
                throw std::invalid_argument("BC direct read request data pointer is null");
            }
            if (request.offset > logical_size_ || request.bytes > logical_size_ - request.offset) {
                throw std::out_of_range("BC direct read request exceeds logical file size");
            }
            if ((request.offset & (options_.alignment - 1U)) != 0U ||
                (request.bytes & (options_.alignment - 1U)) != 0U ||
                (reinterpret_cast<uintptr_t>(request.data) & (options_.alignment - 1U)) != 0U) {
                return false;
            }
            if (request.offset + request.bytes > physical_size_) {
                throw std::out_of_range("BC direct aligned read request exceeds physical file size");
            }
            if (requested_bytes > std::numeric_limits<uint64_t>::max() - request.bytes) {
                throw std::overflow_error("BC direct aligned read requested byte stats overflow");
            }
            requested_bytes += request.bytes;
            ++non_empty;
        }
        if (non_empty == 0U) {
            return true;
        }
        for (const BCFileReadRequest &request : requests) {
            if (request.bytes == 0U) {
                continue;
            }
            detail::direct_read_sync(handle_.get(), request.offset, request.data, request.bytes);
        }
        if (stats != nullptr) {
            stats->request_count += non_empty;
            stats->requested_bytes += requested_bytes;
            stats->backend_io_count += non_empty;
            stats->backend_bytes += requested_bytes;
        }
        return true;
    }

    void read_many_parallel_sync(
        const std::vector<BCFileReadRequest> &requests,
        BCFileIOStats *stats
    ) const {
        const uint32_t worker_count = static_cast<uint32_t>(
            std::min<uint64_t>(options_.queue_depth, requests.size())
        );
        if (worker_count <= 1U) {
            BCDirectFileIOOptions local_options = options_;
            local_options.overlapped = false;
            local_options.queue_depth = 1U;
            BCDirectFileReader reader(path_, local_options);
            reader.read_many(requests, stats);
            return;
        }

        std::vector<std::vector<BCFileReadRequest>> shards(worker_count);
        const size_t chunk = (requests.size() + worker_count - 1U) / worker_count;
        size_t cursor = 0U;
        for (uint32_t worker = 0U; worker < worker_count && cursor < requests.size(); ++worker) {
            const size_t end = std::min<size_t>(requests.size(), cursor + chunk);
            shards[worker].assign(requests.begin() + static_cast<std::ptrdiff_t>(cursor),
                                  requests.begin() + static_cast<std::ptrdiff_t>(end));
            cursor = end;
        }

        std::vector<BCFileIOStats> local_stats(worker_count);
        std::vector<std::exception_ptr> errors(worker_count);
        std::vector<std::thread> threads;
        threads.reserve(worker_count);
        for (uint32_t worker = 0U; worker < worker_count; ++worker) {
            if (shards[worker].empty()) {
                continue;
            }
            threads.emplace_back([&, worker] {
                try {
                    BCDirectFileIOOptions local_options = options_;
                    local_options.overlapped = false;
                    local_options.queue_depth = 1U;
                    BCDirectFileReader reader(path_, local_options);
                    reader.read_many(shards[worker], &local_stats[worker]);
                } catch (...) {
                    errors[worker] = std::current_exception();
                }
            });
        }
        for (std::thread &thread : threads) {
            thread.join();
        }
        for (const std::exception_ptr &error : errors) {
            if (error) {
                std::rethrow_exception(error);
            }
        }

        if (stats != nullptr) {
            for (const BCFileIOStats &local : local_stats) {
                if (stats->request_count > std::numeric_limits<uint64_t>::max() - local.request_count ||
                    stats->requested_bytes > std::numeric_limits<uint64_t>::max() - local.requested_bytes ||
                    stats->backend_io_count > std::numeric_limits<uint64_t>::max() - local.backend_io_count ||
                    stats->backend_bytes > std::numeric_limits<uint64_t>::max() - local.backend_bytes) {
                    throw std::overflow_error("BC direct parallel read stats overflow");
                }
                stats->request_count += local.request_count;
                stats->requested_bytes += local.requested_bytes;
                stats->backend_io_count += local.backend_io_count;
                stats->backend_bytes += local.backend_bytes;
            }
        }
    }

    struct PendingRead {
        OVERLAPPED ov = {};
        detail::BCWinHandle event;
        detail::BCAlignedBuffer buffer;
        uint64_t physical_offset = 0U;
        uint64_t physical_bytes = 0U;
        uint64_t in_physical_offset = 0U;
        void *target = nullptr;
        uint64_t target_bytes = 0U;
        bool direct_to_target = false;
        DWORD immediate_bytes = 0U;
        size_t range_index = 0U;
    };

    struct DirectLogicalRead {
        uint64_t physical_offset = 0U;
        uint64_t physical_bytes = 0U;
        uint64_t in_physical_offset = 0U;
        void *target = nullptr;
        uint64_t target_bytes = 0U;
        bool direct_to_target = false;
    };

    void read_many_overlapped_pipeline(
        const std::vector<BCFileReadRequest> &requests,
        BCFileIOStats *stats
    ) const {
        std::vector<DirectLogicalRead> reads;
        reads.reserve(requests.size());
        uint64_t backend_bytes = 0U;
        for (const BCFileReadRequest &request : requests) {
            if (stats != nullptr) {
                ++stats->request_count;
                if (stats->requested_bytes > std::numeric_limits<uint64_t>::max() - request.bytes) {
                    throw std::overflow_error("BC direct read requested byte stats overflow");
                }
                stats->requested_bytes += request.bytes;
            }
            if (request.bytes == 0U) {
                continue;
            }
            if (request.data == nullptr) {
                throw std::invalid_argument("BC direct read request data pointer is null");
            }
            if (request.offset > logical_size_ || request.bytes > logical_size_ - request.offset) {
                throw std::out_of_range("BC direct read request exceeds logical file size");
            }
            const uint64_t physical_offset = bc_direct_align_down(request.offset, options_.alignment);
            const uint64_t physical_end = bc_direct_align_up(
                request.offset + request.bytes,
                options_.alignment
            );
            if (physical_end > physical_size_) {
                throw std::out_of_range("BC direct read request exceeds physical file size");
            }
            const uint64_t physical_bytes = physical_end - physical_offset;
            if (backend_bytes > std::numeric_limits<uint64_t>::max() - physical_bytes) {
                throw std::overflow_error("BC direct read backend byte stats overflow");
            }
            backend_bytes += physical_bytes;
            const bool direct_to_target =
                physical_offset == request.offset &&
                physical_bytes == request.bytes &&
                (reinterpret_cast<uintptr_t>(request.data) & (options_.alignment - 1U)) == 0U;
            reads.push_back(DirectLogicalRead{
                physical_offset,
                physical_bytes,
                request.offset - physical_offset,
                request.data,
                request.bytes,
                direct_to_target
            });
        }
        if (reads.empty()) {
            return;
        }

        const uint32_t queue_depth = static_cast<uint32_t>(
            std::min<uint64_t>(options_.queue_depth, reads.size())
        );
        std::vector<PendingRead> pending(queue_depth);
        for (PendingRead &request : pending) {
            request.event = detail::BCWinHandle(CreateEventW(nullptr, TRUE, FALSE, nullptr));
            if (!request.event.valid()) {
                throw std::runtime_error(detail::bc_direct_win_error("BC direct read CreateEventW failed"));
            }
        }

        std::vector<uint8_t> active(queue_depth, 0U);
        auto submit_slot = [&](uint32_t slot, size_t read_index) {
            const DirectLogicalRead &read = reads[read_index];
            PendingRead &request = pending[slot];
            if (!read.direct_to_target) {
                request.buffer.ensure_at_least(read.physical_bytes, options_.alignment);
            }
            ResetEvent(request.event.get());
            request.ov = {};
            request.ov.Offset = static_cast<DWORD>(read.physical_offset & 0xFFFFFFFFULL);
            request.ov.OffsetHigh = static_cast<DWORD>((read.physical_offset >> 32U) & 0xFFFFFFFFULL);
            request.ov.hEvent = request.event.get();
            request.physical_offset = read.physical_offset;
            request.physical_bytes = read.physical_bytes;
            request.in_physical_offset = read.in_physical_offset;
            request.target = read.target;
            request.target_bytes = read.target_bytes;
            request.direct_to_target = read.direct_to_target;
            const void *read_buffer = read.direct_to_target ? read.target : request.buffer.data();
            request.immediate_bytes = 0U;
            if (read.physical_bytes > static_cast<uint64_t>(std::numeric_limits<DWORD>::max())) {
                throw std::overflow_error("BC direct overlapped read exceeds DWORD bytes");
            }
            const BOOL ok = ReadFile(
                handle_.get(),
                const_cast<void *>(read_buffer),
                static_cast<DWORD>(read.physical_bytes),
                &request.immediate_bytes,
                &request.ov
            );
            if (!ok) {
                const DWORD error = GetLastError();
                if (error != ERROR_IO_PENDING) {
                    throw std::runtime_error(detail::bc_direct_win_error("BC direct overlapped ReadFile failed", error));
                }
            }
            active[slot] = 1U;
        };
        auto wait_any_slot = [&]() -> uint32_t {
            std::vector<HANDLE> events;
            std::vector<uint32_t> slots;
            events.reserve(queue_depth);
            slots.reserve(queue_depth);
            for (uint32_t slot = 0U; slot < queue_depth; ++slot) {
                if (active[slot] == 0U) {
                    continue;
                }
                events.push_back(pending[slot].event.get());
                slots.push_back(slot);
            }
            if (events.empty()) {
                throw std::logic_error("BC direct overlapped read has no active request");
            }
            const DWORD wait = WaitForMultipleObjects(
                static_cast<DWORD>(events.size()),
                events.data(),
                FALSE,
                INFINITE
            );
            if (wait < WAIT_OBJECT_0 || wait >= WAIT_OBJECT_0 + events.size()) {
                throw std::runtime_error(detail::bc_direct_win_error("BC direct overlapped WaitForMultipleObjects failed"));
            }
            return slots[wait - WAIT_OBJECT_0];
        };

        size_t next_read = 0U;
        size_t completed = 0U;
        for (uint32_t slot = 0U; slot < queue_depth && next_read < reads.size(); ++slot) {
            submit_slot(slot, next_read++);
        }
        while (completed < reads.size()) {
            const uint32_t slot = wait_any_slot();
            PendingRead &request = pending[slot];
            DWORD transferred = 0U;
            if (!GetOverlappedResult(handle_.get(), &request.ov, &transferred, TRUE)) {
                throw std::runtime_error(detail::bc_direct_win_error("BC direct overlapped GetOverlappedResult failed"));
            }
            if (transferred != static_cast<DWORD>(request.physical_bytes)) {
                throw std::runtime_error("BC direct overlapped read transferred unexpected byte count");
            }
            if (!request.direct_to_target) {
                std::memcpy(
                    request.target,
                    request.buffer.data() + static_cast<size_t>(request.in_physical_offset),
                    static_cast<size_t>(request.target_bytes)
                );
            }
            ++completed;
            if (next_read < reads.size()) {
                submit_slot(slot, next_read++);
            } else {
                active[slot] = 0U;
            }
        }
        detail::add_backend_stats(stats, reads.size(), backend_bytes);
    }

    void read_physical_ranges_overlapped(
        std::vector<detail::BCPhysicalRange> &ranges,
        uint32_t queue_depth
    ) const {
        std::vector<PendingRead> pending(queue_depth);
        for (PendingRead &request : pending) {
            request.event = detail::BCWinHandle(CreateEventW(nullptr, TRUE, FALSE, nullptr));
            if (!request.event.valid()) {
                throw std::runtime_error(detail::bc_direct_win_error("BC direct read CreateEventW failed"));
            }
        }

        size_t cursor = 0U;
        while (cursor < ranges.size()) {
            const uint32_t batch = static_cast<uint32_t>(
                std::min<uint64_t>(queue_depth, ranges.size() - cursor)
            );
            for (uint32_t i = 0U; i < batch; ++i) {
                PendingRead &request = pending[i];
                detail::BCPhysicalRange &range = ranges[cursor + i];
                ResetEvent(request.event.get());
                request.ov = {};
                request.ov.Offset = static_cast<DWORD>(range.offset & 0xFFFFFFFFULL);
                request.ov.OffsetHigh = static_cast<DWORD>((range.offset >> 32U) & 0xFFFFFFFFULL);
                request.ov.hEvent = request.event.get();
                request.range_index = cursor + i;
                if (range.bytes > static_cast<uint64_t>(std::numeric_limits<DWORD>::max())) {
                    throw std::overflow_error("BC direct overlapped read exceeds DWORD bytes");
                }
                    const BOOL ok = ReadFile(
                        handle_.get(),
                        range.buffer.data(),
                        static_cast<DWORD>(range.bytes),
                        &request.immediate_bytes,
                        &request.ov
                    );
                if (!ok) {
                    const DWORD error = GetLastError();
                    if (error != ERROR_IO_PENDING) {
                        throw std::runtime_error(detail::bc_direct_win_error("BC direct overlapped ReadFile failed", error));
                    }
                }
            }
            std::vector<HANDLE> events;
            events.reserve(batch);
            for (uint32_t i = 0U; i < batch; ++i) {
                events.push_back(pending[i].event.get());
            }
            const DWORD wait_result = WaitForMultipleObjects(
                batch,
                events.data(),
                TRUE,
                INFINITE
            );
            if (wait_result < WAIT_OBJECT_0 || wait_result >= WAIT_OBJECT_0 + batch) {
                throw std::runtime_error(detail::bc_direct_win_error("BC direct overlapped WaitForMultipleObjects failed"));
            }
            for (uint32_t i = 0U; i < batch; ++i) {
                PendingRead &request = pending[i];
                const detail::BCPhysicalRange &range = ranges[request.range_index];
                DWORD transferred = 0U;
                    if (!GetOverlappedResult(handle_.get(), &request.ov, &transferred, TRUE)) {
                        throw std::runtime_error(detail::bc_direct_win_error("BC direct overlapped GetOverlappedResult failed"));
                    }
                if (transferred != static_cast<DWORD>(range.bytes)) {
                    throw std::runtime_error("BC direct overlapped read transferred unexpected byte count");
                }
            }
            cursor += batch;
        }
    }

    static void copy_requests_from_ranges(
        const std::vector<BCFileReadRequest> &requests,
        const std::vector<detail::BCPhysicalRange> &ranges
    ) {
        for (const BCFileReadRequest &request : requests) {
            if (request.bytes == 0U) {
                continue;
            }
            uint64_t copied = 0U;
            while (copied < request.bytes) {
                const uint64_t cursor = request.offset + copied;
                const auto it = std::upper_bound(
                    ranges.begin(),
                    ranges.end(),
                    cursor,
                    [](uint64_t value, const detail::BCPhysicalRange &range) {
                        return value < range.offset;
                    }
                );
                if (it == ranges.begin()) {
                    throw std::logic_error("BC direct read request is before loaded physical ranges");
                }
                const detail::BCPhysicalRange &range = *(it - 1);
                if (cursor < range.offset || cursor >= range.end()) {
                    throw std::logic_error("BC direct read request is not covered by loaded range");
                }
                const uint64_t in_range = cursor - range.offset;
                const uint64_t take = std::min<uint64_t>(request.bytes - copied, range.bytes - in_range);
                std::memcpy(
                    static_cast<uint8_t *>(request.data) + copied,
                    range.buffer.data() + static_cast<size_t>(in_range),
                    static_cast<size_t>(take)
                );
                copied += take;
            }
        }
    }

    std::filesystem::path path_;
    BCDirectFileIOOptions options_;
    detail::BCWinHandle handle_;
    mutable uint64_t logical_size_ = 0U;
    mutable uint64_t physical_size_ = 0U;
};

// Direct reader wrapper for append/read phase separation. It opens a fresh
// BCDirectFileReader for each operation, so append-only writers may grow the
// file between read phases without sharing a stale no-buffering handle.
class BCLazyDirectFileReader final : public BCReadableFile {
public:
    explicit BCLazyDirectFileReader(
        std::filesystem::path path,
        const BCDirectFileIOOptions &options = BCDirectFileIOOptions{}
    ) : path_(std::move(path)), options_(detail::normalize_direct_options(options)) {}

    void read_at(uint64_t offset, void *data, uint64_t bytes) const override {
        BCDirectFileIOOptions local_options = options_;
        local_options.overlapped = false;
        local_options.queue_depth = 1U;
        auto reader = std::make_unique<BCDirectFileReader>(path_, local_options);
        reader->read_at(offset, data, bytes);
    }

    void read_many(
        const std::vector<BCFileReadRequest> &requests,
        BCFileIOStats *stats = nullptr
    ) const override {
        if (stats != nullptr) {
            *stats = {};
        }
        if (requests.empty()) {
            return;
        }
        const uint32_t worker_count = static_cast<uint32_t>(
            std::min<uint64_t>(std::max<uint32_t>(1U, options_.queue_depth), requests.size())
        );
        if (worker_count <= 1U) {
            BCDirectFileIOOptions local_options = options_;
            local_options.overlapped = false;
            local_options.queue_depth = 1U;
            auto reader = std::make_unique<BCDirectFileReader>(path_, local_options);
            reader->read_many(requests, stats);
            return;
        }

        std::vector<std::vector<BCFileReadRequest>> shards(worker_count);
        const size_t chunk = (requests.size() + worker_count - 1U) / worker_count;
        size_t cursor = 0U;
        for (uint32_t worker = 0U; worker < worker_count && cursor < requests.size(); ++worker) {
            const size_t end = std::min<size_t>(requests.size(), cursor + chunk);
            shards[worker].assign(
                requests.begin() + static_cast<std::ptrdiff_t>(cursor),
                requests.begin() + static_cast<std::ptrdiff_t>(end)
            );
            cursor = end;
        }

        std::vector<BCFileIOStats> local_stats(worker_count);
        std::vector<std::exception_ptr> errors(worker_count);
        std::vector<std::thread> threads;
        threads.reserve(worker_count);
        for (uint32_t worker = 0U; worker < worker_count; ++worker) {
            if (shards[worker].empty()) {
                continue;
            }
            threads.emplace_back([&, worker] {
                try {
                    BCDirectFileIOOptions local_options = options_;
                    local_options.overlapped = false;
                    local_options.queue_depth = 1U;
                    BCDirectFileReader reader(path_, local_options);
                    reader.read_many(shards[worker], &local_stats[worker]);
                } catch (...) {
                    errors[worker] = std::current_exception();
                }
            });
        }
        for (std::thread &thread : threads) {
            thread.join();
        }
        for (const std::exception_ptr &error : errors) {
            if (error) {
                std::rethrow_exception(error);
            }
        }
        if (stats != nullptr) {
            for (const BCFileIOStats &local : local_stats) {
                stats->request_count += local.request_count;
                stats->requested_bytes += local.requested_bytes;
                stats->backend_io_count += local.backend_io_count;
                stats->backend_bytes += local.backend_bytes;
            }
        }
    }

    [[nodiscard]] BCFileIOMode mode() const override {
        return BCFileIOMode::Direct;
    }

    [[nodiscard]] uint64_t size() const override {
        BCDirectFileIOOptions local_options = options_;
        local_options.overlapped = false;
        local_options.queue_depth = 1U;
        auto reader = std::make_unique<BCDirectFileReader>(path_, local_options);
        return reader->size();
    }

private:
    std::filesystem::path path_;
    BCDirectFileIOOptions options_;
};

class BCDirectFileWriter final : public BCWritableFile {
    struct PendingWriteView {
        uint64_t offset = 0U;
        const uint8_t *data = nullptr;
        uint64_t bytes = 0U;
    };

public:
    // Default mode does not preserve unwritten bytes inside aligned physical
    // blocks; it is intended for new-file stream writes or planned full writes
    // where the caller covers all logical bytes exactly once. Set
    // preserve_unwritten_bytes for append-style streams that may update an
    // existing unaligned tail block; that mode performs synchronous RMW and is
    // slower.
    explicit BCDirectFileWriter(
        const std::filesystem::path &path,
        const BCDirectFileIOOptions &options = BCDirectFileIOOptions{}
    )
        : path_(path), options_(detail::normalize_direct_options(options)) {
        handle_ = detail::open_direct_handle(path_, true, options_.overlapped);
    }

    void write_at(uint64_t offset, const void *data, uint64_t bytes) override {
        write_many(std::vector<BCFileWriteRequest>{BCFileWriteRequest{offset, data, bytes}});
    }

    [[nodiscard]] BCFileIOMode mode() const override {
        return BCFileIOMode::Direct;
    }

    void write_many(
        const std::vector<BCFileWriteRequest> &requests,
        BCFileIOStats *stats = nullptr
    ) override {
        if (stats != nullptr) {
            *stats = {};
        }
        if (options_.overlapped &&
            options_.queue_depth > 1U &&
            !options_.preserve_unwritten_bytes) {
            write_many_overlapped_pipeline(requests, stats);
            return;
        }
        std::vector<PendingWriteView> pending;
        pending.reserve(requests.size());
        std::vector<detail::BCPhysicalRange> ranges;
        ranges.reserve(requests.size());
        for (const BCFileWriteRequest &request : requests) {
            if (stats != nullptr) {
                ++stats->request_count;
                if (stats->requested_bytes > std::numeric_limits<uint64_t>::max() - request.bytes) {
                    throw std::overflow_error("BC direct write requested byte stats overflow");
                }
                stats->requested_bytes += request.bytes;
            }
            if (request.bytes == 0U) {
                continue;
            }
            if (request.data == nullptr) {
                throw std::invalid_argument("BC direct write request data pointer is null");
            }
            const uint64_t request_end = request.offset + request.bytes;
            if (request_end < request.offset) {
                throw std::overflow_error("BC direct write request end overflow");
            }
            const uint64_t physical_offset = bc_direct_align_down(request.offset, options_.alignment);
            const uint64_t physical_end = bc_direct_align_up(request_end, options_.alignment);
            pending.push_back(PendingWriteView{
                request.offset,
                static_cast<const uint8_t *>(request.data),
                request.bytes
            });
            ranges.push_back(detail::BCPhysicalRange{physical_offset, physical_end - physical_offset, {}});
        }
        if (pending.empty()) {
            return;
        }

        std::sort(ranges.begin(), ranges.end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.offset != rhs.offset) {
                return lhs.offset < rhs.offset;
            }
            return lhs.bytes < rhs.bytes;
        });
        std::vector<detail::BCPhysicalRange> merged;
        for (detail::BCPhysicalRange &range : ranges) {
            if (merged.empty() || range.offset > merged.back().end()) {
                merged.push_back(detail::BCPhysicalRange{range.offset, range.bytes, {}});
                continue;
            }
            detail::BCPhysicalRange &last = merged.back();
            const uint64_t end = std::max(last.end(), range.end());
            last.bytes = end - last.offset;
        }

        const uint64_t old_physical_size = physical_size_;
        uint64_t max_physical_end = physical_size_;
        for (detail::BCPhysicalRange &range : merged) {
            max_physical_end = std::max(max_physical_end, range.end());
        }
        if (max_physical_end > physical_size_) {
            physical_size_ = max_physical_end;
            detail::set_direct_file_size(handle_.get(), physical_size_);
        }

        for (detail::BCPhysicalRange &range : merged) {
            range.buffer.reset(range.bytes, options_.alignment);
            if (options_.preserve_unwritten_bytes && range.offset < old_physical_size) {
                const uint64_t readable = std::min<uint64_t>(range.bytes, old_physical_size - range.offset);
                if (readable != 0U) {
                    detail::direct_read_sync(handle_.get(), range.offset, range.buffer.data(), readable);
                }
                if (readable < range.bytes) {
                    std::memset(
                        range.buffer.data() + static_cast<size_t>(readable),
                        0,
                        static_cast<size_t>(range.bytes - readable)
                    );
                }
            } else {
                std::memset(range.buffer.data(), 0, range.buffer.size());
            }
        }
        for (const PendingWriteView &request : pending) {
            logical_size_ = std::max(logical_size_, request.offset + request.bytes);
            copy_pending_request_to_ranges(request, merged);
        }
        for (detail::BCPhysicalRange &range : merged) {
            detail::direct_write_sync(handle_.get(), range.offset, range.buffer.data(), range.bytes);
            detail::add_backend_stats(stats, 1U, range.bytes);
        }
    }

    void resize(uint64_t bytes) override {
        logical_size_ = bytes;
        physical_size_ = bc_direct_align_up(bytes, options_.alignment);
        detail::set_direct_file_size(handle_.get(), physical_size_);
    }

    void flush() override {
        if (!FlushFileBuffers(handle_.get())) {
            throw std::runtime_error(detail::bc_direct_win_error("BC direct writer FlushFileBuffers failed"));
        }
    }

private:
    struct PendingWrite {
        OVERLAPPED ov = {};
        detail::BCWinHandle event;
        detail::BCAlignedBuffer buffer;
        const uint8_t *external_data = nullptr;
        DWORD immediate_bytes = 0U;
        uint64_t physical_bytes = 0U;
        size_t write_index = 0U;
    };

    void write_many_overlapped_pipeline(
        const std::vector<BCFileWriteRequest> &requests,
        BCFileIOStats *stats
    ) {
        uint64_t backend_bytes = 0U;
        uint64_t max_physical_end = physical_size_;
        size_t non_empty_count = 0U;
        for (size_t i = 0U; i < requests.size(); ++i) {
            const BCFileWriteRequest &request = requests[i];
            if (stats != nullptr) {
                ++stats->request_count;
                if (stats->requested_bytes > std::numeric_limits<uint64_t>::max() - request.bytes) {
                    throw std::overflow_error("BC direct write requested byte stats overflow");
                }
                stats->requested_bytes += request.bytes;
            }
            if (request.bytes == 0U) {
                continue;
            }
            if (request.data == nullptr) {
                throw std::invalid_argument("BC direct write request data pointer is null");
            }
            const uint64_t request_end = request.offset + request.bytes;
            if (request_end < request.offset) {
                throw std::overflow_error("BC direct write request end overflow");
            }
            const uint64_t physical_offset = bc_direct_align_down(request.offset, options_.alignment);
            const uint64_t physical_end = bc_direct_align_up(request_end, options_.alignment);
            const uint64_t physical_bytes = physical_end - physical_offset;
            if (backend_bytes > std::numeric_limits<uint64_t>::max() - physical_bytes) {
                throw std::overflow_error("BC direct write backend byte stats overflow");
            }
            backend_bytes += physical_bytes;
            max_physical_end = std::max(max_physical_end, physical_end);
            logical_size_ = std::max(logical_size_, request_end);
            ++non_empty_count;
        }
        if (non_empty_count == 0U) {
            return;
        }
        if (max_physical_end > physical_size_) {
            physical_size_ = max_physical_end;
            detail::set_direct_file_size(handle_.get(), physical_size_);
        }

        const uint32_t queue_depth = static_cast<uint32_t>(
            std::min<uint64_t>(options_.queue_depth, non_empty_count)
        );
        std::vector<PendingWrite> pending(queue_depth);
        for (PendingWrite &request : pending) {
            request.event = detail::BCWinHandle(CreateEventW(nullptr, TRUE, FALSE, nullptr));
            if (!request.event.valid()) {
                throw std::runtime_error(detail::bc_direct_win_error("BC direct write CreateEventW failed"));
            }
        }
        std::vector<uint8_t> active(queue_depth, 0U);

        auto submit_slot = [&](uint32_t slot, size_t write_index) {
            const BCFileWriteRequest &write = requests[write_index];
            const uint64_t request_end = write.offset + write.bytes;
            const uint64_t physical_offset = bc_direct_align_down(write.offset, options_.alignment);
            const uint64_t physical_end = bc_direct_align_up(request_end, options_.alignment);
            const uint64_t physical_bytes = physical_end - physical_offset;
            const uint64_t in_physical_offset = write.offset - physical_offset;
            PendingWrite &request = pending[slot];
            const bool use_external_buffer =
                !options_.preserve_unwritten_bytes &&
                in_physical_offset == 0U &&
                physical_bytes == write.bytes &&
                (reinterpret_cast<uintptr_t>(write.data) & (options_.alignment - 1U)) == 0U;
            if (use_external_buffer) {
                request.external_data = static_cast<const uint8_t *>(write.data);
            } else {
                request.external_data = nullptr;
                request.buffer.ensure_at_least(physical_bytes, options_.alignment);
                std::memset(request.buffer.data(), 0, request.buffer.size());
                std::memcpy(
                    request.buffer.data() + static_cast<size_t>(in_physical_offset),
                    write.data,
                    static_cast<size_t>(write.bytes)
                );
            }
            ResetEvent(request.event.get());
            request.ov = {};
            request.ov.Offset = static_cast<DWORD>(physical_offset & 0xFFFFFFFFULL);
            request.ov.OffsetHigh = static_cast<DWORD>((physical_offset >> 32U) & 0xFFFFFFFFULL);
            request.ov.hEvent = request.event.get();
            request.write_index = write_index;
            request.physical_bytes = physical_bytes;
            request.immediate_bytes = 0U;
            if (physical_bytes > static_cast<uint64_t>(std::numeric_limits<DWORD>::max())) {
                throw std::overflow_error("BC direct overlapped write exceeds DWORD bytes");
            }
            const BOOL ok = WriteFile(
                handle_.get(),
                use_external_buffer ? request.external_data : request.buffer.data(),
                static_cast<DWORD>(physical_bytes),
                &request.immediate_bytes,
                &request.ov
            );
            if (!ok) {
                const DWORD error = GetLastError();
                if (error != ERROR_IO_PENDING) {
                    throw std::runtime_error(detail::bc_direct_win_error("BC direct overlapped WriteFile failed", error));
                }
            }
            active[slot] = 1U;
        };

        auto wait_any_slot = [&]() -> uint32_t {
            std::vector<HANDLE> events;
            std::vector<uint32_t> slots;
            events.reserve(queue_depth);
            slots.reserve(queue_depth);
            for (uint32_t slot = 0U; slot < queue_depth; ++slot) {
                if (active[slot] == 0U) {
                    continue;
                }
                events.push_back(pending[slot].event.get());
                slots.push_back(slot);
            }
            if (events.empty()) {
                throw std::logic_error("BC direct overlapped write has no active request");
            }
            const DWORD wait = WaitForMultipleObjects(
                static_cast<DWORD>(events.size()),
                events.data(),
                FALSE,
                INFINITE
            );
            if (wait < WAIT_OBJECT_0 || wait >= WAIT_OBJECT_0 + events.size()) {
                throw std::runtime_error(detail::bc_direct_win_error("BC direct overlapped WaitForMultipleObjects failed"));
            }
            return slots[wait - WAIT_OBJECT_0];
        };

        auto next_non_empty = [&](size_t cursor) {
            while (cursor < requests.size() && requests[cursor].bytes == 0U) {
                ++cursor;
            }
            return cursor;
        };
        size_t next_write = 0U;
        size_t completed = 0U;
        next_write = next_non_empty(next_write);
        for (uint32_t slot = 0U; slot < queue_depth && next_write < requests.size(); ++slot) {
            submit_slot(slot, next_write++);
            next_write = next_non_empty(next_write);
        }
        while (completed < non_empty_count) {
            const uint32_t slot = wait_any_slot();
            PendingWrite &request = pending[slot];
            DWORD transferred = 0U;
            if (!GetOverlappedResult(handle_.get(), &request.ov, &transferred, TRUE)) {
                throw std::runtime_error(detail::bc_direct_win_error("BC direct overlapped GetOverlappedResult failed"));
            }
            if (transferred != static_cast<DWORD>(request.physical_bytes)) {
                throw std::runtime_error("BC direct overlapped write transferred unexpected byte count");
            }
            ++completed;
            if (next_write < requests.size()) {
                submit_slot(slot, next_write++);
                next_write = next_non_empty(next_write);
            } else {
                active[slot] = 0U;
            }
        }
        detail::add_backend_stats(stats, non_empty_count, backend_bytes);
    }

    static void copy_pending_request_to_ranges(
        PendingWriteView request,
        std::vector<detail::BCPhysicalRange> &ranges
    ) {
        uint64_t copied = 0U;
        while (copied < request.bytes) {
            const uint64_t cursor = request.offset + copied;
            const auto it = std::upper_bound(
                ranges.begin(),
                ranges.end(),
                cursor,
                [](uint64_t value, const detail::BCPhysicalRange &range) {
                    return value < range.offset;
                }
            );
            if (it == ranges.begin()) {
                throw std::logic_error("BC direct write request is before physical ranges");
            }
            detail::BCPhysicalRange &range = *(it - 1);
            if (cursor < range.offset || cursor >= range.end()) {
                throw std::logic_error("BC direct write request is not covered by physical ranges");
            }
            const uint64_t in_range = cursor - range.offset;
            const uint64_t take = std::min<uint64_t>(request.bytes - copied, range.bytes - in_range);
            std::memcpy(
                range.buffer.data() + static_cast<size_t>(in_range),
                request.data + copied,
                static_cast<size_t>(take)
            );
            copied += take;
        }
    }

    std::filesystem::path path_;
    BCDirectFileIOOptions options_;
    detail::BCWinHandle handle_;
    uint64_t logical_size_ = 0U;
    uint64_t physical_size_ = 0U;
};

#elif defined(__linux__)

namespace detail {

[[nodiscard]] inline std::string bc_direct_posix_error(const char *label, int error = errno) {
    std::string out(label);
    out += " (";
    out += std::to_string(error);
    out += "): ";
    out += std::strerror(error);
    return out;
}

class BCLinuxFd {
public:
    BCLinuxFd() = default;
    explicit BCLinuxFd(int fd) : fd_(fd) {}
    ~BCLinuxFd() {
        close();
    }

    BCLinuxFd(const BCLinuxFd &) = delete;
    BCLinuxFd &operator=(const BCLinuxFd &) = delete;

    BCLinuxFd(BCLinuxFd &&other) noexcept : fd_(other.fd_) {
        other.fd_ = -1;
    }

    BCLinuxFd &operator=(BCLinuxFd &&other) noexcept {
        if (this != &other) {
            close();
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }

    [[nodiscard]] int get() const {
        return fd_;
    }

    [[nodiscard]] bool valid() const {
        return fd_ >= 0;
    }

    void close() {
        if (valid()) {
            ::close(fd_);
            fd_ = -1;
        }
    }

private:
    int fd_ = -1;
};

class BCAlignedBuffer {
public:
    BCAlignedBuffer() = default;
    BCAlignedBuffer(uint64_t bytes, uint32_t alignment) {
        reset(bytes, alignment);
    }
    ~BCAlignedBuffer() {
        reset();
    }

    BCAlignedBuffer(const BCAlignedBuffer &) = delete;
    BCAlignedBuffer &operator=(const BCAlignedBuffer &) = delete;

    BCAlignedBuffer(BCAlignedBuffer &&other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0U;
    }

    BCAlignedBuffer &operator=(BCAlignedBuffer &&other) noexcept {
        if (this != &other) {
            reset();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0U;
        }
        return *this;
    }

    void reset(uint64_t bytes = 0U, uint32_t alignment = 4096U) {
        if (data_ != nullptr) {
            std::free(data_);
            data_ = nullptr;
            size_ = 0U;
        }
        if (bytes == 0U) {
            return;
        }
        if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::overflow_error("BC direct IO aligned buffer exceeds size_t");
        }
        if (alignment < sizeof(void *) || (alignment % sizeof(void *)) != 0U) {
            throw std::invalid_argument("BC direct IO alignment must be a multiple of pointer size on Linux");
        }
        void *ptr = nullptr;
        const int rc = ::posix_memalign(&ptr, alignment, static_cast<size_t>(bytes));
        if (rc != 0 || ptr == nullptr) {
            if (rc == ENOMEM) {
                throw std::bad_alloc();
            }
            throw std::runtime_error(bc_direct_posix_error("BC direct IO posix_memalign failed", rc));
        }
        data_ = static_cast<uint8_t *>(ptr);
        size_ = static_cast<size_t>(bytes);
    }

    void ensure_at_least(uint64_t bytes, uint32_t alignment = 4096U) {
        if (bytes <= size_) {
            return;
        }
        reset(bytes, alignment);
    }

    [[nodiscard]] uint8_t *data() {
        return data_;
    }

    [[nodiscard]] const uint8_t *data() const {
        return data_;
    }

    [[nodiscard]] size_t size() const {
        return size_;
    }

private:
    uint8_t *data_ = nullptr;
    size_t size_ = 0U;
};

struct BCPhysicalRange {
    uint64_t offset = 0U;
    uint64_t bytes = 0U;
    BCAlignedBuffer buffer;

    [[nodiscard]] uint64_t end() const {
        return offset + bytes;
    }
};

[[nodiscard]] inline BCDirectFileIOOptions normalize_direct_options(const BCDirectFileIOOptions &input) {
    BCDirectFileIOOptions options = input;
    if (options.alignment == 0U || (options.alignment & (options.alignment - 1U)) != 0U) {
        throw std::invalid_argument("BC direct IO alignment must be a non-zero power of two");
    }
    if (options.alignment < sizeof(void *) || (options.alignment % sizeof(void *)) != 0U) {
        throw std::invalid_argument("BC direct IO alignment must be a multiple of pointer size on Linux");
    }
    if (options.queue_depth == 0U) {
        options.queue_depth = 1U;
    }
    return options;
}

[[nodiscard]] inline int open_direct_fd(const std::filesystem::path &path, bool write) {
    int flags = write ? (O_CREAT | O_TRUNC | O_RDWR | O_DIRECT) : (O_RDONLY | O_DIRECT);
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    const int fd = write ? ::open(path.c_str(), flags, 0666) : ::open(path.c_str(), flags);
    if (fd < 0) {
        throw std::runtime_error(bc_direct_posix_error("BC direct IO open failed"));
    }
    return fd;
}

[[nodiscard]] inline uint64_t checked_posix_offset(uint64_t value, const char *label) {
    if (value > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
        throw std::overflow_error(label);
    }
    return value;
}

[[nodiscard]] inline size_t checked_size_t(uint64_t value, const char *label) {
    if (value > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<size_t>(value);
}

[[nodiscard]] inline uint64_t fd_file_size(int fd) {
    struct stat st;
    if (::fstat(fd, &st) != 0) {
        throw std::runtime_error(bc_direct_posix_error("BC direct IO fstat failed"));
    }
    if (st.st_size < 0) {
        throw std::runtime_error("BC direct IO fstat returned negative file size");
    }
    return static_cast<uint64_t>(st.st_size);
}

inline void set_direct_file_size(int fd, uint64_t bytes) {
    (void)checked_posix_offset(bytes, "BC direct IO resize offset exceeds off_t");
    if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
        throw std::runtime_error(bc_direct_posix_error("BC direct IO ftruncate failed"));
    }
}

inline void pread_full(int fd, uint64_t offset, void *data, uint64_t bytes) {
    (void)checked_posix_offset(offset, "BC direct IO read offset exceeds off_t");
    uint8_t *cursor = static_cast<uint8_t *>(data);
    uint64_t remaining = bytes;
    uint64_t file_offset = offset;
    while (remaining != 0U) {
        const uint64_t max_chunk = static_cast<uint64_t>(std::numeric_limits<ssize_t>::max());
        const size_t chunk = checked_size_t(std::min<uint64_t>(remaining, max_chunk),
            "BC direct IO read chunk exceeds size_t");
        const ssize_t got = ::pread(fd, cursor, chunk, static_cast<off_t>(file_offset));
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(bc_direct_posix_error("BC direct IO pread failed"));
        }
        if (got == 0) {
            throw std::runtime_error("BC direct IO pread reached EOF before requested bytes");
        }
        cursor += got;
        remaining -= static_cast<uint64_t>(got);
        file_offset += static_cast<uint64_t>(got);
    }
}

inline void pwrite_full(int fd, uint64_t offset, const void *data, uint64_t bytes) {
    (void)checked_posix_offset(offset, "BC direct IO write offset exceeds off_t");
    const uint8_t *cursor = static_cast<const uint8_t *>(data);
    uint64_t remaining = bytes;
    uint64_t file_offset = offset;
    while (remaining != 0U) {
        const uint64_t max_chunk = static_cast<uint64_t>(std::numeric_limits<ssize_t>::max());
        const size_t chunk = checked_size_t(std::min<uint64_t>(remaining, max_chunk),
            "BC direct IO write chunk exceeds size_t");
        const ssize_t written = ::pwrite(fd, cursor, chunk, static_cast<off_t>(file_offset));
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(bc_direct_posix_error("BC direct IO pwrite failed"));
        }
        if (written == 0) {
            throw std::runtime_error("BC direct IO pwrite transferred zero bytes");
        }
        cursor += written;
        remaining -= static_cast<uint64_t>(written);
        file_offset += static_cast<uint64_t>(written);
    }
}

inline void add_backend_stats(BCFileIOStats *stats, uint64_t io_count, uint64_t bytes) {
    if (stats == nullptr) {
        return;
    }
    if (stats->backend_io_count > std::numeric_limits<uint64_t>::max() - io_count ||
        stats->backend_bytes > std::numeric_limits<uint64_t>::max() - bytes) {
        throw std::overflow_error("BC direct IO backend stats overflow");
    }
    stats->backend_io_count += io_count;
    stats->backend_bytes += bytes;
}

template <class Fn>
inline void run_parallel_jobs(size_t job_count, uint32_t queue_depth, Fn &&fn) {
    if (job_count == 0U) {
        return;
    }
    const uint32_t workers = static_cast<uint32_t>(
        std::min<uint64_t>(std::max<uint32_t>(queue_depth, 1U), job_count)
    );
    if (workers <= 1U || job_count == 1U) {
        for (size_t i = 0U; i < job_count; ++i) {
            fn(i);
        }
        return;
    }
    std::atomic<size_t> next{0U};
    std::mutex error_mutex;
    std::exception_ptr first_error;
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (uint32_t worker = 0U; worker < workers; ++worker) {
        threads.emplace_back([&]() {
            try {
                for (;;) {
                    const size_t index = next.fetch_add(1U, std::memory_order_relaxed);
                    if (index >= job_count) {
                        break;
                    }
                    fn(index);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (first_error == nullptr) {
                    first_error = std::current_exception();
                }
            }
        });
    }
    for (std::thread &thread : threads) {
        thread.join();
    }
    if (first_error != nullptr) {
        std::rethrow_exception(first_error);
    }
}

} // namespace detail

class BCDirectFileReader final : public BCReadableFile {
public:
    explicit BCDirectFileReader(
        const std::filesystem::path &path,
        const BCDirectFileIOOptions &options = BCDirectFileIOOptions{}
    )
        : path_(path), options_(detail::normalize_direct_options(options)) {
        fd_ = detail::BCLinuxFd(detail::open_direct_fd(path_, false));
        physical_size_ = detail::fd_file_size(fd_.get());
        logical_size_ = options_.logical_size.value_or(physical_size_);
        if (logical_size_ > physical_size_) {
            throw std::runtime_error("BC direct reader logical size exceeds physical file size");
        }
        const uint64_t required_physical = bc_direct_align_up(logical_size_, options_.alignment);
        if (required_physical > physical_size_) {
            throw std::runtime_error("BC direct reader requires padded physical file size");
        }
    }

    void read_at(uint64_t offset, void *data, uint64_t bytes) const override {
        read_many(std::vector<BCFileReadRequest>{BCFileReadRequest{offset, data, bytes}});
    }

    [[nodiscard]] BCFileIOMode mode() const override {
        return BCFileIOMode::Direct;
    }

    void read_many(
        const std::vector<BCFileReadRequest> &requests,
        BCFileIOStats *stats = nullptr
    ) const override {
        if (stats != nullptr) {
            *stats = {};
        }

        struct DirectLogicalRead {
            uint64_t physical_offset = 0U;
            uint64_t physical_bytes = 0U;
            uint64_t in_physical_offset = 0U;
            uint8_t *target = nullptr;
            uint64_t target_bytes = 0U;
            bool direct_to_target = false;
            detail::BCAlignedBuffer buffer;
        };

        std::vector<DirectLogicalRead> reads;
        reads.reserve(requests.size());
        uint64_t backend_bytes = 0U;
        for (const BCFileReadRequest &request : requests) {
            if (request.bytes == 0U) {
                continue;
            }
            if (request.data == nullptr) {
                throw std::invalid_argument("BC direct read request data pointer is null");
            }
            if (request.offset > logical_size_ || request.bytes > logical_size_ - request.offset) {
                throw std::out_of_range("BC direct read request exceeds logical file size");
            }
            if (stats != nullptr) {
                ++stats->request_count;
                if (stats->requested_bytes > std::numeric_limits<uint64_t>::max() - request.bytes) {
                    throw std::overflow_error("BC direct read request byte stats overflow");
                }
                stats->requested_bytes += request.bytes;
            }
            const uint64_t physical_offset = bc_direct_align_down(request.offset, options_.alignment);
            const uint64_t physical_end = bc_direct_align_up(request.offset + request.bytes, options_.alignment);
            if (physical_end > physical_size_) {
                throw std::out_of_range("BC direct read request exceeds physical file size");
            }
            const uint64_t physical_bytes = physical_end - physical_offset;
            if (backend_bytes > std::numeric_limits<uint64_t>::max() - physical_bytes) {
                throw std::overflow_error("BC direct read backend byte stats overflow");
            }
            backend_bytes += physical_bytes;
            const bool direct_to_target =
                physical_offset == request.offset &&
                physical_bytes == request.bytes &&
                (reinterpret_cast<uintptr_t>(request.data) & (options_.alignment - 1U)) == 0U;
            reads.push_back(DirectLogicalRead{
                physical_offset,
                physical_bytes,
                request.offset - physical_offset,
                static_cast<uint8_t *>(request.data),
                request.bytes,
                direct_to_target,
                detail::BCAlignedBuffer{}
            });
            if (!direct_to_target) {
                reads.back().buffer.reset(physical_bytes, options_.alignment);
            }
        }
        if (reads.empty()) {
            return;
        }

        detail::run_parallel_jobs(reads.size(), options_.queue_depth, [&](size_t index) {
            DirectLogicalRead &read = reads[index];
            void *target = read.direct_to_target ? read.target : read.buffer.data();
            detail::pread_full(fd_.get(), read.physical_offset, target, read.physical_bytes);
        });

        for (DirectLogicalRead &read : reads) {
            if (read.direct_to_target) {
                continue;
            }
            std::memcpy(
                read.target,
                read.buffer.data() + static_cast<size_t>(read.in_physical_offset),
                detail::checked_size_t(read.target_bytes, "BC direct read target copy exceeds size_t")
            );
        }
        detail::add_backend_stats(stats, reads.size(), backend_bytes);
    }

    [[nodiscard]] uint64_t size() const override {
        return logical_size_;
    }

private:
    std::filesystem::path path_;
    BCDirectFileIOOptions options_;
    detail::BCLinuxFd fd_;
    uint64_t logical_size_ = 0U;
    uint64_t physical_size_ = 0U;
};

// Lazy reader for append-only files. On Linux, BCDirectFileReader already uses
// positioned pread and qd worker threads, so the lazy wrapper mainly refreshes
// file size per call while preserving the same public backend contract.
class BCLazyDirectFileReader final : public BCReadableFile {
public:
    explicit BCLazyDirectFileReader(
        const std::filesystem::path &path,
        const BCDirectFileIOOptions &options = BCDirectFileIOOptions{}
    ) : path_(path), options_(options) {}

    void read_at(uint64_t offset, void *data, uint64_t bytes) const override {
        BCDirectFileIOOptions local_options = options_;
        local_options.logical_size = std::nullopt;
        BCDirectFileReader reader(path_, local_options);
        reader.read_at(offset, data, bytes);
    }

    void read_many(
        const std::vector<BCFileReadRequest> &requests,
        BCFileIOStats *stats = nullptr
    ) const override {
        BCDirectFileIOOptions local_options = options_;
        local_options.logical_size = std::nullopt;
        BCDirectFileReader reader(path_, local_options);
        reader.read_many(requests, stats);
    }

    [[nodiscard]] BCFileIOMode mode() const override {
        return BCFileIOMode::Direct;
    }

    [[nodiscard]] uint64_t size() const override {
        BCDirectFileIOOptions local_options = options_;
        local_options.logical_size = std::nullopt;
        BCDirectFileReader reader(path_, local_options);
        return reader.size();
    }

private:
    std::filesystem::path path_;
    BCDirectFileIOOptions options_;
};

class BCDirectFileWriter final : public BCWritableFile {
    struct PendingWriteView {
        uint64_t offset = 0U;
        const uint8_t *data = nullptr;
        uint64_t bytes = 0U;
    };

public:
    // Linux O_DIRECT writer uses aligned staging buffers for non-4KB-aligned
    // logical requests. With preserve_unwritten_bytes=false it does not
    // read-modify-write untouched bytes inside aligned physical blocks; use it
    // for new files or planned full writes. Set preserve_unwritten_bytes=true
    // for append/tail updates that must preserve existing block contents.
    explicit BCDirectFileWriter(
        const std::filesystem::path &path,
        const BCDirectFileIOOptions &options = BCDirectFileIOOptions{}
    )
        : path_(path), options_(detail::normalize_direct_options(options)) {
        fd_ = detail::BCLinuxFd(detail::open_direct_fd(path_, true));
        logical_size_ = options_.logical_size.value_or(0U);
        physical_size_ = bc_direct_align_up(logical_size_, options_.alignment);
        if (physical_size_ != 0U) {
            detail::set_direct_file_size(fd_.get(), physical_size_);
        }
    }

    void write_at(uint64_t offset, const void *data, uint64_t bytes) override {
        write_many(std::vector<BCFileWriteRequest>{BCFileWriteRequest{offset, data, bytes}});
    }

    [[nodiscard]] BCFileIOMode mode() const override {
        return BCFileIOMode::Direct;
    }

    void write_many(
        const std::vector<BCFileWriteRequest> &requests,
        BCFileIOStats *stats = nullptr
    ) override {
        if (stats != nullptr) {
            *stats = {};
        }
        std::vector<PendingWriteView> pending;
        pending.reserve(requests.size());
        uint64_t max_logical_end = logical_size_;
        for (const BCFileWriteRequest &request : requests) {
            if (request.bytes == 0U) {
                continue;
            }
            if (request.data == nullptr) {
                throw std::invalid_argument("BC direct write request data pointer is null");
            }
            if (request.offset > std::numeric_limits<uint64_t>::max() - request.bytes) {
                throw std::overflow_error("BC direct write request offset overflow");
            }
            const uint64_t logical_end = request.offset + request.bytes;
            max_logical_end = std::max(max_logical_end, logical_end);
            pending.push_back(PendingWriteView{
                request.offset,
                static_cast<const uint8_t *>(request.data),
                request.bytes
            });
            if (stats != nullptr) {
                ++stats->request_count;
                if (stats->requested_bytes > std::numeric_limits<uint64_t>::max() - request.bytes) {
                    throw std::overflow_error("BC direct write request byte stats overflow");
                }
                stats->requested_bytes += request.bytes;
            }
        }
        if (pending.empty()) {
            return;
        }

        bool can_write_external_aligned = !options_.preserve_unwritten_bytes;
        uint64_t direct_backend_bytes = 0U;
        uint64_t previous_end = 0U;
        bool has_previous = false;
        for (const PendingWriteView &request : pending) {
            const bool aligned =
                (request.offset & (options_.alignment - 1U)) == 0U &&
                (request.bytes & (options_.alignment - 1U)) == 0U &&
                (reinterpret_cast<uintptr_t>(request.data) & (options_.alignment - 1U)) == 0U;
            const uint64_t request_end = request.offset + request.bytes;
            const bool ordered_non_overlapping = !has_previous || request.offset >= previous_end;
            if (!aligned || !ordered_non_overlapping) {
                can_write_external_aligned = false;
                break;
            }
            has_previous = true;
            previous_end = request_end;
            if (direct_backend_bytes > std::numeric_limits<uint64_t>::max() - request.bytes) {
                throw std::overflow_error("BC direct write backend byte stats overflow");
            }
            direct_backend_bytes += request.bytes;
        }
        if (can_write_external_aligned) {
            const uint64_t new_physical_size = bc_direct_align_up(max_logical_end, options_.alignment);
            if (new_physical_size > physical_size_) {
                detail::set_direct_file_size(fd_.get(), new_physical_size);
                physical_size_ = new_physical_size;
            }
            detail::run_parallel_jobs(pending.size(), options_.queue_depth, [&](size_t index) {
                const PendingWriteView &request = pending[index];
                detail::pwrite_full(fd_.get(), request.offset, request.data, request.bytes);
            });
            logical_size_ = max_logical_end;
            detail::add_backend_stats(stats, pending.size(), direct_backend_bytes);
            return;
        }

        struct RangePlan {
            uint64_t offset = 0U;
            uint64_t end = 0U;
        };

        std::vector<RangePlan> plans;
        plans.reserve(pending.size());
        for (const PendingWriteView &request : pending) {
            const uint64_t physical_offset = bc_direct_align_down(request.offset, options_.alignment);
            const uint64_t physical_end = bc_direct_align_up(request.offset + request.bytes, options_.alignment);
            plans.push_back(RangePlan{physical_offset, physical_end});
        }
        std::sort(plans.begin(), plans.end(), [](const RangePlan &a, const RangePlan &b) {
            return a.offset < b.offset || (a.offset == b.offset && a.end < b.end);
        });

        std::vector<detail::BCPhysicalRange> ranges;
        ranges.reserve(plans.size());
        for (const RangePlan &plan : plans) {
            if (ranges.empty() || plan.offset > ranges.back().end()) {
                detail::BCPhysicalRange range;
                range.offset = plan.offset;
                range.bytes = plan.end - plan.offset;
                ranges.push_back(std::move(range));
            } else if (plan.end > ranges.back().end()) {
                ranges.back().bytes = plan.end - ranges.back().offset;
            }
        }

        uint64_t backend_bytes = 0U;
        for (detail::BCPhysicalRange &range : ranges) {
            if (backend_bytes > std::numeric_limits<uint64_t>::max() - range.bytes) {
                throw std::overflow_error("BC direct write backend byte stats overflow");
            }
            backend_bytes += range.bytes;
            range.buffer.reset(range.bytes, options_.alignment);
        }

        const uint64_t old_physical_size = physical_size_;
        const uint64_t new_physical_size = bc_direct_align_up(max_logical_end, options_.alignment);
        if (new_physical_size > physical_size_) {
            detail::set_direct_file_size(fd_.get(), new_physical_size);
            physical_size_ = new_physical_size;
        }

        for (detail::BCPhysicalRange &range : ranges) {
            if (options_.preserve_unwritten_bytes && range.offset < old_physical_size) {
                const uint64_t readable = std::min<uint64_t>(range.bytes, old_physical_size - range.offset);
                detail::pread_full(fd_.get(), range.offset, range.buffer.data(), readable);
                if (readable < range.bytes) {
                    std::memset(
                        range.buffer.data() + static_cast<size_t>(readable),
                        0,
                        detail::checked_size_t(range.bytes - readable, "BC direct write zero tail exceeds size_t")
                    );
                }
            } else {
                std::memset(
                    range.buffer.data(),
                    0,
                    detail::checked_size_t(range.bytes, "BC direct write zero fill exceeds size_t")
                );
            }
        }

        for (const PendingWriteView &request : pending) {
            copy_pending_request_to_ranges(request, ranges);
        }

        detail::run_parallel_jobs(ranges.size(), options_.queue_depth, [&](size_t index) {
            const detail::BCPhysicalRange &range = ranges[index];
            detail::pwrite_full(fd_.get(), range.offset, range.buffer.data(), range.bytes);
        });

        logical_size_ = max_logical_end;
        detail::add_backend_stats(stats, ranges.size(), backend_bytes);
    }

    void resize(uint64_t bytes) override {
        logical_size_ = bytes;
        physical_size_ = bc_direct_align_up(bytes, options_.alignment);
        detail::set_direct_file_size(fd_.get(), physical_size_);
    }

    void prepare_full_overwrite(uint64_t bytes) override {
        resize(bytes);
    }

    void flush() override {
        if (::fsync(fd_.get()) != 0) {
            throw std::runtime_error(detail::bc_direct_posix_error("BC direct IO fsync failed"));
        }
    }

private:
    static void copy_pending_request_to_ranges(
        PendingWriteView request,
        std::vector<detail::BCPhysicalRange> &ranges
    ) {
        uint64_t copied = 0U;
        while (copied < request.bytes) {
            const uint64_t cursor = request.offset + copied;
            const auto it = std::upper_bound(
                ranges.begin(),
                ranges.end(),
                cursor,
                [](uint64_t value, const detail::BCPhysicalRange &range) {
                    return value < range.offset;
                }
            );
            if (it == ranges.begin()) {
                throw std::logic_error("BC direct write request is before physical ranges");
            }
            detail::BCPhysicalRange &range = *(it - 1);
            if (cursor < range.offset || cursor >= range.end()) {
                throw std::logic_error("BC direct write request is not covered by physical ranges");
            }
            const uint64_t in_range = cursor - range.offset;
            const uint64_t take = std::min<uint64_t>(request.bytes - copied, range.bytes - in_range);
            std::memcpy(
                range.buffer.data() + static_cast<size_t>(in_range),
                request.data + copied,
                detail::checked_size_t(take, "BC direct write request copy exceeds size_t")
            );
            copied += take;
        }
    }

    std::filesystem::path path_;
    BCDirectFileIOOptions options_;
    detail::BCLinuxFd fd_;
    uint64_t logical_size_ = 0U;
    uint64_t physical_size_ = 0U;
};

#else

class BCDirectFileReader final : public BCReadableFile {
public:
    explicit BCDirectFileReader(const std::filesystem::path &, BCDirectFileIOOptions = {}) {
        throw std::runtime_error("BC direct file reader is only implemented on Windows and Linux");
    }
    void read_at(uint64_t, void *, uint64_t) const override {}
    uint64_t size() const override {
        return 0U;
    }
};

class BCLazyDirectFileReader final : public BCReadableFile {
public:
    explicit BCLazyDirectFileReader(const std::filesystem::path &, BCDirectFileIOOptions = {}) {
        throw std::runtime_error("BC lazy direct file reader is only implemented on Windows and Linux");
    }
    void read_at(uint64_t, void *, uint64_t) const override {}
    uint64_t size() const override {
        return 0U;
    }
};

class BCDirectFileWriter final : public BCWritableFile {
public:
    explicit BCDirectFileWriter(const std::filesystem::path &, BCDirectFileIOOptions = {}) {
        throw std::runtime_error("BC direct file writer is only implemented on Windows and Linux");
    }
    void write_at(uint64_t, const void *, uint64_t) override {}
    void resize(uint64_t) override {}
    void flush() override {}
};

#endif

} // namespace BC
