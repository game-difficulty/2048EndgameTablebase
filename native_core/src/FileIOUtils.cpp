#include "FileIOUtils.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <system_error>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace FileIOUtils {

namespace {

constexpr uint64_t kMiB = 1024ULL * 1024ULL;

[[noreturn]] void throw_io_error(const std::string &message) {
    throw std::runtime_error(message);
}

size_t direct_chunk_bytes(DirectIoConfig config) {
    const uint64_t requested =
        static_cast<uint64_t>(normalize_direct_io_config(config).chunk_mib) * kMiB;
    return static_cast<size_t>(align_up_u64(requested, kDirectIoAlignment));
}

class BufferedAppendWriter {
public:
    explicit BufferedAppendWriter(const std::string &path)
        : path_(path), out_(path, std::ios::binary | std::ios::trunc) {
        if (!out_) {
            throw_io_error("failed to open for write: " + path_);
        }
    }

    void append(const void *src, size_t bytes) {
        if (bytes == 0U) {
            return;
        }
        write_exact(out_, src, bytes, path_);
    }

    void close() {
        out_.flush();
        out_.close();
    }

private:
    std::string path_;
    std::ofstream out_;
};

class BufferedSequentialReader {
public:
    explicit BufferedSequentialReader(const std::string &path)
        : path_(path), in_(path, std::ios::binary) {
        if (!in_) {
            throw_io_error("failed to open for read: " + path_);
        }
    }

    void read(void *dst, size_t bytes) {
        if (bytes == 0U) {
            return;
        }
        read_exact(in_, dst, bytes, path_);
    }

    void close() {
        in_.close();
    }

private:
    std::string path_;
    std::ifstream in_;
};

#ifdef _WIN32

std::runtime_error make_win32_error(const std::string &prefix, DWORD code) {
    return std::runtime_error(prefix + " (win32=" + std::to_string(static_cast<unsigned long>(code)) + ")");
}

void *alloc_aligned_bytes(size_t bytes) {
    if (bytes == 0U) {
        bytes = static_cast<size_t>(kDirectIoAlignment);
    }
    void *ptr = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (ptr == nullptr) {
        throw make_win32_error("VirtualAlloc failed", GetLastError());
    }
    return ptr;
}

void free_aligned_bytes(void *ptr) {
    if (ptr != nullptr) {
        VirtualFree(ptr, 0, MEM_RELEASE);
    }
}

struct DirectWriteSlot {
    void *buffer = nullptr;
    HANDLE event = nullptr;
    OVERLAPPED overlapped{};
    DWORD bytes = 0U;
    bool active = false;
};

class DirectFileReaderWin32 {
public:
    DirectFileReaderWin32(const std::string &path, uint64_t logical_bytes, DirectIoConfig config)
        : path_(path),
          logical_bytes_(logical_bytes),
          aligned_prefix_bytes_(align_down_u64(logical_bytes, kDirectIoAlignment)),
          config_(normalize_direct_io_config(config)),
          chunk_bytes_(direct_chunk_bytes(config_)),
          scratch_(alloc_aligned_bytes(chunk_bytes_)),
          buffered_tail_(path, std::ios::binary) {
        const std::wstring native_path = fs::path(path_).wstring();
        handle_ = CreateFileW(
            native_path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING,
            nullptr
        );
        if (handle_ == INVALID_HANDLE_VALUE) {
            throw make_win32_error("CreateFileW direct read failed", GetLastError());
        }
        if (!buffered_tail_) {
            throw_io_error("failed to open buffered tail reader: " + path_);
        }
    }

    ~DirectFileReaderWin32() {
        free_aligned_bytes(scratch_);
        if (handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
        }
    }

    void read(void *dst, size_t bytes) {
        auto *out = static_cast<uint8_t *>(dst);
        size_t consumed = 0U;
        while (consumed < bytes) {
            if (logical_offset_ < aligned_prefix_bytes_) {
                if (!buffer_covers(logical_offset_)) {
                    fill_direct_buffer();
                }
                const uint64_t buffer_index = logical_offset_ - buffer_offset_;
                const size_t available = static_cast<size_t>(buffer_valid_ - buffer_index);
                const size_t take = std::min(available, bytes - consumed);
                std::memcpy(out + consumed, static_cast<uint8_t *>(scratch_) + buffer_index, take);
                logical_offset_ += static_cast<uint64_t>(take);
                consumed += take;
                continue;
            }

            const size_t tail_bytes = bytes - consumed;
            buffered_tail_.seekg(static_cast<std::streamoff>(logical_offset_), std::ios::beg);
            read_exact(buffered_tail_, out + consumed, tail_bytes, path_);
            logical_offset_ += static_cast<uint64_t>(tail_bytes);
            consumed += tail_bytes;
        }
    }

    void close() {
        buffered_tail_.close();
    }

private:
    bool buffer_covers(uint64_t logical_offset) const {
        return buffer_valid_ != 0ULL &&
               logical_offset >= buffer_offset_ &&
               logical_offset < buffer_offset_ + buffer_valid_;
    }

    void fill_direct_buffer() {
        if (logical_offset_ >= aligned_prefix_bytes_) {
            buffer_valid_ = 0ULL;
            return;
        }
        const uint64_t aligned_offset = align_down_u64(logical_offset_, kDirectIoAlignment);
        const uint64_t request_bytes_u64 = std::min<uint64_t>(
            static_cast<uint64_t>(chunk_bytes_),
            aligned_prefix_bytes_ - aligned_offset
        );
        const DWORD request_bytes = static_cast<DWORD>(request_bytes_u64);
        LARGE_INTEGER file_offset{};
        file_offset.QuadPart = static_cast<LONGLONG>(aligned_offset);
        if (!SetFilePointerEx(handle_, file_offset, nullptr, FILE_BEGIN)) {
            throw make_win32_error("SetFilePointerEx read failed", GetLastError());
        }
        DWORD transferred = 0U;
        if (!ReadFile(handle_, scratch_, request_bytes, &transferred, nullptr)) {
            throw make_win32_error("ReadFile failed", GetLastError());
        }
        if (transferred != request_bytes) {
            throw_io_error("short direct read: " + path_);
        }
        buffer_offset_ = aligned_offset;
        buffer_valid_ = request_bytes_u64;
    }

    std::string path_;
    HANDLE handle_ = INVALID_HANDLE_VALUE;
    uint64_t logical_bytes_ = 0ULL;
    uint64_t aligned_prefix_bytes_ = 0ULL;
    DirectIoConfig config_{};
    size_t chunk_bytes_ = 0U;
    void *scratch_ = nullptr;
    std::ifstream buffered_tail_;
    uint64_t logical_offset_ = 0ULL;
    uint64_t buffer_offset_ = 0ULL;
    uint64_t buffer_valid_ = 0ULL;
};

class DirectFileWriterWin32 {
public:
    DirectFileWriterWin32(const std::string &path, uint64_t logical_bytes, DirectIoConfig config)
        : path_(path),
          logical_bytes_(logical_bytes),
          aligned_bytes_(align_up_u64(logical_bytes, kDirectIoAlignment)),
          config_(normalize_direct_io_config(config)),
          chunk_bytes_(direct_chunk_bytes(config_)) {
        const std::wstring native_path = fs::path(path_).wstring();
        handle_ = CreateFileW(
            native_path.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ,
            nullptr,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
            nullptr
        );
        if (handle_ == INVALID_HANDLE_VALUE) {
            throw make_win32_error("CreateFileW direct write failed", GetLastError());
        }
        preallocate();
        init_slots();
    }

    ~DirectFileWriterWin32() {
        for (auto &slot : slots_) {
            free_aligned_bytes(slot.buffer);
            if (slot.event != nullptr) {
                CloseHandle(slot.event);
            }
        }
        if (handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
        }
    }

    void append(const void *src, size_t bytes) {
        const auto *input = static_cast<const uint8_t *>(src);
        size_t consumed = 0U;
        while (consumed < bytes) {
            if (current_slot_ == kNoSlot) {
                current_slot_ = acquire_slot();
                current_fill_ = 0U;
            }
            DirectWriteSlot &slot = slots_[current_slot_];
            const size_t remaining = chunk_bytes_ - current_fill_;
            const size_t take = std::min(remaining, bytes - consumed);
            std::memcpy(static_cast<uint8_t *>(slot.buffer) + current_fill_, input + consumed, take);
            current_fill_ += take;
            consumed += take;
            if (current_fill_ == chunk_bytes_) {
                submit_current_slot(static_cast<DWORD>(chunk_bytes_));
            }
        }
        written_bytes_ += static_cast<uint64_t>(bytes);
    }

    void close() {
        if (written_bytes_ != logical_bytes_) {
            throw_io_error("direct writer byte count mismatch: " + path_);
        }
        if (current_slot_ != kNoSlot && current_fill_ != 0U) {
            DirectWriteSlot &slot = slots_[current_slot_];
            const DWORD padded = static_cast<DWORD>(align_up_u64(current_fill_, kDirectIoAlignment));
            std::memset(static_cast<uint8_t *>(slot.buffer) + current_fill_, 0, padded - current_fill_);
            submit_current_slot(padded);
        }
        while (true) {
            bool any_active = false;
            for (const auto &slot : slots_) {
                any_active = any_active || slot.active;
            }
            if (!any_active) {
                break;
            }
            drain(true);
        }

        LARGE_INTEGER offset{};
        offset.QuadPart = static_cast<LONGLONG>(logical_bytes_);
        if (!SetFilePointerEx(handle_, offset, nullptr, FILE_BEGIN)) {
            throw make_win32_error("SetFilePointerEx shrink failed", GetLastError());
        }
        if (!SetEndOfFile(handle_)) {
            throw make_win32_error("SetEndOfFile shrink failed", GetLastError());
        }
        if (!FlushFileBuffers(handle_)) {
            throw make_win32_error("FlushFileBuffers failed", GetLastError());
        }
    }

private:
    static constexpr size_t kNoSlot = static_cast<size_t>(-1);

    void preallocate() {
        LARGE_INTEGER offset{};
        offset.QuadPart = static_cast<LONGLONG>(aligned_bytes_);
        if (!SetFilePointerEx(handle_, offset, nullptr, FILE_BEGIN)) {
            throw make_win32_error("SetFilePointerEx preallocate failed", GetLastError());
        }
        if (!SetEndOfFile(handle_)) {
            throw make_win32_error("SetEndOfFile preallocate failed", GetLastError());
        }
    }

    void init_slots() {
        slots_.resize(config_.queue_depth);
        for (auto &slot : slots_) {
            slot.buffer = alloc_aligned_bytes(chunk_bytes_);
            slot.event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            if (slot.event == nullptr) {
                throw make_win32_error("CreateEventW failed", GetLastError());
            }
            slot.overlapped = OVERLAPPED{};
            slot.overlapped.hEvent = slot.event;
        }
    }

    size_t acquire_slot() {
        while (true) {
            if (auto ready = find_ready_slot(false); ready != kNoSlot) {
                return ready;
            }
            drain(true);
        }
    }

    size_t find_ready_slot(bool wait) {
        std::array<HANDLE, 32> active_events{};
        std::array<size_t, 32> active_indices{};
        size_t active_count = 0U;
        for (size_t i = 0U; i < slots_.size(); ++i) {
            if (!slots_[i].active) {
                return i;
            }
            active_events[active_count] = slots_[i].event;
            active_indices[active_count] = i;
            ++active_count;
        }
        if (!wait || active_count == 0U) {
            return kNoSlot;
        }
        const DWORD wait_result = WaitForMultipleObjects(
            static_cast<DWORD>(active_count),
            active_events.data(),
            FALSE,
            INFINITE
        );
        if (wait_result < WAIT_OBJECT_0 || wait_result >= WAIT_OBJECT_0 + active_count) {
            throw make_win32_error("WaitForMultipleObjects failed", GetLastError());
        }
        const size_t slot_index = active_indices[static_cast<size_t>(wait_result - WAIT_OBJECT_0)];
        complete_slot(slot_index);
        return slot_index;
    }

    void complete_slot(size_t index) {
        DirectWriteSlot &slot = slots_[index];
        if (!slot.active) {
            return;
        }
        DWORD transferred = 0U;
        if (!GetOverlappedResult(handle_, &slot.overlapped, &transferred, TRUE)) {
            throw make_win32_error("GetOverlappedResult failed", GetLastError());
        }
        if (transferred != slot.bytes) {
            throw_io_error("short direct write: " + path_);
        }
        slot.active = false;
        slot.bytes = 0U;
        slot.overlapped = OVERLAPPED{};
        slot.overlapped.hEvent = slot.event;
    }

    void drain(bool wait_for_one) {
        if (wait_for_one) {
            (void)find_ready_slot(true);
        }
        for (size_t i = 0U; i < slots_.size(); ++i) {
            if (slots_[i].active && WaitForSingleObject(slots_[i].event, 0U) == WAIT_OBJECT_0) {
                complete_slot(i);
            }
        }
    }

    void submit_current_slot(DWORD bytes) {
        DirectWriteSlot &slot = slots_[current_slot_];
        slot.overlapped = OVERLAPPED{};
        slot.overlapped.Offset = static_cast<DWORD>(next_offset_ & 0xFFFFFFFFULL);
        slot.overlapped.OffsetHigh = static_cast<DWORD>((next_offset_ >> 32U) & 0xFFFFFFFFULL);
        slot.overlapped.hEvent = slot.event;
        slot.bytes = bytes;
        slot.active = true;
        ResetEvent(slot.event);
        if (!WriteFile(handle_, slot.buffer, bytes, nullptr, &slot.overlapped)) {
            const DWORD error = GetLastError();
            if (error != ERROR_IO_PENDING) {
                slot.active = false;
                throw make_win32_error("WriteFile failed", error);
            }
        }
        next_offset_ += static_cast<uint64_t>(bytes);
        current_slot_ = kNoSlot;
        current_fill_ = 0U;
    }

    std::string path_;
    HANDLE handle_ = INVALID_HANDLE_VALUE;
    uint64_t logical_bytes_ = 0ULL;
    uint64_t aligned_bytes_ = 0ULL;
    DirectIoConfig config_{};
    size_t chunk_bytes_ = 0U;
    std::vector<DirectWriteSlot> slots_;
    size_t current_slot_ = kNoSlot;
    size_t current_fill_ = 0U;
    uint64_t next_offset_ = 0ULL;
    uint64_t written_bytes_ = 0ULL;
};

#elif defined(__linux__)

std::runtime_error make_errno_error(const std::string &prefix) {
    return std::runtime_error(prefix + " (errno=" + std::to_string(errno) + ": " + std::strerror(errno) + ")");
}

void *alloc_aligned_bytes(size_t bytes) {
    if (bytes == 0U) {
        bytes = static_cast<size_t>(kDirectIoAlignment);
    }
    void *ptr = nullptr;
    if (posix_memalign(&ptr, static_cast<size_t>(kDirectIoAlignment), bytes) != 0) {
        throw make_errno_error("posix_memalign failed");
    }
    return ptr;
}

void free_aligned_bytes(void *ptr) {
    std::free(ptr);
}

class DirectFileReaderPosix {
public:
    DirectFileReaderPosix(const std::string &path, uint64_t logical_bytes, DirectIoConfig config)
        : path_(path),
          logical_bytes_(logical_bytes),
          aligned_prefix_bytes_(align_down_u64(logical_bytes, kDirectIoAlignment)),
          config_(normalize_direct_io_config(config)),
          chunk_bytes_(direct_chunk_bytes(config_)),
          scratch_(alloc_aligned_bytes(chunk_bytes_)),
          buffered_tail_(path, std::ios::binary) {
        fd_ = ::open(path_.c_str(), O_RDONLY | O_DIRECT);
        if (fd_ < 0) {
            throw make_errno_error("open(O_DIRECT) read failed");
        }
        if (!buffered_tail_) {
            throw_io_error("failed to open buffered tail reader: " + path_);
        }
    }

    ~DirectFileReaderPosix() {
        free_aligned_bytes(scratch_);
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    void read(void *dst, size_t bytes) {
        auto *out = static_cast<uint8_t *>(dst);
        size_t consumed = 0U;
        while (consumed < bytes) {
            if (logical_offset_ < aligned_prefix_bytes_) {
                if (!buffer_covers(logical_offset_)) {
                    fill_direct_buffer();
                }
                const uint64_t buffer_index = logical_offset_ - buffer_offset_;
                const size_t available = static_cast<size_t>(buffer_valid_ - buffer_index);
                const size_t take = std::min(available, bytes - consumed);
                std::memcpy(out + consumed, static_cast<uint8_t *>(scratch_) + buffer_index, take);
                logical_offset_ += static_cast<uint64_t>(take);
                consumed += take;
                continue;
            }

            const size_t tail_bytes = bytes - consumed;
            buffered_tail_.seekg(static_cast<std::streamoff>(logical_offset_), std::ios::beg);
            read_exact(buffered_tail_, out + consumed, tail_bytes, path_);
            logical_offset_ += static_cast<uint64_t>(tail_bytes);
            consumed += tail_bytes;
        }
    }

    void close() {
        buffered_tail_.close();
    }

private:
    bool buffer_covers(uint64_t logical_offset) const {
        return buffer_valid_ != 0ULL &&
               logical_offset >= buffer_offset_ &&
               logical_offset < buffer_offset_ + buffer_valid_;
    }

    void fill_direct_buffer() {
        if (logical_offset_ >= aligned_prefix_bytes_) {
            buffer_valid_ = 0ULL;
            return;
        }
        const uint64_t aligned_offset = align_down_u64(logical_offset_, kDirectIoAlignment);
        const size_t request_bytes = static_cast<size_t>(std::min<uint64_t>(
            static_cast<uint64_t>(chunk_bytes_),
            aligned_prefix_bytes_ - aligned_offset
        ));
        const ssize_t rv = ::pread(fd_, scratch_, request_bytes, static_cast<off_t>(aligned_offset));
        if (rv < 0) {
            throw make_errno_error("pread(O_DIRECT) failed");
        }
        if (static_cast<size_t>(rv) != request_bytes) {
            throw_io_error("short direct read: " + path_);
        }
        buffer_offset_ = aligned_offset;
        buffer_valid_ = static_cast<uint64_t>(request_bytes);
    }

    std::string path_;
    int fd_ = -1;
    uint64_t logical_bytes_ = 0ULL;
    uint64_t aligned_prefix_bytes_ = 0ULL;
    DirectIoConfig config_{};
    size_t chunk_bytes_ = 0U;
    void *scratch_ = nullptr;
    std::ifstream buffered_tail_;
    uint64_t logical_offset_ = 0ULL;
    uint64_t buffer_offset_ = 0ULL;
    uint64_t buffer_valid_ = 0ULL;
};

class DirectFileWriterPosix {
public:
    DirectFileWriterPosix(const std::string &path, uint64_t logical_bytes, DirectIoConfig config)
        : path_(path),
          logical_bytes_(logical_bytes),
          aligned_bytes_(align_up_u64(logical_bytes, kDirectIoAlignment)),
          config_(normalize_direct_io_config(config)),
          chunk_bytes_(direct_chunk_bytes(config_)),
          scratch_(alloc_aligned_bytes(chunk_bytes_)) {
        fd_ = ::open(path_.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_DIRECT, 0644);
        if (fd_ < 0) {
            throw make_errno_error("open(O_DIRECT) failed");
        }
        if (::ftruncate(fd_, static_cast<off_t>(aligned_bytes_)) != 0) {
            throw make_errno_error("ftruncate preallocate failed");
        }
    }

    ~DirectFileWriterPosix() {
        free_aligned_bytes(scratch_);
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    void append(const void *src, size_t bytes) {
        const auto *input = static_cast<const uint8_t *>(src);
        size_t consumed = 0U;
        while (consumed < bytes) {
            const size_t remaining = chunk_bytes_ - current_fill_;
            const size_t take = std::min(remaining, bytes - consumed);
            std::memcpy(static_cast<uint8_t *>(scratch_) + current_fill_, input + consumed, take);
            current_fill_ += take;
            consumed += take;
            if (current_fill_ == chunk_bytes_) {
                flush_current(chunk_bytes_);
            }
        }
        written_bytes_ += static_cast<uint64_t>(bytes);
    }

    void close() {
        if (written_bytes_ != logical_bytes_) {
            throw_io_error("direct writer byte count mismatch: " + path_);
        }
        if (current_fill_ != 0U) {
            const size_t padded = static_cast<size_t>(align_up_u64(current_fill_, kDirectIoAlignment));
            std::memset(static_cast<uint8_t *>(scratch_) + current_fill_, 0, padded - current_fill_);
            flush_current(padded);
        }
        if (::ftruncate(fd_, static_cast<off_t>(logical_bytes_)) != 0) {
            throw make_errno_error("ftruncate shrink failed");
        }
        if (::fsync(fd_) != 0) {
            throw make_errno_error("fsync failed");
        }
    }

private:
    void flush_current(size_t bytes) {
        const ssize_t rv = ::pwrite(
            fd_,
            scratch_,
            bytes,
            static_cast<off_t>(next_offset_)
        );
        if (rv < 0) {
            throw make_errno_error("pwrite(O_DIRECT) failed");
        }
        if (static_cast<size_t>(rv) != bytes) {
            throw_io_error("short direct write: " + path_);
        }
        next_offset_ += static_cast<uint64_t>(bytes);
        current_fill_ = 0U;
    }

    std::string path_;
    int fd_ = -1;
    uint64_t logical_bytes_ = 0ULL;
    uint64_t aligned_bytes_ = 0ULL;
    DirectIoConfig config_{};
    size_t chunk_bytes_ = 0U;
    void *scratch_ = nullptr;
    size_t current_fill_ = 0U;
    uint64_t next_offset_ = 0ULL;
    uint64_t written_bytes_ = 0ULL;
};

#endif

} // namespace

DirectIoConfig normalize_direct_io_config(DirectIoConfig config) {
    config.queue_depth = std::max<uint32_t>(1U, std::min<uint32_t>(32U, config.queue_depth));
    if (config.chunk_mib != 4U && config.chunk_mib != 8U && config.chunk_mib != 16U) {
        config.chunk_mib = 8U;
    }
    return config;
}

DirectIoConfig direct_io_config_from_options(const RunOptions &options) {
    DirectIoConfig config;
    config.enabled = options.direct_io;
    config.queue_depth = static_cast<uint32_t>(std::max(1, options.direct_io_queue_depth));
    config.chunk_mib = static_cast<uint32_t>(std::max(1, options.direct_io_chunk_mib));
    return normalize_direct_io_config(config);
}

std::string temp_write_path(const std::string &final_path) {
    return fs::path(final_path + ".tmp").string();
}

void finalize_temporary_file(const std::string &temp_path, const std::string &final_path) {
#ifdef _WIN32
    const std::wstring temp = fs::path(temp_path).wstring();
    const std::wstring final = fs::path(final_path).wstring();
    if (MoveFileExW(temp.c_str(), final.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        return;
    }
    const DWORD move_error = GetLastError();
    std::error_code remove_error;
    fs::remove(final_path, remove_error);
    std::error_code rename_error;
    fs::rename(temp_path, final_path, rename_error);
    if (!rename_error) {
        return;
    }
    throw std::runtime_error(
        "failed to finalize temporary file: " + final_path +
        " (win32=" + std::to_string(static_cast<unsigned long>(move_error)) + ")"
    );
#else
    std::error_code remove_error;
    fs::remove(final_path, remove_error);
    std::error_code rename_error;
    fs::rename(temp_path, final_path, rename_error);
    if (rename_error) {
        throw std::runtime_error("failed to finalize temporary file: " + final_path);
    }
#endif
}

class DirectAppendWriter::Impl {
public:
    Impl(const std::string &path, uint64_t logical_bytes, DirectIoConfig config)
        : final_path_(path),
          temp_path_(temp_write_path(path)),
          logical_bytes_(logical_bytes),
          config_(normalize_direct_io_config(config)) {
        if (!config_.enabled) {
            buffered_writer_ = std::make_unique<BufferedAppendWriter>(final_path_);
            return;
        }
        std::error_code remove_error;
        fs::remove(temp_path_, remove_error);
#ifdef _WIN32
        direct_writer_ = std::make_unique<DirectFileWriterWin32>(temp_path_, logical_bytes_, config_);
#elif defined(__linux__)
        direct_writer_ = std::make_unique<DirectFileWriterPosix>(temp_path_, logical_bytes_, config_);
#else
        throw_io_error("direct I/O is unsupported on this platform");
#endif
    }

    void append(const void *src, size_t bytes) {
        if (closed_) {
            throw_io_error("append on closed writer: " + final_path_);
        }
        if (written_bytes_ + static_cast<uint64_t>(bytes) > logical_bytes_) {
            throw_io_error("writer overflow: " + final_path_);
        }
        if (direct_writer_) {
            direct_writer_->append(src, bytes);
        } else {
            buffered_writer_->append(src, bytes);
        }
        written_bytes_ += static_cast<uint64_t>(bytes);
    }

    void close() {
        if (closed_) {
            return;
        }
        if (written_bytes_ != logical_bytes_) {
            throw_io_error("writer byte count mismatch: " + final_path_);
        }
        if (direct_writer_) {
            direct_writer_->close();
            direct_writer_.reset();
            finalize_temporary_file(temp_path_, final_path_);
        } else if (buffered_writer_) {
            buffered_writer_->close();
            buffered_writer_.reset();
        }
        closed_ = true;
    }

private:
    std::string final_path_;
    std::string temp_path_;
    uint64_t logical_bytes_ = 0ULL;
    uint64_t written_bytes_ = 0ULL;
    DirectIoConfig config_{};
    bool closed_ = false;
    std::unique_ptr<BufferedAppendWriter> buffered_writer_;
#ifdef _WIN32
    std::unique_ptr<DirectFileWriterWin32> direct_writer_;
#elif defined(__linux__)
    std::unique_ptr<DirectFileWriterPosix> direct_writer_;
#else
    std::nullptr_t direct_writer_ = nullptr;
#endif
};

class DirectSequentialReader::Impl {
public:
    Impl(const std::string &path, uint64_t logical_bytes, DirectIoConfig config)
        : path_(path),
          logical_bytes_(logical_bytes),
          config_(normalize_direct_io_config(config)) {
        if (!config_.enabled) {
            buffered_reader_ = std::make_unique<BufferedSequentialReader>(path_);
            return;
        }
#ifdef _WIN32
        direct_reader_ = std::make_unique<DirectFileReaderWin32>(path_, logical_bytes_, config_);
#elif defined(__linux__)
        direct_reader_ = std::make_unique<DirectFileReaderPosix>(path_, logical_bytes_, config_);
#else
        throw_io_error("direct I/O is unsupported on this platform");
#endif
    }

    void read(void *dst, size_t bytes) {
        if (closed_) {
            throw_io_error("read on closed reader: " + path_);
        }
        if (read_bytes_ + static_cast<uint64_t>(bytes) > logical_bytes_) {
            throw_io_error("reader overflow: " + path_);
        }
        if (direct_reader_) {
            direct_reader_->read(dst, bytes);
        } else {
            buffered_reader_->read(dst, bytes);
        }
        read_bytes_ += static_cast<uint64_t>(bytes);
    }

    void close() {
        if (closed_) {
            return;
        }
        if (read_bytes_ != logical_bytes_) {
            throw_io_error("reader byte count mismatch: " + path_);
        }
        if (direct_reader_) {
            direct_reader_->close();
            direct_reader_.reset();
        } else if (buffered_reader_) {
            buffered_reader_->close();
            buffered_reader_.reset();
        }
        closed_ = true;
    }

private:
    std::string path_;
    uint64_t logical_bytes_ = 0ULL;
    uint64_t read_bytes_ = 0ULL;
    DirectIoConfig config_{};
    bool closed_ = false;
    std::unique_ptr<BufferedSequentialReader> buffered_reader_;
#ifdef _WIN32
    std::unique_ptr<DirectFileReaderWin32> direct_reader_;
#elif defined(__linux__)
    std::unique_ptr<DirectFileReaderPosix> direct_reader_;
#else
    std::nullptr_t direct_reader_ = nullptr;
#endif
};

DirectAppendWriter::DirectAppendWriter()
    : impl_(nullptr) {}

DirectAppendWriter::DirectAppendWriter(const std::string &path, uint64_t logical_bytes, DirectIoConfig config)
    : impl_(std::make_unique<Impl>(path, logical_bytes, config)) {}

DirectAppendWriter::~DirectAppendWriter() = default;
DirectAppendWriter::DirectAppendWriter(DirectAppendWriter &&other) noexcept = default;
DirectAppendWriter &DirectAppendWriter::operator=(DirectAppendWriter &&other) noexcept = default;

void DirectAppendWriter::append(const void *src, size_t bytes) {
    if (!impl_) {
        throw_io_error("writer is not open");
    }
    impl_->append(src, bytes);
}

void DirectAppendWriter::close() {
    if (!impl_) {
        return;
    }
    impl_->close();
}

DirectSequentialReader::DirectSequentialReader()
    : impl_(nullptr) {}

DirectSequentialReader::DirectSequentialReader(const std::string &path, uint64_t logical_bytes, DirectIoConfig config)
    : impl_(std::make_unique<Impl>(path, logical_bytes, config)) {}

DirectSequentialReader::~DirectSequentialReader() = default;
DirectSequentialReader::DirectSequentialReader(DirectSequentialReader &&other) noexcept = default;
DirectSequentialReader &DirectSequentialReader::operator=(DirectSequentialReader &&other) noexcept = default;

void DirectSequentialReader::read(void *dst, size_t bytes) {
    if (!impl_) {
        throw_io_error("reader is not open");
    }
    impl_->read(dst, bytes);
}

void DirectSequentialReader::close() {
    if (!impl_) {
        return;
    }
    impl_->close();
}

} // namespace FileIOUtils
