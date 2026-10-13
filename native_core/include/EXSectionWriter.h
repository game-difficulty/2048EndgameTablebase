#pragma once

#include "FileIOUtils.h"
#include <array>

// Spool the six contiguous EX file sections without holding a whole layer.
// Only finish() publishes the destination; abandoned sections are disposable.
class EXSectionWriter {
public:
    explicit EXSectionWriter(const std::string &path) : path_(path) {}
    ~EXSectionWriter() {
        for (size_t i = 0; i < streams_.size(); ++i) {
            streams_[i].close();
            std::error_code ec;
            NativePath::remove(section_path(i), ec);
        }
    }
    void append(size_t section, const void *data, size_t bytes) {
        if (!bytes) return;
        auto &out = streams_.at(section);
        if (!out.is_open()) out.open(NativePath::from_utf8(section_path(section)), std::ios::binary | std::ios::trunc);
        out.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
        if (!out) throw std::runtime_error("failed to spool EX section: " + path_);
        sizes_[section] += bytes;
    }
    uint64_t size(size_t section) const { return sizes_.at(section); }
    template <typename Header>
    void finish(const Header &header, FileIOUtils::DirectIoConfig config = {}) {
        uint64_t bytes = sizeof(header);
        for (size_t i = 0; i < streams_.size(); ++i) {
            if (streams_[i].is_open()) {
                streams_[i].close();
                if (streams_[i].fail()) throw std::runtime_error("failed to close EX section: " + path_);
            }
            bytes += sizes_[i];
        }
        const auto write = [&](FileIOUtils::DirectIoConfig io) {
            FileIOUtils::DirectAppendWriter out(path_, bytes, io);
            out.append(&header, sizeof(header));
            std::vector<char> buffer(1024U * 1024U);
            for (size_t i = 0; i < streams_.size(); ++i) {
                if (!sizes_[i]) continue;
                std::ifstream in(NativePath::from_utf8(section_path(i)), std::ios::binary);
                for (uint64_t left = sizes_[i]; left;) {
                    const size_t n = static_cast<size_t>(std::min<uint64_t>(left, buffer.size()));
                    in.read(buffer.data(), n);
                    if (!in) throw std::runtime_error("failed to read EX section: " + path_);
                    out.append(buffer.data(), n);
                    left -= n;
                }
            }
            out.close();
        };
        try { write(config); }
        catch (const FileIOUtils::DirectIoUnsupported &) {
            if (!config.enabled) throw;
            config.enabled = false;
            write(config);
        }
    }
private:
    std::string section_path(size_t i) const { return path_ + ".section" + std::to_string(i) + ".tmp"; }
    std::string path_;
    std::array<std::ofstream, 6> streams_;
    std::array<uint64_t, 6> sizes_{};
};
