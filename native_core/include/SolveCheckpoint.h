#pragma once

#include "FileIOUtils.h"
#include "FormationRuntime.h"
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <type_traits>

// Exact recovery snapshots are separate from reader-visible, possibly pruned
// results. Generated positions below the committed frontier remain available.
namespace SolveCheckpoint {

inline void remove_other_copies(const RunOptions &options, int step, const char *suffix, const std::string &published) {
    const auto destination = NativePath::from_utf8(published).lexically_normal();
    for (const auto &prefix : StoragePaths::candidate_pathnames(options, false)) {
        const auto other = NativePath::from_utf8(prefix + std::to_string(logical_layer(options, step)) + suffix);
        std::error_code ec;
        if (other.lexically_normal() != destination && !std::filesystem::equivalent(other, destination, ec))
            std::filesystem::remove_all(other);
    }
}

inline std::string copy_directory(const std::string &source, const RunOptions &options, int step, const char *suffix) {
    const auto src = NativePath::from_utf8(source);
    uint64_t bytes = 0;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(src))
        if (entry.is_regular_file()) bytes += entry.file_size();
    auto lease = StoragePaths::reserve_write_path(options, StoragePaths::ArtifactRole::Hot, step, suffix, bytes);
    const auto dst = NativePath::from_utf8(lease.path());
    const auto tmp = NativePath::from_utf8(lease.path() + ".tmp");
    std::filesystem::remove_all(tmp);
    std::filesystem::copy(src, tmp, std::filesystem::copy_options::recursive);
    std::filesystem::remove_all(dst);
    std::filesystem::rename(tmp, dst);
    remove_other_copies(options, step, suffix, lease.path());
    return lease.path();
}

inline std::string copy_layer(const std::string &source, const RunOptions &options, int step, const char *suffix) {
    const auto bytes = NativePath::file_size(source);
    auto lease = StoragePaths::reserve_write_path(options, StoragePaths::ArtifactRole::Hot, step, suffix, bytes);
    std::ifstream in(NativePath::from_utf8(source), std::ios::binary);
    FileIOUtils::DirectAppendWriter out(lease.path(), bytes, {});
    std::vector<char> buffer(1024 * 1024);
    uint64_t left = bytes;
    while (left) {
        const auto n = static_cast<size_t>(std::min<uint64_t>(left, buffer.size()));
        in.read(buffer.data(), n);
        if (!in) throw std::runtime_error("failed to copy exact solve frontier: " + source);
        out.append(buffer.data(), n);
        left -= n;
    }
    out.close();
    // Restoring onto a fallback drive must not leave a pruned hot copy that
    // normal path lookup would choose instead of the restored exact data.
    remove_other_copies(options, step, suffix, lease.path());
    return lease.path();
}

class State {
    const RunOptions &options_;
    std::string algorithm_, identity_, marker_, suffix_;
    int frontier_ = -2;
    std::array<std::string, 2> files_{};

    std::string marker_for(const std::string &prefix) const {
        return prefix + "solve-" + algorithm_ + ".checkpoint";
    }
    std::string snapshot_prefix(const std::string &prefix) const {
        const auto path = NativePath::from_utf8(prefix);
        return NativePath::to_utf8_string(path.parent_path() /
            NativePath::from_utf8("." + NativePath::to_utf8_string(path.filename()) + "solve-" + algorithm_) / "layer_");
    }
    static void remove_file(const std::string &path) {
        if (path.empty()) return;
        std::error_code ec;
        std::filesystem::remove_all(NativePath::from_utf8(path), ec);
    }
    void publish(int frontier, const std::array<std::string, 2> &files) {
        std::vector<std::string> candidates;
        for (const auto &prefix : StoragePaths::candidate_pathnames(options_, false)) candidates.push_back(marker_for(prefix));
        auto lease = StoragePaths::reserve_write_path_from_candidates(candidates, 16384);
        const std::string path = lease.path();
        const std::string tmp = path + ".tmp";
        std::ofstream out(NativePath::from_utf8(tmp), std::ios::trunc | std::ios::binary);
        out << "SOLVE_CHECKPOINT_1\n" << std::quoted(identity_) << '\n'
            << frontier << '\n' << std::quoted(files[0]) << '\n' << std::quoted(files[1]) << '\n';
        out.close();
        if (!out) throw std::runtime_error("failed to write solve checkpoint: " + tmp);
        FileIOUtils::finalize_temporary_file(tmp, path);
        const auto previous_files = files_;
        const std::string previous_marker = marker_;
        frontier_ = frontier;
        files_ = files;
        marker_ = path;
        test_stop("PUBLISHED", frontier);
        for (const auto &old : previous_files)
            if (old != files[0] && old != files[1]) remove_file(old);
        if (previous_marker != marker_) remove_file(previous_marker);
    }

public:
    bool enabled = false;

    template<class Spec>
    State(const RunOptions &options, const Spec &spec, const char *algorithm, const char *suffix = ".book")
        : options_(options), algorithm_(algorithm), suffix_(suffix) {
        std::ostringstream id;
        id << std::setprecision(17) << algorithm << ' ' << options.steps << ' ' << options.layer_offset
           << ' ' << options.target << ' ' << options.sum_target << ' ' << options.spawn_rate4
           << ' ' << options.success_rate_dtype << ' ' << options.is_variant << ' ' << options.is_free
           << ' ' << options.docheck_step << ' ' << spec.name << ' ' << spec.symm_mode
           << ' ' << unsigned(spec.physical_transform) << ' ' << spec.physical_pattern_signature;
        for (auto mask : spec.pattern_masks) id << ' ' << mask;
        id << " shifts";
        for (auto shift : spec.success_shifts) id << ' ' << unsigned(shift);
        if constexpr (std::is_same_v<Spec, AdvancedPatternSpec>) {
            id << " advanced " << unsigned(spec.num_free_32k) << ' ' << spec.small_tile_sum_limit
               << ' ' << unsigned(spec.target);
            for (auto shift : spec.fixed_32k_shifts) id << ' ' << unsigned(shift);
        }
        identity_ = id.str();
        for (const auto &prefix : StoragePaths::candidate_pathnames(options, false)) {
            const std::string path = marker_for(prefix);
            if (!StoragePaths::filename_exists(path)) continue;
            std::ifstream in(NativePath::from_utf8(path), std::ios::binary);
            std::string magic, identity;
            int frontier;
            std::array<std::string, 2> files;
            if (!(in >> magic >> std::quoted(identity) >> frontier >> std::quoted(files[0]) >> std::quoted(files[1])) ||
                magic != "SOLVE_CHECKPOINT_1" || identity != identity_ || frontier < -1 || frontier >= options.steps)
                throw std::runtime_error("invalid or incompatible solve checkpoint: " + path);
            if (frontier_ == -2 || frontier < frontier_) {
                frontier_ = frontier;
                files_ = std::move(files);
                marker_ = path;
            }
        }
        if (has_frontier()) {
            for (const auto &path : files_)
                if (path.empty() || !StoragePaths::filename_exists(path))
                    throw std::runtime_error("missing exact solve checkpoint file: " + path);
            if (options.optimal_branch_only)
                throw std::runtime_error("finish the checkpointed solve before enabling optimal-branch pruning");
        }
        enabled = exists() || (options.solve_checkpoint_interval > 1 && !options.optimal_branch_only &&
            (options.compress || options.deletion_threshold > 0 || options.relative_deletion_threshold > 0));
    }

    bool exists() const { return frontier_ != -2; }
    bool completed() const { return frontier_ == -1; }
    bool has_frontier() const { return frontier_ >= 0; }
    int frontier() const { return frontier_; }
    const std::string &file(int index) const { return files_.at(index); }
    bool due(int step) const {
        return enabled && step > 0 && (!has_frontier() || frontier_ - step >= std::max(1, options_.solve_checkpoint_interval));
    }
    RunOptions snapshot_options() const {
        RunOptions result = options_;
        result.pathname = snapshot_prefix(result.pathname);
        for (auto &path : result.cold_pathnames) path = snapshot_prefix(path);
        if (enabled) {
            for (const auto &prefix : StoragePaths::candidate_pathnames(result, false)) {
                std::error_code ec;
                std::filesystem::create_directories(NativePath::from_utf8(prefix).parent_path(), ec);
            }
        }
        return result;
    }
    template<class Clean>
    void cleanup(Clean clean, bool positions = true) const {
        if (!exists()) return;
        if (positions)
            for (int step = std::max(0, frontier_); step < options_.steps; ++step) clean(step);
        for (const auto &prefix : StoragePaths::candidate_pathnames(options_, false)) {
            const auto directory = NativePath::from_utf8(snapshot_prefix(prefix)).parent_path();
            std::error_code ec;
            std::filesystem::directory_iterator it(directory, ec), end;
            while (!ec && it != end) {
                const auto path = NativePath::to_utf8_string(it->path());
                const auto name = NativePath::to_utf8_string(it->path().filename());
                // Only this protocol's own snapshot and interrupted-write files.
                const auto ends_with = [&](const std::string &suffix) {
                    return name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
                };
                const bool snapshot = name.compare(0, 6, "layer_") == 0 &&
                    (ends_with(suffix_) || ends_with(suffix_ + ".tmp"));
                const auto normalized = it->path().lexically_normal();
                if (snapshot && normalized != NativePath::from_utf8(files_[0]).lexically_normal() &&
                    normalized != NativePath::from_utf8(files_[1]).lexically_normal()) remove_file(path);
                it.increment(ec);
            }
        }
    }
    template<class Write, class Clean>
    void save(int frontier, Write write, Clean clean) {
        const int previous = has_frontier() ? frontier_ : options_.steps;
        std::array<std::string, 2> files;
        // Interval 1 overlaps the committed pair. Keep that snapshot immutable
        // until the new marker is published, including during a failed write.
        auto snapshot = [&](int index) {
            const int old_index = frontier + index - frontier_;
            return has_frontier() && old_index >= 0 && old_index < 2
                ? files_[old_index] : write(index);
        };
        files[0] = snapshot(0);
        test_stop("FIRST_FILE", frontier);
        files[1] = snapshot(1);
        test_stop("PAIR", frontier);
        publish(frontier, files);
        for (int step = frontier; step < previous; ++step) clean(step);
        cleanup(clean, false);
    }
    template<class Clean>
    void finish(Clean clean) {
        const int previous = has_frontier() ? frontier_ : options_.steps;
        publish(-1, {});
        for (int step = 0; step < previous; ++step) clean(step);
        cleanup(clean);
    }
    static void test_stop(const char *phase, int step) {
        const std::string key = std::string("SOLVE_TEST_STOP_") + phase;
        if (const char *value = std::getenv(key.c_str()); value && std::to_string(step) == value)
            throw std::runtime_error("solve checkpoint test stop: " + std::string(phase));
    }
};

} // namespace SolveCheckpoint
