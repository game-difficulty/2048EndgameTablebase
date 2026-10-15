// Included inside EXPrefix36Runtime's implementation namespace. Both split
// routes use the normal EX kernel, bitmap compaction and on-disk layer format.
uint64_t ex_solve_env_bytes(const char *name) {
    const char *value = std::getenv(name);
    if (!value || !*value) return 0;
    char *end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (*end) throw std::invalid_argument(std::string("invalid byte limit: ") + name);
    return parsed;
}

uint64_t ex_available_memory() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) return status.ullAvailPhys;
#else
    std::ifstream in("/proc/meminfo");
    std::string key, rest;
    uint64_t kib = 0;
    while (in >> key >> kib) {
        std::getline(in, rest);
        if (key == "MemAvailable:") return kib * 1024ULL;
    }
    struct sysinfo info{};
    if (sysinfo(&info) == 0) return static_cast<uint64_t>(info.freeram) * info.mem_unit;
#endif
    return 0;
}

// Archives must be seekable for bounded bucket reads. Extract through the
// existing streaming decoder, never via a full decompressed byte vector.
std::string ex_seekable_input(const RunOptions &options, int step, bool positions_only = false) {
    if (!positions_only) materialize_compressed_layer_input(options, step);
    std::string path = positions_only ? existing_generated_layer_file_path(options, step)
                                      : existing_layer_input_path(options, step);
    if (positions_only && path.empty()) path = existing_generated_layer_archive_path(options, step);
    if (path.empty()) throw std::runtime_error("missing EX generated positions at step " + std::to_string(step));
    if (!has_archive_suffix(path)) return path;
    SevenZipSequentialReader in(path);
    LayerFileHeader header{};
    in.read(&header, sizeof(header));
    validate_layer_header(header, path);
    auto lease = StoragePaths::reserve_write_path(options, StoragePaths::ArtifactRole::Hot,
        step, kGeneratedLayerFileExtension, layer_file_bytes(header));
    // Buffered output is intentional: this is a small, sequential staging copy.
    FileIOUtils::DirectAppendWriter out(lease.path(), layer_file_bytes(header), {});
    out.append(&header, sizeof(header));
    std::vector<char> buffer(1024U * 1024U);
    uint64_t left = layer_file_bytes(header) - sizeof(header);
    while (left) {
        const size_t n = static_cast<size_t>(std::min<uint64_t>(left, buffer.size()));
        in.read(buffer.data(), n);
        out.append(buffer.data(), n);
        left -= n;
    }
    in.close();
    out.close();
    return lease.path();
}

LayerFileHeader ex_input_header(const std::string &path, const LutBundle &lut) {
    std::ifstream in(NativePath::from_utf8(path), std::ios::binary);
    const auto header = read_one_at<LayerFileHeader>(in, 0, path);
    validate_layer_header(header, path);
    if (!physical_metadata_matches(header, lut) || NativePath::file_size(path) < layer_file_bytes(header))
        throw std::runtime_error("EX split input metadata/size mismatch: " + path);
    return header;
}

uint64_t ex_layer_resident_estimate(const LayerFileHeader &header) {
    // Includes rank metadata and the direct query index; generated inputs have
    // no success payload yet, but will acquire one during solving.
    const uint64_t index_bytes = header.live_board_count == 0 ? 0 :
        choose_direct_table_size(header.bucket_count) * (sizeof(uint32_t) + sizeof(BucketEntry));
    return layer_file_bytes(header) + header.bucket_count * (sizeof(uint32_t) + sizeof(BucketEntry)) +
        index_bytes + header.large_bitmap_words * sizeof(uint16_t) +
        (header.success_value_count ? 0ULL : layer_success_storage_count(header) * header.value_size);
}

template <typename T>
class EXLayerChunks {
public:
    EXLayerChunks(const std::string &path, const LutBundle &lut)
        : header(ex_input_header(path, lut)), path_(path), sizes_(lut.dense_lut.size_table) {
        const auto layout = layer_file_layout(header);
        offsets_ = {layout.bucket_keys_offset, layout.bitmap_offsets_offset, layout.success_offsets_offset,
                    layout.small_bitmap_offset, layout.large_bitmap_offset, layout.success_values_offset};
        for (auto &in : streams_) {
            in.open(NativePath::from_utf8(path), std::ios::binary);
            if (!in) throw std::runtime_error("failed to open EX split input: " + path);
        }
    }
    EXLayerChunks(const Prefix36LayerT<T> &layer, const PatternSpec &spec, DTypeMode mode, const LutBundle &lut)
        : header(make_layer_header(layer, spec, mode)), sizes_(lut.dense_lut.size_table), memory_(&layer) {}

    void close() { for (auto &in : streams_) in.close(); }

    bool next(Prefix36LayerT<T> &layer, uint64_t budget) {
        layer = {};
        layer.layer_sum = header.layer_sum;
        layer.threshold_bits = header.threshold_bits;
        layer.success_stride = layer_success_stride(header);
        if (cursor_ == header.bucket_count) return false;
        uint64_t small_begin = UINT64_MAX, small_end = 0, large_begin = UINT64_MAX, large_end = 0;
        uint64_t value_begin = 0, value_end = 0;
        while (cursor_ < header.bucket_count) {
            load_metadata();
            const size_t i = static_cast<size_t>(cursor_ - metadata_begin_);
            const uint64_t key = keys_[i];
            const uint32_t valid = valid_suffix_count_for_key(sizes_, key, path_);
            const bool small = valid <= header.threshold_bits;
            const uint64_t units = small ? bytes_for_bits(valid) : words_for_bits(valid);
            const uint64_t end = static_cast<uint64_t>(bitmaps_[i]) + units;
            const uint64_t next_small_begin = small ? std::min(small_begin, uint64_t(bitmaps_[i])) : small_begin;
            const uint64_t next_large_begin = small ? large_begin : std::min(large_begin, uint64_t(bitmaps_[i]));
            const uint64_t next_small_end = small ? end : small_end;
            const uint64_t next_large_end = small ? large_end : end;
            if (layer.bucket_keys.empty()) value_begin = EXSuccessLayout::decode(values_[i], layer.success_stride);
            if (values_[i + 1] < values_[i]) throw std::runtime_error("EX split success offsets are unordered");
            const uint64_t estimate = (layer.bucket_keys.size() + 1) * 80ULL +
                (next_small_begin == UINT64_MAX ? 0 : next_small_end - next_small_begin) +
                (next_large_begin == UINT64_MAX ? 0 : (next_large_end - next_large_begin) * 10ULL) +
                (EXSuccessLayout::decode(values_[i + 1], layer.success_stride) - value_begin) * sizeof(T);
            if (!layer.bucket_keys.empty() && estimate > budget) break;
            if (!memory_ && layer.bucket_keys.empty() && estimate > budget &&
                !ex_solve_env_bytes("EX_TEST_CURRENT_CHUNK_BYTES"))
                throw std::runtime_error("EX split solve cannot fit a single bucket in the current chunk budget");
            small_begin = next_small_begin; small_end = next_small_end;
            large_begin = next_large_begin; large_end = next_large_end;
            value_end = EXSuccessLayout::decode(values_[i + 1], layer.success_stride);
            layer.bucket_keys.push_back(key);
            layer.bitmap_offsets.push_back(bitmaps_[i]);
            layer.success_offsets.push_back(static_cast<uint32_t>(values_[i] - value_begin / layer.success_stride));
            ++cursor_;
        }
        if (small_begin != UINT64_MAX) {
            if (small_end > header.small_bitmap_bytes) throw std::runtime_error("EX split bitmap range mismatch");
            layer.small_bitmap_bytes.resize(small_end - small_begin);
            read(3, small_begin, layer.small_bitmap_bytes.data(), layer.small_bitmap_bytes.size());
        }
        if (large_begin != UINT64_MAX) {
            if (large_end > header.large_bitmap_words) throw std::runtime_error("EX split bitmap range mismatch");
            layer.large_bitmap_words.resize(large_end - large_begin);
            read(4, large_begin * 8, layer.large_bitmap_words.data(), layer.large_bitmap_words.size() * 8);
        }
        for (size_t i = 0; i < layer.bucket_keys.size(); ++i) {
            const uint32_t valid = valid_suffix_count_for_key(sizes_, layer.bucket_keys[i], path_);
            const bool small = valid <= header.threshold_bits;
            layer.bitmap_offsets[i] -= static_cast<uint32_t>(small ? small_begin : large_begin);
            layer.live_board_count += bucket_live_count(layer, i, valid);
        }
        layer.success_storage_count = value_end - value_begin;
        if (value_end > layer_success_storage_count(header)) throw std::runtime_error("EX split value range mismatch");
        if (header.success_value_count) {
            if (header.value_size != sizeof(T) || value_end > header.success_value_count)
                throw std::runtime_error("EX split dtype/value count mismatch");
            layer.success_values.resize(layer.success_storage_count);
            read(5, value_begin * sizeof(T), layer.success_values.data(), layer.success_values.size() * sizeof(T));
        }
        return true;
    }
    LayerFileHeader header;
private:
    void read(size_t section, uint64_t offset, void *dst, size_t bytes) {
        if (!bytes) return;
        if (memory_) {
            const std::array<const void *, 6> sources = {memory_->bucket_keys.data(), memory_->bitmap_offsets.data(),
                memory_->success_offsets.data(), memory_->small_bitmap_bytes.data(), memory_->large_bitmap_words.data(),
                memory_->success_values.data()};
            std::memcpy(dst, static_cast<const char *>(sources[section]) + offset, bytes);
        } else {
            auto &in = streams_[section];
            in.seekg(static_cast<std::streamoff>(offsets_[section] + offset));
            read_stream_exact(in, dst, bytes, path_);
        }
    }
    void load_metadata() {
        if (cursor_ < metadata_begin_ + keys_.size()) return;
        metadata_begin_ = cursor_;
        const size_t n = static_cast<size_t>(std::min<uint64_t>(4096, header.bucket_count - cursor_));
        keys_.resize(n); bitmaps_.resize(n); values_.resize(n + 1);
        read(0, cursor_ * 8, keys_.data(), n * 8);
        read(1, cursor_ * 4, bitmaps_.data(), n * 4);
        read(2, cursor_ * 4, values_.data(), n * 4);
        values_[n] = cursor_ + n < header.bucket_count ? 0 :
            static_cast<uint32_t>(layer_success_storage_count(header) / layer_success_stride(header));
        if (cursor_ + n < header.bucket_count) read(2, (cursor_ + n) * 4, &values_[n], 4);
    }
    std::string path_;
    const std::vector<uint32_t> &sizes_;
    const Prefix36LayerT<T> *memory_ = nullptr;
    std::array<std::ifstream, 6> streams_;
    std::array<uint64_t, 6> offsets_{};
    uint64_t cursor_ = 0, metadata_begin_ = 0;
    std::vector<uint64_t> keys_;
    std::vector<uint32_t> bitmaps_, values_;
};

template <typename T>
class EXChunkOutput {
public:
    EXChunkOutput(const std::string &path, const LayerFileHeader &header, const std::vector<uint32_t> &sizes)
        : writer_(path), header_(header), sizes_(sizes) {
        header_.bucket_count = header_.live_board_count = header_.success_value_count = 0;
        header_.small_bitmap_bytes = header_.large_bitmap_words = 0;
    }
    void append(const Prefix36LayerT<T> &layer) {
        if (layer.bucket_keys.empty()) return;
        if (layer.success_stride != layer_success_stride(header_))
            throw std::runtime_error("EX split output stride mismatch");
        std::vector<uint32_t> bitmaps(layer.bitmap_offsets), values(layer.success_offsets);
        for (size_t i = 0; i < bitmaps.size(); ++i) {
            const bool small = sizes_[sum_index(key_remaining_sum(layer.bucket_keys[i]))] <= layer.threshold_bits;
            const uint64_t bitmap = bitmaps[i] + (small ? writer_.size(3) : writer_.size(4) / 8);
            const uint64_t value = uint64_t(values[i]) + header_.success_value_count / layer.success_stride;
            if (bitmap > UINT32_MAX || value > UINT32_MAX) throw std::runtime_error("EX split output offset overflow");
            bitmaps[i] = static_cast<uint32_t>(bitmap);
            values[i] = static_cast<uint32_t>(value);
        }
        writer_.append(0, layer.bucket_keys.data(), layer.bucket_keys.size() * 8);
        writer_.append(1, bitmaps.data(), bitmaps.size() * 4);
        writer_.append(2, values.data(), values.size() * 4);
        writer_.append(3, layer.small_bitmap_bytes.data(), layer.small_bitmap_bytes.size());
        writer_.append(4, layer.large_bitmap_words.data(), layer.large_bitmap_words.size() * 8);
        writer_.append(5, layer.success_values.data(), layer.success_values.size() * sizeof(T));
        header_.bucket_count += layer.bucket_keys.size();
        header_.live_board_count += layer.live_board_count;
        header_.success_value_count += layer.success_values.size();
    }
    void finish(FileIOUtils::DirectIoConfig io) {
        finish_header();
        writer_.finish(header_, io);
    }
    void seal_sections() {
        finish_header();
        writer_.seal_sections(header_);
    }
    const LayerFileHeader &header() const { return header_; }
private:
    void finish_header() {
        header_.reserved1 = layer_success_stride(header_) == 1 ? 0 : header_.success_value_count;
        header_.small_bitmap_bytes = writer_.size(3);
        header_.large_bitmap_words = writer_.size(4) / 8;
    }
    EXSectionWriter writer_;
    LayerFileHeader header_;
    const std::vector<uint32_t> &sizes_;
};

#include "EXRetiredLayer.inl"

struct EXSplitWork {
    std::string partial;
    ~EXSplitWork() {
        if (partial.empty()) return;
        std::error_code ec;
        NativePath::remove(partial, ec);
    }
};

void ex_split_test_stop(const char *phase, int step) {
    const std::string key = std::string("EX_TEST_STOP_AFTER_") + phase + "_STEP";
    const char *value = std::getenv(key.c_str());
    if (value && std::stoi(value) == step) throw std::runtime_error("EX split test stop: " + key);
}

template <int Spawn, typename T>
void ex_split_pass(Prefix36LayerT<T> &current, const Prefix36LayerT<T> &future,
                   const PatternSpec &spec, const RunOptions &options, const LutBundle &lut, int step) {
    if (options.is_variant)
        recalculate_current_layer<VBoardMover, T, Spawn>(current, future, future, lut.dense_lut, lut.row_luts,
            spec, options, step > options.docheck_step, thread_count_from_options(options));
    else
        recalculate_current_layer<BoardMover, T, Spawn>(current, future, future, lut.dense_lut, lut.row_luts,
            spec, options, step > options.docheck_step, thread_count_from_options(options));
}

struct EXSplitPipeline {
    int step = kNoSolveResumeStep;
    std::string partial;
    LayerFileHeader future4_header{};
    std::unique_ptr<EXRetiredLayer> retired;
    std::string checkpoint_future;
    ~EXSplitPipeline() {
        if (!partial.empty()) {
            std::error_code ec;
            NativePath::remove(partial, ec);
        }
    }
    bool ready(int current_step) const { return step == current_step && !partial.empty(); }
};

template <typename T, typename LoadFuture>
SolveStepSummary solve_split_step(const PatternSpec &spec, const RunOptions &options, const LutBundle &lut,
    int step, const std::string &current_path, uint64_t chunk_budget, bool resident_current,
    Prefix36LayerT<T> &future, int &cached_step, LoadFuture &&load_future,
    RuntimeControls::DeletionThresholdState thresholds, bool retain_positions = false,
    Prefix36LayerT<T> *retained_current = nullptr, EXSplitPipeline *pipeline_state = nullptr,
    const std::string &next_path = {},
    const std::function<void(const Prefix36LayerT<T> &)> &save_future = {}) {
    const double started = now_seconds();
    const auto mode = dtype_mode_from_name(options.success_rate_dtype);
    const auto io = FileIOUtils::direct_io_config_from_options(options);
    const int threads = thread_count_from_options(options);
    const T zero = zero_value_for_dtype<T>(options.success_rate_dtype);
    SolveStepSummary summary;
    EXSplitPipeline local_pipeline;
    auto &pipeline = pipeline_state ? *pipeline_state : local_pipeline;
    const bool prepared = pipeline.ready(step);
    const auto header = ex_input_header(current_path, lut);
    if (dtype_mode_from_header(header) != mode)
        throw std::runtime_error("EX split solve dtype does not match input; resume with the original dtype");
    summary.input_live = header.live_board_count;
    auto lease = StoragePaths::ReservedWritePath{};
    if (!resident_current || !retained_current)
        lease = StoragePaths::reserve_write_path(options, StoragePaths::ArtifactRole::Hot, step,
            kLayerFileExtension, 3 * (layer_file_bytes(header) + layer_success_storage_count(header) * sizeof(T)));
    EXSplitWork work{prepared ? std::move(pipeline.partial) :
        resident_current ? std::string{} : lease.path() + ".spawn4.tmp"};
    pipeline.partial.clear();
    pipeline.checkpoint_future.clear();
    auto retired = std::move(pipeline.retired);
    Prefix36LayerT<T> current;
    double t0;
    const auto read_chunk = [&](auto &reader, auto &chunk) {
        const double t = now_seconds();
        const bool found = reader.next(chunk, chunk_budget);
        summary.read_seconds += now_seconds() - t;
        return found;
    };
    const auto index_future = [&] {
        const double t = now_seconds();
        ensure_direct_index_built(future, lut.dense_lut.size_table);
        summary.future_index_seconds += now_seconds() - t;
    };
    const auto calculate4 = [&](auto &chunk, int target_step) {
        const double t = now_seconds();
        ex_split_pass<2>(chunk, future, spec, options, lut, target_step);
        summary.recalc_seconds += now_seconds() - t;
    };
    const auto spool4 = [&](const std::string &source, const std::string &output, int target_step) {
        EXLayerChunks<T> positions(source, lut);
        std::ofstream values(NativePath::from_utf8(output), std::ios::binary | std::ios::trunc);
        Prefix36LayerT<T> chunk;
        while (read_chunk(positions, chunk)) {
            calculate4(chunk, target_step);
            const double t = now_seconds();
            FileIOUtils::write_exact(values, chunk.success_values.data(), chunk.success_values.size() * sizeof(T), output);
            summary.current_write_seconds += now_seconds() - t;
        }
        values.close();
        if (!values) throw std::runtime_error("failed to close EX partial values: " + output);
    };
    if (resident_current) {
        t0 = now_seconds();
        current = read_layer_file<T>(current_path, nullptr, io, &lut.dense_lut.size_table, threads);
        summary.read_seconds += now_seconds() - t0;
    }
    if (!prepared) {
        if (cached_step != step + 2) future = {};
        summary.read_seconds += load_future(step + 2, future, cached_step);
        index_future();
        if (resident_current) calculate4(current, step);
        else spool4(current_path, work.partial, step);
        retired = std::make_unique<EXRetiredLayer>();
        stage_retired_layer(*retired, future, spec, options, lut, step + 2, chunk_budget, thresholds, retain_positions, true);
        summary.future_compact_seconds += retired->compact_seconds;
        summary.future_write_seconds += retired->write_seconds;
        future = {};
        cached_step = kNoSolveResumeStep;
    }
    ex_split_test_stop("SPLIT_FIRST_PASS", step);
    summary.read_seconds += load_future(step + 1, future, cached_step);
    index_future();
    const uint64_t future1_live = future.live_board_count;

    // Complete both consumers of this successor before retiring it. The next
    // +4 contribution is disposable; exact checkpoints never depend on it.
    if (!next_path.empty()) {
        const auto next_header = ex_input_header(next_path, lut);
        auto partial_lease = StoragePaths::reserve_write_path(options, StoragePaths::ArtifactRole::Hot,
            step - 1, ".spawn4.tmp", layer_success_storage_count(next_header) * sizeof(T));
        pipeline.partial = partial_lease.path();
        spool4(next_path, pipeline.partial, step - 1);
        partial_lease.release();
        pipeline.step = step - 1;
        pipeline.future4_header = make_layer_header(future, spec, mode);
        ex_split_test_stop("PIPELINE_PREFETCH", step);
    } else {
        pipeline.step = kNoSolveResumeStep;
    }
    const auto calculate2 = [&](auto &chunk) {
        const double t = now_seconds();
        ex_split_pass<1>(chunk, future, spec, options, lut, step);
        summary.recalc_seconds += now_seconds() - t;
    };
    const auto compact = [&](auto &chunk) {
        const double t = now_seconds();
        chunk = compact_layer(chunk, lut.dense_lut, zero, threads, chunk.success_stride);
        summary.compact_seconds += now_seconds() - t;
        summary.output_live += chunk.live_board_count;
        summary.bitmap_bits += layer_bitmap_bits(chunk);
    };
    std::unique_ptr<EXChunkOutput<T>> output;
    if (resident_current) {
        if (prepared) {
            t0 = now_seconds();
            std::ifstream values(NativePath::from_utf8(work.partial), std::ios::binary);
            current.success_values.resize(current.success_storage_count);
            read_stream_exact(values, current.success_values.data(), current.success_values.size() * sizeof(T), work.partial);
            if (values.peek() != std::char_traits<char>::eof())
                throw std::runtime_error("EX partial values do not match positions: " + work.partial);
            summary.read_seconds += now_seconds() - t0;
        }
        calculate2(current);
    } else {
        output = std::make_unique<EXChunkOutput<T>>(lease.path(), header, lut.dense_lut.size_table);
        EXLayerChunks<T> positions(current_path, lut);
        std::ifstream values(NativePath::from_utf8(work.partial), std::ios::binary);
        if (!values) throw std::runtime_error("failed to open EX partial values: " + work.partial);
        while (read_chunk(positions, current)) {
            t0 = now_seconds();
            current.success_values.resize(current.success_storage_count);
            read_stream_exact(values, current.success_values.data(), current.success_values.size() * sizeof(T), work.partial);
            summary.read_seconds += now_seconds() - t0;
            calculate2(current);
            compact(current);
            t0 = now_seconds();
            output->append(current);
            summary.current_write_seconds += now_seconds() - t0;
        }
        if (values.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("EX partial values do not match positions: " + work.partial);
    }
    if (save_future) {
        t0 = now_seconds();
        save_future(future);
        summary.current_write_seconds += now_seconds() - t0;
    }
    if (!next_path.empty()) {
        pipeline.retired = std::make_unique<EXRetiredLayer>();
        stage_retired_layer(*pipeline.retired, future, spec, options, lut,
            step + 1, chunk_budget, thresholds, retain_positions, true);
        summary.future_compact_seconds += pipeline.retired->compact_seconds;
        summary.future_write_seconds += pipeline.retired->write_seconds;
    } else if (retain_positions) {
        // The last two layers retain the existing unpruned tail semantics.
        t0 = now_seconds();
        write_routed_layer_file(options, StoragePaths::ArtifactRole::Hot, step + 1,
            kLayerFileExtension, future, spec, mode, io);
        summary.current_write_seconds += now_seconds() - t0;
    }
    future = {};
    cached_step = kNoSolveResumeStep;
    ex_split_test_stop("PIPELINE_RELEASE", step);
    if (resident_current) compact(current);
    ex_split_test_stop("SPLIT_FINAL_CHUNKS", step);
    t0 = now_seconds();
    if (resident_current) {
        if (retained_current) *retained_current = std::move(current);
        else write_layer_file(lease.path(), current, spec, mode, io);
        current = {};
    } else {
        output->finish(io);
    }
    lease.release();
    summary.current_write_seconds += now_seconds() - t0;
    if (!retain_positions) remove_generated_layer_input(options, step);
    t0 = now_seconds();
    retired->publish(options);
    summary.future_write_seconds += now_seconds() - t0;
    ex_split_test_stop("SPLIT_PUBLISH", step);
    summary.bitmap_live = summary.output_live;
    summary.total_seconds = now_seconds() - started;
    summary.compute_seconds = summary.recalc_seconds + summary.compact_seconds +
        summary.future_index_seconds + summary.future_compact_seconds;
    append_solve_stats(options, resident_current ? "split_solve" : "chunked_solve", step,
        summary.input_live, summary.output_live, future1_live, retired->before, retired->after,
        thresholds.absolute, thresholds.relative, retired->effective_threshold,
        RuntimeControls::retention_ratio(summary.output_live, summary.input_live),
        RuntimeControls::retention_ratio(retired->after, retired->before), summary.recalc_seconds,
        summary.compact_seconds, summary.future_index_seconds, summary.future_compact_seconds,
        summary.current_write_seconds, summary.future_write_seconds, summary.read_seconds, summary.total_seconds,
        summary.compute_seconds, nullptr, summary.output_live, summary.bitmap_bits);
    return summary;
}
