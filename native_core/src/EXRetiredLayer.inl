// Included after EXLayerChunks/EXChunkOutput. Staging releases resident memory
// without replacing an exact dependency before its last consumer commits.
struct EXRetiredLayer {
    StoragePaths::ReservedWritePath lease;
    std::string temporary;
    uint64_t before = 0, after = 0;
    double compact_seconds = 0, write_seconds = 0, effective_threshold = 0;
    bool compressed = false;
    int step = 0;

    ~EXRetiredLayer() {
        if (temporary.empty()) return;
        std::error_code ec;
        NativePath::remove(temporary, ec);
        NativePath::remove(FileIOUtils::temp_write_path(temporary), ec);
    }
    void publish(const RunOptions &options) {
        if (temporary.empty()) return;
        FileIOUtils::finalize_temporary_file(temporary, lease.path());
        temporary.clear();
        lease.release();
        if (compressed) {
            std::error_code ec;
            NativePath::remove(existing_raw_layer_file_path(options, step), ec);
        }
    }
};

template <typename T>
void compact_retired_layer_in_place(Prefix36LayerT<T> &layer, const PatternSpec &spec,
    DTypeMode mode, const LutBundle &lut, T threshold, int threads, uint64_t chunk_budget) {
    // Reuse the normal parallel compactor. With the original stride, pruning
    // only shrinks each section, so copying back cannot reach unread buckets.
    EXLayerChunks<T> chunks(layer, spec, mode, lut);
    layer.bucket_entries = std::vector<BucketEntry>{};
    layer.direct_index = {};
    layer.direct_entry_index = {};
    layer.large_rank_offsets = std::vector<uint32_t>{};
    layer.large_rank_bases = std::vector<uint16_t>{};
    uint64_t buckets = 0, small = 0, large = 0, values = 0, live = 0;
    Prefix36LayerT<T> chunk;
    while (chunks.next(chunk, chunk_budget)) {
        chunk = compact_layer(chunk, lut.dense_lut, threshold, threads, layer.success_stride);
        if (chunk.bucket_keys.empty()) continue;
        for (size_t i = 0; i < chunk.bucket_keys.size(); ++i) {
            const bool is_small = lut.dense_lut.size_table[sum_index(key_remaining_sum(chunk.bucket_keys[i]))]
                <= layer.threshold_bits;
            const uint64_t bitmap = chunk.bitmap_offsets[i] + (is_small ? small : large);
            const uint64_t value = uint64_t(chunk.success_offsets[i]) + values / layer.success_stride;
            if (bitmap > UINT32_MAX || value > UINT32_MAX)
                throw std::runtime_error("EX retired layer offset overflow");
            layer.bucket_keys[buckets + i] = chunk.bucket_keys[i];
            layer.bitmap_offsets[buckets + i] = static_cast<uint32_t>(bitmap);
            layer.success_offsets[buckets + i] = static_cast<uint32_t>(value);
        }
        std::copy(chunk.small_bitmap_bytes.begin(), chunk.small_bitmap_bytes.end(),
            layer.small_bitmap_bytes.begin() + small);
        std::copy(chunk.large_bitmap_words.begin(), chunk.large_bitmap_words.end(),
            layer.large_bitmap_words.begin() + large);
        std::copy(chunk.success_values.begin(), chunk.success_values.end(), layer.success_values.begin() + values);
        buckets += chunk.bucket_keys.size();
        small += chunk.small_bitmap_bytes.size();
        large += chunk.large_bitmap_words.size();
        values += chunk.success_values.size();
        live += chunk.live_board_count;
    }
    layer.bucket_keys.resize(buckets);
    layer.bitmap_offsets.resize(buckets);
    layer.success_offsets.resize(buckets);
    layer.small_bitmap_bytes.resize(small);
    layer.large_bitmap_words.resize(large);
    layer.success_values.resize(values);
    layer.success_storage_count = values;
    layer.live_board_count = live;
}

template <typename T>
void stage_retired_layer(EXRetiredLayer &result, Prefix36LayerT<T> &layer,
    const PatternSpec &spec, const RunOptions &options, const LutBundle &lut,
    int step, uint64_t chunk_budget, RuntimeControls::DeletionThresholdState thresholds,
    bool require_output, bool low_memory = false) {
    const auto mode = dtype_mode_from_name(options.success_rate_dtype);
    const auto io = FileIOUtils::direct_io_config_from_options(options);
    const int threads = thread_count_from_options(options);
    const T zero = zero_value_for_dtype<T>(options.success_rate_dtype);
    const auto max_scale = max_scale_value_for_dtype<T>(options.success_rate_dtype);
    T threshold = RuntimeControls::absolute_deletion_threshold(zero, max_scale, thresholds);
    bool prune = thresholds.absolute > 0;
    const double compact_start = now_seconds();
    if (thresholds.relative > 0) {
        const T maximum = max_success_value(layer, zero);
        if (maximum > zero) {
            threshold = RuntimeControls::max_deletion_threshold(threshold,
                RuntimeControls::relative_deletion_threshold(maximum, zero, thresholds));
            prune = true;
        }
    }
    result.compact_seconds = now_seconds() - compact_start;
    result.step = step;
    result.before = result.after = layer.live_board_count;
    result.compressed = options.compress && !options.optimal_branch_only;
    result.effective_threshold = prune
        ? RuntimeControls::normalized_deletion_threshold(threshold, zero, max_scale) : 0;
    if (!require_output && !prune && !result.compressed) return;
    result.lease = StoragePaths::reserve_write_path(options,
        result.compressed ? StoragePaths::ArtifactRole::Cold : StoragePaths::ArtifactRole::Hot, step,
        result.compressed ? EXCompressedResult::kCompressedLayerFileExtension : kLayerFileExtension,
        2 * layer_file_bytes(layer, mode));
    result.temporary = result.lease.path() + ".retire_tmp";
    if (!low_memory) {
        double t = now_seconds();
        if (prune) compact_retired_layer_in_place(layer, spec, mode, lut, threshold, threads, chunk_budget);
        result.compact_seconds += now_seconds() - t;
        result.after = layer.live_board_count;
        t = now_seconds();
        if (result.compressed) {
            EXCompressedResult::compress_prefix36_layer_view_to_ex_result(
                layer_compression_view(layer, spec, mode), lut_file_path(options.pathname), result.temporary,
                4096U, 65536U, 5,
                {options.result_value_policy.get(), step, thresholds.absolute, thresholds.relative});
        } else {
            write_layer_file(result.temporary, layer, spec, mode, io);
        }
        result.write_seconds += now_seconds() - t;
        result.lease.release();
        return;
    }
    EXLayerChunks<T> chunks(layer, spec, mode, lut);
    EXChunkOutput<T> output(result.temporary, chunks.header, lut.dense_lut.size_table);
    Prefix36LayerT<T> chunk;
    while (chunks.next(chunk, chunk_budget)) {
        double t = now_seconds();
        if (prune) chunk = compact_layer(chunk, lut.dense_lut, threshold, threads, chunk.success_stride);
        result.compact_seconds += now_seconds() - t;
        t = now_seconds();
        output.append(chunk);
        result.write_seconds += now_seconds() - t;
    }
    const double t = now_seconds();
    if (result.compressed) {
        output.seal_sections();
        const std::string header = result.temporary;
        result.temporary = result.lease.path() + ".tmp";
        try {
            EXCompressedResult::compress_zbook_sections_to_ex_result(header, lut_file_path(options.pathname),
                result.temporary, {options.result_value_policy.get(), step, thresholds.absolute, thresholds.relative});
        } catch (...) {
            std::error_code ec;
            NativePath::remove(header, ec);
            throw;
        }
        std::error_code ec;
        NativePath::remove(header, ec);
    } else {
        output.finish(io);
    }
    result.after = output.header().live_board_count;
    result.write_seconds += now_seconds() - t;
    // The staged bytes already consume disk space; only a rename remains.
    result.lease.release();
}
