#pragma once
#include "BCZeroBlockCodec.h"
#include "BCFamilySolvePlan.h"
#include "BCFileIO.h"
#include <array>
#include <chrono>
#include <exception>

namespace BC {
// One arena shared by partial2/partial4/scratch4. IO and codecs run in distinct
// joined waves, never alongside another wave or solve computation.
struct BCZeroTempWorkspace {
    static constexpr size_t kMaxLanes = 16;
    uint32_t lanes;
    std::array<BCFamilyValueVector<uint8_t>, kMaxLanes> buffers;
    double encode_seconds = 0, decode_seconds = 0;
    uint64_t raw_bytes = 0, stored_bytes = 0, packed_blocks = 0, zero_blocks = 0;
    explicit BCZeroTempWorkspace(uint32_t threads) : lanes(std::min<uint32_t>(kMaxLanes, std::max(1U, threads))) {
        for (uint32_t i=0;i<lanes;++i) buffers[i].resize(kBCZeroBlockBytes);
    }
    static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
};

struct BCZeroTempBlock {
    uint64_t offset;
    uint32_t count;
    uint32_t encoded;
};
static_assert(sizeof(BCZeroTempBlock)==16);

template<class T> class BCZeroTempIO {
public:
    void open(BCWritableFile* writer, BCReadableFile* reader, BCZeroTempWorkspace* workspace,
              T zero, uint32_t alignment, uint64_t payload_offset, size_t max_blocks) {
        if (!writer || !reader || !workspace || !alignment ||
            (alignment & (alignment - 1U)) || alignment > kBCZeroBlockBytes || payload_offset % alignment)
            throw std::invalid_argument("BC zero temp IO configuration invalid");
        writer_=writer; reader_=reader; workspace_=workspace; zero_=zero;
        alignment_=alignment; cursor_=payload_offset; max_blocks_=max_blocks;
        std::vector<BCZeroTempBlock>().swap(blocks_);
        blocks_.reserve(max_blocks); // Exact upper bound is charged against the old cache budget.
    }
    bool enabled() const { return workspace_ != nullptr; }
    template<class ValueBuffer = BCFamilyValueVector<T>>
    std::vector<std::pair<uint64_t,uint64_t>> append_many(
            const std::vector<const ValueBuffer*>& values, BCFileIOStats* stats) {
        if (stats) *stats = {};
        std::vector<std::pair<uint64_t,uint64_t>> records;
        records.reserve(values.size());
        uint64_t needed=0;
        for(const auto* v:values) {
            if(!v) throw std::invalid_argument("BC zero temp null values");
            records.emplace_back(blocks_.size()+needed, v->size());
            needed = bc_checked_add_u64(needed, v->size()/kValues + (v->size()%kValues != 0),
                "BC zero temp block count overflow");
        }
        if(needed>max_blocks_-blocks_.size()) throw std::overflow_error("BC zero temp directory budget exceeded");
        const size_t first=blocks_.size();
        blocks_.resize(first+static_cast<size_t>(needed));
        size_t vi=0, value_offset=0;
        for(size_t base=first;base<blocks_.size();) {
            const uint32_t n=static_cast<uint32_t>(std::min<size_t>(workspace_->lanes,blocks_.size()-base));
            std::array<const T*,BCZeroTempWorkspace::kMaxLanes> sources{};
            for(uint32_t j=0;j<n;++j) {
                while(vi<values.size() && value_offset==values[vi]->size()) { ++vi; value_offset=0; }
                if(vi==values.size()) throw std::logic_error("BC zero temp append traversal mismatch");
                auto& block=blocks_[base+j];
                block.count=static_cast<uint32_t>(std::min<size_t>(kValues,values[vi]->size()-value_offset));
                sources[j]=values[vi]->data()+value_offset;
                value_offset+=block.count;
            }
            std::exception_ptr error;
            const double t0=BCZeroTempWorkspace::now();
            #pragma omp parallel for schedule(static) num_threads(n) if(n>1)
            for(int j=0;j<static_cast<int>(n);++j) {
                try {
                    auto& b=blocks_[base+j];
                    b.encoded=bc_zero_encode_block(sources[j],b.count,zero_,workspace_->buffers[j].data(),alignment_);
                } catch(...) {
                    #pragma omp critical(BCZeroTempCodecError)
                    { if(!error) error=std::current_exception(); }
                }
            }
            workspace_->encode_seconds+=BCZeroTempWorkspace::now()-t0;
            if(error) std::rethrow_exception(error);
            std::vector<BCFileWriteRequest> requests;
            requests.reserve(n);
            for(uint32_t j=0;j<n;++j) {
                auto& b=blocks_[base+j];
                const uint32_t stored=b.encoded&~kBCZeroBitmapFlag;
                const bool packed=(b.encoded&kBCZeroBitmapFlag)!=0;
                b.offset=cursor_;
                workspace_->raw_bytes+=static_cast<uint64_t>(b.count)*sizeof(T);
                workspace_->stored_bytes+=stored;
                workspace_->packed_blocks+=packed;
                workspace_->zero_blocks+=stored==0;
                if(stored==0) continue;
                const uint64_t physical=align(stored);
                if(cursor_>UINT64_MAX-physical) throw std::overflow_error("BC zero temp file offset overflow");
                if(packed) {
                    auto* data=workspace_->buffers[j].data();
                    std::memset(data+stored,0,static_cast<size_t>(physical-stored));
                    requests.push_back({cursor_,data,physical});
                } else requests.push_back({cursor_,sources[j],stored});
                cursor_+=physical;
            }
            if(!requests.empty()) {
                BCFileIOStats wave_stats;
                writer_->write_many(requests, &wave_stats);
                bc_success_accumulate_file_stats(stats, wave_stats);
            }
            base+=n; // All writes have joined before a scratch slot can be reused.
        }
        return records;
    }

    void read_many(const std::vector<std::pair<uint64_t,uint64_t>>& records,
                   std::vector<BCFamilyValueVector<T>>& out, BCFileIOStats* stats) {
        if (stats) *stats = {};
        out.clear(); out.resize(records.size());
        if (records.empty()) return;
        for(size_t i=0;i<records.size();++i) {
            const auto [begin,count]=records[i];
            const uint64_t n=count/kValues+(count%kValues!=0);
            if(begin>blocks_.size() || n>blocks_.size()-begin || count>SIZE_MAX)
                throw std::runtime_error("BC zero temp record invalid");
            out[i].resize(static_cast<size_t>(count));
        }
        std::vector<T*> targets;
        targets.reserve(out.size());
        for (auto &values : out) targets.push_back(values.data());
        read_many_into(records, targets, stats);
    }

    // Decode directly into caller-owned arrays. In particular SingleChunk can
    // reuse its aligned sum4 allocation instead of constructing a second array.
    void read_many_into(const std::vector<std::pair<uint64_t,uint64_t>>& records,
                        const std::vector<T*>& destinations, BCFileIOStats* stats) {
        if (stats) *stats = {};
        if (records.size() != destinations.size())
            throw std::invalid_argument("BC zero temp destination count mismatch");
        for (size_t i = 0; i < records.size(); ++i) {
            const auto [begin,count] = records[i];
            const uint64_t n = count/kValues + (count%kValues != 0U);
            if (begin > blocks_.size() || n > blocks_.size()-begin || count > SIZE_MAX ||
                (count != 0U && destinations[i] == nullptr))
                throw std::runtime_error("BC zero temp destination record invalid");
        }
        if(writer_->mode()!=BCFileIOMode::Direct) writer_->flush();
        size_t ri=0, block_in_record=0;
        while(ri<records.size()) {
            std::array<const BCZeroTempBlock*,BCZeroTempWorkspace::kMaxLanes> wave{};
            std::array<T*,BCZeroTempWorkspace::kMaxLanes> targets{};
            uint32_t n=0;
            while(n<workspace_->lanes && ri<records.size()) {
                const uint64_t count=records[ri].second;
                const uint64_t value_offset=static_cast<uint64_t>(block_in_record)*kValues;
                if(block_in_record == count/kValues + (count%kValues != 0U)) {
                    ++ri; block_in_record=0; continue;
                }
                const auto& b=blocks_.at(static_cast<size_t>(records[ri].first)+block_in_record);
                if(b.count!=std::min<uint64_t>(kValues,count-value_offset))
                    throw std::runtime_error("BC zero temp block count mismatch");
                wave[n]=&b; targets[n]=destinations[ri]+value_offset;
                ++n; ++block_in_record;
            }
            if(n==0) continue;
            std::vector<BCFileReadRequest> requests;
            requests.reserve(n);
            bool needs_decode = false;
            for(uint32_t j=0;j<n;++j) {
                const auto& b=*wave[j];
                const uint32_t stored=b.encoded&~kBCZeroBitmapFlag;
                const bool packed=(b.encoded&kBCZeroBitmapFlag)!=0;
                needs_decode = needs_decode || packed;
                if(stored>kBCZeroBlockBytes || (!packed && stored!=b.count*sizeof(T)) ||
                    (packed && stored>=b.count*sizeof(T)) || b.offset%alignment_ ||
                    b.offset>cursor_ || align(stored)>cursor_-b.offset)
                    throw std::runtime_error("BC zero temp block directory invalid");
                if(!stored) continue;
                requests.push_back({b.offset,packed?static_cast<void*>(workspace_->buffers[j].data()):targets[j],
                                    packed?align(stored):stored});
            }
            if(!requests.empty()) {
                BCFileIOStats wave_stats;
                reader_->read_many(requests, &wave_stats);
                bc_success_accumulate_file_stats(stats, wave_stats);
            }
            // RAW blocks already landed in their final arrays and were validated
            // above. Do not launch an OpenMP team solely to revisit their headers.
            if (!needs_decode) continue;
            std::exception_ptr error;
            const double t0=BCZeroTempWorkspace::now();
            #pragma omp parallel for schedule(static) num_threads(n) if(n>1)
            for(int j=0;j<static_cast<int>(n);++j) {
                try {
                    const auto& b=*wave[j];
                    if(b.encoded&kBCZeroBitmapFlag)
                        bc_zero_decode_block(workspace_->buffers[j].data(),b.encoded,b.count,zero_,targets[j]);
                } catch(...) {
                    #pragma omp critical(BCZeroTempCodecError)
                    { if(!error) error=std::current_exception(); }
                }
            }
            workspace_->decode_seconds+=BCZeroTempWorkspace::now()-t0;
            if(error) std::rethrow_exception(error);
        }
    }
private:
    static constexpr size_t kValues=kBCZeroBlockBytes/sizeof(T);
    uint64_t align(uint64_t bytes) const { return (bytes+alignment_-1U)&~static_cast<uint64_t>(alignment_-1U); }
    BCWritableFile* writer_=nullptr;
    BCReadableFile* reader_=nullptr;
    BCZeroTempWorkspace* workspace_=nullptr;
    T zero_{};
    uint32_t alignment_=1;
    uint64_t cursor_=0;
    size_t max_blocks_=0;
    std::vector<BCZeroTempBlock> blocks_;
};
} // namespace BC
