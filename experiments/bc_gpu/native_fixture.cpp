// Isolated fixture generator and production BC CPU oracle.
#include "BCResidentSolve.h"
#include "BCCellBuilder.h"
#include "Calculator.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <unordered_set>
#include <chrono>
#include <random>
#include <cfloat>

namespace fs = std::filesystem;
using namespace BC;
static double now() {return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
template<class T> void dump(const fs::path& p, const std::vector<T>& a) {
    std::ofstream f(p,std::ios::binary); f.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(T));
    if(!f) throw std::runtime_error("write failed: "+p.string());
}
template<class T> std::vector<T> read(const fs::path& p) {
    auto bytes=fs::file_size(p);if(bytes%sizeof(T))throw std::runtime_error("bad input size");
    std::vector<T> a(bytes/sizeof(T));std::ifstream f(p,std::ios::binary);
    f.read(reinterpret_cast<char*>(a.data()),bytes);if(!f)throw std::runtime_error("read failed");return a;
}
static bool terminal(uint64_t b,unsigned target) {for(unsigned p=0;p<16;p++) if(((b>>(4*p))&15)==target)return true;return false;}
static uint64_t mix(uint64_t x){x^=x>>33;x*=0xff51afd7ed558ccdULL;x^=x>>33;x*=0xc4ceb9fe1a85ec53ULL;return x^(x>>33);}
struct Entry {uint64_t key=0;uint32_t word=UINT32_MAX,row=0;};
struct Task {uint64_t key,bits;uint32_t row,rank,sw,se;};
static_assert(sizeof(Entry)==16 && sizeof(Task)==32);

static BCPositionLayerReader make_position(const BCLut& lut, unsigned sum,
                                         const std::unordered_set<uint64_t>& boards, unsigned modulus) {
    auto axis=BCFamilyTable::from_range(sum,2,0,modulus-1);
    BCSolvePreparedQueryEncoder encoder(lut,axis,modulus);
    std::vector<std::unique_ptr<BCCellBuilder>> cells(modulus*modulus);
    for(auto b:boards){BCSolvePreparedQuery q;
        if(!encoder.encode(unpack_board_to_quadrants(b),0,q))throw std::runtime_error("encode failed");
        if(!cells[q.cid])cells[q.cid]=std::make_unique<BCCellBuilder>(lut);
        cells[q.cid]->insert(q.key,q.rank);
    }
    BCPositionLayerWriter writer;writer.begin_layer(axis);
    for(unsigned c=0;c<cells.size();c++)if(cells[c])writer.write_cell(c,cells[c]->finalize());else writer.mark_empty_cell(c);
    return BCPositionLayerReader(writer.finish_layer(),lut);
}
static void export_position(const fs::path& dir,const BCPositionLayerReader& pos) {
    fs::create_directories(dir);dump(dir/"position.bcpos",pos.bytes());
    std::vector<Entry> hash;
    std::vector<uint32_t> cells,base;
    std::vector<uint64_t> words,boards;
    std::vector<Task> tasks;
    uint32_t row_base=0;
    for(unsigned c=0;c<pos.cell_count();c++){
        const auto bs=pos.bucket_entries_for_cell(c);const auto payload=pos.rank_payload_for_cell(c);
        unsigned cap=1;while(cap<std::max(4U,(bs.size*16+4)/5))cap*=2;
        cells.push_back(hash.size());cells.push_back(cap-1);hash.resize(hash.size()+cap);
        for(unsigned j=0;j<bs.size;j++){
            auto b=bs.data[j];auto decoder=BCBucketBoardDecoder(pos.lut(),b.key);
            unsigned len=decoder.bitmap_len();unsigned offset=bc_rank_payload_bitmap_offset(b.rank_payload_offset,len);
            Entry e{b.key,static_cast<uint32_t>(words.size()),row_base+b.success_row_offset};
            unsigned slot=mix(b.key)&(cap-1),start=cells[2*c];
            while(hash[start+slot].word!=UINT32_MAX)slot=(slot+1)&(cap-1);hash[start+slot]=e;
            unsigned ordinal=0;
            for(unsigned w=0;w<words_for_bits(len);w++){
                uint64_t bits=load_u64_le(payload.data+offset+8*w);words.push_back(bits);base.push_back(ordinal);
                if(bits)tasks.push_back({b.key,bits,e.row+ordinal,w*64,decoder.rank_decoder.count_sw,decoder.rank_decoder.count_se});
                auto scan=bits;while(scan){unsigned bit=__builtin_ctzll(scan);scan&=scan-1;boards.push_back(decoder.board(w*64+bit));}
                ordinal+=popcount64(bits);
            }
        }
        row_base+=pos.descriptor(c).success_rows;
    }
    if(boards.size()!=row_base)throw std::runtime_error("export row mismatch");
    dump(dir/"hash.bin",hash);dump(dir/"cells.bin",cells);dump(dir/"words.bin",words);
    dump(dir/"bases.bin",base);dump(dir/"tasks.bin",tasks);dump(dir/"boards.bin",boards);
}
static void export_lut(const fs::path& dir,const BCLut& lut) {
    std::vector<uint32_t> desc(65536*4),offsets(65536,UINT32_MAX),moves(65536);
    std::vector<uint16_t> unrank;
    for(unsigned w=0;w<65536;w++){
        const auto& d=lut.word_desc(w);
        desc[4*w]=d.sum;desc[4*w+1]=d.packed_sum_mask;desc[4*w+2]=d.rank;desc[4*w+3]=d.valid?d.group_count:0;
        if(d.valid && offsets[d.packed_sum_mask]==UINT32_MAX){offsets[d.packed_sum_mask]=unrank.size();auto g=lut.word_group(d.sum_id,d.empty_mask);unrank.insert(unrank.end(),g.words,g.words+g.count);}
        auto l=StandardMergePolicy::_internal_merge(w,false).first;
        auto r=StandardMergePolicy::_internal_merge(w,true).first;
        moves[w]=static_cast<uint32_t>(l)|(static_cast<uint32_t>(r)<<16);
    }
    dump(dir/"lut.bin",desc);dump(dir/"unrank_offsets.bin",offsets);dump(dir/"unrank.bin",unrank);dump(dir/"moves.bin",moves);
}
int main(int argc,char**argv)try{
    // Setup only: production-equivalent freeN initial set and LUTs, no CPU solve.
    if(argc>1 && std::string(argv[1])=="--setup-free") {
        fs::path root=argv[2];unsigned free=std::stoul(argv[3]),target=std::stoul(argv[4]);
        if(free<2||free>16||target<3||target>=15)throw std::runtime_error("setup bounds");
        fs::create_directories(root);std::vector<uint8_t> alphabet{0,15};
        for(unsigned r=1;r<=target;r++)alphabet.push_back(r);BCLut lut(alphabet);export_lut(root,lut);
        std::vector<uint64_t> raw;
        for(unsigned mask=0;mask<65536;mask++)if(__builtin_popcount(mask)==int(16-free)) {
            uint64_t b=0;for(unsigned p=0;p<16;p++)b|=uint64_t((mask>>p)&1?15:1)<<(4*p);
            for(unsigned p=0;p<16;p++)if(!((mask>>p)&1))raw.push_back(b&~(15ULL<<(4*p)));
        }
        auto successors=[](const std::vector<uint64_t>& input){
            std::vector<uint64_t> out;for(auto b:input)for(int d=1;d<=4;d++){
                auto m=BoardMover::move_board(b,d);if(m==Calculator::canonical_full(m))out.push_back(m);
            }std::sort(out.begin(),out.end());out.erase(std::unique(out.begin(),out.end()),out.end());return out;
        };
        auto a=successors(raw),b=successors(a);a.insert(a.end(),b.begin(),b.end());
        std::sort(a.begin(),a.end());a.erase(std::unique(a.begin(),a.end()),a.end());
        a.erase(std::remove_if(a.begin(),a.end(),[](uint64_t b){return
            (b&15)>2&&((b>>12)&15)>2&&((b>>48)&15)>2&&((b>>60)&15)>2;}),a.end());
        dump(root/"initial.bin",a);std::cout<<"{\"initial_rows\":"<<a.size()<<"}"<<std::endl;return 0;
    }
    if(argc>1 && std::string(argv[1])=="--compact") {
        fs::path root=argv[2];unsigned sum=std::stoul(argv[3]),target=std::stoul(argv[4]);int threads=std::stoi(argv[5]);
        std::vector<uint8_t> alphabet{0,15};for(unsigned r=1;r<=target;r++)alphabet.push_back(r);BCLut lut(alphabet);
        BCPositionLayerReader pos(read<uint8_t>(root/std::to_string(sum)/"position.bcpos"),lut);
        BCResidentRawSolveResult<uint32_t> raw;raw.values=read<uint32_t>(root/std::to_string(sum)/"values.bin");
        raw.cell_value_offsets=bc_resident_cell_value_offsets(pos);
        auto start=now();auto compact=bc_resident_compact_zero_in_place(pos,raw,lut,1,BCSuccessDTypeMode::UInt32,0U,threads);double secs=now()-start;
        auto dir=root/"solved"/std::to_string(sum);export_position(dir,compact.position);dump(dir/"values.bin",compact.success_values);
        std::cout<<"{\"sum\":"<<sum<<",\"rows\":"<<compact.success_values.size()<<",\"compact_seconds\":"<<secs<<"}"<<std::endl;return 0;
    }
    if(argc>1 && std::string(argv[1])=="--probes") {
        fs::path root=argv[2];fs::create_directories(root);std::mt19937_64 rng(20260923);
        std::vector<uint64_t> input,expected;
        for(unsigned row=0;row<65536;row++)for(unsigned shift=0;shift<64;shift+=16)input.push_back(uint64_t(row)<<shift);
        for(unsigned i=0;i<65536;i++)input.push_back(rng());
        for(auto b:input){for(int d=1;d<=4;d++)expected.push_back(BoardMover::move_board(b,d));expected.push_back(Calculator::canonical_full(b));}
        dump(root/"primitive_inputs.bin",input);dump(root/"primitive_expected.bin",expected);
        std::vector<uint64_t> s2,s4;std::vector<uint32_t> counts;
        for(unsigned n=1;n<=16;n++){
            for(unsigned i=0;i<30000;i++){counts.push_back(n);s2.push_back(rng()%(4000000000ULL*n+1));s4.push_back(rng()%(4000000000ULL*n+1));}
            for(uint64_t v:{0ULL,1ULL,511ULL,512ULL,1023ULL,1024ULL,2147483647ULL,2147483648ULL,3999999999ULL,4000000000ULL})
                for(int a=-2;a<=2;a++)for(int b=-2;b<=2;b++){
                    int64_t x=int64_t(v*n)+a,y=int64_t(v*n)+b;
                    if(x<0||y<0||x>int64_t(4000000000ULL*n)||y>int64_t(4000000000ULL*n))continue;
                    counts.push_back(n);s2.push_back(x);s4.push_back(y);
                }
        }
        dump(root/"s2.bin",s2);dump(root/"s4.bin",s4);dump(root/"counts.bin",counts);
        std::vector<double> ps{0.,.01,.1,.125,.25,.5,.75,.9,1.};dump(root/"probabilities.bin",ps);
        for(unsigned k=0;k<ps.size();k++){
            std::vector<uint32_t> values;for(unsigned i=0;i<s2.size();i++)values.push_back(bc_solve_reduce_weighted_success_sums<uint32_t>(s2[i],s4[i],counts[i],ps[k],0U));
            dump(root/("rounding_"+std::to_string(k)+".bin"),values);
        }
        std::cout<<"{\"primitive_boards\":"<<input.size()<<",\"rounding_cases_per_probability\":"<<s2.size()<<",\"long_double_mantissa_bits\":"<<LDBL_MANT_DIG<<"}"<<std::endl;
        return 0;
    }
    // Benchmark/recalculate an existing generated layer, retaining its production file format.
    if(argc>1 && (std::string(argv[1])=="--layer" || std::string(argv[1])=="--layer-sparse")) {
        if(argc!=10)throw std::runtime_error("--layer ROOT SUM TARGET THREADS P4 MOD REPS OUTPUT");
        fs::path root=argv[2],output=argv[9];unsigned sum=std::stoul(argv[3]),target=std::stoul(argv[4]);
        int threads=std::stoi(argv[5]),reps=std::stoi(argv[8]);double p4=std::stod(argv[6]);unsigned mod=std::stoul(argv[7]);
        std::vector<uint8_t> alphabet{0,15};for(unsigned r=1;r<=target;r++)alphabet.push_back(r);BCLut lut(alphabet);
        auto current=BCPositionLayerReader(read<uint8_t>(root/std::to_string(sum)/"position.bcpos"),lut);
        BCResidentSolvedLayer<uint32_t> f2,f4;
        fs::path fr=std::string(argv[1])=="--layer-sparse"?root/"solved":root;
        auto begin=now();f2.open(read<uint8_t>(fr/std::to_string(sum+2)/"position.bcpos"),read<uint32_t>(fr/std::to_string(sum+2)/"values.bin"),lut);
        f4.open(read<uint8_t>(fr/std::to_string(sum+4)/"position.bcpos"),read<uint32_t>(fr/std::to_string(sum+4)/"values.bin"),lut);double load_index=now()-begin;
        BCResidentSolveOptions<uint32_t> options;options.num_threads=threads;options.edge_options.success_target_rank=target;
        options.edge_options.success_check_all_cells=true;options.edge_options.future_cell_modulus=mod;options.edge_options.spawn_rate4=p4;
        auto warm=bc_resident_solve_raw_values(current,f2,f4,options);std::vector<double> times;
        BCResidentRawSolveResult<uint32_t> result;
        for(int k=0;k<reps;k++){begin=now();result=bc_resident_solve_raw_values(current,f2,f4,options);times.push_back(now()-begin);}
        dump(output,std::vector<uint32_t>(result.values.begin(),result.values.end()));std::sort(times.begin(),times.end());
        std::cout<<"{\"rows\":"<<result.values.size()<<",\"threads\":"<<threads<<",\"repeats\":"<<reps
                 <<",\"median_seconds\":"<<std::setprecision(12)<<times[times.size()/2]<<",\"load_index_seconds\":"<<load_index<<"}"<<std::endl;
        return 0;
    }
    if(argc<6)throw std::runtime_error("usage: fixture OUT FREE_CELLS TARGET_RANK THREADS P4 [MODULUS]");
    fs::path root=argv[1];unsigned free=std::stoul(argv[2]),target=std::stoul(argv[3]);
    int threads=std::stoi(argv[4]);double p4=std::stod(argv[5]);unsigned modulus=argc>6?std::stoul(argv[6]):7;
    if(free<2||free>12||target<3||target>6)throw std::runtime_error("fixture bounds");
    fs::create_directories(root);
    std::vector<uint8_t> alphabet{0,15};for(unsigned r=1;r<=target;r++)alphabet.push_back(r);
    BCLut lut(alphabet);export_lut(root,lut);
    unsigned walls=16-free,wall_sum=walls*32768U,minsum=2*(free-1),maxsum=free*(1U<<(target-1))+4;
    uint64_t seed=0;for(unsigned p=0;p<walls;p++)seed|=15ULL<<(4*p);
    for(unsigned p=walls;p<15;p++)seed|=1ULL<<(4*p);seed=Calculator::canonical_full(seed);
    std::vector<std::unordered_set<uint64_t>> layers(maxsum/2+3);layers[minsum/2].insert(seed);
    double t0=now();uint64_t generated=0;
    for(unsigned s=minsum;s<=maxsum;s+=2){
        for(auto b:layers[s/2]){
            if(terminal(b,target))continue;
            for(unsigned p=0;p<16;p++)if(((b>>(4*p))&15)==0){
                for(unsigned r=1;r<=2;r++){
                    unsigned ns=s+(1U<<r);if(ns>maxsum)continue;
                    uint64_t spawned=b|(uint64_t(r)<<(4*p));
                    for(int d=1;d<=4;d++){auto m=BoardMover::move_board(spawned,d);if(m!=spawned)layers[ns/2].insert(Calculator::canonical_full(m));}
                }
            }
        }
        generated+=layers[s/2].size();
        std::cout<<"generate sum="<<s<<" rows="<<layers[s/2].size()<<" seconds="<<now()-t0<<std::endl;
    }
    const double generation_seconds=now()-t0;
    std::ofstream csv(root/"cpu_layers.csv");csv<<"small_sum,rows,nonzero,terminal,solve_seconds,index_seconds,export_seconds,value_min,value_max\n";
    BCResidentSolvedLayer<uint32_t> f2,f4;
    std::unordered_set<uint64_t> empty;
    f2.open(make_position(lut,wall_sum+maxsum+2,empty,modulus),{},1);
    f4.open(make_position(lut,wall_sum+maxsum+4,empty,modulus),{},1);
    // Export explicit empty boundary futures for GPU replay.
    for(unsigned s:{maxsum+2,maxsum+4}){auto p=make_position(lut,wall_sum+s,empty,modulus);export_position(root/std::to_string(s),p);dump(root/std::to_string(s)/"values.bin",std::vector<uint32_t>{});}
    BCResidentSolveOptions<uint32_t> options;options.num_threads=threads;
    options.edge_options.success_target_rank=target;options.edge_options.success_check_all_cells=true;
    options.edge_options.future_cell_modulus=modulus;options.edge_options.spawn_rate4=p4;
    for(int s=maxsum;s>=int(minsum);s-=2){
        auto pos=make_position(lut,wall_sum+s,layers[s/2],modulus);
        auto start=now();auto result=bc_resident_solve_raw_values(pos,f2,f4,options);double elapsed=now()-start;
        std::vector<uint32_t> vals(result.values.begin(),result.values.end());
        unsigned nonzero=0,terms=0,lo=UINT32_MAX,hi=0;
        for(auto v:vals){nonzero+=v!=0;lo=std::min(lo,v);hi=std::max(hi,v);}
        for(auto b:layers[s/2])terms+=terminal(b,target);
        auto dir=root/std::to_string(s);start=now();export_position(dir,pos);dump(dir/"values.bin",vals);double ex=now()-start;
        start=now();BCResidentSolvedLayer<uint32_t> solved;solved.open(std::move(pos),std::move(vals),1);double ix=now()-start;
        csv<<s<<','<<layers[s/2].size()<<','<<nonzero<<','<<terms<<','<<std::setprecision(12)<<elapsed<<','<<ix<<','<<ex<<','<<(lo==UINT32_MAX?0:lo)<<','<<hi<<'\n';csv.flush();
        std::cout<<"solve sum="<<s<<" rows="<<layers[s/2].size()<<" seconds="<<elapsed<<" nonzero="<<nonzero<<std::endl;
        f4=std::move(f2);f2=std::move(solved);
        layers[s/2].clear();layers[s/2].rehash(0);
    }
    std::ofstream meta(root/"manifest.json");
    meta<<"{\n\"free_cells\":"<<free<<",\"target_rank\":"<<target<<",\"seed\":\""<<seed<<"\",\"min_sum\":"<<minsum<<",\"max_sum\":"<<maxsum<<",\"wall_sum\":"<<wall_sum<<",\"modulus\":"<<modulus<<",\"threads\":"<<threads<<",\"p4\":"<<std::setprecision(17)<<p4<<",\"terminal_value\":"<<options.terminal_value<<",\"generated_rows\":"<<generated<<",\"generation_seconds\":"<<generation_seconds<<"\n}\n";
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
