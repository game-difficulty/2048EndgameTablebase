// Included after kernels.cu. Experimental streaming BC representation.
// The immutable codec is unchanged; only the on-disk experiment container differs.
struct Mutable {U64* keys; U32* pointers; U64* words; U32* counters; U32 mask,capacity;};
__device__ __forceinline__ bool terminal_board(U64 b,U32 target){
    U64 d=b^(U64(target)*0x1111111111111111ULL);
    return (~(((d&0x7777777777777777ULL)+0x7777777777777777ULL)|d|0x7777777777777777ULL))!=0;
}
__device__ __forceinline__ void keyrank(U64 b,const U32* lut,U64& key,U32& rank,U32& length){
    U16 nw,ne,sw,se;quadrants(b,nw,ne,sw,se);
    const U32 *c=lut+4*U32(ne),*d=lut+4*U32(sw),*e=lut+4*U32(se);
    key=(U64(nw)<<48)|(U64(c[1])<<32)|(U64(d[1])<<16)|e[1];
    rank=(c[2]*d[3]+d[2])*e[3]+e[2];length=(c[3]*d[3]*e[3]+63)/64;
}
__device__ void insert_key(Mutable m,U64 key,U32 rank,U32 length){
    U32 slot=mix(key)&m.mask;
    for(U32 probe=0;probe<=m.mask;probe++,slot=(slot+1)&m.mask){
        U64 old=atomicCAS(m.keys+slot,~0ULL,key);
        if(old==~0ULL){
            U32 offset=atomicAdd(m.counters+1,length);atomicAdd(m.counters,1U);
            if(U64(offset)+length>m.capacity){atomicOr(m.counters+2,2U);offset=0xfffffffeU;}
            __threadfence();atomicExch(m.pointers+slot,offset);
        }
        if(old==~0ULL||old==key){
            U32 offset;
            do{offset=atomicAdd(m.pointers+slot,0U);}while(offset==0xffffffffU);
            if(offset!=0xfffffffeU)atomicOr(m.words+offset+rank/64,1ULL<<(rank&63));
            return;
        }
        // Stop before saturation. Host grows and replays this entire batch.
        if(probe==1023){atomicOr(m.counters+2,1U);return;}
    }
    atomicOr(m.counters+2,1U);
}
extern "C" __global__ void insert_boards(const U64* boards,U32 n,const U32* lut,
    U64* keys,U32* pointers,U64* words,U32* counters,U32 mask,U32 capacity){
    U32 i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    U64 key;U32 rank,len;keyrank(boards[i],lut,key,rank,len);
    insert_key({keys,pointers,words,counters,mask,capacity},key,rank,len);
}
extern "C" __global__ void generate_mutable(const U64* boards,U32 n,const U32* lut,const U32* moves,
    U64* keys,U32* pointers,U64* words,U32* counters,U32 mask,U32 capacity,
    U32 spawn,U32 target,U32 check,U32 only_terminal){
    U32 i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    U64 b=boards[i];if(check&&terminal_board(b,target))return;
    Mutable out{keys,pointers,words,counters,mask,capacity};
    for(U32 p=0;p<16;p++)if(((b>>(4*p))&15)==0){
        U64 s=b|(U64(spawn)<<(4*p));
        #pragma unroll
        for(U32 d=0;d<4;d++){
            U64 m=move(s,moves,d);if(m==s||(only_terminal&&!terminal_board(m,target)))continue;
            U64 key;U32 rank,len;keyrank(canonical(m),lut,key,rank,len);insert_key(out,key,rank,len);
        }
    }
}
extern "C" __global__ void key_metadata(const U64* keys,U32 n,const U32* lut,const U32* groups,U32 mod,
    U32* lengths,U32* cids){
    U32 i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;U64 k=keys[i];
    U32 nw=k>>48,ne=(k>>32)&65535,sw=(k>>16)&65535,se=k&65535;
    U32 a=lut[4*nw],b=groups[2*ne],c=groups[2*sw],d=groups[2*se];
    lengths[i]=(groups[2*ne+1]*groups[2*sw+1]*groups[2*se+1]+63)/64;
    cids[i]=((min(a+b,c+d)/2)%mod)*mod+(min(a+c,b+d)/2)%mod;
}
extern "C" __global__ void gather_words(const U32* srcstarts,const U64* src,const U32* dststarts,
    const U32* lengths,U32 n,U64* dst){
    U32 i=blockIdx.x;if(i>=n)return;
    for(U32 w=threadIdx.x;w<lengths[i];w+=blockDim.x)dst[dststarts[i]+w]=src[srcstarts[i]+w];
}
extern "C" __global__ void rehash_mutable(const U64* oldkeys,const U32* oldptr,U32 n,U64* keys,U32* ptr,U32 mask){
    U32 i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||oldkeys[i]==~0ULL||oldptr[i]>=0xfffffffeU)return;
    U64 key=oldkeys[i];U32 slot=mix(key)&mask;
    while(atomicCAS(keys+slot,~0ULL,key)!=~0ULL)slot=(slot+1)&mask;
    ptr[slot]=oldptr[i];
}
extern "C" __global__ void word_popcount(const U64* words,U32 n,U32* counts){
    U32 i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)counts[i]=__popcll(words[i]);
}
__device__ __forceinline__ U32 owner(const U32* starts,U32 n,U32 w){
    U32 lo=0,hi=n;while(lo+1<hi){U32 mid=(lo+hi)/2;if(starts[mid]<=w)lo=mid;else hi=mid;}return lo;
}
extern "C" __global__ void decode_words(const U64* keys,const U32* starts,U32 nb,const U64* words,
    const U32* rows,U32 first,U32 n,const U32* groups,const U32* offsets,const U16* unrank,U64* boards){
    U32 wi=blockIdx.x*blockDim.x+threadIdx.x;if(wi>=n)return;wi+=first;U64 bits=words[wi];if(!bits)return;
    U32 bi=owner(starts,nb,wi);U64 key=keys[bi];U32 base=(wi-starts[bi])*64;
    U16 nw=key>>48,ne=key>>32,sw=key>>16,se=key;
    U32 row=rows[wi]-rows[first],csw=groups[2*U32(sw)+1],cse=groups[2*U32(se)+1];
    while(bits){U32 r=base+__ffsll(bits)-1;bits&=bits-1;U32 q=r/cse;
        boards[row++]=pack(nw,unrank[offsets[ne]+q/csw],unrank[offsets[sw]+q%csw],unrank[offsets[se]+r%cse]);
    }
}
extern "C" __global__ void compact_words(const U64* words,const U32* rows,const U32* values,U32 n,U64* output){
    U32 wi=blockIdx.x*blockDim.x+threadIdx.x;if(wi>=n)return;
    U64 bits=words[wi],keep=0;U32 row=rows[wi];
    while(bits){U32 p=__ffsll(bits)-1;bits&=bits-1;if(values[row++])keep|=1ULL<<p;}output[wi]=keep;
}
extern "C" __global__ void bucket_live(const U32* starts,const U32* rows,U32 n,U32* live){
    U32 i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)live[i]=rows[starts[i+1]]-rows[starts[i]];
}
extern "C" __global__ void build_cell_index(const U64* keys,const U32* starts,const U32* rows,U32 n,
    Entry* hash,U32 mask,U16* bases){
    U32 i=blockIdx.x;if(i>=n)return;U32 first=starts[i],last=starts[i+1],row=rows[first];
    for(U32 w=first+threadIdx.x;w<last;w+=blockDim.x)bases[w]=U16(rows[w]-row);
    if(threadIdx.x==0){U64 key=keys[i];U32 slot=mix(key)&mask;
        while(atomicCAS(&hash[slot].key,~0ULL,key)!=~0ULL)slot=(slot+1)&mask;
        hash[slot].word=first;hash[slot].row=row;
    }
}
__device__ __forceinline__ U32 lookup_cells(U64 b,const U32* lut,const U64* desc,U32 modulus,U32* missing){
    U16 nw,ne,sw,se;quadrants(b,nw,ne,sw,se);
    const U32 *a=lut+4*U32(nw),*c=lut+4*U32(ne),*d=lut+4*U32(sw),*e=lut+4*U32(se);
    U32 cid=((min(a[0]+c[0],d[0]+e[0])/2)%modulus)*modulus+(min(a[0]+d[0],c[0]+e[0])/2)%modulus;
    const U64* f=desc+6*cid;
    if(!f[0]){atomicAdd(missing,1U);return 0;} // Missing dependency is a correctness error.
    if(!f[5])return 0; // Loaded, genuinely empty future cell.
    const Entry* hash=reinterpret_cast<const Entry*>(f[0]);
    const U64* words=reinterpret_cast<const U64*>(f[1]);
    const U16* bases=reinterpret_cast<const U16*>(f[2]);
    const U32* values=reinterpret_cast<const U32*>(f[3]);U32 mask=f[4];
    U64 key=(U64(nw)<<48)|(U64(c[1])<<32)|(U64(d[1])<<16)|e[1];
    U32 rank=(c[2]*d[3]+d[2])*e[3]+e[2],slot=mix(key)&mask;
    Entry ent=hash[slot];while(ent.word!=0xffffffffU&&ent.key!=key){slot=(slot+1)&mask;ent=hash[slot];}
    if(ent.word==0xffffffffU)return 0;
    U32 wi=ent.word+rank/64,bit=rank&63;U64 bits=words[wi];
    if(!((bits>>bit)&1))return 0;
    return values[ent.row+U32(bases[wi])+__popcll(bits&((1ULL<<bit)-1))];
}
extern "C" __global__ void empty_counts(const U64* boards,U32 n,U32* counts){
    U32 i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;U64 b=boards[i];U32 c=0;
    for(U32 p=0;p<16;p++)c+=((b>>(4*p))&15)==0;counts[i]=c;
}
// mode 0: save per-empty directional max; 1: merge saved max; 2: both axes at once.
// Scratch stores unweighted sum4 in uint64 so the final reduction matches the
// resident CPU80 contract. This intentionally avoids intermediate truncation.
extern "C" __global__ void family_pass(const U64* boards,U32 n,const U32* lut,const U32* moves,
    const U64* future,U32 modulus,U32 target,U32 spawn,U32 directions,U32 mode,
    const U32* empty_offsets,U32* partial,U64* scratch,U32* output,U64 numerator,U32 bits,U32* missing){
    U32 i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;U64 board=boards[i];
    if(terminal_board(board,target)){if(spawn==1&&mode)output[i]=4000000000U;return;}
    U32 off=empty_offsets[i],count=empty_offsets[i+1]-off,j=0;U64 sum=0;
    for(U32 p=0;p<16;p++)if(((board>>(4*p))&15)==0){
        U64 s=board|(U64(spawn)<<(4*p));U32 best=0;
        #pragma unroll
        for(U32 d=0;d<4;d++)if(directions&(1U<<(d/2))){
            U64 m=move(s,moves,d);if(m!=s)best=max(best,lookup_cells(canonical(m),lut,future,modulus,missing));
        }
        if(mode==0)partial[off+j]=best;
        else {if(mode==1)best=max(best,partial[off+j]);sum+=best;}
        j++;
    }
    if(mode){if(spawn==2)scratch[i]=sum;
        else output[i]=count?reduce_cpu80(sum,scratch[i],count,numerator,bits):0;}
}
extern "C" __global__ void resident_cells(const U64* boards,U32 n,const U32* lut,const U32* moves,
    const U64* future2,const U64* future4,U32 modulus,U32 target,U64 numerator,U32 bits,U32* output,U32* missing){
    U32 i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;U64 b=boards[i];
    if(terminal_board(b,target)){output[i]=4000000000U;return;}
    U32 count=0;U64 s2=0,s4=0;
    for(U32 p=0;p<16;p++)if(((b>>(4*p))&15)==0){
        U64 a=b|(1ULL<<(4*p)),c=b|(2ULL<<(4*p));U32 best2=0,best4=0;
        #pragma unroll
        for(U32 d=0;d<4;d++){
            U64 m=move(a,moves,d);if(m!=a)best2=max(best2,lookup_cells(canonical(m),lut,future2,modulus,missing));
            m=move(c,moves,d);if(m!=c)best4=max(best4,lookup_cells(canonical(m),lut,future4,modulus,missing));
        }s2+=best2;s4+=best4;count++;
    }output[i]=count?reduce_cpu80(s2,s4,count,numerator,bits):0;
}
extern "C" __global__ void family_fused(const U64* boards,U32 n,const U32* lut,const U32* moves,
    const U64* future2,const U64* future4,U32 modulus,U32 target,U32 directions,U32 mode,
    const U32* empty_offsets,U32* partial,U32* output,U64 numerator,U32 bits,U32* missing){
    U32 i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;U64 b=boards[i];
    if(terminal_board(b,target)){if(mode)output[i]=4000000000U;return;}
    U32 off=empty_offsets[i],count=empty_offsets[i+1]-off,j=0;U64 sum2=0,sum4=0;
    for(U32 p=0;p<16;p++)if(((b>>(4*p))&15)==0){
        U64 a=b|(1ULL<<(4*p)),c=b|(2ULL<<(4*p));U32 best2=0,best4=0;
        #pragma unroll
        for(U32 d=0;d<4;d++)if(directions&(1U<<(d/2))){
            U64 m=move(a,moves,d);if(m!=a)best2=max(best2,lookup_cells(canonical(m),lut,future2,modulus,missing));
            m=move(c,moves,d);if(m!=c)best4=max(best4,lookup_cells(canonical(m),lut,future4,modulus,missing));
        }
        if(mode==0){partial[2*(off+j)]=best2;partial[2*(off+j)+1]=best4;}
        else {if(mode==1){best2=max(best2,partial[2*(off+j)]);best4=max(best4,partial[2*(off+j)+1]);}
            sum2+=best2;sum4+=best4;}
        j++;
    }
    if(mode)output[i]=count?reduce_cpu80(sum2,sum4,count,numerator,bits):0;
}
