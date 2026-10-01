// Standalone scalar BC CUDA baseline. No AD semantics or family streaming yet.
typedef unsigned long long U64;
typedef unsigned int U32;
typedef unsigned short U16;
struct Entry { U64 key; U32 word,row; };
struct Task { U64 key,bits; U32 row,rank,sw,se; };
struct Future { const Entry* hash; const U32* cells; const U64* words; const U32* bases; const U32* values; };
// Match this project's MinGW x87 long-double (64-bit significand) positive
// multiply/add/divide, then truncate to UInt32. Numerators use exact dyadics.
// Host restricts denominator exponent <=59 so the final divisor/remainder fits U64.
struct Wide {U64 lo,hi;};
__device__ __forceinline__ Wide round64(Wide a) {
    if(!a.hi)return a;
    U32 shift=64-__clzll(a.hi);U64 mask=(1ULL<<shift)-1,tail=a.lo&mask,half=1ULL<<(shift-1);
    bool up=tail>half || (tail==half && ((a.lo>>shift)&1));a.lo&=~mask;
    if(up){U64 before=a.lo;a.lo+=1ULL<<shift;a.hi+=a.lo<before;}
    return a;
}
__device__ __forceinline__ Wide product64(U64 a,U64 b){return round64({a*b,__umul64hi(a,b)});}
__device__ __forceinline__ U32 reduce_cpu80(U64 s2,U64 s4,U32 count,U64 numerator,U32 bits){
    U64 unit=1ULL<<bits;
    Wide a=product64(s2,unit-numerator),b=product64(s4,numerator);
    Wide sum{a.lo+b.lo,a.hi+b.hi};sum.hi+=sum.lo<a.lo;sum=round64(sum);
    U64 whole=bits?((sum.lo>>bits)|(sum.hi<<(64-bits))):sum.lo;
    U64 result=whole/count;
    U64 rem=((whole%count)<<bits)+(sum.lo&(unit-1));
    if(result){
        int exponent=63-__clzll(result),half_shift=int(bits)+exponent-64;
        if(half_shift>=0){
            U64 distance=U64(count)*unit-rem;
            // The next integer is even in the 64-bit-significand representation.
            if(distance<=(U64(count)<<half_shift))++result;
        }
    }
    return U32(result);
}
__device__ __forceinline__ U64 transpose(U64 b) {
    b=(b&0xFF00FF0000FF00FFULL)|((b&0x00FF00FF00000000ULL)>>24)|((b&0x00000000FF00FF00ULL)<<24);
    return (b&0xF0F00F0FF0F00F0FULL)|((b&0x0F0F00000F0F0000ULL)>>12)|((b&0x0000F0F00000F0F0ULL)<<12);
}
__device__ __forceinline__ U64 lr(U64 b) {
    b=((b&0xff00ff00ff00ff00ULL)>>8)|((b&0x00ff00ff00ff00ffULL)<<8);
    return ((b&0xf0f0f0f0f0f0f0f0ULL)>>4)|((b&0x0f0f0f0f0f0f0f0fULL)<<4);
}
__device__ __forceinline__ U64 ud(U64 b) {
    b=(b>>32)|(b<<32);
    return ((b&0xffff0000ffff0000ULL)>>16)|((b&0x0000ffff0000ffffULL)<<16);
}
__device__ __forceinline__ U64 canonical(U64 b) {
    U64 a=lr(b),c=ud(b),d=lr(c),t=transpose(b);
    return min(min(min(b,a),min(c,d)),min(min(t,lr(t)),min(ud(t),lr(ud(t)))));
}
__device__ __forceinline__ U64 horizontal(U64 b,const U32* moves,bool right) {
    U64 out=0;
    #pragma unroll
    for(unsigned i=0;i<4;i++)out|=U64((moves[(b>>(16*i))&65535]>>(right?16:0))&65535)<<(16*i);
    return out;
}
__device__ __forceinline__ U64 move(U64 b,const U32* moves,unsigned d) {
    return d<2?horizontal(b,moves,d==1):transpose(horizontal(transpose(b),moves,d==3));
}
__device__ __forceinline__ U64 mix(U64 x){x^=x>>33;x*=0xff51afd7ed558ccdULL;x^=x>>33;x*=0xc4ceb9fe1a85ec53ULL;return x^(x>>33);}
__device__ __forceinline__ void quadrants(U64 b,U16& nw,U16& ne,U16& sw,U16& se){
    nw=((b>>60)&15)|((b>>52)&240)|((b>>36)&3840)|((b>>28)&61440);
    ne=((b>>52)&15)|((b>>44)&240)|((b>>28)&3840)|((b>>20)&61440);
    sw=((b>>28)&15)|((b>>20)&240)|((b>>4)&3840)|((b<<4)&61440);
    se=((b>>20)&15)|((b>>12)&240)|((b<<4)&3840)|((b<<12)&61440);
}
__device__ __forceinline__ U64 pairbits(U16 word,unsigned byte){U32 x=(word>>(8*byte))&255;return ((x&15)<<4)|(x>>4);}
__device__ __forceinline__ U64 pack(U16 nw,U16 ne,U16 sw,U16 se){
    return (pairbits(nw,0)<<56)|(pairbits(nw,1)<<40)|(pairbits(ne,0)<<48)|(pairbits(ne,1)<<32)
         | (pairbits(sw,0)<<24)|(pairbits(sw,1)<<8)|(pairbits(se,0)<<16)|pairbits(se,1);
}
__device__ __forceinline__ U32 lookup(U64 b,const U32* lut,Future f,U32 modulus){
    U16 nw,ne,sw,se;quadrants(b,nw,ne,sw,se);
    const U32 *a=lut+4*U32(nw),*c=lut+4*U32(ne),*d=lut+4*U32(sw),*e=lut+4*U32(se);
    U32 row=(min(a[0]+c[0],d[0]+e[0])/2)%modulus;
    U32 col=(min(a[0]+d[0],c[0]+e[0])/2)%modulus;
    U32 cid=row*modulus+col, start=f.cells[2*cid],mask=f.cells[2*cid+1];
    U64 key=(U64(nw)<<48)|(U64(c[1])<<32)|(U64(d[1])<<16)|e[1];
    U32 rank=(c[2]*d[3]+d[2])*e[3]+e[2];
    U32 slot=mix(key)&mask;
    Entry entry=f.hash[start+slot];
    while(entry.word!=0xffffffffU && entry.key!=key){slot=(slot+1)&mask;entry=f.hash[start+slot];}
    if(entry.word==0xffffffffU)return 0;
    U32 wi=entry.word+rank/64,bi=rank&63;U64 bits=f.words[wi];
    if(((bits>>bi)&1)==0)return 0;
    U32 ordinal=f.bases[wi]+__popcll(bits&((1ULL<<bi)-1));
    return f.values[entry.row+ordinal];
}
extern "C" __global__ void decode(const Task* tasks,U32 count,const U32* offsets,const U16* unrank,U64* boards){
    U32 i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=count)return;
    Task t=tasks[i];U32 row=t.row;
    U16 nw=t.key>>48,ne=t.key>>32,sw=t.key>>16,se=t.key;
    while(t.bits){U32 r=t.rank+__ffsll(t.bits)-1;t.bits&=t.bits-1;
        U32 z=r/t.se,rs=r-z*t.se,rn=z/t.sw,rw=z-rn*t.sw;
        boards[row++]=pack(nw,unrank[offsets[ne]+rn],unrank[offsets[sw]+rw],unrank[offsets[se]+rs]);
    }
}
extern "C" __global__ void solve(
    const U64* boards,U32 count,const U32* lut,const U32* moves,
    const Entry* h2,const U32* c2,const U64* w2,const U32* b2,const U32* v2,
    const Entry* h4,const U32* c4,const U64* w4,const U32* b4,const U32* v4,
    U32 modulus,U32 target,U32 terminal,double p4,U64 numerator,U32 denominator_bits,U32 cpu80,U32* output){
    U32 i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=count)return;
    U64 board=boards[i];U32 empties=0;
    #pragma unroll
    for(U32 p=0;p<16;p++){U32 r=(board>>(4*p))&15;if(r==target){output[i]=terminal;return;}if(r==0)empties|=1U<<p;}
    const U32 n=__popc(empties);if(!n){output[i]=0;return;}
    Future f2{h2,c2,w2,b2,v2},f4{h4,c4,w4,b4,v4};
    U64 sum2=0,sum4=0;
    while(empties){U32 p=__ffs(empties)-1;empties&=empties-1;
        U64 s2=board|(1ULL<<(4*p)),s4=board|(2ULL<<(4*p));U32 best2=0,best4=0;
        #pragma unroll
        for(U32 d=0;d<4;d++){
            U64 m2=move(s2,moves,d);if(m2!=s2)best2=max(best2,lookup(canonical(m2),lut,f2,modulus));
            U64 m4=move(s4,moves,d);if(m4!=s4)best4=max(best4,lookup(canonical(m4),lut,f4,modulus));
        }
        sum2+=best2;sum4+=best4;
    }
    output[i]=cpu80?reduce_cpu80(sum2,sum4,n,numerator,denominator_bits)
                  :__double2uint_rz((double(sum2)*(1.0-p4)+double(sum4)*p4)/double(n));
}
extern "C" __global__ void rounding_probe(const U64* s2,const U64* s4,const U32* counts,U32 n,
                                         U64 numerator,U32 bits,U32* output){
    U32 i=blockDim.x*blockIdx.x+threadIdx.x;if(i<n)output[i]=reduce_cpu80(s2[i],s4[i],counts[i],numerator,bits);
}
// Cache-capacity diagnostic: same real source boards and exact values, independent
// physically replicated future tables. This is NOT a larger real formation.
extern "C" __global__ void solve_capacity(
    const U64* boards,U32 count,const U32* lut,const U32* moves,
    const Entry* h2,const U32* c2,const U64* w2,const U32* b2,const U32* v2,
    const Entry* h4,const U32* c4,const U64* w4,const U32* b4,const U32* v4,
    U32 modulus,U32 target,U32 terminal,U64 numerator,U32 denominator_bits,
    const U64* strides,U32 replicas,U32* output){
    U32 i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=count)return;
    U64 board=boards[i];U32 empties=0;
    #pragma unroll
    for(U32 p=0;p<16;p++){U32 r=(board>>(4*p))&15;if(r==target){output[i]=terminal;return;}if(r==0)empties|=1U<<p;}
    const U32 n=__popc(empties);if(!n){output[i]=0;return;}
    U32 replica=mix(i)%replicas;
    Future f2{h2+replica*strides[0],c2+replica*strides[1],w2+replica*strides[2],b2+replica*strides[3],v2+replica*strides[4]};
    Future f4{h4+replica*strides[5],c4+replica*strides[6],w4+replica*strides[7],b4+replica*strides[8],v4+replica*strides[9]};
    U64 sum2=0,sum4=0;
    while(empties){U32 p=__ffs(empties)-1;empties&=empties-1;
        U64 s2=board|(1ULL<<(4*p)),s4=board|(2ULL<<(4*p));U32 best2=0,best4=0;
        #pragma unroll
        for(U32 d=0;d<4;d++){
            U64 m2=move(s2,moves,d);if(m2!=s2)best2=max(best2,lookup(canonical(m2),lut,f2,modulus));
            U64 m4=move(s4,moves,d);if(m4!=s4)best4=max(best4,lookup(canonical(m4),lut,f4,modulus));
        }
        sum2+=best2;sum4+=best4;
    }
    output[i]=reduce_cpu80(sum2,sum4,n,numerator,denominator_bits);
}
extern "C" __global__ void primitives(const U64* boards,U32 n,const U32* moves,U64* output){
    U32 i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
    for(U32 d=0;d<4;d++)output[5*i+d]=move(boards[i],moves,d);
    output[5*i+4]=canonical(boards[i]);
}
