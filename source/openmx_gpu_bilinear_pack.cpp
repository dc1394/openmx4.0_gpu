#include "openmx_gpu_bilinear.h"
#include "openmx_host_memory.h"
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

extern "C" int openmx_gpu_local_size_noncollective(void);

namespace {
struct Key {
    const void *pointer;
    size_t offset,stride;
    int count,width,jagged;
    bool operator==(const Key &b) const {
        return pointer==b.pointer && offset==b.offset && stride==b.stride &&
            count==b.count && width==b.width && jagged==b.jagged;
    }
};
struct Hash {
    size_t operator()(const Key &k) const {
        size_t h=reinterpret_cast<uintptr_t>(k.pointer);
        for (size_t v:{k.offset,k.stride,static_cast<size_t>(k.count),
                       static_cast<size_t>(k.width),static_cast<size_t>(k.jagged)})
            h^=v+size_t(0x9e3779b9)+(h<<6)+(h>>2);
        return h;
    }
};
struct Builder {
    int spins;
    size_t budget=SIZE_MAX,used=0;
    std::vector<uint64_t> ptr;
    std::vector<OpenMXBilinearIndex> terms;
    std::vector<double> orbitals,matrices;
    std::unordered_map<Key,uint64_t,Hash> orb_map,dm_map;
    void reserve(size_t count,size_t width) {
        if (count>(budget-used)/width) throw std::bad_alloc();
        used+=count*width;
    }
    uint64_t orbital(const OpenMXBilinearSpan &s) {
        if (s.count<1 || (s.width!=4 && s.width!=8) || (!s.data && !s.rows))
            throw std::invalid_argument("orbital span");
        Key key{s.rows?s.rows:s.data,s.offset,s.stride,s.count,s.width,s.rows!=nullptr};
        auto found=orb_map.find(key);if (found!=orb_map.end()) return found->second;
        /* Include vector growth, the final packet and hash-map nodes. Do
           not let a capacity retry exhaust the node's host/cgroup memory. */
        reserve(1,128);reserve(static_cast<size_t>(s.count),3*sizeof(double));
        uint64_t at=orbitals.size();orb_map.emplace(key,at);
        for (int i=0;i<s.count;i++) orbitals.push_back(openmx_bilinear_value(&s,i));
        return at;
    }
    int term(const OpenMXBilinearTerm &t) {
        OpenMXBilinearIndex x={orbital(t.left),orbital(t.right),0,t.left.count,t.right.count};
        for (int i=0;i<4;i++) x.scale[i]=t.scaled?t.scale[i]:1.0;
        Key key{t.matrix?t.matrix:t.matrix_rows[0],0,static_cast<size_t>(x.n1),x.n0,spins,t.matrix==nullptr};
        if (!key.pointer) return 0;
        auto found=dm_map.find(key);
        if (found!=dm_map.end()) x.matrix=found->second;
        else {
            reserve(1,128);
            reserve(static_cast<size_t>(spins)*x.n0*x.n1,3*sizeof(double));
            x.matrix=matrices.size();dm_map.emplace(key,x.matrix);
            for (int spin=0;spin<spins;spin++) for (int i=0;i<x.n0;i++) for (int j=0;j<x.n1;j++)
                matrices.push_back(openmx_bilinear_matrix(&t,spin,i,j));
        }
        reserve(1,3*sizeof(x));terms.push_back(x);return 1;
    }
};
/* Catch at the C callback boundary: an exception must never unwind through
   OpenMX's C visitor or an OpenMP worker frame. */
int emit(void *context,const OpenMXBilinearTerm *term) {
    try {return static_cast<Builder*>(context)->term(*term);} catch (...) {return 0;}
}
bool append_bytes(size_t &bytes,size_t count,size_t width) {
    if (count>(SIZE_MAX-bytes)/width) return false;
    bytes+=count*width;return true;
}
}

extern "C" int openmx_gpu_bilinear_pack(const OpenMXBilinearSource *source,
    size_t first,size_t last,void **packet,size_t *bytes)
{
    *packet=nullptr;*bytes=0;
    if (!source || !source->visit || last<first || last>source->rows || last-first==SIZE_MAX ||
        (source->spins!=1 && source->spins!=2 && source->spins!=4)) return 0;
    try {
        Builder b;b.spins=source->spins;
        unsigned long long available=openmx_host_mem_available();
        int sharing=openmx_gpu_local_size_noncollective();if (sharing<1) sharing=1;
        if (available) b.budget=static_cast<size_t>(available/(2ULL*sharing));
        b.reserve(last-first+1,3*sizeof(uint64_t));b.ptr.push_back(0);
        for (size_t row=first;row<last;row++) {
            if (!source->visit(source->context,row,emit,&b)) return 0;
            b.ptr.push_back(b.terms.size());
        }
        size_t n=sizeof(OpenMXBilinearHeader);
        if (!append_bytes(n,b.ptr.size(),sizeof(uint64_t)) ||
            !append_bytes(n,b.terms.size(),sizeof(OpenMXBilinearIndex)) ||
            !append_bytes(n,b.orbitals.size(),sizeof(double)) ||
            !append_bytes(n,b.matrices.size(),sizeof(double))) return 0;
        auto *h=static_cast<OpenMXBilinearHeader*>(calloc(n,1));if (!h) return 0;
        *h={last-first,b.terms.size(),b.orbitals.size(),b.matrices.size(),source->spins,source->negate_fourth};
        unsigned char *p=reinterpret_cast<unsigned char*>(h+1);
        memcpy(p,b.ptr.data(),b.ptr.size()*sizeof(uint64_t));p+=b.ptr.size()*sizeof(uint64_t);
        if (!b.terms.empty()) memcpy(p,b.terms.data(),b.terms.size()*sizeof(OpenMXBilinearIndex));p+=b.terms.size()*sizeof(OpenMXBilinearIndex);
        if (!b.orbitals.empty()) memcpy(p,b.orbitals.data(),b.orbitals.size()*sizeof(double));p+=b.orbitals.size()*sizeof(double);
        if (!b.matrices.empty()) memcpy(p,b.matrices.data(),b.matrices.size()*sizeof(double));
        *packet=h;*bytes=n;return 1;
    } catch (...) {return 0;}
}
