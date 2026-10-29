#include "host_simd_abi.hpp"
#include <bit>
#include <cmath>
#include <iostream>
#include <span>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

namespace {
using namespace openmx::core::host;
void require(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }

// An inactive lane must not touch memory, including at a page boundary.
template<class T> struct Guarded {
  size_t page=size_t(sysconf(_SC_PAGESIZE));
  void* mapping=mmap(nullptr,2*page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  T* data;
  explicit Guarded(int count) {
    require(mapping!=MAP_FAILED,"mmap failed");
    require(mprotect(static_cast<char*>(mapping)+page,page,PROT_NONE)==0,"mprotect failed");
    data=reinterpret_cast<T*>(static_cast<char*>(mapping)+page)-count;
  }
  ~Guarded() { munmap(mapping,2*page); }
};

template<class V> void tails() {
  using T=typename V::value_type;
  using I=simd::basic_simd<std::int32_t,typename V::abi_type>;
  using M=typename V::mask_type;
  constexpr int count=V::size();
  for(int live=0;live<=count;++live) {
    Guarded<T> input(live),output(live);
    for(int k=0;k<live;++k){input.data[k]=T(3*k+1);output.data[k]=T(-99);}
    const M mask([&](int k){return k<live;});
    const I indices([&](int k){return k<live?live-1-k:1000000;});
    const auto load=simd::simd_partial_load<V>(input.data,mask);
    const auto gather=simd::unchecked_gather_from<V>(std::span<const T>(input.data,live),mask,indices);
    for(int k=0;k<count;++k) {
      require(load[k]==(k<live?input.data[k]:T(0)),"masked load or inactive lane");
      require(gather[k]==(k<live?input.data[live-1-k]:T(0)),"masked gather or inactive lane");
    }
    simd::simd_partial_store(load,output.data,mask);
    for(int k=0;k<live;++k)require(output.data[k]==input.data[k],"masked store");
    simd::unchecked_scatter_to(gather,std::span<T>(output.data,live),mask,indices);
    for(int k=0;k<live;++k)require(output.data[k]==input.data[k],"masked scatter");
    require(simd::any_of(mask)==(live!=0) && simd::all_of(mask)==(live==count),"mask reduction");
  }
}

void conversions() {
  const double values[]={1.0+std::ldexp(1.0,-24),1.0+3*std::ldexp(1.0,-24),
      -1.0-std::ldexp(1.0,-24),-1.0-3*std::ldexp(1.0,-24),0.0,-0.0,1e-40,-1e-40};
  for(int first=0;first<8;first+=lanes) {
    const Vector d([&](int k){return values[(first+k)%8];});
    const auto f=narrow(d);
    const Vector widened(f);
    for(int k=0;k<lanes;++k) {
      volatile double source=values[(first+k)%8];
      const float expected=static_cast<float>(source);
      require(std::bit_cast<std::uint32_t>(f[k])==std::bit_cast<std::uint32_t>(expected),"float rounding or signed zero");
      require(std::bit_cast<std::uint64_t>(widened[k])==std::bit_cast<std::uint64_t>(double(expected)),"float widening");
    }
  }
  const Vector fractional([](int k){return k%2?double(k)+0.9:-double(k)-0.9;});
  const IndexVector indices(fractional);
  const Vector back(indices);
  for(int k=0;k<lanes;++k)require(indices[k]==int(fractional[k]) && back[k]==double(indices[k]),"integer conversion");
  const auto mask=fractional<Vector(0.0);
  const IndexVector::mask_type imask(mask);
  const Vector::mask_type dmask(imask);
  for(int k=0;k<lanes;++k)require(mask[k]==imask[k] && imask[k]==dmask[k],"mixed element mask conversion");
}

// Exercise every mask pattern across element widths. Inactive gather/scatter
// indices point into protected memory; a conversion must neither move a live
// lane nor enable any of the unused upper lanes of a half-width vector.
template<class V,class SourceMask> void mask_conversions() {
  using T=typename V::value_type;
  using M=typename V::mask_type;
  using I=simd::basic_simd<std::int32_t,typename V::abi_type>;
  constexpr int count=V::size();
  Guarded<T> input(count),output(count);
  for(int k=0;k<count;++k)input.data[k]=T(k+1);
  for(unsigned pattern=0;pattern<(1u<<count);++pattern) {
    const SourceMask source([&](int k){return (pattern>>k)&1u;});
    const M converted(source);
    const SourceMask roundtrip(converted);
    require(simd::any_of(converted)==(pattern!=0),"converted mask any_of");
    require(simd::all_of(converted)==(pattern==(1u<<count)-1),"converted mask all_of");
    const I indices([&](int k){return source[k]?k:count;});
    const auto value=simd::unchecked_gather_from<V>(std::span<const T>(input.data,count),source,indices);
    for(int k=0;k<count;++k) {
      require(converted[k]==source[k] && roundtrip[k]==source[k],"mask pattern conversion");
      require(value[k]==(source[k]?input.data[k]:T(0)),"cross-width masked gather");
      output.data[k]=T(-99);
    }
    simd::unchecked_scatter_to(value,std::span<T>(output.data,count),source,indices);
    for(int k=0;k<count;++k)
      require(output.data[k]==(source[k]?input.data[k]:T(-99)),"cross-width masked scatter");
    // Contiguous access at the page end detects nonzero upper predicate bits.
    for(int k=0;k<count;++k)output.data[k]=T(-99);
    simd::simd_partial_store(simd::simd_partial_load<V>(input.data,converted),output.data,converted);
    for(int k=0;k<count;++k)
      require(output.data[k]==(source[k]?input.data[k]:T(-99)),"converted mask contiguous access");
  }
}
}

extern "C" int OPENMX_HOST_SIMD_FACTORY(openmx_test_abi)() {
  try {
    tails<Vector>();tails<NarrowFloatVector>();tails<IndexVector>();
    tails<FloatVector>();tails<IntVector>();conversions();
    mask_conversions<Vector,IndexVector::mask_type>();
    mask_conversions<Vector,NarrowFloatVector::mask_type>();
    mask_conversions<IndexVector,Vector::mask_type>();
    mask_conversions<NarrowFloatVector,Vector::mask_type>();
    mask_conversions<IndexVector,NarrowFloatVector::mask_type>();
  } catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
  }
  return 0;
}
