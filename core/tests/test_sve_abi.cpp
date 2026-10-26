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
}

extern "C" int OPENMX_HOST_SIMD_FACTORY(openmx_test_abi)() {
  try {
    tails<Vector>();tails<NarrowFloatVector>();tails<IndexVector>();
    tails<FloatVector>();tails<IntVector>();conversions();
  } catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
  }
  return 0;
}
