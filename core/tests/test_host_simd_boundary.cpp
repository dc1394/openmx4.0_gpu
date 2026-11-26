#include "openmx_core_grid.h"
#include "openmx_core_host_simd.h"
#include <Kokkos_SIMD.hpp>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

static void check_row_updates()
{
  for (int mode = OPENMX_GRID_SCALAR; mode < OPENMX_GRID_MODE_COUNT; ++mode) {
    if (!openmx_core_host_simd_usable(mode)) continue;
    const auto update = openmx_core_host_add_rows(mode);
    if (update == nullptr) throw std::runtime_error("a usable host SIMD mode has no row kernel");
    update(nullptr,0,nullptr,0,1,0,1.0);
    for (const size_t columns : {1,3,4,5,7,8,9,13,17,169}) {
      const size_t rows = 3, dst_stride = columns+2, src_stride = columns+5;
      // Offset the start and leave no padding behind the last row.
      std::vector<double> src(1+(rows-1)*src_stride+columns);
      for (size_t i = 0; i < src.size(); ++i) src[i] = 0.13*static_cast<double>(i)-3.1;
      for (const double factor : {1.0,-1.0,0.13}) {
        std::vector<double> dst(1+(rows-1)*dst_stride+columns,-0.3);
        auto expected = dst;
        for (size_t i = 0; i < rows; ++i) {
          for (size_t j = 0; j < columns; ++j) {
            const volatile double product = factor*src[1+i*src_stride+j];
            expected[1+i*dst_stride+j] += product;
          }
        }
        update(dst.data()+1,dst_stride,src.data()+1,src_stride,rows,columns,factor);
        if (std::memcmp(dst.data(),expected.data(),sizeof(double)*dst.size()) != 0)
          throw std::runtime_error("host SIMD row update differs from the scalar loop");
      }
    }
  }
}

int main() {
  try {
    constexpr auto default_width=Kokkos::Experimental::simd<double>::size();
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    static_assert(default_width==1,"GPU-default Kokkos ABI assumption changed");
#endif
    if(Kokkos::is_initialized())throw std::runtime_error("test runtime already initialized");
    char error[256];
    auto check=[&](int ok){if(!ok)throw std::runtime_error(error);};
    std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)> context(
        openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);
    check(bool(context));OpenMXGridWorkspace* w;
    check(openmx_core_grid_workspace(context.get(),&w,error,sizeof(error)));
    const int mode=openmx_core_grid_simd(nullptr),width=openmx_core_grid_simd_width(mode);
    const int expected=mode==OPENMX_GRID_AVX512 || mode==OPENMX_GRID_SVE512?8:
        mode==OPENMX_GRID_AVX2 || mode==OPENMX_GRID_SVE256?4:
        mode==OPENMX_GRID_NEON || mode==OPENMX_GRID_SVE128?2:1;
    if(width!=expected)throw std::runtime_error("host SIMD width follows the device ABI");
    const size_t counts[]={7};OpenMXGridDensity density;
    check(openmx_core_grid_density_prepare(w,1,1,counts,0,nullptr,nullptr,&density,error,sizeof(error)));
    OpenMXGridAtom* atoms;OpenMXGridPair* pair;
    check(openmx_core_grid_plan(w,0,1,1,1,1,1,mode,&atoms,&pair,error,sizeof(error)));
    double values[]={1,2,3,4,5,6,7},*rows[7];const int indices[]={0,1,2,3,4,5,6};
    for(int i=0;i<7;++i)rows[i]=values+i;
    double coefficient=2,*matrix[]={&coefficient};
    atoms[0]={0,1,0};pair[0]={};pair[0].rows=pair[0].columns=1;pair[0].points=7;
    pair[0].center_points=pair[0].neighbor_points=indices;
    pair[0].center=pair[0].neighbor={rows,0,0,8};pair[0].matrix[0]=matrix;
    check(openmx_core_grid_density_worker(w,0,1,nullptr,nullptr,error,sizeof(error)));
    for(int i=0;i<7;++i)if(density.atoms[0][0][i]!=2*values[i]*values[i])
      throw std::runtime_error("host SIMD tail failed");
    context.reset();
    check_row_updates();
    if(Kokkos::is_initialized())throw std::runtime_error("CPU fallback initialized a device runtime");
    std::cout<<"default SIMD width="<<default_width<<"; host SIMD width="<<width<<'\n';
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
