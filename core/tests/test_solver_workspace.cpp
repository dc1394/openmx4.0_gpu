#include "openmx_core_solver_workspace.h"
#include "allocation_faults.hpp"
#include <openmx/core/matrix_scratch.hpp>
#include <openmx/core/unpivoted_inverse.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void equal(double a,double b){require(std::isfinite(a) && std::abs(a-b)<=1e-12*(1+std::abs(b)),"solver numerical value differs");}
void run(OpenMXSolverOperation operation,void* argument=nullptr,int workers=4) {
  char error[256];double result=-1;
  require(openmx_core_solver_run(workers,operation,argument,&result,error,sizeof(error)),error);equal(result,17);
}
int buffers(OpenMXSolverInvocation* owner,void*,double* result) {
  char error[256];const size_t shape[4]={3,2,2,1};OpenMXSolverRealTensor view{};
  const char* key="long-buffer-name-to-check-allocation-free-lookup";
  require(openmx_core_solver_real(owner,0,key,shape,&view,error,sizeof(error)),error);
  for(int i=0;i<12;i++)view.values[i]=i+.25;
  auto saved=view;
  openmx::test::fail_allocation_after(0);
  int ok=openmx_core_solver_real(owner,0,key,shape,&view,error,sizeof(error));
  openmx::test::fail_allocation_after(-1);
  require(ok && view.values==saved.values,"repeated request allocated/moved storage");
  const size_t smaller[4]={2,2,1,1};
  openmx::test::fail_allocation_after(0);
  ok=openmx_core_solver_real(owner,0,key,smaller,&view,error,sizeof(error));
  openmx::test::fail_allocation_after(-1);
  require(ok && view.values==saved.values,"shrinking buffer discarded its capacity");
  for(int i=0;i<4;i++)equal(view.values[i],i+.25);
  OpenMXSolverComplexTensor complex{};int* integers;void* bytes;
  require(openmx_core_solver_complex(owner,0,"z",shape,&complex,error,sizeof(error)),error);
  require(openmx_core_solver_int(owner,0,"i",8,&integers,error,sizeof(error)),error);
  require(openmx_core_solver_bytes(owner,0,"opaque",13,7,&bytes,error,sizeof(error)),error);
  require(reinterpret_cast<std::uintptr_t>(bytes)%alignof(std::max_align_t)==0,"opaque storage alignment");
  complex.values[11]={3,4};integers[7]=29;
  std::memset(bytes,0x4b,13*7);equal(complex.values[11].i,4);require(integers[7]==29,"integer buffer");
  OpenMXSolverIntTensor integer_view;
  const size_t integer_shape[4]={2,4,1,1};
  require(openmx_core_solver_int_tensor(owner,0,"i",integer_shape,&integer_view,error,sizeof(error)),error);
  require(integer_view.values==integers && openmx_solver_int_matrix_row(&integer_view,1)[3]==29,
      "integer tensor lost its storage or row stride");
  *result=17;return 1;
}
int overwrite(OpenMXSolverInvocation* owner,void*,double* result) {
  // Poison actual allocations through the same shared-core fault hook used by
  // the failure tests. Inspect bytes before initializing arithmetic objects.
  struct Fill {
    Fill(){openmx::test::fill_allocations_with(0xa5);}
    ~Fill(){openmx::test::fill_allocations_with(-1);}
  } fill;
  char error[256];const size_t shape[4]={3,5,1,1};
  OpenMXSolverRealTensor view{},zeroed{},worker{},named{};
  require(openmx_core_solver_real_overwrite(owner,0,"overwrite",shape,&view,error,sizeof(error)),error);
  const auto* bytes=reinterpret_cast<const unsigned char*>(view.values);
  for(size_t i=0;i<15*sizeof(double);++i)require(bytes[i]==0xa5,"overwrite storage was initialized");
  require(openmx_core_solver_real(owner,0,"overwrite",shape,&zeroed,error,sizeof(error)),error);
  require(zeroed.values!=view.values,"initialized and overwrite storage alias");
  for(int i=0;i<15;++i)equal(zeroed.values[i],0);
  for(size_t i=0;i<3;++i)for(size_t j=0;j<5;++j)
    openmx_solver_real_matrix_row(&view,i)[j]=double(10*i+j)+.25;
  const auto saved=view;
  openmx::test::fail_allocation_after(0);
  const int reused=openmx_core_solver_real_overwrite(owner,0,"overwrite",shape,&view,error,sizeof(error));
  openmx::test::fail_allocation_after(-1);
  require(reused && view.values==saved.values,"overwrite reuse allocated/moved storage");
  equal(view.values[14],24.25);
  const size_t smaller[4]={2,3,1,1};
  openmx::test::fail_allocation_after(0);
  const int shrunk=openmx_core_solver_real_overwrite(owner,0,"overwrite",smaller,&view,error,sizeof(error));
  openmx::test::fail_allocation_after(-1);
  require(shrunk && view.values==saved.values && view.extent[1]==3,"overwrite shrink lost capacity/shape");
  equal(openmx_solver_real_matrix_row(&view,1)[2],10.25);
  const auto current=view;const size_t larger[4]={4,8,1,1};
  for(long failure=0;failure<2;++failure) {
    openmx::test::fail_allocation_after(failure);
    const int ok=openmx_core_solver_real_overwrite(owner,0,"overwrite",larger,&view,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    require(!ok && error[0] && !std::memcmp(&view,&current,sizeof(view)),"failed overwrite growth changed output");
    equal(current.values[14],24.25);
  }
  require(openmx_core_solver_real_overwrite(owner,1,"overwrite",shape,&worker,error,sizeof(error)),error);
  require(openmx_core_solver_real_overwrite(owner,0,"other",shape,&named,error,sizeof(error)),error);
  require(worker.values!=view.values && named.values!=view.values && named.values!=worker.values,
      "workers/named overwrite buffers alias");
  require(openmx_core_solver_real_overwrite(owner,0,"overwrite",larger,&view,error,sizeof(error)),error);
  bytes=reinterpret_cast<const unsigned char*>(view.values);
  for(size_t i=0;i<32*sizeof(double);++i)require(bytes[i]==0xa5,"grown overwrite storage was initialized");
  const auto grown=view;
  const size_t overflow[4]={SIZE_MAX,2,1,1};
  require(!openmx_core_solver_real_overwrite(owner,0,"overwrite",overflow,&view,error,sizeof(error)),"overwrite overflow accepted");
  require(!std::memcmp(&view,&grown,sizeof(view)),"invalid overwrite shape changed output");
  require(!openmx_core_solver_real_overwrite(owner,-1,"overwrite",shape,&view,error,sizeof(error)),"invalid overwrite worker accepted");
  require(!openmx_core_solver_real_overwrite(owner,0,"",shape,&view,error,sizeof(error)),"empty overwrite name accepted");
  require(!openmx_core_solver_real_overwrite(nullptr,0,"overwrite",shape,&view,error,sizeof(error)),"missing overwrite owner accepted");
  require(!openmx_core_solver_real_overwrite(owner,0,"overwrite",nullptr,&view,error,sizeof(error)),"missing overwrite shape accepted");
  require(!openmx_core_solver_real_overwrite(owner,0,"overwrite",shape,nullptr,error,sizeof(error)),"missing overwrite output accepted");
  const size_t empty[4]={3,0,1,1};
  require(openmx_core_solver_real_overwrite(owner,0,"empty",empty,&view,error,sizeof(error)),error);
  require(!view.values && view.extent[0]==3 && !view.extent[1],"empty overwrite tensor has storage");
  *result=17;return 1;
}
int complex_overwrite(OpenMXSolverInvocation* owner,void*,double* result) {
  struct Fill {
    Fill(){openmx::test::fill_allocations_with(0xa5);}
    ~Fill(){openmx::test::fill_allocations_with(-1);}
  } fill;
  char error[256];const size_t shape[4]={3,4,1,1};
  OpenMXSolverComplexTensor view{},zeroed{},worker{};
  require(openmx_core_solver_complex_overwrite(owner,0,"complex-overwrite",shape,&view,error,sizeof(error)),error);
  const auto* bytes=reinterpret_cast<const unsigned char*>(view.values);
  for(size_t i=0;i<12*sizeof(dcomplex);++i)require(bytes[i]==0xa5,"complex overwrite storage was initialized");
  require(openmx_core_solver_complex(owner,0,"complex-overwrite",shape,&zeroed,error,sizeof(error)),error);
  require(zeroed.values!=view.values,"initialized and overwrite complex buffers alias");
  for(int i=0;i<12;i++){equal(zeroed.values[i].r,0);equal(zeroed.values[i].i,0);view.values[i]={100+i*.25,-i*.5};}
  const auto saved=view;
  openmx::test::fail_allocation_after(0);
  const int reused=openmx_core_solver_complex_overwrite(owner,0,"complex-overwrite",shape,&view,error,sizeof(error));
  openmx::test::fail_allocation_after(-1);
  require(reused && view.values==saved.values,"complex overwrite reuse allocated/moved storage");
  equal(openmx_solver_complex_matrix_row(&view,2)[3].r,102.75);
  equal(openmx_solver_complex_matrix_row(&view,2)[3].i,-5.5);
  const size_t smaller[4]={2,2,1,1};
  openmx::test::fail_allocation_after(0);
  const int shrunk=openmx_core_solver_complex_overwrite(owner,0,"complex-overwrite",smaller,&view,error,sizeof(error));
  openmx::test::fail_allocation_after(-1);
  require(shrunk && view.values==saved.values && view.extent[1]==2,"complex overwrite shrink lost capacity/shape");
  equal(openmx_solver_complex_matrix_row(&view,1)[1].r,100.75);
  const auto current=view;const size_t larger[4]={4,7,1,1};
  for(long failure=0;failure<2;++failure){
    openmx::test::fail_allocation_after(failure);
    const int ok=openmx_core_solver_complex_overwrite(owner,0,"complex-overwrite",larger,&view,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    require(!ok && error[0] && !std::memcmp(&view,&current,sizeof(view)),"failed complex overwrite growth changed output");
    equal(saved.values[11].i,-5.5);
  }
  require(openmx_core_solver_complex_overwrite(owner,1,"complex-overwrite",shape,&worker,error,sizeof(error)),error);
  require(worker.values!=saved.values,"workers alias complex overwrite storage");
  require(openmx_core_solver_complex_overwrite(owner,0,"complex-overwrite",larger,&view,error,sizeof(error)),error);
  bytes=reinterpret_cast<const unsigned char*>(view.values);
  for(size_t i=0;i<28*sizeof(dcomplex);++i)require(bytes[i]==0xa5,"grown complex overwrite storage was initialized");
  const auto grown=view;const size_t overflow[4]={SIZE_MAX,2,1,1};
  require(!openmx_core_solver_complex_overwrite(owner,0,"complex-overwrite",overflow,&view,error,sizeof(error)),"complex overwrite overflow accepted");
  require(!std::memcmp(&view,&grown,sizeof(view)),"invalid complex overwrite shape changed output");
  require(!openmx_core_solver_complex_overwrite(owner,-1,"complex-overwrite",shape,&view,error,sizeof(error)),"invalid complex worker accepted");
  require(!openmx_core_solver_complex_overwrite(owner,0,"",shape,&view,error,sizeof(error)),"empty complex buffer name accepted");
  require(!openmx_core_solver_complex_overwrite(nullptr,0,"complex-overwrite",shape,&view,error,sizeof(error)),"missing complex owner accepted");
  require(!openmx_core_solver_complex_overwrite(owner,0,"complex-overwrite",nullptr,&view,error,sizeof(error)),"missing complex shape accepted");
  require(!openmx_core_solver_complex_overwrite(owner,0,"complex-overwrite",shape,nullptr,error,sizeof(error)),"missing complex output accepted");
  const size_t empty[4]={3,0,1,1};
  require(openmx_core_solver_complex_overwrite(owner,0,"empty",empty,&view,error,sizeof(error)),error);
  require(!view.values && view.extent[0]==3 && !view.extent[1],"empty complex overwrite tensor has storage");
  *result=17;return 1;
}
struct MatrixInput {
  std::array<size_t,5> components{0,2,3,5,6};
  std::array<size_t,7> atoms{0,2,3,4,5,7,8};
  std::array<size_t,9> offsets{0,1,7,13,15,16,22,25,25};
  std::array<int,16> shapes{1,1,2,3,3,2,2,1,1,1,2,3,3,1,0,4};
  std::array<double,25> values{};
  OpenMXElectronicMatrixView view(){return {2,2,4,6,8,25,components.data(),atoms.data(),offsets.data(),shapes.data(),values.data()};}
};
int matrix_scratch(OpenMXSolverInvocation* owner,void*,double* result) {
  char error[256];OpenMXElectronicMatrixView out;
  {
    MatrixInput input;std::fill(input.values.begin(),input.values.end(),-73);
    auto source=input.view();
    struct Fill {
      Fill(){openmx::test::fill_allocations_with(0xa5);}
      ~Fill(){openmx::test::fill_allocations_with(-1);}
    } fill;
    require(openmx_core_solver_matrix_scratch(owner,&source,1,&out,error,sizeof(error)),error);
    const auto* bytes=reinterpret_cast<const unsigned char*>(out.values);
    for(size_t i=0;i<10*sizeof(double);++i)require(bytes[i]==0xa5,"matrix scratch initialized/copied numeric data");
    require(out.groups==1 && out.channels==2 && out.components==2 && out.atoms==3 && out.blocks==4 && out.elements==10,"selected group has wrong shape");
    require(out.component_offsets!=source.component_offsets && out.atom_offsets!=source.atom_offsets &&
        out.shapes!=source.shapes && out.values!=source.values,"scratch borrows source metadata/storage");
    for(auto v:input.values)equal(v,-73);
    for(size_t i=0;i<10;++i)out.values[i]=double(i)+.25;
    equal(openmx_electronic_row(&out,0,1,0,1)[2],6.25);
    equal(openmx_electronic_row(&out,0,1,1,2)[0],9.25);
    require(openmx_electronic_row(&out,1,0,0,0)==out.values+10,"zero-sized block has wrong offset");
    openmx::core::MatrixScratch cpp(source,1);
    auto block=cpp.block(0,1,0);openmx::core::element(block,1,2)=41;
    const auto cpp_view=cpp.view();equal(openmx_electronic_row(&cpp_view,0,1,0,1)[2],41);
    for(auto index:{size_t(2),SIZE_MAX}){
      bool rejected=false;try{(void)cpp.block(index,0,0);}catch(const std::out_of_range&){rejected=true;}
      require(rejected,"invalid C++ matrix component accepted");
    }
    const auto saved=out;
    for(int bad=0;bad<9;++bad){
      auto invalid=source;
      if(bad==0)invalid.groups=SIZE_MAX;
      if(bad==1)invalid.components=3;
      if(bad==2)invalid.atoms=SIZE_MAX;
      if(bad==3)invalid.blocks=SIZE_MAX;
      if(bad==4)invalid.component_offsets=nullptr;
      if(bad==5)invalid.atom_offsets=nullptr;
      if(bad==6)invalid.shapes=nullptr;
      if(bad==7)input.components[2]=1;
      if(bad==8)input.shapes[9]=-1;
      require(!openmx_core_solver_matrix_scratch(owner,&invalid,1,&out,error,sizeof(error)) && error[0],"invalid matrix shape accepted");
      require(!std::memcmp(&out,&saved,sizeof(out)),"failed matrix acquisition changed output");
      input.components[2]=3;input.shapes[9]=1;
    }
    require(!openmx_core_solver_matrix_scratch(owner,&source,2,&out,error,sizeof(error)),"invalid matrix group accepted");
    require(!openmx_core_solver_matrix_scratch(nullptr,&source,0,&out,error,sizeof(error)),"missing matrix invocation accepted");
    require(!openmx_core_solver_matrix_scratch(owner,nullptr,0,&out,error,sizeof(error)),"missing matrix shape accepted");
    require(!openmx_core_solver_matrix_scratch(owner,&source,0,nullptr,error,sizeof(error)),"missing matrix output accepted");
    // All source descriptors and values die here, including both source groups.
  }
  require(out.component_offsets[2]==3 && out.atom_offsets[3]==4 && out.shapes[3]==3,"scratch shape expired with source metadata");
  equal(openmx_electronic_row(&out,0,1,0,1)[2],6.25);
  const size_t components[]={0,1},atoms[]={0,1};const int shapes[]={0,3};
  const OpenMXElectronicMatrixView empty={1,1,1,1,1,0,components,atoms,nullptr,shapes,nullptr};
  require(openmx_core_solver_matrix_scratch(owner,&empty,0,&out,error,sizeof(error)),error);
  require(!out.elements && out.values && openmx_electronic_row(&out,0,0,0,0)==out.values,"empty matrix lacks a safe zero-offset view");
  *result=17;return 1;
}
int unpivoted_inverse(OpenMXSolverInvocation* owner,void*,double* result) {
  struct Fill {
    Fill(){openmx::test::fill_allocations_with(0xff);}
    ~Fill(){openmx::test::fill_allocations_with(-1);}
  } fill;
  std::array<double,80> a,b;char error[256];
  const OpenMXSolverRealTensor input={{8,10,1,1},a.data()},output={{10,8,1,1},b.data()};
  const auto seed=[&](int n){
    a.fill(-71);b.fill(-91);
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)a[i*10+j]=i==j?4+i:.125*(1+i+2*j);
  };
  const auto identity=[&](int n,const std::array<double,80>& original){
    require(a==original,"unpivoted inverse did not restore full input");
    for(int i=0;i<n;++i)for(int j=0;j<n;++j){long double sum=0;
      for(int k=0;k<n;++k)sum+=static_cast<long double>(original[i*10+k])*b[k*8+j];
      require(std::isfinite(sum) && std::abs(sum-(i==j?1.L:0.L))<1e-11L,"unpivoted inverse identity differs");}
    for(int i=0;i<10;++i)for(int j=0;j<8;++j)if(i>=n || j>=n)equal(b[i*8+j],-91);
  };
  for(int n:{1,3,2,4,3}){
    seed(n);const auto original=a;
    require(openmx_core_solver_inverse_unpivoted(owner,0,n-1,&input,&output,error,sizeof(error)),error);identity(n,original);
  }
  seed(3);auto original=a;openmx::test::fail_allocation_after(0);
  const int reused=openmx_core_solver_inverse_unpivoted(owner,0,2,&input,&output,error,sizeof(error));openmx::test::fail_allocation_after(-1);
  require(reused,"unpivoted inverse reuse allocated");identity(3,original);
  // A replacement owns its complete scratch before publication. Failure at
  // each of object/x/y/backup preserves both arrays and the previous capacity.
  for(long failure=0;failure<4;++failure){
    seed(6);const auto source=a,untouched=b;const auto live=openmx::test::live_allocations();
    openmx::test::fail_allocation_after(failure);
    const int ok=openmx_core_solver_inverse_unpivoted(owner,0,5,&input,&output,error,sizeof(error));openmx::test::fail_allocation_after(-1);
    require(!ok && error[0] && a==source && b==untouched && openmx::test::live_allocations()==live,"failed inverse growth changed arrays/leaked");
    seed(4);original=a;openmx::test::fail_allocation_after(0);
    const int retained=openmx_core_solver_inverse_unpivoted(owner,0,3,&input,&output,error,sizeof(error));openmx::test::fail_allocation_after(-1);
    require(retained,"failed inverse growth discarded old capacity");identity(4,original);
  }
  seed(6);original=a;require(openmx_core_solver_inverse_unpivoted(owner,0,5,&input,&output,error,sizeof(error)),error);identity(6,original);
  seed(2);const auto source=a,untouched=b;openmx::test::fail_allocation_after(0);
  const int other=openmx_core_solver_inverse_unpivoted(owner,1,1,&input,&output,error,sizeof(error));openmx::test::fail_allocation_after(-1);
  require(!other && a==source && b==untouched,"another worker reused private inverse scratch");
  require(openmx_core_solver_inverse_unpivoted(owner,1,1,&input,&output,error,sizeof(error)),error);identity(2,source);
  for(int bad=0;bad<9;++bad){
    seed(2);const auto saved_a=a,saved_b=b;auto invalid_input=input,invalid_output=output;int n=1;
    if(bad==0)n=-2;if(bad==1)n=8;
    if(bad==2)invalid_input.extent[2]=2;
    if(bad==3)invalid_input.extent[0]=SIZE_MAX;
    if(bad==4)invalid_input.values=nullptr;
    if(bad==5)invalid_output.values=nullptr;
    if(bad==6)invalid_output.extent[1]=1;
    if(bad==7)invalid_output.values=a.data();
    if(bad==8){invalid_output.values=a.data()+3;invalid_output.extent[0]=7;}
    require(!openmx_core_solver_inverse_unpivoted(owner,0,n,&invalid_input,&invalid_output,error,sizeof(error)) && error[0],"invalid inverse request accepted");
    require(a==saved_a && b==saved_b,"invalid inverse request changed input/output");
  }
  require(!openmx_core_solver_inverse_unpivoted(nullptr,0,1,&input,&output,error,sizeof(error)),"missing inverse scope accepted");
  require(!openmx_core_solver_inverse_unpivoted(owner,-1,1,&input,&output,error,sizeof(error)),"invalid inverse worker accepted");
  require(!openmx_core_solver_inverse_unpivoted(owner,0,1,nullptr,&output,error,sizeof(error)),"missing inverse input accepted");
  require(!openmx_core_solver_inverse_unpivoted(owner,0,1,&input,nullptr,error,sizeof(error)),"missing inverse output accepted");
  for(int singular=0;singular<3;++singular){
    seed(2);int n=singular?2:1;
    if(!singular)a[0]=0;
    else if(singular==1){a[0]=0;a[1]=1;a[10]=1;a[11]=2;}
    else {a[0]=1;a[1]=2;a[10]=2;a[11]=4;}
    const auto saved=a;
    require(openmx_core_solver_inverse_unpivoted(owner,0,n-1,&input,&output,error,sizeof(error)),"legacy singular arithmetic was replaced by a failure");
    bool nonfinite=false;for(int i=0;i<n;++i)for(int j=0;j<n;++j)nonfinite|=!std::isfinite(b[i*8+j]);
    require(nonfinite && a==saved,"zero-pivot/singular fallback contract or input restoration changed");
  }
  a.fill(7);b.fill(-91);openmx::test::fail_allocation_after(0);
  const int clear=openmx_core_solver_inverse_unpivoted(owner,0,-1,&input,&output,error,sizeof(error));openmx::test::fail_allocation_after(-1);
  require(clear,"inverse sentinel allocated");for(auto value:a)equal(value,0);for(auto value:b)equal(value,-91);
  const OpenMXSolverRealTensor empty={{0,0,1,1},nullptr};
  require(openmx_core_solver_inverse_unpivoted(owner,0,-1,&empty,&output,error,sizeof(error)),error);
  // Ill-conditioned Gram/constraint input captured from independently
  // compiled native C references (NVHPC and ROCm Clang agree bit for bit).
  // Separate multiply/subtract changes this inverse and its SCF trajectory.
  const double legacy_input[]={0x1.bc6b979a6a6d2p-5,0x1.bc6b8e1c0045bp-5,-0x1.0000000000000p+0,0x1.bc6b8e1c0045bp-5,0x1.bc6b849d98536p-5,-0x1.0000000000000p+0,-0x1.0000000000000p+0,-0x1.0000000000000p+0,0x0.0p+0};
  const double legacy_inverse[]={0x1.cfd3a046ee85dp+43,-0x1.cfd3a046ee85dp+43,0x1.13377208dc849p+18,-0x1.cfd3a046ee85dp+43,0x1.cfd3a046ee85dp+43,-0x1.1337b208dc849p+18,0x1.13377208232dap+18,-0x1.1337b208232d9p+18,-0x1.9398246cb62a2p-5};
  seed(3);for(int i=0;i<3;++i)for(int j=0;j<3;++j)a[i*10+j]=legacy_input[i*3+j];
  original=a;
  require(openmx_core_solver_inverse_unpivoted(owner,0,2,&input,&output,error,sizeof(error)),error);
  require(a==original,"native-rounding regression changed input");
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)
    require(!std::memcmp(&b[i*8+j],&legacy_inverse[i*3+j],sizeof(double)),"unpivoted inverse changed the native FMA rounding contract");
  // The C++ module uses the same independent identity with unequal strides.
  seed(3);original=a;openmx::core::UnpivotedInverse cpp(3);
  cpp.apply(openmx::core::UnpivotedInverse::matrix_view{a.data(),8,10},
      openmx::core::UnpivotedInverse::matrix_view{b.data(),10,8},3);identity(3,original);
  *result=17;return 1;
}
int rows(OpenMXSolverInvocation* owner,void*,double* result) {
  char error[256];double** matrix;dcomplex** complex;
  const size_t dimensions[4]={4,3,1,1};OpenMXSolverRealTensor original;
  require(openmx_core_solver_real(owner,0,"matrix",dimensions,&original,error,sizeof(error)),error);
  original.values[11]=31;
  require(openmx_core_solver_real_rows(owner,0,"matrix",4,3,&matrix,error,sizeof(error)),error);
  require(matrix[0]==original.values && matrix[3]==matrix[0]+9,"matrix rows copied or changed values");equal(matrix[3][2],31);
  require(openmx_core_solver_complex_rows(owner,0,"z",2,5,&complex,error,sizeof(error)),error);
  complex[1][4]={7,-3};require(complex[1]==complex[0]+5,"complex matrix stride");equal(complex[1][4].i,-3);
  require(openmx_core_solver_real_rows(owner,0,"empty",3,0,&matrix,error,sizeof(error)),error);
  require(matrix[0]==matrix[2],"zero-column rows are inconsistent");
  double first[24]{},second[24]{};
  const OpenMXSolverRealTensor planes={{2,3,4,1},first};double*** aliases;
  require(openmx_core_solver_real_alias(owner,0,"planes",&planes,&aliases,error,sizeof(error)),error);
  aliases[1][2][3]=19;
  require(first[23]==19 && aliases[1][0]==first+12,"borrowed real planes have incorrect strides");
  auto saved=aliases;auto rebound=planes;rebound.values=second;
  openmx::test::fail_allocation_after(0);
  const int ok=openmx_core_solver_real_alias(owner,0,"planes",&rebound,&aliases,error,sizeof(error));
  openmx::test::fail_allocation_after(-1);
  require(ok && saved==aliases && aliases[1][2]==second+20,"rebinding row metadata allocated or retained old values");
  const OpenMXSolverRealTensor invalid_shape={{SIZE_MAX,2,4,1},second};
  require(!openmx_core_solver_real_alias(owner,0,"planes",&invalid_shape,&aliases,error,sizeof(error)),"alias overflow accepted");
  require(aliases==saved && aliases[1][2]==second+20,"failed alias replacement changed existing rows");
  const OpenMXSolverComplexTensor zplanes={{1,2,5,1},complex[0]};dcomplex*** zaliases;
  require(openmx_core_solver_complex_alias(owner,0,"zplanes",&zplanes,&zaliases,error,sizeof(error)),error);
  equal(zaliases[0][1][4].i,-3);
  const size_t scratch_shape[4]={3,2,5,1};OpenMXSolverComplexTensor scratch;
  require(openmx_core_solver_complex(owner,0,"nonlocal",scratch_shape,&scratch,error,sizeof(error)),error);
  for(size_t c=0;c<3;++c)for(size_t i=0;i<2;++i)for(size_t j=0;j<5;++j)
    openmx_solver_complex_plane_row(&scratch,c,i)[j]={double(100*c+10*i+j),-double(c+i+j)};
  for(size_t k=0;k<30;++k)
    equal(scratch.values[k].r,100*(k/10)+10*((k/5)%2)+k%5);
  const auto previous=scratch;
  openmx::test::fail_allocation_after(0);
  const int reused=openmx_core_solver_complex(owner,0,"nonlocal",scratch_shape,&scratch,error,sizeof(error));
  openmx::test::fail_allocation_after(-1);
  require(reused && previous.values==scratch.values,"nonlocal scratch did not reuse its numeric owner");
  equal(openmx_solver_complex_plane_row(&scratch,2,1)[4].r,214);
  *result=17;return 1;
}
int orbitals(OpenMXSolverInvocation* owner,void*,double* result) {
  char error[256];const OpenMXOrbitalShape shape{4,3,1};OpenMXOrbitalBuffers view{};
  const char* key="force-orbital-sampler-with-allocation-free-lookup";
  require(openmx_core_solver_orbitals(owner,0,key,&shape,&view,error,sizeof(error)),error);
  require(view.angular_channels==5 && view.multiplicity==3 && view.magnetic_capacity==11,"orbital cache dimensions");
  require(view.dRF==view.RF+15 && view.dAFQ==view.AF+55 && view.dAFP==view.AF+110,"orbital cache channel strides");
  for(int i=0;i<15;i++){equal(view.RF[i],0);equal(view.dRF[i],0);view.RF[i]=i+.25;}
  for(int i=0;i<55;i++){equal(view.AF[i],0);equal(view.dAFQ[i],0);equal(view.dAFP[i],0);view.dAFP[i]=i-.5;}
  const auto saved=view;
  openmx::test::fail_allocation_after(0);
  const int reused=openmx_core_solver_orbitals(owner,0,key,&shape,&view,error,sizeof(error));
  openmx::test::fail_allocation_after(-1);
  require(reused && view.RF==saved.RF && view.AF==saved.AF,"equal orbital shape allocated or changed buffers");
  equal(view.RF[14],14.25);equal(view.dAFP[54],53.5);
  const OpenMXOrbitalShape larger{5,4,1};
  for(long failure=0;failure<3;failure++) {
    openmx::test::fail_allocation_after(failure);
    const int ok=openmx_core_solver_orbitals(owner,0,key,&larger,&view,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    require(!ok && error[0] && view.RF==saved.RF,"failed orbital replacement changed output");
    equal(saved.RF[14],14.25);equal(saved.dAFP[54],53.5);
  }
  require(openmx_core_solver_orbitals(owner,1,key,&shape,&view,error,sizeof(error)),error);
  require(view.RF!=saved.RF && view.AF!=saved.AF,"workers alias orbital buffers");
  equal(view.RF[14],0);equal(view.dAFP[54],0);
  const OpenMXOrbitalShape value_only{2,2,0};
  require(openmx_core_solver_orbitals(owner,0,key,&value_only,&view,error,sizeof(error)),error);
  require(view.angular_channels==3 && view.multiplicity==2 && view.magnetic_capacity==7 &&
      !view.dRF && !view.dAFQ && !view.dAFP,"value-only orbital replacement retained derivative channels");
  equal(view.RF[5],0);equal(view.AF[20],0);
  const auto current=view;
  for(const auto bad:std::array<OpenMXOrbitalShape,4>{{{-1,2,1},{15,2,1},{2,0,1},{2,2,2}}}) {
    require(!openmx_core_solver_orbitals(owner,0,key,&bad,&view,error,sizeof(error)),"invalid orbital shape accepted");
    require(view.RF==current.RF && view.angular_channels==current.angular_channels,"invalid shape changed orbital output");
  }
  require(!openmx_core_solver_orbitals(owner,-1,key,&shape,&view,error,sizeof(error)),"invalid orbital worker accepted");
  require(!openmx_core_solver_orbitals(owner,0,"",&shape,&view,error,sizeof(error)),"empty orbital key accepted");
  require(!openmx_core_solver_orbitals(nullptr,0,key,&shape,&view,error,sizeof(error)),"missing orbital scope accepted");
  require(!openmx_core_solver_orbitals(owner,0,key,nullptr,&view,error,sizeof(error)),"missing orbital shape accepted");
  require(!openmx_core_solver_orbitals(owner,0,key,&shape,nullptr,error,sizeof(error)),"missing orbital output accepted");
  *result=17;return 1;
}
int fields(OpenMXSolverInvocation* owner,void*,double* result) {
  char error[256];OpenMXSolverRealFieldView view;
  require(openmx_core_solver_field_create(owner,2,3,&view,error,sizeof(error)),error);
  const size_t lengths[3]={1,4,7};
  for(size_t s=0;s<2;s++)for(size_t a=0;a<3;a++) {
    require(openmx_core_solver_field_resize(view.owner,s,a,lengths[a],error,sizeof(error)),error);
    for(size_t k=0;k<lengths[a];k++)openmx_solver_field_row(&view,s,a)[k]=100*s+10*a+k;
  }
  for(size_t s=0;s<2;s++)for(size_t a=0;a<3;a++)for(size_t k=0;k<lengths[a];k++)
    equal(openmx_solver_field_row(&view,s,a)[k],100*s+10*a+k);
  const auto saved=view.blocks[5];
  openmx::test::fail_allocation_after(0);
  int ok=openmx_core_solver_field_resize(view.owner,1,2,7,error,sizeof(error));
  openmx::test::fail_allocation_after(-1);
  require(ok && view.blocks[5].values==saved.values,"same spectral row allocated");
  equal(openmx_solver_field_row(&view,1,2)[6],126);
  require(!openmx_core_solver_field_resize(view.owner,1,2,SIZE_MAX,error,sizeof(error)),"field overflow accepted");
  require(view.blocks[5].values==saved.values && view.blocks[5].count==7,"failed field resize changed view");
  *result=17;return 1;
}
OpenMXSolverResidueView residue(OpenMXSolverInvocation* owner,int complex_values,int padding) {
  int map[3]={-1,2,1},species[3]={-1,0,1},orbitals[2]={2,3},neighbors[3]={-1,1,1};
  int first[2]={1,2},second[2]={2,1},*lists[3]={nullptr,first,second};
  const OpenMXSolverResidueInput input={2,2,2,complex_values?3:2,padding,complex_values,map,species,orbitals,neighbors,lists};
  OpenMXSolverResidueView view;char error[256];
  require(openmx_core_solver_residue_create(owner,&input,&view,error,sizeof(error)),error);
  map[1]=-1;species[1]=-1;orbitals[0]=-1;neighbors[1]=-1;first[0]=-1;
  return view; // Borrowed input dies here; the residue layout must own its snapshot.
}
int residues(OpenMXSolverInvocation* owner,void*,double* result) {
  char error[256];auto real=residue(owner,0,1),complex=residue(owner,1,0);
  require(real.atoms==3 && real.atom_offsets[3]==6 && real.pair_offsets[6]==27,"legacy dummy/mixed-orbital residue layout");
  require(complex.atoms==2 && complex.atom_offsets[2]==4 && complex.pair_offsets[4]==25,"noncollinear residue layout");
  size_t real_count=0,complex_count=0;
  for(size_t s=0;s<2;s++)for(size_t a=0;a<3;a++) {
    const size_t states=a==0?1:a==1?4:3;
    require(openmx_core_solver_residue_resize(real.owner,s,a,states,error,sizeof(error)),error);
    const auto* begin=real.blocks[s*3+a].values;
    const int r=a==0?1:a==1?3:2;
    size_t offset=0;
    for(int h=0;h<2;h++) {
      const int columns=a==0?1:h==0?r:r==3?2:3;
      for(int i=0;i<r;i++)for(int j=0;j<columns;j++) {
        auto* row=openmx_solver_residue_row(&real,s,a,h,i,j);
        require(row==begin+offset,"residue has a gap or overlaps rows");
        for(size_t k=0;k<states;k++){row[k]=s*1000+a*100+h*30+i*10+j+.1*k;real_count++;}
        offset+=states;
      }
    }
  }
  require(real_count==184,"real residue element count differs from frozen dimensions");
  for(size_t s=0;s<3;s++)for(size_t a=0;a<2;a++) {
    const size_t states=a==0?2:5;
    require(openmx_core_solver_residue_resize(complex.owner,s,a,states,error,sizeof(error)),error);
    const auto* begin=complex.blocks[s*2+a].complex_values;const int r=a==0?3:2;size_t offset=0;
    for(int h=0;h<2;h++)for(int i=0;i<r;i++)for(int j=0;j<(h==0?r:5-r);j++) {
      auto* row=openmx_solver_complex_residue_row(&complex,s,a,h,i,j);
      require(row==begin+offset,"complex residue stride");
      for(size_t k=0;k<states;k++){row[k]={double(100*s+10*i+j),-double(k)};complex_count++;}
      offset+=states;
    }
  }
  require(complex_count==240,"complex residue element count differs from frozen dimensions");
  equal(openmx_solver_residue_row(&real,1,1,1,2,1)[3],1151.3);
  const auto saved=real.blocks[4];
  require(!openmx_core_solver_residue_resize(real.owner,1,1,SIZE_MAX,error,sizeof(error)),"residue overflow accepted");
  require(real.blocks[4].values==saved.values && real.blocks[4].states==saved.states,"failed residue resize changed owner");
  require(openmx_core_solver_residue_resize(complex.owner,1,1,0,error,sizeof(error)),error);
  require(!openmx_solver_complex_residue_row(&complex,1,1,1,0,0),"zero-state block has data");
  *result=17;return 1;
}
int invalid(OpenMXSolverInvocation* owner,void*,double* result) {
  char error[256];const size_t shape[4]={SIZE_MAX,2,1,1};OpenMXSolverRealTensor view{};view.values=reinterpret_cast<double*>(17);auto saved=view;
  require(!openmx_core_solver_real(owner,0,"bad",shape,&view,error,sizeof(error)),"overflow tensor accepted");
  require(!std::memcmp(&saved,&view,sizeof(view)),"failure changed output");
  const size_t valid[4]={2,3,1,1};
  require(!openmx_core_solver_real(owner,4,"bad",valid,&view,error,sizeof(error)),"invalid worker accepted");
  require(!openmx_core_solver_real(owner,0,nullptr,valid,&view,error,sizeof(error)),"null name accepted");
  void* pointer=reinterpret_cast<void*>(19);
  require(!openmx_core_solver_bytes(owner,0,"overflow",SIZE_MAX,2,&pointer,error,sizeof(error)),"byte overflow accepted");
  require(pointer==reinterpret_cast<void*>(19),"failure changed opaque output");
  auto r=residue(owner,0,1);
  require(!openmx_core_solver_residue_resize(r.owner,2,0,1,error,sizeof(error)),"invalid residue component accepted");
  OpenMXSolverResidueInput input{};OpenMXSolverResidueView output=r;
  require(!openmx_core_solver_residue_create(owner,&input,&output,error,sizeof(error)),"invalid residue topology accepted");
  require(!std::memcmp(&r,&output,sizeof(output)),"invalid topology changed view");
  *result=17;return 1;
}
int lifetime_callback(OpenMXSolverInvocation* owner,void* mode,double* result) {
  double ignored;buffers(owner,nullptr,&ignored);overwrite(owner,nullptr,&ignored);rows(owner,nullptr,&ignored);orbitals(owner,nullptr,&ignored);
  complex_overwrite(owner,nullptr,&ignored);
  matrix_scratch(owner,nullptr,&ignored);
  unpivoted_inverse(owner,nullptr,&ignored);
  fields(owner,nullptr,&ignored);residues(owner,nullptr,&ignored);
  const auto behavior=*static_cast<int*>(mode);
  if(behavior==1)return 0;
  if(behavior==2)throw std::runtime_error("callback exception");
  *result=17;return 1;
}
void lifetime_test() {
  for(int mode=0;mode<3;mode++) {
    const auto before=openmx::test::live_allocations();char error[256];double result=29;
    const int ok=openmx_core_solver_run(4,lifetime_callback,&mode,&result,error,sizeof(error));
    require(ok==(mode==0) && openmx::test::live_allocations()==before,"solver scope leaked on callback return/exception");
    equal(result,mode==0?17:29);
  }
}
int allocating(OpenMXSolverInvocation* owner,void*,double* result) {
  char error[256];const size_t shape[4]={3,4,2,1};OpenMXSolverRealTensor view;
  if(!openmx_core_solver_real(owner,0,"a",shape,&view,error,sizeof(error)))return 0;
  if(!openmx_core_solver_real_overwrite(owner,0,"a",shape,&view,error,sizeof(error)))return 0;
  OpenMXSolverComplexTensor complex;
  if(!openmx_core_solver_complex_overwrite(owner,0,"a",shape,&complex,error,sizeof(error)))return 0;
  MatrixInput matrix_input;auto matrix_source=matrix_input.view();OpenMXElectronicMatrixView matrix{};
  const auto unchanged_matrix=matrix;
  if(!openmx_core_solver_matrix_scratch(owner,&matrix_source,1,&matrix,error,sizeof(error))){
    require(!std::memcmp(&matrix,&unchanged_matrix,sizeof(matrix)),"failed matrix construction changed output");return 0;
  }
  double inverse_input[]={4,1,2,3},inverse_output[4]={};
  const OpenMXSolverRealTensor inverse_a={{2,2,1,1},inverse_input},inverse_b={{2,2,1,1},inverse_output};
  if(!openmx_core_solver_inverse_unpivoted(owner,0,1,&inverse_a,&inverse_b,error,sizeof(error))){
    require(inverse_input[0]==4 && inverse_input[1]==1 && inverse_input[2]==2 && inverse_input[3]==3 &&
        inverse_output[0]==0 && inverse_output[1]==0 && inverse_output[2]==0 && inverse_output[3]==0,
        "partial inverse construction changed arrays");return 0;
  }
  double** rows;if(!openmx_core_solver_real_rows(owner,0,"b",5,8,&rows,error,sizeof(error)))return 0;
  double*** aliases;const OpenMXSolverRealTensor borrowed={{3,4,2,1},view.values};
  if(!openmx_core_solver_real_alias(owner,0,"alias",&borrowed,&aliases,error,sizeof(error)))return 0;
  OpenMXSolverIntTensor integers;
  if(!openmx_core_solver_int_tensor(owner,0,"integers",shape,&integers,error,sizeof(error)))return 0;
  OpenMXSolverRealFieldView field;if(!openmx_core_solver_field_create(owner,2,2,&field,error,sizeof(error)))return 0;
  if(!openmx_core_solver_field_resize(field.owner,0,0,8,error,sizeof(error)))return 0;
  auto r=residue(owner,0,1);
  if(!openmx_core_solver_residue_resize(r.owner,1,1,8,error,sizeof(error)))return 0;
  OpenMXOrbitalBuffers sampler;const OpenMXOrbitalShape orbital_shape{4,3,1};
  if(!openmx_core_solver_orbitals(owner,0,"sampler",&orbital_shape,&sampler,error,sizeof(error)))return 0;
  const OpenMXOrbitalShape larger{6,4,0};
  if(!openmx_core_solver_orbitals(owner,0,"sampler",&larger,&sampler,error,sizeof(error)))return 0;
  *result=17;return 1;
}
void allocation_test() {
  char error[256];double result=29;
  openmx::test::fail_allocation_after(10000);
  int ok=openmx_core_solver_run(4,allocating,nullptr,&result,error,sizeof(error));
  auto allocations=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  require(ok && allocations>15,"allocation interception failed");
  for(long failure=0;failure<allocations;failure++) {
    const auto before=openmx::test::live_allocations();result=29;
    openmx::test::fail_allocation_after(failure);ok=openmx_core_solver_run(4,allocating,nullptr,&result,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    require(!ok && error[0] && openmx::test::live_allocations()==before,"partial solver construction leaked");equal(result,29);
  }
}
int concurrent(OpenMXSolverInvocation* owner,void*,double* result) {
  OpenMXSolverRealFieldView fields;char error[256];require(openmx_core_solver_field_create(owner,1,4,&fields,error,sizeof(error)),error);
  std::array<int,4> status{};std::vector<std::thread> threads;
  for(int worker=0;worker<4;worker++)threads.emplace_back([&,worker]{try {
    const size_t shape[4]={3,3,1,1};OpenMXSolverRealTensor view;
    const OpenMXOrbitalShape orbital_shape{3,2,1};OpenMXOrbitalBuffers sampler;
    for(int n=0;n<10;n++) {
      require(openmx_core_solver_real(owner,worker,"worker",shape,&view,nullptr,0),"concurrent buffer request");
      for(int i=0;i<9;i++)view.values[i]=worker*100+n*10+i;
      require(openmx_core_solver_real_overwrite(owner,worker,"worker",shape,&view,nullptr,0),"concurrent overwrite request");
      for(int i=0;i<9;i++)view.values[i]=worker*100+n*10+i;
      OpenMXSolverComplexTensor z;
      require(openmx_core_solver_complex_overwrite(owner,worker,"worker",shape,&z,nullptr,0),"concurrent complex overwrite request");
      for(int i=0;i<9;i++)z.values[i]={double(worker*100+n*10+i),double(-worker)};
      equal(z.values[8].r,worker*100+n*10+8);equal(z.values[8].i,-worker);
      double inverse_input[]={4.+worker,1,2,3},inverse_output[4];
      const OpenMXSolverRealTensor inverse_a={{2,2,1,1},inverse_input},inverse_b={{2,2,1,1},inverse_output};
      require(openmx_core_solver_inverse_unpivoted(owner,worker,1,&inverse_a,&inverse_b,nullptr,0),"concurrent inverse request");
      equal((4.+worker)*inverse_output[0]+inverse_output[2],1);
      equal(inverse_input[0],4.+worker);equal(inverse_input[3],3);
      require(openmx_core_solver_orbitals(owner,worker,"sampler",&orbital_shape,&sampler,nullptr,0),"concurrent orbital request");
      for(int i=0;i<8;i++)sampler.dRF[i]=worker*100+n*10+i;
      require(openmx_core_solver_field_resize(fields.owner,0,worker,n+1,nullptr,0),"concurrent field resize");
    }
    equal(view.values[8],worker*100+98);equal(sampler.dRF[7],worker*100+97);status[worker]=1;
  }catch(...){status[worker]=-1;}});
  for(auto& t:threads)t.join();for(auto s:status)require(s==1,"worker state raced or aliased");
  *result=17;return 1;
}
}
int main(int argc,char** argv) {
  try {
    const std::string name=argc==2?argv[1]:"";
    if(name=="buffers")run(buffers);else if(name=="overwrite")run(overwrite);else if(name=="rows")run(rows);else if(name=="fields")run(fields);
    else if(name=="complex-overwrite")run(complex_overwrite);
    else if(name=="matrix-scratch")run(matrix_scratch);
    else if(name=="unpivoted-inverse")run(unpivoted_inverse);
    else if(name=="orbitals")run(orbitals);
    else if(name=="residues")run(residues);else if(name=="invalid")run(invalid);else if(name=="lifetime")lifetime_test();
    else if(name=="allocation")allocation_test();else if(name=="concurrent")run(concurrent);else throw std::runtime_error("unknown solver workspace test");
    std::cout<<"PASS solver workspace "<<name<<'\n';return 0;
  }catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}
