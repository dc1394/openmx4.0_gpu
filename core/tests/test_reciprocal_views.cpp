#include "reciprocal_fixture.hpp"
#include <openmx/core/radial.hpp>
#include "allocation_faults.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

namespace {
using namespace openmx::test::reciprocal;
void check4(OpenMXRadialView4 flat,double ****rows,std::size_t a,
    std::size_t b,std::size_t c,std::size_t d) {
  require(flat.extent[0]==a && flat.extent[1]==b && flat.extent[2]==c && flat.extent[3]==d,
      "reciprocal extent or legacy padding changed");
  std::size_t index=0;
  for (std::size_t i=0;i<a;++i) for (std::size_t j=0;j<b;++j)
    for (std::size_t k=0;k<c;++k) for (std::size_t l=0;l<d;++l,++index) {
      auto *value=openmx_radial_row4(&flat,i,j,k)+l;
      require(value==flat.values+index && value==&rows[i][j][k][l],"flat/legacy row mismatch");
      require(*value==0,"reciprocal values or padding not initialized");
      *value=static_cast<double>(index+1);
      require(rows[i][j][k][l]==index+1,"reciprocal values copied rather than borrowed");
    }
}
void layout_test() {
  auto owner=context();Fixture f;f.allocate_all(owner.get());auto v=get(owner.get());
  check4(v.pao_transform,f.pao.transforms,3,3,2,7);
  check4(v.nonlocal_transform,f.vps.transforms,2,3,6,7);
  check4(v.neutral_projectors,f.vps.neutral_projectors,3,3,2,5);
  check4(v.neutral_transform,f.vps.neutral_transforms,3,3,2,11);
  require(v.neutral_energies.extent[0]==3 && v.neutral_energies.extent[1]==3 &&
      v.neutral_energies.extent[2]==2,"neutral energy dimensions");
  for (int s=0;s<3;++s) for (int l=0;l<3;++l) for (int m=0;m<2;++m) {
    f.vps.neutral_energies[s][l][m]=100*s+10*l+m;
    require(openmx_radial_value3(&v.neutral_energies,s,l,m)==100*s+10*l+m,"flat neutral energy lookup");
  }
  require(v.potential_transform.extent[0]==3 && v.potential_transform.extent[1]==11,"crude potential padding");
  for (int s=0;s<3;++s) for (int k=0;k<11;++k) {
    require(openmx_radial_row2(&v.potential_transform,s)+k==&f.vps.potential_transform[s][k],"potential row mismatch");
    require(f.vps.potential_transform[s][k]==0,"potential padding not initialized");
  }
  require(v.pao.orbitals==f.pao.orbitals && v.vps.projectors==f.vps.projectors &&
      v.products.rows==f.products.rows,"reader adapters changed");
}
void products_test() {
  auto owner=context();Fixture f;require(f.allocate(owner.get(),2),f.error);
  const auto v=get(owner.get()).product_transform;
  require(v.species==3 && v.stride==11,"product dimensions/padding");
  std::size_t row=0;
  for (int s=0;s<3;++s) {
    require(v.layout[s].first_row==row,"product species offset");
    for (int l1=0;l1<=f.lmax[s];++l1) for (int m1=0;m1<f.counts[s][l1];++m1)
      for (int l2=l1;l2<=f.lmax[s];++l2) for (int m2=0;m2<f.counts[s][l2];++m2)
        for (int l=0;l<=2*l2;++l,++row) for (int k=0;k<11;++k) {
          double *value=openmx_product_radial_row(&v,s,l1,m1,l2,m2,l)+k;
          require(value==v.values+row*11+k && value==&f.products.rows[s][l1][m1][l2][m2][l][k],
              "ragged product row order differs from restart/native packing");
          require(*value==0,"product padding not initialized");*value=row*11+k+1;
        }
    for (int l=0;l<=f.lmax[s];++l)
      if (!f.counts[s][l]) require(!f.products.rows[s][l],"empty product group allocated");
  }
  require(row==v.rows,"product terminal row offset");
  // Input metadata may disappear/change after allocation. Flat indices are owned.
  f.n0[0]=0;f.n1[1]=0;
  require(openmx_product_radial_row(&v,0,0,1,2,0,4)[3]>0,"product view retained input metadata");
  require(openmx_product_radial_row(&v,1,1,1,1,0,2)[10]>0,"second product species layout lost");
}
void lifecycle_test() {
  auto first=context(),second=context();Fixture f;f.allocate_all(first.get());
  require(!get(second.get()).pao_transform.values,"independent context inherited table storage");
  auto v=get(first.get());v.pao_transform.values[0]=91;v.neutral_energies.values[0]=13;
  const auto live=openmx::test::live_allocations();
  openmx::test::fail_allocation_after(0);
  auto same=get(first.get());
  const auto remaining=openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  require(remaining==0 && openmx::test::live_allocations()==live,"view getter allocates or caches state");
  require(same.pao_transform.values==v.pao_transform.values,"getter changed table address");
  openmx_core_product_release(first.get());same=get(first.get());
  require(!same.product_transform.values && same.pao_transform.values[0]==91 &&
      same.neutral_energies.values[0]==13,"product release affected other owners");
  openmx_core_vps_release(first.get());same=get(first.get());
  require(!same.nonlocal_transform.values && !same.neutral_energies.values &&
      same.pao_transform.values[0]==91,"VPS release affected PAO");
  openmx_core_pao_release(first.get());
  for (int round=0;round<8;++round) {
    f.pao_shape.radial_points=5+round;f.neutral_shape.wavevectors=3+round;
    f.allocate_all(first.get());same=get(first.get());
    require(same.pao_transform.values[0]==0 && same.neutral_transform.extent[3]==std::size_t(5+round),
        "reallocation retained stale views or values");
    openmx_core_product_release(first.get());openmx_core_vps_release(first.get());openmx_core_pao_release(first.get());
  }
  f.allocate_all(second.get());require(get(second.get()).pao_transform.values[0]==0,"independent table data not zero");
  openmx_core_finish(first.get());openmx_core_finish(first.get());
}
void host_transform() {
  auto owner=context();Fixture f;require(f.allocate(owner.get(),0),f.error);
  double radii[5]{0,.25,.5,.75,1};const int counts[1]{1};
  for (int r=0;r<5;++r) f.pao.orbitals[0][0][0][r]=2;
  const OpenMXPaoTable input{0,counts,5,radii,f.pao.orbitals[0],f.pao.transforms[0]};
  const OpenMXUniformRadialGrid grid{4096,3,0,1./4096,.25};
  require(openmx_core_transform_pao(owner.get(),0,&input,&grid,f.error,sizeof(f.error)),f.error);
  const auto v=get(owner.get());
  require(std::abs(v.pao_transform.values[0]-2./3)<3e-8,"host-owned radial quadrature");
  openmx_core_finish(owner.get());
}
void runtime_test(bool finalize) {
  if (finalize) {
    Kokkos::initialize();
    for (int round=0;round<5;++round) {
      host_transform();
      require(Kokkos::is_initialized() && !Kokkos::is_finalized(),
          "host-only calculation finalized an externally owned runtime");
    }
    Kokkos::finalize();
  }
  const bool initialized=Kokkos::is_initialized(),finished=Kokkos::is_finalized();
  for (int round=0;round<5;++round) {
    host_transform();
    require(Kokkos::is_initialized()==initialized && Kokkos::is_finalized()==finished,
        "host-only calculation changed borrowed/global runtime lifetime");
  }
}
void invalid_test() {
  auto owner=context();Fixture f;require(f.allocate(owner.get(),1),f.error);
  const auto v=get(owner.get());require(!v.pao_transform.values && !v.product_transform.values,"absent owner not empty");
  auto output=v;
  require(!openmx_core_reciprocal_get(nullptr,&output,f.error,sizeof(f.error)) && f.error[0],"null context accepted");
  require(!std::memcmp(&v,&output,sizeof(v)),"failed getter changed output");
  require(!openmx_core_reciprocal_get(owner.get(),nullptr,nullptr,0),"null output accepted");
  char small[2]{'x','y'};
  require(!openmx_core_reciprocal_get(nullptr,&output,small,1) && !small[0] && small[1]=='y',"error buffer overflow");
  openmx_core_finish(owner.get());
  require(!openmx_core_reciprocal_get(owner.get(),&output,nullptr,0) &&
      !std::memcmp(&v,&output,sizeof(v)),"finished getter published stale state");
  auto plain=context();OpenMXVpsViews no_neutral{};
  require(openmx_core_vps_allocate(plain.get(),&f.vps_shape,nullptr,&no_neutral,f.error,sizeof(f.error)),f.error);
  const auto disabled=get(plain.get());
  require(disabled.nonlocal_transform.values && !disabled.neutral_transform.values &&
      !disabled.neutral_projectors.values && !disabled.neutral_energies.values &&
      !disabled.potential_transform.values,"VNA-off views not empty");
}
void allocation_test() {
  for (int kind=0;kind<3;++kind) {
    long count=0;
    {
      auto owner=context();Fixture f;
      openmx::test::fail_allocation_after(10000);const int ok=f.allocate(owner.get(),kind);
      count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
      require(ok && count>1,"allocation interception unavailable");
    }
    for (long failure=0;failure<count;++failure) {
      auto owner=context();Fixture f;
      require(f.allocate(owner.get(),(kind+1)%3),f.error);const auto saved=get(owner.get());
      const auto live=openmx::test::live_allocations();
      openmx::test::fail_allocation_after(failure);const int ok=f.allocate(owner.get(),kind);
      openmx::test::fail_allocation_after(-1);
      require(!ok && f.error[0] && openmx::test::live_allocations()==live,"failed allocation leaked table state");
      const auto current=get(owner.get());
      require(current.pao_transform.values==saved.pao_transform.values &&
          current.nonlocal_transform.values==saved.nonlocal_transform.values &&
          current.product_transform.values==saved.product_transform.values,"failed allocation changed independent table owners");
    }
  }
  char error[64];openmx::test::fail_allocation_after(0);
  auto *bad=openmx_core_create_host(error,sizeof(error));openmx::test::fail_allocation_after(-1);
  require(!bad && error[0],"host context allocation failure not reported");
}
void concurrent_test() {
  int result[4]{};std::thread jobs[4];
  for (int thread=0;thread<4;++thread) jobs[thread]=std::thread([&,thread] {
    try {
      for (int round=0;round<8;++round) {
        auto owner=context();Fixture f;f.allocate_all(owner.get());auto v=get(owner.get());
        v.pao_transform.values[0]=thread+round;v.product_transform.values[0]=thread-round;
        require(get(owner.get()).pao_transform.values[0]==thread+round &&
            get(owner.get()).product_transform.values[0]==thread-round,"concurrent contexts share tables");
      }
      result[thread]=1;
    } catch (...) { result[thread]=-1; }
  });
  for (auto& job:jobs) job.join();for (int value:result) require(value==1,"concurrent host table ownership failed");
}
}
int main(int argc,char **argv) {
  try {
    const std::string name=argc==2?argv[1]:"";
    if (name=="layout") layout_test();else if (name=="products") products_test();
    else if (name=="lifecycle") lifecycle_test();else if (name=="host-runtime") runtime_test(false);
    else if (name=="finalized-runtime") runtime_test(true);else if (name=="invalid") invalid_test();
    else if (name=="allocation") allocation_test();else if (name=="concurrent") concurrent_test();
    else throw std::runtime_error("unknown reciprocal view test");
    std::cout<<"PASS reciprocal views "<<name<<'\n';return 0;
  } catch (const std::exception& e) {
    openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;
  }
}
