#include "openmx_core_vna_host_tables.h"
#include <openmx/core/vna_host_tables.hpp>
#include "allocation_faults.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {
using Prep=std::unique_ptr<OpenMXVnaPreparation,decltype(&openmx_core_vna_preparation_destroy)>;
using Projectors=std::unique_ptr<OpenMXVnaHostProjectors,decltype(&openmx_core_vna_host_projectors_destroy)>;
using Products=std::unique_ptr<OpenMXVnaHostProducts,decltype(&openmx_core_vna_host_products_destroy)>;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
bool bits(double a,double b){return std::memcmp(&a,&b,sizeof(a))==0;}
constexpr int species=3,points=5,source_points=7,width=7;
constexpr std::size_t source_orbitals=8,species_rows=8*8*9+3,source_rows=species*species_rows+2;
double value(std::size_t row,int point){return row%17==0 && point==0?-0.0:100.0*row+point+0.25;}
Prep prepare(bool empty=false,int projectors=2)
{
  int angular[]{empty?-1:2,-1,empty?-1:4};int first[]{1,0,2},third[]{0,1,1,0,1};
  int* counts[]{first,nullptr,third};OpenMXVnaPreparationBuffers out{};char error[160];
  const OpenMXVnaPreparationShape shape{4,2,2,projectors,points,species,angular,counts};
  Prep result(openmx_core_vna_preparation_create(&shape,&out,error,sizeof(error)),openmx_core_vna_preparation_destroy);
  require(bool(result),error);return result;
}
struct Inputs {
  std::vector<double> projectors=std::vector<double>(species*4*3*source_points);
  std::vector<dcomplex> conversion=std::vector<dcomplex>(3*width*width);
  std::vector<double> products=std::vector<double>(source_rows*source_points);
  std::array<std::size_t,6> offsets{{1,3,4,5,6,8}};
  std::array<std::size_t,64> pairs{};
  std::array<OpenMXProductSpeciesView,species> layout{};
  Inputs() {
    for(int s=0;s<species;++s)for(int l=0;l<4;++l)for(int r=0;r<3;++r)for(int p=0;p<source_points;++p)
      projectors[((s*4+l)*3+r)*source_points+p]=1000*s+100*l+10*r+p+0.125;
    for(int l=0;l<3;++l)for(int r=0;r<width;++r)for(int c=0;c<width;++c) {
      const double x=100*l+10*r+c+0.375;
      conversion[(l*width+r)*width+c]={x,-x};
    }
    for(std::size_t row=0;row<source_rows;++row)for(int p=0;p<source_points;++p)
      products[row*source_points+p]=p<points?value(row,p):std::numeric_limits<double>::quiet_NaN();
    for(std::size_t i=0;i<pairs.size();++i)pairs[i]=(13*i%64)*9;
    for(int s=0;s<species;++s)layout[s]={source_orbitals,std::size_t(s)*species_rows+1,offsets.data(),pairs.data()};
  }
  OpenMXVnaHostProjectorInput projector_input(int convert=1) {
    return {{projectors.data(),{species,4,3,source_points}},conversion.data(),width,convert};
  }
  OpenMXProductRadialView product_input(){return {products.data(),species,source_rows,source_points,layout.data()};}
  void discard() {
    std::vector<double>().swap(projectors);std::vector<double>().swap(products);
    std::vector<dcomplex>().swap(conversion);offsets.fill(SIZE_MAX);pairs.fill(SIZE_MAX);layout.fill({});
  }
};
Projectors pack(const Prep& prep,Inputs& input,OpenMXVnaHostProjectorData& out,int convert=1)
{
  auto v=input.projector_input(convert);char error[160];
  Projectors result(openmx_core_vna_host_projectors_create(prep.get(),&v,&out,error,sizeof(error)),openmx_core_vna_host_projectors_destroy);
  require(bool(result),error);return result;
}
Products pack(const Prep& prep,Inputs& input,OpenMXVnaHostProductData& out)
{
  auto v=input.product_input();char error[160];
  Products result(openmx_core_vna_host_products_create(prep.get(),&v,&out,error,sizeof(error)),openmx_core_vna_host_products_destroy);
  require(bool(result),error);return result;
}
void check(const OpenMXVnaHostProjectorData& out,bool convert=true)
{
  require(out.stride==8 && out.width==width && out.projectors,"projector shape differs");
  for(int s=0;s<species;++s)for(int p=0;p<points;++p)for(int column=0;column<8;++column) {
    const double expected=column<6?1000*s+100*(column/2)+10*(column%2)+p+0.125:0.;
    require(bits(out.projectors[(s*points+p)*8+column],expected),"projector transpose/padding differs");
  }
  if(!convert){require(!out.conversion_real && !out.conversion_imag,"unrequested conversion allocated");return;}
  for(int l=0;l<3;++l)for(int column=0;column<width;++column)for(int row=0;row<width;++row) {
    const bool active=row<=2*l && column<=2*l;const double x=100*l+10*row+column+0.375;
    const auto at=(l*width+column)*width+row;
    require(bits(out.conversion_real[at],active?x:0.) && bits(out.conversion_imag[at],active?-x:0.),"conversion transpose/padding differs");
  }
}
void check(const OpenMXVnaHostProductData& out)
{
  require(out.order_stride==16 && out.radials_max==3 && out.blocks==18 && out.products,"product shape differs");
  const std::size_t expected_offsets[]{0,9,9,18};const int expected_counts[]{3,0,3};
  for(int s=0;s<4;++s)require(out.species_offsets[s]==expected_offsets[s],"product species offset differs");
  for(int s=0;s<3;++s)require(out.radials[s]==expected_counts[s],"product radial count differs");
  const int angular[2][3]={{0,2,2},{1,2,4}},radial[2][3]={{0,0,1},{0,0,0}};
  const int source_offset[]{1,3,4,5,6};
  for(int group=0;group<2;++group)for(int left=0;left<3;++left)for(int right=0;right<3;++right) {
    const auto block=group*9+left*3+right;const int l0=angular[group][left],l1=angular[group][right];
    const int a=source_offset[l0]+radial[group][left],b=source_offset[l1]+radial[group][right];
    const auto row=std::size_t(2*group)*species_rows+1+((a*8+b)*13%64)*9;
    for(int p=0;p<points;++p)for(int l=0;l<16;++l) {
      const double expected=l0<=l1 && l<=2*l1?value(row+l,p):0.;
      require(bits(out.products[(block*points+p)*16+l],expected),"ragged product transpose/padding differs");
    }
  }
}
bool same(const OpenMXVnaHostProjectorData& a,const OpenMXVnaHostProjectorData& b)
{return a.stride==b.stride && a.width==b.width && a.projectors==b.projectors && a.conversion_real==b.conversion_real && a.conversion_imag==b.conversion_imag;}
bool same(const OpenMXVnaHostProductData& a,const OpenMXVnaHostProductData& b)
{return a.order_stride==b.order_stride && a.radials_max==b.radials_max && a.blocks==b.blocks && a.species_offsets==b.species_offsets && a.radials==b.radials && a.products==b.products;}
void contents()
{
  static_assert(std::is_const_v<std::remove_reference_t<decltype(*OpenMXVnaHostProductData{}.products)>>);
  static_assert(!std::is_copy_constructible_v<openmx::core::VnaHostProducts>);
  static_assert(!std::is_move_constructible_v<openmx::core::VnaHostProjectors>);
  auto prep=prepare();Inputs input;OpenMXVnaHostProjectorData a{},b{};OpenMXVnaHostProductData c{};
  auto one=pack(prep,input,a),two=pack(prep,input,b,0);auto three=pack(prep,input,c);
  require(a.projectors!=b.projectors && a.projectors!=input.projectors.data() && c.products!=input.products.data(),"packing aliases caller/another owner");
  input.discard();prep.reset();check(a);check(b,false);check(c);
}
void empty()
{
  auto prep=prepare(true,0);char error[160];OpenMXVnaHostProjectorData a{};OpenMXVnaHostProductData b{};
  OpenMXVnaHostProjectorInput pi{};OpenMXProductRadialView vi{};
  Projectors first(openmx_core_vna_host_projectors_create(prep.get(),&pi,&a,error,sizeof(error)),openmx_core_vna_host_projectors_destroy);
  require(bool(first) && a.stride==0 && !a.projectors && !a.conversion_real && !a.conversion_imag,error);
  Products second(openmx_core_vna_host_products_create(prep.get(),&vi,&b,error,sizeof(error)),openmx_core_vna_host_products_destroy);
  require(bool(second) && b.order_stride==8 && !b.radials_max && !b.blocks && !b.products,error);
  for(int s=0;s<species;++s)require(!b.radials[s] && !b.species_offsets[s],"empty species owns blocks");
  require(!b.species_offsets[species],"empty product sentinel differs");
}
void invalid()
{
  auto prep=prepare();Inputs input;OpenMXVnaHostProjectorData a{41,43,input.projectors.data(),nullptr,nullptr};const auto kept_a=a;
  OpenMXVnaHostProductData b{31,37,41,input.offsets.data(),nullptr,input.products.data()};const auto kept_b=b;char error[160];
  for(int kind=0;kind<12;++kind) {
    auto pi=input.projector_input();
    if(kind==0)pi.convert=2;if(kind==1)pi.projectors.values=nullptr;
    if(kind>=2 && kind<6)pi.projectors.extent[kind-2]=0;
    if(kind==6)pi.projectors.extent[0]=SIZE_MAX;
    if(kind==7)pi.width=4;if(kind==8)pi.width=SIZE_MAX;if(kind==9)pi.conversion=nullptr;
    const auto live=openmx::test::live_allocations();
    Projectors owner(openmx_core_vna_host_projectors_create(kind==10?nullptr:prep.get(),kind==11?nullptr:&pi,&a,error,sizeof(error)),openmx_core_vna_host_projectors_destroy);
    require(!owner && *error && same(a,kept_a) && openmx::test::live_allocations()==live,"invalid projector input escaped, leaked or changed output");
  }
  for(int kind=0;kind<12;++kind) {
    auto vi=input.product_input();const auto saved_layout=input.layout;
    const auto saved_offset=input.offsets[0],saved_pair=input.pairs[9];
    if(kind==0)vi.values=nullptr;if(kind==1)vi.layout=nullptr;
    if(kind==2)vi.species=2;if(kind==3)vi.stride=4;if(kind==4)vi.rows=SIZE_MAX;
    if(kind==5)input.layout[0].orbital_offsets=nullptr;if(kind==6)input.layout[0].pair_rows=nullptr;
    if(kind==7)input.layout[0].orbitals=SIZE_MAX;if(kind==8)input.offsets[0]=SIZE_MAX;
    if(kind==9)input.layout[0].first_row=vi.rows;if(kind==10)input.pairs[9]=vi.rows;
    const auto live=openmx::test::live_allocations();
    Products owner(openmx_core_vna_host_products_create(prep.get(),kind==11?nullptr:&vi,&b,error,sizeof(error)),openmx_core_vna_host_products_destroy);
    input.layout=saved_layout;input.offsets[0]=saved_offset;input.pairs[9]=saved_pair;
    require(!owner && *error && same(b,kept_b) && openmx::test::live_allocations()==live,"invalid product input escaped, leaked or changed output");
  }
  auto pi=input.projector_input();auto vi=input.product_input();
  require(!openmx_core_vna_host_projectors_create(prep.get(),&pi,nullptr,nullptr,0),"null projector output accepted");
  require(!openmx_core_vna_host_products_create(prep.get(),&vi,nullptr,nullptr,0),"null product output accepted");
  require(!openmx_core_vna_host_products_create(nullptr,&vi,&b,error,sizeof(error)) && same(b,kept_b),"null product preparation accepted");
  auto one=pack(prep,input,a);auto two=pack(prep,input,b);check(a);check(b);
}
void allocation()
{
  auto prep=prepare();Inputs input;OpenMXVnaHostProjectorData a{};OpenMXVnaHostProductData b{};
  auto kept_projectors=pack(prep,input,a);auto kept_products=pack(prep,input,b);
  for(int kind=0;kind<3;++kind) {
    long count;
    {
      openmx::test::fail_allocation_after(1000);
      if(kind<2){OpenMXVnaHostProjectorData out{};auto trial=pack(prep,input,out,kind);count=1000-openmx::test::remaining_allocations();}
      else {OpenMXVnaHostProductData out{};auto trial=pack(prep,input,out);count=1000-openmx::test::remaining_allocations();}
      openmx::test::fail_allocation_after(-1);
    }
    require(count>=(kind==0?2:kind==1?3:4),"packing allocations not intercepted");
    for(long failure=0;failure<count;++failure) {
      const auto live=openmx::test::live_allocations();char error[160];
      auto pi=input.projector_input(kind==1);auto vi=input.product_input();auto out_a=a;auto out_b=b;
      openmx::test::fail_allocation_after(failure);
      if(kind<2) {
        auto* trial=openmx_core_vna_host_projectors_create(prep.get(),&pi,&out_a,error,sizeof(error));
        openmx::test::fail_allocation_after(-1);
        require(!trial && *error && same(a,out_a),"failed projector publication changed output");
      }else {
        auto* trial=openmx_core_vna_host_products_create(prep.get(),&vi,&out_b,error,sizeof(error));
        openmx::test::fail_allocation_after(-1);
        require(!trial && *error && same(b,out_b),"failed product publication changed output");
      }
      require(openmx::test::live_allocations()==live,"partial packing leaked");check(a);check(b);
      if(kind<2){auto retried=pack(prep,input,out_a,kind);check(out_a,kind==1);}
      else {auto retried=pack(prep,input,out_b);check(out_b);}
    }
    std::cout<<"packing allocation points "<<kind<<": "<<count<<'\n';
  }
}
void lifetime()
{
  const auto live=openmx::test::live_allocations();
  {
    auto prep=prepare();Inputs input;OpenMXVnaHostProjectorData a{},b{};OpenMXVnaHostProductData c{},d{};
    auto one=pack(prep,input,a),two=pack(prep,input,b);auto three=pack(prep,input,c),four=pack(prep,input,d);
    one.reset();three.reset();input.discard();prep.reset();check(b);check(d);
  }
  openmx_core_vna_host_projectors_destroy(nullptr);openmx_core_vna_host_products_destroy(nullptr);
  require(openmx::test::live_allocations()==live,"packed table ownership leaked");
}
void concurrent()
{
  auto prep=prepare();Inputs input;OpenMXVnaHostProjectorData a{};OpenMXVnaHostProductData b{};
  auto first=pack(prep,input,a);auto second=pack(prep,input,b);
  std::array<std::future<void>,8> tasks;
  for(auto& task:tasks)task=std::async(std::launch::async,[&] {
    for(int repeat=0;repeat<5;++repeat) {
      check(a);check(b);OpenMXVnaHostProjectorData x{};OpenMXVnaHostProductData y{};
      auto other=pack(prep,input,x);auto product=pack(prep,input,y);check(x);check(y);
      require(x.projectors!=a.projectors && y.products!=b.products,"concurrent packing reused mutable storage");
    }
  });
  for(auto& task:tasks)task.get();
}
} // namespace
int main(int argc,char** argv)
{
  try {
    require(argc==2,"expected VNA host table test name");const std::string_view name=argv[1];
    if(name=="contents")contents();else if(name=="empty")empty();else if(name=="invalid")invalid();
    else if(name=="allocation")allocation();else if(name=="lifetime")lifetime();else if(name=="concurrent")concurrent();
    else throw std::invalid_argument("unknown VNA host table test");
    std::cout<<"PASS VNA host tables "<<name<<'\n';
  }catch(const std::exception& error){openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1;}
}
