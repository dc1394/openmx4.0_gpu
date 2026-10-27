#include <openmx/basis/orbital_radial.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include "allocation_gate.hpp"

extern "C" void reference_orbital_radial(int,const double*,const double*,std::size_t,double,double*);
namespace {
using namespace openmx::basis;
const std::array<double,8> grid{.15,.3,.6,1.1,1.7,2.6,3.8,5.2};
const std::array<double,8> row{.7,1.4,-.2,.9,-.4,.3,.1,0};
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void near(double a,double b,double tolerance=3e-12)
{require(std::isfinite(a) && std::abs(a-b)<=tolerance*(1+std::abs(b)),"orbital radial analytic value differs");}
bool bits(double a,double b){return std::memcmp(&a,&b,sizeof(a))==0;}
template<class Function> void rejects(Function&& f)
{bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught,"invalid orbital radial input accepted");}
std::array<RadialChannel,5> channels(std::span<const double> values)
{return {{{0,values},{1,values},{2,values},{4,values},{29,values}}};}
void legacy()
{
  std::uint64_t random=0x8a654e129bd301afULL;std::size_t checked=0;
  auto next=[&]{random^=random<<13;random^=random>>7;random^=random<<17;return random;};
  for(int seed=0;seed<24;++seed)for(int n:{5,6,8,53}) {
    std::vector<double> radii(n),values(n);const double scale=seed%3==0?1e-5:seed%3==1?1.:1e5;
    double radius=seed%4==0?0.:0.01*scale;
    for(int i=0;i<n;++i) {
      radii[i]=radius;radius+=scale*(.01+double(next()%4096)/4096.);
      values[i]=seed%7==0?-0.0:(double(int(next()%8192)-4096)/1024.);
    }
    const auto desc=channels(values);const OrbitalRadialTable owned(radii,desc);const OrbitalRadialTableView view(radii,desc);
    std::vector<double> points{0.,-0.,radii[0]*.5,std::nextafter(radii.back(),INFINITY),radii.back()+scale};
    for(double r:radii) {
      points.push_back(r);points.push_back(std::nextafter(r,INFINITY));
      if(r>0)points.push_back(std::nextafter(r,0.));
    }
    for(int i=0;i<50;++i)points.push_back(radii.back()*double(next()%10000)/10000.);
    for(double r:points) {
      std::array<OrbitalRadialSample,6> batch{},borrowed{};batch[5]={71.,73.};borrowed[5]=batch[5];
      owned.evaluate(r,batch);view.evaluate(r,borrowed);
      require(batch[5].value==71. && batch[5].radial_derivative==73. && borrowed[5].value==71.,"batch overwrote output tail");
      for(std::size_t c=0;c<desc.size();++c) {
        double expected[2];reference_orbital_radial(desc[c].angular,radii.data(),values.data(),radii.size(),r,expected);
        const auto a=owned.sample(c,r),b=view.sample(c,r);
        for(auto sample:{a,b,batch[c],borrowed[c]})
          if(!bits(sample.value,expected[0]) || !bits(sample.radial_derivative,expected[1])) {
            std::cerr<<"seed="<<seed<<" mesh="<<n<<" l="<<desc[c].angular<<" radius="<<r<<'\n';
            throw std::runtime_error("orbital radial sample differs from frozen C bits");
          }
        ++checked;
      }
    }
  }
  std::cout<<"frozen C coefficient pairs checked: "<<checked<<'\n';
}
void analytic()
{
  std::array<std::array<double,8>,3> values;
  for(std::size_t i=0;i<grid.size();++i){const double r=grid[i];values[0][i]=2.+4.*r*r;values[1][i]=3.*r;values[2][i]=5.*r*r;}
  const std::array<RadialChannel,3> desc{{{0,values[0]},{1,values[1]},{4,values[2]}}};
  const OrbitalRadialTable table(grid,desc);
  for(double r:{0.,.01,.075,.15,.2,.3,.4,.85,1.7,2.5,3.8,4.9,5.2}) {
    const auto s=table.sample(0,r),p=table.sample(1,r),g=table.sample(2,r);
    near(s.value,2.+4.*r*r);near(s.radial_derivative,8.*r);
    near(p.value,3.*r);near(p.radial_derivative,3.);
    near(g.value,5.*r*r);near(g.radial_derivative,10.*r);
  }
  for(std::size_t c=0;c<3;++c){const auto x=table.sample(c,5.3);require(bits(x.value,0.) && bits(x.radial_derivative,0.),"outside mesh is not zero");}
}
void derivatives()
{
  const auto desc=channels(row);const OrbitalRadialTable table(grid,desc);
  for(std::size_t c=0;c<desc.size();++c)for(double r:{.025,.08,.12,.2,.45,.85,1.4,2.1,3.2,4.5,5.1}) {
    const double h=1e-6;const double slope=(table.sample(c,r+h).value-table.sample(c,r-h).value)/(2*h);
    near(table.sample(c,r).radial_derivative,slope,2e-7);
  }
}
void policy()
{
  const RadialChannel c{0,row};const OrbitalRadialTable orbital(grid,{&c,1});const RadialTable phif(RadialKind::pao,grid,{&c,1});
  near(orbital.sample(0,grid.front()).value,row.front());
  require(std::abs(orbital.sample(0,0).value-phif.sample(0,0))>1e-4 &&
      std::abs(orbital.sample(0,grid.front()).value-phif.sample(0,grid.front()))>1e-4,"orbital and PhiF endpoint contracts were conflated");
}
void invalid()
{
  const RadialChannel c{0,row};
  rejects([&]{OrbitalRadialTable t{std::span<const double>(grid).first(4),std::span<const RadialChannel>(&c,1)};});
  for(int kind=0;kind<7;++kind) {
    auto r=grid;auto v=row;RadialChannel bad{0,v};
    if(kind==0)r[0]=-1.;if(kind==1)r[3]=r[2];if(kind==2)r[2]=NAN;
    if(kind==3)v[3]=INFINITY;if(kind==4)bad.angular=-1;if(kind==5)bad.angular=30;
    if(kind==6)bad.values=std::span(v).first(7);
    rejects([&]{OrbitalRadialTableView view(r,{&bad,1});});
  }
  const OrbitalRadialTable table(grid,{&c,1});const OrbitalRadialTableView view(grid,{&c,1});
  std::array<OrbitalRadialSample,2> out{{{71.,73.},{79.,83.}}};
  for(double radius:{-1.,double(NAN),double(INFINITY)}) {
    rejects([&]{(void)table.sample(0,radius);});rejects([&]{view.evaluate(radius,out);});
    require(out[0].value==71. && out[0].radial_derivative==73.,"invalid radius changed output");
  }
  rejects([&]{table.evaluate(1.,{});});rejects([&]{(void)table.sample(1,1.);});rejects([&]{(void)view.angular(1);});
  char error[128];OpenMXOrbitalRadialSample sample{41.,43.};
  for(int kind=0;kind<6;++kind) {
    const int ok=openmx_orbital_radial_sample(kind==0?30:0,kind==1?nullptr:grid.data(),kind==2?nullptr:row.data(),kind==3?4:grid.size(),kind==4?-1.:.2,kind==5?nullptr:&sample,error,sizeof(error));
    require(!ok && *error && sample.value==41. && sample.radial_derivative==43.,"invalid C input accepted or changed output");
  }
}
void ownership()
{
  auto radii=grid,values=row;auto desc=channels(values);OrbitalRadialTable first(radii,desc);
  const auto expected=first.sample(0,.08);OrbitalRadialTable copied=first,moved=std::move(first);
  require(copied.channel_count()==5 && moved.angular(3)==4,"copy/move lost metadata");
  std::fill(radii.begin(),radii.end(),NAN);std::fill(values.begin(),values.end(),NAN);desc.fill({-1,{}});
  near(copied.sample(0,.08).value,expected.value);near(moved.sample(0,.08).radial_derivative,expected.radial_derivative);
  rejects([&]{(void)first.sample(0,.08);});rejects([&]{first.evaluate(.08,{});});
  first=copied;copied=std::move(moved);near(first.sample(0,.08).value,expected.value);
  const OrbitalRadialTable empty(grid,{});require(empty.channel_count()==0,"empty basis owns channels");
  empty.evaluate(0.,{});rejects([&]{(void)empty.sample(0,0.);});
}
void allocation()
{
  const auto desc=channels(row);const OrbitalRadialTable source(grid,desc);const OrbitalRadialTableView view(grid,desc);
  long failures=0;
  for(long index=0;index<12;++index) {
    bool success=false;allocations_before_failure=index;
    try{OrbitalRadialTable trial(grid,desc);success=true;}catch(const std::bad_alloc&){}
    allocations_before_failure=-1;if(success)break;++failures;
  }
  require(failures==3,"construction fault sweep missed owned grid/values/metadata");
  std::array<double,8> other=row;other[4]=1.7;const RadialChannel c{0,other};OrbitalRadialTable target(grid,{&c,1});
  const auto kept=target.sample(0,.08);failures=0;
  require(!bits(kept.value,source.sample(0,.08).value),"copy failure probe must distinguish source and target data");
  for(long index=0;index<12;++index) {
    bool success=false;allocations_before_failure=index;
    try{target=source;success=true;}catch(const std::bad_alloc&){}
    allocations_before_failure=-1;if(success)break;++failures;
    require(target.channel_count()==1 && bits(target.sample(0,.08).value,kept.value),"failed copy assignment damaged owner");
  }
  require(failures==3 && target.channel_count()==5,"copy fault sweep missed owned buffers");
  require(bits(target.sample(0,.08).value,source.sample(0,.08).value),"successful assignment lost sampled values");
  std::array<OrbitalRadialSample,5> out{};char error[128];OpenMXOrbitalRadialSample scalar{};
  allocations_before_failure=0;
  source.evaluate(.2,out);view.evaluate(.2,out);const auto single=source.sample(0,.2);
  const int ok=openmx_orbital_radial_sample(0,grid.data(),row.data(),grid.size(),.2,&scalar,error,sizeof(error));
  allocations_before_failure=-1;
  require(ok && bits(single.value,scalar.value),"successful evaluation allocated or changed C result");
}
void concurrent()
{
  const auto desc=channels(row);const OrbitalRadialTable table(grid,desc);const OrbitalRadialTableView view(grid,desc);
  std::array<std::future<void>,8> tasks;
  for(auto& task:tasks)task=std::async(std::launch::async,[&]{
    std::array<OrbitalRadialSample,5> out{};
    for(int i=0;i<100;++i) {
      const double r=.02+i*.04;table.evaluate(r,out);
      for(std::size_t c=0;c<5;++c){const auto other=view.sample(c,r);require(bits(out[c].value,other.value) && bits(out[c].radial_derivative,other.radial_derivative),"concurrent evaluation differs");}
    }
  });
  for(auto& task:tasks)task.get();
}
} // namespace
int main(int argc,char** argv)
{
  try {
    require(argc==2,"expected orbital radial test name");const std::string_view name=argv[1];
    if(name=="legacy")legacy();else if(name=="analytic")analytic();else if(name=="derivatives")derivatives();
    else if(name=="policy")policy();else if(name=="invalid")invalid();else if(name=="ownership")ownership();
    else if(name=="allocation")allocation();else if(name=="concurrent")concurrent();else throw std::invalid_argument("unknown test");
    std::cout<<"PASS orbital radial "<<name<<'\n';
  }catch(const std::exception& error){allocations_before_failure=-1;std::cerr<<error.what()<<'\n';return 1;}
}
