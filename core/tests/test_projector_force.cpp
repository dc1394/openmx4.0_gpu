#include <openmx/core/projector_force.hpp>
#include "allocation_faults.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
using namespace openmx::core;
namespace {
void require(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
void near(double a,double b){require(std::isfinite(a) && std::abs(a-b)<1e-13*(1+std::abs(b)),"projector value differs");}
struct Input {
  int counts[3]{0,4,3},components[3]{0,16,15},relativistic[3]{0,1,0};
  int angular[3][7]{{0},{0,2,0,1,3},{0,1,1,4}};
  double energies[2][3][6]{};
  Input(){for(int c=0;c<2;++c)for(int w=0;w<3;++w)for(int p=0;p<6;++p)energies[c][w][p]=.25+.75*c+.125*w-.0625*p;}
  OpenMXAtomicViews view(){
    OpenMXAtomicViews v{};v.species.species=3;v.species.radial_projectors=counts;
    v.species.projector_components=components;v.species.relativistic=relativistic;
    v.projector_angular=&angular[0][0];v.projector_angular_species=3;v.projector_angular_stride=7;
    v.projector_energies={&energies[0][0][0],{2,3,6}};return v;
  }
};
void layouts()
{
  for(int spin:{0,1,3})for(int so:{0,1}) {
    Input in;ProjectorForce owner(in.view(),spin,so);const auto v=owner.view();
    require(v.spin==spin && v.spin_orbit==so && v.species==3,"projector policy differs");
    require(v.species_views[0].count==0 && !v.species_views[0].channels,"empty projector species allocated channels");
    for(std::size_t w=1;w<3;++w) {
      const auto& species=v.species_views[w];std::size_t offset=0;
      require(species.relativistic==in.relativistic[w] && species.count==std::size_t(in.counts[w]),"projector species shape differs");
      for(std::size_t j=0;j<species.count;++j) {
        const auto& channel=species.channels[j];const int l=in.angular[w][j+1];
        require(channel.offset==offset && channel.twice_angular==2*l,"projector component offset differs");
        near(channel.weight[0],double(l+1)/(2*l+1));near(channel.weight[1],double(l)/(2*l+1));
        near(channel.energy[0],in.energies[0][w][j]);near(channel.energy[1],in.relativistic[w]?in.energies[1][w][j]:0);
        offset+=2*l+1;
      }
      require(offset==species.components && offset==std::size_t(in.components[w]),"projector magnetic component count differs");
    }
  }
  OpenMXAtomicViews empty{};ProjectorForce zero(empty,0,0);require(zero.view().species==0,"empty calculation gained species");
  Input in;in.counts[1]=1;in.components[1]=1;in.angular[1][1]=0;
  ProjectorForce scalar(in.view(),3,1);std::array<double,1> derivative{.75},value{-.5};
  OpenMXProjectorForceRows row{{{derivative.data(),derivative.data(),derivative.data()},
      {derivative.data(),derivative.data(),derivative.data()}},{value.data(),value.data()},1,-1,1};
  std::array<dcomplex,9> sum{};projector_force_trace(scalar.view(),1,row,sum);
  for(int s=0;s<3;++s)for(int axis=0;axis<3;++axis){near(sum[3*s+axis].r,s<2?-.75*in.energies[0][1][0]*-.5:0);near(sum[3*s+axis].i,0);}
}
void blocks()
{
  Input in;ProjectorForce owner(in.view(),3,1);const auto view=owner.view();
  double derivative[2][3][60],value[2][63];
  for(int c=0;c<2;++c) {
    for(int i=0;i<63;++i)value[c][i]=std::sin(.31*i+.17*c);
    for(int axis=0;axis<3;++axis)for(int i=0;i<60;++i)derivative[c][axis][i]=std::cos(.23*i+.71*c-.19*axis);
  }
  OpenMXProjectorForceTerm terms[2]{};
  for(auto& term:terms) {
    term.derivative_rows=term.value_rows=3;term.derivative_stride=20;term.value_stride=21;term.sign=1;
    for(int c=0;c<2;++c){term.value[c]=value[c];for(int axis=0;axis<3;++axis)term.derivative[c][axis]=derivative[c][axis];}
  }
  terms[1].sign=-1;terms[1].reverse_energy=1;terms[1].transpose=1;
  std::array<dcomplex,135> output;for(int i=0;i<135;++i)output[i]={std::sin(.13*i),std::cos(.17*i)};
  const auto original=output;auto expected=output;
  OpenMXProjectorForceBlock block{2,2,3,5,2,terms,{output.data(),output.data()+45,output.data()+90},1};
  for(std::size_t m=0;m<2;++m)for(std::size_t n=0;n<2;++n) {
    std::array<dcomplex,9> sum{};
    for(const auto& term:terms) {
      OpenMXProjectorForceRows row{};row.columns=16;row.sign=term.sign;row.reverse_energy=term.reverse_energy;
      for(int c=0;c<2;++c){row.value[c]=term.value[c]+(term.transpose?m:n)*term.value_stride;
        for(int axis=0;axis<3;++axis)row.derivative[c][axis]=term.derivative[c][axis]+(term.transpose?n:m)*term.derivative_stride;}
      projector_force_trace(view,1,row,sum);
    }
    for(int axis=0;axis<3;++axis)for(int c=0;c<3;++c){auto& cell=expected[45*axis+15*c+5*m+n];cell.r+=sum[3*c+axis].r;cell.i+=sum[3*c+axis].i;}
  }
  char error[256];openmx::test::fail_allocation_after(0);
  const int ok=openmx_core_projector_force_block(&view,1,&block,error,sizeof(error));openmx::test::fail_allocation_after(-1);
  require(ok,error);for(int i=0;i<135;++i){near(output[i].r,expected[i].r);near(output[i].i,expected[i].i);}
  const auto saved=terms[1];
  for(int mode=0;mode<10;++mode) {
    output=original;terms[1]=saved;auto bad=block;
    if(mode==0)terms[1].value_rows=1;if(mode==1)terms[1].derivative_stride=15;
    if(mode==2)terms[1].sign=std::numeric_limits<double>::quiet_NaN();if(mode==3)terms[1].value[1]=nullptr;
    if(mode==4)bad.output[2]=nullptr;if(mode==5)bad.output_stride=std::numeric_limits<std::size_t>::max();
    if(mode==6)bad.term_count=3;if(mode==7)terms[1].transpose=2;if(mode==8)terms[1].value_first=2;
    if(mode==9)bad.rows=4;
    require(!openmx_core_projector_force_block(&view,1,&bad,error,sizeof(error)) && error[0],"invalid projector block accepted");
    for(int i=0;i<135;++i)require(output[i].r==original[i].r && output[i].i==original[i].i,"invalid late block term changed output");
  }
  terms[1]=saved;auto empty=block;empty.rows=0;for(auto& pointer:empty.output)pointer=nullptr;
  require(openmx_core_projector_force_block(&view,1,&empty,error,sizeof(error)),error);
}
int snapshot(OpenMXSolverInvocation *scope,void*,double *result)
{
  Input in;auto atomic=in.view();OpenMXProjectorForceView first,last;char error[256];
  require(openmx_core_projector_force_prepare(scope,&atomic,3,1,&first,error,sizeof(error)),error);
  const auto* channels=first.species_views[1].channels;
  in.angular[1][1]=0;in.energies[0][1][0]=-19;
  require(channels[0].twice_angular==4 && channels[1].offset==5,"projector owner borrowed mutable angular metadata");
  near(channels[0].energy[0],.375);
  for(int i=0;i<20;++i) {
    Input next;next.energies[0][1][0]=i+.5;auto view=next.view();
    require(openmx_core_projector_force_prepare(scope,&view,1,0,&last,error,sizeof(error)),error);
    near(last.species_views[1].channels[0].energy[0],i+.5);
    require(first.species_views[1].channels==channels,"later preparation moved old projector view");
    near(channels[0].energy[0],.375);
  }
  const auto saved=last;require(!openmx_core_projector_force_prepare(scope,&atomic,-1,0,&last,error,sizeof(error)),"invalid projector spin accepted");
  require(last.species_views==saved.species_views,"failed projector preparation changed output");
  *result=19;return 1;
}
void invalid()
{
  for(int mode=0;mode<8;++mode) {
    Input in;auto v=in.view();
    if(mode==0)in.counts[1]=-1;if(mode==1)in.components[1]=15;
    if(mode==2)v.projector_energies.extent[2]=2;if(mode==3)v.projector_angular_stride=3;
    if(mode==4)in.energies[1][1][2]=std::numeric_limits<double>::infinity();
    if(mode==5)in.angular[1][1]=4;if(mode==6)v.projector_angular=nullptr;
    if(mode==7)v.projector_energies.extent[0]=std::numeric_limits<std::size_t>::max();
    bool threw=false;try{ProjectorForce rejected(v,3,1);}catch(const std::exception&){threw=true;}
    require(threw,"invalid projector metadata accepted");
  }
  Input in;ProjectorForce owner(in.view(),3,1);const auto v=owner.view();std::array<double,16> row{};
  OpenMXProjectorForceRows input{{{row.data(),row.data(),row.data()},{row.data(),row.data(),row.data()}},
      {row.data(),row.data()},16,1,0};char error[256];
  std::array<dcomplex,9> sums;for(int i=0;i<9;++i)sums[i]={double(i),-double(i)};const auto saved=sums;
  for(int mode=0;mode<7;++mode) {
    auto trace=input;auto view=v;std::size_t w=1;std::array<OpenMXForceProjectorSpecies,3> species;
    std::array<OpenMXForceProjectorChannel,4> channels;
    if(mode==0)trace.columns=15;if(mode==1)trace.derivative[1][2]=nullptr;
    if(mode==2)trace.sign=std::numeric_limits<double>::quiet_NaN();if(mode==3)trace.reverse_energy=2;
    if(mode==4)w=3;if(mode==5)view.spin=2;
    if(mode==6){std::copy(v.species_views,v.species_views+3,species.begin());
      std::copy(species[1].channels,species[1].channels+4,channels.begin());
      channels[3].offset=10;species[1].channels=channels.data();view.species_views=species.data();}
    require(!openmx_core_projector_force_trace(&view,w,&trace,sums.data(),error,sizeof(error)) && error[0],"invalid projector trace accepted");
    for(int i=0;i<9;++i)require(sums[i].r==saved[i].r && sums[i].i==saved[i].i,"failed projector trace partially changed output");
  }
  Input none;none.counts[1]=none.components[1]=0;ProjectorForce empty(none.view(),3,1);
  OpenMXProjectorForceRows blank{};blank.sign=1;projector_force_trace(empty.view(),1,blank,sums);
  for(int i=0;i<9;++i)require(sums[i].r==saved[i].r && sums[i].i==saved[i].i,"empty projector trace changed output");
}
int lifetime_callback(OpenMXSolverInvocation *scope,void *opaque,double *result)
{
  snapshot(scope,nullptr,result);const int mode=*static_cast<int*>(opaque);
  if(mode==1)return 0;if(mode==2)throw std::runtime_error("injected projector exception");return 1;
}
void lifetime()
{
  for(int mode=0;mode<3;++mode) {
    const auto live=openmx::test::live_allocations();char error[256];double value=31;
    const int ok=openmx_core_solver_run(2,lifetime_callback,&mode,&value,error,sizeof(error));
    require(ok==(mode==0) && openmx::test::live_allocations()==live,"projector invocation leaked");near(value,mode==0?19:31);
  }
}
int allocate(OpenMXSolverInvocation *scope,void*,double *result)
{
  Input in;const auto v=in.view();OpenMXProjectorForceView out;char error[256];
  for(int i=0;i<3;++i)if(!openmx_core_projector_force_prepare(scope,&v,3,1,&out,error,sizeof(error)))return 0;
  *result=19;return 1;
}
void allocation()
{
  char error[256];double result=31;openmx::test::fail_allocation_after(10000);
  const int ok=openmx_core_solver_run(2,allocate,nullptr,&result,error,sizeof(error));
  const long count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  require(ok && count>10,"projector allocation interception failed");
  for(long i=0;i<count;++i) {
    const auto live=openmx::test::live_allocations();result=31;openmx::test::fail_allocation_after(i);
    const int passed=openmx_core_solver_run(2,allocate,nullptr,&result,error,sizeof(error));openmx::test::fail_allocation_after(-1);
    require(!passed && openmx::test::live_allocations()==live,"partial projector preparation leaked");near(result,31);
  }
}
void concurrent()
{
  Input in;ProjectorForce owner(in.view(),3,1);const auto view=owner.view();std::array<int,4> done;std::array<std::thread,4> threads;
  for(int worker=0;worker<4;++worker)threads[worker]=std::thread([&,worker]{try {
    std::array<double,16> row;for(int i=0;i<16;++i)row[i]=std::sin(i*.1+worker*.3);
    OpenMXProjectorForceRows trace{{{row.data(),row.data(),row.data()},{row.data(),row.data(),row.data()}},
        {row.data(),row.data()},16,worker%2?-1.0:1.0,worker%2};
    std::array<dcomplex,9> a{},b{};projector_force_trace(view,1,trace,a);
    for(int i=0;i<20;++i)projector_force_trace(view,1,trace,b);
    for(int i=0;i<9;++i){near(b[i].r,20*a[i].r);near(b[i].i,20*a[i].i);}done[worker]=1;
  }catch(...){done[worker]=-1;}});
  for(auto& thread:threads)thread.join();for(auto status:done)require(status==1,"shared projector coefficients raced");
}
}
int main(int argc,char **argv)
{
  try {
    const std::string name=argc==2?argv[1]:"";
    if(name=="layouts"){layouts();blocks();}else if(name=="invalid")invalid();else if(name=="lifetime")lifetime();
    else if(name=="allocation")allocation();else if(name=="concurrent")concurrent();else if(name=="snapshot") {
      char error[256];double result;require(openmx_core_solver_run(1,snapshot,nullptr,&result,error,sizeof(error)),error);near(result,19);
    }else throw std::runtime_error("unknown projector Force test");
    std::cout<<"PASS projector-force "<<name<<'\n';return 0;
  }catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}
