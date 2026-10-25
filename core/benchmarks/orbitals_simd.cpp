#include "openmx_core_orbitals_simd.h"
#include "openmx_core_host_simd.h"
#include "../tests/host_orbitals_reference.hpp"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Orbital values on the grid points of one atom sphere (an s2p2d1 species,
// 13 orbitals, a 500-point logarithmic PAO mesh, 0.2 bohr grid), against the
// scalar device routine SOG_point_eval executed on the host.
int main(int argc,char** argv) {
  try {
    const int points=argc>1?std::atoi(argv[1]):65536,repeats=argc>2?std::atoi(argv[2]):10;
    if(points<1 || repeats<1)throw std::invalid_argument("positive point and repetition counts required");
    const int mesh=500,nb[4]={2,2,1,0},maxl=2,no=13;const double rcut=7.0,r0=1.0e-4;
    std::vector<double> rv(mesh),rwf;
    for(int m=0;m<mesh;++m)rv[m]=std::exp(std::log(r0)+(std::log(rcut)-std::log(r0))*m/double(mesh-1));
    for(int L0=0;L0<=maxl;++L0)for(int Mul0=0;Mul0<nb[L0];++Mul0)for(int m=0;m<mesh;++m)
      rwf.push_back(std::pow(rv[m],L0)*std::exp(-(0.7+0.3*Mul0)*rv[m])*(1.0+0.25*std::sin((2.0+L0)*rv[m]+Mul0)));
    const size_t rv_off[1]={0},rwf_base[1]={0};const int meshes[1]={mesh},maxls[1]={maxl};
    const int nbs[4]={nb[0],nb[1],nb[2],nb[3]};
    const double spacing=0.2;const int ng=96;
    OpenMXOrbitalsFrame frame{};frame.ng3=ng;frame.ng23=ng*ng;frame.max_no=no;
    for(int i=0;i<3;++i){frame.frame[4*i]=spacing;frame.origin[i]=-0.5*spacing*ng;}
    std::vector<double> translations(27*3);
    for(int c=0;c<27;++c){const int l[3]={c/9-1,c/3%3-1,c%3-1};for(int a=0;a<3;++a)translations[3*c+a]=l[a]*spacing*ng;}
    const OpenMXOrbitalsTables tables={rv.data(),rwf.data(),translations.data(),rv_off,rwf_base,meshes,maxls,nbs,1,3,8,27};
    // The atom near the origin; keep the grid points inside its sphere.
    std::mt19937 rng(42u);std::uniform_int_distribution<int> grid(0,ng*ng*ng-1);
    std::vector<OpenMXOrbitalsPoint> q;
    while(int(q.size())<points) {
      OpenMXOrbitalsPoint p{};p.wan=0;p.no=no;p.grid=grid(rng);p.cell=13;p.xyz[0]=0.013;p.xyz[1]=-0.021;p.xyz[2]=0.007;
      const int n1=p.grid/frame.ng23,n2=(p.grid-n1*frame.ng23)/frame.ng3,n3=p.grid-n1*frame.ng23-n2*frame.ng3;
      const double x=n1*spacing+frame.origin[0]-p.xyz[0],y=n2*spacing+frame.origin[1]-p.xyz[1],z=n3*spacing+frame.origin[2]-p.xyz[2];
      if(x*x+y*y+z*z<=rcut*rcut)q.push_back(p);
    }
    std::vector<float> reference(size_t(points)*no),out(size_t(points)*no);
    auto seconds=[&](auto&& body){const auto t0=std::chrono::steady_clock::now();for(int r=0;r<repeats;++r)body();
      return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();};
    const double legacy=seconds([&]{for(int i=0;i<points;++i){const auto& p=q[i];
      openmx::test::sog_point_eval(p.wan,p.no,p.xyz[0],p.xyz[1],p.xyz[2],p.shift[0],p.shift[1],p.shift[2],p.grid,p.cell,
        translations.data(),rv.data(),rwf.data(),rv_off,rwf_base,meshes,maxls,nbs,frame.ng23,frame.ng3,
        frame.frame[0],frame.frame[1],frame.frame[2],frame.frame[3],frame.frame[4],frame.frame[5],frame.frame[6],frame.frame[7],frame.frame[8],
        frame.origin[0],frame.origin[1],frame.origin[2],reference.data()+size_t(i)*no);}});
    std::cout<<std::fixed<<std::setprecision(1)<<"points="<<points<<" orbitals="<<no<<" mesh="<<mesh<<" repeats="<<repeats<<'\n'
        <<std::setw(24)<<"SOG_point_eval (host)"<<std::setw(10)<<1e9*legacy/(double(points)*repeats)<<" ns/point\n";
    for(int mode=OPENMX_GRID_SCALAR;mode<=OPENMX_GRID_NEON;++mode) {
      if(!openmx_core_host_simd_usable(mode))continue;
      const double s=seconds([&]{if(!openmx_core_orbitals_points(&tables,&frame,q.data(),q.size(),out.data(),mode))throw std::runtime_error("kernel refused");});
      if(std::memcmp(out.data(),reference.data(),out.size()*sizeof(float))!=0)throw std::runtime_error("host kernel differs from SOG_point_eval");
      std::cout<<std::setw(24)<<openmx_core_host_simd_name(mode)<<std::setw(10)<<1e9*s/(double(points)*repeats)<<" ns/point  ("<<std::setprecision(2)<<legacy/s<<"x)\n"<<std::setprecision(1);
    }
    // Derivatives at Force3 sizes: the same sphere, displacements from the atom.
    std::vector<OpenMXOrbitalsDisplacement> d;
    for(const auto& p:q) {
      const int n1=p.grid/frame.ng23,n2=(p.grid-n1*frame.ng23)/frame.ng3,n3=p.grid-n1*frame.ng23-n2*frame.ng3;
      OpenMXOrbitalsDisplacement e{};e.wan=0;e.no=no;
      e.x=n1*spacing+frame.origin[0]-p.xyz[0];e.y=n2*spacing+frame.origin[1]-p.xyz[1];e.z=n3*spacing+frame.origin[2]-p.xyz[2];
      d.push_back(e);
    }
    std::vector<double> dreference(size_t(points)*no*3),dout(size_t(points)*no*3);
    const int nbs4[4]={nb[0],nb[1],nb[2],nb[3]};
    const double dlegacy=seconds([&]{for(int i=0;i<points;++i)
      openmx::test::get_dorbitals_reference(maxl,nbs4,mesh,rv.data(),rwf.data(),d[i].x,d[i].y,d[i].z,dreference.data()+size_t(i)*no*3);});
    // The libm floor of the exact derivatives: the six transcendental calls per point.
    volatile double sink=0.0;
    const double dlibm=seconds([&]{double acc=0.0;for(int i=0;i<points;++i){double S[3];
      openmx::test::xyz2spherical_reference(d[i].x,d[i].y,d[i].z,S);
      acc+=std::sin(S[1])+std::cos(S[1])+std::sin(S[2])+std::cos(S[2]);}sink=sink+acc;});
    std::cout<<"derivatives (Get_dOrbitals_work), "<<points<<" points\n"
        <<std::setw(24)<<"Get_dOrbitals_work (host)"<<std::setw(10)<<1e9*dlegacy/(double(points)*repeats)<<" ns/point\n"
        <<std::setw(24)<<"  libm angles alone"<<std::setw(10)<<1e9*dlibm/(double(points)*repeats)<<" ns/point\n";
    for(int mode=OPENMX_GRID_SCALAR;mode<=OPENMX_GRID_NEON;++mode) {
      if(!openmx_core_host_simd_usable(mode))continue;
      const double s=seconds([&]{if(!openmx_core_orbitals_dpoints(&tables,d.data(),d.size(),dout.data(),mode))throw std::runtime_error("derivative kernel refused");});
      if(std::memcmp(dout.data(),dreference.data(),dout.size()*sizeof(double))!=0)throw std::runtime_error("host derivative kernel differs from Get_dOrbitals_work");
      std::cout<<std::setw(24)<<openmx_core_host_simd_name(mode)<<std::setw(10)<<1e9*s/(double(points)*repeats)<<" ns/point  ("<<std::setprecision(2)<<dlegacy/s<<"x)\n"<<std::setprecision(1);
      const double sc=seconds([&]{if(!openmx_core_orbitals_dpoints_cosines(&tables,d.data(),d.size(),dout.data(),mode))throw std::runtime_error("cosine kernel refused");});
      std::cout<<std::setw(24)<<(std::string(openmx_core_host_simd_name(mode))+" cosines")<<std::setw(10)<<1e9*sc/(double(points)*repeats)<<" ns/point  ("<<std::setprecision(2)<<dlegacy/sc<<"x)\n"<<std::setprecision(1);
    }
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
