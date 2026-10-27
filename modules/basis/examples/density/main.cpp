#include <openmx/basis/pao_file.hpp>
#include <openmx/basis/density_table.hpp>
#include <iostream>

int main(int argc,char** argv)
{
  try{
    if(argc!=2){std::cerr<<"usage: density_example FILE.pao\n";return 1;}
    const auto pao=openmx::basis::PaoFile::from_file(argv[1]);
    const openmx::basis::LogDensityTable density(pao.log_radii(),pao.radii(),pao.density());
    const openmx::basis::GaussLegendreRule quadrature(1500);
    std::cout<<"rho(r=0.85 bohr)="<<density.sample(.85)
             <<" d(rho)/dr="<<density.derivative(.85)
             <<" Hartree="<<openmx::basis::atomic_hartree_potential(density.view(),.85,quadrature.view())<<'\n';
    return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
