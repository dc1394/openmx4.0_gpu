#include "openmx_core_atomic_density.h"
#include <openmx/basis/density_table.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
std::span<double> row(const OpenMXRadialView2& view,std::size_t species,std::size_t points)
{
  const auto limit=static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double);
  if(!view.values || !view.extent[1] || view.extent[0]>limit/view.extent[1] ||
     species>=view.extent[0] || points>view.extent[1])
    throw std::invalid_argument("invalid atomic density row capacity");
  return {openmx_radial_row2(&view,species),points};
}
}

extern "C" int openmx_core_atomic_hartree(const OpenMXAtomicViews* atomic,std::size_t species,
    int quadrature_count,const double* nodes,const double* weights,char* error,std::size_t capacity)
{
  if(error && capacity)*error='\0';
  try {
    if(!atomic || species>=atomic->species.species || !atomic->species.atomic_number ||
       !atomic->species.pao_mesh || !atomic->species.vps_mesh || quadrature_count<=0 || !nodes || !weights)
      throw std::invalid_argument("invalid atomic Hartree request");
    const int n=atomic->species.pao_mesh[species],m=atomic->species.vps_mesh[species];
    if(n<6 || n>INT_MAX-2 || m<1 || m>INT_MAX-2)
      throw std::invalid_argument("invalid atomic Hartree mesh size");
    const openmx::basis::LogDensityView density(row(atomic->pao_log_radii,species,n),
        row(atomic->pao_radii,species,n),row(atomic->density,species,static_cast<std::size_t>(n)+2));
    const openmx::basis::GaussLegendreView quadrature({nodes,static_cast<std::size_t>(quadrature_count)},
        {weights,static_cast<std::size_t>(quadrature_count)});
    const auto radii=row(atomic->vps_radii,species,m);
    const auto output=row(atomic->hartree_potential,species,static_cast<std::size_t>(m)+2);
    for(double r:radii)if(!std::isfinite(r) || r<=0)throw std::invalid_argument("invalid atomic Hartree output radius");
    std::vector<double> values(m,0.);
    if(atomic->species.atomic_number[species]!=0)
      for(std::size_t i=0;i<values.size();++i)
        values[i]=openmx::basis::atomic_hartree_potential(density,radii[i],quadrature);
    std::copy(values.begin(),values.end(),output.begin()+1);
    return 1;
  }catch(const std::exception& exception){if(error && capacity)std::snprintf(error,capacity,"%s",exception.what());}
  catch(...){if(error && capacity)std::snprintf(error,capacity,"unknown atomic Hartree error");}
  return 0;
}
