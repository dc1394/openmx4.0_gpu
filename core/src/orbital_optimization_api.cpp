#include "calculation_context.hpp"
#include <cstdio>

namespace {
template<class F> int checked(char* error,std::size_t capacity,F&& f) noexcept
{
  if(error && capacity)*error='\0';
  try {f();return 1;}
  catch(const std::exception& e){if(error && capacity)std::snprintf(error,capacity,"%s",e.what());}
  catch(...){if(error && capacity)std::snprintf(error,capacity,"unexpected orbital optimization error");}
  return 0;
}
void context_ok(OpenMXCoreContext* c)
{ if(!c || c->implementation.finished())throw std::invalid_argument("missing or finished orbital optimization context"); }
}
extern "C" int openmx_core_orbopt_begin(OpenMXCoreContext* c,const OpenMXOrbOptInput* in,
    OpenMXOrbOptViews* out,char* error,std::size_t capacity)
{
  return checked(error,capacity,[&]{
    context_ok(c);if(!in || !out)throw std::invalid_argument("missing orbital optimization input/output");
    auto owner=std::make_unique<openmx::core::OrbitalOptimization>(*in);
    const auto views=owner->views();c->orbital_optimization=std::move(owner);*out=views;
  });
}
extern "C" int openmx_core_orbopt_get(OpenMXCoreContext* c,OpenMXOrbOptViews* out,char* error,std::size_t capacity)
{
  return checked(error,capacity,[&]{context_ok(c);if(!out)throw std::invalid_argument("missing optimization output");
    *out=c->orbital_optimization ? c->orbital_optimization->views():OpenMXOrbOptViews{};});
}
extern "C" void openmx_core_orbopt_end(OpenMXCoreContext* c)
{ if(c)c->orbital_optimization.reset(); }
extern "C" int openmx_core_orbopt_step(OpenMXCoreContext* c,int iteration,double energy,
    const OpenMXElectronicViews* e,const OpenMXLocalBasisViews* l,const OpenMXOrbOptCallbacks* services,
    double* norm,char* error,std::size_t capacity)
{
  return checked(error,capacity,[&]{context_ok(c);
    if(!c->orbital_optimization || !e || !l || !services || !norm)throw std::invalid_argument("missing optimization state or arguments");
    *norm=c->orbital_optimization->step(iteration,energy,*e,*l,*services);
  });
}
extern "C" int openmx_core_orbopt_gradient(OpenMXCoreContext* c,const OpenMXElectronicViews* e,
    const OpenMXLocalBasisViews* l,const OpenMXOrbOptCallbacks* services,double* norm,char* error,std::size_t capacity)
{
  return checked(error,capacity,[&]{context_ok(c);
    if(!c->orbital_optimization || !e || !l || !services || !norm)throw std::invalid_argument("missing optimization state or arguments");
    *norm=c->orbital_optimization->gradient(*e,*l,*services);
  });
}
