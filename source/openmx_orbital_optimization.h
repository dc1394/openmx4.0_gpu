#ifndef OPENMX_ORBITAL_OPTIMIZATION_H
#define OPENMX_ORBITAL_OPTIMIZATION_H
#include "openmx_core_atomic.h"
#include "openmx_core_orbital_optimization.h"

void Begin_Orbital_Optimization(OpenMXCoreContext*,const OpenMXAtomicViews*);
double Opt_Contraction(OpenMXCoreContext*,int,double,
    const OpenMXElectronicViews*,const OpenMXLocalBasisViews*);
#endif
