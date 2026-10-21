#include <stdio.h>
#include <math.h>
#include "openmx_common.h"

double Nonlocal_Basis(const OpenMXAtomicViews *atomic, int wan, int Lnum_index, int Mnum, int so,
                      double r, double theta, double phi)
{
  int Lnum;
  double NWF,Radial_WF,Angular_WF;

  Lnum = openmx_atomic_projector_angular(atomic,wan)[Lnum_index];
  Radial_WF = Nonlocal_RadialF(atomic, wan,Lnum_index-1,so,r);
  Angular_WF = AngularF(Lnum,Mnum,theta,phi,0,0.0,0.0,0.0,0.0);
  NWF = Radial_WF*Angular_WF;

  return NWF;
} 
