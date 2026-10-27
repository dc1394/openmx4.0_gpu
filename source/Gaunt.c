#include "openmx_basis_gaunt.h"

/* Native compatibility boundary. The original f[40] factorial sum is shared
   with the standalone basis library. Reject unsupported degrees before any
   factorial access; selection-rule zeros with |m|>l remain valid here because
   rectangular projector tables also request those entries. */
double Gaunt(int l, int m, int l1, int m1, int l2, int m2)
{
  if(!openmx_basis_gaunt_degrees_valid(l,l1,l2) ||
      m < -76 || m>76 || m1 < -76 || m1>76 || m2 < -76 || m2>76)
    return NAN;
  return openmx_basis_gaunt_kernel(l,m,l1,m1,l2,m2);
}
