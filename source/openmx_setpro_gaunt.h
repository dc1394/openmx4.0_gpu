#ifndef OPENMX_SETPRO_GAUNT_H
#define OPENMX_SETPRO_GAUNT_H
/* Gaunt coefficients Gaunt(L0,M0,L1,M1,LL,M0-M1) tabulated once per call for
   the host SIMD paths of openmx_dsvna_cpu.h and openmx_vna_cpu.h: the m sums
   of the projector expansions only ever ask for M0-M1, so the table replaces
   a pure function whose factorial loops dominated the m loops. Values are
   the function's own, bitwise. Relies on the includer's Gaunt(). */
#include <string.h>
#include "openmx_setpro_host.h"

typedef struct { int l0max,l1max,llmax; size_t row_stride; double *values; } SetProGauntTable;

/* values[(((L0*(L0+1)+M0)*(llmax+1)+LL)*(l1max+1)+L1)*(2*l1max+1)+(L1+M1)]:
   M1 is the contiguous dimension, so the Core kernels load the 2*L1+1
   coefficients of one (L0,M0,LL,L1) as vector lanes. */
static inline size_t SetPro_GauntTable_Row(const SetProGauntTable *t,int L0,int M0,int LL,int L1)
{
  return ((((size_t)(L0*(L0+1)+M0))*(size_t)(t->llmax+1)+(size_t)LL)*(size_t)(t->l1max+1)+(size_t)L1)*t->row_stride;
}
static inline void SetPro_GauntTable_Build(SetProGauntTable *t,int l0max,int l1max,int llmax)
{
  const size_t n0=(size_t)(l0max+1)*(l0max+1);
  int L0,M0,L1,M1,LL;
  t->l0max=l0max;t->l1max=l1max;t->llmax=llmax;t->row_stride=(size_t)(2*l1max+1);
  t->values=(double*)SetPro_HostAlloc(sizeof(double)*n0*(size_t)(llmax+1)*(size_t)(l1max+1)*t->row_stride);
  memset(t->values,0,sizeof(double)*n0*(size_t)(llmax+1)*(size_t)(l1max+1)*t->row_stride);
  for (L0=0; L0<=l0max; L0++) for (M0=-L0; M0<=L0; M0++)
    for (LL=0; LL<=llmax; LL++)
      for (L1=0; L1<=l1max; L1++) for (M1=-L1; M1<=L1; M1++)
        t->values[SetPro_GauntTable_Row(t,L0,M0,LL,L1)+(size_t)(L1+M1)] = Gaunt(L0,M0,L1,M1,LL,M0-M1);
}
static inline double SetPro_GauntTable_At(const SetProGauntTable *t,int L0,int M0,int L1,int M1,int LL)
{
  return t->values[SetPro_GauntTable_Row(t,L0,M0,LL,L1)+(size_t)(L1+M1)];
}
static inline void SetPro_GauntTable_Free(SetProGauntTable *t) { free(t->values); t->values=NULL; }
#endif
