/* Harness-only adapter from the frozen small geometry arrays to the real
   production owner. The owner must copy these automatic row tables. */
#include "openmx_core_atom_pairs.h"
static OpenMXAtomPairPlan *test_vna_pairs(OpenMXAtomPairBuffers *view)
{
  int counts[NA+1],*neighbor_rows[NA+1],*image_rows[NA+1];
  double *positions[NA+1],*translations[NH];
  for (int g=0;g<=NA;++g) {
    counts[g]=NH-1;neighbor_rows[g]=natn[g];image_rows[g]=ncn[g];positions[g]=Gxyz[g];
  }
  for (int h=0;h<NH;++h) translations[h]=atv[h];
  const OpenMXAtomPairInput input={NA,NA,NS,NH,M2G,WhatSpecies,counts,
      neighbor_rows,image_rows,positions,translations};
  char error[256];
  OpenMXAtomPairPlan *plan=openmx_core_atom_pairs_create(&input,view,error,sizeof(error));
  if (!plan) { fprintf(stderr,"%s\n",error);abort(); }
  return plan;
}
