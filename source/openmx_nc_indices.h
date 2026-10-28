#ifndef OPENMX_NC_INDICES_H
#define OPENMX_NC_INDICES_H

#include "openmx_common.h"

/* The (2n x 2n) block-cyclic layout of the non-collinear Hamiltonian and
   overlap matrices: one element (i1, j1) of a spin block of the n x n layout
   of process id1 lands on process *idXX at local (*iXX, *jXX) for the four
   blocks 11, 22, 12 and 21 (21 is the transpose of 12).  Shared by the
   band and cluster Hs2/Ss2 builders; n is the basis size of the n x n
   layout, counted once per call by openmx_nc_basis_size (the former copies
   recounted it over every atom for every element). */
static inline int openmx_nc_basis_size(void)
{
  int i,n=0;
  for (i=1; i<=atomnum; i++) n += Spe_Total_CNO[WhatSpecies[i]];
  return n;
}

static inline void openmx_nc_trans_indices_n_to_n2( int n, int id1, int i1, int j1,
                            int *id11, int *i11, int *j11,
                            int *id22, int *i22, int *j22,
                            int *id12, int *i12, int *j12,
                            int *id21, int *i21, int *j21)
{
  int ig,jg,il,jl,loop,si,sj;
  int my_prow1,my_pcol1,id2;
  int brow2,bcol2,prow2,pcol2;

  my_prow1 = id1/np_cols;
  my_pcol1 = id1%np_cols;

  for (loop=1; loop<=4; loop++){

    if      (loop==1){ si = 0; sj = 0; }
    else if (loop==2){ si = n; sj = n; }
    else if (loop==3){ si = 0; sj = n; }
    else if (loop==4){ si = 0; sj = n; }

    if (loop!=4){
      ig = np_rows*nblk*((i1)/nblk) + (i1)%nblk + ((np_rows+my_prow1)%np_rows)*nblk + 1 + si;
      jg = np_cols*nblk*((j1)/nblk) + (j1)%nblk + ((np_cols+my_pcol1)%np_cols)*nblk + 1 + sj;
    }
    else{
      jg = np_rows*nblk*((i1)/nblk) + (i1)%nblk + ((np_rows+my_prow1)%np_rows)*nblk + 1 + si;
      ig = np_cols*nblk*((j1)/nblk) + (j1)%nblk + ((np_cols+my_pcol1)%np_cols)*nblk + 1 + sj;
    }

    brow2 = (ig-1)/nblk2;
    bcol2 = (jg-1)/nblk2;
    prow2 = brow2%np_rows2;
    pcol2 = bcol2%np_cols2;

    id2 = prow2*np_cols2 + pcol2; 

    il = (brow2/np_rows2+1)*nblk2+1;
    jl = (bcol2/np_cols2+1)*nblk2+1;

    if (((prow2+np_rows2)%np_rows2) >= (brow2%np_rows2)){
      il = il+(ig-1)%nblk2;
      il = il-nblk2;
    }

    if (((pcol2+np_cols2)%np_cols2) >= (bcol2%np_cols2)){
      jl = jl+(jg-1)%nblk2;
      jl = jl-nblk2;
    }

    if      (loop==1){ *id11 = id2; *i11 = il; *j11 = jl; }
    else if (loop==2){ *id22 = id2; *i22 = il; *j22 = jl; }
    else if (loop==3){ *id12 = id2; *i12 = il; *j12 = jl; }
    else if (loop==4){ *id21 = id2; *i21 = il; *j21 = jl; }
  }
}

#endif
