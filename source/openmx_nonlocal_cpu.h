#ifndef OPENMX_NONLOCAL_CPU_H
#define OPENMX_NONLOCAL_CPU_H

/* Internal adapter shared by CUDA/HIP. Legacy validation and NLRF_BesselF
   declarations precede this include; the numerical owner has no globals. */
#include "openmx_core_nonlocal.h"
#include "openmx_two_center.h"
#include "openmx_host_simd.h"
#include "openmx_core_vna_simd.h"
#include <stdlib.h>
#include <string.h>

static OpenMXNonlocalWorkspace *SetNonlocal_Workspace(const OpenMXAtomicViews *atomic, int points, const double *wavevectors,
    int cache_entries, OpenMXNonlocalBuffers *buffers)
{
  const OpenMXNonlocalShape shape = {List_YOUSO[25],List_YOUSO[24],List_YOUSO[30],
      SpeciesNum,atomic->species.radial_projectors,atomic->projector_angular,
      atomic->projector_angular_stride,points,wavevectors,
      cache_entries > 0 ? (size_t)cache_entries : 0};
  char error[256], message[512];
  OpenMXNonlocalWorkspace *owner = openmx_core_nonlocal_create(&shape,buffers,error,sizeof(error));
  if (owner == NULL) {
    snprintf(message,sizeof(message),"Set_Nonlocal workspace: %s",error);
    SetNonlocal_AbortWithMessage(message);
    abort();
  }
  return owner;
}

/* Shared CPU and GPU-radial postprocessing. No legacy global input is
   read here; each worker borrows its own C++ numerical workspace. */
static void SetNonlocal_TransformDSNL(const OpenMXAtomicViews *atomic,
    const OpenMXNonlocalBuffers *buffers,int central_species,int neighbor_species,
    int so,int local,int neighbor,double ******matrix,const dcomplex *conversion,
    size_t conversion_orders,size_t conversion_width,const OpenMXTwoCenterGeometry *geometry)
{
  const size_t species_offset=buffers->species_offsets[neighbor_species];
  const size_t *offsets=buffers->projector_offsets+species_offset;
  const int *angular=buffers->projector_angular+species_offset;
  int row=0;
  for (int l0=0; l0<=atomic->species.basis_lmax[central_species]; ++l0){
    for (int radial=0; radial<atomic->species.basis[(size_t)central_species*atomic->species.angular_channels+l0]; ++radial){
      int column=0;
      for (int p=1; p<=buffers->projector_counts[neighbor_species]; ++p){
        const int l1=angular[p];
        OpenMX_TwoCenterTransform(&buffers->two_center,buffers->angular[0],
            buffers->orbital_capacity,buffers->projector_capacity,
            openmx_core_orbital_index(buffers->multiplicity,l0,radial,0),offsets[p],l0,l1,
            conversion,conversion_orders,conversion_width,geometry);
        for (int m0=0; m0<2*l0+1; ++m0)
          for (int m1=0; m1<2*l1+1; ++m1)
            for (int channel=0; channel<4; ++channel)
              matrix[so][channel][local][neighbor][row+m0][column+m1]=
                  OpenMX_TwoCenterValue(&buffers->two_center,channel,m0,m1);
        column+=2*l1+1;
      }
      row+=2*l0+1;
    }
  }
}

/* The host SIMD path of one pair's radial sums: lanes across the (basis
   function, projector) pairs and every order, the legacy per-element order,
   so the sums are bitwise those of the C loop of SetNonlocal_CalcDSNL_CPU
   (openmx_core_vna_nonlocal_radial). The rows of a species pair are
   transposed to [point][combo] once per (species pair, spin-orbit index), in
   tables owned by the calling thread; the sums of every order land in
   sums[2][orders][combos] (S, then Sr) with the combos in the legacy
   (L0, Mul0, L) order. Returns 0 under OPENMX_HOST_SIMD=legacy or scalar (the
   C loop shares each basis row's products across the projectors, which one
   lane cannot: the scalar Core kernel runs at 0.86x of it), when a table
   cannot be allocated, or when the Core refuses; the caller then runs the C
   loop. Rank-local. */
typedef struct {
  double *rows;        /* [2][points][combos]: basis rows, then projector rows */
  double *sums;        /* [2][orders][combos] */
  size_t row_capacity, sum_capacity;  /* doubles */
  int Cwan, Hwan, combos;             /* the species pair the rows hold; -1: none */
} SetNonlocalHostRadial;

/* Releases one thread's tables. The members are read through a pointer: nvc
   26.9 (nvc and nvc++) miscompiles a read of a pointer member at a constant
   index of an array declared inside an OpenMP region -- host_radial[0].rows
   evaluates to the member's address, and free() aborts. */
static void SetNonlocal_RadialHostRelease(SetNonlocalHostRadial *host)
{
  free(host->rows);
  free(host->sums);
}

static int SetNonlocal_RadialHost(SetNonlocalHostRadial *host, int mode, const OpenMXAtomicViews *atomic,
    int Cwan, int Hwan, int projector_count, const double *rf_cache, const double *nlrf_cache,
    const double *Normk_grid, int grid_dim, double h, const double *SphB, const double *SphBp,
    int orders, int onsite)
{
  int combos = 0;
  if (mode==OPENMX_HOST_SIMD_LEGACY || mode==OPENMX_GRID_SCALAR) return 0;
  if (grid_dim<=0 || orders<=0 || projector_count<=0) return 0;
  for (int L0=0; L0<=atomic->species.basis_lmax[Cwan]; L0++)
    combos += atomic->species.basis[(size_t)(Cwan)*atomic->species.angular_channels+(L0)];
  combos *= projector_count;
  if (combos<=0) return 0;
  const size_t points = (size_t)grid_dim, n = (size_t)combos;
  if (host->Cwan!=Cwan || host->Hwan!=Hwan || host->combos!=combos){
    if (2*points*n>host->row_capacity){
      double *grown = (double*)realloc(host->rows,2*points*n*sizeof(double));
      if (grown==NULL) return 0;
      host->rows = grown;
      host->row_capacity = 2*points*n;
    }
    /* point-major, so the writes stream (the source rows stay cached) */
    for (size_t i=0; i<points; i++){
      double *row0 = host->rows+i*n, *row1 = host->rows+(points+i)*n;
      size_t c = 0;
      for (int L0=0; L0<=atomic->species.basis_lmax[Cwan]; L0++){
        for (int Mul0=0; Mul0<atomic->species.basis[(size_t)(Cwan)*atomic->species.angular_channels+(L0)]; Mul0++){
          const double basis_value = rf_cache[(size_t)(L0*List_YOUSO[24]+Mul0)*points+i];
          for (int L=1; L<=projector_count; L++,c++){
            row0[c] = basis_value;
            row1[c] = nlrf_cache[(size_t)L*points+i];
          }
        }
      }
    }
    host->Cwan = Cwan;
    host->Hwan = Hwan;
    host->combos = combos;
  }
  if (2*(size_t)orders*n>host->sum_capacity){
    double *grown = (double*)realloc(host->sums,2*(size_t)orders*n*sizeof(double));
    if (grown==NULL) return 0;
    host->sums = grown;
    host->sum_capacity = 2*(size_t)orders*n;
  }
  OpenMXVnaNonlocalRadial q = {0};
  q.points = grid_dim; q.orders = orders; q.combos = combos; q.onsite = onsite;
  q.h = h; q.k = Normk_grid; q.rft = host->rows; q.nlrft = host->rows+points*n; q.table_stride = n;
  q.sb = SphB; q.sbp = SphBp; q.sb_stride = points;
  q.out[0] = host->sums; q.out[1] = host->sums+(size_t)orders*n; q.out_stride = n;
  return openmx_core_vna_nonlocal_radial(&q,mode);
}

static void SetNonlocal_CalcDSNL_CPU(const OpenMXAtomicViews *atomic, const OpenMXReciprocalViews *reciprocal, double ******DS_NL, const double *Normk_grid,
    int grid_dim, double grid_h, const int *OneD2Mc_AN, const int *OneD2h_AN,
    int OneD_Nloop, int bessel_cache_entries)
{
  const int host_simd_mode = OpenMX_HostSimdMode();
#pragma omp parallel shared(List_YOUSO,time_per_atom,DS_NL,Comp2Real,OneD_Grid,PAO_Nkmax,atv,Gxyz,WhatSpecies,ncn,natn,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop,Ngrid_NormK,reciprocal,atomic)
    {

      int OMPID,Nthrds,Nprocs,Nloop;
      int Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN;
      int Rnh,Hwan,so,i,j,k,l,m;
      int LL,L0,L1,L,Mul0,M0,M1,num0,num1;
      int Lmax,Lmax_Four_Int,Ls;

      double dx,dy,dz,r;
      double S_coordinate[3];
      double siT,coT,siP,coP;
      double gant,theta,phi;
      double SH[2],dSHt[2],dSHp[2];
      double Stime_atom, Etime_atom;
      double Normk,kmin,kmax;
      double Bessel_Pro0,Bessel_Pro1;
      double h,coe0,sj,sjp;
      double tmp0,tmp1,tmp2,tmp3,tmp4;
      const double *SphB,*SphBp;
      OpenMXNonlocalWorkspace *workspace;
      OpenMXNonlocalBuffers buffers;
      char bessel_error[256];
      double *RF_BesselCache,*NLRF_BesselCache[2];
      double *SumNL0;
      double *SumNLr0;
      int cached_Cwan,cached_Hwan[2];

      dcomplex Ctmp2,Cpow;
      dcomplex CY,CYt,CYp,CY1,CYt1,CYp1;
      dcomplex *TmpNL;
      dcomplex *TmpNLr;
      dcomplex *TmpNLt;
      dcomplex *TmpNLp;
      SetNonlocalHostRadial host_radial[2];

    memset(host_radial,0,sizeof(host_radial));
    host_radial[0].Cwan = host_radial[1].Cwan = -1;
    workspace = SetNonlocal_Workspace(atomic, grid_dim,Normk_grid,bessel_cache_entries,&buffers);
    TmpNL = buffers.angular[0];
    TmpNLr = buffers.angular[1];
    TmpNLt = buffers.angular[2];
    TmpNLp = buffers.angular[3];
    SumNL0 = buffers.sums[0];
    SumNLr0 = buffers.sums[1];
    RF_BesselCache = buffers.radial;
    NLRF_BesselCache[0] = buffers.nonlocal_radial[0];
    NLRF_BesselCache[1] = buffers.nonlocal_radial[1];

    cached_Cwan = -1;
    cached_Hwan[0] = -1;
    cached_Hwan[1] = -1;

    /* get info. on OpenMP */

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    /* one-dimensionalized loop */

    for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

      dtime(&Stime_atom);

      /* get Mc_AN and h_AN */

      Mc_AN = OneD2Mc_AN[Nloop];
      h_AN  = OneD2h_AN[Nloop];

      /* set data on Mc_AN */

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];

      /* set data on h_AN */

      Gh_AN = natn[Gc_AN][h_AN];
      Rnh = ncn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      const size_t *projector_offsets = buffers.projector_offsets+buffers.species_offsets[Hwan];
      const int *projector_angular = buffers.projector_angular+buffers.species_offsets[Hwan];
      const int projector_count = buffers.projector_counts[Hwan];

      dx = Gxyz[Gh_AN][1] + atv[Rnh][1] - Gxyz[Gc_AN][1];
      dy = Gxyz[Gh_AN][2] + atv[Rnh][2] - Gxyz[Gc_AN][2];
      dz = Gxyz[Gh_AN][3] + atv[Rnh][3] - Gxyz[Gc_AN][3];

      xyz2spherical(dx,dy,dz,0.0,0.0,0.0,S_coordinate);
      r     = S_coordinate[0];
      theta = S_coordinate[1];
      phi   = S_coordinate[2];

      /* for empty atoms or finite elemens basis */

      if (r<1.0e-10) r = 1.0e-10;

      /* precalculation of sin and cos */

      siT = sin(theta);
      coT = cos(theta);
      siP = sin(phi);
      coP = cos(phi);

      kmin = Radial_kmin;
      kmax = PAO_Nkmax;
      h = grid_h;

      if (cached_Cwan != Cwan){
        for (L0=0; L0<=atomic->species.basis_lmax[Cwan]; L0++){
          for (Mul0=0; Mul0<atomic->species.basis[(size_t)(Cwan)*atomic->species.angular_channels+(L0)]; Mul0++){
            int rf_offset;

            rf_offset = (L0*List_YOUSO[24] + Mul0)*grid_dim;
            for (i=0; i<grid_dim; i++){
              RF_BesselCache[rf_offset + i] = RF_BesselF(reciprocal, Cwan,L0,Mul0,Normk_grid[i]);
            }
          }
        }
        cached_Cwan = Cwan;
      }

      Lmax = -10;
      for (L=1; L<=projector_count; L++){
        if (Lmax<projector_angular[L]) Lmax = projector_angular[L];
      }
      if (atomic->species.basis_lmax[Cwan]<Lmax)
        Lmax_Four_Int = 2*Lmax;
      else
        Lmax_Four_Int = 2*atomic->species.basis_lmax[Cwan];

      OpenMXPairBesselTable bessel_table;
      if (!openmx_core_nonlocal_sample(workspace,r,Lmax_Four_Int,
          &bessel_table,bessel_error,sizeof(bessel_error))) {
        fprintf(stderr,"Set_Nonlocal: %s\n",bessel_error);
        abort();
      }
      SphB = bessel_table.values;
      SphBp = bessel_table.derivatives;

      for (so=0; so<=atomic->species.relativistic[Hwan]; so++){

	/****************************************************
           Evaluate ovelap integrals <chi0|P> between PAOs
           and progectors of nonlocal potentials.
	****************************************************/
	/****************************************************
                \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3
	****************************************************/

	for (L0=0; L0<=atomic->species.basis_lmax[Cwan]; L0++){
	  for (Mul0=0; Mul0<atomic->species.basis[(size_t)(Cwan)*atomic->species.angular_channels+(L0)]; Mul0++){
	    for (L=1; L<=projector_count; L++){
	      L1 = projector_angular[L];
	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){
		  const size_t pair_index = openmx_core_nonlocal_pair(&buffers,projector_offsets,L0,Mul0,L0+M0,L,L1+M1);
		  TmpNL[pair_index]  = Complex(0.0,0.0);
		  TmpNLr[pair_index] = Complex(0.0,0.0);
		  TmpNLt[pair_index] = Complex(0.0,0.0);
		  TmpNLp[pair_index] = Complex(0.0,0.0);
		}
	      }
	    }
	  }
	}

        if (cached_Hwan[so] != Hwan){
          for (L=1; L<=projector_count; L++){
            int nlrf_offset;

            nlrf_offset = L*grid_dim;
            for (i=0; i<grid_dim; i++){
              NLRF_BesselCache[so][nlrf_offset + i] = NLRF_BesselF(reciprocal, Hwan,L,so,Normk_grid[i]);
            }
          }
          cached_Hwan[so] = Hwan;
        }

	/* the radial sums of every order on the host SIMD path (rank-local) */
	const int host_sums = SetNonlocal_RadialHost(&host_radial[so],host_simd_mode,atomic,Cwan,Hwan,
	    projector_count,RF_BesselCache,NLRF_BesselCache[so],Normk_grid,grid_dim,h,SphB,SphBp,
	    Lmax_Four_Int+1,h_AN==0);

	/* LL loop */

	for(LL=0; LL<=Lmax_Four_Int; LL++){

	  if (host_sums){
	    /* this order's sums from the host SIMD path, in the legacy (L0, Mul0, L) order */
	    const size_t combos = (size_t)host_radial[so].combos;
	    const double *host_S = host_radial[so].sums+(size_t)LL*combos;
	    const double *host_Sr = host_radial[so].sums+(size_t)(Lmax_Four_Int+1+LL)*combos;
	    size_t c = 0;
	    for (L0=0; L0<=atomic->species.basis_lmax[Cwan]; L0++){
	      for (Mul0=0; Mul0<atomic->species.basis[(size_t)(Cwan)*atomic->species.angular_channels+(L0)]; Mul0++){
		for (L=1; L<=projector_count; L++,c++){
		  SumNL0[openmx_core_nonlocal_sum(&buffers,L0,Mul0,L)] = host_S[c];
		  SumNLr0[openmx_core_nonlocal_sum(&buffers,L0,Mul0,L)] = host_Sr[c];
		}
	      }
	    }
	  }

	  else {
	  for (L0=0; L0<=atomic->species.basis_lmax[Cwan]; L0++){
	    for (Mul0=0; Mul0<atomic->species.basis[(size_t)(Cwan)*atomic->species.angular_channels+(L0)]; Mul0++){
	      for (L=1; L<=projector_count; L++){
		SumNL0[openmx_core_nonlocal_sum(&buffers,L0,Mul0,L)] = 0.0;
		SumNLr0[openmx_core_nonlocal_sum(&buffers,L0,Mul0,L)] = 0.0;
	      }
	    }
	  }

	  for (i=0; i<grid_dim; i++){

	    if (i==0 || i==(grid_dim-1)) coe0 = 0.50;
	    else                      coe0 = 1.00;

	    Normk = Normk_grid[i];

	    sj  =  SphB[LL*grid_dim + i];
	    sjp = SphBp[LL*grid_dim + i];

	    for (L0=0; L0<=atomic->species.basis_lmax[Cwan]; L0++){
	      for (Mul0=0; Mul0<atomic->species.basis[(size_t)(Cwan)*atomic->species.angular_channels+(L0)]; Mul0++){

		Bessel_Pro0 = RF_BesselCache[(L0*List_YOUSO[24] + Mul0)*grid_dim + i];
		tmp0 = coe0*h*Normk*Normk*Bessel_Pro0;
		tmp1 = tmp0*sj;
		tmp2 = tmp0*Normk*sjp;

		for (L=1; L<=projector_count; L++){

		  Bessel_Pro1 = NLRF_BesselCache[so][L*grid_dim + i];

		  tmp3 = tmp1*Bessel_Pro1;
		  tmp4 = tmp2*Bessel_Pro1;
		  SumNL0[openmx_core_nonlocal_sum(&buffers,L0,Mul0,L)]  += tmp3;
		  SumNLr0[openmx_core_nonlocal_sum(&buffers,L0,Mul0,L)] += tmp4;
		}
	      }
	    }
	  }
	  } /* the C loop */

	  /* derivatives of "on site" */
	  if (h_AN==0){
	    for (L0=0; L0<=atomic->species.basis_lmax[Cwan]; L0++){
	      for (Mul0=0; Mul0<atomic->species.basis[(size_t)(Cwan)*atomic->species.angular_channels+(L0)]; Mul0++){
		for (L=1; L<=projector_count; L++){
		  SumNLr0[openmx_core_nonlocal_sum(&buffers,L0,Mul0,L)] = 0.0;
		}
	      }
	    }
	  }

	  /****************************************************
            For "overlap",
            sum_m 8*(-i)^{-L0+L1+l}*
                  C_{L0,-M0,L1,M1,l,m}*
                  \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3,
	  ****************************************************/

	  for(m=-LL; m<=LL; m++){

	    ComplexSH(LL,m,theta,phi,SH,dSHt,dSHp);
	    SH[1]   = -SH[1];
	    dSHt[1] = -dSHt[1];
	    dSHp[1] = -dSHp[1];

	    CY  = Complex(SH[0],SH[1]);
	    CYt = Complex(dSHt[0],dSHt[1]);
	    CYp = Complex(dSHp[0],dSHp[1]);

	    for (L0=0; L0<=atomic->species.basis_lmax[Cwan]; L0++){
	      for (Mul0=0; Mul0<atomic->species.basis[(size_t)(Cwan)*atomic->species.angular_channels+(L0)]; Mul0++){
		for (L=1; L<=projector_count; L++){

		  L1 = projector_angular[L];
		  Ls = -L0 + L1 + LL;

		  if (abs(L1-LL)<=L0 && L0<=(L1+LL) ){

		    Cpow = Im_pow(-1,Ls);
		    CY1  = Cmul(Cpow,CY);
		    CYt1 = Cmul(Cpow,CYt);
		    CYp1 = Cmul(Cpow,CYp);

		    for (M0=-L0; M0<=L0; M0++){

		      M1 = M0 - m;

		      if (abs(M1)<=L1){
                        const size_t pair_index = openmx_core_nonlocal_pair(&buffers,projector_offsets,L0,Mul0,L0+M0,L,L1+M1);

			gant = Gaunt(L0,M0,L1,M1,LL,m);

			/* S */

			tmp0 = gant*SumNL0[openmx_core_nonlocal_sum(&buffers,L0,Mul0,L)];
			Ctmp2 = CRmul(CY1,tmp0);
			TmpNL[pair_index] =
			  Cadd(TmpNL[pair_index],Ctmp2);

			/* dS/dr */

			tmp0 = gant*SumNLr0[openmx_core_nonlocal_sum(&buffers,L0,Mul0,L)];
			Ctmp2 = CRmul(CY1,tmp0);
			TmpNLr[pair_index] =
			  Cadd(TmpNLr[pair_index],Ctmp2);

			/* dS/dt */

			tmp0 = gant*SumNL0[openmx_core_nonlocal_sum(&buffers,L0,Mul0,L)];
			Ctmp2 = CRmul(CYt1,tmp0);
			TmpNLt[pair_index] =
			  Cadd(TmpNLt[pair_index],Ctmp2);

			/* dS/dp */

			tmp0 = gant*SumNL0[openmx_core_nonlocal_sum(&buffers,L0,Mul0,L)];
			Ctmp2 = CRmul(CYp1,tmp0);
			TmpNLp[pair_index] =
			  Cadd(TmpNLp[pair_index],Ctmp2);

		      }
		    }
		  }

		}
	      }
	    }
	  }
	} /* LL */

	/****************************************************
                           complex to real
	****************************************************/

        const OpenMXTwoCenterGeometry geometry={r,siT,coT,siP,coP,h_AN==0};
        SetNonlocal_TransformDSNL(atomic,&buffers,Cwan,Hwan,so,Mc_AN,h_AN,DS_NL,
            &Comp2Real[0][0][0], sizeof(Comp2Real)/sizeof(Comp2Real[0]), sizeof(Comp2Real[0][0])/sizeof(Comp2Real[0][0][0]),&geometry);
      } /* so */


      dtime(&Etime_atom);
#pragma omp atomic
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    SetNonlocal_RadialHostRelease(&host_radial[0]);
    SetNonlocal_RadialHostRelease(&host_radial[1]);
    openmx_core_nonlocal_destroy(workspace);

#pragma omp flush(DS_NL)

    } /* #pragma omp parallel */
  }

#endif
