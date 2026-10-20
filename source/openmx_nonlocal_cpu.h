#ifndef OPENMX_NONLOCAL_CPU_H
#define OPENMX_NONLOCAL_CPU_H

/* Internal adapter shared by CUDA/HIP. Legacy validation and NLRF_BesselF
   declarations precede this include; the numerical owner has no globals. */
#include "openmx_core_nonlocal.h"

static OpenMXNonlocalWorkspace *SetNonlocal_Workspace(int points, const double *wavevectors,
    int cache_entries, OpenMXNonlocalBuffers *buffers)
{
  const OpenMXNonlocalShape shape = {List_YOUSO[25],List_YOUSO[24],List_YOUSO[30],
      SpeciesNum,Spe_Num_RVPS,Spe_VPS_List,points,wavevectors,
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

static void SetNonlocal_CalcDSNL_CPU(const OpenMXReciprocalViews *reciprocal, double ******DS_NL, const double *Normk_grid,
    int grid_dim, double grid_h, const int *OneD2Mc_AN, const int *OneD2h_AN,
    int OneD_Nloop, int bessel_cache_entries)
{
#pragma omp parallel shared(List_YOUSO,time_per_atom,DS_NL,Comp2Real,OneD_Grid,Spe_Num_RVPS,Spe_VPS_List,Spe_Num_Basis,Spe_MaxL_Basis,PAO_Nkmax,VPS_j_dependency,atv,Gxyz,WhatSpecies,ncn,natn,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop,Ngrid_NormK,reciprocal)
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

      dcomplex CsumNL0,CsumNLr,CsumNLt,CsumNLp;
      dcomplex Ctmp0,Ctmp1,Ctmp2,Cpow;
      dcomplex CY,CYt,CYp,CY1,CYt1,CYp1;
      dcomplex *TmpNL;
      dcomplex *TmpNLr;
      dcomplex *TmpNLt;
      dcomplex *TmpNLp;
      dcomplex *CmatNL0;
      dcomplex *CmatNLr;
      dcomplex *CmatNLt;
      dcomplex *CmatNLp;

    workspace = SetNonlocal_Workspace(grid_dim,Normk_grid,bessel_cache_entries,&buffers);
    const size_t conversion_columns = buffers.conversion_columns;
    TmpNL = buffers.angular[0];
    TmpNLr = buffers.angular[1];
    TmpNLt = buffers.angular[2];
    TmpNLp = buffers.angular[3];
    CmatNL0 = buffers.conversion[0];
    CmatNLr = buffers.conversion[1];
    CmatNLt = buffers.conversion[2];
    CmatNLp = buffers.conversion[3];
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
        for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
          for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
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
      if (Spe_MaxL_Basis[Cwan]<Lmax)
        Lmax_Four_Int = 2*Lmax;
      else
        Lmax_Four_Int = 2*Spe_MaxL_Basis[Cwan];

      OpenMXPairBesselTable bessel_table;
      if (!openmx_core_nonlocal_sample(workspace,r,Lmax_Four_Int,
          &bessel_table,bessel_error,sizeof(bessel_error))) {
        fprintf(stderr,"Set_Nonlocal: %s\n",bessel_error);
        abort();
      }
      SphB = bessel_table.values;
      SphBp = bessel_table.derivatives;

      for (so=0; so<=VPS_j_dependency[Hwan]; so++){

	/****************************************************
           Evaluate ovelap integrals <chi0|P> between PAOs
           and progectors of nonlocal potentials.
	****************************************************/
	/****************************************************
                \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3
	****************************************************/

	for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
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

	/* LL loop */

	for(LL=0; LL<=Lmax_Four_Int; LL++){

	  for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	    for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
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

	    for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){

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

	  /* derivatives of "on site" */
	  if (h_AN==0){
	    for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
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

	    for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
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

	num0 = 0;
	for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){

	    num1 = 0;
	    for (L=1; L<=projector_count; L++){
	      L1 = projector_angular[L];

	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

		  CsumNL0 = Complex(0.0,0.0);
		  CsumNLr = Complex(0.0,0.0);
		  CsumNLt = Complex(0.0,0.0);
		  CsumNLp = Complex(0.0,0.0);

		  for (k=-L0; k<=L0; k++){
                    const size_t pair_index = openmx_core_nonlocal_pair(&buffers,projector_offsets,L0,Mul0,L0+k,L,L1+M1);

		    Ctmp1 = Conjg(Comp2Real[L0][L0+M0][L0+k]);

		    /* S */

		    Ctmp0 = TmpNL[pair_index];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumNL0 = Cadd(CsumNL0,Ctmp2);

		    /* dS/dr */

		    Ctmp0 = TmpNLr[pair_index];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumNLr = Cadd(CsumNLr,Ctmp2);

		    /* dS/dt */

		    Ctmp0 = TmpNLt[pair_index];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumNLt = Cadd(CsumNLt,Ctmp2);

		    /* dS/dp */

		    Ctmp0 = TmpNLp[pair_index];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumNLp = Cadd(CsumNLp,Ctmp2);

		  }

		  CmatNL0[(size_t)(L0+M0)*conversion_columns+(L1+M1)] = CsumNL0;
		  CmatNLr[(size_t)(L0+M0)*conversion_columns+(L1+M1)] = CsumNLr;
		  CmatNLt[(size_t)(L0+M0)*conversion_columns+(L1+M1)] = CsumNLt;
		  CmatNLp[(size_t)(L0+M0)*conversion_columns+(L1+M1)] = CsumNLp;
		}
	      }

	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

		  CsumNL0 = Complex(0.0,0.0);
		  CsumNLr = Complex(0.0,0.0);
		  CsumNLt = Complex(0.0,0.0);
		  CsumNLp = Complex(0.0,0.0);

		  for (k=-L1; k<=L1; k++){

		    /* S */

		    Ctmp1 = Cmul(CmatNL0[(size_t)(L0+M0)*conversion_columns+(L1+k)],Comp2Real[L1][L1+M1][L1+k]);
		    CsumNL0 = Cadd(CsumNL0,Ctmp1);

		    /* dS/dr */

		    Ctmp1 = Cmul(CmatNLr[(size_t)(L0+M0)*conversion_columns+(L1+k)],Comp2Real[L1][L1+M1][L1+k]);
		    CsumNLr = Cadd(CsumNLr,Ctmp1);

		    /* dS/dt */

		    Ctmp1 = Cmul(CmatNLt[(size_t)(L0+M0)*conversion_columns+(L1+k)],Comp2Real[L1][L1+M1][L1+k]);
		    CsumNLt = Cadd(CsumNLt,Ctmp1);

		    /* dS/dp */

		    Ctmp1 = Cmul(CmatNLp[(size_t)(L0+M0)*conversion_columns+(L1+k)],Comp2Real[L1][L1+M1][L1+k]);
		    CsumNLp = Cadd(CsumNLp,Ctmp1);

		  }

		  DS_NL[so][0][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 8.0*CsumNL0.r;

		  if (h_AN!=0){

		    if (fabs(siT)<10e-14){

		      DS_NL[so][1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*coP*CsumNLr.r + coT*coP/r*CsumNLt.r);

		      DS_NL[so][2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*siP*CsumNLr.r + coT*siP/r*CsumNLt.r);

		      DS_NL[so][3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(coT*CsumNLr.r - siT/r*CsumNLt.r);

		    }

		    else{

		      DS_NL[so][1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*coP*CsumNLr.r + coT*coP/r*CsumNLt.r
			     - siP/siT/r*CsumNLp.r);

		      DS_NL[so][2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*siP*CsumNLr.r + coT*siP/r*CsumNLt.r
			     + coP/siT/r*CsumNLp.r);

		      DS_NL[so][3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(coT*CsumNLr.r - siT/r*CsumNLt.r);

		    }
		  }

		  else{
		    DS_NL[so][1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		    DS_NL[so][2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		    DS_NL[so][3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		  }

		}
	      }

	      num1 = num1 + 2*L1 + 1;
	    }

	    num0 = num0 + 2*L0 + 1;
	  }
	}
      } /* so */


      dtime(&Etime_atom);
#pragma omp atomic
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    openmx_core_nonlocal_destroy(workspace);

#pragma omp flush(DS_NL)

    } /* #pragma omp parallel */
  }

#endif
