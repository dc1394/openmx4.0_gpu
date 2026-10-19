#ifndef OPENMX_VNA_CPU_H
#define OPENMX_VNA_CPU_H
#include "openmx_core_vna.h"

/* Internal host adapter shared by CUDA/HIP and both pair orientations.
   Geometry, basis and output still borrow the legacy invocation's globals. */
static void SetPro_VNA23_CPU(double *****HVNA23, int mirror, const double *GL_NormK,
    int OneD_Nloop, const int *OneD2Mc_AN, const int *OneD2h_AN)
{
#pragma omp parallel shared(List_YOUSO,time_per_atom,HVNA23,Comp2Real,GL_Weight,Spe_ProductRF_Bessel,Spe_CrudeVNA_Bessel,GL_NormK,Spe_Num_Basis,Spe_MaxL_Basis,PAO_Nkmax,atv,Gxyz,WhatSpecies,ncn,natn,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop)
  {
    int OMPID,Nthrds,Nprocs,Nloop;
    int Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN;
    int Rnh,Hwan,L0,Mul0,L1,Mul1,M0,M1,LL;
    int Lmax_Four_Int,i,j,k,l,m,num0,num1;
    double dx,dy,dz;
    double siT,coT,siP,coP;
    double Normk,kmin,kmax,Sk,Dk;
    double gant,r,theta,phi;
    double SH[2],dSHt[2],dSHp[2];
    double S_coordinate[3];
    double Stime_atom,Etime_atom;
    double sum,sumr,sj,sjp,tmp0,tmp1,tmp10;
    double *SphB,*SphBp;
    double *tmp_SphB,*tmp_SphBp;

    dcomplex Ctmp0,Ctmp1,Ctmp2,Cpow;
    dcomplex Csum,Csumr,Csumt,Csump;
    dcomplex CsumS0,CsumSr,CsumSt,CsumSp;
    dcomplex CY,CYt,CYp,CY1,CYt1,CYp1;
    dcomplex *CmatS0;
    dcomplex *CmatSr;
    dcomplex *CmatSt;
    dcomplex *CmatSp;
    dcomplex *TmpHNA;
    dcomplex *TmpHNAr;
    dcomplex *TmpHNAt;
    dcomplex *TmpHNAp;

    OpenMXVnaBuffers buffers;
    const OpenMXVnaShape shape = {List_YOUSO[25],List_YOUSO[24],GL_Mesh};
    char error[256];
    OpenMXVnaWorkspace *workspace = openmx_core_vna_create(&shape,&buffers,error,sizeof(error));
    if (workspace == NULL) {
      fprintf(stderr,"Set_ProExpn_VNA workspace: %s\n",error);
      SetPro_gpu_abort("Cannot allocate neutral-potential workspace.");
      abort();
    }
    TmpHNA = buffers.angular[0];
    TmpHNAr = buffers.angular[1];
    TmpHNAt = buffers.angular[2];
    TmpHNAp = buffers.angular[3];
    CmatS0 = buffers.conversion[0];
    CmatSr = buffers.conversion[1];
    CmatSt = buffers.conversion[2];
    CmatSp = buffers.conversion[3];
    SphB = buffers.bessel[0];
    SphBp = buffers.bessel[1];
    tmp_SphB = buffers.scratch[0];
    tmp_SphBp = buffers.scratch[1];

    if (!mirror && omp_get_thread_num() == 0) {
      const size_t bytes = sizeof(dcomplex)*buffers.orbital_capacity*buffers.orbital_capacity;
      PrintMemory("Set_ProExpn_VNA: TmpHNA",bytes,NULL);
      PrintMemory("Set_ProExpn_VNA: TmpHNAr",bytes,NULL);
      PrintMemory("Set_ProExpn_VNA: TmpHNAt",bytes,NULL);
      PrintMemory("Set_ProExpn_VNA: TmpHNAp",bytes,NULL);
    }

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

      const int product_species = mirror ? Hwan : Cwan;
      const int potential_species = mirror ? Cwan : Hwan;
      if (mirror) {
        dx = Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rnh][1];
        dy = Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rnh][2];
        dz = Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rnh][3];
      } else {
        dx = Gxyz[Gh_AN][1] + atv[Rnh][1] - Gxyz[Gc_AN][1];
        dy = Gxyz[Gh_AN][2] + atv[Rnh][2] - Gxyz[Gc_AN][2];
        dz = Gxyz[Gh_AN][3] + atv[Rnh][3] - Gxyz[Gc_AN][3];
      }

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

      /****************************************************
         evaluate ovelap integrals <Phi_{LM,L'M'}|VNA>
         between VNA and PAO.
      ****************************************************/
      /****************************************************
       \int VNA(k)*Spe_ProductRF_Bessel(k)*jl(k*R) k^2 dk^3
      ****************************************************/

      kmin = Radial_kmin;
      kmax = PAO_Nkmax;
      Sk = kmax + kmin;
      Dk = kmax - kmin;

      for (L0=0; L0<=Spe_MaxL_Basis[product_species]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[product_species][L0]; Mul0++){
	  for (L1=0; L1<=Spe_MaxL_Basis[product_species]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[product_species][L1]; Mul1++){
	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){
		  TmpHNA[openmx_core_vna_pair(&buffers,L0,Mul0,L0+M0,L1,Mul1,L1+M1)] = Complex(0.0,0.0);
		  TmpHNAr[openmx_core_vna_pair(&buffers,L0,Mul0,L0+M0,L1,Mul1,L1+M1)] = Complex(0.0,0.0);
		  TmpHNAt[openmx_core_vna_pair(&buffers,L0,Mul0,L0+M0,L1,Mul1,L1+M1)] = Complex(0.0,0.0);
		  TmpHNAp[openmx_core_vna_pair(&buffers,L0,Mul0,L0+M0,L1,Mul1,L1+M1)] = Complex(0.0,0.0);
		}
	      }
	    }
	  }
	}
      }

      /* Reuse this worker's Bessel table; preserve the Gauss recurrence. */
      Lmax_Four_Int = 2*Spe_MaxL_Basis[product_species];
      for (i=0; i<GL_Mesh; i++){
        Normk = GL_NormK[i];
        Spherical_Bessel2(Normk*r,Lmax_Four_Int,tmp_SphB,tmp_SphBp);
        for (LL=0; LL<=Lmax_Four_Int; LL++){
          SphB[LL*GL_Mesh+i] = tmp_SphB[LL];
          SphBp[LL*GL_Mesh+i] = tmp_SphBp[LL];
        }
      }

      /* loops for L0, Mul0, L1, and Mul1 */

      for (L0=0; L0<=Spe_MaxL_Basis[product_species]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[product_species][L0]; Mul0++){
	  for (L1=0; L1<=Spe_MaxL_Basis[product_species]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[product_species][L1]; Mul1++){

	      if (L0<=L1){
		Lmax_Four_Int = 2*L1;

		/* sum over LL */

		for(LL=0; LL<=Lmax_Four_Int; LL++){

		  if (abs(L1-LL)<=L0 && L0<=(L1+LL) ){

		    sum  = 0.0;
		    sumr = 0.0;

		    /* Gauss-Legendre quadrature */

		    for (i=0; i<GL_Mesh; i++){

		      Normk = GL_NormK[i];

		      sj  =  SphB[LL*GL_Mesh+i];
		      sjp = SphBp[LL*GL_Mesh+i];

		      tmp0 = Spe_CrudeVNA_Bessel[potential_species][i];
		      tmp1 = Spe_ProductRF_Bessel[product_species][L0][Mul0][L1][Mul1][LL][i];
		      tmp10 = 0.50*Dk*tmp0*tmp1*GL_Weight[i]*Normk*Normk;

		      sum  += tmp10*sj;
		      sumr += tmp10*sjp*Normk;
		    }

		    /* sum over m */

		    for (M0=-L0; M0<=L0; M0++){
		      for (M1=-L1; M1<=L1; M1++){

			Csum = Complex(0.0,0.0);
			Csumr = Complex(0.0,0.0);
			Csumt = Complex(0.0,0.0);
			Csump = Complex(0.0,0.0);

			for(m=-LL; m<=LL; m++){

			  if ( (M1-m)==M0){

			    ComplexSH(LL,m,theta,phi,SH,dSHt,dSHp);
			    CY  = Complex(SH[0],SH[1]);
			    CYt = Complex(dSHt[0],dSHt[1]);
			    CYp = Complex(dSHp[0],dSHp[1]);

			    gant = pow(-1.0,(double)abs(m))*Gaunt(L0,M0,L1,M1,LL,-m);

			    /* S */

			    Ctmp2 = CRmul(CY,gant);
			    Csum = Cadd(Csum,Ctmp2);

			    /* dS/dt */

			    Ctmp2 = CRmul(CYt,gant);
			    Csumt = Cadd(Csumt,Ctmp2);

			    /* dS/dp */

			    Ctmp2 = CRmul(CYp,gant);
			    Csump = Cadd(Csump,Ctmp2);
			  }

			}

			/* S */
			Ctmp1 = CRmul(Csum,sum);
			TmpHNA[openmx_core_vna_pair(&buffers,L0,Mul0,L0+M0,L1,Mul1,L1+M1)]
			  = Cadd(TmpHNA[openmx_core_vna_pair(&buffers,L0,Mul0,L0+M0,L1,Mul1,L1+M1)],Ctmp1);

			/* dS/dr */
			Ctmp1 = CRmul(Csum,sumr);
			TmpHNAr[openmx_core_vna_pair(&buffers,L0,Mul0,L0+M0,L1,Mul1,L1+M1)]
			  = Cadd(TmpHNAr[openmx_core_vna_pair(&buffers,L0,Mul0,L0+M0,L1,Mul1,L1+M1)],Ctmp1);

			/* dS/dt */
			Ctmp1 = CRmul(Csumt,sum);
			TmpHNAt[openmx_core_vna_pair(&buffers,L0,Mul0,L0+M0,L1,Mul1,L1+M1)]
			  = Cadd(TmpHNAt[openmx_core_vna_pair(&buffers,L0,Mul0,L0+M0,L1,Mul1,L1+M1)],Ctmp1);

			/* dS/dp */
			Ctmp1 = CRmul(Csump,sum);
			TmpHNAp[openmx_core_vna_pair(&buffers,L0,Mul0,L0+M0,L1,Mul1,L1+M1)]
			  = Cadd(TmpHNAp[openmx_core_vna_pair(&buffers,L0,Mul0,L0+M0,L1,Mul1,L1+M1)],Ctmp1);
		      }
		    }
		  }
		} /* LL */
	      } /* if (L0<=L1) */
	    }
	  }
	}
      }

      /* copy the upper part to the lower part */

      for (L0=0; L0<=Spe_MaxL_Basis[product_species]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[product_species][L0]; Mul0++){
	  for (L1=0; L1<=Spe_MaxL_Basis[product_species]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[product_species][L1]; Mul1++){
	      if (L0<=L1){
		for (M0=-L0; M0<=L0; M0++){
		  for (M1=-L1; M1<=L1; M1++){

		    TmpHNA[openmx_core_vna_pair(&buffers,L1,Mul1,L1+M1,L0,Mul0,L0+M0)]
		      = Conjg(TmpHNA[openmx_core_vna_pair(&buffers,L0,Mul0,L0+M0,L1,Mul1,L1+M1)]);

		    TmpHNAr[openmx_core_vna_pair(&buffers,L1,Mul1,L1+M1,L0,Mul0,L0+M0)]
		      = Conjg(TmpHNAr[openmx_core_vna_pair(&buffers,L0,Mul0,L0+M0,L1,Mul1,L1+M1)]);

		    TmpHNAt[openmx_core_vna_pair(&buffers,L1,Mul1,L1+M1,L0,Mul0,L0+M0)]
		      = Conjg(TmpHNAt[openmx_core_vna_pair(&buffers,L0,Mul0,L0+M0,L1,Mul1,L1+M1)]);

		    TmpHNAp[openmx_core_vna_pair(&buffers,L1,Mul1,L1+M1,L0,Mul0,L0+M0)]
		      = Conjg(TmpHNAp[openmx_core_vna_pair(&buffers,L0,Mul0,L0+M0,L1,Mul1,L1+M1)]);

		  }
		}
	      }
	    }
	  }
	}
      }

      /****************************************************
                       Complex to Real
      ****************************************************/

      num0 = 0;
      for (L0=0; L0<=Spe_MaxL_Basis[product_species]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[product_species][L0]; Mul0++){

	  num1 = 0;
	  for (L1=0; L1<=Spe_MaxL_Basis[product_species]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[product_species][L1]; Mul1++){

	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

		  CsumS0 = Complex(0.0,0.0);
		  CsumSr = Complex(0.0,0.0);
		  CsumSt = Complex(0.0,0.0);
		  CsumSp = Complex(0.0,0.0);

		  for (k=-L0; k<=L0; k++){

		    Ctmp1 = Conjg(Comp2Real[L0][L0+M0][L0+k]);

		    /* S */

		    Ctmp0 = TmpHNA[openmx_core_vna_pair(&buffers,L0,Mul0,L0+k,L1,Mul1,L1+M1)];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumS0 = Cadd(CsumS0,Ctmp2);

		    /* dS/dr */

		    Ctmp0 = TmpHNAr[openmx_core_vna_pair(&buffers,L0,Mul0,L0+k,L1,Mul1,L1+M1)];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumSr = Cadd(CsumSr,Ctmp2);

		    /* dS/dt */

		    Ctmp0 = TmpHNAt[openmx_core_vna_pair(&buffers,L0,Mul0,L0+k,L1,Mul1,L1+M1)];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumSt = Cadd(CsumSt,Ctmp2);

		    /* dS/dp */

		    Ctmp0 = TmpHNAp[openmx_core_vna_pair(&buffers,L0,Mul0,L0+k,L1,Mul1,L1+M1)];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumSp = Cadd(CsumSp,Ctmp2);

		  }

		  CmatS0[(size_t)(L0+M0)*buffers.magnetic_capacity+(L1+M1)] = CsumS0;
		  CmatSr[(size_t)(L0+M0)*buffers.magnetic_capacity+(L1+M1)] = CsumSr;
		  CmatSt[(size_t)(L0+M0)*buffers.magnetic_capacity+(L1+M1)] = CsumSt;
		  CmatSp[(size_t)(L0+M0)*buffers.magnetic_capacity+(L1+M1)] = CsumSp;

		}
	      }

	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

		  CsumS0 = Complex(0.0,0.0);
		  CsumSr = Complex(0.0,0.0);
		  CsumSt = Complex(0.0,0.0);
		  CsumSp = Complex(0.0,0.0);

		  for (k=-L1; k<=L1; k++){

		    /* S */

		    Ctmp1 = Cmul(CmatS0[(size_t)(L0+M0)*buffers.magnetic_capacity+(L1+k)],Comp2Real[L1][L1+M1][L1+k]);
		    CsumS0 = Cadd(CsumS0,Ctmp1);

		    /* dS/dr */

		    Ctmp1 = Cmul(CmatSr[(size_t)(L0+M0)*buffers.magnetic_capacity+(L1+k)],Comp2Real[L1][L1+M1][L1+k]);
		    CsumSr = Cadd(CsumSr,Ctmp1);

		    /* dS/dt */

		    Ctmp1 = Cmul(CmatSt[(size_t)(L0+M0)*buffers.magnetic_capacity+(L1+k)],Comp2Real[L1][L1+M1][L1+k]);
		    CsumSt = Cadd(CsumSt,Ctmp1);

		    /* dS/dp */

		    Ctmp1 = Cmul(CmatSp[(size_t)(L0+M0)*buffers.magnetic_capacity+(L1+k)],Comp2Real[L1][L1+M1][L1+k]);
		    CsumSp = Cadd(CsumSp,Ctmp1);

		  }

		  HVNA23[0][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 8.0*CsumS0.r;

		  if (h_AN!=0){

		    if (fabs(siT)<10e-14){

		      HVNA23[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*coP*CsumSr.r + coT*coP/r*CsumSt.r);

		      HVNA23[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*siP*CsumSr.r + coT*siP/r*CsumSt.r);

		      HVNA23[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(coT*CsumSr.r - siT/r*CsumSt.r);
		    }

		    else{

		      HVNA23[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*coP*CsumSr.r + coT*coP/r*CsumSt.r
			     - siP/siT/r*CsumSp.r);

		      HVNA23[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*siP*CsumSr.r + coT*siP/r*CsumSt.r
			     + coP/siT/r*CsumSp.r);

		      HVNA23[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(coT*CsumSr.r - siT/r*CsumSt.r);
		    }
		  }
		  else{
		    HVNA23[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		    HVNA23[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		    HVNA23[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		  }

		}
	      }

	      num1 = num1 + 2*L1 + 1;
	    }
	  }

	  num0 = num0 + 2*L0 + 1;
	}
      }

      dtime(&Etime_atom);
#pragma omp atomic update
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    openmx_core_vna_destroy(workspace);

#pragma omp flush(HVNA23)

  } /* #pragma omp parallel */

}
#endif
