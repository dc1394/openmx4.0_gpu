#ifndef OPENMX_VNA_CPU_H
#define OPENMX_VNA_CPU_H
#include "openmx_two_center.h"
#include "openmx_core_vna.h"
#include "openmx_core_atom_pairs.h"
#include "openmx_vna_tables.h"
#include "openmx_core_vna_matrix.h"

/* Internal host adapter shared by CUDA/HIP and both pair orientations.
   Basis and geometry are owned snapshots; Fourier/output tables are explicit borrowed views. */
static void SetPro_VNA23_CPU(const OpenMXDoubleBlockMatrix *HVNA23, int mirror, const double *GL_NormK,
    const OpenMXVnaPreparationBuffers *preparation,const OpenMXAtomPairBuffers *pairs,
    const SetProVnaTables *tables)
{
  const int OneD_Nloop=pairs->count,points=preparation->points;
  const int *OneD2Mc_AN=pairs->local,*OneD2h_AN=pairs->neighbor;
#pragma omp parallel
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
    dcomplex CY,CYt,CYp,CY1,CYt1,CYp1;
    dcomplex *TmpHNA;
    dcomplex *TmpHNAr;
    dcomplex *TmpHNAt;
    dcomplex *TmpHNAp;

    OpenMXVnaBuffers buffers;
    const OpenMXVnaShape shape = {preparation->basis_lmax,preparation->basis_multiplicity,preparation->points};
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

      Gc_AN=pairs->central_global[Nloop];
      Cwan=pairs->central_species[Nloop];
      Hwan=pairs->neighbor_species[Nloop];
      const int product_species=mirror?Hwan:Cwan;
      const int potential_species=mirror?Cwan:Hwan;
      const double *displacement=pairs->displacements+(size_t)Nloop*6+mirror*3;
      dx=displacement[0];dy=displacement[1];dz=displacement[2];

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

      kmin = tables->minimum_wavevector;
      kmax = tables->maximum_wavevector;
      Sk = kmax + kmin;
      Dk = kmax - kmin;

      for (L0=0; L0<=preparation->maximum_angular[product_species]; L0++){
	for (Mul0=0; Mul0<preparation->basis_counts[(size_t)(product_species)*preparation->angular_channels+(L0)]; Mul0++){
	  for (L1=0; L1<=preparation->maximum_angular[product_species]; L1++){
	    for (Mul1=0; Mul1<preparation->basis_counts[(size_t)(product_species)*preparation->angular_channels+(L1)]; Mul1++){
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
      Lmax_Four_Int = 2*preparation->maximum_angular[product_species];
      for (i=0; i<points; i++){
        Normk = GL_NormK[i];
        Spherical_Bessel2(Normk*r,Lmax_Four_Int,tmp_SphB,tmp_SphBp);
        for (LL=0; LL<=Lmax_Four_Int; LL++){
          SphB[LL*points+i] = tmp_SphB[LL];
          SphBp[LL*points+i] = tmp_SphBp[LL];
        }
      }

      /* loops for L0, Mul0, L1, and Mul1 */

      for (L0=0; L0<=preparation->maximum_angular[product_species]; L0++){
	for (Mul0=0; Mul0<preparation->basis_counts[(size_t)(product_species)*preparation->angular_channels+(L0)]; Mul0++){
	  for (L1=0; L1<=preparation->maximum_angular[product_species]; L1++){
	    for (Mul1=0; Mul1<preparation->basis_counts[(size_t)(product_species)*preparation->angular_channels+(L1)]; Mul1++){

	      if (L0<=L1){
		Lmax_Four_Int = 2*L1;

		/* sum over LL */

		for(LL=0; LL<=Lmax_Four_Int; LL++){

		  if (abs(L1-LL)<=L0 && L0<=(L1+LL) ){

		    sum  = 0.0;
		    sumr = 0.0;

		    /* Gauss-Legendre quadrature */

		    for (i=0; i<points; i++){

		      Normk = GL_NormK[i];

		      sj  =  SphB[LL*points+i];
		      sjp = SphBp[LL*points+i];

		      tmp0 = openmx_radial_row2(&tables->crude,potential_species)[i];
		      tmp1 = openmx_product_radial_row(&tables->products,product_species,L0,Mul0,L1,Mul1,LL)[i];
		      tmp10 = 0.50*Dk*tmp0*tmp1*tables->weights[i]*Normk*Normk;

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

      for (L0=0; L0<=preparation->maximum_angular[product_species]; L0++){
	for (Mul0=0; Mul0<preparation->basis_counts[(size_t)(product_species)*preparation->angular_channels+(L0)]; Mul0++){
	  for (L1=0; L1<=preparation->maximum_angular[product_species]; L1++){
	    for (Mul1=0; Mul1<preparation->basis_counts[(size_t)(product_species)*preparation->angular_channels+(L1)]; Mul1++){
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

      const OpenMXTwoCenterGeometry geometry={r,siT,coT,siP,coP,h_AN==0};
      num0=0;
      for (L0=0; L0<=preparation->maximum_angular[product_species]; ++L0){
        for (Mul0=0; Mul0<preparation->basis_counts[(size_t)product_species*preparation->angular_channels+L0]; ++Mul0){
          num1=0;
          for (L1=0; L1<=preparation->maximum_angular[product_species]; ++L1){
            for (Mul1=0; Mul1<preparation->basis_counts[(size_t)product_species*preparation->angular_channels+L1]; ++Mul1){
              OpenMX_TwoCenterTransform(&buffers.two_center,buffers.angular[0],
                  buffers.orbital_capacity,buffers.orbital_capacity,
                  openmx_core_orbital_index(buffers.multiplicity,L0,Mul0,0),
                  openmx_core_orbital_index(buffers.multiplicity,L1,Mul1,0),L0,L1,
                  tables->conversion,(size_t)preparation->basis_lmax+1,tables->conversion_width,&geometry);
              for (M0=0; M0<2*L0+1; ++M0)
                for (M1=0; M1<2*L1+1; ++M1)
                  for (int channel=0; channel<4; ++channel)
                    openmx_double_block_row(HVNA23,channel,Mc_AN,h_AN,num0+M0)[num1+M1]=
                        OpenMX_TwoCenterValue(&buffers.two_center,channel,M0,M1);
              num1+=2*L1+1;
            }
          }
          num0+=2*L0+1;
        }
      }

      dtime(&Etime_atom);
#pragma omp atomic update
      tables->atom_seconds[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    openmx_core_vna_destroy(workspace);

#pragma omp flush(HVNA23)

  } /* #pragma omp parallel */

}
#endif
