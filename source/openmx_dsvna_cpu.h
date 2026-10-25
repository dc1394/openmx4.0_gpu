#ifndef OPENMX_DSVNA_CPU_H
#define OPENMX_DSVNA_CPU_H
#include "openmx_core_dsvna.h"
#include "openmx_core_atom_pairs.h"
#include "openmx_vna_tables.h"
#include "openmx_projector_matrix.h"
#include "openmx_host_simd.h"
#include "openmx_core_vna_simd.h"

/* Gaunt coefficients Gaunt(L0,M0,L1,M1,LL,M0-M1) tabulated once per call:
   the m sums of the projector expansions only ever ask for M0-M1, so the
   table replaces a pure function whose factorial loops dominated the m
   loops. Values are the function's own, bitwise. */
typedef struct { int l0max,l1max,llmax; size_t n1; double *values; } SetProGauntTable;

static void SetPro_GauntTable_Build(SetProGauntTable *t,int l0max,int l1max,int llmax)
{
  const size_t n0=(size_t)(l0max+1)*(l0max+1);
  int L0,M0,L1,M1,LL;
  t->l0max=l0max;t->l1max=l1max;t->llmax=llmax;t->n1=(size_t)(l1max+1)*(l1max+1);
  t->values=(double*)SetPro_checked_malloc(sizeof(double)*n0*t->n1*(size_t)(llmax+1));
  for (L0=0; L0<=l0max; L0++) for (M0=-L0; M0<=L0; M0++)
    for (L1=0; L1<=l1max; L1++) for (M1=-L1; M1<=L1; M1++)
      for (LL=0; LL<=llmax; LL++)
        t->values[(((size_t)(L0*(L0+1)+M0))*t->n1+(size_t)(L1*(L1+1)+M1))*(size_t)(llmax+1)+LL]
          = Gaunt(L0,M0,L1,M1,LL,M0-M1);
}
static inline double SetPro_GauntTable_At(const SetProGauntTable *t,int L0,int M0,int L1,int M1,int LL)
{
  return t->values[(((size_t)(L0*(L0+1)+M0))*t->n1+(size_t)(L1*(L1+1)+M1))*(size_t)(t->llmax+1)+LL];
}
static void SetPro_GauntTable_Free(SetProGauntTable *t) { free(t->values); t->values=NULL; }

/* Read-only inputs of the Core's host kernels for DS_VNA, shared by all
   workers: the projector table transposed to [point][projector] per species
   (padded to a multiple of eight rows, zero padding) and the Gaunt table.
   A legacy mode leaves the tables empty and every loop below unchanged. */
typedef struct {
  int mode, points, projectors;
  size_t table_stride;
  double *projector_table;   /* [species][points][table_stride] */
  SetProGauntTable gaunt;
} SetProDsvnaHost;

static void SetPro_DsvnaHost_Build(SetProDsvnaHost *host,const OpenMXVnaPreparationBuffers *preparation,
    const SetProVnaTables *tables)
{
  const int species=preparation->species,points=preparation->points,projectors=preparation->projectors;
  int spe,L,L1,Mul1,i,Lmax=0,l0max=0,llmax=0;
  memset(host,0,sizeof(*host));
  host->mode=OpenMX_HostSimdMode();
  if (host->mode==OPENMX_HOST_SIMD_LEGACY) return;
  host->points=points;host->projectors=projectors;
  host->table_stride=((size_t)projectors+7)/8*8;
  /* the flat projector index of the sums is L1*multiplicity+Mul1; otherwise keep the loops */
  if ((preparation->projector_lmax+1)*preparation->projector_multiplicity==projectors && projectors>0) {
    host->projector_table=(double*)SetPro_checked_malloc(sizeof(double)*((size_t)species*points*host->table_stride+1));
    memset(host->projector_table,0,sizeof(double)*(size_t)species*points*host->table_stride);
    for (spe=0; spe<species; spe++){
      double *table=host->projector_table+(size_t)spe*points*host->table_stride;
      L=0;
      for (L1=0; L1<=preparation->projector_lmax; L1++){
        for (Mul1=0; Mul1<preparation->projector_multiplicity; Mul1++,L++){
          const double *row=openmx_radial_row4(&tables->projectors,spe,L1,Mul1);
          for (i=0; i<points; i++) table[(size_t)i*host->table_stride+L]=row[i];
        }
      }
    }
  }
  for (L=0; L<projectors; L++) if (Lmax<preparation->projector_angular[L]) Lmax=preparation->projector_angular[L];
  for (spe=0; spe<species; spe++){
    const int lfi=preparation->maximum_angular[spe]<Lmax?2*Lmax:2*preparation->maximum_angular[spe];
    if (l0max<preparation->maximum_angular[spe]) l0max=preparation->maximum_angular[spe];
    if (llmax<lfi) llmax=lfi;
  }
  SetPro_GauntTable_Build(&host->gaunt,l0max,preparation->projector_lmax,llmax);
}
static void SetPro_DsvnaHost_Free(SetProDsvnaHost *host)
{
  if (host->mode==OPENMX_HOST_SIMD_LEGACY) return;
  free(host->projector_table);host->projector_table=NULL;
  SetPro_GauntTable_Free(&host->gaunt);
}

/* Shared CPU projector assembly. Preparation outlives all workers; only
   Fourier values and output retain explicit borrowed views; geometry is owned. */
static void SetPro_DSVNA_CPU(const OpenMXProjectorMatrix *DS_VNA, const OpenMXVnaPreparation *owner,
    const OpenMXVnaPreparationBuffers *preparation, const double *GL_NormK,
    const OpenMXAtomPairBuffers *pairs,const SetProVnaTables *tables,
    double *time1, double *time2, double *time3)
{
  const int points = preparation->points, Num_RVNA = preparation->projectors;
  const int OneD_Nloop=pairs->count;
  const int *OneD2Mc_AN=pairs->local,*OneD2h_AN=pairs->neighbor;
  const int *VNA_List = preparation->projector_angular;
  const double *Bessel_Pro00 = preparation->weighted[0], *Bessel_Pro01 = preparation->weighted[1];
  SetProDsvnaHost host;
  SetPro_DsvnaHost_Build(&host,preparation,tables);
#pragma omp parallel

  {
    int i,j,k,l,m,num0,num1;
    int Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN;
    int Rnh,Hwan,L0,Mul0,L,L1,M0,M1,LL,Mul1,Ls;
    int OMPID,Nthrds,Nprocs,Nloop;
    int Lmax,Lmax_Four_Int;
    double Stime_atom,Etime_atom;
    double dx,dy,dz;
    double gant,SH[2],dSHt[2],dSHp[2];
    double S_coordinate[3];
    double r,theta,phi;
    double Normk,tmp0,tmp1,tmp2;
    double siT,coT,siP,coP;
    double *SumNL0;
    double *SumNLr0;
    double *Bes00;
    double *Bes01;
    double *tmp_SphB;
    double *tmp_SphBp;
    double *SphB;
    double *SphBp;
    double stime,etime;

    dcomplex Ctmp1,Ctmp0,Ctmp2;
    dcomplex CsumNL0,CsumNLr,CsumNLt,CsumNLp;
    dcomplex CY,CYt,CYp,CY1,CYt1,CYp1,Cpow;
    dcomplex *TmpNL;
    dcomplex *TmpNLr;
    dcomplex *TmpNLt;
    dcomplex *TmpNLp;
    dcomplex *CmatNL0;
    dcomplex *CmatNLr;
    dcomplex *CmatNLt;
    dcomplex *CmatNLp;

    OpenMXDsvnaBuffers buffers;
    char error[256];
    OpenMXDsvnaWorkspace *workspace = openmx_core_dsvna_create(owner,&buffers,error,sizeof(error));
    if (workspace == NULL) {
      fprintf(stderr,"DS_VNA workspace: %s\n",error);
      SetPro_gpu_abort("Cannot allocate neutral-projector workspace.");
      abort();
    }
    TmpNL = buffers.angular[0]; TmpNLr = buffers.angular[1];
    TmpNLt = buffers.angular[2]; TmpNLp = buffers.angular[3];
    CmatNL0 = buffers.conversion[0]; CmatNLr = buffers.conversion[1];
    CmatNLt = buffers.conversion[2]; CmatNLp = buffers.conversion[3];
    SumNL0 = buffers.sums[0]; SumNLr0 = buffers.sums[1];
    Bes00 = buffers.radial[0]; Bes01 = buffers.radial[1];
    SphB = buffers.bessel[0]; SphBp = buffers.bessel[1];
    tmp_SphB = buffers.scratch[0]; tmp_SphBp = buffers.scratch[1];
    /* host SIMD path: Bessel arguments and the radial-function rows of a pair */
    double *simd_x = NULL, *simd_w = NULL;
    const double **simd_a0 = NULL, **simd_a1 = NULL, **simd_w0 = NULL, **simd_w1 = NULL;
    if (host.mode!=OPENMX_HOST_SIMD_LEGACY) {
      simd_x = (double*)SetPro_checked_malloc(sizeof(double)*(size_t)points);
      simd_w = (double*)SetPro_checked_malloc(sizeof(double)*(2*buffers.radial_capacity*(size_t)points+1));
      simd_a0 = (const double**)SetPro_checked_malloc(sizeof(double*)*(buffers.radial_capacity+1));
      simd_a1 = (const double**)SetPro_checked_malloc(sizeof(double*)*(buffers.radial_capacity+1));
      simd_w0 = (const double**)SetPro_checked_malloc(sizeof(double*)*(buffers.radial_capacity+1));
      simd_w1 = (const double**)SetPro_checked_malloc(sizeof(double*)*(buffers.radial_capacity+1));
      for (i=0; i<(int)buffers.radial_capacity; i++){
        simd_w0[i] = simd_w+(size_t)(2*i)*points;
        simd_w1[i] = simd_w+(size_t)(2*i+1)*points;
      }
    }
    if (omp_get_thread_num() == 0) {
      const size_t radial_bytes = sizeof(double)*buffers.radial_capacity*buffers.projectors;
      const size_t angular_bytes = sizeof(dcomplex)*buffers.orbital_capacity*buffers.projector_capacity;
      PrintMemory("Set_ProExpn_VNA: SumNL0",radial_bytes,NULL);
      PrintMemory("Set_ProExpn_VNA: SumNLr0",radial_bytes,NULL);
      PrintMemory("Set_ProExpn_VNA: TmpNL",angular_bytes,NULL);
      PrintMemory("Set_ProExpn_VNA: TmpNLr",angular_bytes,NULL);
      PrintMemory("Set_ProExpn_VNA: TmpNLt",angular_bytes,NULL);
      PrintMemory("Set_ProExpn_VNA: TmpNLp",angular_bytes,NULL);
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
      const double *displacement=pairs->displacements+(size_t)Nloop*6;
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
         evaluate ovelap integrals <chi0|P> between PAOs
         and progectors of nonlocal VNA potentials.
      ****************************************************/
      /****************************************************
              \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3
      ****************************************************/


      for (L0=0; L0<=preparation->maximum_angular[Cwan]; L0++){
        for (Mul0=0; Mul0<preparation->basis_counts[(size_t)Cwan*preparation->angular_channels+L0]; Mul0++){
          for (L=0; L<Num_RVNA; L++){
            L1 = VNA_List[L];
            for (M0=-L0; M0<=L0; M0++){
	      for (M1=-L1; M1<=L1; M1++){
                const size_t pair_index = openmx_core_dsvna_pair(&buffers,preparation,Cwan,L0,Mul0,L0+M0,L,L1+M1);
		TmpNL[pair_index] = Complex(0.0,0.0);
		TmpNLr[pair_index] = Complex(0.0,0.0);
		TmpNLt[pair_index] = Complex(0.0,0.0);
		TmpNLp[pair_index] = Complex(0.0,0.0);
	      }
	    }
	  }
	}
      }

      Lmax = 0;
      for (L=0; L<Num_RVNA; L++){
	if (Lmax<VNA_List[L]) Lmax = VNA_List[L];
      }
      if (preparation->maximum_angular[Cwan]<Lmax)
	Lmax_Four_Int = 2*Lmax;
      else
        Lmax_Four_Int = 2*preparation->maximum_angular[Cwan];

      /* calculate SphB and SphBp */
      int simd_bessel = 0;
      if (host.mode!=OPENMX_HOST_SIMD_LEGACY) {
        OpenMXVnaBessel bessel;
        for (i=0; i<points; i++) simd_x[i] = GL_NormK[i]*r;
        bessel.points = points; bessel.lmax = Lmax_Four_Int; bessel.x = simd_x;
        bessel.sb = SphB; bessel.dsb = SphBp; bessel.stride = (size_t)points;
        bessel.transposed = NULL; bessel.transposed_derivative = NULL; bessel.transposed_stride = 0;
        simd_bessel = openmx_core_vna_bessel(&bessel,host.mode);
      }
      if (!simd_bessel) {
#ifdef kcomp
#else
#pragma forceinline recursive
#endif
      for (i=0; i<points; i++){
        Normk = GL_NormK[i];
        Spherical_Bessel2(Normk*r,Lmax_Four_Int,tmp_SphB,tmp_SphBp);
        for(LL=0; LL<=Lmax_Four_Int; LL++){
          SphB[(size_t)LL*points+i]  = tmp_SphB[LL];
          SphBp[(size_t)LL*points+i] = tmp_SphBp[LL];
	}
      }
      }

      /* the radial functions of this pair in the legacy order, for the Core's
         contraction (its outputs are the species-local SumNL0/SumNLr0 rows) */
      int simd_radials = -1;
      if (host.mode!=OPENMX_HOST_SIMD_LEGACY && host.projector_table) {
        simd_radials = 0;
        for (L0=0; L0<=preparation->maximum_angular[Cwan]; L0++){
          for (Mul0=0; Mul0<preparation->basis_counts[(size_t)Cwan*preparation->angular_channels+L0]; Mul0++){
            if (openmx_core_vna_basis_radial(preparation,Cwan,L0,Mul0)!=(size_t)simd_radials ||
                openmx_core_dsvna_sum(&buffers,preparation,Cwan,L0,Mul0,0)!=(size_t)simd_radials*(size_t)Num_RVNA ||
                (size_t)simd_radials>=buffers.radial_capacity) simd_radials = -1;
            if (simd_radials<0) break;
            simd_a0[simd_radials] = Bessel_Pro00+openmx_core_vna_radial_offset(preparation,Cwan,L0,Mul0);
            simd_a1[simd_radials] = Bessel_Pro01+openmx_core_vna_radial_offset(preparation,Cwan,L0,Mul0);
            simd_radials++;
          }
          if (simd_radials<0) break;
        }
      }

      /* LL loop */

      for(LL=0; LL<=Lmax_Four_Int; LL++){

	for (L0=0; L0<=preparation->maximum_angular[Cwan]; L0++){
	  for (Mul0=0; Mul0<preparation->basis_counts[(size_t)Cwan*preparation->angular_channels+L0]; Mul0++){
	    for (L=0; L<Num_RVNA; L++){
	      SumNL0[openmx_core_dsvna_sum(&buffers,preparation,Cwan,L0,Mul0,L)] = 0.0;
	      SumNLr0[openmx_core_dsvna_sum(&buffers,preparation,Cwan,L0,Mul0,L)] = 0.0;
	    }
	  }
	}

        /* Gauss-Legendre quadrature */

        if (measure_time) dtime(&stime);

        int simd_contracted = 0;
        if (simd_radials>=0) {
          OpenMXVnaContract contract;
          /* the legacy Bes00/Bes01 rows of every radial function */
          for (k=0; k<simd_radials; k++){
            double *w0 = (double*)simd_w0[k], *w1 = (double*)simd_w1[k];
            for (i=0; i<points; i++){
              w0[i] = simd_a0[k][i]*SphB[(size_t)LL*points+i];
              w1[i] = simd_a1[k][i]*SphBp[(size_t)LL*points+i];
            }
          }
          contract.points = points; contract.rows = Num_RVNA; contract.count = simd_radials;
          contract.table_stride = host.table_stride; contract.out_stride = (size_t)Num_RVNA;
          contract.table = host.projector_table+(size_t)Hwan*points*host.table_stride;
          contract.w0 = simd_w0; contract.w1 = simd_w1;
          contract.out0 = SumNL0; contract.out1 = SumNLr0;
          simd_contracted = openmx_core_vna_contract(&contract,host.mode);
        }
        if (!simd_contracted)
        for (L0=0; L0<=preparation->maximum_angular[Cwan]; L0++){
	  for (Mul0=0; Mul0<preparation->basis_counts[(size_t)Cwan*preparation->angular_channels+L0]; Mul0++){

            const size_t radial_offset = openmx_core_vna_radial_offset(preparation,Cwan,L0,Mul0);
            for (i=0; i<points; i++){
              Bes00[i] = Bessel_Pro00[radial_offset+i]*SphB[(size_t)LL*points+i];
              Bes01[i] = Bessel_Pro01[radial_offset+i]*SphBp[(size_t)LL*points+i];
            }

            L = 0;
            for (L1=0; L1<=preparation->projector_lmax; L1++){
              for (Mul1=0; Mul1<preparation->projector_multiplicity; Mul1++){

                tmp1 = 0.0;
                tmp2 = 0.0;

                for (i=0; i<points; i++){
                  tmp1 += Bes00[i]*openmx_radial_row4(&tables->projectors,Hwan,L1,Mul1)[i];
                  tmp2 += Bes01[i]*openmx_radial_row4(&tables->projectors,Hwan,L1,Mul1)[i];
		}

		SumNL0[openmx_core_dsvna_sum(&buffers,preparation,Cwan,L0,Mul0,L)]  = tmp1;
		SumNLr0[openmx_core_dsvna_sum(&buffers,preparation,Cwan,L0,Mul0,L)] = tmp2;

                L++;
	      }
	    }
	  }
	}

        if (measure_time && OMPID==0){
          dtime(&etime);
          *time1 += etime - stime;
	}

	/* derivatives of "on site" */

	if (h_AN==0){
	  for (L0=0; L0<=preparation->maximum_angular[Cwan]; L0++){
	    for (Mul0=0; Mul0<preparation->basis_counts[(size_t)Cwan*preparation->angular_channels+L0]; Mul0++){
	      for (L=0; L<Num_RVNA; L++){
		SumNLr0[openmx_core_dsvna_sum(&buffers,preparation,Cwan,L0,Mul0,L)] = 0.0;
	      }
	    }
	  }
	}

	/****************************************************
         for "overlap",
	    sum_m 8*(-i)^{-L0+L1+l}*
	          C_{L0,-M0,L1,M1,l,m}*
	          \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3,
	****************************************************/

        if (measure_time && OMPID==0) dtime(&stime);

	for(m=-LL; m<=LL; m++){

	  ComplexSH(LL,m,theta,phi,SH,dSHt,dSHp);

	  SH[1]   = -SH[1];
	  dSHt[1] = -dSHt[1];
	  dSHp[1] = -dSHp[1];

	  CY  = Complex(SH[0],SH[1]);
	  CYt = Complex(dSHt[0],dSHt[1]);
	  CYp = Complex(dSHp[0],dSHp[1]);

	  for (L0=0; L0<=preparation->maximum_angular[Cwan]; L0++){
	    for (Mul0=0; Mul0<preparation->basis_counts[(size_t)Cwan*preparation->angular_channels+L0]; Mul0++){
	      for (L=0; L<Num_RVNA; L++){

		L1 = VNA_List[L];
		Ls = -L0 + L1 + LL;

                if ( abs(L1-LL)<=L0 && L0<=(L1+LL) ){

		  Cpow = Im_pow(-1,Ls);
		  CY1  = Cmul(Cpow,CY);
		  CYt1 = Cmul(Cpow,CYt);
		  CYp1 = Cmul(Cpow,CYp);

		  for (M0=-L0; M0<=L0; M0++){

		    M1 = M0 - m;

                    if (abs(M1)<=L1){
                      const size_t pair_index = openmx_core_dsvna_pair(&buffers,preparation,Cwan,L0,Mul0,L0+M0,L,L1+M1);

		      gant = host.mode!=OPENMX_HOST_SIMD_LEGACY
		        ? SetPro_GauntTable_At(&host.gaunt,L0,M0,L1,M1,LL) : Gaunt(L0,M0,L1,M1,LL,m);
		      tmp2 = gant*SumNL0[openmx_core_dsvna_sum(&buffers,preparation,Cwan,L0,Mul0,L)];

		      /* S */

		      TmpNL[pair_index].r += CY1.r*tmp2;
		      TmpNL[pair_index].i += CY1.i*tmp2;

		      /* dS/dr */

		      tmp0 = gant*SumNLr0[openmx_core_dsvna_sum(&buffers,preparation,Cwan,L0,Mul0,L)];

		      TmpNLr[pair_index].r += CY1.r*tmp0;
		      TmpNLr[pair_index].i += CY1.i*tmp0;

		      /* dS/dt */

		      TmpNLt[pair_index].r += CYt1.r*tmp2;
		      TmpNLt[pair_index].i += CYt1.i*tmp2;

		      /* dS/dp */

		      TmpNLp[pair_index].r += CYp1.r*tmp2;
		      TmpNLp[pair_index].i += CYp1.i*tmp2;

		    }
		  }
		}

	      }
	    }
	  }
	} /* m */

        if (measure_time && OMPID==0){
          dtime(&etime);
          *time2 += etime - stime;
	}

      } /* LL */

      /****************************************************
                        Complex to Real
      ****************************************************/

      if (measure_time) dtime(&stime);

      num0 = 0;
      for (L0=0; L0<=preparation->maximum_angular[Cwan]; L0++){
	for (Mul0=0; Mul0<preparation->basis_counts[(size_t)Cwan*preparation->angular_channels+L0]; Mul0++){

	  num1 = 0;
	  for (L=0; L<Num_RVNA; L++){
	    L1 = VNA_List[L];

	    for (M0=-L0; M0<=L0; M0++){
	      for (M1=-L1; M1<=L1; M1++){

		CsumNL0 = Complex(0.0,0.0);
		CsumNLr = Complex(0.0,0.0);
		CsumNLt = Complex(0.0,0.0);
		CsumNLp = Complex(0.0,0.0);

		for (k=-L0; k<=L0; k++){
                  const size_t pair_index = openmx_core_dsvna_pair(&buffers,preparation,Cwan,L0,Mul0,L0+k,L,L1+M1);

		  Ctmp1 = Conjg(SetPro_Conversion(tables,L0,L0+M0,L0+k));

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

		CmatNL0[(size_t)(L0+M0)*buffers.conversion_columns+(L1+M1)] = CsumNL0;
		CmatNLr[(size_t)(L0+M0)*buffers.conversion_columns+(L1+M1)] = CsumNLr;
		CmatNLt[(size_t)(L0+M0)*buffers.conversion_columns+(L1+M1)] = CsumNLt;
		CmatNLp[(size_t)(L0+M0)*buffers.conversion_columns+(L1+M1)] = CsumNLp;

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

		  CsumNL0.r += (CmatNL0[(size_t)(L0+M0)*buffers.conversion_columns+(L1+k)].r*SetPro_Conversion(tables,L1,L1+M1,L1+k).r
				- CmatNL0[(size_t)(L0+M0)*buffers.conversion_columns+(L1+k)].i*SetPro_Conversion(tables,L1,L1+M1,L1+k).i);

		  CsumNL0.i += (CmatNL0[(size_t)(L0+M0)*buffers.conversion_columns+(L1+k)].r*SetPro_Conversion(tables,L1,L1+M1,L1+k).i
				+ CmatNL0[(size_t)(L0+M0)*buffers.conversion_columns+(L1+k)].i*SetPro_Conversion(tables,L1,L1+M1,L1+k).r);

		  /* dS/dr */

		  CsumNLr.r += (CmatNLr[(size_t)(L0+M0)*buffers.conversion_columns+(L1+k)].r*SetPro_Conversion(tables,L1,L1+M1,L1+k).r
				- CmatNLr[(size_t)(L0+M0)*buffers.conversion_columns+(L1+k)].i*SetPro_Conversion(tables,L1,L1+M1,L1+k).i);

		  CsumNLr.i += (CmatNLr[(size_t)(L0+M0)*buffers.conversion_columns+(L1+k)].r*SetPro_Conversion(tables,L1,L1+M1,L1+k).i
				+ CmatNLr[(size_t)(L0+M0)*buffers.conversion_columns+(L1+k)].i*SetPro_Conversion(tables,L1,L1+M1,L1+k).r);

		  /* dS/dt */

		  CsumNLt.r += (CmatNLt[(size_t)(L0+M0)*buffers.conversion_columns+(L1+k)].r*SetPro_Conversion(tables,L1,L1+M1,L1+k).r
				- CmatNLt[(size_t)(L0+M0)*buffers.conversion_columns+(L1+k)].i*SetPro_Conversion(tables,L1,L1+M1,L1+k).i);

		  CsumNLt.i += (CmatNLt[(size_t)(L0+M0)*buffers.conversion_columns+(L1+k)].r*SetPro_Conversion(tables,L1,L1+M1,L1+k).i
				+ CmatNLt[(size_t)(L0+M0)*buffers.conversion_columns+(L1+k)].i*SetPro_Conversion(tables,L1,L1+M1,L1+k).r);

		  /* dS/dp */

		  CsumNLp.r += (CmatNLp[(size_t)(L0+M0)*buffers.conversion_columns+(L1+k)].r*SetPro_Conversion(tables,L1,L1+M1,L1+k).r
				- CmatNLp[(size_t)(L0+M0)*buffers.conversion_columns+(L1+k)].i*SetPro_Conversion(tables,L1,L1+M1,L1+k).i);

		  CsumNLp.i += (CmatNLp[(size_t)(L0+M0)*buffers.conversion_columns+(L1+k)].r*SetPro_Conversion(tables,L1,L1+M1,L1+k).i
				+ CmatNLp[(size_t)(L0+M0)*buffers.conversion_columns+(L1+k)].i*SetPro_Conversion(tables,L1,L1+M1,L1+k).r);

		}

		openmx_projector_block_row(DS_VNA,0,Mc_AN,h_AN,num0+L0+M0)[num1+L1+M1] = 8.0*(Type_DS_VNA)CsumNL0.r;

		if (h_AN!=0){

		  if (fabs(siT)<10e-14){

		    openmx_projector_block_row(DS_VNA,1,Mc_AN,h_AN,num0+L0+M0)[num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(siT*coP*CsumNLr.r + coT*coP/r*CsumNLt.r);

		    openmx_projector_block_row(DS_VNA,2,Mc_AN,h_AN,num0+L0+M0)[num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(siT*siP*CsumNLr.r + coT*siP/r*CsumNLt.r);

		    openmx_projector_block_row(DS_VNA,3,Mc_AN,h_AN,num0+L0+M0)[num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(coT*CsumNLr.r - siT/r*CsumNLt.r);
		  }

		  else{

		    openmx_projector_block_row(DS_VNA,1,Mc_AN,h_AN,num0+L0+M0)[num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(siT*coP*CsumNLr.r + coT*coP/r*CsumNLt.r
					- siP/siT/r*CsumNLp.r);

		    openmx_projector_block_row(DS_VNA,2,Mc_AN,h_AN,num0+L0+M0)[num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(siT*siP*CsumNLr.r + coT*siP/r*CsumNLt.r
					+ coP/siT/r*CsumNLp.r);

		    openmx_projector_block_row(DS_VNA,3,Mc_AN,h_AN,num0+L0+M0)[num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(coT*CsumNLr.r - siT/r*CsumNLt.r);
		  }

		}

		else{
		  openmx_projector_block_row(DS_VNA,1,Mc_AN,h_AN,num0+L0+M0)[num1+L1+M1] = 0.0;
		  openmx_projector_block_row(DS_VNA,2,Mc_AN,h_AN,num0+L0+M0)[num1+L1+M1] = 0.0;
		  openmx_projector_block_row(DS_VNA,3,Mc_AN,h_AN,num0+L0+M0)[num1+L1+M1] = 0.0;
		}

	      }
	    }

	    num1 = num1 + 2*L1 + 1;
	  }

	  num0 = num0 + 2*L0 + 1;
	} /* M0 */
      } /* L0 */

      if (measure_time && OMPID==0){
	dtime(&etime);
	*time3 += etime - stime;
      }

      dtime(&Etime_atom);
#pragma omp atomic update
      tables->atom_seconds[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    openmx_core_dsvna_destroy(workspace);
    free(simd_x); free(simd_w); free(simd_a0); free(simd_a1); free(simd_w0); free(simd_w1);

#pragma omp flush(DS_VNA)

  } /* #pragma omp parallel */

  SetPro_DsvnaHost_Free(&host);
}
#endif
