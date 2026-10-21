/* DS_VNA CPU block frozen from 36ad25d918c1052e90d0053f17a0c7bea3458204.
   Kept independently of the new packed workspace for numerical regression. */
#pragma omp parallel shared(time_per_atom,DS_VNA,Comp2Real,Spe_VNA_Bessel,Bessel_Pro01,Bessel_Pro00,GL_NormK,List_YOUSO,VNA_List,Num_RVNA,Spe_Num_Basis,Spe_MaxL_Basis,PAO_Nkmax,atv,Gxyz,ncn,natn,Spe_Atom_Cut1,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop,time1,time2,time3)

  {
    int i,j,k,l,m,num0,num1;
    int Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN;
    int Rnh,Hwan,L0,Mul0,L,L1,M0,M1,LL,Mul1,Ls;
    int OMPID,Nthrds,Nprocs,Nloop;
    int Lmax,Lmax_Four_Int;
    double Stime_atom,Etime_atom;
    double rcutA,rcutB,rcut;
    double dx,dy,dz;
    double kmin,kmax,Sk,Dk;
    double gant,SH[2],dSHt[2],dSHp[2];
    double S_coordinate[3];
    double r,theta,phi;
    double Normk,tmp0,tmp1,tmp2;
    double siT,coT,siP,coP;
    double ***SumNL0;
    double ***SumNLr0;
    double *Bes00;
    double *Bes01;
    double tmp_SphB[30];
    double tmp_SphBp[30];
    double SphB[30][GL_Mesh];
    double SphBp[30][GL_Mesh];
    double stime,etime;

    dcomplex Ctmp1,Ctmp0,Ctmp2;
    dcomplex CsumNL0,CsumNLr,CsumNLt,CsumNLp;
    dcomplex CY,CYt,CYp,CY1,CYt1,CYp1,Cpow;
    dcomplex *****TmpNL;
    dcomplex *****TmpNLr;
    dcomplex *****TmpNLt;
    dcomplex *****TmpNLp;
    dcomplex **CmatNL0;
    dcomplex **CmatNLr;
    dcomplex **CmatNLt;
    dcomplex **CmatNLp;

    /* allocation of arrays */

    TmpNL = (dcomplex*****)malloc(sizeof(dcomplex****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpNL[i] = (dcomplex****)malloc(sizeof(dcomplex***)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNL[i][j] = (dcomplex***)malloc(sizeof(dcomplex**)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpNL[i][j][k] = (dcomplex**)malloc(sizeof(dcomplex*)*Num_RVNA);
	  for (l=0; l<Num_RVNA; l++){
	    TmpNL[i][j][k][l] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
	  }
	}
      }
    }

    TmpNLr = (dcomplex*****)malloc(sizeof(dcomplex****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpNLr[i] = (dcomplex****)malloc(sizeof(dcomplex***)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNLr[i][j] = (dcomplex***)malloc(sizeof(dcomplex**)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpNLr[i][j][k] = (dcomplex**)malloc(sizeof(dcomplex*)*Num_RVNA);
	  for (l=0; l<Num_RVNA; l++){
	    TmpNLr[i][j][k][l] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
	  }
	}
      }
    }

    TmpNLt = (dcomplex*****)malloc(sizeof(dcomplex****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpNLt[i] = (dcomplex****)malloc(sizeof(dcomplex***)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNLt[i][j] = (dcomplex***)malloc(sizeof(dcomplex**)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpNLt[i][j][k] = (dcomplex**)malloc(sizeof(dcomplex*)*Num_RVNA);
	  for (l=0; l<Num_RVNA; l++){
	    TmpNLt[i][j][k][l] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
	  }
	}
      }
    }

    TmpNLp = (dcomplex*****)malloc(sizeof(dcomplex****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpNLp[i] = (dcomplex****)malloc(sizeof(dcomplex***)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNLp[i][j] = (dcomplex***)malloc(sizeof(dcomplex**)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpNLp[i][j][k] = (dcomplex**)malloc(sizeof(dcomplex*)*Num_RVNA);
	  for (l=0; l<Num_RVNA; l++){
	    TmpNLp[i][j][k][l] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
	  }
	}
      }
    }

    SumNL0 = (double***)malloc(sizeof(double**)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      SumNL0[i] = (double**)malloc(sizeof(double*)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	SumNL0[i][j] = (double*)malloc(sizeof(double)*Num_RVNA);
      }
    }

    SumNLr0 = (double***)malloc(sizeof(double**)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      SumNLr0[i] = (double**)malloc(sizeof(double*)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	SumNLr0[i][j] = (double*)malloc(sizeof(double)*Num_RVNA);
      }
    }

    CmatNL0 = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatNL0[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
    }

    CmatNLr = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatNLr[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
    }

    CmatNLt = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatNLt[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
    }

    CmatNLp = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatNLp[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[35]+1));
    }

    Bes00 = (double*)malloc(sizeof(double)*GL_Mesh);
    Bes01 = (double*)malloc(sizeof(double)*GL_Mesh);

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
      rcutA = Spe_Atom_Cut1[Cwan];

      /* set data on h_AN */

      Gh_AN = natn[Gc_AN][h_AN];
      Rnh = ncn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      rcutB = Spe_Atom_Cut1[Hwan];
      rcut = rcutA + rcutB;

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

      /****************************************************
         evaluate ovelap integrals <chi0|P> between PAOs
         and progectors of nonlocal VNA potentials.
      ****************************************************/
      /****************************************************
              \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3
      ****************************************************/

      kmin = Radial_kmin;
      kmax = PAO_Nkmax;
      Sk = kmax + kmin;
      Dk = kmax - kmin;

      for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
        for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
          for (L=0; L<Num_RVNA; L++){
            L1 = VNA_List[L];
            for (M0=-L0; M0<=L0; M0++){
	      for (M1=-L1; M1<=L1; M1++){
		TmpNL[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0);
		TmpNLr[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0);
		TmpNLt[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0);
		TmpNLp[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0);
	      }
	    }
	  }
	}
      }

      Lmax = -10;
      for (L=0; L<Num_RVNA; L++){
	if (Lmax<VNA_List[L]) Lmax = VNA_List[L];
      }
      if (Spe_MaxL_Basis[Cwan]<Lmax)
	Lmax_Four_Int = 2*Lmax;
      else
        Lmax_Four_Int = 2*Spe_MaxL_Basis[Cwan];

      /* calculate SphB and SphBp */
#ifdef kcomp
#else
#pragma forceinline recursive
#endif
      for (i=0; i<GL_Mesh; i++){
        Normk = GL_NormK[i];
        Spherical_Bessel2(Normk*r,Lmax_Four_Int,tmp_SphB,tmp_SphBp);
        for(LL=0; LL<=Lmax_Four_Int; LL++){
          SphB[LL][i]  = tmp_SphB[LL];
          SphBp[LL][i] = tmp_SphBp[LL];
	}
      }

      /* LL loop */

      for(LL=0; LL<=Lmax_Four_Int; LL++){

	for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	    for (L=0; L<Num_RVNA; L++){
	      SumNL0[L0][Mul0][L] = 0.0;
	      SumNLr0[L0][Mul0][L] = 0.0;
	    }
	  }
	}

        /* Gauss-Legendre quadrature */

        if (measure_time) dtime(&stime);

        for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){

            const size_t radial_offset = openmx_core_vna_radial_offset(&preparation,Cwan,L0,Mul0);
            for (i=0; i<GL_Mesh; i++){
              Bes00[i] = Bessel_Pro00[radial_offset+i]*SphB[LL][i];
              Bes01[i] = Bessel_Pro01[radial_offset+i]*SphBp[LL][i];
            }

            L = 0;
            for (L1=0; L1<=List_YOUSO[35]; L1++){
              for (Mul1=0; Mul1<List_YOUSO[34]; Mul1++){

                tmp1 = 0.0;
                tmp2 = 0.0;

                for (i=0; i<GL_Mesh; i++){
                  tmp1 += Bes00[i]*Spe_VNA_Bessel[Hwan][L1][Mul1][i];
                  tmp2 += Bes01[i]*Spe_VNA_Bessel[Hwan][L1][Mul1][i];
		}

		SumNL0[L0][Mul0][L]  = tmp1;
		SumNLr0[L0][Mul0][L] = tmp2;

                L++;
	      }
	    }
	  }
	}

        if (measure_time && OMPID==0){
          dtime(&etime);
          time1 += etime - stime;
	}

	/* derivatives of "on site" */

	if (h_AN==0){
	  for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	    for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	      for (L=0; L<Num_RVNA; L++){
		SumNLr0[L0][Mul0][L] = 0.0;
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

	  for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	    for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
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

		      gant = Gaunt(L0,M0,L1,M1,LL,m);
		      tmp2 = gant*SumNL0[L0][Mul0][L];

		      /* S */

		      TmpNL[L0][Mul0][L0+M0][L][L1+M1].r += CY1.r*tmp2;
		      TmpNL[L0][Mul0][L0+M0][L][L1+M1].i += CY1.i*tmp2;

		      /* dS/dr */

		      tmp0 = gant*SumNLr0[L0][Mul0][L];

		      TmpNLr[L0][Mul0][L0+M0][L][L1+M1].r += CY1.r*tmp0;
		      TmpNLr[L0][Mul0][L0+M0][L][L1+M1].i += CY1.i*tmp0;

		      /* dS/dt */

		      TmpNLt[L0][Mul0][L0+M0][L][L1+M1].r += CYt1.r*tmp2;
		      TmpNLt[L0][Mul0][L0+M0][L][L1+M1].i += CYt1.i*tmp2;

		      /* dS/dp */

		      TmpNLp[L0][Mul0][L0+M0][L][L1+M1].r += CYp1.r*tmp2;
		      TmpNLp[L0][Mul0][L0+M0][L][L1+M1].i += CYp1.i*tmp2;

		    }
		  }
		}

	      }
	    }
	  }
	} /* m */

        if (measure_time && OMPID==0){
          dtime(&etime);
          time2 += etime - stime;
	}

      } /* LL */

      /****************************************************
                        Complex to Real
      ****************************************************/

      if (measure_time) dtime(&stime);

      num0 = 0;
      for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){

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

		  Ctmp1 = Conjg(Comp2Real[L0][L0+M0][L0+k]);

		  /* S */

		  Ctmp0 = TmpNL[L0][Mul0][L0+k][L][L1+M1];
		  Ctmp2 = Cmul(Ctmp1,Ctmp0);
		  CsumNL0 = Cadd(CsumNL0,Ctmp2);

		  /* dS/dr */

		  Ctmp0 = TmpNLr[L0][Mul0][L0+k][L][L1+M1];
		  Ctmp2 = Cmul(Ctmp1,Ctmp0);
		  CsumNLr = Cadd(CsumNLr,Ctmp2);

		  /* dS/dt */

		  Ctmp0 = TmpNLt[L0][Mul0][L0+k][L][L1+M1];
		  Ctmp2 = Cmul(Ctmp1,Ctmp0);
		  CsumNLt = Cadd(CsumNLt,Ctmp2);

		  /* dS/dp */

		  Ctmp0 = TmpNLp[L0][Mul0][L0+k][L][L1+M1];
		  Ctmp2 = Cmul(Ctmp1,Ctmp0);
		  CsumNLp = Cadd(CsumNLp,Ctmp2);

		}

		CmatNL0[L0+M0][L1+M1] = CsumNL0;
		CmatNLr[L0+M0][L1+M1] = CsumNLr;
		CmatNLt[L0+M0][L1+M1] = CsumNLt;
		CmatNLp[L0+M0][L1+M1] = CsumNLp;

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

		  CsumNL0.r += (CmatNL0[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].r
				- CmatNL0[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].i);

		  CsumNL0.i += (CmatNL0[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].i
				+ CmatNL0[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].r);

		  /* dS/dr */

		  CsumNLr.r += (CmatNLr[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].r
				- CmatNLr[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].i);

		  CsumNLr.i += (CmatNLr[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].i
				+ CmatNLr[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].r);

		  /* dS/dt */

		  CsumNLt.r += (CmatNLt[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].r
				- CmatNLt[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].i);

		  CsumNLt.i += (CmatNLt[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].i
				+ CmatNLt[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].r);

		  /* dS/dp */

		  CsumNLp.r += (CmatNLp[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].r
				- CmatNLp[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].i);

		  CsumNLp.i += (CmatNLp[L0+M0][L1+k].r*Comp2Real[L1][L1+M1][L1+k].i
				+ CmatNLp[L0+M0][L1+k].i*Comp2Real[L1][L1+M1][L1+k].r);

		}

		DS_VNA[0][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 8.0*(Type_DS_VNA)CsumNL0.r;

		if (h_AN!=0){

		  if (fabs(siT)<10e-14){

		    DS_VNA[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(siT*coP*CsumNLr.r + coT*coP/r*CsumNLt.r);

		    DS_VNA[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(siT*siP*CsumNLr.r + coT*siP/r*CsumNLt.r);

		    DS_VNA[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(coT*CsumNLr.r - siT/r*CsumNLt.r);
		  }

		  else{

		    DS_VNA[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(siT*coP*CsumNLr.r + coT*coP/r*CsumNLt.r
					- siP/siT/r*CsumNLp.r);

		    DS_VNA[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(siT*siP*CsumNLr.r + coT*siP/r*CsumNLt.r
					+ coP/siT/r*CsumNLp.r);

		    DS_VNA[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
		      -8.0*(Type_DS_VNA)(coT*CsumNLr.r - siT/r*CsumNLt.r);
		  }

		}

		else{
		  DS_VNA[1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		  DS_VNA[2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		  DS_VNA[3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
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
	time3 += etime - stime;
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of arrays */

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNL0[i]);
    }
    free(CmatNL0);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNLr[i]);
    }
    free(CmatNLr);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNLt[i]);
    }
    free(CmatNLt);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNLp[i]);
    }
    free(CmatNLp);

    free(Bes00);
    free(Bes01);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	free(SumNLr0[i][j]);
      }
      free(SumNLr0[i]);
    }
    free(SumNLr0);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	free(SumNL0[i][j]);
      }
      free(SumNL0[i]);
    }
    free(SumNL0);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<Num_RVNA; l++){
	    free(TmpNLp[i][j][k][l]);
	  }
          free(TmpNLp[i][j][k]);
	}
        free(TmpNLp[i][j]);
      }
      free(TmpNLp[i]);
    }
    free(TmpNLp);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<Num_RVNA; l++){
	    free(TmpNLt[i][j][k][l]);
	  }
	  free(TmpNLt[i][j][k]);
	}
	free(TmpNLt[i][j]);
      }
      free(TmpNLt[i]);
    }
    free(TmpNLt);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<Num_RVNA; l++){
	    free(TmpNLr[i][j][k][l]);
	  }
	  free(TmpNLr[i][j][k]);
	}
	free(TmpNLr[i][j]);
      }
      free(TmpNLr[i]);
    }
    free(TmpNLr);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<Num_RVNA; l++){
	    free(TmpNL[i][j][k][l]);
	  }
	  free(TmpNL[i][j][k]);
	}
	free(TmpNL[i][j]);
      }
      free(TmpNL[i]);
    }
    free(TmpNL);

#pragma omp flush(DS_VNA)

  } /* #pragma omp parallel */
