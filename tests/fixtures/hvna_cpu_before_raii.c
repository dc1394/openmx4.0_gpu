/* Frozen CPU HVNA assembly from 6227226a. Only test wrappers, names and
   trailing whitespace differ; the contraction and parallel loops are retained. */
static void reference_hvna_halo(Type_DS_VNA *****DS_VNA,double ****HVNA,
    int Original_Mc_AN,int received_global)
{
  int Mc_AN,Gc_AN,Cwan,fan,n,jg,j0,jg0,Mj_AN0,po,j,Hwan,k,kg,wakg,kl;
  int m,L,L1,GL,Mul1,L2,L3,i1,j1,IDR=0;
  double Stime_atom,Etime_atom,rcutA,rcutB,rcut,sum,ene,tmp0,dmp;
  double **NLH=malloc(sizeof(double*)*List_YOUSO[7]);
  for(int i=0;i<List_YOUSO[7];++i)NLH[i]=malloc(sizeof(double)*List_YOUSO[7]);
  Rcv_GAN[0][0]=received_global;

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	  dtime(&Stime_atom);

	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  fan = FNAN[Gc_AN];
	  rcutA = Spe_Atom_Cut1[Cwan];

	  n = F_Rcv_Num_WK[IDR];
	  jg = Rcv_GAN[IDR][n];

	  for (j0=0; j0<=fan; j0++){

	    jg0 = natn[Gc_AN][j0];
	    Mj_AN0 = F_G2M[jg0];

	    po = 0;
	    if (Original_Mc_AN==Mj_AN0){
	      po = 1;
	      j = j0;
	    }

	    if (po==1){

	      Hwan = WhatSpecies[jg];
	      rcutB = Spe_Atom_Cut1[Hwan];
	      rcut = rcutA + rcutB;

	      for (k=0; k<=fan; k++){

		kg = natn[Gc_AN][k];
		wakg = WhatSpecies[kg];
		kl = RMI1[Mc_AN][j][k];

		if (0<=kl){
#pragma omp parallel for private(m, n, sum, L ,L1, GL, Mul1, ene, L2 ,tmp0, L3) default(shared) if(Spe_Total_NO[Cwan] > 8)
		  for (m=0; m<Spe_Total_NO[Cwan]; m++){
		    for (n=0; n<Spe_Total_NO[Hwan]; n++){

		      sum = 0.0;
		      L = 0;

		      for (L1=0; L1<Num_RVNA; L1++){

			GL = VNA_List[L1];
			Mul1 = VNA_List2[L1];

			ene = VNA_proj_ene[wakg][GL][Mul1];
			L2 = 2*VNA_List[L1];

			tmp0 = 0.0;

			for (L3=0; L3<=L2; L3++){

			  tmp0 += fp32 ? (double)(DS_VNA[0][Mc_AN][k][m][L] * DS_VNA[0][Matomnum+1][kl][n][L])
			                   : (double)DS_VNA[0][Mc_AN][k][m][L] * (double)DS_VNA[0][Matomnum+1][kl][n][L];

			  L++;
			}

			sum += ene*tmp0;

		      }

		      if (k==0)  NLH[m][n]  = sum;
		      else       NLH[m][n] += sum;

		    } /* n */
		  } /* m */
		} /* if (0<=kl)*/
	      } /* k */

	      /****************************************************
                            NLH to HVNA
	      ****************************************************/

	      dmp = dampingF(rcut,Dis[Gc_AN][j]);

	      for (i1=0; i1<Spe_Total_NO[Cwan]; i1++){
		for (j1=0; j1<Spe_Total_NO[Hwan]; j1++){
		  HVNA[Mc_AN][j][i1][j1] = dmp*NLH[i1][j1];
		}
	      }

	    } /* if (po==1) */

	  } /* j */

	  dtime(&Etime_atom);
	  time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	} /* Mc_AN */


  for(int i=0;i<List_YOUSO[7];++i)free(NLH[i]);
  free(NLH);
}
static void reference_hvna_local(Type_DS_VNA *****DS_VNA,double ****HVNA)
{

#pragma omp parallel shared(List_YOUSO,time_per_atom,HVNA,Dis,DS_VNA,VNA_proj_ene,VNA_List2,VNA_List,Num_RVNA,Spe_Total_NO,RMI1,F_G2M,natn,Spe_Atom_Cut1,FNAN,WhatSpecies,M2G,Matomnum)
  {

    int OMPID,Nthrds,Nprocs;
    int Mc_AN,Gc_AN,Cwan,fan,Mj_AN,Hwan;
    int L,L1,GL,Mul1,L2,L3,i1,j1;
    int k,kg,wakg,kl,i,j,jg,m,n;
    double **NLH;
    double rcutA,rcutB,rcut,sum,ene,tmp0,dmp;
    double Stime_atom,Etime_atom;

    /* allocation of array */

    NLH = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
    for (i=0; i<List_YOUSO[7]; i++){
      NLH[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    }

    /* get info. on OpenMP */

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      fan = FNAN[Gc_AN];
      rcutA = Spe_Atom_Cut1[Cwan];

      for (j=0; j<=fan; j++){

	jg = natn[Gc_AN][j];
	Mj_AN = F_G2M[jg];

	if (Mj_AN<=Matomnum){

	  Hwan = WhatSpecies[jg];
	  rcutB = Spe_Atom_Cut1[Hwan];
	  rcut = rcutA + rcutB;

	  for (k=0; k<=fan; k++){

	    kg = natn[Gc_AN][k];
	    wakg = WhatSpecies[kg];
	    kl = RMI1[Mc_AN][j][k];

	    if (0<=kl){

	      for (m=0; m<Spe_Total_NO[Cwan]; m++){
		for (n=0; n<Spe_Total_NO[Hwan]; n++){

		  sum = 0.0;
		  L = 0;

		  for (L1=0; L1<Num_RVNA; L1++){

		    GL = VNA_List[L1];
		    Mul1 = VNA_List2[L1];

		    ene = VNA_proj_ene[wakg][GL][Mul1];
		    L2 = 2*VNA_List[L1];

		    tmp0 = 0.0;

		    for (L3=0; L3<=L2; L3++){
		      tmp0 += fp32 ? (double)(DS_VNA[0][Mc_AN][k][m][L] * DS_VNA[0][Mj_AN][kl][n][L])
		                       : (double)DS_VNA[0][Mc_AN][k][m][L] * (double)DS_VNA[0][Mj_AN][kl][n][L];
		      L++;
		    }

		    sum += ene*tmp0;

		  }

		  if (k==0)  NLH[m][n]  = sum;
		  else       NLH[m][n] += sum;

		}
	      }
	    }
	  } /* k */

	  /****************************************************
                            NLH to HVNA
	  ****************************************************/

	  dmp = dampingF(rcut,Dis[Gc_AN][j]);

	  for (i1=0; i1<Spe_Total_NO[Cwan]; i1++){
	    for (j1=0; j1<Spe_Total_NO[Hwan]; j1++){
	      HVNA[Mc_AN][j][i1][j1] = dmp*NLH[i1][j1];
	    }
	  }

	} /* if (Mj_AN<=Matomnum) */
      } /* j */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of arrays */

    for (i=0; i<List_YOUSO[7]; i++){
      free(NLH[i]);
    }
    free(NLH);

#pragma omp flush(HVNA)

  } /* #pragma omp parallel */


}
