/**********************************************************************
  Overlap_Cluster_LNO_Ss.c:

     Overlap_Cluster_LNO_Ss.c is a subroutine to make an overlap matrix
     contracted with localized natural orbitals (LNOs) for cluster or 
     molecular systems, which is distributed over MPI cores according 
     to data distribution of ScaLAPACK.

  Log of Overlap_Cluster_LNO_Ss.c:

     02/Nov/2023  Released by T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"
#include "mpi.h"

void Overlap_Cluster_LNO_Ss(double ****OLP0, double *Ss, int *MP, int spin, int myworld1)
{
  int i,j,k,p;
  int MA_AN,MB_AN,GA_AN,LB_AN,GB_AN,AN;
  int wanA,wanB,tnoA,tnoB,tnoA0,tnoB0;
  int Anum,Bnum,NUM;
  int num,tnum,num_orbitals;
  int ID,myid,numprocs,tag=999;
  int *My_NZeros;
  int *is1,*ie1,*is2;
  int *My_Matomnum,*order_GA;
  double *S1,*Sc,*Stmp,sum,sumS;
  MPI_Status stat;
  MPI_Request request;

  int ig,jg,il,jl,prow,pcol,brow,bcol;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /* allocation of arrays */

  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  My_Matomnum = (int*)malloc(sizeof(int)*numprocs);
  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);
  is2 = (int*)malloc(sizeof(int)*numprocs);
  order_GA = (int*)malloc(sizeof(int)*(atomnum+2));
  Sc = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  Stmp = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);

  /* find my total number of non-zero elements in myid */

  My_NZeros[myid] = 0;
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = LNOs_Num_predefined[wanA];

    num = 0;      
    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = LNOs_Num_predefined[wanB];
      num += tnoB;
    }

    My_NZeros[myid] += tnoA*num;
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_NZeros[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  tnum = 0;
  for (ID=0; ID<numprocs; ID++){
    tnum += My_NZeros[ID];
  }  

  is1[0] = 0;
  ie1[0] = My_NZeros[0] - 1;

  for (ID=1; ID<numprocs; ID++){
    is1[ID] = ie1[ID-1] + 1;
    ie1[ID] = is1[ID] + My_NZeros[ID] - 1;
  }  

  /* set is2 and order_GA */

  My_Matomnum[myid] = Matomnum;
  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_Matomnum[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  is2[0] = 1;
  for (ID=1; ID<numprocs; ID++){
    is2[ID] = is2[ID-1] + My_Matomnum[ID-1];
  }
  
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    order_GA[is2[myid]+MA_AN-1] = M2G[MA_AN];
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&order_GA[is2[ID]],My_Matomnum[ID],MPI_INT,ID,mpi_comm_level1);
  }

  /* set MP */

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    wanA = WhatSpecies[i];
    Anum += LNOs_Num_predefined[wanA];
  }
  NUM = Anum - 1;

  /* set S1 */

  S1 = (double*)malloc(sizeof(double)*(tnum+1));

  p = is1[myid];
  
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){

    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA0 = Spe_Total_CNO[wanA];
    tnoA = LNO_Num[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      MB_AN = S_G2M[GB_AN];
      
      wanB = WhatSpecies[GB_AN];
      tnoB0 = Spe_Total_CNO[wanB];
      tnoB = LNO_Num[GB_AN];

      /* transformation of representation for OLP0 */

      for (i=0; i<tnoA0; i++){
	for (j=0; j<tnoB0; j++){
	  Sc[j*tnoA0+i] = OLP0[MA_AN][LB_AN][i][j];
	}
      }
      
      for (i=0; i<tnoA; i++){
	for (j=0; j<tnoB0; j++){

	  sumS = 0.0;

	  for (k=0; k<tnoA0; k++){
	    sumS += LNO_coes[spin][MA_AN][tnoA0*i+k]*Sc[tnoA0*j+k];
	  }

	  Stmp[i*tnoB0+j] = sumS;
	  
	} // j
      } // i

      for (i=0; i<tnoA; i++){
	for (j=0; j<tnoB; j++){

	  sumS = 0.0;

	  for (k=0; k<tnoB0; k++){
	    sumS += Stmp[i*tnoB0+k]*LNO_coes[spin][MB_AN][tnoB0*j+k];
	  }

	  S1[p] = sumS;
	  p++;
	  
	} // j
      } // i
      
    } // LB_AN
  } // MA_AN

  /* MPI S1 */

  for (ID=0; ID<numprocs; ID++){
    k = is1[ID];
    MPI_Bcast(&S1[k], My_NZeros[ID], MPI_DOUBLE, ID, mpi_comm_level1);
  }

  /* S1 -> Ss */

  if (spin==myworld1){
  
    for(i=0;i<na_rows*na_cols;i++){
      Ss[i] = 0.0;
    }

    k = 0;
    for (AN=1; AN<=atomnum; AN++){
      GA_AN = order_GA[AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = LNO_Num[GA_AN];
      Anum = MP[GA_AN];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = LNO_Num[GB_AN];
	Bnum = MP[GB_AN];

	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

	    ig = Anum+i;
	    jg = Bnum+j;
	    
	    brow = (ig-1)/nblk;
	    bcol = (jg-1)/nblk;

	    prow = brow%np_rows;
	    pcol = bcol%np_cols;

	    if (my_prow==prow && my_pcol==pcol){

	      il = (brow/np_rows+1)*nblk+1;
	      jl = (bcol/np_cols+1)*nblk+1;

	      if(((my_prow+np_rows)%np_rows) >= (brow%np_rows)){
		if(my_prow==prow){
		  il = il+(ig-1)%nblk;
		}
		il = il-nblk;
	      }

	      if(((my_pcol+np_cols)%np_cols) >= (bcol%np_cols)){
		if(my_pcol==pcol){
		  jl = jl+(jg-1)%nblk;
		}
		jl = jl-nblk;
	      }

	      Ss[(jl-1)*na_rows+il-1] += S1[k];
	    }
	    
	    k++;
	  }
	}
      }
    }

    /*
    printf("myid=%2d\n",myid);fflush(stdout);
    for(i=0; i<na_rows; i++){
      for(j=0; j<na_cols; j++){
        printf("%7.4f ",Ss[j*na_rows+i]);fflush(stdout);
      }
      printf("\n");fflush(stdout);
    }
    */
    
  } // if (spin==myworld1)

  /* freeing of arrays */

  free(S1);
  free(Stmp);
  free(Sc);
  free(order_GA);
  free(is2);
  free(ie1);
  free(is1);
  free(My_Matomnum);
  free(My_NZeros);
}

