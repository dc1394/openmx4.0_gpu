/**********************************************************************
  Exnapnd_ReCoulomb_Kernel.c:

   Exnapnd_ReCoulomb_Kernel.c is a subroutine to expand the reciprocal 
   Coulomb kernel into a separable form of \sum_{i} \lambda_i A_i(G1)B_i{G2}.

  Log of Exnapnd_ReCoulomb_Kernel.c:

   10/Feb./2024  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "openmx_common.h"
#include "mpi.h"

void Exnapnd_ReCoulomb_Kernel()
{
  int i,j,k,p,i1,j1,k1,i2,j2,k2,Ng1,Nrank,Ndim,n,m;
  int *jun_sv;
  double x1,y1,z1,x2,y2,z2,anal;
  double q,dq,qmin,qmax,c,sx,sy,sz,max_diff,tensor;
  double *A,*xgrid,*ko;

  /* set parameters */
  
  Ng1 = Ng1_Rec_Coulomb;
  Nrank = Nrank_Rec_Coulomb;
  Ndim = Ng1*Ng1*Ng1;
  qmin = xmin_Rec_Coulomb;
  qmax = xmax_Rec_Coulomb;
  dq = (qmax-qmin)/(double)(Ng1-1);
  c = Yukawa_Exponent_Rec_Coulomb;
    
  /* allocation of arrays */

  xgrid = (double*)malloc(sizeof(double)*Ng1);
  A = (double*)malloc(sizeof(double)*Ndim*Ndim);
  ko = (double*)malloc(sizeof(double)*Ndim);
  jun_sv = (int*)malloc(sizeof(int)*Ndim);

  /* set xgrid */
  
  for (i1=0; i1<Ng1; i1++){
    xgrid[i1] = dq*(double)i1;
    if ((0.50*qmax)<=xgrid[i1]) xgrid[i1] = xgrid[i1] - qmax;
  }

  /* construct the matrix by the discretization of 1/(|r1+r2|^2 + c) */

  n = 0;
  for (i1=0; i1<Ng1; i1++){
    x1 = xgrid[i1];
    for (j1=0; j1<Ng1; j1++){
      y1 = xgrid[j1];
      for (k1=0; k1<Ng1; k1++){
        z1 = xgrid[k1];

	m = 0;
	for (i2=0; i2<Ng1; i2++){
          x2 = xgrid[i2];
	  sx = x1 + x2;
	  for (j2=0; j2<Ng1; j2++){
            y2 = xgrid[j2];
	    sy = y1 + y2;
	    for (k2=0; k2<Ng1; k2++){
              z2 = xgrid[k2];
              sz = z1 + z2;

	      A[m*Ndim+n] = 1.0/(sx*sx+sy*sy+sz*sz+c);
	      
	      m++;  
	    }
	  }
	}

	n++;
      }
    }
  }

  /* singular value decomposition (SVD) of A */

  /*
  printf("Original\n");
  for (i=0; i<Ndim; i++){
    for (j=0; j<Ndim; j++){
      printf("%8.4f ",A[i*Ndim+j]);
    }
    printf("\n");
  }
  */
  
  Eigen_lapack3(A, ko, Ndim, Ndim);

  /*
  printf("Tensor\n");
  for (i=0; i<Ndim; i++){
    for (j=0; j<Ndim; j++){

      tensor = 0.0;
      for (k=0; k<Ndim; k++){
        tensor += ko[k]*A[k*Ndim+i]*A[k*Ndim+j];
      }
      printf("%8.4f ",tensor);
    }
    printf("\n");
  }
  */

  
  for (i=0; i<Ndim; i++){
    SVals_Rec_Coulomb[i] = fabs(ko[i]);
    jun_sv[i] = i; 
  }

  qsort_double_int2(Ndim, SVals_Rec_Coulomb, jun_sv);

  for (i=0; i<Ndim; i++){
    printf("ABC1 i=%2d sv=%15.12f jun=%2d\n",i,SVals_Rec_Coulomb[i],jun_sv[i]);
  }


  /*
  for (i=0; i<Ndim; i++){
    printf("ABC2 i=%2d ko=%15.12f jun=%2d\n",i,ko[i],jun_sv[i]);
  }
  */

  /* the sign is attached to the singular value */
  for (i=0; i<Ndim; i++){
    k = jun_sv[i];
    SVals_Rec_Coulomb[i] = ko[k];
  }

  /* store singular vectors */ 

  for (i=0; i<Nrank; i++){
    k = jun_sv[i];
    for (j=0; j<Ndim; j++){
      SVecs_Rec_Coulomb[i][j] = A[k*Ndim+j];
    }
  }

  /* check the accuracy of the approximation */ 
  
  max_diff = -1.0;
  
  m = 0;
  for (i1=0; i1<Ng1; i1++){
    x1 = xgrid[i1];
    for (j1=0; j1<Ng1; j1++){
      y1 = xgrid[j1];
      for (k1=0; k1<Ng1; k1++){
        z1 = xgrid[k1];

	n = 0;
	for (i2=0; i2<Ng1; i2++){
          x2 = xgrid[i2];
	  sx = x1 + x2;
	  for (j2=0; j2<Ng1; j2++){
            y2 = xgrid[j2];
	    sy = y1 + y2;
	    for (k2=0; k2<Ng1; k2++){
              z2 = xgrid[k2];
              sz = z1 + z2;

	      anal = 1.0/(sx*sx+sy*sy+sz*sz+c);

	      tensor = 0.0;
              for (p=0; p<Nrank; p++){
                tensor += SVals_Rec_Coulomb[p]*SVecs_Rec_Coulomb[p][m]*SVecs_Rec_Coulomb[p][n];
	      }

              if (max_diff<fabs(anal-tensor)) max_diff = fabs(anal-tensor);
	      
	      n++;  
	    }
	  }
	}

	m++;
      }
    }
  }

  printf("Nrank=%2d max_diff=%15.12f\n",Nrank,max_diff);
  
  MPI_Finalize();
  exit(0);

  
  /* freeing of arrays */

  free(xgrid);
  free(A);
  free(ko);
  free(jun_sv);
  
}

double Approximate_ReCoulomb_Kernel(
       double x1, double y1, double z1,
       double x2, double y2, double z2 )
{
  int i,i1,j1,k1,i2,j2,k2;
  int Ng1,Nrank,Ndim,qmin;
  double qmin,qmax,dq,c;
  
  Ng1 = Ng1_Rec_Coulomb;
  Nrank = Nrank_Rec_Coulomb;
  Ndim = Ng1*Ng1*Ng1;
  qmin = xmin_Rec_Coulomb;
  qmax = xmax_Rec_Coulomb;
  dq = (qmax-qmin)/(double)(Ng1-1);
  c = Yukawa_Exponent_Rec_Coulomb;
  
  for (i1=0; i1<Ng1; i1++){
    xg[i1] = dq*(double)i1;
  }

  for (i=0; i<Ng1; i++){
    if ((dq*(double)i)<=x1 && x1<(dq*(double)(i+1))) i1 = i;
  }

  for (i=0; i<Ng1; i++){
    if ((dq*(double)i)<=y1 && y1<(dq*(double)(i+1))) j1 = i;
  }
  
  for (i=0; i<Ng1; i++){
    if ((dq*(double)i)<=z1 && z1<(dq*(double)(i+1))) k1 = i;
  }

  for (i=0; i<Ng1; i++){
    if ((dq*(double)i)<=x2 && x2<(dq*(double)(i+1))) i2 = i;
  }

  for (i=0; i<Ng1; i++){
    if ((dq*(double)i)<=y2 && y2<(dq*(double)(i+1))) j2 = i;
  }
  
  for (i=0; i<Ng1; i++){
    if ((dq*(double)i)<=z2 && z2<(dq*(double)(i+1))) k2 = i;
  }
  
  

  
}





