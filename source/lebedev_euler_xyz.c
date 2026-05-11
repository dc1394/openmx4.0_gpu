#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define PI            3.1415926535897932384626

int main(int argc, char *argv[]) 
{
  int Np,i;
  double *leb_theta,*leb_phi,*leb_w;
  double x,y,z;
  FILE *fp;

  /*
  printf("argv[0]=%s\n",argv[0]);
  printf("argv[1]=%s\n",argv[1]);
  printf("argv[2]=%s\n",argv[2]);
  */

  Np = atoi(argv[1]);

  /* allocate arrays */

  leb_theta = (double*)malloc(sizeof(double)*Np);
  leb_phi = (double*)malloc(sizeof(double)*Np);
  leb_w = (double*)malloc(sizeof(double)*Np);

  /* read the file */

  if ((fp = fopen(argv[2],"r")) != NULL){

    for (i=0; i<Np; i++){
      fscanf(fp,"%lf %lf %lf",&leb_theta[i],&leb_phi[i],&leb_w[i]);

      leb_theta[i] = leb_theta[i]/180.0*PI;
      leb_phi[i] = leb_phi[i]/180.0*PI;
    }

    fclose(fp);
  }  

  /* Euler angle to xyz */

  for (i=0; i<Np; i++){


    x = cos(leb_theta[i])*sin(leb_phi[i]); 
    y = sin(leb_theta[i])*sin(leb_phi[i]); 
    z = cos(leb_phi[i]);

    //printf("%2d %15.12f %15.12f %15.12f  %15.12f %15.12f\n",i,leb_theta[i],leb_phi[i],leb_w[i],cos(leb_theta[i]),sin(leb_phi[i]));

    printf("Leb_Grid_XYZW[%5d][%d] = %18.15f;\n",i,0,x);
    printf("Leb_Grid_XYZW[%5d][%d] = %18.15f;\n",i,1,y);
    printf("Leb_Grid_XYZW[%5d][%d] = %18.15f;\n",i,2,z);
    printf("Leb_Grid_XYZW[%5d][%d] = %18.15f;\n\n",i,3,leb_w[i]);
   
  }




  /* free arrays */

  free(leb_theta);
  free(leb_phi);
  free(leb_w);


}
