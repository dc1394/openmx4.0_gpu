/**********************************************************************
  spectra.c

   spectra.c is a program which calculates a smeared spectrum from 
   excited energies and intensities.

      Usage:

         ./spectra input output

  Log of spectra.c:

     16/Apr./2020  Released by T. Ozaki 

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define eV2Hartree      27.2113845                

void XANES(int argc, char *argv[]);

int main(int argc, char *argv[]) 
{
  if (argc!=3){
    printf("Usage:\n");
    printf("  ./spectra input output\n");
    exit(0);
  }

  if (argv[1],".xanes") XANES(argc,argv);

}

void XANES(int argc, char *argv[]) 
{
  int num,num_mesh=2001;
  int i,j,jmin,jmax,Num_XANES_Out;
  double tmp0,tmp1,tmp2,tmp3,tmp4,tmp5;
  double rEx,rEy,rEz,iEx,iEy,iEz;
  double **XANES_Out,**XANES_Spec;
  double e,px,py,pz,rx,ry,rz,ix,iy,iz;
  double f,x,E,E0,Emin,Emax,dE,gwidth;
  char ctmp[4096];
  FILE *fp1,*fp2;
  char buf[1000];

  /*******************************************
              read the first file 
  *******************************************/

  if ((fp1 = fopen(argv[1],"r")) != NULL){

    /* get the amount of information */

    fgets(ctmp,1000,fp1);
    fgets(ctmp,1000,fp1);
    fgets(ctmp,1000,fp1);

    num = 0;
    while (fgets(ctmp, sizeof(ctmp), fp1) != NULL) {
      num++;
    }
    fclose(fp1);

    Num_XANES_Out = num;

    /* allocations of arrays */

    XANES_Out = (double**)malloc(sizeof(double*)*num);
    for (i=0; i<num; i++){
      XANES_Out[i] = (double*)malloc(sizeof(double)*10);
    }

    XANES_Spec = (double**)malloc(sizeof(double*)*num_mesh);
    for (i=0; i<num_mesh; i++){
      XANES_Spec[i] = (double*)malloc(sizeof(double)*6);
      for (j=0; j<6; j++){
        XANES_Spec[i][j] = 0.0;
      }
    }

    /* read data and store them */

    if ((fp1 = fopen(argv[1],"r")) != NULL){

      fgets(ctmp,1000,fp1);
      fgets(ctmp,1000,fp1);
      fgets(ctmp,1000,fp1);

      num = 0;
      while (fgets(ctmp, sizeof(ctmp), fp1) != NULL) {

	char *p = ctmp;
	while (*p == ' ' || *p == '\t' || *p == '\r') p++;
	if (*p == '\0' || *p == '\n') continue;

	int idx;
	double e, px, py, pz;

	if (sscanf(p, "%d %lf %lf %lf %lf %lf", &idx, &e, &px, &py, &pz) == 5) {

	  XANES_Out[num][0] = e;
	  XANES_Out[num][1] = px;
	  XANES_Out[num][2] = py;
	  XANES_Out[num][3] = pz;
	  num++;
	}
	else {
	}
      }
      fclose(fp1);
    }   

    /* find the lowest and highest energies */

    Emin = 1.0e+10;
    Emax =-1.0e+10;

    for (i=0; i<Num_XANES_Out; i++){
      if (XANES_Out[i][0]<Emin) Emin = XANES_Out[i][0];
      if (XANES_Out[i][0]>Emax) Emax = XANES_Out[i][0];
    }

    /* interactively ask the electric field vector for the monochromatic light */

    printf("Please input the real part of the electric field vector: (e.g., 1.0 0.0 1.0)\n"); 
    fgets(buf,1000,stdin); sscanf(buf,"%lf %lf %lf",&rEx,&rEy,&rEz); 

    printf("Please input the imaginary part of the electric field vector (e.g., 0.0 0.0 0.0)\n"); 
    fgets(buf,1000,stdin); sscanf(buf,"%lf %lf %lf",&iEx,&iEy,&iEz); 

    /* interactively ask the Gaussian width */

    printf("Please input a value of gaussian width (eV)\n"); 
    fgets(buf,1000,stdin); sscanf(buf,"%lf",&gwidth); 

    /* print the input data */

    if (0){
      printf("rEx=%15.12f rEy=%15.12f rEz=%15.12f\n",rEx,rEy,rEz);
      printf("iEx=%15.12f iEy=%15.12f iEz=%15.12f\n",iEx,iEy,iEz);
      printf("gwidth=%15.12f\n",gwidth);
    }

    /* normalization of E */

    tmp0 = 1.0/sqrt(rEx*rEx + iEx*iEx + rEy*rEy + iEy*iEy + rEz*rEz + iEz*iEz); 

    rEx = tmp0*rEx;    
    rEy = tmp0*rEy;    
    rEz = tmp0*rEz;    

    iEx = tmp0*iEx;    
    iEy = tmp0*iEy;    
    iEz = tmp0*iEz;    

    if (0){
      printf("rEx=%15.12f rEy=%15.12f rEz=%15.12f\n",rEx,rEy,rEz);
      printf("iEx=%15.12f iEy=%15.12f iEz=%15.12f\n",iEx,iEy,iEz);
      printf("gwidth=%15.12f\n",gwidth);
      exit(0);
    }

    /* update Emin and Emax */

    Emin -= 4.0*gwidth;
    Emax += 4.0*gwidth;
 
    /* set energies on mesh */

    dE = (Emax - Emin)/(double)(num_mesh-1);
    for (i=0; i<num_mesh; i++){
      XANES_Spec[i][0] = Emin + (double)i*dE;
    }

    /* calculate XANES_Spec */

    for (i=0; i<Num_XANES_Out; i++){
     
      E0 = XANES_Out[i][0];

      jmin = (int)((E0-4.0*gwidth-Emin)/dE);
      jmax = (int)((E0+4.0*gwidth-Emin)/dE);

      if (jmin<0)            jmin = 0;
      if ((num_mesh-1)<jmax) jmax = num_mesh - 1;

      for (j=jmin; j<=jmax; j++){

        x = (XANES_Spec[j][0]-E0)/gwidth;
        f = exp(-x*x);

        e = XANES_Spec[i][0]/eV2Hartree;
        px = XANES_Out[i][1];
        py = XANES_Out[i][2];
        pz = XANES_Out[i][3];

        rx = px*rEx;
        ry = py*rEy;
        rz = pz*rEz;

        ix = px*iEx;
        iy = py*iEy;
        iz = pz*iEz;

        //printf("%15.12f %15.12f %15.12f\n",py,ry,rEy);
        
        tmp1 = rx + ry + rz;
        tmp2 = ix + iy + iz;
        tmp3 = tmp1*tmp1 + tmp2*tmp2;

        XANES_Spec[j][1] += 2.0*f*tmp3/e;  // oscillator strength for the given E
        XANES_Spec[j][2] += 2.0*f*px*px/e; // x-component 
        XANES_Spec[j][3] += 2.0*f*py*py/e; // y-component 
        XANES_Spec[j][4] += 2.0*f*pz*pz/e; // z-component 
      }
    }

    /* output of results */

    if ((fp2 = fopen(argv[2],"w")) != NULL){

      fprintf(fp2,"# 1: Energy (eV)\n");
      fprintf(fp2,"# 2: Oscillator strength for the given E\n");
      fprintf(fp2,"# 3: x-component of oscillator strength, px^2/delta_E\n");
      fprintf(fp2,"# 4: y-component of oscillator strength, py^2/delta_E\n");
      fprintf(fp2,"# 5: z-component of oscillator strength, pz^2/delta_E\n");

      for (i=0; i<num_mesh; i++){
	fprintf(fp2,"%15.12f %15.12f %15.12f %15.12f %15.12f\n",
		XANES_Spec[i][0],
		XANES_Spec[i][1],
		XANES_Spec[i][2],
		XANES_Spec[i][3],
		XANES_Spec[i][4]);
      }

      /* fclose of fp2 */
      fclose(fp2);
    }

    /* freeing of arrays */

    for (i=0; i<num_mesh; i++){
      free(XANES_Spec[i]);
    }
    free(XANES_Spec);

    for (i=0; i<num; i++){
      free(XANES_Out[i]);
    }
    free(XANES_Out);

  }
  else{
    printf("error in scanfing %s\n",argv[1]);
  }
}


