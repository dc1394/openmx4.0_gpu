/**********************************************************************
  iprod_gcube.c

     iprod_gcube.c is a program which calculates the inner product of 
     two funtions expressed in the Gaussian cube format. 

      Usage:

         ./iprod_gcube input1.cube input2.cube

  Log of iprod_gcube.c:

     20/March/2023  Released by T. Ozaki 

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define fp_bsize         1048576     /* buffer size for setvbuf */

void Cross_Product(double a[4], double b[4], double c[4]);
double Dot_Product(double a[4], double b[4]);


int main(int argc, char *argv[]) 
{
  static int i,j,itmp,n1,n2,n3,po;
  static int atomnum1,atomnum2;
  static int Ngrid1_1,Ngrid1_2,Ngrid1_3;
  static int Ngrid2_1,Ngrid2_2,Ngrid2_3;
  static double **Gxyz1,**Gxyz2;
  static double ***CubeData1,***CubeData2;
  static double Grid_Origin1[4];
  static double Grid_Origin2[4];
  static double gtv1[4][4],gtv2[4][4];
  static double dtmp;
  static char ctmp[100];
  char buf[1000],buf2[1000],*c;
  FILE *fp1,*fp2;
  char fp_buf[fp_bsize];          /* setvbuf */

  if (argc!=3){
    printf("Usage:\n");
    printf("  ./iprod_gcube input1.cube input2.cube\n");
    exit(0);
  }

  /*******************************************
               read the first file 
  *******************************************/

  if ((fp1 = fopen(argv[1],"r")) != NULL){

#ifdef xt3
    setvbuf(fp1,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

    /* scanf cube tile */

    fscanf(fp1,"%s",ctmp);
    fscanf(fp1,"%s",ctmp);
    fscanf(fp1,"%d",&atomnum1);
    fscanf(fp1,"%lf %lf %lf",&Grid_Origin1[1],&Grid_Origin1[2],&Grid_Origin1[3]);
    fscanf(fp1,"%d",&Ngrid1_1);
    fscanf(fp1,"%lf %lf %lf",&gtv1[1][1],&gtv1[1][2],&gtv1[1][3]);
    fscanf(fp1,"%d",&Ngrid1_2);
    fscanf(fp1,"%lf %lf %lf",&gtv1[2][1],&gtv1[2][2],&gtv1[2][3]);
    fscanf(fp1,"%d",&Ngrid1_3);
    fscanf(fp1,"%lf %lf %lf",&gtv1[3][1],&gtv1[3][2],&gtv1[3][3]);

    /* allocation of arrays */

    Gxyz1 = (double**)malloc(sizeof(double*)*(atomnum1+1)); 
    for (i=0; i<(atomnum1+1); i++){
      Gxyz1[i] = (double*)malloc(sizeof(double)*4); 
    }

    CubeData1 = (double***)malloc(sizeof(double**)*Ngrid1_1); 
    for (i=0; i<Ngrid1_1; i++){
      CubeData1[i] = (double**)malloc(sizeof(double*)*Ngrid1_2); 
      for (j=0; j<Ngrid1_2; j++){
        CubeData1[i][j] = (double*)malloc(sizeof(double)*Ngrid1_3); 
      }
    }

    /* scanf xyz coordinates */

    for (i=1; i<=atomnum1; i++){
      fscanf(fp1,"%lf %lf %lf %lf %lf",&Gxyz1[i][0],&dtmp,&Gxyz1[i][1],&Gxyz1[i][2],&Gxyz1[i][3]);
    }

    /* scanf cube data */

    for (n1=0; n1<Ngrid1_1; n1++){
      for (n2=0; n2<Ngrid1_2; n2++){
        for (n3=0; n3<Ngrid1_3; n3++){
          fscanf(fp1,"%lf",&CubeData1[n1][n2][n3]);
	}
      }
    }

    fclose(fp1);
  }
  else{
    printf("error in scanfing %s\n",argv[1]);
  }

  /*******************************************
               scanf the second file 
  *******************************************/

  if ((fp2 = fopen(argv[2],"r")) != NULL){

    /* scanf cube tile */

    fscanf(fp2,"%s",ctmp);
    fscanf(fp2,"%s",ctmp);
    fscanf(fp2,"%d",&atomnum2);
    fscanf(fp2,"%lf %lf %lf",&Grid_Origin2[1],&Grid_Origin2[2],&Grid_Origin2[3]);
    fscanf(fp2,"%d",&Ngrid2_1);
    fscanf(fp2,"%lf %lf %lf",&gtv2[1][1],&gtv2[1][2],&gtv2[1][3]);
    fscanf(fp2,"%d",&Ngrid2_2);
    fscanf(fp2,"%lf %lf %lf",&gtv2[2][1],&gtv2[2][2],&gtv2[2][3]);
    fscanf(fp2,"%d",&Ngrid2_3);
    fscanf(fp2,"%lf %lf %lf",&gtv2[3][1],&gtv2[3][2],&gtv2[3][3]);

    /* allocation of arrays */

    Gxyz2 = (double**)malloc(sizeof(double*)*(atomnum2+1)); 
    for (i=0; i<(atomnum2+1); i++){
      Gxyz2[i] = (double*)malloc(sizeof(double)*4); 
    }

    CubeData2 = (double***)malloc(sizeof(double**)*Ngrid2_1); 
    for (i=0; i<Ngrid2_1; i++){
      CubeData2[i] = (double**)malloc(sizeof(double*)*Ngrid2_2); 
      for (j=0; j<Ngrid2_2; j++){
        CubeData2[i][j] = (double*)malloc(sizeof(double)*Ngrid2_3); 
      }
    }

    /* check */
   
    po = 0;

    if (Ngrid1_1!=Ngrid2_1){
      printf("Found a difference in the number of grid on a-axis\n");  
      po = 1;      
    }
    
    if (Ngrid1_2!=Ngrid2_2){
      printf("Found a difference in the number of grid on b-axis\n");  
      po = 1;      
    }

    if (Ngrid1_3!=Ngrid2_3){
      printf("Found a difference in the number of grid on c-axis\n");  
      po = 1;      
    }

    if (atomnum1!=atomnum2){
      printf("Found a difference in the number of atoms\n");  
      po = 1;      
    }

    if (Grid_Origin1[1]!=Grid_Origin2[1]){
      printf("Found a difference in x-coordinate of the origin\n");  
      po = 1;      
    }

    if (Grid_Origin1[2]!=Grid_Origin2[2]){
      printf("Found a difference in y-coordinate of the origin\n");  
      po = 1;      
    }

    if (Grid_Origin1[3]!=Grid_Origin2[3]){
      printf("Found a difference in z-coordinate of the origin\n");  
      po = 1;      
    }

    if ( (gtv1[1][1]!=gtv2[1][1]) 
        || 
         (gtv1[1][2]!=gtv2[1][2]) 
        || 
         (gtv1[1][3]!=gtv2[1][3]) 
       ){
      printf("Found a difference in the vector of a-axis\n");  
      po = 1;      
    }

    if ( (gtv1[2][1]!=gtv2[2][1]) 
        || 
         (gtv1[2][2]!=gtv2[2][2]) 
        || 
         (gtv1[2][3]!=gtv2[2][3]) 
       ){
      printf("Found a difference in the vector of b-axis\n");  
      po = 1;      
    }

    if ( (gtv1[3][1]!=gtv2[3][1]) 
        || 
         (gtv1[3][2]!=gtv2[3][2]) 
        || 
         (gtv1[3][3]!=gtv2[3][3]) 
       ){
      printf("Found a difference in the vector of c-axis\n");  
      po = 1;      
    }

    /*
    if (po==1){
      do {
        printf("Are you sure you want to continue?, yes(1) or no(2)\n");
        fgets(buf,1000,stdin); sscanf(buf,"%d",itmp);

        if      (itmp==2) exit(0);
        else if (itmp==1) po = 0;

      } while (po==1);
    }   
    */

    /* scanf xyz coordinates */

    for (i=1; i<=atomnum2; i++){
      fscanf(fp2,"%lf %lf %lf %lf %lf",&Gxyz2[i][0],&dtmp,&Gxyz2[i][1],&Gxyz2[i][2],&Gxyz2[i][3]);
    }

    /* scanf cube data */

    for (n1=0; n1<Ngrid2_1; n1++){
      for (n2=0; n2<Ngrid2_2; n2++){
        for (n3=0; n3<Ngrid2_3; n3++){
          fscanf(fp2,"%lf",&CubeData2[n1][n2][n3]);
	}
      }
    }

    fclose(fp2);
  }
  else{
    printf("error in reading %s\n",argv[2]);
  }

  /*******************************************
         calculate the inner product
  *******************************************/

  double CellV_grid,tmp[4],sum;

  Cross_Product(gtv1[2],gtv1[3],tmp);
  CellV_grid = fabs(Dot_Product(gtv1[1],tmp)); 

  sum = 0.0;

  for (n1=0; n1<Ngrid1_1; n1++){
    for (n2=0; n2<Ngrid1_2; n2++){
      for (n3=0; n3<Ngrid1_3; n3++){
	sum += CubeData1[n1][n2][n3]*CubeData2[n1][n2][n3];
      }
    }
  }

  sum *= CellV_grid;

  printf("The inner product: %15.12f\n",sum);

  /* freeing of arrays */

  for (i=0; i<(atomnum1+1); i++){
    free(Gxyz1[i]);
  }
  free(Gxyz1);

  for (i=0; i<Ngrid1_1; i++){
    for (j=0; j<Ngrid1_2; j++){
      free(CubeData1[i][j]);
    }
    free(CubeData1[i]);
  }
  free(CubeData1);

  for (i=0; i<(atomnum2+1); i++){
    free(Gxyz2[i]);
  }
  free(Gxyz2);

  for (i=0; i<Ngrid2_1; i++){
    for (j=0; j<Ngrid2_2; j++){
      free(CubeData2[i][j]);
    }
    free(CubeData2[i]);
  }
  free(CubeData2);
}



void Cross_Product(double a[4], double b[4], double c[4])
{
  c[1] = a[2]*b[3] - a[3]*b[2]; 
  c[2] = a[3]*b[1] - a[1]*b[3]; 
  c[3] = a[1]*b[2] - a[2]*b[1];
}

double Dot_Product(double a[4], double b[4])
{
  double sum;
  sum = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]; 
  return sum;
}


