#include <stdio.h>
#include <stdlib.h>  
#include <math.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

double rnd(double width);

int main(int argc, char *argv[]) 
{
  int I,J,i,j,k,i1,j1,k1,i2,j2,k2,m,n;
  int Nd1,Ng1,Nrank,loop,po;
  double xmin,xmax,dx1,x,y,z,x1,x2,y1,y2,z1,z2;
  double tensor,d,F,G,max_diff,xd,yd,zd,xd1,yd1,zd1;
  double ai,aj,ak,sumA,sumB,lambdaF,lambdaG;
  double ScaleFactor;
  double xmin2,xmax2,dx2,xx1,yy1,zz1,xx2,yy2,zz2;
  double fx1,fy1,fz1,fx2,fy2,fz2;
  double al = 1.0, be=0.1;
  double ****Vec1,****Vec2;
  double ****dFdA,****dFdB;

  be = pow(1.0/be,1.0/6.0);
  Nd1 = 2;
  Nrank = Nd1*Nd1*Nd1;
  
  //Nrank = 4;
  
  Ng1 = 12;
  lambdaF = 1.0;
  lambdaG = 0.0;
  
  xmin =  0.0;
  xmax = 12.0;  
  xmin2 = sqrt(xmin);
  xmax2 = sqrt(xmax);

  dx1 = (xmax - xmin)/(double)(Ng1-1);
  dx2 = (xmax2 - xmin2)/(double)(Ng1-1);

  Vec1 = (double****)malloc(sizeof(double***)*Nrank);
  for (i=0; i<Nrank; i++){
    Vec1[i] = (double***)malloc(sizeof(double**)*Ng1);
    for (j=0; j<Ng1; j++){
      Vec1[i][j] = (double**)malloc(sizeof(double*)*Ng1);
      for (k=0; k<Ng1; k++){
        Vec1[i][j][k] = (double*)malloc(sizeof(double)*Ng1);
      }
    }
  }
  
  Vec2 = (double****)malloc(sizeof(double***)*Nrank);
  for (i=0; i<Nrank; i++){
    Vec2[i] = (double***)malloc(sizeof(double**)*Ng1);
    for (j=0; j<Ng1; j++){
      Vec2[i][j] = (double**)malloc(sizeof(double*)*Ng1);
      for (k=0; k<Ng1; k++){
        Vec2[i][j][k] = (double*)malloc(sizeof(double)*Ng1);
      }
    }
  }

  dFdA = (double****)malloc(sizeof(double***)*Nrank);
  for (i=0; i<Nrank; i++){
    dFdA[i] = (double***)malloc(sizeof(double**)*Ng1);
    for (j=0; j<Ng1; j++){
      dFdA[i][j] = (double**)malloc(sizeof(double*)*Ng1);
      for (k=0; k<Ng1; k++){
        dFdA[i][j][k] = (double*)malloc(sizeof(double)*Ng1);
      }
    }
  }

  dFdB = (double****)malloc(sizeof(double***)*Nrank);
  for (i=0; i<Nrank; i++){
    dFdB[i] = (double***)malloc(sizeof(double**)*Ng1);
    for (j=0; j<Ng1; j++){
      dFdB[i][j] = (double**)malloc(sizeof(double*)*Ng1);
      for (k=0; k<Ng1; k++){
        dFdB[i][j][k] = (double*)malloc(sizeof(double)*Ng1);
      }
    }
  }
  
  /* setting initial factor matrices */

  if (0){
  
    for (I=0; I<Nrank; I++){

      /* set Vec1 and Vec2 on grid */
	
      for (i1=0; i1<Ng1; i1++){

	xx1 = xmin2 + (double)i1*dx2;
	x = xx1*xx1;

	if      (I==0) { fx1 = 1.47*exp(-x*x);    fx2 = 1.47*exp(-x*x);   }
	else if (I==1) { fx1 = -x*exp(-2.0*x*x);  fx2 = -x*exp(-2.0*x*x);   }
	else if (I==2) { fx1 = x*x*exp(-2.0*x*x); fx2 = x*x*exp(-2.0*x*x);    }
	else if (I==3) { fx1 = exp(-5.0*x);       fx2 = exp(-5.0*x);   }
	else if (I==4) { fx1 = exp(-12.0*x);      fx2 = exp(-12.0*x);   }
	else {
	  fx1 = rnd(1.0);  fx2 = rnd(1.0);
	}
      
	for (j1=0; j1<Ng1; j1++){
	    
	  yy1 = xmin2 + (double)j1*dx2;
	  y = yy1*yy1;
	  
	  if      (I==0) { fy1 = 1.47*exp(-y*y);    fy2 = 1.47*exp(-y*y);     }
	  else if (I==1) { fy1 = -y*exp(-2.0*y*y);  fy2 = -y*exp(-2.0*y*y);   }
	  else if (I==2) { fy1 = y*y*exp(-2.0*y*y); fy2 = y*y*exp(-2.0*y*y);    }
	  else if (I==3) { fy1 = exp(-5.0*y);       fy2 = exp(-5.0*y);   }
	  else if (I==4) { fy1 = exp(-12.0*y);      fy2 = exp(-12.0*y);   }
	  else {
	    fy1 = rnd(1.0);  fy2 = rnd(1.0);
	  }
	      
	  for (k1=0; k1<Ng1; k1++){
	      
	    zz1 = xmin2 + (double)k1*dx2;
	    z = zz1*zz1;
	    
	    if      (I==0) { fz1 = 1.47*exp(-z*z);    fz2 = 1.47*exp(-z*z);  }
	    else if (I==1) { fz1 = -z*exp(-2.0*z*z);  fz2 = -z*exp(-2.0*z*z);  }
	    else if (I==2) { fy1 = z*z*exp(-2.0*y*y); fy2 = y*y*exp(-2.0*y*y);    }
	    else if (I==2) { fz1 = exp(-5.0*z); fz2 = exp(-5.0*z);  }
	    else if (I==3) { fz1 = -exp(-14.0*(z-1.0)*(z-1.0)); fz2 = -exp(-14.0*(z-1.0)*(z-1.0));  }
	    else if (I==4) { fz1 = exp(-12.0*z); fz2 = exp(-12.0*z);  }
	    else {
	      fz1 = rnd(1.0);  fz2 = rnd(1.0);
	    }
		
	    Vec1[I][i1][j1][k1] = fx1*fy1*fz1;
	    Vec2[I][i1][j1][k1] = fx2*fy2*fz2;

	    //printf("ABC I=%2d i1=%2d j1=%2d k1=%2d %15.12f\n",I,i1,j1,k1,Vec1[I][i1][j1][k1]);
	  }
	}
      }
    }
  }

  else{

    I = 0;
    for (i=0; i<Nd1; i++){
      for (j=0; j<Nd1; j++){
	for (k=0; k<Nd1; k++){

	  /* set Vec1 and Vec2 on grid */

	  for (i1=0; i1<Ng1; i1++){

	    xx1 = xmin2 + (double)i1*dx2;
	    x = xx1*xx1;

	    if      (i==0) { fx1 = exp(-al*x*x);                     fx2 = fx1;  }
	    else if (i==1) { fx1 =-0.1/(x+1.0);                      fx2 = 0.1*x;    }

	    /*
	    else if (i==1) { fx1 =-2.0*al*x*exp(-al*x*x);            fx2 = fx1;  }
	    else if (i==2) { fx1 = 2.0*al*al*x*x*exp(-al*x*x);       fx2 = fx1;  }
	    else if (i==3) { fx1 =-0.75*al*al*al*x*x*x*exp(-al*x*x); fx2 = fx1;  }
	    */
	    
	    for (j1=0; j1<Ng1; j1++){
	    
	      yy1 = xmin2 + (double)j1*dx2;
	      y = yy1*yy1;

	      if      (j==0) { fy1 = exp(-al*y*y);                     fy2 = fy1;  }
	      else if (j==1) { fy1 =-0.1/(y+1.0);                      fy2 = 0.1*y;   }

	      /*
	      else if (j==1) { fy1 =-2.0*al*y*exp(-al*y*y);            fy2 = fy1;  }
	      else if (j==2) { fy1 = 2.0*al*al*y*y*exp(-al*y*y);       fy2 = fy1;  }
	      else if (j==3) { fy1 =-0.75*al*al*al*y*y*y*exp(-al*y*y); fy2 = fy1;  }
	      */
	      
	      for (k1=0; k1<Ng1; k1++){
	      
		zz1 = xmin2 + (double)k1*dx2;
		z = zz1*zz1;

		if      (k==0) { fz1 = exp(-al*z*z);                     fz2 = fz1;  }
		else if (k==1) { fz1 =-0.1/(z+1.0);                      fz2 = 0.1*z;    }

		/*
		else if (k==1) { fz1 =-2.0*al*z*exp(-al*z*z);            fz2 = fz1;  }
		else if (k==2) { fz1 = 2.0*al*al*z*z*exp(-al*z*z);       fz2 = fz1;  }
		else if (k==3) { fz1 =-0.75*al*al*al*z*z*z*exp(-al*z*z); fz2 = fz1;  }
		*/
		
		Vec1[I][i1][j1][k1] = fx1*fy1*fz1;
		Vec2[I][i1][j1][k1] = fx2*fy2*fz2;

		//printf("ABC I=%2d i1=%2d j1=%2d k1=%2d %15.12f\n",I,i1,j1,k1,Vec1[I][i1][j1][k1]);
	  
	      }
	    }
	  }

	  I++;
	}
      }
    }

  }
  
  /* optimization loop */

  ScaleFactor = 1.0e-5;
  po = 0;
  loop = 1;
  
  do {
  
    /* calculation of F */

    F = 0.0;
    max_diff = -1.0;
    for (i1=0; i1<Ng1; i1++){

      xx1 = xmin2 + (double)i1*dx2;
      x1 = xx1*xx1;
      
      for (j1=0; j1<Ng1; j1++){
	
	yy1 = xmin2 + (double)j1*dx2;
        y1 = yy1*yy1; 
	
	for (k1=0; k1<Ng1; k1++){

	  zz1 = xmin2 + (double)k1*dx2;
          z1 = zz1*zz1;
	  
	  for (i2=0; i2<Ng1; i2++){
	    
	    xx2 = xmin2 + (double)i2*dx2;
            x2 = xx2*xx2;
	    xd = x1 + x2;
	    
	    for (j2=0; j2<Ng1; j2++){
	      
	      yy2 = xmin2 + (double)j2*dx2;
	      y2 = yy2*yy2;
	      yd = y1 + y2;	    

	      for (k2=0; k2<Ng1; k2++){
		
		zz2 = xmin2 + (double)k2*dx2;
		z2 = zz2*zz2;
		zd = z1 + z2;	    

		tensor = 0.0;
		for (I=0; I<Nrank; I++){
		  tensor += Vec1[I][i1][j1][k1]*Vec2[I][i2][j2][k2];
		}

		d = tensor - 1.0/(xd*xd + yd*yd + zd*zd + 0.1);

		if (max_diff<fabs(d)){
		  max_diff = fabs(d);
		  xd1 = xd;
		  yd1 = yd;
		  zd1 = zd;
		}

		/*
		if (loop%4000==0){
                printf("BBB loop=%2d i1=%2d j1=%2d k1=%2d i2=%2d j2=%2d k2=%2d %15.12f %15.12f %15.12f\n",
		       loop,i1,j1,k1,i2,j2,k2,tensor,1.0/(xd*xd + yd*yd + zd*zd + 0.1),d);
		}
		*/
		
		F += lambdaF*d*d;  

	      }
	    }
	  }
	}
      }
    }

    /* calculation of dF/dA and dF/dB */

    for (I=0; I<Nrank; I++){

      for (i1=0; i1<Ng1; i1++){

        xx1 = xmin2 + (double)i1*dx2;
        x1 = xx1*xx1;
	
	for (j1=0; j1<Ng1; j1++){

    	  yy1 = xmin2 + (double)j1*dx2;
          y1 = yy1*yy1; 

	  for (k1=0; k1<Ng1; k1++){
	    
   	    zz1 = xmin2 + (double)k1*dx2;
            z1 = zz1*zz1;

	    sumA = 0.0;
	    sumB = 0.0;
	  
	    for (i2=0; i2<Ng1; i2++){

	      xx2 = xmin2 + (double)i2*dx2;
	      x2 = xx2*xx2;
	      xd = x1 + x2;

	      for (j2=0; j2<Ng1; j2++){
		
		yy2 = xmin2 + (double)j2*dx2;
		y2 = yy2*yy2;
		yd = y1 + y2;	    
		
		for (k2=0; k2<Ng1; k2++){

		  zz2 = xmin2 + (double)k2*dx2;
		  z2 = zz2*zz2;
		  zd = z1 + z2;	    

		  tensor = 0.0;
		  for (J=0; J<Nrank; J++){
		    tensor += Vec1[J][i1][j1][k1]*Vec2[J][i2][j2][k2];
		  }

		  d = tensor - 1.0/(xd*xd + yd*yd + zd*zd + 0.1);

		  sumA += 2.0*d*Vec2[I][i2][j2][k2];
		  sumB += 2.0*d*Vec1[I][i2][j2][k2];

		}
	      }
	    }

	    dFdA[I][i1][j1][k1] = lambdaF*sumA;           
	    dFdB[I][i1][j1][k1] = lambdaF*sumB;           

	  }
	}
      }
    }

    /* calculation of G */
    
    G = 0.0;
    for (I=0; I<Nrank; I++){

      for (i1=0; i1<(Ng1-1); i1++){
	for (j1=0; j1<Ng1; j1++){
	  for (k1=0; k1<Ng1; k1++){

	    d = Vec1[I][i1+1][j1][k1] - Vec1[I][i1][j1][k1]; 
            G += lambdaG*d*d;

	    d = Vec2[I][i1+1][j1][k1] - Vec2[I][i1][j1][k1]; 
            G += lambdaG*d*d;
	  }
	}
      }

      for (j1=0; j1<(Ng1-1); j1++){
        for (i1=0; i1<Ng1; i1++){
	  for (k1=0; k1<Ng1; k1++){

	    d = Vec1[I][i1][j1+1][k1] - Vec1[I][i1][j1][k1]; 
            G += lambdaG*d*d;

	    d = Vec2[I][i1][j1+1][k1] - Vec2[I][i1][j1][k1]; 
            G += lambdaG*d*d;
	  }
	}
      }

      for (k1=0; k1<(Ng1-1); k1++){
        for (i1=0; i1<Ng1; i1++){
   	  for (j1=0; j1<Ng1; j1++){

	    d = Vec1[I][i1][j1][k1+1] - Vec1[I][i1][j1][k1]; 
            G += lambdaG*d*d;

	    d = Vec2[I][i1][j1][k1+1] - Vec2[I][i1][j1][k1]; 
            G += lambdaG*d*d;
	  }
	}
      }
    }
    
    /* calculation of dG/dA and dG/dB */

    for (I=0; I<Nrank; I++){

      /* along i1 */ 

      for (i1=0; i1<(Ng1-1); i1++){
	for (j1=0; j1<Ng1; j1++){
	  for (k1=0; k1<Ng1; k1++){
            dFdA[I][i1][j1][k1] += -2.0*lambdaG*(Vec1[I][i1+1][j1][k1] - Vec1[I][i1][j1][k1]);
            dFdB[I][i1][j1][k1] += -2.0*lambdaG*(Vec2[I][i1+1][j1][k1] - Vec2[I][i1][j1][k1]);
	  }
	}
      }

      for (i1=1; i1<Ng1; i1++){
	for (j1=0; j1<Ng1; j1++){
	  for (k1=0; k1<Ng1; k1++){
            dFdA[I][i1][j1][k1] += 2.0*lambdaG*(Vec1[I][i1][j1][k1] - Vec1[I][i1-1][j1][k1]);
            dFdB[I][i1][j1][k1] += 2.0*lambdaG*(Vec2[I][i1][j1][k1] - Vec2[I][i1-1][j1][k1]);
	  }
	}
      }

      /* along j1 */ 
      
      for (i1=0; i1<Ng1; i1++){
	for (j1=0; j1<(Ng1-1); j1++){
	  for (k1=0; k1<Ng1; k1++){
            dFdA[I][i1][j1][k1] += -2.0*lambdaG*(Vec1[I][i1][j1+1][k1] - Vec1[I][i1][j1][k1]);
            dFdB[I][i1][j1][k1] += -2.0*lambdaG*(Vec2[I][i1][j1+1][k1] - Vec2[I][i1][j1][k1]);
	  }
	}
      }

      for (i1=0; i1<Ng1; i1++){
	for (j1=1; j1<Ng1; j1++){
	  for (k1=0; k1<Ng1; k1++){
            dFdA[I][i1][j1][k1] += 2.0*lambdaG*(Vec1[I][i1][j1][k1] - Vec1[I][i1][j1-1][k1]);
            dFdB[I][i1][j1][k1] += 2.0*lambdaG*(Vec2[I][i1][j1][k1] - Vec2[I][i1][j1-1][k1]);
	  }
	}
      }

      /* along k1 */ 
      
      for (i1=0; i1<Ng1; i1++){
	for (j1=0; j1<Ng1; j1++){
	  for (k1=0; k1<(Ng1-1); k1++){
            dFdA[I][i1][j1][k1] += -2.0*lambdaG*(Vec1[I][i1][j1][k1+1] - Vec1[I][i1][j1][k1]);
            dFdB[I][i1][j1][k1] += -2.0*lambdaG*(Vec2[I][i1][j1][k1+1] - Vec2[I][i1][j1][k1]);
	  }
	}
      }

      for (i1=0; i1<Ng1; i1++){
	for (j1=0; j1<Ng1; j1++){
	  for (k1=1; k1<Ng1; k1++){
            dFdA[I][i1][j1][k1] += 2.0*lambdaG*(Vec1[I][i1][j1][k1] - Vec1[I][i1][j1][k1-1]);
            dFdB[I][i1][j1][k1] += 2.0*lambdaG*(Vec2[I][i1][j1][k1] - Vec2[I][i1][j1][k1-1]);
	  }
	}
      }
    }
    
    /* update the factor matrices */

    for (I=0; I<Nrank; I++){

      m = 0;
      for (i1=0; i1<Ng1; i1++){
	for (j1=0; j1<Ng1; j1++){
	  for (k1=0; k1<Ng1; k1++){

            Vec1[I][i1][j1][k1] -= ScaleFactor*dFdA[I][i1][j1][k1];
            Vec2[I][i1][j1][k1] -= ScaleFactor*dFdB[I][i1][j1][k1];
	    
	    m++;
	  }
	}
      }
    }

    printf("loop=%3d F=%12.8f G=%12.8f F+G=%12.8f max_diff=%12.8f %6.3f %6.3f %6.3f\n",
	   loop,F,G,F+G,max_diff,xd1,yd1,zd1);  

    /*
    if      (F<20000.0) ScaleFactor = 0.001;
    else if (F<10000.0) ScaleFactor = 0.003;
    else if (F<5000.0)  ScaleFactor = 0.007;
    else if (F<1000.0)  ScaleFactor = 0.02;
    else if (F<500.0)   ScaleFactor = 0.04;
    else if (F<100.0)   ScaleFactor = 0.07;
    else if (F<10.0)    ScaleFactor = 0.1;
    */

    if      (F<20000.0) ScaleFactor = 0.0001;
    else if (F<15000.0) ScaleFactor = 0.0060;
    else if (F<12000.0) ScaleFactor = 0.0100;
    else if (F<7000.0)  ScaleFactor = 0.010;
    else if (F<1000.0)  ScaleFactor = 0.03;
    else if (F<500.0)   ScaleFactor = 0.10;
    else if (F<100.0)   ScaleFactor = 0.20;
    else if (F<10.0)    ScaleFactor = 0.30;
    
    loop++;

    if (F<1.0e-7) po = 1;
    
  } while (po==0 && loop<200000);  

  I = 1;
  for (i1=0; i1<Ng1; i1++){
    for (j1=0; j1<Ng1; j1++){
      for (k1=0; k1<Ng1; k1++){
	printf("I=%2d i1=%2d j1=%2d k1=%2d Vec=%15.12f %15.12f\n",
	       I,i1,j1,k1,Vec1[I][i1][j1][k1],Vec2[I][i1][j1][k1]);
      }
    }
  }

  exit(0);

}

/*
double TensorF(double x1, double y1, double z1, double x2, double y2, double z2,
	       int Nd1, int Nrank, double xmin, double xmax, double dx,
	       double Vec1[125][10][10][10], double Vec2[125][10][10][10])
{
  int i1,i2,j1,j2,k1,k2,n;
  double a[64];
  double x,y,z;

  i1 = (int)((x1-xmin)/dx);
  i2 = (int)((x2-xmin)/dx);
  j1 = (int)((y1-xmin)/dx);
  j2 = (int)((y2-xmin)/dx);
  k1 = (int)((z1-xmin)/dx);
  k2 = (int)((z2-xmin)/dx);

  n = 0;
  for (i=i1; i<=(i1+1); i++){
    x = xmin + (double)i*dx;
      
    for (j=j1; j<=(j1+1); j++){
      y = xmin + (double)j*dx;
      for (k=k1; k<=(k1+1); k++){
        z = xmin + (double)k*dx;
        a[n] = pow();
	
      }
    }
  }
  
  1, x, y, z, xy, yz, zx, xyz

  
}
*/


double rnd(double width)
{
  /****************************************************
       This rnd() function generates random number
                -width/2 to width/2
  ****************************************************/

  double result;

  result = rand();
  result = result*width/(double)RAND_MAX - 0.5*width;
  return result;
}
