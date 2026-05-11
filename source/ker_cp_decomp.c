#include <stdio.h>
#include <stdlib.h>  
#include <math.h>
#include <string.h>
#include <time.h>


int main(int argc, char *argv[]) 
{
  int I,J,i,j,k,i1,j1,k1,i2,j2,k2,m,n;
  int Nd1,Ng1,Nrank,loop,po;
  double xmin,xmax,dx,dx1,x,y,z,x1,x2,y1,y2,z1,z2;
  double tensor,d,F,G,xd,yd,zd,ai,aj,ak,sumA,sumB,lambdaF,lambdaG;
  double ScaleFactor,max_diff,xd1,yd1,zd1;
  double fx1,fy1,fz1,fx2,fy2,fz2;
  double al = 0.0002, be=0.000000001;
  double ****Vec1,****Vec2;
  double ****dFdA,****dFdB;

  Nd1 = 2;
  Nrank = Nd1*Nd1*Nd1;
  Ng1 = 12;
  lambdaF = 1.0;
  lambdaG = 0.0;
  
  xmin = 0.0;
  xmax = 12.0;

  dx = (xmax - xmin)/(double)(Nd1-1);
  dx1 = (xmax - xmin)/(double)(Ng1-1);

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
  
    I = 0;
    for (i=0; i<Nd1; i++){
      ai = xmin + (double)i*dx;
      for (j=0; j<Nd1; j++){
	aj = xmin + (double)j*dx;
	for (k=0; k<Nd1; k++){
	  ak = xmin + (double)k*dx;

	  /* set Vec1 and Vec2 on grid */
	
	  for (i1=0; i1<Ng1; i1++){

	    x = xmin + (double)i1*dx1;
	    x2 = -(x-ai)*(x-ai); 
	  
	    for (j1=0; j1<Ng1; j1++){
	    
	      y = xmin + (double)j1*dx1;
	      y2 = -(y-aj)*(y-aj); 

	      for (k1=0; k1<Ng1; k1++){
	      
		z = xmin + (double)k1*dx1;
		z2 = -(z-ak)*(z-ak);

		//Vec1[I][i1][j1][k1] = exp( 0.006*(x2+y2+z2) );

		Vec1[I][i1][j1][k1] = exp( 0.06*(x2+y2+z2) );
		Vec2[I][i1][j1][k1] = Vec1[I][i1][j1][k1];

		//printf("ABC I=%2d %15.12f %15.12f %15.12f %15.12f\n",I,x2,y2,z2,Vec1[I][i1][j1][k1]);
	      
	      
	      }
	    }
	  }

	  I++;
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

	    x = xmin + (double)i1*dx1;

	    /* x^2 + 2 xy + y^2 */
	    /* x^4 + 4 x^3 y + 6 x^2 y^2 + 4 x y^3 + y^4 */

	    if      (i==0) { fx1 = exp(-0.1*x);          fx2 = exp(-0.1*x);            }
	    else if (i==1) { fx1 = exp(-1.0*x);          fx2 = exp(-1.0*x);            }
	    else if (i==2) { fx1 = exp(-3.0*x);          fx2 = exp(-3.0*x);            }
	    else if (i==3) { fx1 = exp(-4.0*x);          fx2 = exp(-4.0*x);            }

	    
	    /*
	    if      (i==0) { fx1 = pow(1.0/0.1,1.0/6.0); fx2 = pow(1.0/0.1,1.0/6.0);   }
	    else if (i==1) { fx1 = al*x*x;               fx2 = 1.0;                    }
	    else if (i==2) { fx1 = sqrt(al*2.0)*x;       fx2 = sqrt(al*2.0)*x;         }
	    else if (i==3) { fx1 = 1.0;                  fx2 = 0.0002*x*x;             }
	    else if (i==4) { fx1 = be*x*x*x*x;           fx2 = 1.0;                    }
	    else if (i==5) { fx1 = sqrt(4.0*be)*x*x*x;   fx2 = sqrt(be)*x;             }
	    else if (i==6) { fx1 = sqrt(6.0*be)*x*x;     fx2 = sqrt(6.0*be)*x*x;       }
	    else if (i==7) { fx1 = sqrt(4.0*be)*x;       fx2 = sqrt(4.0*be)*x*x*x;     }
	    else if (i==8) { fx1 = 1.0;                  fx2 = be*x*x*x*x;             }
	    */
	    
	    for (j1=0; j1<Ng1; j1++){
	    
	      y = xmin + (double)j1*dx1;

	      if      (j==0) { fy1 = exp(-0.1*y);  fy2 = exp(-0.1*y);   }
	      else if (j==1) { fy1 = exp(-1.0*y);  fy2 = exp(-1.0*y);   }
	      else if (j==2) { fy1 = exp(-3.0*y);  fy2 = exp(-3.0*y);   }
	      else if (j==3) { fy1 = exp(-4.0*y);  fy2 = exp(-4.0*y);   }
	      
	      /*
	      if      (j==0) { fy1 = pow(1.0/0.1,1.0/6.0); fy2 = pow(1.0/0.1,1.0/6.0);   }
	      else if (j==1) { fy1 = al*y*y;               fy2 = 1.0;                    }
	      else if (j==2) { fy1 = sqrt(al*2.0)*y;       fy2 = sqrt(al*2.0)*y;         }
	      else if (j==3) { fy1 = 1.0;                  fy2 = al*y*y;                 }
	      else if (j==4) { fy1 = be*y*y*y*y;           fy2 = 1.0;                    }
	      else if (j==5) { fy1 = sqrt(4.0*be)*y*y*y;   fy2 = sqrt(be)*y;             }
	      else if (j==6) { fy1 = sqrt(6.0*be)*y*y;     fy2 = sqrt(6.0*be)*y*y;       }
	      else if (j==7) { fy1 = sqrt(4.0*be)*y;       fy2 = sqrt(4.0*be)*y*y*y;     }
	      else if (j==8) { fy1 = 1.0;                  fy2 = be*y*y*y*y;             }
	      */
	      
	      for (k1=0; k1<Ng1; k1++){
	      
		z = xmin + (double)k1*dx1;

		if      (k==0) { fz1 = exp(-0.0*z); fz2 = exp(-0.1*z);  }
		else if (k==1) { fz1 = exp(-1.0*z); fz2 = exp(-1.0*z);  }
		else if (k==2) { fz1 = exp(-3.0*z); fz2 = exp(-3.0*z);  }
		else if (k==3) { fz1 = exp(-4.0*z); fz2 = exp(-4.0*z);  }
		
		/*
		if      (k==0) { fz1 = pow(1.0/0.1,1.0/6.0); fz2 = pow(1.0/0.1,1.0/6.0);  }
		else if (k==1) { fz1 = al*z*z;               fz2 = 1.0;                   }
		else if (k==2) { fz1 = sqrt(al*2.0)*z;       fz2 = sqrt(al*2.0)*z;        }
		else if (k==3) { fz1 = 1.0;                  fz2 = al*z*z;                }
		else if (k==4) { fz1 = be*z*z*z*z;           fz2 = 1.0;                    }
		else if (k==5) { fz1 = sqrt(4.0*be)*z*z*z;   fz2 = sqrt(be)*z;             }
		else if (k==6) { fz1 = sqrt(6.0*be)*z*z;     fz2 = sqrt(6.0*be)*z*z;       }
		else if (k==7) { fz1 = sqrt(4.0*be)*z;       fz2 = sqrt(4.0*be)*z*z*z;     }
		else if (k==8) { fz1 = 1.0;                  fz2 = be*z*z*z*z;             }
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
      x1 = xmin + (double)i1*dx1;
      for (j1=0; j1<Ng1; j1++){
	y1 = xmin + (double)j1*dx1;
	for (k1=0; k1<Ng1; k1++){
	  z1 = xmin + (double)k1*dx1;

	  for (i2=0; i2<Ng1; i2++){
	    x2 = xmin + (double)i2*dx1;
	    xd = x1 + x2;
	    for (j2=0; j2<Ng1; j2++){
	      y2 = xmin + (double)j2*dx1;
	      yd = y1 + y2;	    
	      for (k2=0; k2<Ng1; k2++){
		z2 = xmin + (double)k2*dx1;
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
		if (loop%100==1){
                printf("BBB loop=%2d i1=%2d j1=%2d k1=%2d i2=%2d j2=%2d k2=%2d %15.12f %15.12f\n",
		       loop,i1,j1,k1,i2,j2,k2,tensor,1.0/(xd*xd + yd*yd + zd*zd + 0.1));
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
	x1 = xmin + (double)i1*dx1;
	for (j1=0; j1<Ng1; j1++){
	  y1 = xmin + (double)j1*dx1;
	  for (k1=0; k1<Ng1; k1++){
	    z1 = xmin + (double)k1*dx1;

	    sumA = 0.0;
	    sumB = 0.0;
	  
	    for (i2=0; i2<Ng1; i2++){
	      x2 = xmin + (double)i2*dx1;
	      xd = x1 + x2;
	      for (j2=0; j2<Ng1; j2++){
		y2 = xmin + (double)j2*dx1;
		yd = y1 + y2;	    
		for (k2=0; k2<Ng1; k2++){
		  z2 = xmin + (double)k2*dx1;
		  zd = z1 + z2;	    

		  tensor = 0.0;
		  for (J=0; J<Nrank; J++){
		    tensor += Vec1[J][i1][j1][k1]*Vec2[J][i2][j2][k2];
		  }

		  d = tensor - 1.0/(xd*xd + yd*yd + zd*zd + 0.1);
		
		  sumA += 2.0*d*Vec2[I][i2][j2][k2];
		  sumB += 2.0*d*Vec1[I][i2][j2][k2];
		
		  n++; 
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
    else if (F<12000.0) ScaleFactor = 0.0350;
    else if (F<7000.0)  ScaleFactor = 0.050;
    else if (F<1000.0)  ScaleFactor = 0.10;
    else if (F<500.0)   ScaleFactor = 0.20;
    else if (F<100.0)   ScaleFactor = 0.40;
    else if (F<10.0)    ScaleFactor = 0.90;
    
    loop++;

    if (F<1.0e-7) po = 1;
    
  } while (po==0 && loop<100000);  

  I = 4;
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
