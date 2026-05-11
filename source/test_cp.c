#include <stdio.h>
#include <stdlib.h>  
#include <math.h>
#include <string.h>
#include <time.h>

#define PI              3.1415926535897932384626

double phi1(double x, double L);
double phi2(double x, double L);


int main(int argc, char *argv[]) 
{
  int i,j,l,p,Npbc,Ng,Ng0,method;
  double L,a,G,Gq,Gp,G0,xmin,xmax,x1,x2,dx,sum;
  double sumr,sumi;
  double rphi1_G[2000],iphi1_G[2000];
  double rphi2_G[2000],iphi2_G[2000];
  double rA[2000],iA[2000],rB[2000],iB[2000];

  method = 4;
  
  L = 2.0;
  Npbc = 10;
  a = 0.08;
  Ng = 500;
  xmin = 0.0;
  xmax = L;
  dx = (xmax-xmin)/(double)Ng;
  G0 = 2.0*PI/L;

  /**************************************/
  /*     direct method in real space   */
  /*************************************/

  if (method==1){
  
    for (i=0; i<Ng; i++){  
      x1 = xmin + (double)i*dx;

      sum = 0.0;
      for (j=0; j<180*Ng; j++){
	x2 = xmin - 90.0*L + (double)j*dx;
	sum += phi1(x2,L)*1.0/((x1-x2)*(x1-x2)+a*a)*dx;
      }

      sum = sum*phi2(x1,L);
    
      printf("ABC1 %15.12f %15.12f\n",x1,sum);
    }
  }
  
  /*************************************/
  /*  conventional Fourier transform   */
  /*************************************/

  if (method==2){
    
    for (i=0; i<Ng; i++){

      G = (double)i*G0;

      sumr = 0.0;
      sumi = 0.0;

      for (j=0; j<Ng; j++){  
	x2 = xmin + (double)j*dx;
	sumr += cos(G*x2)*phi1(x2,L);
	sumi -= sin(G*x2)*phi1(x2,L);
      }

      rphi1_G[i] = sumr;  
      iphi1_G[i] = sumi;

      //printf("ABC2 %15.12f %15.12f %15.12f\n",G,sumr,sumi);
    
    }

    for (i=0; i<Ng; i++){  
      x1 = xmin + (double)i*dx;

      sumr = 0.0;
      sumi = 0.0;
    
      for (j=0; j<Ng; j++){
	G = (double)j*G0;
	sumr += cos(G*x1)*exp(-a*G)*rphi1_G[j] - sin(G*x1)*exp(-a*G)*iphi1_G[j];
      }
    
      sumr = sumr*PI/a*phi2(x1,L)/(0.5*(double)Ng);  

      printf("ABC2 %15.12f %15.12f\n",x1,sumr);
    
    }
  }

  /*************************************/
  /*          the new method           */
  /*************************************/

  if (method==3){

    /* FT of phi2 */
  
    for (i=0; i<Ng; i++){

      G = (double)i*G0;
     
      sumr = 0.0;
      sumi = 0.0;

      for (j=0; j<Ng; j++){  
	x1 = xmin + (double)j*dx;
	sumr += cos(G*x1)*phi2(x1,L);
	sumi += sin(G*x1)*phi2(x1,L);

	//printf("%10.7f %10.7f %10.7f\n",sumr,cos(G*x1),cos(2.0*x1/L*2.0*PI));
      }

      rphi2_G[i] = sumr;  
      iphi2_G[i] = sumi;

      //printf("ABC3 %15.12f %15.12f %15.12f\n",G,rphi2_G[i],iphi2_G[i]);
    }

  
    /* FT for A(x2) */
  
    for (i=0; i<Ng; i++){
      
      x2 = xmin + (double)i*dx;

      sumr = 0.0;
      sumi = 0.0;

      for (j=0; j<Ng; j++){
	G = (double)j*G0;
	sumr += (cos(G*x2)*rphi2_G[j] + sin(G*x2)*iphi2_G[j])*exp(-a*G);
	sumi += (cos(G*x2)*iphi2_G[j] - sin(G*x2)*rphi2_G[j])*exp(-a*G);
      }

      rA[i] = sumr;
      iA[i] = sumi; 

      //printf("ABC3 %15.12f %15.12f %15.12f\n",x2,rA[i],iA[i]);
    }  

    /* FT */

    for (i=0; i<Ng; i++){

      G = (double)i*G0;
     
      sumr = 0.0;
      sumi = 0.0;

      for (j=0; j<Ng; j++){  
	x2 = xmin + (double)j*dx;
	sumr += (cos(G*x2)*rA[j] + sin(G*x2)*iA[j])*phi1(x2,L);
	sumi += (cos(G*x2)*iA[j] - sin(G*x2)*rA[j])*phi1(x2,L);
      }

      rB[i] = sumr*exp(-a*G);
      iB[i] = sumi*exp(-a*G);

      //printf("ABC3 %15.12f %15.12f %15.12f\n",G,rB[i],iB[i]);
    
    }

    /* FT of B */

    for (i=0; i<Ng; i++){  
      x1 = xmin + (double)i*dx;

      sumr = 0.0;
      sumi = 0.0;

      for (j=0; j<Ng; j++){
	G = (double)j*G0;
	sumr += cos(G*x1)*rB[j] - sin(G*x1)*iB[j];
	sumi += cos(G*x1)*iB[j] + sin(G*x1)*rB[j];
      } 
    
      printf("ABC3 %15.12f %15.12f %15.12f\n",x1,sumr,sumi);
    
    }
  }

  /*************************************/
  /*         Eq. in the note           */
  /*************************************/

  if (method==4){

    /* FT of phi2 */
  
    for (i=0; i<Ng; i++){

      G = (double)i*G0;
     
      sumr = 0.0;
      sumi = 0.0;

      for (j=0; j<Ng; j++){  
	x1 = xmin + (double)j*dx;
	sumr += cos(G*x1)*phi2(x1,L);
	sumi -= sin(G*x1)*phi2(x1,L);

	//printf("%10.7f %10.7f %10.7f\n",sumr,cos(G*x1),cos(2.0*x1/L*2.0*PI));
      }

      rphi2_G[i] = sumr;  
      iphi2_G[i] = sumi;

      //printf("ABC3 %15.12f %15.12f %15.12f\n",G,rphi2_G[i],iphi2_G[i]);
    }
  
    /* FT for A(x2) */

    for (p=0; p<Ng; p++){
    
      for (i=0; i<Ng; i++){
      
	x2 = xmin + (double)i*dx;

	sumr = 0.0;
	sumi = 0.0;

	for (j=0; j<Ng; j++){

	  l = p-j;
	  if      (l<0)   l = l + Ng;
	  else if (Ng<=l) l = l - Ng;
	
	  Gq = (double)j*G0;
	  sumr += (cos(Gq*x2)*rphi2_G[l] + sin(Gq*x2)*iphi2_G[l])*exp(-a*Gq);
	  sumi += (cos(Gq*x2)*iphi2_G[l] - sin(Gq*x2)*rphi2_G[l])*exp(-a*Gq);

	} // j

	rA[i] = sumr;
	iA[i] = sumi; 

	//printf("ABC3 %15.12f %15.12f %15.12f\n",x2,rA[i],iA[i]);

      } // i  

      sumr = 0.0;
      sumi = 0.0;
      
      for (i=0; i<Ng; i++){
	x2 = xmin + (double)i*dx;
        sumr += phi1(x2,L)*rA[i];
        sumi += phi1(x2,L)*iA[i];
      }

      rB[p] = sumr*PI/(double)Ng;
      iB[p] = sumi*PI/(double)Ng;
      
    } // p

    /* FT of B */

    for (i=0; i<Ng; i++){
      
      x1 = xmin + (double)i*dx;

      sumr = 0.0;
      sumi = 0.0;

      for (j=0; j<Ng; j++){
	G = (double)j*G0;
	sumr += cos(G*x1)*rB[j] - sin(G*x1)*iB[j];
	sumi += cos(G*x1)*iB[j] + sin(G*x1)*rB[j];
      } 
    
      sumr = sumr/(double)Ng;
      sumi = sumi/(double)Ng;

      printf("ABC3 %15.12f %15.12f %15.12f\n",x1,sumr,sumi);
    
    }
  }

  /*************************************/
  /*  the last Eq. in the note         */
  /*************************************/

  if (method==5){

    /* FT of phi2 */

    for (i=0; i<Ng; i++){

      G = (double)i*G0;
     
      sumr = 0.0;
      sumi = 0.0;

      for (j=0; j<Ng; j++){  
	x1 = xmin + (double)j*dx;
	sumr += cos(G*x1)*phi2(x1,L);
	sumi -= sin(G*x1)*phi2(x1,L);
      }

      if (i<(Ng/5)){
        rphi2_G[i] = sumr;  
        iphi2_G[i] = sumi;
      }
      else{
        rphi2_G[i] = 0.0;
        iphi2_G[i] = 0.0;
      }

      //printf("ABC3 %15.12f %15.12f %15.12f\n",G,rphi2_G[i],iphi2_G[i]);
      
    }

    /* FT for A(x2) */

    if (1){
    for (i=0; i<Ng; i++){
      
      x2 = xmin + (double)i*dx;

      sumr = 0.0;
      sumi = 0.0;

      for (j=0; j<Ng; j++){

	G = (double)j*G0;
	
	sumr += (cos(G*x2)*rphi2_G[j] - sin(G*x2)*iphi2_G[j])*exp(a*G);
	sumi += (cos(G*x2)*iphi2_G[j] + sin(G*x2)*rphi2_G[j])*exp(a*G);

      } // j

      rA[i] = sumr;
      iA[i] = sumi;

      //printf("ABC3 %15.12f %15.12f %15.12f\n",x2,rA[i],iA[i]);

    } // i  
    }

    if (0){
    for (i=0; i<Ng; i++){
      
      x2 = xmin + (double)i*dx;

      sumr = 0.0;
      sumi = 0.0;

      for (j=0; j<Ng; j++){

	G = (double)j*G0;

	sumr += (cos(G*x2)*rphi2_G[j] + sin(G*x2)*(-iphi2_G[j]))*exp(-a*G);
	sumi += (cos(G*x2)*(-iphi2_G[j]) - sin(G*x2)*rphi2_G[j])*exp(-a*G);
	
      } // j

      rA[i] = sumr;
      iA[i] = sumi;

      //printf("ABC3 %15.12f %15.12f %15.12f\n",x2,rA[i],iA[i]);

    } // i  
    }
    
    //exit(0);
    
    for (i=0; i<Ng; i++){

      G = (double)i*G0;

      sumr = 0.0;
      sumi = 0.0;

      for (j=0; j<Ng; j++){
        x2 = xmin + (double)j*dx;
	sumr += (cos(G*x2)*rA[j] + sin(G*x2)*iA[j])*phi1(x2,L);
	sumi += (cos(G*x2)*iA[j] - sin(G*x2)*rA[j])*phi1(x2,L);
      }
	
      rB[i] = sumr*exp(-a*G); 
      iB[i] = sumi*exp(-a*G); 
    }

    /* FT of B */

    for (i=0; i<Ng; i++){
      
      x1 = xmin + (double)i*dx;

      sumr = 0.0;
      sumi = 0.0;

      for (j=0; j<Ng; j++){
	G = (double)j*G0;
	sumr += cos(G*x1)*rB[j] - sin(G*x1)*iB[j];
	sumi += cos(G*x1)*iB[j] + sin(G*x1)*rB[j];
      } 
    
      sumr = sumr/(double)Ng;
      sumi = sumi/(double)Ng;

      printf("ABC3 %15.12f %15.12f %15.12f\n",x1,sumr,sumi);
    
    }
  }
  
  exit(0);

}

double phi1(double x, double L)
{
  return sin(1.0*x/L*2.0*PI) + cos(4.0*x/L*2.0*PI) - cos(8.0*x/L*2.0*PI);
}


double phi2(double x, double L)
{
  return sin(5.0*x/L*2.0*PI) - cos(2.0*x/L*2.0*PI) + cos(2.0*x/L*2.0*PI)*sin(4.0*x/L*2.0*PI);
}
