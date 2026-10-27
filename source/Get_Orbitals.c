/**********************************************************************
  Get_Orbitals.c:

     Get_Orbitals.c is a subroutine to calculate basis orbitals

  Log of Get_Orbitals.c:

     24/April/2002  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"
#include "openmx_orbitals.h"

void Get_Orbitals(const OpenMXAtomicViews *atomic, int wan, double x, double y, double z, double *Chi)
{

  int i,j,L0,Mul0,M0,i1;
  double S_coordinate[3];
  double dum,dum1,dum2,dum3,dum4,a,b,c,d;
  double siQ,coQ,siP,coP,Q,P,R,Rmin;
  double rm,df,sum0,sum1;
  double *RF;
  double *AF;
  double SH[Supported_MaxL*2+1][2];
  double dSHt[Supported_MaxL*2+1][2];
  double dSHp[Supported_MaxL*2+1][2];

  /* Radial */
  int mp_min,mp_max,m,po;
  double h1,h2,h3,f1,f2,f3,f4;
  double g1,g2,x1,x2,y1,y2,y12,y22,f;

  /* Flat worker-local orbital scratch, owned by the C++ workspace. */

  OpenMXOrbitalBuffers orbital_buffers;
  OpenMXOrbitalWorkspace *orbital_workspace=OpenMX_OrbitalWorkspace(
      (int)atomic->orbitals.extent[1]-1,(int)atomic->orbitals.extent[2],0,&orbital_buffers);
  RF=orbital_buffers.RF;
  AF=orbital_buffers.AF;

  /* start calc. */

  Rmin = 10e-14;

  xyz2spherical(x,y,z,0.0,0.0,0.0,S_coordinate);
  R = S_coordinate[0];
  Q = S_coordinate[1];
  P = S_coordinate[2];

  po = OpenMX_OrbitalRadial(atomic,wan,R,0,&orbital_buffers);

  if (po==0){

    /* Angular */
    siQ = sin(Q);
    coQ = cos(Q);
    siP = sin(P);
    coP = cos(P);
    if (atomic->species.basis_lmax[wan] >= 4)
      openmx_orbital_polar(x, y, z, R, &siQ, &coQ, &siP, &coP, &P);

    for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){

      if (L0==0){
        AF[(size_t)(0)*orbital_buffers.magnetic_capacity+(0)] = 0.282094791773878;
      }
      else if (L0==1){
        dum = 0.48860251190292*siQ;
        AF[(size_t)(1)*orbital_buffers.magnetic_capacity+(0)] = dum*coP;
        AF[(size_t)(1)*orbital_buffers.magnetic_capacity+(1)] = dum*siP;
        AF[(size_t)(1)*orbital_buffers.magnetic_capacity+(2)] = 0.48860251190292*coQ;
      }
      else if (L0==2){
        dum1 = siQ*siQ;
        dum2 = 1.09254843059208*siQ*coQ;
        AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(0)] = 0.94617469575756*coQ*coQ - 0.31539156525252;
        AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(1)] = 0.54627421529604*dum1*(1.0 - 2.0*siP*siP);
        AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(2)] = 1.09254843059208*dum1*siP*coP;
        AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(3)] = dum2*coP;
        AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(4)] = dum2*siP;
      }

      else if (L0==3){
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(0)] = 0.373176332590116*(5.0*coQ*coQ*coQ - 3.0*coQ);
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(1)] = 0.457045799464466*coP*siQ*(5.0*coQ*coQ - 1.0);
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(2)] = 0.457045799464466*siP*siQ*(5.0*coQ*coQ - 1.0);
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(3)] = 1.44530572132028*siQ*siQ*coQ*(coP*coP-siP*siP);
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(4)] = 2.89061144264055*siQ*siQ*coQ*siP*coP;
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(5)] = 0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP - 3.0*coP);
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(6)] = 0.590043589926644*siQ*siQ*siQ*(3.0*siP - 4.0*siP*siP*siP);
      }

      else if (4<=L0){

        /* calculation of complex spherical harmonics functions */
        for(m=-L0; m<=L0; m++){ 
          openmx_orbital_complex_sh(L0, m, siQ, coQ, P,SH[L0+m],dSHt[L0+m],dSHp[L0+m]);
	}

        /* transformation of complex to real */
        for (i=0; i<(L0*2+1); i++){

          sum0 = 0.0;
          sum1 = 0.0; 
	  for (j=0; j<(L0*2+1); j++){
            sum0 += openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).r*SH[j][0] - openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).i*SH[j][1];
            sum1 += openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).r*SH[j][1] + openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).i*SH[j][0];
	  }
          AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = sum0 + sum1;
	}              

      }
    }
  }

  /* Chi */  
  i1 = -1;
  for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){
    for (Mul0=0; Mul0<atomic->species.basis[(size_t)(wan)*atomic->species.angular_channels+(L0)]; Mul0++){
      for (M0=0; M0<=2*L0; M0++){
        i1++;
        Chi[i1] = (po==0) ? RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)]*AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)] : 0.0;
      }
    }
  }


  openmx_core_orbital_destroy(orbital_workspace);

}
