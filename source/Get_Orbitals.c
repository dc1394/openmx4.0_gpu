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

  po = 0;
  mp_min = 0;
  mp_max = atomic->species.pao_mesh[wan] - 1;

  if (openmx_radial_row2(&atomic->pao_radii,wan)[atomic->species.pao_mesh[wan]-1]<R){

    for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){
      for (Mul0=0; Mul0<atomic->species.basis[(size_t)(wan)*atomic->species.angular_channels+(L0)]; Mul0++){
        RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = 0.0;
      }
    }

    po = 1;
  }

  else if (R<openmx_radial_row2(&atomic->pao_radii,wan)[0]){

    m = 4;
    rm = openmx_radial_row2(&atomic->pao_radii,wan)[m];

    /* Use reflected endpoint samples before forming the spline weights. */
    h1 = (m==1) ? openmx_radial_row2(&atomic->pao_radii,wan)[m-1] - openmx_radial_row2(&atomic->pao_radii,wan)[m+1]
      : openmx_radial_row2(&atomic->pao_radii,wan)[m-1] - openmx_radial_row2(&atomic->pao_radii,wan)[m-2];
    h2 = openmx_radial_row2(&atomic->pao_radii,wan)[m]   - openmx_radial_row2(&atomic->pao_radii,wan)[m-1];
    h3 = (m==atomic->species.pao_mesh[wan]-1) ? openmx_radial_row2(&atomic->pao_radii,wan)[m-2] - openmx_radial_row2(&atomic->pao_radii,wan)[m]
      : openmx_radial_row2(&atomic->pao_radii,wan)[m+1] - openmx_radial_row2(&atomic->pao_radii,wan)[m];

    x1 = rm - openmx_radial_row2(&atomic->pao_radii,wan)[m-1];
    x2 = rm - openmx_radial_row2(&atomic->pao_radii,wan)[m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;

    dum = h1 + h2;
    dum1 = h1/h2/dum;
    dum2 = h2/h1/dum;
    dum = h2 + h3;
    dum3 = h2/h3/dum;
    dum4 = h3/h2/dum;

    for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){
      for (Mul0=0; Mul0<atomic->species.basis[(size_t)(wan)*atomic->species.angular_channels+(L0)]; Mul0++){

        f1 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[(m==1) ? m+1 : m-2];
        f2 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[m-1];
        f3 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[m];
        f4 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[(m==atomic->species.pao_mesh[wan]-1) ? m-2 : m+1];

        dum = f3 - f2;
        g1 = dum*dum1 + (f2-f1)*dum2;
        g2 = (f4-f3)*dum3 + dum*dum4;

        f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
           + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

        df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
           + y22*(2.0*f2 + h2*g1)/h2
           + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
           - y12*(2.0*f3 - h2*g2)/h2;

        if (L0==0){
          a = 0.0;
          b = 0.5*df/rm;
          c = 0.0;
          d = f - b*rm*rm;
        }

        else if (L0==1){
          a = (rm*df - f)/(2.0*rm*rm*rm);
          b = 0.0;
          c = df - 3.0*a*rm*rm;
          d = 0.0;
        }

        else{
          b = (3.0*f - rm*df)/(rm*rm);
          a = (f - b*rm*rm)/(rm*rm*rm);
          c = 0.0;
          d = 0.0;
        }

        RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = a*R*R*R + b*R*R + c*R + d;

      }
    }

  }

  else{

    do{
      m = (mp_min + mp_max)/2;
      if (openmx_radial_row2(&atomic->pao_radii,wan)[m]<R)
        mp_min = m;
      else 
        mp_max = m;
    }
    while((mp_max-mp_min)!=1);
    m = mp_max;

    h1 = (m==1) ? openmx_radial_row2(&atomic->pao_radii,wan)[m-1] - openmx_radial_row2(&atomic->pao_radii,wan)[m+1]
      : openmx_radial_row2(&atomic->pao_radii,wan)[m-1] - openmx_radial_row2(&atomic->pao_radii,wan)[m-2];
    h2 = openmx_radial_row2(&atomic->pao_radii,wan)[m]   - openmx_radial_row2(&atomic->pao_radii,wan)[m-1];
    h3 = (m==atomic->species.pao_mesh[wan]-1) ? openmx_radial_row2(&atomic->pao_radii,wan)[m-2] - openmx_radial_row2(&atomic->pao_radii,wan)[m]
      : openmx_radial_row2(&atomic->pao_radii,wan)[m+1] - openmx_radial_row2(&atomic->pao_radii,wan)[m];

    x1 = R - openmx_radial_row2(&atomic->pao_radii,wan)[m-1];
    x2 = R - openmx_radial_row2(&atomic->pao_radii,wan)[m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;

    dum = h1 + h2;
    dum1 = h1/h2/dum;
    dum2 = h2/h1/dum;
    dum = h2 + h3;
    dum3 = h2/h3/dum;
    dum4 = h3/h2/dum;

    for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){
      for (Mul0=0; Mul0<atomic->species.basis[(size_t)(wan)*atomic->species.angular_channels+(L0)]; Mul0++){

        f1 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[(m==1) ? m+1 : m-2];
        f2 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[m-1];
        f3 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[m];
        f4 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[(m==atomic->species.pao_mesh[wan]-1) ? m-2 : m+1];

        dum = f3 - f2;
        g1 = dum*dum1 + (f2-f1)*dum2;
        g2 = (f4-f3)*dum3 + dum*dum4;

        f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
           + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

        RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = f;

      }
    } 

  }

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
            sum0 += Comp2Real[L0][i][j].r*SH[j][0] - Comp2Real[L0][i][j].i*SH[j][1]; 
            sum1 += Comp2Real[L0][i][j].r*SH[j][1] + Comp2Real[L0][i][j].i*SH[j][0]; 
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
