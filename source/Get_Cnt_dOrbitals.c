/**********************************************************************
  Get_Cnt_dOrbitals.c:

     Get_Cnt_dOrbitals.c is a subrutine to calculate
     derivatives of contracted basis orbitals

  Log of Get_Orbitals.c:

     24/April/2002  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"
#include "openmx_orbitals.h"

void Get_Cnt_dOrbitals(const OpenMXAtomicViews *atomic, int Mc_AN, double x, double y, double z, double **dChi)
{

  int Gc_AN,i,j,L0,Mul0,M0,i1,al,wan,p;
  int mp_min,mp_max,m,po;
  double S_coordinate[3];
  double dum,dum1,dum2,dum3,dum4;
  double siQ,coQ,siP,coP,Q,P,R,Rmin;
  double *RF;
  double *dRF;
  double *AF;
  double *dAFQ;
  double *dAFP;
  double h1,h2,h3,f1,f2,f3,f4,a,b;
  double g1,g2,x1,x2,y1,y2,y12,y22,f,df;
  double dRx,dRy,dRz,dQx,dQy,dQz,dPx,dPy,dPz;
  double dChiR,dChiQ,dChiP,tmp0,tmp1;
  double SH[Supported_MaxL*2+1][2];
  double dSHt[Supported_MaxL*2+1][2];
  double dSHp[Supported_MaxL*2+1][2];

  /* Flat worker-local orbital scratch, owned by the C++ workspace. */
  
  OpenMXOrbitalBuffers orbital_buffers;
  OpenMXOrbitalWorkspace *orbital_workspace=OpenMX_OrbitalWorkspace(
      (int)atomic->orbitals.extent[1]-1,(int)atomic->orbitals.extent[2],1,&orbital_buffers);
  RF=orbital_buffers.RF;
  dRF=orbital_buffers.dRF;
  AF=orbital_buffers.AF;
  dAFQ=orbital_buffers.dAFQ;
  dAFP=orbital_buffers.dAFP;

  /* start calc. */

  /* Grid halo callers also pass extended local indices. */
  Gc_AN = F_M2G[Mc_AN];
  Rmin = 10e-12;
  wan = WhatSpecies[Gc_AN];
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
        dRF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = 0.0;
      }
    }
    po = 1;
  }
  else if (R<openmx_radial_row2(&atomic->pao_radii,wan)[0]){

    h1 = openmx_radial_row2(&atomic->pao_radii,wan)[0] - openmx_radial_row2(&atomic->pao_radii,wan)[1];

    for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){
      for (Mul0=0; Mul0<atomic->species.basis[(size_t)(wan)*atomic->species.angular_channels+(L0)]; Mul0++){

        y1 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[0] - openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[1];
        a = y1/h1;
        b = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[0] - a*openmx_radial_row2(&atomic->pao_radii,wan)[0];

         RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = a*R + b;
        dRF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = a;
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

    /* Use reflected endpoint samples before forming the spline weights. */
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

        df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
           + y2*y2*(2.0*f2 + h2*g1)/h2
           + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
           - y1*y1*(2.0*f3 - h2*g2)/h2;

         RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = f;
        dRF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = df;

      }
    } 
  }

  /* dr/dx,y,z, dQ/dx,y,z, dP/dx,y,z and dAngular */

  if (po==0){

    /* Angular */
    siQ = sin(Q);
    coQ = cos(Q);
    siP = sin(P);
    coP = cos(P);

    dRx = siQ*coP;
    dRy = siQ*siP;
    dRz = coQ;

    if (Rmin<R){
      dQx = coQ*coP/R;
      dQy = coQ*siP/R;
      dQz = -siQ/R;
    }
    else{
      dQx = 0.0;
      dQy = 0.0;
      dQz = 0.0;
    }

    /* RICS note 72P */

    if (Rmin<R){
      dPx = -siP/R;
      dPy = coP/R;
      dPz = 0.0;
    }
    else{
      dPx = 0.0;
      dPy = 0.0;
      dPz = 0.0;
    }

    for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){
      if (L0==0){
        AF[(size_t)(0)*orbital_buffers.magnetic_capacity+(0)] = 0.282094791773878;
        dAFQ[(size_t)(0)*orbital_buffers.magnetic_capacity+(0)] = 0.0;
        dAFP[(size_t)(0)*orbital_buffers.magnetic_capacity+(0)] = 0.0;
      }
      else if (L0==1){
        dum = 0.48860251190292*siQ;

        AF[(size_t)(1)*orbital_buffers.magnetic_capacity+(0)] = dum*coP;
        AF[(size_t)(1)*orbital_buffers.magnetic_capacity+(1)] = dum*siP;
        AF[(size_t)(1)*orbital_buffers.magnetic_capacity+(2)] = 0.48860251190292*coQ;

        dAFQ[(size_t)(1)*orbital_buffers.magnetic_capacity+(0)] = 0.48860251190292*coQ*coP;
        dAFQ[(size_t)(1)*orbital_buffers.magnetic_capacity+(1)] = 0.48860251190292*coQ*siP;
        dAFQ[(size_t)(1)*orbital_buffers.magnetic_capacity+(2)] = -0.48860251190292*siQ;

        dAFP[(size_t)(1)*orbital_buffers.magnetic_capacity+(0)] = -0.48860251190292*siP;
        dAFP[(size_t)(1)*orbital_buffers.magnetic_capacity+(1)] = 0.48860251190292*coP;
        dAFP[(size_t)(1)*orbital_buffers.magnetic_capacity+(2)] = 0.0;
      }
      else if (L0==2){
        dum1 = siQ*siQ;
        dum2 = 1.09254843059208*siQ*coQ;
        AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(0)] = 0.94617469575756*coQ*coQ - 0.31539156525252;
        AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(1)] = 0.54627421529604*dum1*(1.0 - 2.0*siP*siP);
        AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(2)] = 1.09254843059208*dum1*siP*coP;
        AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(3)] = dum2*coP;
        AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(4)] = dum2*siP;

        dAFQ[(size_t)(2)*orbital_buffers.magnetic_capacity+(0)] = -1.89234939151512*siQ*coQ;
        dAFQ[(size_t)(2)*orbital_buffers.magnetic_capacity+(1)] = 1.09254843059208*siQ*coQ*(1.0 - 2.0*siP*siP);
        dAFQ[(size_t)(2)*orbital_buffers.magnetic_capacity+(2)] = 2.18509686118416*siQ*coQ*siP*coP;
        dAFQ[(size_t)(2)*orbital_buffers.magnetic_capacity+(3)] = 1.09254843059208*(1.0 - 2.0*siQ*siQ)*coP;
        dAFQ[(size_t)(2)*orbital_buffers.magnetic_capacity+(4)] = 1.09254843059208*(1.0 - 2.0*siQ*siQ)*siP;

        /* RICS note 72P */

        dAFP[(size_t)(2)*orbital_buffers.magnetic_capacity+(0)] = 0.0;
        dAFP[(size_t)(2)*orbital_buffers.magnetic_capacity+(1)] = -2.18509686118416*siQ*siP*coP;
        dAFP[(size_t)(2)*orbital_buffers.magnetic_capacity+(2)] = 1.09254843059208*siQ*(1.0 - 2.0*siP*siP);
        dAFP[(size_t)(2)*orbital_buffers.magnetic_capacity+(3)] = -1.09254843059208*coQ*siP;
        dAFP[(size_t)(2)*orbital_buffers.magnetic_capacity+(4)] = 1.09254843059208*coQ*coP;
      }
      else if (L0==3){
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(0)] = 0.373176332590116*(5.0*coQ*coQ*coQ - 3.0*coQ);
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(1)] = 0.457045799464466*coP*siQ*(5.0*coQ*coQ - 1.0);
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(2)] = 0.457045799464466*siP*siQ*(5.0*coQ*coQ - 1.0);
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(3)] = 1.44530572132028*siQ*siQ*coQ*(coP*coP-siP*siP);
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(4)] = 2.89061144264055*siQ*siQ*coQ*siP*coP;
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(5)] = 0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP - 3.0*coP);
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(6)] = 0.590043589926644*siQ*siQ*siQ*(3.0*siP - 4.0*siP*siP*siP);

        dAFQ[(size_t)(3)*orbital_buffers.magnetic_capacity+(0)] = 0.373176332590116*siQ*(-15.0*coQ*coQ + 3.0);
        dAFQ[(size_t)(3)*orbital_buffers.magnetic_capacity+(1)] = 0.457045799464466*coP*coQ*(15.0*coQ*coQ - 11.0);
        dAFQ[(size_t)(3)*orbital_buffers.magnetic_capacity+(2)] = 0.457045799464466*siP*coQ*(15.0*coQ*coQ - 11.0);
        dAFQ[(size_t)(3)*orbital_buffers.magnetic_capacity+(3)] = 1.44530572132028*(coP*coP-siP*siP)*siQ*(2.0*coQ*coQ-siQ*siQ);
        dAFQ[(size_t)(3)*orbital_buffers.magnetic_capacity+(4)] = 2.89061144264055*coP*siP*siQ*(2.0*coQ*coQ - siQ*siQ);
        dAFQ[(size_t)(3)*orbital_buffers.magnetic_capacity+(5)] = 1.770130769779932*coP*coQ*siQ*siQ*(-3.0 + 4.0*coP*coP);
        dAFQ[(size_t)(3)*orbital_buffers.magnetic_capacity+(6)] = 1.770130769779932*coQ*siP*siQ*siQ*( 3.0 - 4.0*siP*siP);

        /* RICS note 72P */

        dAFP[(size_t)(3)*orbital_buffers.magnetic_capacity+(0)] = 0.0;
        dAFP[(size_t)(3)*orbital_buffers.magnetic_capacity+(1)] = 0.457045799464466*siP*(-5.0*coQ*coQ + 1.0);
        dAFP[(size_t)(3)*orbital_buffers.magnetic_capacity+(2)] = 0.457045799464466*coP*( 5.0*coQ*coQ - 1.0);
        dAFP[(size_t)(3)*orbital_buffers.magnetic_capacity+(3)] = -5.781222885281120*coP*coQ*siP*siQ;
        dAFP[(size_t)(3)*orbital_buffers.magnetic_capacity+(4)] = 2.89061144264055*coQ*siQ*(coP*coP - siP*siP);
        dAFP[(size_t)(3)*orbital_buffers.magnetic_capacity+(5)] = 1.770130769779932*siP*siQ*siQ*(1.0 - 4.0*coP*coP);
        dAFP[(size_t)(3)*orbital_buffers.magnetic_capacity+(6)] = 1.770130769779932*coP*siQ*siQ*(1.0 - 4.0*siP*siP);
      }

      else if (4<=L0){

        for(m=-L0; m<=L0; m++){ 
          ComplexSH(L0,m,Q,P,SH[L0+m],dSHt[L0+m],dSHp[L0+m]);
	}

        AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(0)]   =   -SH[L0][0];
        dAFQ[(size_t)(L0)*orbital_buffers.magnetic_capacity+(0)] = -dSHt[L0][0];
        dAFP[(size_t)(L0)*orbital_buffers.magnetic_capacity+(0)] = -dSHp[L0][0];

        j = -1;
        for (i=1; i<(L0*2+1); i=i+4){
          j++;
          AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*SH[L0-(2*j+1)][0];
          dAFQ[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*dSHt[L0-(2*j+1)][0];
          dAFP[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*dSHp[L0-(2*j+1)][0];
	}

	j = 0;
	for (i=3; i<(L0*2+1); i=i+4){
	  j++;
          AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*SH[L0-2*j][0];
          dAFQ[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*dSHt[L0-2*j][0];
          dAFP[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*dSHp[L0-2*j][0];
	}

	j = -1;
	for (i=2; i<(L0*2+1); i=i+4){
	  j++;
          AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*SH[L0-(2*j+1)][1];
          dAFQ[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*dSHt[L0-(2*j+1)][1];
          dAFP[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*dSHp[L0-(2*j+1)][1];
	}

	j = 0;
	for (i=4; i<(L0*2+1); i=i+4){
	  j++;
          AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*SH[L0-2*j][1];
          dAFQ[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*dSHt[L0-2*j][1];
          dAFP[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*dSHp[L0-2*j][1];
	}
      }

    }
  }

  /* Contracted Chi */
  al = -1;
  for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){
    for (Mul0=0; Mul0<atomic->species.contracted_basis[(size_t)(wan)*atomic->species.angular_channels+(L0)]; Mul0++){
      for (M0=0; M0<=2*L0; M0++){
        al++;
        if (po!=0){
          dChi[0][al] = dChi[1][al] = 0.0;
          dChi[2][al] = dChi[3][al] = 0.0;
          continue;
        }

        tmp0 = 0.0;
        tmp1 = 0.0;
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
          tmp0 = tmp0 + CntCoes[Mc_AN][al][p]*RF[(size_t)(L0)*orbital_buffers.multiplicity+(p)];
          tmp1 = tmp1 + CntCoes[Mc_AN][al][p]*dRF[(size_t)(L0)*orbital_buffers.multiplicity+(p)];
        }        

        dChiR = tmp1*AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)];
        dChiQ = tmp0*dAFQ[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)];
        dChiP = tmp0*dAFP[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)];

        dChi[0][al] = tmp0*AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)];
        dChi[1][al] = -dRx*dChiR - dQx*dChiQ - dPx*dChiP;
        dChi[2][al] = -dRy*dChiR - dQy*dChiQ - dPy*dChiP;
        dChi[3][al] = -dRz*dChiR - dQz*dChiQ - dPz*dChiP;
      }
    }
  }


  openmx_core_orbital_destroy(orbital_workspace);

}


void Get_Cnt_dOrbitals2(const OpenMXAtomicViews *atomic, int Mc_AN, double x, double y, double z, double **dChi)
{

  int Gc_AN,i,j,L0,Mul0,M0,i1,al,wan,p;
  int mp_min,mp_max,m,po;
  double S_coordinate[3];
  double dum,dum1,dum2,dum3,dum4;
  double siQ,coQ,siP,coP,Q,P,R,Rmin;
  double *RF;
  double *dRF;
  double *AF;
  double *dAFQ;
  double *dAFP;
  double h1,h2,h3,f1,f2,f3,f4,a,b;
  double g1,g2,x1,x2,y1,y2,y12,y22,f,df;
  double dRx,dRy,dRz,dQx,dQy,dQz,dPx,dPy,dPz;
  double dChiR,dChiQ,dChiP,tmp0,tmp1;
  double SH[Supported_MaxL*2+1][2];
  double dSHt[Supported_MaxL*2+1][2];
  double dSHp[Supported_MaxL*2+1][2];

  /* Flat worker-local orbital scratch, owned by the C++ workspace. */
  
  OpenMXOrbitalBuffers orbital_buffers;
  OpenMXOrbitalWorkspace *orbital_workspace=OpenMX_OrbitalWorkspace(
      (int)atomic->orbitals.extent[1]-1,(int)atomic->orbitals.extent[2],1,&orbital_buffers);
  RF=orbital_buffers.RF;
  dRF=orbital_buffers.dRF;
  AF=orbital_buffers.AF;
  dAFQ=orbital_buffers.dAFQ;
  dAFP=orbital_buffers.dAFP;

  /* start calc. */

  Gc_AN = F_M2G[Mc_AN];
  Rmin = 10e-12;
  wan = WhatSpecies[Gc_AN];
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
        dRF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = 0.0;
      }
    }
    po = 1;
  }
  else if (R<openmx_radial_row2(&atomic->pao_radii,wan)[0]){

    h1 = openmx_radial_row2(&atomic->pao_radii,wan)[0] - openmx_radial_row2(&atomic->pao_radii,wan)[1];

    for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){
      for (Mul0=0; Mul0<atomic->species.basis[(size_t)(wan)*atomic->species.angular_channels+(L0)]; Mul0++){

        y1 = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[0] - openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[1];
        a = y1/h1;
        b = openmx_radial_row4(&atomic->orbitals,wan,L0,Mul0)[0] - a*openmx_radial_row2(&atomic->pao_radii,wan)[0];

         RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = a*R + b;
        dRF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = a;
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

        df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
           + y2*y2*(2.0*f2 + h2*g1)/h2
           + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
           - y1*y1*(2.0*f3 - h2*g2)/h2;

         RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = f;
        dRF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)] = df;

      }
    } 
  }

  /* dr/dx,y,z, dQ/dx,y,z, dP/dx,y,z and dAngular */

  if (po==0){

    /* Angular */
    siQ = sin(Q);
    coQ = cos(Q);
    siP = sin(P);
    coP = cos(P);

    dRx = siQ*coP;
    dRy = siQ*siP;
    dRz = coQ;

    if (Rmin<R){
      dQx = coQ*coP/R;
      dQy = coQ*siP/R;
      dQz = -siQ/R;
    }
    else{
      dQx = 0.0;
      dQy = 0.0;
      dQz = 0.0;
    }

    /* RICS note 72P */

    if (Rmin<R){
      dPx = -siP/R;
      dPy = coP/R;
      dPz = 0.0;
    }
    else{
      dPx = 0.0;
      dPy = 0.0;
      dPz = 0.0;
    }

    for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){
      if (L0==0){
        AF[(size_t)(0)*orbital_buffers.magnetic_capacity+(0)] = 0.282094791773878;
        dAFQ[(size_t)(0)*orbital_buffers.magnetic_capacity+(0)] = 0.0;
        dAFP[(size_t)(0)*orbital_buffers.magnetic_capacity+(0)] = 0.0;
      }
      else if (L0==1){
        dum = 0.48860251190292*siQ;

        AF[(size_t)(1)*orbital_buffers.magnetic_capacity+(0)] = dum*coP;
        AF[(size_t)(1)*orbital_buffers.magnetic_capacity+(1)] = dum*siP;
        AF[(size_t)(1)*orbital_buffers.magnetic_capacity+(2)] = 0.48860251190292*coQ;

        dAFQ[(size_t)(1)*orbital_buffers.magnetic_capacity+(0)] = 0.48860251190292*coQ*coP;
        dAFQ[(size_t)(1)*orbital_buffers.magnetic_capacity+(1)] = 0.48860251190292*coQ*siP;
        dAFQ[(size_t)(1)*orbital_buffers.magnetic_capacity+(2)] = -0.48860251190292*siQ;

        dAFP[(size_t)(1)*orbital_buffers.magnetic_capacity+(0)] = -0.48860251190292*siP;
        dAFP[(size_t)(1)*orbital_buffers.magnetic_capacity+(1)] = 0.48860251190292*coP;
        dAFP[(size_t)(1)*orbital_buffers.magnetic_capacity+(2)] = 0.0;
      }
      else if (L0==2){
        dum1 = siQ*siQ;
        dum2 = 1.09254843059208*siQ*coQ;
        AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(0)] = 0.94617469575756*coQ*coQ - 0.31539156525252;
        AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(1)] = 0.54627421529604*dum1*(1.0 - 2.0*siP*siP);
        AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(2)] = 1.09254843059208*dum1*siP*coP;
        AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(3)] = dum2*coP;
        AF[(size_t)(2)*orbital_buffers.magnetic_capacity+(4)] = dum2*siP;

        dAFQ[(size_t)(2)*orbital_buffers.magnetic_capacity+(0)] = -1.89234939151512*siQ*coQ;
        dAFQ[(size_t)(2)*orbital_buffers.magnetic_capacity+(1)] = 1.09254843059208*siQ*coQ*(1.0 - 2.0*siP*siP);
        dAFQ[(size_t)(2)*orbital_buffers.magnetic_capacity+(2)] = 2.18509686118416*siQ*coQ*siP*coP;
        dAFQ[(size_t)(2)*orbital_buffers.magnetic_capacity+(3)] = 1.09254843059208*(1.0 - 2.0*siQ*siQ)*coP;
        dAFQ[(size_t)(2)*orbital_buffers.magnetic_capacity+(4)] = 1.09254843059208*(1.0 - 2.0*siQ*siQ)*siP;

        /* RICS note 72P */

        dAFP[(size_t)(2)*orbital_buffers.magnetic_capacity+(0)] = 0.0;
        dAFP[(size_t)(2)*orbital_buffers.magnetic_capacity+(1)] = -2.18509686118416*siQ*siP*coP;
        dAFP[(size_t)(2)*orbital_buffers.magnetic_capacity+(2)] = 1.09254843059208*siQ*(1.0 - 2.0*siP*siP);
        dAFP[(size_t)(2)*orbital_buffers.magnetic_capacity+(3)] = -1.09254843059208*coQ*siP;
        dAFP[(size_t)(2)*orbital_buffers.magnetic_capacity+(4)] = 1.09254843059208*coQ*coP;
      }
      else if (L0==3){
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(0)] = 0.373176332590116*(5.0*coQ*coQ*coQ - 3.0*coQ);
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(1)] = 0.457045799464466*coP*siQ*(5.0*coQ*coQ - 1.0);
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(2)] = 0.457045799464466*siP*siQ*(5.0*coQ*coQ - 1.0);
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(3)] = 1.44530572132028*siQ*siQ*coQ*(coP*coP-siP*siP);
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(4)] = 2.89061144264055*siQ*siQ*coQ*siP*coP;
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(5)] = 0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP - 3.0*coP);
        AF[(size_t)(3)*orbital_buffers.magnetic_capacity+(6)] = 0.590043589926644*siQ*siQ*siQ*(3.0*siP - 4.0*siP*siP*siP);

        dAFQ[(size_t)(3)*orbital_buffers.magnetic_capacity+(0)] = 0.373176332590116*siQ*(-15.0*coQ*coQ + 3.0);
        dAFQ[(size_t)(3)*orbital_buffers.magnetic_capacity+(1)] = 0.457045799464466*coP*coQ*(15.0*coQ*coQ - 11.0);
        dAFQ[(size_t)(3)*orbital_buffers.magnetic_capacity+(2)] = 0.457045799464466*siP*coQ*(15.0*coQ*coQ - 11.0);
        dAFQ[(size_t)(3)*orbital_buffers.magnetic_capacity+(3)] = 1.44530572132028*(coP*coP-siP*siP)*siQ*(2.0*coQ*coQ-siQ*siQ);
        dAFQ[(size_t)(3)*orbital_buffers.magnetic_capacity+(4)] = 2.89061144264055*coP*siP*siQ*(2.0*coQ*coQ - siQ*siQ);
        dAFQ[(size_t)(3)*orbital_buffers.magnetic_capacity+(5)] = 1.770130769779932*coP*coQ*siQ*siQ*(-3.0 + 4.0*coP*coP);
        dAFQ[(size_t)(3)*orbital_buffers.magnetic_capacity+(6)] = 1.770130769779932*coQ*siP*siQ*siQ*( 3.0 - 4.0*siP*siP);

        /* RICS note 72P */

        dAFP[(size_t)(3)*orbital_buffers.magnetic_capacity+(0)] = 0.0;
        dAFP[(size_t)(3)*orbital_buffers.magnetic_capacity+(1)] = 0.457045799464466*siP*(-5.0*coQ*coQ + 1.0);
        dAFP[(size_t)(3)*orbital_buffers.magnetic_capacity+(2)] = 0.457045799464466*coP*( 5.0*coQ*coQ - 1.0);
        dAFP[(size_t)(3)*orbital_buffers.magnetic_capacity+(3)] = -5.781222885281120*coP*coQ*siP*siQ;
        dAFP[(size_t)(3)*orbital_buffers.magnetic_capacity+(4)] = 2.89061144264055*coQ*siQ*(coP*coP - siP*siP);
        dAFP[(size_t)(3)*orbital_buffers.magnetic_capacity+(5)] = 1.770130769779932*siP*siQ*siQ*(1.0 - 4.0*coP*coP);
        dAFP[(size_t)(3)*orbital_buffers.magnetic_capacity+(6)] = 1.770130769779932*coP*siQ*siQ*(1.0 - 4.0*siP*siP);
      }

      else if (4<=L0){

        for(m=-L0; m<=L0; m++){ 
          ComplexSH(L0,m,Q,P,SH[L0+m],dSHt[L0+m],dSHp[L0+m]);
  }

        AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(0)]   =   -SH[L0][0];
        dAFQ[(size_t)(L0)*orbital_buffers.magnetic_capacity+(0)] = -dSHt[L0][0];
        dAFP[(size_t)(L0)*orbital_buffers.magnetic_capacity+(0)] = -dSHp[L0][0];

        j = -1;
        for (i=1; i<(L0*2+1); i=i+4){
          j++;
          AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*SH[L0-(2*j+1)][0];
          dAFQ[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*dSHt[L0-(2*j+1)][0];
          dAFP[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*dSHp[L0-(2*j+1)][0];
  }

  j = 0;
  for (i=3; i<(L0*2+1); i=i+4){
    j++;
          AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*SH[L0-2*j][0];
          dAFQ[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*dSHt[L0-2*j][0];
          dAFP[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*dSHp[L0-2*j][0];
  }

  j = -1;
  for (i=2; i<(L0*2+1); i=i+4){
    j++;
          AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*SH[L0-(2*j+1)][1];
          dAFQ[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*dSHt[L0-(2*j+1)][1];
          dAFP[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*dSHp[L0-(2*j+1)][1];
  }

  j = 0;
  for (i=4; i<(L0*2+1); i=i+4){
    j++;
          AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*SH[L0-2*j][1];
          dAFQ[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*dSHt[L0-2*j][1];
          dAFP[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = -2.0/sqrt(2.0)*dSHp[L0-2*j][1];
  }
      }

    }
  }

  /* Contracted Chi */
  al = -1;
  for (L0=0; L0<=atomic->species.basis_lmax[wan]; L0++){
    for (Mul0=0; Mul0<atomic->species.contracted_basis[(size_t)(wan)*atomic->species.angular_channels+(L0)]; Mul0++){
      for (M0=0; M0<=2*L0; M0++){
        al++;
        if (po!=0){
          dChi[0][al] = dChi[1][al] = 0.0;
          dChi[2][al] = dChi[3][al] = 0.0;
          continue;
        }

        tmp0 = 0.0;
        tmp1 = 0.0;
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
          tmp0 = tmp0 + CntCoes[Mc_AN][al][p]*RF[(size_t)(L0)*orbital_buffers.multiplicity+(p)];
          tmp1 = tmp1 + CntCoes[Mc_AN][al][p]*dRF[(size_t)(L0)*orbital_buffers.multiplicity+(p)];
        }        

        dChiR = tmp1*AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)];
        dChiQ = tmp0*dAFQ[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)];
        dChiP = tmp0*dAFP[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)];

        dChi[0][al] = tmp0*AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)];
        dChi[1][al] = dRx*dChiR + dQx*dChiQ + dPx*dChiP;
        dChi[2][al] = dRy*dChiR + dQy*dChiQ + dPy*dChiP;
        dChi[3][al] = dRz*dChiR + dQz*dChiQ + dPz*dChiP;
      }
    }
  }


  openmx_core_orbital_destroy(orbital_workspace);

}


