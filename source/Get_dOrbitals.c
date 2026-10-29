/**********************************************************************
  Get_dOrbitals.c:

     Get_dOrbital.c is a subrutine to calculate derivatives
     of basis orbitals

  Log of Get_dOrbitals.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"
#include "openmx_orbitals.h"

void Get_dOrbitals(const OpenMXAtomicViews *atomic, int wan, double x, double y, double z, double **dChi)
{

  int i,i1,i2,i3,i4,j,l,l1;
  int po,L0,Mul0,M0;
  int mp_min,mp_max,m;
  double dum,dum1,dum2,dum3,dum4;
  double siQ,coQ,siP,coP,a,b,c;
  double dx,rm,tmp0,tmp1,id,d;
  double drx,dry,drz,R,Q,P,Rmin;
  double S_coordinate[3];
  double *RF;
  double *dRF;
  double *AF;
  double *dAFQ;
  double *dAFP;
  double h1,h2,h3,f1,f2,f3,f4,dfx,dfx2;
  double g1,g2,x1,x2,y1,y2,y12,y22,f,df,df2;
  double dRx,dRy,dRz,dQx,dQy,dQz,dPx,dPy,dPz;
  double dChiR,dChiQ,dChiP,h,sum0,sum1;
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

  Rmin = 10e-14;

  xyz2spherical(x,y,z,0.0,0.0,0.0,S_coordinate);
  R = S_coordinate[0];
  Q = S_coordinate[1];
  P = S_coordinate[2];

  if (R<Rmin){
    x = x + Rmin;
    y = y + Rmin;
    z = z + Rmin;
    xyz2spherical(x,y,z,0.0,0.0,0.0,S_coordinate);
    R = S_coordinate[0];
    Q = S_coordinate[1];
    P = S_coordinate[2];
  }  

  po = OpenMX_OrbitalRadial(atomic,wan,R,1,&orbital_buffers);

  if (po==0){

    /* Angular */
    siQ = sin(Q);
    coQ = cos(Q);
    siP = sin(P);
    coP = cos(P);
    if (atomic->species.basis_lmax[wan] >= 4)
      openmx_orbital_polar(x, y, z, R, &siQ, &coQ, &siP, &coP, &P);

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

          sum0 = 0.0;
          sum1 = 0.0; 
	  for (j=0; j<(L0*2+1); j++){
            sum0 += openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).r*dSHt[j][0] - openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).i*dSHt[j][1];
            sum1 += openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).r*dSHt[j][1] + openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).i*dSHt[j][0];
	  }
          dAFQ[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = sum0 + sum1;

          sum0 = 0.0;
          sum1 = 0.0; 
	  for (j=0; j<(L0*2+1); j++){
            sum0 += openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).r*dSHp[j][0] - openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).i*dSHp[j][1];
            sum1 += openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).r*dSHp[j][1] + openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).i*dSHp[j][0];
	  }
          dAFP[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = sum0 + sum1;
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
        if (po!=0){
          dChi[0][i1] = dChi[1][i1] = 0.0;
          dChi[2][i1] = dChi[3][i1] = 0.0;
          continue;
        }
       
        dChiR = dRF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)]*AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)];
        dChiQ = RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)]*dAFQ[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)];
        dChiP = RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)]*dAFP[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)];
  
        dChi[0][i1] = RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)]*AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)];
        dChi[1][i1] = -dRx*dChiR - dQx*dChiQ - dPx*dChiP;
        dChi[2][i1] = -dRy*dChiR - dQy*dChiQ - dPy*dChiP;
        dChi[3][i1] = -dRz*dChiR - dQz*dChiQ - dPz*dChiP;
      }
    }
  }


  openmx_core_orbital_destroy(orbital_workspace);

}

/* AITUNE reduce the cost of memory allocation */


void Get_dOrbitals_work(const OpenMXAtomicViews *atomic, int wan, double x, double y, double z, double **dChi, OpenMXOrbitalBuffers buffer)
{

  int i, i1, i2, i3, i4, j, l, l1;
  int po, L0, Mul0, M0;
  int mp_min, mp_max, m;
  double dum, dum1, dum2, dum3, dum4;
  double siQ, coQ, siP, coP, a, b, c;
  double dx, rm, tmp0, tmp1, id, d;
  double drx, dry, drz, R, Q, P, Rmin;
  double S_coordinate[3];
  double *RF = buffer.RF;
  double *dRF = buffer.dRF;
  double *AF = buffer.AF;
  double *dAFQ = buffer.dAFQ;
  double *dAFP = buffer.dAFP;
  double h1, h2, h3, f1, f2, f3, f4, dfx, dfx2;
  double g1, g2, x1, x2, y1, y2, y12, y22, f, df, df2;
  double dRx, dRy, dRz, dQx, dQy, dQz, dPx, dPy, dPz;
  double dChiR, dChiQ, dChiP, h, sum0, sum1;
  double SH[Supported_MaxL * 2 + 1][2];
  double dSHt[Supported_MaxL * 2 + 1][2];
  double dSHp[Supported_MaxL * 2 + 1][2];

  /* start calc. */

  Rmin = 10e-14;

  xyz2spherical(x, y, z, 0.0, 0.0, 0.0, S_coordinate);
  R = S_coordinate[0];
  Q = S_coordinate[1];
  P = S_coordinate[2];

  if (R < Rmin) {
    x = x + Rmin;
    y = y + Rmin;
    z = z + Rmin;
    xyz2spherical(x, y, z, 0.0, 0.0, 0.0, S_coordinate);
    R = S_coordinate[0];
    Q = S_coordinate[1];
    P = S_coordinate[2];
  }

  po = OpenMX_OrbitalRadial(atomic,wan,R,1,&buffer);

  if (po == 0) {

    /* Angular */
    siQ = sin(Q);
    coQ = cos(Q);
    siP = sin(P);
    coP = cos(P);
    if (atomic->species.basis_lmax[wan] >= 4)
      openmx_orbital_polar(x, y, z, R, &siQ, &coQ, &siP, &coP, &P);

    dRx = siQ*coP;
    dRy = siQ*siP;
    dRz = coQ;

    if (Rmin < R) {
      dQx = coQ*coP / R;
      dQy = coQ*siP / R;
      dQz = -siQ / R;
    } else {
      dQx = 0.0;
      dQy = 0.0;
      dQz = 0.0;
    }

    /* RICS note 72P */

    if (Rmin < R) {
      dPx = -siP / R;
      dPy = coP / R;
      dPz = 0.0;
    } else {
      dPx = 0.0;
      dPy = 0.0;
      dPz = 0.0;
    }

    for (L0 = 0; L0 <= atomic->species.basis_lmax[wan]; L0++) {
      if (L0 == 0) {
	AF[(size_t)(0)*buffer.magnetic_capacity+(0)] = 0.282094791773878;
	dAFQ[(size_t)(0)*buffer.magnetic_capacity+(0)] = 0.0;
	dAFP[(size_t)(0)*buffer.magnetic_capacity+(0)] = 0.0;
      } else if (L0 == 1) {
	dum = 0.48860251190292*siQ;

	AF[(size_t)(1)*buffer.magnetic_capacity+(0)] = dum*coP;
	AF[(size_t)(1)*buffer.magnetic_capacity+(1)] = dum*siP;
	AF[(size_t)(1)*buffer.magnetic_capacity+(2)] = 0.48860251190292*coQ;

	dAFQ[(size_t)(1)*buffer.magnetic_capacity+(0)] = 0.48860251190292*coQ*coP;
	dAFQ[(size_t)(1)*buffer.magnetic_capacity+(1)] = 0.48860251190292*coQ*siP;
	dAFQ[(size_t)(1)*buffer.magnetic_capacity+(2)] = -0.48860251190292*siQ;

	dAFP[(size_t)(1)*buffer.magnetic_capacity+(0)] = -0.48860251190292*siP;
	dAFP[(size_t)(1)*buffer.magnetic_capacity+(1)] = 0.48860251190292*coP;
	dAFP[(size_t)(1)*buffer.magnetic_capacity+(2)] = 0.0;
      } else if (L0 == 2) {

	dum1 = siQ*siQ;
	dum2 = 1.09254843059208*siQ*coQ;
	AF[(size_t)(2)*buffer.magnetic_capacity+(0)] = 0.94617469575756*coQ*coQ - 0.31539156525252;
	AF[(size_t)(2)*buffer.magnetic_capacity+(1)] = 0.54627421529604*dum1*(1.0 - 2.0*siP*siP);
	AF[(size_t)(2)*buffer.magnetic_capacity+(2)] = 1.09254843059208*dum1*siP*coP;
	AF[(size_t)(2)*buffer.magnetic_capacity+(3)] = dum2*coP;
	AF[(size_t)(2)*buffer.magnetic_capacity+(4)] = dum2*siP;

	dAFQ[(size_t)(2)*buffer.magnetic_capacity+(0)] = -1.89234939151512*siQ*coQ;
	dAFQ[(size_t)(2)*buffer.magnetic_capacity+(1)] = 1.09254843059208*siQ*coQ*(1.0 - 2.0*siP*siP);
	dAFQ[(size_t)(2)*buffer.magnetic_capacity+(2)] = 2.18509686118416*siQ*coQ*siP*coP;
	dAFQ[(size_t)(2)*buffer.magnetic_capacity+(3)] = 1.09254843059208*(1.0 - 2.0*siQ*siQ)*coP;
	dAFQ[(size_t)(2)*buffer.magnetic_capacity+(4)] = 1.09254843059208*(1.0 - 2.0*siQ*siQ)*siP;

	/* RICS note 72P */

	dAFP[(size_t)(2)*buffer.magnetic_capacity+(0)] = 0.0;
	dAFP[(size_t)(2)*buffer.magnetic_capacity+(1)] = -2.18509686118416*siQ*siP*coP;
	dAFP[(size_t)(2)*buffer.magnetic_capacity+(2)] = 1.09254843059208*siQ*(1.0 - 2.0*siP*siP);
	dAFP[(size_t)(2)*buffer.magnetic_capacity+(3)] = -1.09254843059208*coQ*siP;
	dAFP[(size_t)(2)*buffer.magnetic_capacity+(4)] = 1.09254843059208*coQ*coP;
      }

      else if (L0 == 3) {
	AF[(size_t)(3)*buffer.magnetic_capacity+(0)] = 0.373176332590116*(5.0*coQ*coQ*coQ - 3.0*coQ);
	AF[(size_t)(3)*buffer.magnetic_capacity+(1)] = 0.457045799464466*coP*siQ*(5.0*coQ*coQ - 1.0);
	AF[(size_t)(3)*buffer.magnetic_capacity+(2)] = 0.457045799464466*siP*siQ*(5.0*coQ*coQ - 1.0);
	AF[(size_t)(3)*buffer.magnetic_capacity+(3)] = 1.44530572132028*siQ*siQ*coQ*(coP*coP - siP*siP);
	AF[(size_t)(3)*buffer.magnetic_capacity+(4)] = 2.89061144264055*siQ*siQ*coQ*siP*coP;
	AF[(size_t)(3)*buffer.magnetic_capacity+(5)] = 0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP - 3.0*coP);
	AF[(size_t)(3)*buffer.magnetic_capacity+(6)] = 0.590043589926644*siQ*siQ*siQ*(3.0*siP - 4.0*siP*siP*siP);

	dAFQ[(size_t)(3)*buffer.magnetic_capacity+(0)] = 0.373176332590116*siQ*(-15.0*coQ*coQ + 3.0);
	dAFQ[(size_t)(3)*buffer.magnetic_capacity+(1)] = 0.457045799464466*coP*coQ*(15.0*coQ*coQ - 11.0);
	dAFQ[(size_t)(3)*buffer.magnetic_capacity+(2)] = 0.457045799464466*siP*coQ*(15.0*coQ*coQ - 11.0);
	dAFQ[(size_t)(3)*buffer.magnetic_capacity+(3)] = 1.44530572132028*(coP*coP - siP*siP)*siQ*(2.0*coQ*coQ - siQ*siQ);
	dAFQ[(size_t)(3)*buffer.magnetic_capacity+(4)] = 2.89061144264055*coP*siP*siQ*(2.0*coQ*coQ - siQ*siQ);
	dAFQ[(size_t)(3)*buffer.magnetic_capacity+(5)] = 1.770130769779932*coP*coQ*siQ*siQ*(-3.0 + 4.0*coP*coP);
	dAFQ[(size_t)(3)*buffer.magnetic_capacity+(6)] = 1.770130769779932*coQ*siP*siQ*siQ*(3.0 - 4.0*siP*siP);

	/* RICS note 72P */

	dAFP[(size_t)(3)*buffer.magnetic_capacity+(0)] = 0.0;
	dAFP[(size_t)(3)*buffer.magnetic_capacity+(1)] = 0.457045799464466*siP*(-5.0*coQ*coQ + 1.0);
	dAFP[(size_t)(3)*buffer.magnetic_capacity+(2)] = 0.457045799464466*coP*(5.0*coQ*coQ - 1.0);
	dAFP[(size_t)(3)*buffer.magnetic_capacity+(3)] = -5.781222885281120*coP*coQ*siP*siQ;
	dAFP[(size_t)(3)*buffer.magnetic_capacity+(4)] = 2.89061144264055*coQ*siQ*(coP*coP - siP*siP);
	dAFP[(size_t)(3)*buffer.magnetic_capacity+(5)] = 1.770130769779932*siP*siQ*siQ*(1.0 - 4.0*coP*coP);
	dAFP[(size_t)(3)*buffer.magnetic_capacity+(6)] = 1.770130769779932*coP*siQ*siQ*(1.0 - 4.0*siP*siP);
      }

      else if (4 <= L0) {

	/* calculation of complex spherical harmonics functions */
	for (m = -L0; m <= L0; m++) {
	  openmx_orbital_complex_sh(L0, m, siQ, coQ, P, SH[L0 + m], dSHt[L0 + m], dSHp[L0 + m]);
	}

	/* transformation of complex to real */
	for (i = 0; i < (L0 * 2 + 1); i++) {

	  sum0 = 0.0;
	  sum1 = 0.0;
	  for (j = 0; j < (L0 * 2 + 1); j++) {
	    sum0 += openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).r*SH[j][0] - openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).i*SH[j][1];
	    sum1 += openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).r*SH[j][1] + openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).i*SH[j][0];
	  }
	  AF[(size_t)(L0)*buffer.magnetic_capacity+(i)] = sum0 + sum1;

	  sum0 = 0.0;
	  sum1 = 0.0;
	  for (j = 0; j < (L0 * 2 + 1); j++) {
	    sum0 += openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).r*dSHt[j][0] - openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).i*dSHt[j][1];
	    sum1 += openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).r*dSHt[j][1] + openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).i*dSHt[j][0];
	  }
	  dAFQ[(size_t)(L0)*buffer.magnetic_capacity+(i)] = sum0 + sum1;

	  sum0 = 0.0;
	  sum1 = 0.0;
	  for (j = 0; j < (L0 * 2 + 1); j++) {
	    sum0 += openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).r*dSHp[j][0] - openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).i*dSHp[j][1];
	    sum1 += openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).r*dSHp[j][1] + openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).i*dSHp[j][0];
	  }
	  dAFP[(size_t)(L0)*buffer.magnetic_capacity+(i)] = sum0 + sum1;
	}

      }
    }
  }

  /* Chi */
  i1 = -1;
  for (L0 = 0; L0 <= atomic->species.basis_lmax[wan]; L0++) {
    for (Mul0 = 0; Mul0 < atomic->species.basis[(size_t)(wan)*atomic->species.angular_channels+(L0)]; Mul0++) {
      for (M0 = 0; M0 <= 2 * L0; M0++) {

	i1++;
	if (po!=0){
	  dChi[0][i1] = dChi[1][i1] = 0.0;
	  dChi[2][i1] = dChi[3][i1] = 0.0;
	  continue;
	}

	dChiR = dRF[(size_t)(L0)*buffer.multiplicity+(Mul0)] * AF[(size_t)(L0)*buffer.magnetic_capacity+(M0)];
	dChiQ = RF[(size_t)(L0)*buffer.multiplicity+(Mul0)] * dAFQ[(size_t)(L0)*buffer.magnetic_capacity+(M0)];
	dChiP = RF[(size_t)(L0)*buffer.multiplicity+(Mul0)] * dAFP[(size_t)(L0)*buffer.magnetic_capacity+(M0)];

	dChi[0][i1] = RF[(size_t)(L0)*buffer.multiplicity+(Mul0)] * AF[(size_t)(L0)*buffer.magnetic_capacity+(M0)];
	dChi[1][i1] = -dRx*dChiR - dQx*dChiQ - dPx*dChiP;
	dChi[2][i1] = -dRy*dChiR - dQy*dChiQ - dPy*dChiP;
	dChi[3][i1] = -dRz*dChiR - dQz*dChiQ - dPz*dChiP;
      }
    }
  }

	
}


void Get_dOrbitals2(const OpenMXAtomicViews *atomic, int wan, double x, double y, double z, double **dChi)
{

  int i,i1,i2,i3,i4,j,l,l1;
  int po,L0,Mul0,M0;
  int mp_min,mp_max,m;
  double dum,dum1,dum2,dum3,dum4;
  double siQ,coQ,siP,coP,a,b,c;
  double dx,rm,tmp0,tmp1,id,d;
  double drx,dry,drz,R,Q,P,Rmin;
  double S_coordinate[3];
  double *RF;
  double *dRF;
  double *AF;
  double *dAFQ;
  double *dAFP;
  double h1,h2,h3,f1,f2,f3,f4,dfx,dfx2;
  double g1,g2,x1,x2,y1,y2,y12,y22,f,df,df2;
  double dRx,dRy,dRz,dQx,dQy,dQz,dPx,dPy,dPz;
  double dChiR,dChiQ,dChiP,h,sum0,sum1;
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

  Rmin = 10e-14;

  xyz2spherical(x,y,z,0.0,0.0,0.0,S_coordinate);
  R = S_coordinate[0];
  Q = S_coordinate[1];
  P = S_coordinate[2];

  if (R<Rmin){
    x = x + Rmin;
    y = y + Rmin;
    z = z + Rmin;
    xyz2spherical(x,y,z,0.0,0.0,0.0,S_coordinate);
    R = S_coordinate[0];
    Q = S_coordinate[1];
    P = S_coordinate[2];
  }  

  po = OpenMX_OrbitalRadial(atomic,wan,R,1,&orbital_buffers);

  if (po==0){

    /* Angular */
    siQ = sin(Q);
    coQ = cos(Q);
    siP = sin(P);
    coP = cos(P);
    if (atomic->species.basis_lmax[wan] >= 4)
      openmx_orbital_polar(x, y, z, R, &siQ, &coQ, &siP, &coP, &P);

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

          sum0 = 0.0;
          sum1 = 0.0; 
    for (j=0; j<(L0*2+1); j++){
            sum0 += openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).r*dSHt[j][0] - openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).i*dSHt[j][1];
            sum1 += openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).r*dSHt[j][1] + openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).i*dSHt[j][0];
    }
          dAFQ[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = sum0 + sum1;

          sum0 = 0.0;
          sum1 = 0.0; 
    for (j=0; j<(L0*2+1); j++){
            sum0 += openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).r*dSHp[j][0] - openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).i*dSHp[j][1];
            sum1 += openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).r*dSHp[j][1] + openmx_real_harmonic_coefficient(&atomic->real_harmonics,L0,i,j).i*dSHp[j][0];
    }
          dAFP[(size_t)(L0)*orbital_buffers.magnetic_capacity+(i)] = sum0 + sum1;
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
        if (po!=0){
          dChi[0][i1] = dChi[1][i1] = 0.0;
          dChi[2][i1] = dChi[3][i1] = 0.0;
          continue;
        }
       
        dChiR = dRF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)]*AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)];
        dChiQ = RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)]*dAFQ[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)];
        dChiP = RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)]*dAFP[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)];
  
        dChi[0][i1] = RF[(size_t)(L0)*orbital_buffers.multiplicity+(Mul0)]*AF[(size_t)(L0)*orbital_buffers.magnetic_capacity+(M0)];
        dChi[1][i1] = dRx*dChiR + dQx*dChiQ + dPx*dChiP;
        dChi[2][i1] = dRy*dChiR + dQy*dChiQ + dPy*dChiP;
        dChi[3][i1] = dRz*dChiR + dQz*dChiQ + dPz*dChiP;
      }
    }
  }


  openmx_core_orbital_destroy(orbital_workspace);

}


