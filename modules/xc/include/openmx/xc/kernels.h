#ifndef OPENMX_XC_KERNELS_H
#define OPENMX_XC_KERNELS_H

/* The arithmetic of the OpenMX exchange-correlation functionals as inline C
   functions: one implementation for the C++ library (functionals.cpp), for
   C callers, and for the application's GPU device code (OpenACC routines
   under nvc, OpenMP target regions under hipcc/clang).  The expressions keep
   the operand order and the literal constants of OpenMX 4.0's XC_*.c
   routines (T. Ozaki); the library's tests compare them bitwise with the
   frozen originals.  Hartree atomic units.

   Conventions: dens[2] is (up, down), gdens[3][2] is [direction][spin].
   `symmetrize` is the LDA+U cFLL rule of the originals: both spins see the
   averaged density and the polarization is zero.  The pw92 and pbe kernels
   rewrite dens with the floored densities as the originals did.

   Defining OPENMX_XC_KERNELS_OMP_TARGET before the include places the
   functions in an OpenMP declare-target region; under OpenACC each function
   is a sequential routine.

   The functions are inlined into the including unit, so they agree bitwise
   with the library only when that unit is compiled without floating-point
   contraction (-ffp-contract=off, as the library is); a unit that contracts
   (GPU device code, or a host compiler's default) differs in the last bits. */

#include <math.h>

#if defined(_OPENACC)
#define OPENMX_XC_KERNEL _Pragma("acc routine seq") static inline
#else
#define OPENMX_XC_KERNEL static inline
#endif

#define OPENMX_XC_PI 3.1415926535897932384626

#if defined(OPENMX_XC_KERNELS_OMP_TARGET)
#pragma omp declare target
#endif

OPENMX_XC_KERNEL double openmx_xc_kernel_largest(double a, double b)
{
  return b <= a ? a : b;
}

/* Ceperley-Alder LDA (Perdew-Zunger).  quantity: 0 energy density,
   1 potential, 2 energy minus potential, 3 derivative of the energy density;
   zero below 1e-15 bohr^-3 and for any other quantity. */
OPENMX_XC_KERNEL double openmx_xc_kernel_ceperley_alder(double den, int quantity)
{
  const double coe = 0.6203504908994;  /* pow(3.0/4.0/PI,1.0/3.0); */
  double rs, tmp0, Ex, dEx, Ec, dEc;
  if (den <= 1.0e-15) return 0.0;
  rs = coe*pow(den,-0.3333333333333333333);
  tmp0 = 0.458165293632163/rs;
  Ex = -tmp0;
  dEx = tmp0/rs;
  if (1.0<=rs){
    double dum, tmp1;
    tmp0 = sqrt(rs);
    dum = (1.0 + 1.0529*tmp0 + 0.3334*rs);
    tmp1 = 0.1423/dum;
    Ec = -tmp1;
    dEc = tmp1/dum*(0.52645/tmp0 + 0.3334);
  }
  else{
    tmp0 = log(rs);
    Ec = -0.0480 + 0.0311*tmp0 + rs*(0.0020*tmp0 - 0.0116);
    dEc = 0.0311/rs + 0.0020*tmp0 - 0.0096;
  }
  if (quantity == 0) return Ex + Ec;
  if (quantity == 1) return Ex + Ec - 0.33333333333333333333*rs*(dEx + dEc);
  if (quantity == 2) return 0.3333333333333333333*rs*(dEx + dEc);
  if (quantity == 3) return -0.3333333333333333333/(coe*coe*coe)*rs*rs*rs*rs*(dEx + dEc);
  return 0.0;
}

/* Ceperley-Alder LSDA.  quantity: 0 energy density, 1 potential, 2 energy
   minus potential; xc, x, c receive the per-spin values (zeros below 1e-15
   bohr^-3 total density and for any other quantity). */
OPENMX_XC_KERNEL void openmx_xc_kernel_ca_lsda(double den0, double den1, int quantity, int symmetrize,
                                               double xc[2], double x[2], double c[2])
{
  const double min_den = 1.0e-15;
  const double coe = 0.6203504908994;  /* pow(3.0/4.0/PI,1.0/3.0); */
  const double tden = den0 + den1;
  double zeta, rs, tmp0, ExP, dExP, ExF, dExF, EcP, dEcP, EcF, dEcF;
  double X0, X1, C0, C1;
  xc[0] = 0.0; xc[1] = 0.0; x[0] = 0.0; x[1] = 0.0; c[0] = 0.0; c[1] = 0.0;
  if (tden<min_den) return;

  zeta = (den0 - den1)/tden;
  if (1.0<zeta)  zeta =  1.0 - min_den;
  if (zeta<-1.0) zeta = -1.0 + min_den;
  if (symmetrize) zeta = 0.0;
  if (tden<min_den)
    rs = 62035.04908994;  /* coe*pow(1.0e-15,-1.0/3.0); */
  else
    rs = coe*pow(tden,-0.3333333333333333333);

  tmp0 = 0.458165293632163/rs;
  ExP  = -tmp0;
  dExP = tmp0/rs;
  ExF  = 1.25992104989487*ExP;
  dExF = 1.25992104989487*dExP;
  if (1.0<=rs){
    double dum, tmp1;
    tmp0 = sqrt(rs);
    dum = (1.0 + 1.0529*tmp0 + 0.3334*rs);
    tmp1 = 0.1423/dum;
    EcP = -tmp1;
    dEcP = tmp1/dum*(0.52645/tmp0 + 0.3334);
    dum = (1.0 + 1.3981*tmp0 + 0.2611*rs);
    tmp1 = 0.0843/dum;
    EcF = -tmp1;
    dEcF = tmp1/dum*(0.69905/tmp0 + 0.2611);
  }
  else{
    tmp0 = log(rs);
    EcP = -0.0480 + 0.0311*tmp0 + rs*(0.0020*tmp0 - 0.0116);
    dEcP = 0.0311/rs + 0.0020*tmp0 - 0.0096;
    EcF = -0.0269 + 0.01555*tmp0 + rs*(0.0007*tmp0 - 0.0048);
    dEcF = 0.01555/rs + 0.0007*tmp0 - 0.0041;
  }

  if (quantity == 0){
    const double z0 = pow(1.0+zeta,1.33333333333333333);
    const double z1 = pow(1.0-zeta,1.33333333333333333);
    const double fzeta = 1.92366105093154*(z0 + z1 - 2.0);
    X0 = ExP + (ExF - ExP)*fzeta;
    X1 = X0;
    C0 = EcP + (EcF - EcP)*fzeta;
    C1 = C0;
  }
  else if (quantity == 1){
    const double z0 = pow(1.0+zeta,0.33333333333333333);
    const double z1 = pow(1.0-zeta,0.33333333333333333);
    const double z02 = z0*z0, z04 = z02*z02, z12 = z1*z1, z14 = z12*z12;
    const double fzeta  = 1.92366105093154*(z04 + z14 - 2.0);
    const double dfzeta = 2.56488140124205*(z0 - z1);
    double Exc, dExc, Vxc;
    tmp0 = ExF - ExP;
    Exc = ExP + tmp0*fzeta;
    dExc = dExP + (dExF - dExP)*fzeta;
    Vxc = Exc - 0.33333333333333333333*rs*dExc;
    X0 = Vxc + tmp0*( 1.0 - zeta)*dfzeta;
    X1 = Vxc + tmp0*(-1.0 - zeta)*dfzeta;
    tmp0 = EcF - EcP;
    Exc = EcP + tmp0*fzeta;
    dExc = dEcP + (dEcF - dEcP)*fzeta;
    Vxc = Exc - 0.33333333333333333333*rs*dExc;
    C0 = Vxc + tmp0*( 1.0 - zeta)*dfzeta;
    C1 = Vxc + tmp0*(-1.0 - zeta)*dfzeta;
  }
  else if (quantity == 2){
    const double z0 = pow(1.0+zeta,0.33333333333333333);
    const double z1 = pow(1.0-zeta,0.33333333333333333);
    const double z02 = z0*z0, z04 = z02*z02, z12 = z1*z1, z14 = z12*z12;
    const double fzeta  = 1.92366105093154*(z04 + z14 - 2.0);
    const double dfzeta = 2.56488140124205*(z0 - z1);
    double Exc, dExc, tmp1, tmp2;
    tmp0 = ExF - ExP;
    Exc = ExP + tmp0*fzeta;
    dExc = dExP + (dExF - dExP)*fzeta;
    tmp1 = 0.33333333333333333333*rs*dExc;
    tmp2 = tmp0*dfzeta;
    X0 = tmp1 - tmp2*( 1.0 - zeta);
    X1 = tmp1 - tmp2*(-1.0 - zeta);
    tmp0 = EcF - EcP;
    Exc = EcP + tmp0*fzeta;
    dExc = dEcP + (dEcF - dEcP)*fzeta;
    tmp1 = 0.33333333333333333333*rs*dExc;
    tmp2 = tmp0*dfzeta;
    C0 = tmp1 - tmp2*( 1.0 - zeta);
    C1 = tmp1 - tmp2*(-1.0 - zeta);
    (void)Exc;
  }
  else return;
  xc[0] = X0 + C0;
  xc[1] = X1 + C1;
  x[0] = X0;
  x[1] = X1;
  c[0] = C0;
  c[1] = C1;
}

/* Perdew-Wang 92 correlation of the uniform gas: ec[0] energy density, vc
   the two potentials; dens is floored to 0.5e-14 per spin below 1e-14. */
OPENMX_XC_KERNEL void openmx_xc_kernel_pw92(double dens[2], int symmetrize, double ec[1], double vc[2])
{
  const double den_min = 1.0e-14, den_min_half = 0.5*1.0e-14;
  /* parameters from Table I of Perdew and Wang, PRB, 45, 13244 (92) */
  const double A[3]      = {0.0310910,  0.0155450,  0.0168870};
  const double alpha1[3] = {0.2137000,  0.2054800,  0.1112500};
  const double beta1[3]  = {7.5957000, 14.1189000, 10.3570000};
  const double beta2[3]  = {3.5876000,  6.1977000,  3.6231000};
  const double beta3[3]  = {1.6382000,  3.3662000,  0.8802600};
  const double beta4[3]  = {0.4929400,  0.6251700,  0.4967100};
  const double coe = 0.6203504908994;  /* pow(3.0/4.0/PI,1.0/3.0);   */
  double dtot = dens[0] + dens[1];
  double rs, tmp0, zeta, drsdd, dzdd[2], srs, G[3], dGdrs[3];
  double c, fpp0, dum1, dum2, tmp1, tmp2, tmp12, tmp22, tmp14, tmp24, f, dfdz, dum;
  double dEcdrs, dEcdz, dEcdd[2];
  int i;
  if (dtot<=den_min){
    rs = 6203.504908994;  /* coe*pow(1.0e-12,-1.0/3.0); */
    dens[0] = den_min_half;
    dens[1] = den_min_half;
    dtot = den_min;
  }
  else
    rs = coe*pow(dtot,-0.33333333333333333333333);
  tmp0 = 1.0/dtot;
  zeta = tmp0*(dens[0] - dens[1]);
  if (0.99<zeta) zeta =  0.99;
  if (zeta<-1.0) zeta = -0.99;
  if (symmetrize) zeta = 0.0;
  drsdd = -0.3333333333333333333333*rs*tmp0;
  dzdd[0] = tmp0*( 1.0 - zeta);
  dzdd[1] = tmp0*(-1.0 - zeta);

  srs = sqrt(rs);
  for (i=0; i<=2; i++){
    const double b = beta1[i]*srs + rs*(beta2[i] + beta3[i]*srs + beta4[i]*rs);
    const double dbdrs =  beta1[i]*0.50/srs
                        + beta2[i]
                        + beta3[i]*1.50*srs
                        + beta4[i]*2.0*rs;
    const double cc = 1.0 + 1.0/(2.0*A[i]*b);
    const double dcdrs = -(cc - 1.0)*dbdrs/b;
    const double lg = log(cc);
    const double d1 = 1.0 + alpha1[i]*rs;
    G[i] = -2.0*A[i]*d1*lg;
    dGdrs[i] = -2.0*A[i]*(alpha1[i]*lg + d1*dcdrs/cc);
  }
  c = 1.92366105093154;       /* 1/(2*(2^{1/3}-1)) */
  fpp0 = 1.70992093416137;
  dum1 = 1.0 + zeta;
  dum2 = 1.0 - zeta;
  tmp1  = pow(dum1,0.333333333333333333);
  tmp2  = pow(dum2,0.333333333333333333);
  tmp12 = tmp1*tmp1;
  tmp22 = tmp2*tmp2;
  tmp14 = tmp12*tmp12;
  tmp24 = tmp22*tmp22;
  f = (tmp14 + tmp24 - 2.0)*c;
  dfdz = 1.333333333333333333*(tmp1 - tmp2)*c;
  dum1 = zeta*zeta*zeta;
  dum  = dum1*zeta;
  ec[0] = G[0] - G[2]*f/fpp0*(1.0 - dum) + (G[1] - G[0])*f*dum;
  dEcdrs =   dGdrs[0] - dGdrs[2]*f/fpp0*(1.0 - dum)
           + (dGdrs[1] - dGdrs[0])*f*dum;
  dEcdz = - G[2]/fpp0*(dfdz*(1.0 - dum) - f*4.0*dum1)
          + (G[1] - G[0])*(dfdz*dum + f*4.0*dum1);
  dum = dEcdrs*drsdd;
  dEcdd[0] = dum + dEcdz*dzdd[0];
  dEcdd[1] = dum + dEcdz*dzdd[1];
  vc[0] = ec[0] + dtot*dEcdd[0];
  vc[1] = ec[0] + dtot*dEcdd[1];
}

/* Slater exchange of the unpolarized uniform gas (XC_EX with NSP = 1):
   ex[0] energy density, vx[0] potential; evaluated at the floor below
   1e-14 bohr^-3. */
OPENMX_XC_KERNEL void openmx_xc_kernel_exchange(double density, double ex[1], double vx[1])
{
  const double TRD = 0.333333333333333;
  const double den_min = 1.0e-14;
  const double COE1 = 0.6203504908994;   /* pow(3.0/4.0/PI,1.0/3.0); */
  const double COE2 = 0.610887057710857; /* 3.0*ALP/(2.0*PI*A0)      */
  double RS, VXP;
  if (density<=den_min){
    RS = COE1*pow(den_min,-1.0/3.0);
  }
  else{
    RS = COE1*pow(density,-TRD);
  }
  VXP = -COE2/RS;
  vx[0] = VXP;
  ex[0] = 0.750*VXP;
}

/* GGA-PBE: exc[0] exchange and exc[1] correlation energy density; dexdd,
   decdd per spin; dexdgd, decdgd [direction][spin].  dens is floored to
   0.5e-14 per spin. */
OPENMX_XC_KERNEL void openmx_xc_kernel_pbe(double dens[2], double gdens[3][2], int symmetrize,
                                           double exc[2], double dexdd[2], double decdd[2],
                                           double dexdgd[3][2], double decdgd[3][2])
{
  const double HALF = 0.50, THD = (1.0/3.0), TWO = 2.0, TWOTHD = (2.0/3.0);
  const double beta = 0.06672455060314922, kappa = 0.8040;
  const double gamma = (1.0 - log(TWO))/(OPENMX_XC_PI*OPENMX_XC_PI);
  const double mu = beta*OPENMX_XC_PI*OPENMX_XC_PI/3.0;
  const double den_min = 1.0e-14, gd_min = 1.0e-14;
  int IS, IX;
  double D[2], dt, GD[3][2], GDT[3], GDM[2], GDMT;
  double Ec_unif[1], Vc_unif[2];
  double rs, kF, ks, zeta, phi, t, f1c, f2, A, f3, f4, H, Fc;
  double DRSDD, DKFDD, DKSDD, DZDD[2], DPDZ, DFCDD[2], DFCDGD[3][2];
  double Fx, DFXDD[2], DFXDGD[3][2];
  const double coe = 1.0;

  dens[0] = openmx_xc_kernel_largest(0.5*den_min,dens[0]);
  dens[1] = openmx_xc_kernel_largest(0.5*den_min,dens[1]);
  if (symmetrize){
    D[0] = (dens[0]+dens[1])/2.0;
    D[1] = (dens[0]+dens[1])/2.0;
  }
  else{
    D[0] = dens[0];
    D[1] = dens[1];
  }
  dt = dens[0] + dens[1];
  for (IX=0; IX<=2; IX++){
    if (symmetrize){
      GD[IX][0] = (gdens[IX][0]+gdens[IX][1])/2.0;
      GD[IX][1] = (gdens[IX][0]+gdens[IX][1])/2.0;
    }
    else{
      GD[IX][0] = gdens[IX][0];
      GD[IX][1] = gdens[IX][1];
    }
    GDT[IX] = gdens[IX][0] + gdens[IX][1];
  }
  GDM[0] = sqrt(GD[0][0]*GD[0][0] + GD[1][0]*GD[1][0] + GD[2][0]*GD[2][0]);
  GDM[1] = sqrt(GD[0][1]*GD[0][1] + GD[1][1]*GD[1][1] + GD[2][1]*GD[2][1]);
  GDMT = sqrt(GDT[0]*GDT[0] + GDT[1]*GDT[1] + GDT[2]*GDT[2]);
  GDMT = openmx_xc_kernel_largest(gd_min, GDMT);

  /* local correlation energy and potential (the original passed its floored
     density array, which the PW92 routine could rewrite) */
  openmx_xc_kernel_pw92(dens, symmetrize, Ec_unif, Vc_unif);

  /* total correlation energy */
  rs = pow(3.0/(4.0*OPENMX_XC_PI*dt),THD);
  kF = pow(3.0*OPENMX_XC_PI*OPENMX_XC_PI*dt,THD);
  ks = sqrt(4.0*kF/OPENMX_XC_PI);
  zeta = (dens[0] - dens[1])/dt;
  if (0.99<zeta) zeta =  0.99;
  if (zeta<-1.0) zeta = -0.99;
  if (symmetrize) zeta = 0.0;
  phi = 0.50*(pow(1.0 + zeta,TWOTHD)
            + pow(1.0 - zeta,TWOTHD));
  t = GDMT/(2.0*phi*ks*dt);
  f1c = Ec_unif[0]/(gamma*phi*phi*phi);
  f2 = exp(-f1c);
  A = beta/gamma/(f2 - 1.0);
  f3 = t*t + A*t*t*t*t;
  f4 = beta/gamma * f3/(1.0 + A*f3);
  H = gamma*phi*phi*phi*log(1.0 + f4);
  Fc = Ec_unif[0] + H;

  /* correlation energy derivatives */
  DRSDD = -(THD*rs/dt);
  DKFDD = THD*kF/dt;
  DKSDD = HALF*ks*DKFDD/kF;
  DZDD[0] = 1.0/dt - zeta/dt;
  DZDD[1] = -(1.0/dt) - zeta/dt;
  DPDZ = HALF*TWOTHD*(1.0/pow(1.0+zeta,THD) - 1.0/pow(1.0-zeta,THD));
  for (IS=0; IS<=1; IS++){
    const double DECUDD = (Vc_unif[IS] - Ec_unif[0])/dt;
    const double DPDD = DPDZ*DZDD[IS];
    const double DTDD = (-t)*(DPDD/phi + DKSDD/ks + 1.0/dt);
    const double DF1DD = f1c*(DECUDD/Ec_unif[0] - 3.0*DPDD/phi);
    const double DF2DD = (-f2)*DF1DD;
    const double DADD = (-A)*DF2DD/(f2 - 1.0);
    const double DF3DD = (2.0*t + 4.0*A*t*t*t) * DTDD + DADD*t*t*t*t;
    const double DF4DD = f4*(DF3DD/f3 - (DADD*f3+A*DF3DD)/(1.0 + A*f3));
    double DHDD = 3.0*H*DPDD/phi;
    DHDD = DHDD + gamma*phi*phi*phi*DF4DD/(1.0 + f4);
    DFCDD[IS] = Vc_unif[IS] + H + dt * DHDD;
    for (IX=0; IX<=2; IX++){
      const double DTDGD = (t/GDMT)*GDT[IX]/GDMT;
      const double DF3DGD = DTDGD*(2.0*t + 4.0*A*t*t*t);
      const double DF4DGD = f4*DF3DGD*(1.0/f3 - A/(1.0 + A*f3));
      const double DHDGD = gamma*phi*phi*phi*DF4DGD/(1.0 + f4);
      DFCDGD[IX][IS] = dt*DHDGD;
    }
  }
  (void)DRSDD;

  /* exchange energy and potential */
  Fx = 0.0;
  for (IS=0; IS<=1; IS++){
    const double DS = openmx_xc_kernel_largest(den_min,2.0*D[IS]);
    const double GDMS = openmx_xc_kernel_largest(gd_min, 2.0*GDM[IS]);
    const double KFS = pow(3.0*OPENMX_XC_PI*OPENMX_XC_PI*DS,THD);
    const double s = GDMS/(2.0*KFS*DS);
    const double f1 = 1.0 + mu*s*s/kappa;
    const double f = 1.0 + kappa - kappa/f1;
    double Ex_unif[1], Vx_unif[1];
    double DKFDDx, DSDD, DF1DD, DFDD;
    openmx_xc_kernel_exchange(DS, Ex_unif, Vx_unif);   /* nspin = 1 in the original call */
    Fx = Fx + DS*Ex_unif[0]*f;
    DKFDDx = THD * KFS/DS;
    DSDD = s*(-(DKFDDx/KFS) - 1.0/DS);
    DF1DD = 2.0*(f1 - 1.0)*DSDD/s;
    DFDD = kappa*DF1DD/(f1*f1);
    DFXDD[IS] = Vx_unif[0]*f + DS*Ex_unif[0]*DFDD;
    for (IX=0; IX<=2; IX++){
      const double GDS = 2.0*GD[IX][IS];
      const double DSDGD = (s/GDMS)*GDS/GDMS;
      const double DF1DGD = 2.0*mu*s*DSDGD/kappa;
      const double DFDGD = kappa*DF1DGD/(f1*f1);
      DFXDGD[IX][IS] = DS*Ex_unif[0]*DFDGD;
    }
  }
  Fx = HALF*Fx/dt;

  exc[0] = coe*Fx;
  exc[1] = Fc;
  for (IS=0; IS<=1; IS++){
    dexdd[IS] = coe*DFXDD[IS];
    decdd[IS] = DFCDD[IS];
    for (IX=0; IX<=2; IX++){
      dexdgd[IX][IS] = coe*DFXDGD[IX][IS];
      decdgd[IX][IS] = DFCDGD[IX][IS];
    }
  }
}

#if defined(OPENMX_XC_KERNELS_OMP_TARGET)
#pragma omp end declare target
#endif

#endif
