#include <openmx/xc/functionals.hpp>

#include <cmath>

// Faithful ports of OpenMX 4.0's XC_Ceperly_Alder.c, XC_CA_LSDA.c,
// XC_PW92C.c, XC_EX.c and XC_PBE.c (T. Ozaki; PBE after the SIESTA Fortran
// by the SIESTA group).  Every expression keeps the original operand order
// and literal constants: this translation unit is built without
// floating-point contraction, and the tests compare it bitwise with the
// frozen originals.  Only the global flag dc_Type became the explicit
// `symmetrize_spins` argument.

namespace openmx::xc {
namespace {

constexpr double PI = 3.1415926535897932384626;

double largest(double a, double b) noexcept { return b <= a ? a : b; }

}  // namespace

double ceperley_alder(double den, CeperleyAlderQuantity quantity) noexcept
{
  if (den <= 1.0e-15) return 0.0;
  const double coe = 0.6203504908994;  /* pow(3.0/4.0/PI,1.0/3.0); */
  const double rs = coe*std::pow(den,-0.3333333333333333333);
  double tmp0 = 0.458165293632163/rs;
  const double Ex = -tmp0;
  const double dEx = tmp0/rs;
  double Ec, dEc;
  if (1.0<=rs){
    tmp0 = std::sqrt(rs);
    const double dum = (1.0 + 1.0529*tmp0 + 0.3334*rs);
    const double tmp1 = 0.1423/dum;
    Ec = -tmp1;
    dEc = tmp1/dum*(0.52645/tmp0 + 0.3334);
  }
  else{
    tmp0 = std::log(rs);
    Ec = -0.0480 + 0.0311*tmp0 + rs*(0.0020*tmp0 - 0.0116);
    dEc = 0.0311/rs + 0.0020*tmp0 - 0.0096;
  }
  switch (quantity) {
  case CeperleyAlderQuantity::energy_density:         return Ex + Ec;
  case CeperleyAlderQuantity::potential:              return Ex + Ec - 0.33333333333333333333*rs*(dEx + dEc);
  case CeperleyAlderQuantity::energy_minus_potential: return 0.3333333333333333333*rs*(dEx + dEc);
  case CeperleyAlderQuantity::energy_derivative:      return -0.3333333333333333333/(coe*coe*coe)*rs*rs*rs*rs*(dEx + dEc);
  }
  return 0.0;
}

LsdaResult ca_lsda(SpinDensity density, LsdaQuantity quantity, bool symmetrize_spins) noexcept
{
  LsdaResult r = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  const double den0 = density.up, den1 = density.down;
  const double min_den = 1.0e-15;
  const double tden = den0 + den1;
  if (tden<min_den) return r;

  double zeta = (den0 - den1)/tden;
  if (1.0<zeta)  zeta =  1.0 - min_den;
  if (zeta<-1.0) zeta = -1.0 + min_den;
  if (symmetrize_spins) zeta = 0.0;
  const double coe = 0.6203504908994;  /* pow(3.0/4.0/PI,1.0/3.0); */
  double rs;
  if (tden<min_den)
    rs = 62035.04908994;  /* coe*pow(1.0e-15,-1.0/3.0); */
  else
    rs = coe*std::pow(tden,-0.3333333333333333333);

  double tmp0 = 0.458165293632163/rs;
  const double ExP  = -tmp0;
  const double dExP = tmp0/rs;
  const double ExF  = 1.25992104989487*ExP;
  const double dExF = 1.25992104989487*dExP;
  double EcP, dEcP, EcF, dEcF;
  if (1.0<=rs){
    tmp0 = std::sqrt(rs);
    double dum = (1.0 + 1.0529*tmp0 + 0.3334*rs);
    double tmp1 = 0.1423/dum;
    EcP = -tmp1;
    dEcP = tmp1/dum*(0.52645/tmp0 + 0.3334);
    dum = (1.0 + 1.3981*tmp0 + 0.2611*rs);
    tmp1 = 0.0843/dum;
    EcF = -tmp1;
    dEcF = tmp1/dum*(0.69905/tmp0 + 0.2611);
  }
  else{
    tmp0 = std::log(rs);
    EcP = -0.0480 + 0.0311*tmp0 + rs*(0.0020*tmp0 - 0.0116);
    dEcP = 0.0311/rs + 0.0020*tmp0 - 0.0096;
    EcF = -0.0269 + 0.01555*tmp0 + rs*(0.0007*tmp0 - 0.0048);
    dEcF = 0.01555/rs + 0.0007*tmp0 - 0.0041;
  }

  double X0, X1, C0, C1;
  if (quantity == LsdaQuantity::energy_density){
    const double z0 = std::pow(1.0+zeta,1.33333333333333333);
    const double z1 = std::pow(1.0-zeta,1.33333333333333333);
    const double fzeta = 1.92366105093154*(z0 + z1 - 2.0);
    X0 = ExP + (ExF - ExP)*fzeta;
    X1 = X0;
    C0 = EcP + (EcF - EcP)*fzeta;
    C1 = C0;
  }
  else if (quantity == LsdaQuantity::potential){
    const double z0 = std::pow(1.0+zeta,0.33333333333333333);
    const double z1 = std::pow(1.0-zeta,0.33333333333333333);
    const double z02 = z0*z0, z04 = z02*z02, z12 = z1*z1, z14 = z12*z12;
    const double fzeta  = 1.92366105093154*(z04 + z14 - 2.0);
    const double dfzeta = 2.56488140124205*(z0 - z1);
    tmp0 = ExF - ExP;
    double Exc = ExP + tmp0*fzeta;
    double dExc = dExP + (dExF - dExP)*fzeta;
    double Vxc = Exc - 0.33333333333333333333*rs*dExc;
    X0 = Vxc + tmp0*( 1.0 - zeta)*dfzeta;
    X1 = Vxc + tmp0*(-1.0 - zeta)*dfzeta;
    tmp0 = EcF - EcP;
    Exc = EcP + tmp0*fzeta;
    dExc = dEcP + (dEcF - dEcP)*fzeta;
    Vxc = Exc - 0.33333333333333333333*rs*dExc;
    C0 = Vxc + tmp0*( 1.0 - zeta)*dfzeta;
    C1 = Vxc + tmp0*(-1.0 - zeta)*dfzeta;
  }
  else if (quantity == LsdaQuantity::energy_minus_potential){
    const double z0 = std::pow(1.0+zeta,0.33333333333333333);
    const double z1 = std::pow(1.0-zeta,0.33333333333333333);
    const double z02 = z0*z0, z04 = z02*z02, z12 = z1*z1, z14 = z12*z12;
    const double fzeta  = 1.92366105093154*(z04 + z14 - 2.0);
    const double dfzeta = 2.56488140124205*(z0 - z1);
    tmp0 = ExF - ExP;
    double Exc = ExP + tmp0*fzeta;
    double dExc = dExP + (dExF - dExP)*fzeta;
    double tmp1 = 0.33333333333333333333*rs*dExc;
    double tmp2 = tmp0*dfzeta;
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
  else return r;
  r.xc[0] = X0 + C0;
  r.xc[1] = X1 + C1;
  r.exchange[0] = X0;
  r.exchange[1] = X1;
  r.correlation[0] = C0;
  r.correlation[1] = C1;
  return r;
}

Pw92Result pw92_correlation(SpinDensity density, bool symmetrize_spins) noexcept
{
  const double den_min = 1.0e-14, den_min_half = 0.5*1.0e-14;
  double dens[2] = {density.up, density.down};
  /****************************************************
               parameters from Table I of
            Perdew and Wang, PRB, 45, 13244 (92)
  ****************************************************/
  const double A[3]      = {0.0310910,  0.0155450,  0.0168870};
  const double alpha1[3] = {0.2137000,  0.2054800,  0.1112500};
  const double beta1[3]  = {7.5957000, 14.1189000, 10.3570000};
  const double beta2[3]  = {3.5876000,  6.1977000,  3.6231000};
  const double beta3[3]  = {1.6382000,  3.3662000,  0.8802600};
  const double beta4[3]  = {0.4929400,  0.6251700,  0.4967100};

  const double coe = 0.6203504908994;  /* pow(3.0/4.0/PI,1.0/3.0);   */
  double dtot = dens[0] + dens[1];
  double rs;
  if (dtot<=den_min){
    rs = 6203.504908994;  /* coe*pow(1.0e-12,-1.0/3.0); */
    dens[0] = den_min_half;
    dens[1] = den_min_half;
    dtot = den_min;
  }
  else
    rs = coe*std::pow(dtot,-0.33333333333333333333333);
  const double tmp0 = 1.0/dtot;
  double zeta = tmp0*(dens[0] - dens[1]);
  if (0.99<zeta) zeta =  0.99;
  if (zeta<-1.0) zeta = -0.99;
  if (symmetrize_spins) zeta = 0.0;
  const double drsdd = -0.3333333333333333333333*rs*tmp0;
  const double dzdd[2] = {tmp0*( 1.0 - zeta), tmp0*(-1.0 - zeta)};

  const double srs = std::sqrt(rs);
  double G[3], dGdrs[3];
  for (int i=0; i<=2; i++){
    const double b = beta1[i]*srs + rs*(beta2[i] + beta3[i]*srs + beta4[i]*rs);
    const double dbdrs =  beta1[i]*0.50/srs
                        + beta2[i]
                        + beta3[i]*1.50*srs
                        + beta4[i]*2.0*rs;
    const double c = 1.0 + 1.0/(2.0*A[i]*b);
    const double dcdrs = -(c - 1.0)*dbdrs/b;
    const double dum = std::log(c);
    const double dum1 = 1.0 + alpha1[i]*rs;
    G[i] = -2.0*A[i]*dum1*dum;
    dGdrs[i] = -2.0*A[i]*(alpha1[i]*dum + dum1*dcdrs/c);
  }
  const double c = 1.92366105093154;       /* 1/(2*(2^{1/3}-1)) */
  const double fpp0 = 1.70992093416137;
  double dum1 = 1.0 + zeta;
  const double dum2 = 1.0 - zeta;
  const double tmp1  = std::pow(dum1,0.333333333333333333);
  const double tmp2  = std::pow(dum2,0.333333333333333333);
  const double tmp12 = tmp1*tmp1;
  const double tmp22 = tmp2*tmp2;
  const double tmp14 = tmp12*tmp12;
  const double tmp24 = tmp22*tmp22;
  const double f = (tmp14 + tmp24 - 2.0)*c;
  const double dfdz = 1.333333333333333333*(tmp1 - tmp2)*c;
  dum1 = zeta*zeta*zeta;
  double dum  = dum1*zeta;
  Pw92Result r;
  r.energy_density = G[0] - G[2]*f/fpp0*(1.0 - dum) + (G[1] - G[0])*f*dum;
  const double dEcdrs =   dGdrs[0] - dGdrs[2]*f/fpp0*(1.0 - dum)
                        + (dGdrs[1] - dGdrs[0])*f*dum;
  const double dEcdz = - G[2]/fpp0*(dfdz*(1.0 - dum) - f*4.0*dum1)
                       + (G[1] - G[0])*(dfdz*dum + f*4.0*dum1);
  dum = dEcdrs*drsdd;
  const double dEcdd[2] = {dum + dEcdz*dzdd[0], dum + dEcdz*dzdd[1]};
  r.potential[0] = r.energy_density + dtot*dEcdd[0];
  r.potential[1] = r.energy_density + dtot*dEcdd[1];
  r.density_used = {dens[0], dens[1]};
  return r;
}

ExchangeResult lda_exchange(double density) noexcept
{
  /* XC_EX with NSP = 1, the only form the original callers used (its NSP = 2
     branch read uninitialized spin densities and is not provided). */
  const double TRD = 0.333333333333333;
  const double den_min = 1.0e-14;
  const double COE1 = 0.6203504908994;   /* pow(3.0/4.0/PI,1.0/3.0); */
  const double COE2 = 0.610887057710857; /* 3.0*ALP/(2.0*PI*A0)      */
  double RS;
  if (density<=den_min){
    RS = COE1*std::pow(den_min,-1.0/3.0);
  }
  else{
    RS = COE1*std::pow(density,-TRD);
  }
  const double VXP = -COE2/RS;
  const double EXP = 0.750*VXP;
  return {EXP, VXP};
}

PbeResult pbe(SpinDensity density, const SpinGradient& gradient, bool symmetrize_spins) noexcept
{
  const double HALF = 0.50, THD = (1.0/3.0), TWO = 2.0, TWOTHD = (2.0/3.0);
  const double beta = 0.06672455060314922, kappa = 0.8040;
  const double gamma = (1.0 - std::log(TWO))/(PI*PI);
  const double mu = beta*PI*PI/3.0;
  const double den_min = 1.0e-14, gd_min = 1.0e-14;
  const double (&GDENS)[3][2] = gradient.component;
  int IS, IX;

  double dens[2] = {density.up, density.down};
  dens[0] = largest(0.5*den_min,dens[0]);
  dens[1] = largest(0.5*den_min,dens[1]);
  double D[2];
  if (symmetrize_spins){
    D[0] = (dens[0]+dens[1])/2.0;
    D[1] = (dens[0]+dens[1])/2.0;
  }
  else{
    D[0] = dens[0];
    D[1] = dens[1];
  }
  const double dt = dens[0] + dens[1];
  double GD[3][2], GDT[3];
  for (IX=0; IX<=2; IX++){
    if (symmetrize_spins){
      GD[IX][0] = (GDENS[IX][0]+GDENS[IX][1])/2.0;
      GD[IX][1] = (GDENS[IX][0]+GDENS[IX][1])/2.0;
    }
    else{
      GD[IX][0] = GDENS[IX][0];
      GD[IX][1] = GDENS[IX][1];
    }
    GDT[IX] = GDENS[IX][0] + GDENS[IX][1];
  }
  double GDM[2];
  GDM[0] = std::sqrt(GD[0][0]*GD[0][0] + GD[1][0]*GD[1][0] + GD[2][0]*GD[2][0]);
  GDM[1] = std::sqrt(GD[0][1]*GD[0][1] + GD[1][1]*GD[1][1] + GD[2][1]*GD[2][1]);
  double GDMT = std::sqrt(GDT[0]*GDT[0] + GDT[1]*GDT[1] + GDT[2]*GDT[2]);
  GDMT = largest(gd_min, GDMT);

  /* local correlation energy and potential (the original passed its floored
     density array, which the PW92 routine could rewrite) */
  const Pw92Result pw = pw92_correlation({dens[0], dens[1]}, symmetrize_spins);
  dens[0] = pw.density_used.up;
  dens[1] = pw.density_used.down;
  const double Ec_unif = pw.energy_density;
  const double Vc_unif[2] = {pw.potential[0], pw.potential[1]};

  /* total correlation energy */
  const double rs = std::pow(3.0/(4.0*PI*dt),THD);
  const double kF = std::pow(3.0*PI*PI*dt,THD);
  const double ks = std::sqrt(4.0*kF/PI);
  double zeta = (dens[0] - dens[1])/dt;
  if (0.99<zeta) zeta =  0.99;
  if (zeta<-1.0) zeta = -0.99;
  if (symmetrize_spins) zeta = 0.0;
  const double phi = 0.50*(std::pow(1.0 + zeta,TWOTHD)
                         + std::pow(1.0 - zeta,TWOTHD));
  const double t = GDMT/(2.0*phi*ks*dt);
  const double f1c = Ec_unif/(gamma*phi*phi*phi);
  const double f2 = std::exp(-f1c);
  const double A = beta/gamma/(f2 - 1.0);
  const double f3 = t*t + A*t*t*t*t;
  const double f4 = beta/gamma * f3/(1.0 + A*f3);
  const double H = gamma*phi*phi*phi*std::log(1.0 + f4);
  const double Fc = Ec_unif + H;

  /* correlation energy derivatives */
  const double DRSDD = -(THD*rs/dt);
  const double DKFDD = THD*kF/dt;
  const double DKSDD = HALF*ks*DKFDD/kF;
  const double DZDD[2] = {1.0/dt - zeta/dt, -(1.0/dt) - zeta/dt};
  const double DPDZ = HALF*TWOTHD*(1.0/std::pow(1.0+zeta,THD) - 1.0/std::pow(1.0-zeta,THD));
  double DFCDD[2], DFCDGD[3][2];
  for (IS=0; IS<=1; IS++){
    const double DECUDD = (Vc_unif[IS] - Ec_unif)/dt;
    const double DPDD = DPDZ*DZDD[IS];
    const double DTDD = (-t)*(DPDD/phi + DKSDD/ks + 1.0/dt);
    const double DF1DD = f1c*(DECUDD/Ec_unif - 3.0*DPDD/phi);
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
  double Fx = 0.0;
  double DFXDD[2], DFXDGD[3][2];
  for (IS=0; IS<=1; IS++){
    const double DS = largest(den_min,2.0*D[IS]);
    const double GDMS = largest(gd_min, 2.0*GDM[IS]);
    const double KFS = std::pow(3.0*PI*PI*DS,THD);
    const double s = GDMS/(2.0*KFS*DS);
    const double f1 = 1.0 + mu*s*s/kappa;
    const double f = 1.0 + kappa - kappa/f1;
    const ExchangeResult ex = lda_exchange(DS);   /* nspin = 1 in the original call */
    const double Ex_unif = ex.energy_density, Vx_unif = ex.potential;
    Fx = Fx + DS*Ex_unif*f;
    const double DKFDDx = THD * KFS/DS;
    const double DSDD = s*(-(DKFDDx/KFS) - 1.0/DS);
    const double DF1DD = 2.0*(f1 - 1.0)*DSDD/s;
    const double DFDD = kappa*DF1DD/(f1*f1);
    DFXDD[IS] = Vx_unif*f + DS*Ex_unif*DFDD;
    for (IX=0; IX<=2; IX++){
      const double GDS = 2.0*GD[IX][IS];
      const double DSDGD = (s/GDMS)*GDS/GDMS;
      const double DF1DGD = 2.0*mu*s*DSDGD/kappa;
      const double DFDGD = kappa*DF1DGD/(f1*f1);
      DFXDGD[IX][IS] = DS*Ex_unif*DFDGD;
    }
  }
  Fx = HALF*Fx/dt;

  PbeResult r;
  const double coe = 1.0;
  r.exchange_energy_density = coe*Fx;
  r.correlation_energy_density = Fc;
  for (IS=0; IS<=1; IS++){
    r.dexchange_ddensity[IS] = coe*DFXDD[IS];
    r.dcorrelation_ddensity[IS] = DFCDD[IS];
    for (IX=0; IX<=2; IX++){
      r.dexchange_dgradient[IX][IS] = coe*DFXDGD[IX][IS];
      r.dcorrelation_dgradient[IX][IS] = DFCDGD[IX][IS];
    }
  }
  r.density_used = {dens[0], dens[1]};
  return r;
}

}  // namespace openmx::xc
