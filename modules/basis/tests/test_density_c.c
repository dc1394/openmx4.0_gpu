#include <openmx/basis/density.h>
#include <math.h>
#include <stddef.h>

int main(void)
{
  double x[8],r[8],rho[10];
  int i;
  for(i=0;i<8;++i){x[i]=-3.0+.5*i;r[i]=exp(x[i]);}
  for(i=0;i<10;++i)rho[i]=1.;
  if(openmx_basis_density_value(8,-1.,x,r,rho)!=1.)return 1;
  if(openmx_basis_density_derivative(8,-1.,exp(-1.),x,r,rho)!=0.)return 2;
  if(!isnan(openmx_basis_density_value(5,0.,x,r,rho)))return 3;
  if(!isnan(openmx_basis_density_value(8,0.,NULL,r,rho)))return 4;
  return 0;
}
