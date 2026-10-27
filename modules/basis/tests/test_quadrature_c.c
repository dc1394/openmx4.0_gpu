#include <openmx/basis/quadrature.h>
#include <math.h>

int main(void)
{
  double nodes[5]={-99,-99,-99,-99,-99},weights[5]={-99,-99,-99,-99,-99};
  char error[128]; double moment=0; int i;
  if(!openmx_basis_gauss_legendre(3,nodes,weights,error,sizeof error) || *error)return 1;
  for(i=0;i<3;++i)moment+=weights[i]*nodes[i]*nodes[i];
  if(fabs(moment-2./3)>1e-12)return 2;
  if(nodes[3]!=-99 || weights[3]!=-99 || nodes[4]!=-99 || weights[4]!=-99)return 3;
  if(!openmx_basis_gauss_legendre(1,nodes,weights,0,0))return 4;
  if(nodes[0]!=0 || weights[0]!=2)return 5;
  return 0;
}
