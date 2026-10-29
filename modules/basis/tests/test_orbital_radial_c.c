#include <openmx/basis/orbital_radial.h>
#include <math.h>
#include <stdio.h>
int main(void)
{
  const double radii[]={.1,.2,.4,.8,1.6,3.2};
  double values[6];OpenMXOrbitalRadialSample sample={71.,73.};char error[128];
  for(int i=0;i<6;++i)values[i]=2.+4.*radii[i]*radii[i];
  if(!openmx_orbital_radial_sample(0,radii,values,6,.05,&sample,error,sizeof(error)))return 1;
  if(fabs(sample.value-2.01)>1e-12 || fabs(sample.radial_derivative-.4)>1e-12)return 2;
  if(openmx_orbital_radial_sample(0,radii,values,4,.05,&sample,error,sizeof(error)))return 3;
  if(fabs(sample.value-2.01)>1e-12 || !*error)return 4;
  if(!openmx_orbital_radial_sample(0,radii,values,6,4.,&sample,0,0) || sample.value!=0. || sample.radial_derivative!=0.)return 5;
  puts("PASS real C orbital radial ABI");return 0;
}
