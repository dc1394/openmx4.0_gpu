#include <openmx/basis/real_harmonics.h>
#include <math.h>
#include <stdio.h>
int main(void)
{
  dcomplex data[5*9*9];char error[128];
  if(!openmx_basis_real_harmonics_fill(4,data,sizeof data/sizeof *data,error,sizeof error)) {
    fprintf(stderr,"%s\n",error);return 1;
  }
  const OpenMXRealHarmonicData view={5,9,data};
  for(size_t l=0;l<view.orders;++l)for(size_t row=0;row<=2*l;++row) {
    double norm=0.;
    for(size_t col=0;col<=2*l;++col) {
      const dcomplex z=openmx_real_harmonic_coefficient(&view,l,row,col);norm+=z.r*z.r+z.i*z.i;
    }
    if(fabs(norm-1)>1e-14)return 2;
  }
  if(openmx_real_harmonic_coefficient(&view,1,2,1).r!=1)return 3;
  puts("PASS C real-harmonic coefficients");return 0;
}
