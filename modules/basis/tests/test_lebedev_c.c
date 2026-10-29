#include <openmx/basis/lebedev.h>
#include <math.h>
#include <stdio.h>

int main(void)
{
  double storage[4*2030];char error[160];
  const int counts[3]={590,1202,2030};
  for(int c=0;c<3;++c) {
    if(!openmx_basis_lebedev_fill(counts[c],storage,sizeof storage/sizeof *storage,error,sizeof error)) {
      fprintf(stderr,"%s\n",error);return 1;
    }
    const OpenMXLebedevRuleData rule={(size_t)counts[c],storage};
    double sum=0,x2=0;
    for(size_t i=0;i<rule.count;++i) {
      const double *point=openmx_lebedev_point(&rule,i);
      sum+=point[3];x2+=point[3]*point[0]*point[0];
    }
    if(fabs(sum-1)>3e-12 || fabs(x2-1.0/3)>3e-12)return 2;
  }
  puts("PASS C Lebedev fill and borrowed rule");return 0;
}
