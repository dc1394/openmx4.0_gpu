#include <openmx/basis/gaunt.h>
#include <math.h>
#include <stdio.h>
int main(void)
{
  char error[128];double value=17.;double table[72];OpenMXGauntTableData view;
  if(!openmx_basis_gaunt(0,0,0,0,0,0,&value,error,sizeof(error)) || *error ||
      fabs(value-1/sqrt(4*3.1415926535897932384626))>1e-14)return 1;
  if(!openmx_basis_gaunt_table_fill(1,1,1,table,72,&view,error,sizeof(error)) || *error || view.values!=table || view.row_stride!=3)return 2;
  if(fabs(table[openmx_gaunt_table_row_offset(&view,1,0,0,1)+1]-value)>1e-14)return 3;
  value=17.;if(openmx_basis_gaunt(39,0,0,0,0,0,&value,error,sizeof(error)) || value!=17. || !*error)return 4;
  puts("PASS real C Gaunt scalar/table API");return 0;
}
