#include <openmx/basis/radial.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Always active, including Release/NDEBUG. */
#define CHECK(expression) do { if (!(expression)) { \
  fprintf(stderr,"line %d: %s\n",__LINE__,#expression); return 1; } } while (0)

int main(void)
{
  double radii[8] = {.15,.3,.6,1.1,1.7,2.6,3.8,5.2};
  /* Two rows with padding; padding is deliberately non-finite and ignored. */
  double values[18] = {.7,1.4,-.2,.9,-.4,.3,.1,0,NAN,NAN,
                      1.4,2.8,-.4,1.8,-.8,.6,.2,0};
  int angular[2] = {0,2};
  char error[128] = "old error";
  double output = -99;
  CHECK(openmx_basis_radial_sample(OPENMX_BASIS_RADIAL_PAO,0,8,radii,values,
      0,.85,&output,error,sizeof error));
  CHECK(fabs(output-.17978219696969672) < 1e-13 && error[0] == '\0');
  CHECK(openmx_basis_reciprocal_sample(OPENMX_BASIS_RECIPROCAL_PAO,8,2.4/8,
      values,0,&output,error,sizeof error));
  CHECK(fabs(output+4.3) < 1e-13);
  output = 123;
  CHECK(!openmx_basis_radial_sample(OPENMX_BASIS_RADIAL_PAO,0,8,NULL,values,
      0,.85,&output,error,sizeof error));
  CHECK(output == 123 && error[0] != '\0');
  CHECK(!openmx_basis_radial_sample(OPENMX_BASIS_RADIAL_PAO,0,SIZE_MAX,radii,values,
      0,.85,&output,error,sizeof error));
  CHECK(!openmx_basis_reciprocal_sample(OPENMX_BASIS_RECIPROCAL_PAO,0,.3,
      values,0,&output,error,sizeof error));
  CHECK(output == 123);
  char short_error[3] = {'a','b','c'};
  CHECK(!openmx_basis_radial_sample(OPENMX_BASIS_RADIAL_PAO,0,8,radii,values,
      0,-1,&output,short_error,2));
  CHECK(short_error[0] != '\0' && short_error[1] == '\0' && short_error[2] == 'c');
  CHECK(!openmx_basis_reciprocal_sample(OPENMX_BASIS_RECIPROCAL_PAO,8,.3,
      NULL,0,&output,short_error,0));
  CHECK(short_error[0] != '\0');

  OpenMXBasisRadialTable *table = openmx_basis_radial_table_create(OPENMX_BASIS_RADIAL_PAO,
      8,radii,2,angular,10,values,0,error,sizeof error);
  CHECK(table && error[0] == '\0');
  /* Both rows, metadata and mesh are now independently owned. */
  for (int i = 0; i < 8; ++i) radii[i] = -1;
  for (int i = 0; i < 18; ++i) values[i] = -1;
  angular[0] = 27;
  CHECK(openmx_basis_radial_table_sample(table,0,0,&output,error,sizeof error));
  CHECK(output == .7);
  CHECK(openmx_basis_radial_table_sample(table,1,.85,&output,error,sizeof error));
  CHECK(fabs(output-2*.17978219696969672) < 1e-13);
  output = 123;
  CHECK(!openmx_basis_radial_table_sample(table,2,.85,&output,error,sizeof error));
  CHECK(!openmx_basis_radial_table_sample(NULL,0,0,&output,NULL,0));
  CHECK(!openmx_basis_radial_table_sample(table,0,0,NULL,error,sizeof error));
  CHECK(output == 123);
  CHECK(!openmx_basis_radial_table_create(OPENMX_BASIS_RADIAL_PAO,8,radii,2,
      angular,10,values,0,error,sizeof error));
  const double valid[3] = {0,1,2};
  CHECK(!openmx_basis_radial_table_create(OPENMX_BASIS_RADIAL_PAO,3,valid,2,
      angular,SIZE_MAX,values,0,error,sizeof error));
  CHECK(!openmx_basis_radial_table_create(OPENMX_BASIS_RADIAL_PAO,3,valid,2,
      angular,2,values,0,error,sizeof error));
  OpenMXBasisRadialTable *empty = openmx_basis_radial_table_create(OPENMX_BASIS_RADIAL_PAO,
      3,valid,0,NULL,3,NULL,0,error,sizeof error);
  CHECK(empty);
  CHECK(!openmx_basis_radial_table_sample(empty,0,0,&output,error,sizeof error));
  openmx_basis_radial_table_destroy(empty);
  openmx_basis_radial_table_destroy(table);
  openmx_basis_radial_table_destroy(NULL);
  puts("PASS C caller: strided input, lifetime, invalid input and error isolation");
  return 0;
}
