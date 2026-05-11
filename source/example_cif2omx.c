#include <stdio.h>
#include "cif2omx.h"

int main(void)
{
  FILE *fp_cif = fopen("Si_mp149.cif", "r");
  FILE *fp_out = fopen("Si.dat", "w");
  CIF2OMX_Options opt;

  if (fp_cif == NULL){
    fprintf(stderr, "Failed to open Si_mp149.cif.\n");
    return 1;
  }
  if (fp_out == NULL){
    fprintf(stderr, "Failed to open Si.dat.\n");
    fclose(fp_cif);
    return 1;
  }

  cif2omx_options_init(&opt);
  opt.system_name = "Si";

  // choose COORD_XYZ or COORD_FRAC
  opt.coord_mode = COORD_XYZ;      

  // choose BASIS_QUICK, BASIS_STANDARD, or BASIS_PRECISE
  opt.basis_mode = BASIS_STANDARD; 

  /* If opt.has_kspacing=1, you can specify opt.kspacing.
     If opt.has_kspacing=0, the default value of 0.33 (1/Ang.) 
     is set to opt.kspacing.
  */
  opt.has_kspacing = 1;
  opt.kspacing = 0.20;

  /* If opt.has_energycutoff=1, you can specify opt.energycutoff.
     If opt.has_energycutoff=0, the default value of 220 Ryd.
     is set to opt.energycutoff.
  */
  opt.has_energycutoff = 1;
  opt.energycutoff = 300.0;

  if (!cif_fp_to_openmx_opt(fp_cif, fp_out, NULL, &opt)){
    fprintf(stderr, "cif2omx failed.\n");
    fclose(fp_cif);
    fclose(fp_out);
    return 1;
  }

  fclose(fp_cif);
  fclose(fp_out);
  return 0;
}
