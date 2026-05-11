#ifndef CIF2OMX_H
#define CIF2OMX_H

#include <stdio.h>

typedef struct {
  char symbol[8];
  double f[3];
  double r[3];
} AtomOut;

typedef struct {
  AtomOut *v;
  int n, cap;
  double tv[3][3];
  int has_tv;
} CIF_Result;

typedef enum {
  BASIS_QUICK = 0,
  BASIS_STANDARD = 1,
  BASIS_PRECISE = 2
} BasisMode;

typedef enum {
  COORD_XYZ = 0,
  COORD_FRAC = 1
} CoordMode;

typedef struct {
  const char *system_name;
  CoordMode coord_mode;
  BasisMode basis_mode;
  int has_kspacing;
  double kspacing;
  int has_energycutoff;
  double energycutoff;
} CIF2OMX_Options;

void cif2omx_options_init(CIF2OMX_Options *opt);

int cif_fp_to_openmx_opt(FILE *fp_cif, FILE *fp_out, CIF_Result *res_out,
                         const CIF2OMX_Options *opt);

void cif_result_free(CIF_Result *res);

#endif
