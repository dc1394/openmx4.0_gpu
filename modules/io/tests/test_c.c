#include <openmx/io/keyword_file.h>

#include <stdio.h>
#include <string.h>

/* Always active, including Release/NDEBUG. */
#define CHECK(expression) do { if (!(expression)) { \
  fprintf(stderr,"line %d: %s\n",__LINE__,#expression); return 1; } } while (0)

int main(void)
{
  const char *path = "openmx_io_c_test.dat";
  FILE *out = fopen(path, "w");
  OpenMXKeywordFiles *files;
  int i = 0, flag = 0, grid[3], kind = 0, three[3] = {1, 1, 1};
  double d = 0.0, range[2], zero2[2] = {0.0, 0.0};
  char text[64], first[16], second[16], *pair[2] = {first, second};
  const char *defaults[2] = {"x", "y"};
  const char *xc_names[3] = {"LDA", "LSDA-CA", "GGA-PBE"};
  const int xc_values[3] = {1, 2, 4};
  FILE *block;

  CHECK(out != NULL);
  fputs("System.Name  h2o\n"
        "scf.maxIter\t40\n"
        "scf.EnergyCutoff 150.5 # Ry\n"
        "scf.SpinPolarization  On\n"
        "scf.XcType  gga-pbe\n"
        "scf.Kgrid  4 4 1\n"
        "Dos.Erange -10.0 10.0\n"
        "Pair.Key  alpha beta\n"
        "bad.int  seven\n"
        "<Definition.of.Atomic.Species\n"
        " H  H5.0-s2p1  H_PBE19\n"
        "Definition.of.Atomic.Species>\n", out);
  fclose(out);

  files = openmx_io_keyword_files_create(14, NULL);
  CHECK(files != NULL);
  CHECK(openmx_io_depth(files) == 0);
  CHECK(openmx_io_int(files, "scf.maxIter", &i, 1) == -1);     /* nothing open */
  CHECK(openmx_io_open(files, "does-not-exist.dat") == 0);
  CHECK(openmx_io_open(files, path) == 1 && openmx_io_depth(files) == 1);

  CHECK(openmx_io_string(files, "system.name", text, sizeof text, "none") == 1 && strcmp(text, "h2o") == 0);
  CHECK(openmx_io_int(files, "SCF.MAXITER", &i, 1) == 1 && i == 40);
  CHECK(openmx_io_double(files, "scf.EnergyCutoff", &d, 1.0) == 1 && d == 150.5);
  CHECK(openmx_io_logical(files, "scf.SpinPolarization", &flag, 0) == 1 && flag == 1);
  CHECK(openmx_io_string2int(files, "scf.XcType", &kind, 3, xc_names, xc_values) == 1 && kind == 4);
  CHECK(openmx_io_intv(files, "scf.Kgrid", 3, grid, three) == 1 && grid[0] == 4 && grid[2] == 1);
  CHECK(openmx_io_doublev(files, "Dos.Erange", 2, range, zero2) == 1 && range[0] == -10.0 && range[1] == 10.0);
  CHECK(openmx_io_stringv(files, "Pair.Key", 2, pair, sizeof first, defaults) == 1);
  CHECK(strcmp(first, "alpha") == 0 && strcmp(second, "beta") == 0);
  CHECK(openmx_io_error_count(files) == 0);

  /* missing keeps the fallback; invalid records an error and keeps it too */
  CHECK(openmx_io_int(files, "scf.Mixing.Type", &i, 7) == 0 && i == 7);
  CHECK(openmx_io_int(files, "bad.int", &i, 7) == -1 && i == 7);
  CHECK(openmx_io_error_count(files) == 1);
  CHECK(openmx_io_string(files, "System.Name", text, 3, "none") == -1 && strcmp(text, "no") == 0);
  CHECK(openmx_io_error_count(files) == 1);   /* a too-small buffer is the caller's error, not the file's */

  block = openmx_io_find(files, "<Definition.of.Atomic.Species");
  CHECK(block != NULL);
  CHECK(fscanf(block, "%15s %15s %63s", first, second, text) == 3 && strcmp(second, "H5.0-s2p1") == 0);
  CHECK(openmx_io_last(files, "Definition.of.Atomic.Species>") == 1);
  CHECK(openmx_io_find(files, "<Missing.Block") == NULL);

  CHECK(openmx_io_cmpstring("Yes", &flag, 2, defaults, xc_values) == 0);
  CHECK(openmx_io_cmpstring("GGA-pbe", &kind, 3, xc_names, xc_values) == 1 && kind == 4);

  /* nesting: the second open resets the counter, closing adds the inner errors back */
  CHECK(openmx_io_open(files, path) == 1 && openmx_io_depth(files) == 2 && openmx_io_error_count(files) == 0);
  CHECK(openmx_io_int(files, "bad.int", &i, 0) == -1 && openmx_io_error_count(files) == 1);
  CHECK(openmx_io_close(files) == 0 && openmx_io_depth(files) == 1 && openmx_io_error_count(files) == 1);
  CHECK(openmx_io_close(files) == 0 && openmx_io_depth(files) == 0);
  CHECK(openmx_io_close(files) == EOF);
  CHECK(openmx_io_int(NULL, "scf.maxIter", &i, 1) == -1 && openmx_io_open(NULL, path) == 0);
  openmx_io_keyword_files_destroy(files);
  openmx_io_keyword_files_destroy(NULL);
  remove(path);
  printf("PASS io C interface\n");
  return 0;
}
