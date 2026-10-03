/*
 * Compiler check: nvc 24.5 and 24.7 (NVHPC) vectorize an access a[i][i+c]
 * of a pointer-to-pointer array as if the row pointer were loop invariant,
 * so a diagonal store fills part of one row instead.  nvc 22.2 and GCC are
 * correct; -Mnovect (placed after -O2/-O3) or -Mvect=nosimd avoids it.
 * Generate_pMatrix() in Krylov.c builds its starting vectors with the
 * second loop below and returns a wrong subspace when it is miscompiled.
 *
 *   nvc -O3 tests/nvc_diag_vectorizer_bug.c -o check && ./check
 *
 * Exit status 0: both loops compiled correctly; 1: miscompiled.
 */
#include <stdio.h>
#include <stdlib.h>

#define N 37

__attribute__((noinline))
static void set_diagonal(double **a, int n)
{
  int i;

  for (i = 0; i < n; i++) a[i][i] = 1.0;
}

__attribute__((noinline))
static int set_unit_vectors(double **vec, int offset, const int *count)
{
  int i, m = 0;

  for (i = 0; i < count[0]; i++) {
    vec[m][offset + i] = 1.0;
    m++;
  }
  return m;
}

static int wrong(double **a, int shift)
{
  int i, j, bad = 0;

  for (i = 0; i < 2 * N; i++) {
    for (j = 0; j < 2 * N; j++) {
      if (a[i][j] != ((i < N && j == i + shift) ? 1.0 : 0.0)) bad++;
      a[i][j] = 0.0;
    }
  }
  return bad;
}

int main(void)
{
  double **a = malloc(sizeof(double *) * 2 * N);
  int count = N, i, bad1, bad2;

  for (i = 0; i < 2 * N; i++) a[i] = calloc(2 * N, sizeof(double));

  set_diagonal(a, N);
  bad1 = wrong(a, 0);
  set_unit_vectors(a, 4, &count);
  bad2 = wrong(a, 4);

  printf("a[i][i] = 1.0: %s; vec[m][offset+i] = 1.0, m++: %s\n",
         bad1 ? "MISCOMPILED" : "ok", bad2 ? "MISCOMPILED" : "ok");
  return bad1 != 0 || bad2 != 0;
}
