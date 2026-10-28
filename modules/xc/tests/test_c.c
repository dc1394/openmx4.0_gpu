#include <openmx/xc/xc.h>

#include <math.h>
#include <stdio.h>

/* Always active, including Release/NDEBUG. */
#define CHECK(expression) do { if (!(expression)) { \
  fprintf(stderr,"line %d: %s\n",__LINE__,#expression); return 1; } } while (0)

static int close_to(double a, double b, double relative)
{
  const double scale = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
  return fabs(a - b) <= relative * (scale > 1e-300 ? scale : 1.0);
}

int main(void)
{
  double dens[2] = {0.02, 0.005}, gdens[3][2] = {{0.01, 0.002}, {-0.003, 0.0}, {0.0, 0.004}};
  double exc[2], dexdd[2], decdd[2], dexdgd[3][2], decdgd[3][2];
  double xc[2], x[2], c[2], ec[1], vc[2], ex[1], vx[1];
  double lda[2] = {0.02, 0.005};
  double zero[3][2] = {{0, 0}, {0, 0}, {0, 0}};
  int d, s;

  /* Ceperley-Alder LDA: potential below the energy density at this density,
     the two differences consistent, zero below the floor, unknown quantities 0 */
  CHECK(openmx_xc_ceperley_alder(0.01, 0) < 0.0);
  CHECK(close_to(openmx_xc_ceperley_alder(0.01, 2),
                 openmx_xc_ceperley_alder(0.01, 0) - openmx_xc_ceperley_alder(0.01, 1), 1e-12));
  CHECK(openmx_xc_ceperley_alder(0.0, 1) == 0.0);
  CHECK(openmx_xc_ceperley_alder(0.01, 7) == 0.0);

  /* LSDA: unpolarized densities give the LDA, the spin-symmetrized call too */
  CHECK(openmx_xc_ca_lsda(0.005, 0.005, 0, 0, xc, x, c));
  CHECK(close_to(xc[0], openmx_xc_ceperley_alder(0.01, 0), 1e-13) && xc[0] == xc[1]);
  CHECK(close_to(xc[0], x[0] + c[0], 1e-15));
  CHECK(openmx_xc_ca_lsda(0.009, 0.001, 0, 1, xc, x, c));
  CHECK(close_to(xc[0], openmx_xc_ceperley_alder(0.01, 0), 1e-13));
  CHECK(!openmx_xc_ca_lsda(0.009, 0.001, 5, 0, xc, x, c));
  CHECK(openmx_xc_ca_lsda(0.009, 0.001, 1, 0, NULL, NULL, NULL));   /* nothing to write is fine */

  /* PW92 correlation and Slater exchange */
  CHECK(openmx_xc_pw92_correlation(lda, 0, ec, vc));
  CHECK(ec[0] < 0.0 && vc[0] < 0.0 && vc[1] < 0.0 && lda[0] == 0.02);
  CHECK(!openmx_xc_pw92_correlation(NULL, 0, ec, vc));
  CHECK(openmx_xc_lda_exchange(0.025, ex, vx));
  CHECK(close_to(ex[0], -0.75 * pow(3.0 / 3.1415926535897932384626, 1.0 / 3.0) * pow(0.025, 1.0 / 3.0), 1e-12));
  CHECK(close_to(vx[0], 4.0 / 3.0 * ex[0], 1e-13));
  CHECK(!openmx_xc_lda_exchange(0.025, NULL, vx));

  /* PBE: finite outputs, the input densities are kept above the floor, and
     without a gradient the pieces reduce to Slater exchange and PW92 */
  CHECK(openmx_xc_pbe(dens, gdens, 0, exc, dexdd, decdd, dexdgd, decdgd));
  CHECK(isfinite(exc[0]) && isfinite(exc[1]) && dens[0] == 0.02 && dens[1] == 0.005);
  for (d = 0; d < 3; d++) for (s = 0; s < 2; s++) CHECK(isfinite(dexdgd[d][s]) && isfinite(decdgd[d][s]));
  {
    double even[2] = {0.0125, 0.0125}, same[2] = {0.0125, 0.0125};
    CHECK(openmx_xc_pbe(even, zero, 0, exc, dexdd, decdd, dexdgd, decdgd));
    CHECK(openmx_xc_lda_exchange(0.025, ex, vx) && openmx_xc_pw92_correlation(same, 0, ec, vc));
    CHECK(close_to(exc[0], ex[0], 1e-10) && close_to(exc[1], ec[0], 1e-10));
    CHECK(close_to(dexdd[0], vx[0], 1e-10) && close_to(decdd[1], vc[1], 1e-10));
  }
  CHECK(!openmx_xc_pbe(dens, gdens, 0, NULL, dexdd, decdd, dexdgd, decdgd));
  printf("PASS xc C interface\n");
  return 0;
}
