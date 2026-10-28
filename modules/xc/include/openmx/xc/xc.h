#ifndef OPENMX_XC_XC_H
#define OPENMX_XC_XC_H

/* C interface of the OpenMX exchange-correlation functionals (see
   functionals.hpp for the physics and the conventions).  Array arguments use
   the layouts of the original OpenMX routines so that existing callers can
   switch without reshaping: dens[2] is (up, down), gdens[3][2] is
   [direction][spin].  symmetrize_spins is 0 or 1.  Every function is pure
   and may be called concurrently; NULL outputs are never written and a
   NULL required input returns 0 from the int-returning entries. */

#ifdef __cplusplus
extern "C" {
#endif

/* quantity: 0 energy density, 1 potential, 2 energy minus potential,
   3 derivative of the energy density; other values return 0.0 */
double openmx_xc_ceperley_alder(double density, int quantity);

/* quantity: 0 energy density, 1 potential, 2 energy minus potential.
   xc, x, c receive the per-spin values; returns 0 for an unknown quantity. */
int openmx_xc_ca_lsda(double den_up, double den_down, int quantity, int symmetrize_spins,
                      double xc[2], double x[2], double c[2]);

/* dens is read and, below the density floor, rewritten to the floor values
   as the original routine did.  ec receives the energy density, vc the two
   potentials.  Returns 0 without a dens or output array. */
int openmx_xc_pw92_correlation(double dens[2], int symmetrize_spins, double ec[1], double vc[2]);

/* ex receives the exchange energy density, vx the potential. */
int openmx_xc_lda_exchange(double density, double ex[1], double vx[1]);

/* dens is read and rewritten to the floored values as the original routine
   did.  exc[0] exchange, exc[1] correlation energy density; dexdd/decdd per
   spin; dexdgd/decdgd [direction][spin].  Returns 0 without an input or
   output array. */
int openmx_xc_pbe(double dens[2], double gdens[3][2], int symmetrize_spins,
                  double exc[2], double dexdd[2], double decdd[2],
                  double dexdgd[3][2], double decdgd[3][2]);

#ifdef __cplusplus
}
#endif

#endif
