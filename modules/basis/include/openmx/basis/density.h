#ifndef OPENMX_BASIS_DENSITY_H
#define OPENMX_BASIS_DENSITY_H

#ifdef __cplusplus
extern "C" {
#endif

/* Allocation-free legacy scalar boundary. Input rows are borrowed; each grid
   has count entries and density has count+2 entries (physical values at +1).
   count >= 6, finite values and a strictly increasing uniform logarithmic
   grid are caller preconditions. These hot functions do not rescan the rows.
   Invalid shape, null rows or NaN coordinates return NaN. log_radius=-inf
   evaluates the origin continuation. The derivative is with respect to r;
   below the first mesh point it uses exp(log_radius), as legacy Dr_KumoF did. */
double openmx_basis_density_value(int count, double log_radius,
    const double *log_grid, const double *radii, const double *padded_density);
double openmx_basis_density_derivative(int count, double log_radius, double radius,
    const double *log_grid, const double *radii, const double *padded_density);

#ifdef __cplusplus
}
#endif
#endif
