#ifndef OPENMX_BAND_KPATH_POINTS_H
#define OPENMX_BAND_KPATH_POINTS_H

/* Path endpoints and axes are one-based. A one-point segment samples its start. */
static inline double openmx_band_kpath_coordinate(double *const *path, int axis, int point, int count)
{
    const double start = path[1][axis];
    return count == 1 ? start : start + (path[2][axis] - start) * point / (count - 1);
}

#endif
