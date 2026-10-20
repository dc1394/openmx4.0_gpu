#ifndef OPENMX_PERIODIC_H
#define OPENMX_PERIODIC_H

#include <limits.h>
#include <stdint.h>

/* OpenMX enumerates [-radius,radius]^3 lexicographically, moving the
   origin to cell zero. No lookup table or simulation state is needed. */
static inline int openmx_periodic_count(int radius)
{
  int64_t width, count;
  if (radius < 0 || radius > 644) return -1;
  width = 2*(int64_t)radius + 1;
  count = width*width*width;
  return count <= INT_MAX ? (int)count : -1;
}

static inline int openmx_periodic_index(int radius, int i, int j, int k)
{
  int64_t width, ordinal, origin;
  if (openmx_periodic_count(radius) < 0 ||
      i < -radius || i > radius || j < -radius || j > radius ||
      k < -radius || k > radius) return -1;
  width = 2*(int64_t)radius + 1;
  ordinal = ((i + radius)*width + j + radius)*width + k + radius;
  origin = (width*width*width - 1)/2;
  return ordinal == origin ? 0 : (int)(ordinal + (ordinal < origin));
}

#endif
