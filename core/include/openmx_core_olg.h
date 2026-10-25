#ifndef OPENMX_CORE_OLG_H
#define OPENMX_CORE_OLG_H
#include "openmx_core_grid.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* One neighbour grid point of the overlap-grid search of UCell_Box: its
   global grid index, its cell index in the neighbour's list, and the
   neighbour's cell shift (atv_ijk of the pair's cell). */
typedef struct { int grid, cell, l1, l2, l3; } OpenMXOlgQuery;

/* The centre atom's grid list (grid ascending, repeated grid indices with
   different cells allowed) and the periodic tables of UCell_Box:
   shifts[3*c+a] = atv_ijk[c][a+1] for c < cells, and
   periodic[((l1+cpy)*rdim + (l2+cpy))*rdim + (l3+cpy)] with rdim = 2*cpy+1.
   Everything is borrowed for the duration of a call. */
typedef struct {
  const int *grid, *cell;
  int nc;
  const int *shifts;
  int cells;
  const int *periodic;
  int cpy;
} OpenMXOlgLists;

/* out[i] = the index into the centre list whose grid index equals query i's
   and whose cell is the query's periodic image, or -1: exactly what the
   device kernel of UCell_Box produces (leftmost match, any query order; sorted
   queries run as a vector merge). Returns 0 for an unsupported mode or a
   missing buffer, with nothing written, else 1. Pure, re-entrant and
   allocation-free: callers split the queries across threads. */
int openmx_core_olg_intersect(const OpenMXOlgLists*,const OpenMXOlgQuery*,size_t count,int *out,int mode);

/* The same answers for a neighbour list given as UCell_Box holds it: its grid
   and cell arrays plus the one cell shift of the pair, without building query
   records. */
typedef struct { const int *grid, *cell; int l1, l2, l3; } OpenMXOlgNeighbours;
int openmx_core_olg_intersect_lists(const OpenMXOlgLists*,const OpenMXOlgNeighbours*,size_t count,int *out,int mode);

/* The same answers through a map from the global grid index to the leftmost
   centre entry with that index (-1 elsewhere): built once per centre atom and
   reused for every neighbour, so a query costs a few gathers instead of a
   merge, and the queries need not be sorted. map holds extent ints; the
   caller fills it with -1 once (map_fill writes the centre's entries, map_reset
   restores -1 at them; both return 0 when a centre grid index is outside
   [0, extent), with nothing written). */
int openmx_core_olg_map_fill(const OpenMXOlgLists*,int *map,size_t extent,int mode);
int openmx_core_olg_map_reset(const OpenMXOlgLists*,int *map,size_t extent,int mode);
int openmx_core_olg_intersect_map(const OpenMXOlgLists*,const OpenMXOlgNeighbours*,const int *map,
    size_t extent,size_t count,int *out,int mode);

#ifdef __cplusplus
}
#endif
#endif
