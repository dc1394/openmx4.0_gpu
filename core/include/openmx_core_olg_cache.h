#ifndef OPENMX_CORE_OLG_CACHE_H
#define OPENMX_CORE_OLG_CACHE_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Identity of one overlap-grid sizing/final-pass pair. Native atom maps and
   neighbour counts are one-based; the cache API's atom/pair indices are
   zero-based. Inputs are borrowed only for the call and copied by prepare. */
typedef struct {
  int md_iter, copied_cells;
  size_t local_atoms, global_atoms;
  const int *local_to_global, *neighbor_counts;
} OpenMXOlgCacheInput;

/* take_atom transfers both pointer tables AND their pair buffers to the
   native consumer, which must free each pair and then each pointer table
   with free(). counts remains borrowed until prepare/release/finish/destroy.
   Pair buffers are not copied during the transfer. */
typedef struct {
  const int *counts;
  int **center, **neighbor;
  size_t pairs;
} OpenMXOlgCacheAtom;

/* Host-only, calculation-owned storage; no default context or GPU runtime.
   prepare replaces an old cache only on success. store copies a single pair
   transactionally; disjoint pairs may be stored concurrently after prepare.
   matches is true only for a complete cache with no atom taken yet. These
   operations and take_atom leave outputs/previous storage intact on failure.
   prepare/matches/take/release require callers to join all store workers. */
int openmx_core_olg_cache_prepare(OpenMXCoreContext*,const OpenMXOlgCacheInput*,char*,size_t);
int openmx_core_olg_cache_matches(OpenMXCoreContext*,const OpenMXOlgCacheInput*,int*,char*,size_t);
int openmx_core_olg_cache_store(OpenMXCoreContext*,size_t atom,size_t pair,size_t count,
    const int *center,const int *neighbor,char*,size_t);
int openmx_core_olg_cache_take_atom(OpenMXCoreContext*,size_t atom,OpenMXOlgCacheAtom*,char*,size_t);
void openmx_core_olg_cache_release(OpenMXCoreContext*);

#ifdef __cplusplus
}
#endif
#endif
