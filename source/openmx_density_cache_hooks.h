#ifndef OPENMX_DENSITY_CACHE_HOOKS_H
#define OPENMX_DENSITY_CACHE_HOOKS_H
/* Borrowed for one density-grid invocation. Memory admission can return
   retained solver eigenvectors to the host without a current/global solver. */
typedef struct {
  void *context;
  void (*demote_eigenvectors)(void *context);
} OpenMXDensityCacheHooks;
#endif
