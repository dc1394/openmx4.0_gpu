/**********************************************************************
  openmx_stdout_level.h

  level.of.stdout (level_stdout of openmx_common.h) for the GPU sources
  that do not include openmx_common.h.  The GPU paths print by it:

  level.of.stdout 1 (default)  errors, a GPU that cannot be used, and a
                               GPU eigensolver given up for lack of device
                               memory or after a failure
  level.of.stdout 2            also the device plans and banners, the stage
                               switches of the adaptive precision, and the
                               fallbacks and retries of single stages and
                               solves
  level.of.stdout 3            also the per-step, per-k-point and per-rank
                               details (refinement residuals, warm starts)

  The reference is weak so that the standalone test builds of these
  sources link without openmx_common.h; they print as at level 3.
***********************************************************************/
#ifndef OPENMX_STDOUT_LEVEL_H
#define OPENMX_STDOUT_LEVEL_H

#ifdef __cplusplus
extern "C" {
#endif

extern int level_stdout __attribute__((weak));

#ifdef __cplusplus
}
#endif

#define OPENMX_STDOUT_LEVEL(level) (&level_stdout == 0 || (level) <= level_stdout)

#endif
