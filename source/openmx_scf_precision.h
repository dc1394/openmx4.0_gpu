#ifndef OPENMX_SCF_PRECISION_H
#define OPENMX_SCF_PRECISION_H

#include <mpi.h>
#include "openmx_precision.h"

typedef struct {
    int step;
    int usage;
    int previous_usage;
} OpenMXScfPrecision;

static inline int OpenMX_ScfPrecisionUsage(const OpenMXScfPrecision *precision, MPI_Comm comm)
{
    int usage = openmx_fp32_usage() | precision->previous_usage;
    MPI_Allreduce(MPI_IN_PLACE, &usage, 1, MPI_INT, MPI_BOR, comm);
    return usage;
}

/* Reserve the final allowed step even when the mixed run has not converged.
   Return the usage flags once, so any stored mixed orbitals can be rebuilt. */
static inline int OpenMX_ScfPrecisionBeginStep(OpenMXScfPrecision *precision, int step, int limit, MPI_Comm comm)
{
    if (!openmx_fp32_enabled()) return 0;
    precision->previous_usage = openmx_fp32_begin_step();
    if (step < limit && precision->step != step) return 0;
    if (precision->step != step) {
        precision->step = step;
        precision->usage = OpenMX_ScfPrecisionUsage(precision, comm);
    }
    openmx_fp32_enter_reference();
    return precision->usage;
}

/* All ranks must decide to stop together. Some of them may have used a
   different kernel or fallen back to the host. Once requested, the reference
   phase cannot be cancelled by a new stop request on this mixed step. */
static inline int OpenMX_ScfPrecisionStop(OpenMXScfPrecision *precision, int step, int limit,
                                           int converged, MPI_Comm comm)
{
    if (precision->step == step + 1) return 0;
    if (precision->step == step && precision->usage && step < limit) return 0;
    if ((converged || limit <= step) && openmx_fp32_enabled()) {
        const int usage = OpenMX_ScfPrecisionUsage(precision, comm);
        if (usage) {
            precision->step = step + 1;
            precision->usage = usage;
            return 0;
        }
    }
    return converged;
}

#endif
