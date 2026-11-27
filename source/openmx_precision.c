#include "openmx_precision.h"
#include <stdatomic.h>

enum {
    FP32_STEP_USAGE = OPENMX_FP32_USED | OPENMX_FP32_LOSSY_GRID,
    FP32_USAGE = FP32_STEP_USAGE | OPENMX_FP32_ORBITAL_TILES | OPENMX_FP32_CACHED_ORBITALS,
    FP32_REQUESTED = 16,
    FP32_REFERENCE = 32
};

/* GPU workers can report mixed kernels concurrently. The SCF controller
   changes the phase only between calls, after the workers have joined. */
static atomic_int fp32_policy = FP32_REQUESTED;

int openmx_fp32_requested(void)
{
    return (atomic_load_explicit(&fp32_policy, memory_order_relaxed) & FP32_REQUESTED) != 0;
}

int openmx_fp32_enabled(void)
{
    const int policy = atomic_load_explicit(&fp32_policy, memory_order_relaxed);
    return (policy & (FP32_REQUESTED | FP32_REFERENCE)) == FP32_REQUESTED;
}

void openmx_set_fp32_enabled(int enabled)
{
    atomic_store_explicit(&fp32_policy, enabled ? FP32_REQUESTED : 0, memory_order_relaxed);
}

void openmx_fp32_begin_scf(void)
{
    openmx_set_fp32_enabled(openmx_fp32_requested());
}

int openmx_fp32_begin_step(void)
{
    const int policy = atomic_load_explicit(&fp32_policy, memory_order_relaxed);
    /* Contractions and solves are recomputed each step. Orbital tiles and
       rows may be cached and used again without launching their kernels. */
    if (policy & FP32_STEP_USAGE)
        atomic_store_explicit(&fp32_policy, policy & ~FP32_STEP_USAGE, memory_order_relaxed);
    return policy & FP32_USAGE;
}

void openmx_fp32_enter_reference(void)
{
    const int policy = atomic_load_explicit(&fp32_policy, memory_order_relaxed);
    atomic_store_explicit(&fp32_policy, policy | FP32_REFERENCE, memory_order_relaxed);
}

void openmx_fp32_note_use(int usage)
{
    const int policy = atomic_load_explicit(&fp32_policy, memory_order_relaxed);
    if ((policy & (FP32_REQUESTED | FP32_REFERENCE)) == FP32_REQUESTED && (policy & usage) != usage)
        atomic_fetch_or_explicit(&fp32_policy, usage, memory_order_relaxed);
}

int openmx_fp32_usage(void)
{
    return atomic_load_explicit(&fp32_policy, memory_order_relaxed) & FP32_USAGE;
}
