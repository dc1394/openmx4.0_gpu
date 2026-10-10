/**********************************************************************
  grid_precision.c  (see grid_precision.h)
***********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "grid_precision.h"

static int    configured = 0;
static int    enabled = 0;        /* FP32 stages at all */
static int    df_mode = 0;        /* the double-float kernels for the whole SCF */
static double until = 1.0e-5;     /* FP32 while NormRD > until */
static int    restart = 1;        /* restart the mixing history at the switch */
static int    fp32_now = 0;       /* this step */
static int    switched = 0;       /* the cycle went to FP64 for good */
static int    pending_reset = 0;
static int    announced = 0;

static void configure(void)
{
    const char *value;

    if (configured) return;
    configured = 1;
    /* the lossy FP32 stage is an experiment (it stalls the SCF near its
       own error level, 1e-5 of NormRD on sidia333): off unless asked */
    value = getenv("OPENMX_GRID_FP32");
    enabled = (value != NULL && value[0] != '\0') ? (atoi(value) != 0) : 0;
    value = getenv("OPENMX_GRID_PRECISION");
    if (value != NULL && value[0] != '\0') {
        df_mode = (strcmp(value, "df") == 0 || strcmp(value, "DF") == 0 || strcmp(value, "2") == 0);
    }
    else {
        int device = 0, perf_ratio = 0;

        df_mode = (cudaGetDevice(&device) == cudaSuccess &&
                   cudaDeviceGetAttribute(&perf_ratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, device) ==
                       cudaSuccess &&
                   8 <= perf_ratio);
    }
    value = getenv("OPENMX_GRID_FP32_UNTIL");
    if (value != NULL && value[0] != '\0') {
        double parsed = atof(value);
        if (isfinite(parsed)) until = fmax(0.0, parsed);
    }
    value = getenv("OPENMX_GRID_FP32_RESTART");
    if (value != NULL && value[0] != '\0') restart = atoi(value) != 0;
}

int Grid_Precision_Fp32(void)
{
    return fp32_now;
}

int Grid_Precision_Kernel(void)
{
    configure();
    if (fp32_now) return 1;
    return df_mode ? 2 : 0;
}

void Grid_Precision_BeginStep(int SCF_iter, int SCF_max, double normrd_prev, int myid, int verbose)
{
    int next;

    configure();
    if (SCF_iter == 1) {
        static int df_announced = 0;

        switched = 0;
        pending_reset = 0;
        if (df_mode && !df_announced && myid == 0 && 0 < verbose) {
            printf("<DFT>  grid integrals: double-float kernels (FP32 arithmetic with error-free transformations; "
                   "OPENMX_GRID_PRECISION=fp64 restores the FP64 kernels)\n");
            fflush(stdout);
            df_announced = 1;
        }
    }
    /* Even an unconverged run publishes energies and forces at the limit.
       Its last allowed step must use the non-lossy kernels as well. */
    next = enabled && !switched && SCF_iter < SCF_max &&
           (SCF_iter == 1 || (isfinite(normrd_prev) && until < normrd_prev));
    if (next && !announced && myid == 0 && 0 < verbose) {
        printf("<DFT>  grid integrals: FP32 (compensated sums) while NormRD > %.1e (OPENMX_GRID_FP32, "
               "OPENMX_GRID_FP32_UNTIL)\n", until);
        fflush(stdout);
        announced = 1;
    }
    if (fp32_now && !next) {
        switched = 1;
        pending_reset = restart;
        if (myid == 0 && 0 < verbose) {
            printf("<DFT>  grid integrals: FP64 from SCF step %d (NormRD %.1e)%s\n", SCF_iter, normrd_prev,
                   restart ? ", mixing history restarted" : "");
            fflush(stdout);
        }
    }
    fp32_now = next;
}

int Grid_Precision_StopCheck(int po, int myid, int verbose)
{
    if (!po || !fp32_now) return po;
    switched = 1;
    pending_reset = restart;
    if (myid == 0 && 0 < verbose) {
        printf("<DFT>  grid integrals: the stop test passed on an FP32 step; FP64 from the next step, the SCF goes on\n");
        fflush(stdout);
    }
    return 0;
}

int Grid_Precision_TakeHistoryReset(void)
{
    int value = pending_reset;

    pending_reset = 0;
    return value;
}

void Grid_Precision_EndCycle(void)
{
    switched = 0;
    fp32_now = 0;
    pending_reset = 0;
}
