#ifndef OPENMX_VNA_STATUS_H
#define OPENMX_VNA_STATUS_H
#include "openmx_core_vna_matrix.h"
#include "openmx_core_reciprocal.h"
/* Shared application interface with calculation-scoped matrix views.
   The stage marker follows the backend's existing diagnostic policy and is
   returned to the caller instead of being kept in process-global storage.
   A null status pointer is accepted by the compatibility wrapper. */
#ifdef __cplusplus
extern "C" {
#endif
double Set_ProExpn_VNA_WithStatus(const OpenMXVnaMatrixViews *vna,const OpenMXReciprocalViews *reciprocal, int *used_gpu);
#ifdef __cplusplus
}
#endif
#endif
