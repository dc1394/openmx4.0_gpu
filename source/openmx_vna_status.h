#ifndef OPENMX_VNA_STATUS_H
#define OPENMX_VNA_STATUS_H
/* HIP application interface; include after the Type_DS_VNA definition.
   The stage marker follows the backend's existing diagnostic policy and is
   returned to the caller instead of being kept in process-global storage.
   A null status pointer is accepted by the compatibility wrapper. */
#ifdef __cplusplus
extern "C" {
#endif
double Set_ProExpn_VNA_WithStatus(double ****hvna, double *****hvna2,
    Type_DS_VNA *****ds_vna, int *used_gpu);
#ifdef __cplusplus
}
#endif
#endif
