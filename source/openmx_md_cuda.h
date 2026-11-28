#ifndef OPENMX_MD_CUDA_H
#define OPENMX_MD_CUDA_H

#include "../core/include/openmx_core_md.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct OpenMXMDCuda OpenMXMDCuda;
OpenMXMDCuda *OpenMX_MD_Cuda_Create(void);
void OpenMX_MD_Cuda_Destroy(OpenMXMDCuda *);
void *OpenMX_MD_Cuda_Workspace(OpenMXMDCuda *, size_t bytes);
int OpenMX_MD_Cuda_Move(OpenMXMDCuda *, OpenMXMDMoveTile *, size_t, const OpenMXMDStep *);
int OpenMX_MD_Cuda_Periodic(OpenMXMDCuda *, OpenMXMDPeriodicTile *, size_t, const OpenMXMDCell *);
int OpenMX_MD_Cuda_NPT(OpenMXMDCuda *, OpenMXMDNPTTile *, size_t, const OpenMXMDNPTStep *);
#ifdef __cplusplus
}
#endif

#endif
