#ifndef OPENMX_MD_GPU_H
#define OPENMX_MD_GPU_H

#include "../core/include/openmx_core_md.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct OpenMXMDGPU OpenMXMDGPU;
OpenMXMDGPU *OpenMX_MD_GPU_Create(void);
void OpenMX_MD_GPU_Destroy(OpenMXMDGPU *);
void *OpenMX_MD_GPU_Workspace(OpenMXMDGPU *, size_t bytes);
int OpenMX_MD_GPU_Move(OpenMXMDGPU *, OpenMXMDMoveTile *, size_t, const OpenMXMDStep *);
int OpenMX_MD_GPU_Periodic(OpenMXMDGPU *, OpenMXMDPeriodicTile *, size_t, const OpenMXMDCell *);
int OpenMX_MD_GPU_NPT(OpenMXMDGPU *, OpenMXMDNPTTile *, size_t, const OpenMXMDNPTStep *);
#ifdef __cplusplus
}
#endif

#endif
