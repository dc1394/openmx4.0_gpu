/**********************************************************************
  openmx_host_memory.h

  Host memory of this node as a batch job sees it: /proc/meminfo bounded
  by the memory limits of this process's cgroup and its parents (cgroup
  v1 or v2; a batch system's limit can sit on a parent).  Used by the
  capacity decision of the cluster solvers (openmx_cluster_capacity.c)
  and by the gpusolver2 bridge (elpa_cosma_bridge.c).  No MPI.
***********************************************************************/
#ifndef OPENMX_HOST_MEMORY_H
#define OPENMX_HOST_MEMORY_H

#ifdef __cplusplus
extern "C" {
#endif

/* Bytes this job may still allocate on this node: MemAvailable, bounded
   by the room left under the cgroup limits (at least 1 when known, even
   with no room left); 0 when unknown. */
unsigned long long openmx_host_mem_available(void);

/* Bytes of this node the job may use in all: MemTotal bounded by the
   cgroup limits; 0 when unknown. */
unsigned long long openmx_host_mem_total(void);

#ifdef __cplusplus
}
#endif

#endif /* OPENMX_HOST_MEMORY_H */
