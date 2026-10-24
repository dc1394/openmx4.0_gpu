#include "set_cuda_default_device_from_local_rank.h"
#include "set_openacc_device_from_local_rank.h"
#include "openmx_mgpu_runtime.h"
#include "openmx_env.h"
#include <limits.h>
#include <stdint.h>

int set_cuda_default_device_from_local_rank(void) { return -1; }
int set_cuda_default_device_from_local_rank_noncollective(void) { return -1; }
int set_openacc_device_from_local_rank(acc_device_t type) { (void)type;return 0; }
int set_openacc_nvidia_device_from_local_rank(void) { return 0; }
int set_openacc_device_from_local_rank_noncollective(acc_device_t type) { (void)type;return 0; }
int set_openacc_nvidia_device_from_local_rank_noncollective(void) { return 0; }
int openmx_gpu_map_rank_to_device(int rank,int ranks,int devices) {
  (void)rank;(void)ranks;(void)devices;return -1;
}
static int local_environment(const int *ids) {
  for(int i=0;ids[i]>=0;i++) {
    long value=openmx_env_long_at(ids[i],-1);
    if(value>=0 && value<=INT_MAX)return (int)value;
  }
  return -1;
}
int openmx_gpu_local_rank_noncollective(void) {
  const int ids[]={OPENMX_ENV_EXT_OMPI_COMM_WORLD_LOCAL_RANK,OPENMX_ENV_EXT_MV2_COMM_WORLD_LOCAL_RANK,
    OPENMX_ENV_EXT_SLURM_LOCALID,OPENMX_ENV_EXT_PMI_LOCAL_RANK,-1};
  int local=local_environment(ids);if(local>=0)return local;
  int initialized=0,rank=0;MPI_Initialized(&initialized);
  if(initialized)MPI_Comm_rank(MPI_COMM_WORLD,&rank);return rank;
}
int openmx_gpu_local_size_noncollective(void) {
  const int ids[]={OPENMX_ENV_EXT_OMPI_COMM_WORLD_LOCAL_SIZE,OPENMX_ENV_EXT_MV2_COMM_WORLD_LOCAL_SIZE,
    OPENMX_ENV_EXT_SLURM_NTASKS_PER_NODE,OPENMX_ENV_EXT_PMI_LOCAL_SIZE,-1};
  int local=local_environment(ids);if(local>0)return local;
  int initialized=0,size=1;MPI_Initialized(&initialized);
  if(initialized)MPI_Comm_size(MPI_COMM_WORLD,&size);return size;
}
cudaError_t OpenMX_GpuMemGetInfo(size_t *f,size_t *t) { return cudaMemGetInfo(f,t); }
void OpenMX_GpuPresizeKernelStack(void) {}

/* GPU transforms are unavailable. The existing bridge chooses ScaLAPACK. */
double openmx_gs2_cosma_host_bytes(char ta,char tb,int m,int n,int k,
    const int *da,const int *db,const int *dc,int context,int complex_values) {
  (void)ta;(void)tb;(void)m;(void)n;(void)k;(void)da;(void)db;(void)dc;
  (void)context;(void)complex_values;return -1.0;
}

#define CPU_PGEMM_PARAMETERS const char *ta,const char *tb,const int *m,const int *n, \
  const int *k,const double *alpha,const double *a,const int *ia,const int *ja, \
  const int *da,const double *b,const int *ib,const int *jb,const int *db, \
  const double *beta,double *c,const int *ic,const int *jc,const int *dc
#define CPU_PGEMM_ARGUMENTS ta,tb,m,n,k,alpha,a,ia,ja,da,b,ib,jb,db,beta,c,ic,jc,dc
extern void pdgemm_(CPU_PGEMM_PARAMETERS);
extern void pzgemm_(CPU_PGEMM_PARAMETERS);
void cosma_pdgemm_(CPU_PGEMM_PARAMETERS) { pdgemm_(CPU_PGEMM_ARGUMENTS); }
void cosma_pzgemm_(CPU_PGEMM_PARAMETERS) { pzgemm_(CPU_PGEMM_ARGUMENTS); }
#undef CPU_PGEMM_PARAMETERS
#undef CPU_PGEMM_ARGUMENTS
