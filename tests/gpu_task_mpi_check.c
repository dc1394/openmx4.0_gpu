/* Real CUDA/OpenACC and MPI integration. Only the reported capacity is
   constrained; no artificial multi-gigabyte allocation is needed. */
#include "openmx_gpu_task.h"
#include "openmx_gpu_workers.h"
#include "openmx_mgpu_runtime.h"
#include "set_cuda_default_device_from_local_rank.h"
#include <openacc.h>
#include <omp.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static int rank_id,mode,cap=2,mask,cpu_calls,gpu_calls,logical_nodes;
int MPI_Comm_split_type(MPI_Comm comm,int type,int key,MPI_Info info,MPI_Comm *out){
 if(logical_nodes && type==MPI_COMM_TYPE_SHARED)return PMPI_Comm_dup(MPI_COMM_SELF,out);
 return PMPI_Comm_split_type(comm,type,key,info,out);
}
static int device_base(void){return logical_nodes?rank_id*(logical_nodes==2?1:2):0;}
static size_t need;
static void *task_allocations[128];
static int task_allocations_live;
/* Link with -Wl,--wrap=cudaMalloc. Exercise an allocation that fails only
   after the capacity query admitted the whole range on its owner. */
cudaError_t __real_cudaMalloc(void **pointer,size_t bytes);
cudaError_t __wrap_cudaMalloc(void **pointer,size_t bytes){
 if(mode==7 && bytes>need*3/4){*pointer=NULL;return cudaErrorMemoryAllocation;}
 cudaError_t status=__real_cudaMalloc(pointer,bytes);
 if(status==cudaSuccess && *pointer){
#pragma omp critical(task_allocations)
  {int found=0;for(int i=0;i<128;i++)if(!task_allocations[i]){task_allocations[i]=*pointer;task_allocations_live++;found=1;break;}assert(found);}
 }
 return status;
}
cudaError_t __real_cudaFree(void *pointer);
cudaError_t __wrap_cudaFree(void *pointer){
 cudaError_t status=__real_cudaFree(pointer);
 if(status==cudaSuccess && pointer){
#pragma omp critical(task_allocations)
  {int found=0;for(int i=0;i<128;i++)if(task_allocations[i]==pointer){task_allocations[i]=NULL;task_allocations_live--;found=1;break;}assert(found);}
 }
 return status;
}
static _Thread_local int scope=1;
int openmx_mgpu_scope_enter(int allowed){int old=scope;scope=allowed;return old;}
void openmx_mgpu_scope_leave(int old){scope=old;}
int openmx_mgpu_pool_devices(int *ids,int n){if(n>cap)n=cap;for(int i=0;i<n;i++)ids[i]=i+device_base();return n;}
int openmx_mgpu_scope_devices(int *ids,int n){return openmx_mgpu_pool_devices(ids,n);}
void *openmx_mgpu_admit_devices(const int *ids,int n){(void)ids;(void)n;return (void*)(uintptr_t)1;}
void openmx_mgpu_release_devices(void *handle){(void)handle;}
int openmx_gpu_map_rank_to_device(int r,int size,int count){(void)size;return count?r%count:0;}
int gpu_rank_device_usable(void){return mode!=6 && !(mode==4 && rank_id==0);}
void OpenMX_GpuPresizeKernelStack(void){}
cudaError_t OpenMX_GpuMemGetInfo(size_t *free_bytes,size_t *total){
 cudaError_t err=cudaMemGetInfo(free_bytes,total);if(err!=cudaSuccess)return err;
 int device=0;cudaGetDevice(&device);
 double fraction=mode==6?(device==device_base()?0:2):mode==0||mode==7?2:mode==1||mode==8?.6:mode==2||mode==5?.3:mode==3?0:rank_id?2:0;
 size_t budget=256ULL*1024*1024+(size_t)(need*fraction);
 if(*free_bytes>budget)*free_bytes=budget;return err;
}
static void kernel(size_t first,size_t count,const void *parameters,void *const views[],int gpu){
 double factor=*(const double*)parameters;const double *a=views[0];double *b=views[1],*sum=views[2];double total=0;
 if(gpu){int device=-1;assert(cudaGetDevice(&device)==cudaSuccess);
#pragma omp critical
 {mask|=1<<device;gpu_calls++;}
 }else cpu_calls++;
#pragma acc parallel loop reduction(+:total) deviceptr(a,b) if(gpu)
 for(size_t i=0;i<count;i++){b[i]=factor*a[i]+(double)(first+i);total+=b[i];}
#pragma acc serial deviceptr(sum) if(gpu)
 {sum[0]=total;}
}
int main(int argc,char **argv){
 logical_nodes=argc>1 && strcmp(argv[1],"--logical-nodes")==0;
 if(argc>1 && strcmp(argv[1],"--logical-single")==0)logical_nodes=2;
 int local_only=argc>1 && strcmp(argv[1],"--local")==0;
 int single_gpu=argc>1 && strcmp(argv[1],"--single-gpu-per-node")==0;
 int single=argc>1 && strcmp(argv[1],"--single")==0;
 MPI_Init(&argc,&argv);int size;MPI_Comm_rank(MPI_COMM_WORLD,&rank_id);MPI_Comm_size(MPI_COMM_WORLD,&size);assert(size==((local_only || single)?1:2));
 MPI_Comm local;MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank_id,MPI_INFO_NULL,&local);int local_size;MPI_Comm_size(local,&local_size);assert(local_size==1);MPI_Comm_free(&local);
 int devices=0;assert(cudaGetDeviceCount(&devices)==cudaSuccess && devices>=((single || single_gpu)?1:logical_nodes==1?4:2));
 char host[MPI_MAX_PROCESSOR_NAME];int host_length=0;
 MPI_Get_processor_name(host,&host_length);
 printf("placement rank=%d host=%.*s local_size=%d visible_gpus=%d single_gpu=%d\n",
        rank_id,host_length,host,local_size,devices,single_gpu);
 cudaSetDevice(device_base());acc_set_device_num(device_base(),acc_device_nvidia);openmx_gpu_workers_configure(0,1,2);
 enum {N=65536};double *a=malloc(N*sizeof(double)),*b=malloc(N*sizeof(double)),sum,factor=1.25+rank_id;
 for(int i=0;i<N;i++)a[i]=(i%19)*.125+rank_id;
 need=2*N*sizeof(double)+16;
 unsetenv("OPENMX_GPU_WORKERS");setenv("OPENMX_MGPU_VERBOSE","1",1);
 for(mode=0;mode<9;mode++){
  if(local_only && (mode==2 || mode==4 || mode==8))continue;
  if((single_gpu || logical_nodes==2) && (mode==1 || mode==2 || mode==6 || mode==7))continue;
  if(single && (mode==1 || mode==2 || mode==4 || mode==6 || mode==8))continue;
  cap=mode==5 || mode==8 || single || single_gpu || logical_nodes==2?1:2;mask=cpu_calls=gpu_calls=0;
  for(int i=0;i<N;i++)b[i]=-999;
  OpenMXGpuTask task={0};task.first=17;task.count=N;task.parameters=&factor;task.parameter_bytes=sizeof(factor);
  task.inputs=1;task.outputs=2;task.kernel=kernel;
  task.view[0]=(OpenMXGpuTaskView){a,0,sizeof(double)};task.view[1]=(OpenMXGpuTaskView){b,0,sizeof(double)};task.view[2]=(OpenMXGpuTaskView){&sum,sizeof(double),0};
  int done=openmx_gpu_task_execute(MPI_COMM_WORLD,&task,1,"capacity-integration");
  assert(task_allocations_live==0);
  double expected_sum=0;for(int i=0;i<N;i++){double expected=factor*a[i]+17+i;assert(b[i]==expected);expected_sum+=expected;}
  assert(fabs(sum-expected_sum)<1e-6);
  if(mode==0)assert(done && (mask>>device_base())==1 && !cpu_calls);
  if(mode==1||mode==2||(mode==7 && !single))assert(done && (mask>>device_base())==3 && !cpu_calls);
  if(mode==7 && single)assert(!done && cpu_calls && !gpu_calls);
  if(mode==3||mode==5)assert(!done && cpu_calls && !gpu_calls);
  if(mode==6)assert(done && !cpu_calls && (mask>>device_base())==2);
  if(mode==8)assert(done && !cpu_calls && (mask>>device_base())==1);
  if(mode==4)assert(done && !cpu_calls && (rank_id?gpu_calls>=2:!gpu_calls));
  MPI_Barrier(MPI_COMM_WORLD);
  if(!rank_id){printf("PASS mode=%d (owner/local/remote/CPU/occupied-owner/GPU-cap/failed-owner-init/late-OOM)\n",mode);fflush(stdout);}
 }
 free(a);free(b);MPI_Finalize();if(single)acc_shutdown(acc_device_nvidia);return 0;
}
