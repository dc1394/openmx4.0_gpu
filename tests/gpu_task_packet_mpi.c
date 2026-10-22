/* Real C controller/transport + C++ packets/CPU recovery. Only device
   execution and physical-node grouping are modeled; every MPI call is real. */
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_gpu_task_packet.h"
#include "openmx_gpu_task_cpu.h"

int level_stdout=0;
static int rank_id,case_id,in_remote,did_fail,remote_calls,tracking,live;
static void *allocations[512];
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL task-packet case=%d rank=%d line=%d: %s\n",case_id,rank_id,__LINE__,#x);MPI_Abort(MPI_COMM_WORLD,1);}}while(0)
static int node_split(MPI_Comm comm,int type,int key,MPI_Info info,MPI_Comm *out){
    (void)type;(void)info;int rank;MPI_Comm_rank(comm,&rank);
    return MPI_Comm_split(comm,case_id==9?0:rank/2,key,out);
}
static void remember(void *pointer){
    if(!tracking || !pointer)return;
    for(int i=0;i<512;++i)if(!allocations[i]){allocations[i]=pointer;++live;return;}
    CHECK(0);
}
void *__real_malloc(size_t);
void *__real_calloc(size_t,size_t);
void __real_free(void*);
void *__wrap_malloc(size_t bytes){
    if(tracking && case_id==7 && rank_id==2 && !did_fail && bytes>=sizeof(OpenMXGpuTaskPacket)){did_fail=1;return NULL;}
    void *p=__real_malloc(bytes);remember(p);return p;
}
void *__wrap_calloc(size_t count,size_t width){
    if(tracking && !did_fail && width==1 &&
       ((case_id==5 && rank_id==0 && count>=sizeof(OpenMXGpuTaskPacket)) ||
        (case_id==6 && rank_id==2 && in_remote))){did_fail=1;return NULL;}
    void *p=__real_calloc(count,width);remember(p);return p;
}
void __wrap_free(void *pointer){
    if(pointer)for(int i=0;i<512;++i)if(allocations[i]==pointer){allocations[i]=NULL;--live;break;}
    __real_free(pointer);
}
int __real_openmx_gpu_task_remote(void*,const void*,size_t,void**,size_t*);
int __wrap_openmx_gpu_task_remote(void *context,const void *packet,size_t bytes,void **result,size_t *result_bytes){
    in_remote=1;
    int rc=__real_openmx_gpu_task_remote(context,packet,bytes,result,result_bytes);
    in_remote=0;return rc;
}
#define OPENMX_GPU_BACKEND 0
#define OPENMX_GPU_DEVICE_H
#define OPENMX_GPU_RUNTIME_H
#define MPI_Comm_split_type node_split
#include "openmx_gpu_collective.c"
#undef MPI_Comm_split_type
static int gpu_rank_device_usable(void){return 0;}
int openmx_gpu_workers_capacity_enabled(void){return 1;}
#include "openmx_gpu_task.c"

typedef struct {double factor;} Parameters;
static void kernel(size_t first,size_t count,const void *opaque,void *const view[],int gpu){
    CHECK(!gpu);const Parameters *p=opaque;
    const double *input=view[0];const size_t *offsets=view[1];
    const int *items=openmx_gpu_task_ragged_data(view[1],count),*lookup=view[2];
    double *out=view[3],*sum=view[4],*scratch=view[5];sum[0]=sum[1]=0;
    for(size_t i=0;i<count;++i){
        scratch[i]=p->factor*input[i]+lookup[(first+i)%3];
        for(size_t j=offsets[i];j<offsets[i+1];++j)scratch[i]+=items[j];
        out[i]=scratch[i];sum[0]+=out[i];sum[1]+=first+i;
    }
}
int openmx_gpu_task_local(OpenMXGpuTask *task,const char *label){
    if(strcmp(label,"remote-range"))return 0;
    ++remote_calls;
    if(case_id==4 || (case_id==3 && rank_id==2))return 0;
    openmx_gpu_task_cpu(task);return 1;
}

int main(int argc,char **argv){
    MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank_id);
    int ranks;MPI_Comm_size(MPI_COMM_WORLD,&ranks);CHECK(ranks==6);
    for(case_id=0;case_id<11;++case_id){
        enum {N=12};double input[N],output[N+2],sum[2]={991,991};
        size_t offsets[N+1]={0};int items[2*N],lookup[3]={7,-2,5};
        for(int i=0;i<N;++i){input[i]=rank_id*10+i;offsets[i+1]=offsets[i]+(size_t)(i%3);}
        for(int i=0;i<2*N;++i)items[i]=i-3;
        for(int i=0;i<N+2;++i)output[i]=-1234;
        const int pending=case_id==2?(rank_id==0 || rank_id==4):(case_id>0 && rank_id==0);
        const size_t n=pending?(case_id==10?1:N):0;
        Parameters parameters={(double)(rank_id+1)};
        OpenMXGpuTask task={0};task.first=(size_t)rank_id*100+5;task.count=n;
        task.parameters=&parameters;task.parameter_bytes=sizeof(parameters);
        task.inputs=3;task.outputs=2;task.scratch=1;task.scope_kind=2;task.kernel=kernel;
        task.view[0]=(OpenMXGpuTaskView){input,0,sizeof(double),NULL,0};
        task.view[1]=(OpenMXGpuTaskView){items,0,0,offsets,sizeof(int)};
        task.view[2]=(OpenMXGpuTaskView){lookup,sizeof(lookup),0,NULL,0};
        task.view[3]=(OpenMXGpuTaskView){output+1,0,sizeof(double),NULL,0};
        task.view[4]=(OpenMXGpuTaskView){sum,sizeof(sum),0,NULL,0};
        task.view[5]=(OpenMXGpuTaskView){NULL,0,sizeof(double),NULL,0};
        did_fail=remote_calls=in_remote=0;CHECK(!live);tracking=1;
        int rc=openmx_gpu_task_execute(MPI_COMM_WORLD,&task,case_id!=8,"task-packet-mpi");
        CHECK(!live);tracking=0;
        const int cpu_only=pending && (case_id==4 || case_id==8 || case_id==9);
        CHECK(rc==!cpu_only);
        long double expected_sum=0,expected_indices=0;
        for(size_t i=0;i<n;++i){
            long double expected=(rank_id+1)*(rank_id*10+(long double)i)+lookup[(task.first+i)%3];
            for(size_t k=offsets[i];k<offsets[i+1];++k)expected+=items[k];
            CHECK(output[i+1]==expected);expected_sum+=expected;expected_indices+=task.first+i;
        }
        for(size_t i=n+1;i<N+2;++i)CHECK(output[i]==-1234);
        CHECK(output[0]==-1234 && sum[0]==expected_sum && sum[1]==expected_indices);
        for(int i=0;i<N;++i)CHECK(input[i]==rank_id*10+i && offsets[i+1]-offsets[i]==(size_t)(i%3));
        for(int i=0;i<2*N;++i)CHECK(items[i]==i-3);
        CHECK(lookup[0]==7 && lookup[1]==-2 && lookup[2]==5);
        int failures=0,calls=0;
        MPI_Allreduce(&did_fail,&failures,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
        MPI_Allreduce(&remote_calls,&calls,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
        CHECK(failures==((case_id==5 || case_id==6 || case_id==7)?1:0));
        CHECK((calls>0)==(case_id>0 && case_id!=8 && case_id!=9));
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank_id==0)printf("PASS: real MPI task packet case %d; source-owned host allocations balanced\n",case_id);
    }
    MPI_Finalize();return 0;
}
