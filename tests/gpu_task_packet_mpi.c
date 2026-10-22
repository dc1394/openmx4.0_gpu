/* Real C controller/transport + C++ packets/CPU recovery. Only device
   execution and physical-node grouping are modeled; every MPI call is real. */
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_gpu_task_packet.h"
#include "openmx_gpu_task_cpu.h"

#define OPENMX_TEST_PACKET_NAME "task"
#define OPENMX_TEST_PACKET_MINIMUM sizeof(OpenMXGpuTaskPacket)
#include "fixtures/gpu_packet_mpi.h"

int __real_openmx_gpu_task_pack(void*,size_t,size_t,void**,size_t*);
int __wrap_openmx_gpu_task_pack(void *context,size_t first,size_t last,void **packet,size_t *bytes){
    in_pack=1;int rc=__real_openmx_gpu_task_pack(context,first,last,packet,bytes);in_pack=0;return rc;
}
int __real_openmx_gpu_task_remote(void*,const void*,size_t,void**,size_t*);
int __wrap_openmx_gpu_task_remote(void *context,const void *packet,size_t bytes,void **result,size_t *result_bytes){
    in_remote=1;
    int rc=__real_openmx_gpu_task_remote(context,packet,bytes,result,result_bytes);
    in_remote=0;return rc;
}
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
        packet_model_begin();
        int rc=openmx_gpu_task_execute(MPI_COMM_WORLD,&task,case_id!=8,"task-packet-mpi");
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
        packet_model_finish(rc,pending,"task");
    }
    MPI_Finalize();return 0;
}
