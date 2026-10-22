/* Actual C controller/transport and C++ packets; only device work and
   physical-node grouping are modeled by this SDK-free integration test. */
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_gpu_vector_packet.h"
#define OPENMX_TEST_PACKET_NAME "vector"
#define OPENMX_TEST_PACKET_MINIMUM 128
#include "fixtures/gpu_packet_mpi.h"
int __real_openmx_gpu_vector_pack(void*,size_t,size_t,void**,size_t*);
int __wrap_openmx_gpu_vector_pack(void *context,size_t first,size_t last,void **packet,size_t *bytes){
    in_pack=1;int rc=__real_openmx_gpu_vector_pack(context,first,last,packet,bytes);in_pack=0;return rc;
}
int __real_openmx_gpu_vector_remote(void*,const void*,size_t,void**,size_t*);
int __wrap_openmx_gpu_vector_remote(void *context,const void *packet,size_t bytes,void **result,size_t *result_bytes){
    in_remote=1;int rc=__real_openmx_gpu_vector_remote(context,packet,bytes,result,result_bytes);in_remote=0;return rc;
}
#include "openmx_gpu_vector.c"
int openmx_gpu_vector_local(OpenMXGpuVector *j){
    if(!in_remote)return 0;
    ++remote_calls;if(case_id==4 || (case_id==3 && rank_id==2))return 0;
    if(j->mode){
        for(size_t i=0;i<j->n;++i){
            double value=0;for(int col=0;col<j->columns;++col)value+=j->coeff[col]*j->data[(size_t)col*j->ld+i];
            j->output[i]=value;
        }
    }else{
        for(int p=0;p<j->pairs;++p){
            double value=0;for(size_t i=0;i<j->n;++i)value+=j->weight[i]*j->data[(size_t)j->left[p]*j->ld+i]*j->data[(size_t)j->right[p]*j->ld+i];
            j->output[p]=value;
        }
    }
    return 1;
}
int main(int argc,char **argv){
    MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank_id);
    int ranks;MPI_Comm_size(MPI_COMM_WORLD,&ranks);CHECK(ranks==6);
    for(int mode=0;mode<2;++mode)for(case_id=0;case_id<11;++case_id){
        enum{N=12,LD=15,C=3,P=4};double data[C*LD],weights[N],coefficients[C]={2,-1,3},output[N+2];
        int left[P]={2,0,1,2},right[P]={0,1,1,2};
        for(int col=0;col<C;++col)for(int i=0;i<LD;++i)data[col*LD+i]=(rank_id+1)*(col+1)+i;
        for(int i=0;i<N;++i)weights[i]=i%3?1:-2;
        for(int i=0;i<N+2;++i)output[i]=-1234;
        const int pending=case_id==2?(rank_id==0 || rank_id==4):(case_id>0 && rank_id==0);
        const size_t n=pending?(case_id==10?1:N):0;
        OpenMXGpuVector job={0};job.n=n;job.ld=LD;job.mode=mode;job.columns=C;job.pairs=P;
        job.data=data;job.weight=weights;job.coeff=coefficients;job.left=left;job.right=right;job.output=output+1;
        packet_model_begin();int rc=openmx_gpu_vector_execute(MPI_COMM_WORLD,&job,case_id!=8,"vector-packet-mpi");
        size_t outputs=mode?n:P;
        for(size_t p=0;p<outputs;++p){
            long double expected=0;
            if(mode)for(int col=0;col<C;++col)expected+=coefficients[col]*(long double)data[col*LD+p];
            else for(size_t i=0;i<n;++i)expected+=(long double)weights[i]*data[(size_t)left[p]*LD+i]*data[(size_t)right[p]*LD+i];
            CHECK(output[p+1]==expected);
        }
        CHECK(output[0]==-1234);for(size_t i=outputs+1;i<N+2;++i)CHECK(output[i]==-1234);
        for(int col=0;col<C;++col)for(int i=0;i<LD;++i)CHECK(data[col*LD+i]==(rank_id+1)*(col+1)+i);
        for(int i=0;i<N;++i)CHECK(weights[i]==(i%3?1:-2));
        CHECK(coefficients[0]==2 && coefficients[1]==-1 && coefficients[2]==3);
        const int want_left[P]={2,0,1,2},want_right[P]={0,1,1,2};
        CHECK(!memcmp(left,want_left,sizeof(left)) && !memcmp(right,want_right,sizeof(right)));
        packet_model_finish(rc,pending,mode?"vector mode=1":"vector mode=0");
    }
    MPI_Finalize();return 0;
}
