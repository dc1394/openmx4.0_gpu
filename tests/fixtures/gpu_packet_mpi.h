#ifndef OPENMX_TEST_GPU_PACKET_MPI_H
#define OPENMX_TEST_GPU_PACKET_MPI_H
/* Real transport, modeled physical grouping and allocator failures shared by
   task and vector packet tests. Drivers define their packet size threshold. */
int level_stdout=0;
static int rank_id,case_id,in_pack,in_remote,did_fail,remote_calls,tracking,live;
static void *allocations[512];
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL %s-packet case=%d rank=%d line=%d: %s\n",OPENMX_TEST_PACKET_NAME,case_id,rank_id,__LINE__,#x);MPI_Abort(MPI_COMM_WORLD,1);}}while(0)
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
    if(tracking && !did_fail && ((case_id==7 && rank_id==2 && bytes>=OPENMX_TEST_PACKET_MINIMUM) ||
       (case_id==5 && rank_id==0 && in_pack))){did_fail=1;return NULL;}
    void *p=__real_malloc(bytes);remember(p);return p;
}
void *__wrap_calloc(size_t count,size_t width){
    if(tracking && !did_fail &&
       ((case_id==5 && rank_id==0 && in_pack) ||
        (case_id==6 && rank_id==2 && in_remote))){did_fail=1;return NULL;}
    void *p=__real_calloc(count,width);remember(p);return p;
}
void __wrap_free(void *pointer){
    if(pointer)for(int i=0;i<512;++i)if(allocations[i]==pointer){allocations[i]=NULL;--live;break;}
    __real_free(pointer);
}
#define OPENMX_GPU_BACKEND 0
#define OPENMX_GPU_DEVICE_H
#define OPENMX_GPU_RUNTIME_H
#define MPI_Comm_split_type node_split
#include "openmx_gpu_collective.c"
#undef MPI_Comm_split_type
static int gpu_rank_device_usable(void){return 0;}
int openmx_gpu_workers_capacity_enabled(void){return 1;}

static void packet_model_begin(void){
    did_fail=remote_calls=in_pack=in_remote=0;CHECK(!live);tracking=1;
}
static void packet_model_finish(int rc,int pending,const char *label){
    CHECK(!live);tracking=0;
    const int cpu_only=pending && (case_id==4 || case_id==8 || case_id==9);
    CHECK(rc==!cpu_only);
    int failures=0,calls=0;
    MPI_Allreduce(&did_fail,&failures,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(&remote_calls,&calls,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    CHECK(failures==((case_id==5 || case_id==6 || case_id==7)?1:0));
    CHECK((calls>0)==(case_id>0 && case_id!=8 && case_id!=9));
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank_id==0)printf("PASS: real MPI %s packet case %d; source-owned host allocations balanced\n",label,case_id);
}
#endif
