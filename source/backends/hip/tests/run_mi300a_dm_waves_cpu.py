#!/usr/bin/env python3
"""Exercise production MI300A density-wave rollback with concurrent MPI mocks."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

# Prefer the production unit staged by CMake, falling back to the HIP overlay.
root = Path(__file__).resolve().parents[1]
source_path = root/'source/Band_DFT_NonCol.c'
if not source_path.is_file():
    root = Path(__file__).resolve().parents[4]
    source_path = root/'source/backends/hip/src/Band_DFT_NonCol.c'
source = source_path.read_text()
helper = source[source.index('static int BandNonCol_AccumulateDMCachedKWaves_HIP('):
                source.index('static void BandNonCol_CalcDMRootDenseAllK1_HIP(')]
prefix = r'''
#define _XOPEN_SOURCE 700
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_gpu_k_waves.h"
#define CHECK(x) do {if(!(x)){fprintf(stderr,"rank %d line %d: %s\n",rank_id,__LINE__,#x);abort();}}while(0)
#define RANKS 4
#define POINTS 6
#define SIZE_H 3
#define MPI_INT 0
#define MPI_MIN 0
#define OPENMX_ENV_MI300A_BAND_DM_WAVES 0
static int mpi_comm_level1;
typedef struct {double r,i;} dcomplex;
typedef struct {int entry_count,pair_count;int *basis0,*basis1,*phase_index;double *phase_r,*phase_i;} BandNonColDMEntryCache;
typedef struct {double occ[4],eig_occ[4];} BandNonColDMOccWorkspace;
typedef struct {int mi300a_k_enabled,mgpu_host_dense;void *resources,*chemical;
    BandNonColDMEntryCache dm_entry_cache;BandNonColDMOccWorkspace dm_occ_workspace;} BandNonColState;
static pthread_barrier_t barrier;
static _Thread_local int rank_id,round_id,kernel_calls,allocated;
static int inputs[RANKS],results[RANKS],calls[RANKS],rounds[RANKS];
static int missing=-1,fail_point=-1,fail_malloc_rank=-1,remote_fail_round=-1,disabled_rank=-1,recovery=0;
static int owner[POINTS]={0,1,0,2,1,0},order[POINTS],offset[POINTS+1],waves;
static double sums[RANKS][8][SIZE_H],grid[POINTS],eigen_data[POINTS][4],*eigen[POINTS];
static dcomplex values[POINTS][16],*panels[RANKS][POINTS];
static int MPI_Allreduce(const int *in,int *out,int count,int type,int op,int comm){
    (void)type;(void)op;(void)comm;CHECK(count==1);
    inputs[rank_id]=*in;pthread_barrier_wait(&barrier);
    int result=1;for(int r=0;r<RANKS;r++)if(inputs[r]<result)result=inputs[r];
    ++round_id;if(round_id==remote_fail_round)result=0;
    *out=result;pthread_barrier_wait(&barrier);return 0;
}
static int openmx_env_flag_at(int id,int fallback){(void)id;return rank_id==disabled_rank?0:fallback;}
static int openmx_gpu_capacity_test_recovery(void){return recovery;}
static void *openmx_core_native_malloc(void *r,size_t n,size_t bytes,int zero){
    (void)r;(void)zero;if(rank_id==fail_malloc_rank)return NULL;
    void *p=malloc(n*bytes);CHECK(p);allocated++;return p;
}
static void openmx_core_native_free(void *r,void *p){(void)r;if(p){allocated--;free(p);}}
static void BandNonCol_DMOccWorkspace_Ensure(BandNonColState *s,int n){(void)s;CHECK(n==3);}
static void BandNonCol_DMEntryCache_Ensure(BandNonColState *s,int *mp,int n){(void)mp;CHECK(n==2);s->dm_entry_cache.entry_count=SIZE_H;s->dm_entry_cache.pair_count=1;}
static void BandNonCol_AbortWithMessage(const char *s){fprintf(stderr,"%s\n",s);abort();}
static int BandNonCol_BuildOccupationWeightsDense(void *c,int a,int b,const double *e,double *w,double *ew){
    (void)c;CHECK(a==1 && b==3 && e);for(int k=0;k<b;k++){w[k]=1;ew[k]=2;}return b;
}
static void BandNonCol_UpdateDMEntryPhases(BandNonColState *s,double a,double b,double c){(void)s;CHECK(a==b && b==c);}
static int hipGetLastError(void){return 0;}
static int BandNonCol_CalcDMRootDenseHostPanel_HIP(int entries,int pairs,int size,int n,int n2,int nk,
    const int *a,const int *b,const int *ph,const double *pr,const double *pi,const double *occ,const double *ew,
    const dcomplex *v,int ld,double *out){
    (void)a;(void)b;(void)ph;(void)pr;(void)pi;(void)occ;(void)ew;
    CHECK(entries==SIZE_H && size==SIZE_H && pairs==1 && n==2 && n2==4 && nk==3 && ld==3);
    int p=(int)v[0].r;CHECK(owner[p]==rank_id);kernel_calls++;
    if(p==fail_point)return 1;
    for(int c=0;c<8;c++)for(int i=0;i<size;i++)out[c*size+i]=(c+1)*(p+1)+i;
    return 0;
}
'''
main = r'''
static void *worker(void *arg){
    rank_id=(int)(size_t)arg;round_id=kernel_calls=allocated=0;
    BandNonColState state={0};state.mi300a_k_enabled=1;
    double *out[8];for(int c=0;c<8;c++)out[c]=sums[rank_id][c];
    results[rank_id]=BandNonCol_AccumulateDMCachedKWaves_HIP(&state,rank_id,POINTS,waves,
        offset,order,owner,SIZE_H,NULL,2,4,3,grid,grid,grid,eigen,panels[rank_id],out);
    CHECK(!allocated);calls[rank_id]=kernel_calls;rounds[rank_id]=round_id;return NULL;
}
static void run(int expected){
    pthread_t threads[RANKS];memset(sums,0,sizeof sums);
    for(int r=0;r<RANKS;r++)for(int p=0;p<POINTS;p++)panels[r][p]=(owner[p]==r && p!=missing)?values[p]:NULL;
    for(int r=0;r<RANKS;r++)CHECK(!pthread_create(&threads[r],NULL,worker,(void*)(size_t)r));
    for(int r=0;r<RANKS;r++)CHECK(!pthread_join(threads[r],NULL));
    for(int r=0;r<RANKS;r++){
        CHECK(results[r]==expected && rounds[r]==rounds[0]);
        for(int p=0;p<POINTS;p++)CHECK(panels[r][p]==((owner[p]==r && p!=missing)?values[p]:NULL));
        for(int c=0;c<8;c++)for(int i=0;i<SIZE_H;i++){
            double value=0;if(expected)for(int p=0;p<POINTS;p++)if(owner[p]==r)value+=(c+1)*(p+1)+i;
            CHECK(sums[r][c][i]==value);
        }
    }
}
int main(void){
    int devices[RANKS]={0,0,1,1},limit[2]={1,1};
    waves=openmx_gpu_k_waves(POINTS,RANKS,2,owner,devices,limit,1,order,offset);CHECK(waves==2);
    for(int p=0;p<POINTS;p++){values[p][0].r=p;eigen[p]=eigen_data[p];grid[p]=p;}
    CHECK(!pthread_barrier_init(&barrier,NULL,RANKS));
    run(1);CHECK(calls[0]==3 && calls[1]==2 && calls[2]==1 && calls[3]==0 && rounds[0]==4);
    missing=4;run(0);CHECK(rounds[0]==1 && calls[0]==0);missing=-1;
    fail_malloc_rank=1;run(0);CHECK(rounds[0]==2 && calls[0]==0);fail_malloc_rank=-1;
    fail_point=2;run(0);CHECK(rounds[0]==3 && calls[0]==2);fail_point=-1;
    remote_fail_round=4;run(0);CHECK(calls[0]==3 && calls[1]==2);remote_fail_round=-1;
    disabled_rank=2;run(0);CHECK(rounds[0]==1);disabled_rank=-1;
    recovery=1;run(0);CHECK(rounds[0]==1);recovery=0;
    run(1);CHECK(!pthread_barrier_destroy(&barrier));
    puts("PASS: concurrent density waves; compact stride; empty owners; missing panels; local/remote failures; rollback; panels retained for replay; recovery gate");
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-mi300a-dm-waves-') as temp:
    c = Path(temp)/'test.c'; exe = Path(temp)/'test'
    c.write_text(prefix+helper+main)
    subprocess.run(shlex.split(os.environ.get('CC','cc'))+['-std=c11','-O1','-Wall','-Wextra','-Werror',
        *shlex.split(os.environ.get('CFLAGS','')),'-I'+str(root/'source'),str(c),'-pthread','-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True,timeout=30)
