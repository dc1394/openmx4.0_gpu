#!/usr/bin/env python3
"""Check serial collinear-cluster workspace lifetimes without a GPU.

Compile the production serial spin path and cleanup helper against tracking
stubs. Verify MI300A returns both native and real-MAGMA scratch before each
turn barrier, preserves cached host eigenvectors, handles cached overlaps and
both assembly routes, and leaves non-MI300A release behavior unchanged.
This is a lifetime/dispatch check, not GPU numerical or performance validation.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    match = re.search(r'^static void\s+' + re.escape(name) + r'\([^;{}]*\)\s*\{', source, re.M)
    if match is None:
        raise ValueError(f'Missing production function {name}')
    pos, depth = source.index('{', match.start()) + 1, 1
    while depth:
        depth += (source[pos] == '{') - (source[pos] == '}')
        pos += 1
    return source[match.start():pos]


PREFIX = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(2); } } while (0)
#define wait_hipfunc(x) CHECK((x)==0)
#define MPI_INT 1
#define MPI_MIN 2
#define hipMemcpyHostToDevice 3
static int SpinP_switch=1,mpi_comm_level1=1;
static int mi300a,packed,world_rank,overlap_solves,h_solves,barriers;
static int native_live=1,real_live=1,z_live=1;
static int native_releases,real_releases,z_releases,gemm_releases,patch_builds,packed_builds;
static int start_calls,end_calls;
static double *saved_evec[2],*ClusterCol_SerialOverlap;
static int ClusterCol_SerialOverlapN;
static double packed_matrix[4]={1,0,0,1};
static int packed_order[2]={0,1};
static struct { double *d_S,*d_H; int stream,transformed_s_valid,transformed_s_dim; } ClusterCol_gpusolver_ctx;
static int openmx_gpu_is_mi300a(void) { return mi300a; }
static int openmx_hipsolver_release_workspace(void) {
    CHECK(!ClusterCol_gpusolver_ctx.d_S && !ClusterCol_gpusolver_ctx.d_H);
    native_live=0;native_releases++;return 0;
}
static int openmx_magma_release_d_workspace(void) {
    CHECK(!ClusterCol_gpusolver_ctx.d_S && !ClusterCol_gpusolver_ctx.d_H);
    real_live=0;real_releases++;return 0;
}
static int openmx_magma_release_z_workspace(void) {
    CHECK(!ClusterCol_gpusolver_ctx.d_S && !ClusterCol_gpusolver_ctx.d_H);
    z_live=0;z_releases++;return 0;
}
static void openmx_gemmul8ReleaseWorkspaces(void) { gemm_releases++; }
static int Set_Hamiltonian_GpuSolver_Packed_CacheReady(void) { return packed; }
static int Set_Hamiltonian_GpuSolver_Packed_OrderMode(void) { return 0; }
static void Set_Hamiltonian_GpuSolver_SetMP(int *mp) { CHECK(mp!=NULL); }
static int Set_Hamiltonian_GpuSolver_Packed_Size(void) { return 2; }
static int *Set_Hamiltonian_GpuSolver_Packed_OrderGA(void) { return packed_order; }
static double *Set_Hamiltonian_GpuSolver_Packed_H(int spin) { CHECK(spin==0 || spin==1);return packed_matrix; }
static double *Set_Hamiltonian_GpuSolver_Packed_Overlap(void) { return packed_matrix; }
static int Set_Hamiltonian_GpuSolver_Packed_OwnsCache(void) { return 1; }
static void ClusterCol_RefineBeginSolve(int scf,int mode,int peer) {
    CHECK(scf>0 && mode==2 && peer==-1);start_calls++;
}
static void ClusterCol_RefineEndSolve(void) { end_calls++; }
static void ClusterCol_ReleaseGpuSolverCachedEVec(int rank) {
    if(rank==0)for(int spin=0;spin<2;spin++){free(saved_evec[spin]);saved_evec[spin]=NULL;}
}
static void ClusterCol_DetailTimer_Begin(int rank,int scf,int n,int maxn) {
    CHECK(rank==world_rank && scf>0 && n==2 && maxn==2);
}
static void ClusterCol_GpuSolver_Destroy(void) {
    free(ClusterCol_gpusolver_ctx.d_S);free(ClusterCol_gpusolver_ctx.d_H);
    memset(&ClusterCol_gpusolver_ctx,0,sizeof(ClusterCol_gpusolver_ctx));
}
static void ClusterCol_GpuSolver_EnsureMatrixCapacity(int n) {
    CHECK(n==2 && !ClusterCol_gpusolver_ctx.d_S && !ClusterCol_gpusolver_ctx.d_H);
    ClusterCol_gpusolver_ctx.d_S=calloc(4,sizeof(double));
    ClusterCol_gpusolver_ctx.d_H=calloc(4,sizeof(double));
    CHECK(ClusterCol_gpusolver_ctx.d_S && ClusterCol_gpusolver_ctx.d_H);
}
static int MPI_Barrier(int comm) {
    CHECK(comm==mpi_comm_level1);
    if(world_rank==0){
        CHECK(!ClusterCol_gpusolver_ctx.d_S && !ClusterCol_gpusolver_ctx.d_H && !z_live);
        if(mi300a)CHECK(!native_live && !real_live);
    }
    barriers++;return 0;
}
static int MPI_Allreduce(const int *src,int *dst,int count,int type,int operation,int comm) {
    CHECK(count==1 && type==MPI_INT && operation==MPI_MIN && comm==mpi_comm_level1);
    *dst=*src;return 0;
}
static int hipMemcpyAsync(double *dst,const double *src,size_t size,int kind,int stream) {
    CHECK(dst && src && size==4*sizeof(double) && kind==hipMemcpyHostToDevice && stream==0);
    memcpy(dst,src,size);return 0;
}
static int hipStreamSynchronize(int stream) { CHECK(stream==0);return 0; }
static int ClusterCol_GpuVerbose(void) { return 0; }
static void ClusterCol_AbortWithMessage(const char *msg) { fprintf(stderr,"%s\n",msg);exit(3); }
static void *ClusterCol_MallocArray(size_t count,size_t size,const char *label) {
    CHECK(count>0 && size>0 && label);void *p=calloc(count,size);CHECK(p);return p;
}
static size_t ClusterCol_CheckedMulCount(size_t a,size_t b,const char *label) {
    CHECK(label && (!a || b<=SIZE_MAX/a));return a*b;
}
static void ClusterCol_BuildDenseIndex(const int *order,int *mp,int n,int count,int *index) {
    CHECK(order && mp && n==2 && count==2 && index);index[0]=0;index[1]=3;
}
static void ClusterCol_BuildDeviceDenseFromPacked(const double *a,const int *index,int count,int n,double *dst) {
    CHECK(a && index && count==2 && n==2 && dst);packed_builds++;
}
static void Patch2Device_Cluster_Owner(double ****a,int *mp,int owner,int n,double *dst) {
    (void)a;CHECK(mp && n==2 && (dst!=NULL)==owner);if(owner)patch_builds++;
}
static void ClusterCol_GpuSolver_PrepareTransformedSDevice(int rebuild,int n,double *ko) {
    CHECK(rebuild==1 && n==2 && ko && ClusterCol_gpusolver_ctx.d_S);
    native_live=real_live=z_live=1;overlap_solves++;
}
static void ClusterCol_SerialOverlapSave(int n) {
    CHECK(n==2);if(!ClusterCol_SerialOverlap)ClusterCol_SerialOverlap=calloc(4,sizeof(double));
    CHECK(ClusterCol_SerialOverlap);ClusterCol_SerialOverlapN=n;
}
static void ClusterCol_GpuSolver_SolveHamiltonianDevice(int n,int maxn,double *ko,double *evec,int spin,int shared) {
    CHECK(n==2 && maxn==2 && ko && evec && spin==-1 && shared==0);
    CHECK(ClusterCol_gpusolver_ctx.d_H);native_live=real_live=z_live=1;
    h_solves++;for(int i=0;i<4;i++)evec[i]=100*h_solves+i;
}
static void ClusterCol_StashGpuSolverDenseEVec(int rank,int spin,int n,int maxn,double **evec) {
    CHECK(rank==0 && spin>=0 && spin<=1 && n==2 && maxn==2 && evec && *evec);
    saved_evec[spin]=*evec;*evec=NULL;
}
'''


MAIN = r'''
int main(int argc,char **argv) {
    CHECK(argc==7);
    mi300a=atoi(argv[1]);packed=atoi(argv[2]);int cached=atoi(argv[3]);
    int ranks=atoi(argv[4]),world=atoi(argv[5]);world_rank=atoi(argv[6]);
    double k0[3]={0},k1[3]={0};double *ko[2]={k0,k1};int mp[2]={0};
    double ****nh[2]={NULL,NULL};
    if(cached){ClusterCol_SerialOverlap=calloc(4,sizeof(double));CHECK(ClusterCol_SerialOverlap);ClusterCol_SerialOverlapN=2;}
    const int owned=world_rank ? 0 : ranks==1 ? 2 : 1;
    const int per_call_releases=world_rank ? 0 : 1+owned;
    for(int iteration=1;iteration<=2;iteration++){
        ClusterCol_GpuSolverRootDenseSerialPath(iteration,ko,nh,NULL,ranks,world,world_rank,mp,2,2);
        CHECK(h_solves==iteration*owned && barriers==3*iteration);
        CHECK(start_calls==iteration && end_calls==iteration);
        CHECK(z_releases==iteration*per_call_releases && gemm_releases==z_releases);
        CHECK(native_releases==(mi300a ? z_releases : 0) && real_releases==native_releases);
        CHECK(overlap_solves==(!cached && owned>0 ? 1 : 0));
        if(!world_rank){
            for(int spin=0;spin<2;spin++){
                int expected=(ranks==1 || spin==world);
                CHECK((saved_evec[spin]!=NULL)==expected);
                if(expected)CHECK(saved_evec[spin][3]>100 && saved_evec[spin][3]<500);
            }
        }
        if(packed)CHECK(patch_builds==0 && packed_builds==h_solves+overlap_solves);
        else CHECK(packed_builds==0 && patch_builds==h_solves+overlap_solves);
    }
    for(int spin=0;spin<2;spin++)free(saved_evec[spin]);
    free(ClusterCol_SerialOverlap);
    puts("PASS");return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC', 'cc'))
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    source = (ROOT / 'source/Cluster_DFT_Col.c').read_text()
    names = ('ClusterCol_ReleaseSerialSolverWorkspaces', 'ClusterCol_GpuSolverRootDenseSerialPath')
    harness = '\n'.join([PREFIX, *(function(source, name) for name in names), MAIN])
    checks = 0
    with tempfile.TemporaryDirectory(prefix='openmx-cluster-serial-') as tmp:
        src, exe = Path(tmp) / 'check.c', Path(tmp) / 'check'
        src.write_text(harness)
        flags = ['-std=c11', '-O2', '-Wall', '-Wextra']
        if args.sanitize:
            flags += ['-O1', '-g', '-fsanitize=address,undefined', '-fno-sanitize-recover=all']
        subprocess.run([*shlex.split(args.cc), *flags, str(src), '-o', str(exe)], check=True)
        for mi300a in (0, 1):
            for packed in (0, 1):
                for cached in (0, 1):
                    for ranks, world, rank in ((1, 0, 0), (24, 0, 0), (24, 1, 0), (24, 0, 1), (24, 1, 1)):
                        run = subprocess.run([str(exe), *map(str, (mi300a, packed, cached, ranks, world, rank))],
                                             capture_output=True, text=True)
                        if run.returncode:
                            raise RuntimeError(f'{mi300a=}, {packed=}, {cached=}, {ranks=}, {world=}, {rank=}: {run.stderr}')
                        checks += 1
    print(f'PASS: {checks} serial cluster configurations, two SCF calls each; MI300A scratch returned before barriers and host eigenvectors retained')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
