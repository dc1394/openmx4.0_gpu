#!/usr/bin/env python3
"""Keep DC-LNO on the CUDA/OpenACC device assigned before its first SCF step."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from run_dclno_solver_retry import function


PREFIX = r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

typedef int MPI_Comm;
typedef int cudaError_t;
enum { MPI_COMM_TYPE_SHARED=1, MPI_INFO_NULL=0, cudaSuccess=0,
       OPENMX_ENV_GPU_RANK_MAP=1 };
typedef struct {
    int gpu_proxy_initialized, node_rank, node_size, ngpu, gpu_id;
    int gpu_group_rank, gpu_group_size, is_gpu_owner;
    MPI_Comm node_comm, gpu_group_comm;
} DCLNOState;

static int SCF_Gpu_Num, local_rank, local_size, node_rank, node_size;
static int visible_devices, cuda_device, acc_device, expected_device;
static int device_queries, device_bindings;
static const char *rank_map;

static const char *openmx_env_at(int key) { assert(key==OPENMX_ENV_GPU_RANK_MAP);return rank_map; }
static int openmx_gpu_local_rank_noncollective(void) { return local_rank; }
static int openmx_gpu_local_size_noncollective(void) { return local_size; }
static MPI_Comm DCLNO_GPUProxy_BaseComm(void) { return 1; }
static void MPI_Comm_split_type(MPI_Comm comm,int kind,int key,int info,MPI_Comm *out)
{ assert(comm==1 && kind==MPI_COMM_TYPE_SHARED);*out=2; }
static void MPI_Comm_rank(MPI_Comm comm,int *rank) { *rank=comm==2?node_rank:0; }
static void MPI_Comm_size(MPI_Comm comm,int *size) { *size=comm==2?node_size:1; }
static void MPI_Comm_split(MPI_Comm comm,int color,int key,MPI_Comm *out)
{
    if(color!=expected_device) {
        fprintf(stderr,"DC-LNO changed rank %d/%d from device %d to %d\n",
                local_rank,local_size,expected_device,color);
        exit(2);
    }
    *out=3;
}
static void MPI_Abort(MPI_Comm comm,int code) { abort(); }
static int cudaGetDeviceCount(int *count) { *count=visible_devices;device_queries++;return cudaSuccess; }
static int cudaSetDevice(int device) { cuda_device=device;device_bindings++;return cudaSuccess; }
#define wait_cudafunc(call) assert((call)==cudaSuccess)
'''

MAIN = r'''
int main(void)
{
    int checks=0;
    for(int mode=0;mode<2;mode++) {
        rank_map=mode?"mod":NULL;
        const int rank_counts[]={32,1,2,3,4,11,18,25,33};
        for(int rank_case=0;rank_case<9;rank_case++) {
            int ranks=rank_counts[rank_case];
            local_size=ranks;
            for(int devices=1;devices<=4;devices++) {
                visible_devices=devices;
                for(int cap=0;cap<=devices;cap++) {
                    SCF_Gpu_Num=cap;
                    int allowed=cap>0 && cap<devices?cap:devices;
                    for(local_rank=0;local_rank<local_size;local_rank++) {
                        for(int subset=0;subset<2;subset++) {
                            node_rank=subset?0:local_rank;
                            node_size=subset?1:local_size;
                            expected_device=openmx_gpu_map_rank_to_device(local_rank,local_size,allowed);
                            cuda_device=acc_device=expected_device;
                            device_queries=device_bindings=0;
                            DCLNOState state={0};
                            DCLNO_GPUProxy_Init(&state);
                            assert(state.gpu_id==expected_device && state.ngpu==allowed);
                            assert(cuda_device==acc_device && cuda_device==expected_device);
                            assert(state.gpu_proxy_initialized && state.is_gpu_owner);
                            assert(device_queries==1 && device_bindings==1);
                            DCLNO_GPUProxy_Init(&state);
                            assert(device_queries==1 && device_bindings==1);
                            checks++;
                        }
                    }
                }
            }
        }
    }
    puts("PASS: DC-LNO preserves shared CUDA/OpenACC rank mapping, caps, communicator subsets and initialization reuse");
    printf("Binding checks: %d\n",checks);
    return 0;
}
'''


def main():
    repository = Path(__file__).resolve().parents[1]
    solver = (repository / 'source/Divide_Conquer_LNO.c').read_text()
    mapping = (repository / 'source/set_cuda_default_device_from_local_rank.c').read_text()
    mapping = mapping.replace('int openmx_gpu_map_rank_to_device(', 'static int openmx_gpu_map_rank_to_device(', 1)
    with tempfile.TemporaryDirectory(prefix='openmx-dclno-device-') as directory:
        source = Path(directory) / 'binding.c'
        source.write_text(PREFIX + function(mapping, 'openmx_gpu_map_rank_to_device') + '\n' +
                          function(solver, 'DCLNO_GPUProxy_Init') + '\n' + MAIN)
        binary = Path(directory) / 'binding'
        compiler = shlex.split(os.environ.get('CC', 'cc'))
        subprocess.run(compiler + ['-std=c11', '-O2', str(source), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
