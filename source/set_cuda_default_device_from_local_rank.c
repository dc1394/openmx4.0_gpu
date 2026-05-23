#include "openmx_common.h"
#include "set_cuda_default_device_from_local_rank.h"
#include <cuda_runtime.h>
#include <mpi.h>
#include <stdlib.h>

static int get_local_rank_noncollective(void)
{
    const char *env_names[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "PMI_LOCAL_RANK",
        NULL
    };

    for (int i = 0; env_names[i] != NULL; i++) {
        const char *value = getenv(env_names[i]);
        if (value != NULL && value[0] != '\0') {
            int local_rank = atoi(value);
            if (0 <= local_rank) {
                return local_rank;
            }
        }
    }

    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    }
}

int set_cuda_default_device_from_local_rank()
{
    // MPI_COMM_WORLD 内でノード共有 communicator を作り、ノード内 local rank を得る。
    // この関数は MPI_COMM_WORLD 上の全 rank が collective に呼ぶ前提。
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);

    int deviceCount;
    wait_cudafunc(cudaGetDeviceCount(&deviceCount));

    int local_rank;
    MPI_Comm_rank(shmcomm, &local_rank);

    int dev = -1;
    if (deviceCount > 0) {
        dev = local_rank % deviceCount;
        wait_cudafunc(cudaSetDevice(dev));
    }

    MPI_Comm_free(&shmcomm);

    return dev;
}

int set_cuda_default_device_from_local_rank_noncollective(void)
{
    int deviceCount;
    int local_rank;
    int dev = -1;

    wait_cudafunc(cudaGetDeviceCount(&deviceCount));

    if (deviceCount > 0) {
        local_rank = get_local_rank_noncollective();
        dev = local_rank % deviceCount;
        wait_cudafunc(cudaSetDevice(dev));
    }

    return dev;
}
