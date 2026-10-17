#include "openmx_common.h"
#include "openmx_gpu_workers.h"
#include "set_openmp_device_from_local_rank.h"
#include "set_hip_default_device_from_local_rank.h"
#include <omp.h>
#include <stdlib.h>

int set_openmp_device_from_local_rank(void)
{
    // MPI_COMM_WORLD 内でノード共有 communicator を作り、ノード内 local rank を得る。
    // この関数は MPI_COMM_WORLD 上の全 rank が collective に呼ぶ前提。
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);

    int local_rank = 0, local_size = 1;
    MPI_Comm_rank(shmcomm, &local_rank);
    MPI_Comm_size(shmcomm, &local_size);

    int ndev = omp_get_num_devices();
    int dev  = -1;

    if (0 < SCF_Gpu_Num && SCF_Gpu_Num < ndev) ndev = SCF_Gpu_Num;

    if (ndev > 0) {
        dev = openmx_gpu_map_rank_to_device(local_rank, local_size, ndev);
        omp_set_default_device(dev);
    }

    MPI_Comm_free(&shmcomm);
    return dev;  // -1: デバイス無し
}

/* 便利ラッパ（従来の nvidia 固定版と同じ呼び出し規約） */
int set_openmp_nvidia_device_from_local_rank(void)
{
    return set_openmp_device_from_local_rank();
}

int set_openmp_device_from_local_rank_noncollective(void)
{
    const int worker=openmx_gpu_worker_device();
    if (worker>=0) { omp_set_default_device(worker); return worker; }
    int ndev = omp_get_num_devices();
    int dev  = -1;

    if (0 < SCF_Gpu_Num && SCF_Gpu_Num < ndev) ndev = SCF_Gpu_Num;

    if (ndev > 0) {
        dev = openmx_gpu_map_rank_to_device(openmx_gpu_local_rank_noncollective(),
                                            openmx_gpu_local_size_noncollective(),
                                            ndev);
        omp_set_default_device(dev);
    }

    return dev;
}

int set_openmp_nvidia_device_from_local_rank_noncollective(void)
{
    return set_openmp_device_from_local_rank_noncollective();
}
