#include "openmx_common.h"
#include "set_hip_default_device_from_local_rank.h"
#include "openmx_gpu_workers.h"
#include "hip_runtime_compat.h"
#include <mpi.h>
#include <omp.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/file.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif

static int get_nonnegative_env_int(const char *env_names[])
{
    int i;

    for (i = 0; env_names[i] != NULL; i++) {
        const char *value = getenv(env_names[i]);
        if (value != NULL && value[0] != '\0') {
            int parsed = atoi(value);
            if (0 <= parsed) return parsed;
        }
    }
    return -1;
}

int openmx_gpu_local_rank_noncollective(void)
{
    const char *env_names[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "PMI_LOCAL_RANK",
        NULL
    };
    int local_rank = get_nonnegative_env_int(env_names);

    if (0 <= local_rank) return local_rank;

    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    }
}

int openmx_gpu_local_size_noncollective(void)
{
    const char *env_names[] = {
        "OMPI_COMM_WORLD_LOCAL_SIZE",
        "MV2_COMM_WORLD_LOCAL_SIZE",
        "SLURM_NTASKS_PER_NODE",
        "PMI_LOCAL_SIZE",
        NULL
    };
    int local_size = get_nonnegative_env_int(env_names);

    if (0 < local_size) return local_size;

    {
        int size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        return size;
    }
}

int openmx_gpu_map_rank_to_device(int local_rank, int local_size, int device_count)
{
    const char *mode = getenv("OPENMX_GPU_RANK_MAP");
    int ranks_per_device, dev;

    if (device_count <= 0) return -1;
    if (local_rank < 0) local_rank = 0;

    if (mode != NULL && (mode[0] == 'm' || mode[0] == 'M')) {
        return local_rank % device_count;
    }

    if (local_size < local_rank + 1) local_size = local_rank + 1;
    ranks_per_device = (local_size + device_count - 1) / device_count;
    dev = local_rank / ranks_per_device;
    if (device_count <= dev) dev = device_count - 1;
    return dev;
}

/* Number of MPI ranks that share one GPU, computed without communication
   from the launcher-provided node-local size and the visible device count
   (clamped by scf.Gpu.Num, matching the rank->device binding).  Used by the
   dense-GPU eigensolver paths to decide whether the device is so
   oversubscribed that the all-ranks-parallel CPU (ELPA2) path is the better
   and safer choice.  Returns at least 1. */
int openmx_gpu_ranks_per_device_noncollective(void)
{
    int local_size = openmx_gpu_local_size_noncollective();
    int ndev = omp_get_num_devices();
    int per;

    if (0 < SCF_Gpu_Num && SCF_Gpu_Num < ndev) ndev = SCF_Gpu_Num;
    if (ndev <= 0) ndev = 1;
    if (local_size <= 0) local_size = 1;

    per = (local_size + ndev - 1) / ndev;
    return (per < 1) ? 1 : per;
}

/* True when one GPU is shared by more ranks than the dense-GPU band
   eigensolver can use well.  Default limit 16 keeps the validated 8-rank GPU
   configuration on the GPU and demotes the 24-ranks-per-APU case to the CPU
   path; override with OPENMX_BAND_GPU_MAX_DEVICE_RANKS (0 disables the
   demotion entirely). */
#define BAND_GPU_DEFAULT_MAX_DEVICE_RANKS 16

int openmx_band_gpu_dense_oversubscribed(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *value = getenv("OPENMX_BAND_GPU_MAX_DEVICE_RANKS");
        int limit = BAND_GPU_DEFAULT_MAX_DEVICE_RANKS;

        if (value != NULL && value[0] != '\0') {
            int parsed = atoi(value);
            if (0 <= parsed) limit = parsed;
        }
        cached = (0 < limit && limit < openmx_gpu_ranks_per_device_noncollective()) ? 1 : 0;
    }
    return cached;
}

#define GPU_PROBE_DEFAULT_RESERVE_MB 256

static size_t gpu_probe_reserve_bytes(void)
{
    const char *value = getenv("OPENMX_GPU_PROBE_RESERVE_MB");

    if (value != NULL && value[0] != '\0') {
        long mb = atol(value);
        if (0 <= mb) return (size_t)mb * 1024U * 1024U;
    }
    return (size_t)GPU_PROBE_DEFAULT_RESERVE_MB * 1024U * 1024U;
}

/* Many ranks sharing one device can exhaust it with their HIP contexts
   alone.  The probe is serialized across ranks on a node so the free-memory
   margin checked by one rank still exists while its OpenMP target module is
   initialized.  OPENMX_GPU=0 disables GPU use for the whole run. */
int gpu_rank_device_usable(void)
{
    if (openmx_gpu_worker_device()>=0) return 1;
    static int checked = 0, usable = 0;
    int hip_devices = 0, omp_devices = 0, device_count, dev;
    int lock_fd = -1, devices_seen = 0;
    size_t free_bytes = 0, total_bytes = 0;
    char lock_path[64];
    const char *env = getenv("OPENMX_GPU");

    if (checked) return usable;
    checked = 1;

    if (env != NULL && env[0] != '\0' && atoi(env) == 0) {
        omp_set_default_device(omp_get_initial_device());
        return usable;
    }

    if (hipGetDeviceCount(&hip_devices) != hipSuccess || hip_devices <= 0) {
        omp_set_default_device(omp_get_initial_device());
        return usable;
    }

    omp_devices = omp_get_num_devices();
    if (omp_devices <= 0) {
        omp_set_default_device(omp_get_initial_device());
        return usable;
    }
    devices_seen = 1;

    device_count = (hip_devices < omp_devices) ? hip_devices : omp_devices;
    if (0 < SCF_Gpu_Num && SCF_Gpu_Num < device_count) device_count = SCF_Gpu_Num;
    dev = openmx_gpu_map_rank_to_device(openmx_gpu_local_rank_noncollective(),
                                        openmx_gpu_local_size_noncollective(),
                                        device_count);

    if (hipSetDevice(dev) != hipSuccess) {
        omp_set_default_device(omp_get_initial_device());
        fprintf(stderr,
                "gpu_rank_device_usable: hipSetDevice(%d) failed on this rank; using host paths.\n",
                dev);
        fflush(stderr);
        return usable;
    }

    snprintf(lock_path, sizeof(lock_path), "/tmp/.openmx_gpu_probe.%ld",
             (long)getuid());
    lock_fd = open(lock_path, O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (0 <= lock_fd) {
        while (flock(lock_fd, LOCK_EX) != 0 && errno == EINTR);
    }

    if (hipFree(0) == hipSuccess &&
        hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess &&
        gpu_probe_reserve_bytes() <= free_bytes) {
        void *touch;
        int offloaded = 0;

        omp_set_default_device(dev);
        touch = omp_target_alloc((size_t)1024 * 1024, dev);
        if (touch != NULL) {
            /* Force the program's OpenMP device image to load while the
               checked memory margin is protected by the node lock. */
#pragma omp target map(from:offloaded) device(dev)
            {
                offloaded = omp_is_initial_device() ? 0 : 1;
            }
            omp_target_free(touch, dev);
            if (offloaded) usable = 1;
        }
    }

    if (0 <= lock_fd) {
        (void)flock(lock_fd, LOCK_UN);
        (void)close(lock_fd);
    }

    if (!usable) {
        (void)hipDeviceReset();
        omp_set_default_device(omp_get_initial_device());
        if (devices_seen) {
            fprintf(stderr,
                    "gpu_rank_device_usable: GPU %d cannot be initialized on this rank"
                    " (%.1f MiB free of %.1f MiB, %.1f MiB reserve); using host paths.\n",
                    dev,
                    (double)free_bytes / (1024.0 * 1024.0),
                    (double)total_bytes / (1024.0 * 1024.0),
                    (double)gpu_probe_reserve_bytes() / (1024.0 * 1024.0));
            fflush(stderr);
        }
    }
    return usable;
}

int set_hip_default_device_from_local_rank()
{
    // MPI_COMM_WORLD 内でノード共有 communicator を作り、ノード内 local rank を得る。
    // この関数は MPI_COMM_WORLD 上の全 rank が collective に呼ぶ前提。
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);

    int deviceCount;
    wait_hipfunc(hipGetDeviceCount(&deviceCount));

    if (0 < SCF_Gpu_Num && SCF_Gpu_Num < deviceCount) deviceCount = SCF_Gpu_Num;

    int local_rank, local_size;
    MPI_Comm_rank(shmcomm, &local_rank);
    MPI_Comm_size(shmcomm, &local_size);

    int dev = -1;
    if (deviceCount > 0) {
        dev = openmx_gpu_map_rank_to_device(local_rank, local_size, deviceCount);
        wait_hipfunc(hipSetDevice(dev));
        omp_set_default_device(dev);
    }

    MPI_Comm_free(&shmcomm);

    return dev;
}

int set_hip_default_device_from_local_rank_noncollective(void)
{
    const int worker=openmx_gpu_worker_device();
    if (worker>=0) {
        wait_hipfunc(hipSetDevice(worker));
        omp_set_default_device(worker);
        return worker;
    }
    int deviceCount;
    int local_rank;
    int dev = -1;

    wait_hipfunc(hipGetDeviceCount(&deviceCount));

    if (0 < SCF_Gpu_Num && SCF_Gpu_Num < deviceCount) deviceCount = SCF_Gpu_Num;

    if (deviceCount > 0) {
        local_rank = openmx_gpu_local_rank_noncollective();
        dev = openmx_gpu_map_rank_to_device(local_rank,
                                            openmx_gpu_local_size_noncollective(),
                                            deviceCount);
        wait_hipfunc(hipSetDevice(dev));
        omp_set_default_device(dev);
    }

    return dev;
}

int OpenMX_GpuMemGetInfo(size_t *free_bytes, size_t *total_bytes)
{
    return (int)hipMemGetInfo(free_bytes, total_bytes);
}
