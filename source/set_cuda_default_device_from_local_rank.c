#include "openmx_common.h"
#include "set_cuda_default_device_from_local_rank.h"
#include <cuda_runtime.h>
#include <nvml.h>
#include <mpi.h>
#include <openacc.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/file.h>
#include <sys/types.h>
#include <unistd.h>

static int get_positive_env_int(const char *env_names[])
{
    for (int i = 0; env_names[i] != NULL; i++) {
        const char *value = getenv(env_names[i]);
        if (value != NULL && value[0] != '\0') {
            int parsed = atoi(value);
            if (0 <= parsed) {
                return parsed;
            }
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
    int local_rank = get_positive_env_int(env_names);

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
    int local_size = get_positive_env_int(env_names);

    if (0 < local_size) return local_size;

    {
        int size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        return size;
    }
}

static int get_local_rank_noncollective(void)
{
    return openmx_gpu_local_rank_noncollective();
}

/* See set_cuda_default_device_from_local_rank.h for why this is a block map
   and why every device-binding site must go through it. */
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

/* Free device memory a rank must still see right after creating its own
   context before it may claim the GPU: headroom for the device-module
   load (which aborts, not fails, when it cannot allocate) and for the
   runtime's transients. */
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

/* The physical free-memory cap of OpenMX_GpuMemGetInfo (see the header).
   NVML is resolved at run time: a load-time dependency on libnvidia-ml.so.1
   would keep the binary from starting on hosts without the driver (login
   nodes, CPU-only nodes), which the GPU build otherwise handles by taking
   the host paths.  nvml.h supplies only the types.  The library is opened
   and initialized once per process and the handle of a CUDA device is
   resolved once, through its PCI bus id.  A failure of any step only
   disables the cap. */
#define GPU_MEMINFO_MAX_DEVICES 64

static nvmlReturn_t (*gpu_meminfo_nvmlInit)(void);
static nvmlReturn_t (*gpu_meminfo_nvmlDeviceGetHandleByPciBusId)(const char *, nvmlDevice_t *);
static nvmlReturn_t (*gpu_meminfo_nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t *);

static int gpu_meminfo_nvml_enabled(void)
{
    static int enabled = -1;

    if (enabled < 0) {
        const char *value = getenv("OPENMX_GPU_MEMINFO_NVML");
        enabled = (value != NULL && value[0] != '\0' && atoi(value) == 0) ? 0 : 1;
    }
    return enabled;
}

static int gpu_meminfo_nvml_load(void)
{
    void *lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);

    if (lib == NULL) return 0;
    /* the exported names; nvml.h maps the unversioned ones onto these */
    gpu_meminfo_nvmlInit =
        (nvmlReturn_t (*)(void))dlsym(lib, "nvmlInit_v2");
    gpu_meminfo_nvmlDeviceGetHandleByPciBusId =
        (nvmlReturn_t (*)(const char *, nvmlDevice_t *))dlsym(lib, "nvmlDeviceGetHandleByPciBusId_v2");
    gpu_meminfo_nvmlDeviceGetMemoryInfo =
        (nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t *))dlsym(lib, "nvmlDeviceGetMemoryInfo");
    if (gpu_meminfo_nvmlInit == NULL || gpu_meminfo_nvmlDeviceGetHandleByPciBusId == NULL ||
        gpu_meminfo_nvmlDeviceGetMemoryInfo == NULL)
        return 0;
    return gpu_meminfo_nvmlInit() == NVML_SUCCESS;
}

static int gpu_meminfo_nvml_device(int dev, nvmlDevice_t *handle)
{
    static int nvml_state = 0;                               /* 0 untried, 1 up, -1 unavailable */
    static int known[GPU_MEMINFO_MAX_DEVICES];               /* 0 untried, 1 resolved, -1 failed */
    static nvmlDevice_t handles[GPU_MEMINFO_MAX_DEVICES];
    char bus_id[32];

    if (dev < 0 || GPU_MEMINFO_MAX_DEVICES <= dev) return 0;
    if (nvml_state == 0) nvml_state = gpu_meminfo_nvml_load() ? 1 : -1;
    if (nvml_state < 0) return 0;
    if (known[dev] == 0) {
        known[dev] = (cudaDeviceGetPCIBusId(bus_id, (int)sizeof(bus_id), dev) == cudaSuccess &&
                      gpu_meminfo_nvmlDeviceGetHandleByPciBusId(bus_id, &handles[dev]) == NVML_SUCCESS) ? 1 : -1;
        if (known[dev] < 0) (void)cudaGetLastError();
    }
    if (known[dev] < 0) return 0;
    *handle = handles[dev];
    return 1;
}

cudaError_t OpenMX_GpuMemGetInfo(size_t *free_bytes, size_t *total_bytes)
{
    static int announced = 0;
    const size_t gap_min = (size_t)256 * 1024 * 1024;
    size_t cuda_free = 0, cuda_total = 0;
    cudaError_t status = cudaMemGetInfo(&cuda_free, &cuda_total);
    nvmlDevice_t handle;
    nvmlMemory_t memory;
    int dev = -1;

    if (status != cudaSuccess) return status;
    if (free_bytes != NULL) *free_bytes = cuda_free;
    if (total_bytes != NULL) *total_bytes = cuda_total;
    if (!gpu_meminfo_nvml_enabled() || free_bytes == NULL) return cudaSuccess;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        (void)cudaGetLastError();
        return cudaSuccess;
    }
    if (!gpu_meminfo_nvml_device(dev, &handle)) return cudaSuccess;
    if (gpu_meminfo_nvmlDeviceGetMemoryInfo(handle, &memory) != NVML_SUCCESS) return cudaSuccess;
    if ((size_t)memory.free < cuda_free) {
        if (!announced && gap_min < cuda_free - (size_t)memory.free) {
            /* On Linux the two figures agree, but a peer's allocation that
               lands between the two queries shows as a gap too; a second
               CUDA query after the NVML one tells the two apart (it already
               includes that allocation), so only a gap that persists is
               reported.  The cap itself needs no such care: the newer
               figure is the right one either way. */
            size_t again_free = 0, again_total = 0;

            if (cudaMemGetInfo(&again_free, &again_total) == cudaSuccess &&
                gap_min < again_free - (size_t)memory.free) {
                int initialized = 0, myid = 0;

                announced = 1;
                MPI_Initialized(&initialized);
                if (initialized) MPI_Comm_rank(mpi_comm_level1, &myid);
                if (myid == 0 && 0 < level_stdout) {
                    printf("<GPU> device memory: the CUDA runtime reports %.2f GiB free on device %d but %.2f GiB is physically free (NVML);"
                           " on WDDM platforms such as WSL2 the CUDA figure omits the allocations of other processes,"
                           " so the device-memory budgets use the smaller figure (OPENMX_GPU_MEMINFO_NVML=0 restores the CUDA one).\n",
                           (double)(again_free < cuda_free ? again_free : cuda_free) / 1073741824.0, dev,
                           (double)memory.free / 1073741824.0);
                    fflush(stdout);
                }
            }
        }
        *free_bytes = (size_t)memory.free;
    }
    return cudaSuccess;
}

/* Many ranks sharing one device can exhaust it with their CUDA contexts
   alone (~200-300 MiB each); past that point the OpenACC entry points do
   not fail, they abort ("Could not find symbol ... Rebuild this file
   with -gpu=ccNN" when the device-module load runs out of memory).  So
   the context creation runs here through CUDA runtime calls, which
   report errors, and the first OpenACC touch happens immediately after
   a verified free-memory margin.  An flock serializes the probes of the
   node's ranks so the margin one rank checked is still there when its
   module loads.  OPENMX_GPU=0 disables the GPU for the whole run. */
int gpu_rank_device_usable(void)
{
    static int checked = 0, usable = 0;
    int cuda_devices = 0, acc_devices = 0, device_count, dev;
    int lock_fd = -1, devices_seen = 0;
    size_t free_bytes = 0, total_bytes = 0;
    char lock_path[64];
    const char *env = getenv("OPENMX_GPU");

    if (checked) return usable;
    checked = 1;

    if (env != NULL && env[0] != '\0' && atoi(env) == 0) {
        acc_set_device_type(acc_device_host);
        return usable;
    }

    if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices <= 0) {
        (void)cudaGetLastError();
        acc_set_device_type(acc_device_host);
        return usable;
    }

    acc_devices = acc_get_num_devices(acc_device_nvidia);
    if (acc_devices <= 0) {
        acc_set_device_type(acc_device_host);
        return usable;
    }

    devices_seen = 1;

    /* the same rank-to-device mapping DFT_GPU_DeviceInit commits to later,
       so the context created here is the one the run keeps using */
    device_count = (cuda_devices < acc_devices) ? cuda_devices : acc_devices;
    if (0 < SCF_Gpu_Num && SCF_Gpu_Num < device_count) device_count = SCF_Gpu_Num;
    dev = openmx_gpu_map_rank_to_device(get_local_rank_noncollective(),
                                        openmx_gpu_local_size_noncollective(),
                                        device_count);

    if (cudaSetDevice(dev) != cudaSuccess) {
        (void)cudaGetLastError();
        acc_set_device_type(acc_device_host);
        fprintf(stderr,
                "gpu_rank_device_usable: cudaSetDevice(%d) failed on this rank; using host paths.\n",
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

    if (cudaFree(0) == cudaSuccess &&                   /* context creation */
        OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
        gpu_probe_reserve_bytes() <= free_bytes) {

        void *touch;

        /* align the OpenACC runtime with the CUDA device, then force its
           device initialization while the verified margin still holds */
        acc_set_device_num(dev, acc_device_nvidia);
        touch = acc_malloc((size_t)1024 * 1024);
        if (touch != NULL) {
            acc_free(touch);
            usable = 1;
        }
    }

    if (0 <= lock_fd) {
        (void)flock(lock_fd, LOCK_UN);
        (void)close(lock_fd);
    }

    if (!usable) {
        (void)cudaGetLastError();
        /* release whatever context the failed probe created; this rank
           never touches the device again */
        (void)cudaDeviceReset();
        acc_set_device_type(acc_device_host);
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

int set_cuda_default_device_from_local_rank()
{
    // MPI_COMM_WORLD 内でノード共有 communicator を作り、ノード内 local rank を得る。
    // この関数は MPI_COMM_WORLD 上の全 rank が collective に呼ぶ前提。
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);

    int deviceCount;
    wait_cudafunc(cudaGetDeviceCount(&deviceCount));

    /* scf.Gpu.Num caps the GPUs used per node (debugging aid) */
    if (0 < SCF_Gpu_Num && SCF_Gpu_Num < deviceCount) deviceCount = SCF_Gpu_Num;

    int local_rank, local_size;
    MPI_Comm_rank(shmcomm, &local_rank);
    MPI_Comm_size(shmcomm, &local_size);

    int dev = -1;
    if (deviceCount > 0) {
        dev = openmx_gpu_map_rank_to_device(local_rank, local_size, deviceCount);
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

    if (0 < SCF_Gpu_Num && SCF_Gpu_Num < deviceCount) deviceCount = SCF_Gpu_Num;

    if (deviceCount > 0) {
        local_rank = get_local_rank_noncollective();
        dev = openmx_gpu_map_rank_to_device(local_rank,
                                            openmx_gpu_local_size_noncollective(),
                                            deviceCount);
        wait_cudafunc(cudaSetDevice(dev));
    }

    return dev;
}
