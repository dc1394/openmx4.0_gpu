#ifndef OPENMX_FORCE_ARCHIVE_LEASE_H
#define OPENMX_FORCE_ARCHIVE_LEASE_H

/* Optional CUDA archive admission is independent of the resident context.
   Its descriptor has invocation lifetime; never register a C stack address.
   Temporary device pointers are borrowed only by the current batch. */
typedef struct {
    OpenMXNativeResources *resources;
    void *admission;
    float *first,*second;
} Force4BArchiveLease;
enum { OPENMX_FORCE_RESOURCE_ADMISSION=8 };

static int Force4B_ArchiveRelease(void *pointer,int device)
{
    Force4BArchiveLease *lease=(Force4BArchiveLease*)pointer;
    int previous,status=OpenMX_Resource_BindDevice(device,&previous);
    if(status)return status;
    if(lease->second)(void)cudaFree(lease->second);
    if(lease->first)(void)cudaFree(lease->first);
    (void)cudaGetLastError();
    openmx_mgpu_release_devices(lease->admission);
    memset(lease,0,sizeof(*lease));
    OpenMX_Resource_RestoreDevice(device,previous);
    return 0;
}

static void Force4B_ArchiveEnd(Force4BArchiveLease *lease)
{
    int status;
    if(!lease || !lease->admission)return;
    (void)openmx_core_native_release(lease->resources,OPENMX_FORCE_RESOURCE_ADMISSION,lease,&status);
}

/* When all ranks' archives cannot coexist, upload an indexed archive under
   admission before expanding per-pair recovery packets. Host data survives
   until result publication. GeForce and datacenter admission policies remain
   in the caller; HIP continues to use its existing resident/rank-turn paths. */
static int Force4B_ArchiveBegin(Force4BGpuContext *g,int kind,Force4BArchiveLease **output)
{
    int device;char error[256];
    size_t free_bytes=0,total_bytes=0,workspace1,workspace2;
    const float *first=kind==1?g->flat_host:g->c2_host;
    const float *second=kind==1?g->halo_host:NULL;
    size_t first_bytes=(kind==1?(g->flat_stride?4*g->flat_stride:4)
                                             :(g->c2_stride?4*g->c2_stride:4))*sizeof(float);
    size_t second_bytes=kind==1?(g->halo_count?g->halo_count:1)*sizeof(float):0;
    *output=NULL;
    if(!first || (kind==1 && (!second || g->flat_dev || g->halo_dev)) ||
        (kind==2 && g->c2_dev) || !gpu_rank_device_usable() ||
        openmx_gpu_capacity_test_recovery() || cudaGetDevice(&device)!=cudaSuccess)return 0;
    Force4B_GpuWorkspaceSizes(&workspace1,&workspace2);
    size_t workspace=kind==1?workspace1:workspace2;
    size_t reserve=Force_gpu_node_reserve(openmx_gpu_local_size_noncollective());
    size_t need=first_bytes+second_bytes+workspace+(size_t)64*1024*1024;
    void *state=NULL;
    if(!openmx_core_native_state(g->resources,"archive-lease",sizeof(Force4BArchiveLease),&state,error,sizeof(error)))return 0;
    Force4BArchiveLease *lease=(Force4BArchiveLease*)state;
    if(lease->admission)return 0;
    lease->resources=g->resources;
    lease->admission=openmx_mgpu_admit_devices(&device,1);
    if(!lease->admission)return 0;
    if(!openmx_core_native_adopt(g->resources,OPENMX_FORCE_RESOURCE_ADMISSION,lease,device,
        Force4B_ArchiveRelease,ForceGpu_DeviceSync,error,sizeof(error))) {
        (void)Force4B_ArchiveRelease(lease,device);return 0;
    }
    acc_wait_all();
    if(cudaDeviceSynchronize()!=cudaSuccess)goto decline;
    acc_clear_freelists();
    if(OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess ||
        free_bytes<=reserve || need>free_bytes-reserve)goto decline;
    if(cudaMalloc((void**)&lease->first,first_bytes)!=cudaSuccess ||
        (second_bytes && cudaMalloc((void**)&lease->second,second_bytes)!=cudaSuccess))goto decline;
    if(cudaMemcpy(lease->first,first,first_bytes,cudaMemcpyHostToDevice)!=cudaSuccess ||
        (second_bytes && cudaMemcpy(lease->second,second,second_bytes,cudaMemcpyHostToDevice)!=cudaSuccess))goto decline;
    *output=lease;
    if(openmx_env_flag_at(OPENMX_ENV_FORCE_PROFILE,0)) {
        fprintf(stderr,"<Force4B> rank %d case=%d archived GPU batch: %.1f MiB uploaded once\n",
            MYID_MPI_COMM_WORLD,kind,(double)(first_bytes+second_bytes)/1048576.0);
    }
    return 1;
decline:
    Force4B_ArchiveEnd(lease);
    return 0;
}
#endif
