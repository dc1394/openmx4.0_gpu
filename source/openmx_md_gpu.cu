#include "openmx_md_gpu.h"
#include "openmx_gpu_memory.hpp"
#include "../core/include/openmx_md_integrator.hpp"

#include <limits>
#include <memory>
#include <new>

struct OpenMXMDGPU {
  struct HostFree {
    void operator()(void *pointer) const noexcept { (void)OpenMXGpuFreeHost(pointer); }
  };
  int device=-1;
  OpenMXGpuStream stream=nullptr;
  std::unique_ptr<void,HostFree> host;
  size_t host_bytes=0;
  openmx::gpu::DeviceBuffer<unsigned char> workspace;

  ~OpenMXMDGPU()
  {
    if (stream) {
      openmx::gpu::DeviceBinding binding(device);
      if (binding) {
        (void)OpenMXGpuStreamSynchronize(stream);
        (void)OpenMXGpuStreamDestroy(stream);
      }
    }
  }
};

namespace {

__global__ void move_atoms(OpenMXMDMoveTile *tiles, size_t count, OpenMXMDStep step)
{
  const size_t k=(size_t)blockIdx.x*blockDim.x+threadIdx.x, t=k/OPENMX_MD_TILE;
  if (t>=count) return;
  const int i=k%OPENMX_MD_TILE;
  auto &tile=tiles[t];
  openmx::md::move(tile.x[i],tile.v[i],tile.gradient[i],tile.mass[i],tile.scale[i],tile.previous_gradient[i],step);
}

__global__ void wrap_atoms(OpenMXMDPeriodicTile *tiles, size_t count, OpenMXMDCell cell)
{
  const size_t k=(size_t)blockIdx.x*blockDim.x+threadIdx.x, t=k/OPENMX_MD_TILE;
  if (t>=count) return;
  const int i=k%OPENMX_MD_TILE;
  auto &tile=tiles[t];
  const double x=tile.xyz[0][i]-cell.origin[0];
  const double y=tile.xyz[1][i]-cell.origin[1];
  const double z=tile.xyz[2][i]-cell.origin[2];
  double frac[3];
  for (int axis=0; axis<3; ++axis) {
    const double f=openmx::md::dot3(x,y,z,cell.reciprocal[axis])*0.5/cell.pi;
    const double shift=openmx::md::translation(f);
    frac[axis]=f+shift;
    tile.frac[axis][i]=frac[axis];
    tile.shift[axis][i]=shift;
  }
  for (int j=0; j<3; ++j)
    tile.xyz[j][i]=frac[0]*cell.real[0][j]+frac[1]*cell.real[1][j]
                 +frac[2]*cell.real[2][j]+cell.origin[j];
}

__global__ void npt_atoms(OpenMXMDNPTTile *tiles, size_t count, OpenMXMDNPTStep step)
{
  const size_t k=(size_t)blockIdx.x*blockDim.x+threadIdx.x, t=k/OPENMX_MD_TILE;
  if (t>=count) return;
  const int i=k%OPENMX_MD_TILE;
  auto &tile=tiles[t];
  double pos[3], vel[3], grad[3];
  for (int d=0; d<3; ++d) {
    pos[d]=tile.position[d][i];
    vel[d]=tile.velocity[d][i];
    grad[d]=tile.gradient[d][i];
  }
  tile.kinetic[i]=openmx::md::npt_kick(pos,vel,grad,tile.mass[i],step);
  for (int d=0; d<3; ++d) {
    tile.position[d][i]=pos[d];
    tile.velocity[d][i]=vel[d];
  }
}

template<class Tile, class Launch>
int execute(OpenMXMDGPU *gpu, Tile *host, size_t count, Launch launch)
{
  constexpr unsigned threads=128;
  if (count>std::numeric_limits<size_t>::max()/sizeof(Tile)) return 0;
  const size_t bytes=count*sizeof(Tile);
  const size_t grids=(count+threads/OPENMX_MD_TILE-1)/(threads/OPENMX_MD_TILE);
  if (!grids || grids>std::numeric_limits<int>::max()) return 0;
  openmx::gpu::DeviceBinding binding(gpu->device);
  if (!binding) return 0;
  if (gpu->workspace.size()<bytes) {
    gpu->workspace.reset();
    if (!gpu->workspace.allocate(bytes)) return 0;
  }
  auto *device=reinterpret_cast<Tile*>(gpu->workspace.get());
  if (OpenMXGpuMemcpyAsync(device,host,bytes,OPENMX_GPU_COPY_HOST_TO_DEVICE,gpu->stream)!=OPENMX_GPU_SUCCESS) return 0;
  launch(device,(unsigned)grids,threads,gpu->stream);
  if (OpenMXGpuGetLastError()!=OPENMX_GPU_SUCCESS) return 0;
  if (OpenMXGpuMemcpyAsync(host,device,bytes,OPENMX_GPU_COPY_DEVICE_TO_HOST,gpu->stream)!=OPENMX_GPU_SUCCESS) return 0;
  return OpenMXGpuStreamSynchronize(gpu->stream)==OPENMX_GPU_SUCCESS;
}

}

extern "C" OpenMXMDGPU *OpenMX_MD_GPU_Create()
{
  std::unique_ptr<OpenMXMDGPU> gpu(new(std::nothrow) OpenMXMDGPU);
  if (!gpu || OpenMXGpuGetDevice(&gpu->device)!=OPENMX_GPU_SUCCESS ||
      OpenMXGpuStreamCreateWithFlags(&gpu->stream,OPENMX_GPU_STREAM_NONBLOCKING)!=OPENMX_GPU_SUCCESS) return nullptr;
  return gpu.release();
}

extern "C" void OpenMX_MD_GPU_Destroy(OpenMXMDGPU *gpu)
{
  delete gpu;
}

extern "C" void *OpenMX_MD_GPU_Workspace(OpenMXMDGPU *gpu, size_t bytes)
{
  if (gpu->host_bytes<bytes) {
    gpu->host.reset();
    gpu->host_bytes=0;
    void *host=nullptr;
    if (OpenMXGpuHostAlloc(&host,bytes,0)!=OPENMX_GPU_SUCCESS) return nullptr;
    gpu->host.reset(host);
    gpu->host_bytes=bytes;
  }
  return gpu->host.get();
}

extern "C" int OpenMX_MD_GPU_Move(OpenMXMDGPU *gpu, OpenMXMDMoveTile *host,
    size_t count, const OpenMXMDStep *step)
{
  return execute(gpu,host,count,[&](auto *device, unsigned blocks, unsigned threads, OpenMXGpuStream stream) {
    move_atoms<<<blocks,threads,0,stream>>>(device,count,*step);
  });
}

extern "C" int OpenMX_MD_GPU_Periodic(OpenMXMDGPU *gpu, OpenMXMDPeriodicTile *host,
    size_t count, const OpenMXMDCell *cell)
{
  return execute(gpu,host,count,[&](auto *device, unsigned blocks, unsigned threads, OpenMXGpuStream stream) {
    wrap_atoms<<<blocks,threads,0,stream>>>(device,count,*cell);
  });
}

extern "C" int OpenMX_MD_GPU_NPT(OpenMXMDGPU *gpu, OpenMXMDNPTTile *host,
    size_t count, const OpenMXMDNPTStep *step)
{
  return execute(gpu,host,count,[&](auto *device, unsigned blocks, unsigned threads, OpenMXGpuStream stream) {
    npt_atoms<<<blocks,threads,0,stream>>>(device,count,*step);
  });
}
