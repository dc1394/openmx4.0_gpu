#include "openmx_md_cuda.h"
#include "openmx_gpu_memory.hpp"
#include "../core/include/openmx_md_integrator.hpp"

#include <limits>
#include <memory>
#include <new>

struct OpenMXMDCuda {
  struct HostFree {
    void operator()(void *pointer) const noexcept { (void)cudaFreeHost(pointer); }
  };
  int device=-1;
  cudaStream_t stream=nullptr;
  std::unique_ptr<void,HostFree> host;
  size_t host_bytes=0;
  openmx::gpu::DeviceBuffer<unsigned char> workspace;

  ~OpenMXMDCuda()
  {
    if (stream) {
      openmx::gpu::DeviceBinding binding(device);
      if (binding) {
        (void)cudaStreamSynchronize(stream);
        (void)cudaStreamDestroy(stream);
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
int execute(OpenMXMDCuda *cuda, Tile *host, size_t count, Launch launch)
{
  constexpr unsigned threads=128;
  if (count>std::numeric_limits<size_t>::max()/sizeof(Tile)) return 0;
  const size_t bytes=count*sizeof(Tile);
  const size_t grids=(count+threads/OPENMX_MD_TILE-1)/(threads/OPENMX_MD_TILE);
  if (!grids || grids>std::numeric_limits<int>::max()) return 0;
  openmx::gpu::DeviceBinding binding(cuda->device);
  if (!binding) return 0;
  if (cuda->workspace.size()<bytes) {
    cuda->workspace.reset();
    if (!cuda->workspace.allocate(bytes)) return 0;
  }
  auto *device=reinterpret_cast<Tile*>(cuda->workspace.get());
  if (cudaMemcpyAsync(device,host,bytes,cudaMemcpyHostToDevice,cuda->stream)!=cudaSuccess) return 0;
  launch(device,(unsigned)grids,threads,cuda->stream);
  if (cudaGetLastError()!=cudaSuccess) return 0;
  if (cudaMemcpyAsync(host,device,bytes,cudaMemcpyDeviceToHost,cuda->stream)!=cudaSuccess) return 0;
  return cudaStreamSynchronize(cuda->stream)==cudaSuccess;
}

}

extern "C" OpenMXMDCuda *OpenMX_MD_Cuda_Create()
{
  std::unique_ptr<OpenMXMDCuda> cuda(new(std::nothrow) OpenMXMDCuda);
  if (!cuda || cudaGetDevice(&cuda->device)!=cudaSuccess ||
      cudaStreamCreateWithFlags(&cuda->stream,cudaStreamNonBlocking)!=cudaSuccess) return nullptr;
  return cuda.release();
}

extern "C" void OpenMX_MD_Cuda_Destroy(OpenMXMDCuda *cuda)
{
  delete cuda;
}

extern "C" void *OpenMX_MD_Cuda_Workspace(OpenMXMDCuda *cuda, size_t bytes)
{
  if (cuda->host_bytes<bytes) {
    cuda->host.reset();
    cuda->host_bytes=0;
    void *host=nullptr;
    if (cudaMallocHost(&host,bytes)!=cudaSuccess) return nullptr;
    cuda->host.reset(host);
    cuda->host_bytes=bytes;
  }
  return cuda->host.get();
}

extern "C" int OpenMX_MD_Cuda_Move(OpenMXMDCuda *cuda, OpenMXMDMoveTile *host,
    size_t count, const OpenMXMDStep *step)
{
  return execute(cuda,host,count,[&](auto *device, unsigned blocks, unsigned threads, cudaStream_t stream) {
    move_atoms<<<blocks,threads,0,stream>>>(device,count,*step);
  });
}

extern "C" int OpenMX_MD_Cuda_Periodic(OpenMXMDCuda *cuda, OpenMXMDPeriodicTile *host,
    size_t count, const OpenMXMDCell *cell)
{
  return execute(cuda,host,count,[&](auto *device, unsigned blocks, unsigned threads, cudaStream_t stream) {
    wrap_atoms<<<blocks,threads,0,stream>>>(device,count,*cell);
  });
}

extern "C" int OpenMX_MD_Cuda_NPT(OpenMXMDCuda *cuda, OpenMXMDNPTTile *host,
    size_t count, const OpenMXMDNPTStep *step)
{
  return execute(cuda,host,count,[&](auto *device, unsigned blocks, unsigned threads, cudaStream_t stream) {
    npt_atoms<<<blocks,threads,0,stream>>>(device,count,*step);
  });
}
