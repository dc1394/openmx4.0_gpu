#include "openmx_mo_device.h"
#include "openmx_gpu_memory.hpp"

#include <algorithm>
#include <memory>
#include <new>

struct OpenMXMODevice {
    int device, points, orbitals, channels, max_batch;
    openmx::gpu::DeviceBuffer<int> offsets;
    openmx::gpu::DeviceBuffer<OpenMXMOTerm> terms;
    openmx::gpu::DeviceBuffer<OpenMXMODeviceAtom> atoms;
    openmx::gpu::DeviceBuffer<float> paos;
    openmx::gpu::DeviceBuffer<double> phases, coefficients, output;
};

namespace {
template<class T>
bool upload(openmx::gpu::DeviceBuffer<T>& dst, const T* src, size_t count)
{
    return count==0 || (dst.allocate(count) &&
        OpenMXGpuMemcpy(dst.get(),src,count*sizeof(T),OPENMX_GPU_COPY_HOST_TO_DEVICE)==OPENMX_GPU_SUCCESS);
}

/* Work is independent across grid points and MOs. The reduction within
   one point reads its AO contributions without any device atomics. */
template<bool Complex>
__global__ void mo_grid(int points, int orbitals, const int* offsets, const OpenMXMOTerm* terms,
                         const OpenMXMODeviceAtom* atoms, const float* paos,
                         const double* phases, const double* coefficients, double* result)
{
    const int point=blockIdx.x*blockDim.x+threadIdx.x;
    if (point>=points) return;
    constexpr int channels=Complex ? 2 : 1;
    const size_t mo=(size_t)blockIdx.y*channels;
    const double* real=coefficients+mo*orbitals;
    const double* imag=real+orbitals;
    double re=0.0, im=0.0;
    for (int j=offsets[point]; j<offsets[point+1]; ++j) {
        const auto term=terms[j];
        const auto atom=atoms[term.atom];
        const float* p=paos+atom.first+(size_t)term.sample*atom.stride;
        double a=0.0,b=0.0;
        for (int i=0; i<atom.count; ++i) {
            a=fma(double(p[i]),real[atom.coefficient+i],a);
            if constexpr (Complex) b=fma(double(p[i]),imag[atom.coefficient+i],b);
        }
        if constexpr (Complex) {
            const double c=phases[2*(size_t)term.cell],s=phases[2*(size_t)term.cell+1];
            re+=c*a-s*b;
            im+=s*a+c*b;
        } else re+=a;
    }
    result[mo*points+point]=re;
    if constexpr (Complex) result[(mo+1)*points+point]=im;
}
}

extern "C" OpenMXMODevice *openmx_mo_device_create(int device, int atoms, const OpenMXMODeviceAtom *atom,
    int points, const int *offsets, const OpenMXMOTerm *terms, size_t samples, const float *ao,
    int orbitals, int cells, int channels, int max_batch)
{
    openmx::gpu::DeviceBinding binding(device);
    if (!binding) return nullptr;
    const size_t coeff_count=(size_t)max_batch*channels*orbitals;
    const size_t output_count=(size_t)max_batch*channels*points;
    const size_t phase_count=(size_t)2*cells;
    const size_t bytes=sizeof(float)*samples+sizeof(OpenMXMODeviceAtom)*atoms+
        sizeof(int)*((size_t)points+1)+sizeof(OpenMXMOTerm)*offsets[points]+
        sizeof(double)*(coeff_count+output_count+phase_count);
    size_t free=0,total=0;
    // Leave memory for other MPI ranks and later device work.
    constexpr size_t reserve=512UL*1024*1024;
    if (OpenMXGpuMemGetInfo(&free,&total)!=OPENMX_GPU_SUCCESS) {
        (void)OpenMXGpuGetLastError();
        return nullptr;
    }
    if (free<reserve || bytes>free-reserve) return nullptr;
    std::unique_ptr<OpenMXMODevice> state(new(std::nothrow) OpenMXMODevice);
    if (!state) return nullptr;
    if (!upload(state->offsets,offsets,(size_t)points+1) || !upload(state->terms,terms,offsets[points]) ||
        !upload(state->atoms,atom,atoms) || !upload(state->paos,ao,samples) ||
        !state->phases.allocate(phase_count) || !state->coefficients.allocate(coeff_count) ||
        !state->output.allocate(output_count)) {
        (void)OpenMXGpuGetLastError();
        return nullptr;
    }
    state->device=device;
    state->points=points;
    state->orbitals=orbitals;
    state->channels=channels;
    state->max_batch=max_batch;
    return state.release();
}

extern "C" void openmx_mo_device_destroy(OpenMXMODevice *state)
{
    delete state;
}

extern "C" int openmx_mo_device_evaluate(OpenMXMODevice *state, int batch, const double *phase,
    const double *coefficients, double *output)
{
    if (!state || batch<1 || batch>state->max_batch) return 0;
    openmx::gpu::DeviceBinding binding(state->device);
    if (!binding) return 0;
    const size_t coeff_bytes=(size_t)batch*state->channels*state->orbitals*sizeof(double);
    if (OpenMXGpuMemcpy(state->coefficients.get(),coefficients,coeff_bytes,
                        OPENMX_GPU_COPY_HOST_TO_DEVICE)!=OPENMX_GPU_SUCCESS ||
        (state->channels==2 && phase &&
         OpenMXGpuMemcpy(state->phases.get(),phase,state->phases.size()*sizeof(double),
                         OPENMX_GPU_COPY_HOST_TO_DEVICE)!=OPENMX_GPU_SUCCESS)) {
        (void)OpenMXGpuGetLastError();
        return 0;
    }
    constexpr int threads=128;
    const dim3 grid(((size_t)state->points+threads-1)/threads,batch);
    if (state->channels==2)
        mo_grid<true><<<grid,threads>>>(state->points,state->orbitals,state->offsets.get(),state->terms.get(),
            state->atoms.get(),state->paos.get(),state->phases.get(),state->coefficients.get(),state->output.get());
    else
        mo_grid<false><<<grid,threads>>>(state->points,state->orbitals,state->offsets.get(),state->terms.get(),
            state->atoms.get(),state->paos.get(),state->phases.get(),state->coefficients.get(),state->output.get());
    const size_t output_bytes=(size_t)batch*state->channels*state->points*sizeof(double);
    const bool done=OpenMXGpuGetLastError()==OPENMX_GPU_SUCCESS &&
        OpenMXGpuMemcpy(output,state->output.get(),output_bytes,
                        OPENMX_GPU_COPY_DEVICE_TO_HOST)==OPENMX_GPU_SUCCESS;
    if (!done) (void)OpenMXGpuGetLastError();
    return done;
}
