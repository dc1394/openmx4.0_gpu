#include "openmx_mo_grid.h"
#include "openmx_mo_device.h"
#include "openmx_core_host_simd.h"
#include "openmx_gpu_backend.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <unistd.h>

namespace {
constexpr int batch_limit=8;
constexpr size_t grid_budget=64U*1024*1024;

/* Stable grouping by grid index turns the scatter over atoms into an
   owner-computes loop. Each entry refers to the existing FP32 AO samples. */
struct GridPlan {
    std::vector<int> points, offsets;
    std::vector<OpenMXMOTerm> terms;
    std::vector<OpenMXMODeviceAtom> atoms;
    int orbitals=0;
    size_t samples=0;

    bool prepare(const OpenMXMOGridInput& in)
    {
        std::vector<int> cursor((size_t)in.grid_size+1);
        int total=0;
        atoms.resize(in.atom_count);
        for (int a=0; a<in.atom_count; ++a) {
            const auto& atom=in.atoms[a];
            if (atom.samples<0 || atom.orbitals<0 || atom.orbitals>atom.stride ||
                atom.samples>std::numeric_limits<int>::max()-total ||
                atom.orbitals>std::numeric_limits<int>::max()-orbitals) return false;
            if (atom.orbitals==0) continue;
            const size_t count=(size_t)atom.samples*atom.stride;
            if (count>std::numeric_limits<size_t>::max()-samples) return false;
            atoms[a]={atom.stride,atom.orbitals,orbitals,samples};
            samples+=count;
            orbitals+=atom.orbitals;
            total+=atom.samples;
            for (int i=0; i<atom.samples; ++i) {
                const int grid=atom.grid[i], cell=in.kpoints ? atom.cell[i] : 0;
                if (grid<0 || grid>=in.grid_size || cell<0 || cell>=in.cell_count) return false;
                ++cursor[grid+1];
            }
        }
        std::partial_sum(cursor.begin(),cursor.end(),cursor.begin());
        for (int point=0; point<in.grid_size; ++point) {
            if (cursor[point]==cursor[point+1]) continue;
            points.push_back(point);
            offsets.push_back(cursor[point]);
        }
        offsets.push_back(total);
        terms.resize(total);
        for (int a=0; a<in.atom_count; ++a) {
            const auto& atom=in.atoms[a];
            if (atom.orbitals==0) continue;
            for (int i=0; i<atom.samples; ++i)
                terms[cursor[atom.grid[i]]++]={a,i,in.kpoints ? atom.cell[i] : 0};
        }
        return true;
    }
};

struct Orbital {
    dcomplex** coefficients;
    int spin, index;
    bool lumo;
};

Orbital orbital_at(const OpenMXMOGridInput& in, int k, int homo, int lumo, int ordinal)
{
    const int spin=ordinal/(homo+lumo);
    const int state=ordinal%(homo+lumo);
    if (state<homo) return {in.homos[k][spin][state],spin,state,false};
    return {in.lumos[k][spin][state-homo],spin,state-homo,true};
}

void phase_table(const OpenMXMOGridInput& in, int k, double* phase)
{
    constexpr double two_pi=6.283185307179586476925;
    if (!in.kpoints) {
        phase[0]=1.0;
        phase[1]=0.0;
        return;
    }
    const double* point=in.kpoints[k];
    for (int cell=0; cell<in.cell_count; ++cell) {
        const int* t=in.translations[cell];
        // The cell displacement in the wavefunction is opposite to the
        // translation used to assemble the Bloch Hamiltonian.
        const double a=-two_pi*(point[1]*t[1]+point[2]*t[2]+point[3]*t[3]);
        phase[2*(size_t)cell]=std::cos(a);
        phase[2*(size_t)cell+1]=std::sin(a);
    }
}

void copy_coefficients(const OpenMXMOGridInput& in, const GridPlan& plan,
                       int count, const Orbital* jobs, double* coeff)
{
    for (int j=0; j<count; ++j) {
        double* re=coeff+(size_t)j*in.part_count*plan.orbitals;
        double* im=in.part_count==2 ? re+plan.orbitals : nullptr;
        for (int a=0; a<in.atom_count; ++a) {
            const auto& atom=plan.atoms[a];
            if (atom.count==0) continue;
            const dcomplex* src=jobs[j].coefficients[in.atoms[a].global_atom];
            for (int i=0; i<atom.count; ++i) {
                re[atom.coefficient+i]=src[i].r;
                if (im) im[atom.coefficient+i]=src[i].i;
            }
        }
    }
}

void evaluate_cpu(const OpenMXMOGridInput& in, const GridPlan& plan,
                  int batch, const double* phase, const double* coeff, double* output)
{
    const bool complex=in.part_count==2;
#pragma omp parallel if (plan.points.size()>=1024)
    {
        OpenMXHostMODot dot=openmx_core_host_mo_dot(in.simd);
        if (!dot) dot=openmx_core_host_mo_dot(OPENMX_GRID_SCALAR);
#pragma omp for schedule(static)
        for (size_t p=0; p<plan.points.size(); ++p) {
            double real[batch_limit]{}, imag[batch_limit]{};
            for (int i=plan.offsets[p]; i<plan.offsets[p+1]; ++i) {
                const auto t=plan.terms[i];
                const auto& atom=plan.atoms[t.atom];
                const auto* paos=in.atoms[t.atom].paos+(size_t)t.sample*atom.stride;
                const double c=phase[2*(size_t)t.cell],s=phase[2*(size_t)t.cell+1];
                for (int j=0; j<batch; ++j) {
                    const double* re=coeff+(size_t)j*in.part_count*plan.orbitals+atom.coefficient;
                    double v[2];
                    dot(atom.count,paos,re,complex ? re+plan.orbitals : nullptr,v);
                    real[j]+=c*v[0]-s*v[1];
                    if (complex) imag[j]+=s*v[0]+c*v[1];
                }
            }
            const int grid=plan.points[p];
            for (int j=0; j<batch; ++j) {
                const size_t first=(size_t)j*in.part_count*in.grid_size+grid;
                output[first]=real[j];
                if (complex) output[first+in.grid_size]=imag[j];
            }
        }
    }
}

#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
using Device=std::unique_ptr<OpenMXMODevice,decltype(&openmx_mo_device_destroy)>;

Device prepare_device(const OpenMXMOGridInput& in, const GridPlan& plan, int batch) try
{
    if (in.gpu<0 || plan.points.empty()) return {nullptr,openmx_mo_device_destroy};
    std::vector<float> paos(plan.samples);
#pragma omp parallel for schedule(static) if (plan.samples>=65536)
    for (int a=0; a<in.atom_count; ++a) {
        const auto& atom=in.atoms[a];
        if (atom.samples && atom.orbitals)
            std::copy_n(atom.paos,(size_t)atom.samples*atom.stride,paos.data()+plan.atoms[a].first);
    }
    return {openmx_mo_device_create(in.gpu,in.atom_count,plan.atoms.data(),int(plan.points.size()),
        plan.offsets.data(),plan.terms.data(),paos.size(),paos.data(),plan.orbitals,in.cell_count,
        in.part_count,batch),openmx_mo_device_destroy};
} catch (const std::bad_alloc&) {
    return {nullptr,openmx_mo_device_destroy};
} catch (const std::length_error&) {
    return {nullptr,openmx_mo_device_destroy};
}
#endif

bool write_text(FILE* fp, int grid_size, int nz, const double* values)
{
    std::array<char,65536> buffer;
    size_t used=0;
    for (int i=0, z=0, group=0; i<grid_size; ++i) {
        if (buffer.size()-used<64) {
            if (std::fwrite(buffer.data(),1,used,fp)!=used) return false;
            used=0;
        }
        constexpr size_t field=13;
        char* dest=buffer.data()+used;
#if defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE >= 11
        // Keep the cube format, including its fixed field width and exponent
        // case, using the locale-independent floating-point conversion.
        char number[field];
        const auto converted=std::to_chars(number,number+field,values[i],std::chars_format::scientific,3);
        if (converted.ec==std::errc{} && std::isfinite(values[i])) {
            const size_t n=converted.ptr-number;
            std::replace(number,converted.ptr,'e','E');
            std::fill_n(dest,field-n,' ');
            std::memcpy(dest+field-n,number,n);
        } else
#endif
        {
            if (std::snprintf(dest,field+1,"%13.3E",values[i])!=(int)field) return false;
        }
        used+=field;
        ++z;
        ++group;
        if (z==nz || group==6) {
            buffer[used++]='\n';
            group=0;
            if (z==nz) z=0;
        }
    }
    return std::fwrite(buffer.data(),1,used,fp)==used;
}

bool write_cube(const OpenMXMOGridInput& in, int k, const Orbital& job, int part, const double* values)
{
    char path[2048];
    const char* kind=job.lumo ? "lumo" : "homo";
    const char* suffix=in.part_count==1 ? "_r" : (part==0 ? "_r" : "_i");
    const int length=in.kpoints
        ? std::snprintf(path,sizeof(path),"%s%s.%s%d_%d_%d%s%s",in.directory,in.name,kind,k,job.spin,
                        job.index,suffix,in.extension)
        : std::snprintf(path,sizeof(path),"%s%s.%s%d_%d%s%s",in.directory,in.name,kind,job.spin,
                        job.index,suffix,in.extension);
    if (length<0 || (size_t)length>=sizeof(path)) return false;
    FILE* fp=std::fopen(path,in.binary ? "wb" : "w");
    if (!fp) return false;
    in.cube_header(in.chemical,fp,1,job.coefficients[0][0].r,0.0);
    const bool written=in.binary ? std::fwrite(values,sizeof(double),in.grid_size,fp)==(size_t)in.grid_size
                                 : write_text(fp,in.grid_size,in.last_axis,values);
    const bool synced=std::fflush(fp)==0 && fsync(fileno(fp))==0;
    return std::fclose(fp)==0 && written && synced;
}

void reduce_grids(const OpenMXMOGridInput& in, int rank, double* grid, size_t entries)
{
    constexpr int chunk_size=1<<27;
    for (size_t first=0; first<entries; first+=chunk_size) {
        double* part=grid+first;
        const int count=int(std::min((size_t)chunk_size,entries-first));
        if (rank==in.root) MPI_Reduce(MPI_IN_PLACE,part,count,MPI_DOUBLE,MPI_SUM,in.root,in.comm);
        else MPI_Reduce(part,nullptr,count,MPI_DOUBLE,MPI_SUM,in.root,in.comm);
    }
}
}

extern "C" int openmx_mo_grid_write(const OpenMXMOGridInput* input)
{
    const auto& in=*input;
    int rank;
    MPI_Comm_rank(in.comm,&rank);
    const double start=MPI_Wtime();
    GridPlan plan;
    std::vector<double> output, phase, coefficients;
    int batch=0, ready=0, all_ready;
    try {
        if (in.simd!=OPENMX_HOST_SIMD_LEGACY && in.grid_size>0 && in.cell_count>0 && plan.prepare(in)) {
            const size_t per_orbital=(size_t)in.grid_size*in.part_count;
            batch=int(std::max(size_t(1),std::min((size_t)batch_limit,grid_budget/sizeof(double)/per_orbital)));
            output.resize(batch*per_orbital);
            coefficients.resize((size_t)batch*in.part_count*plan.orbitals);
            phase.resize(2*(size_t)in.cell_count);
            ready=1;
        }
    } catch (const std::bad_alloc&) {
    } catch (const std::length_error&) {
    }
    MPI_Allreduce(&ready,&all_ready,1,MPI_INT,MPI_MIN,in.comm);
    if (!all_ready) return 0;

    int on_gpu=0;
#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
    auto device=prepare_device(in,plan,batch);
    std::vector<double> compact;
    if (device) {
        try {
            compact.resize((size_t)batch*in.part_count*plan.points.size());
            on_gpu=1;
        } catch (const std::bad_alloc&) { device.reset(); }
          catch (const std::length_error&) { device.reset(); }
    }
#endif
    int gpu_ranks;
    MPI_Reduce(&on_gpu,&gpu_ranks,1,MPI_INT,MPI_SUM,in.root,in.comm);
    if (rank==in.root && in.verbosity>0) {
        const char *host=openmx_core_host_simd_name(in.simd);
        if (gpu_ranks>0)
            std::printf("<MO> Grid projection with %s, %d %s rank(s)\n",host,gpu_ranks,OPENMX_GPU_RUNTIME_NAME);
        else
            std::printf("<MO> Grid projection with %s\n",host);
    }
    int errors=0, retries=0, files=0;
    for (int k=0; k<in.point_count; ++k) {
        const int homo=in.nhomos[k], lumo=in.nlumos[k];
        const int jobs=(homo+lumo)*in.spin_count;
        phase_table(in,k,phase.data());
        for (int first=0; first<jobs; first+=batch) {
            const int count=std::min(batch,jobs-first);
            const int components=count*in.part_count;
            std::array<Orbital,batch_limit> work;
            for (int j=0; j<count; ++j) work[j]=orbital_at(in,k,homo,lumo,first+j);
            std::fill_n(output.data(),(size_t)components*in.grid_size,0.0);
            if (!plan.points.empty()) {
                copy_coefficients(in,plan,count,work.data(),coefficients.data());
                bool evaluated=false;
#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
                if (device) {
                    evaluated=openmx_mo_device_evaluate(device.get(),count,first==0 ? phase.data() : nullptr,
                                                         coefficients.data(),compact.data());
                    if (evaluated) {
#pragma omp parallel for schedule(static) if (plan.points.size()>=1024)
                        for (int c=0; c<components; ++c) {
                            double* target=output.data()+(size_t)c*in.grid_size;
                            const double* source=compact.data()+(size_t)c*plan.points.size();
                            for (size_t p=0; p<plan.points.size(); ++p) target[plan.points[p]]=source[p];
                        }
                    } else {
                        device.reset();
                        ++retries;
                    }
                }
#endif
                if (!evaluated) evaluate_cpu(in,plan,count,phase.data(),coefficients.data(),output.data());
            }
            reduce_grids(in,rank,output.data(),(size_t)components*in.grid_size);
            if (rank==in.root) {
#pragma omp parallel for schedule(static,1) reduction(+:errors) if(components>1)
                for (int c=0; c<components; ++c)
                    if (!write_cube(in,k,work[c/in.part_count],c%in.part_count,output.data()+(size_t)c*in.grid_size))
                        ++errors;
            }
            files+=components;
        }
    }
    int total_retries;
    MPI_Reduce(&retries,&total_retries,1,MPI_INT,MPI_SUM,in.root,in.comm);
    if (rank==in.root) {
        if (errors) std::fprintf(stderr,"<MO> Failed to save %d orbital grid file(s).\n",errors);
        if (total_retries) std::fprintf(stderr,"<MO> %d %s rank(s) completed their grid work on the CPU.\n",
                                       total_retries,OPENMX_GPU_RUNTIME_NAME);
        if (in.verbosity>0) { std::printf("<MO> Projection and output of %d grid files: %.3f s\n",files,MPI_Wtime()-start); std::fflush(stdout); }
    }
    return 1;
}
