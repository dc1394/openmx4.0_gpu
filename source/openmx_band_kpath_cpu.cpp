/* FP64 post-SCF band solver. Each worker owns its matrices and LAPACK
   scratch. The MPI adapter supplies an immutable set of real-space blocks. */
#include "openmx_band_kpath.h"
#include "openmx_band_dos.h"
#include "openmx_core_host_simd.h"
#include "openmx_dense_cpu.h"
#include "openmx_exact_exchange_host.h"
#include "openmx_host_buffer.hpp"
#include "f77func.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <omp.h>
#include <stdexcept>
#include <vector>

extern "C" {
void F77_NAME(zpotrf,ZPOTRF)(const char*, const int*, double*, const int*, int*);
void F77_NAME(zhegst,ZHEGST)(const int*, const char*, const int*, double*, const int*, const double*, const int*, int*);
void F77_NAME(ztrsm,ZTRSM)(const char*, const char*, const char*, const char*, const int*, const int*,
    const double*, const double*, const int*, double*, const int*);
void F77_NAME(zheev,ZHEEV)(const char*, const char*, const int*, double*, const int*, double*, double*, const int*, double*, int*);
void F77_NAME(zheevd,ZHEEVD)(const char*, const char*, const int*, double*, const int*, double*,
    double*, const int*, double*, const int*, int*, const int*, int*);
#if defined(__ELF__)
/* These symbols are optional in vendor and static LAPACK/BLAS builds. */
decltype(F77_NAME(zheevd,ZHEEVD)) F77_NAME(zheevd_2stage,ZHEEVD_2STAGE) __attribute__((weak));
int MKL_Set_Num_Threads_Local(int) __attribute__((weak));
#endif
}

namespace {
using openmx::host::Buffer;
constexpr char lower = 'L', vectors = 'V', values = 'N';
constexpr double two_pi = 6.283185307179586476925;

struct BlasThreads {
    explicit BlasThreads(bool outer_parallel)
    {
        if (outer_parallel) {
            previous_blas = Exact_Exchange_Host_Blas_Single();
            previous_levels = omp_get_max_active_levels();
            omp_set_max_active_levels(1);
        }
    }
    ~BlasThreads()
    {
        Exact_Exchange_Host_Blas_Restore(previous_blas);
        if (previous_levels >= 0) omp_set_max_active_levels(previous_levels);
    }
    int previous_blas = 0, previous_levels = -1;
};

struct LocalBlasThreads {
    explicit LocalBlasThreads(bool outer_parallel)
    {
#if defined(__ELF__)
        if (outer_parallel && MKL_Set_Num_Threads_Local) previous = MKL_Set_Num_Threads_Local(1);
#else
        (void)outer_parallel;
#endif
    }
    ~LocalBlasThreads()
    {
#if defined(__ELF__)
        if (previous >= 0) MKL_Set_Num_Threads_Local(previous);
#endif
    }
    int previous = -1;
};

template<class T>
Buffer<T> allocate(size_t count)
{
    if (count > std::numeric_limits<size_t>::max()/sizeof(T)) return {};
    return Buffer<T>(static_cast<T*>(std::malloc(count*sizeof(T))));
}

using Translation = std::array<int,3>;
using Phase = std::array<double,2>;

Translation translation(const OpenMXKPathBlock& block)
{
    return {block.translation[0],block.translation[1],block.translation[2]};
}

struct PhasePlan {
    explicit PhasePlan(const OpenMXKPathMatrices& m) : shifts(m.block_count), indices(m.block_count)
    {
        std::transform(m.blocks,m.blocks+m.block_count,shifts.begin(),translation);
        std::sort(shifts.begin(),shifts.end());
        shifts.erase(std::unique(shifts.begin(),shifts.end()),shifts.end());
        for (int i=0; i<m.block_count; ++i) {
            const auto found = std::lower_bound(shifts.begin(),shifts.end(),translation(m.blocks[i]));
            indices[i] = int(found-shifts.begin());
        }
    }

    void evaluate(const OpenMXKPathPoint& k, Phase* phases) const
    {
        for (size_t i=0; i<shifts.size(); ++i) {
            const auto& t = shifts[i];
            const double angle = two_pi*(k.xyz[0]*t[0]+k.xyz[1]*t[1]+k.xyz[2]*t[2]);
            phases[i] = {std::cos(angle),std::sin(angle)};
        }
    }

    std::vector<Translation> shifts;
    std::vector<int> indices;
};

double* element(double* a, int ld, int i, int j)
{
    return a+2*(size_t(j)*ld+i);
}

void assemble(const OpenMXKPathMatrices& m, const PhasePlan& plan, const Phase* phase,
              OpenMXHostBlochAdd add, int spin, double* s, double* h, int order)
{
    const int n = m.order;
    if (s) std::fill_n(s,2*size_t(n)*n,0.0);
    std::fill_n(h,2*size_t(order)*order,0.0);
    for (int i=0; i<m.block_count; ++i) {
        const auto& b = m.blocks[i];
        const auto& p = phase[plan.indices[i]];
        const size_t plane = size_t(b.rows)*b.cols;
        const double* v = m.values+b.value_offset;
        double* up = element(h,order,b.row,b.col);
        if (b.value_offset >= m.exchange_offset) {
            add(up,order,v+spin*plane,nullptr,b.rows,b.cols,p[0],p[1]);
        } else if (m.noncollinear) {
            if (s) add(element(s,n,b.row,b.col),n,v,nullptr,b.rows,b.cols,p[0],p[1]);
            const double* i_up = m.imaginary_diagonals ? v+5*plane : nullptr;
            const double* i_down = m.imaginary_diagonals ? v+6*plane : nullptr;
            add(up,order,v+plane,i_up,b.rows,b.cols,p[0],p[1]);
            add(element(h,order,b.row+n,b.col+n),order,v+2*plane,i_down,b.rows,b.cols,p[0],p[1]);
            add(element(h,order,b.row,b.col+n),order,v+3*plane,v+4*plane,b.rows,b.cols,p[0],p[1]);
        } else {
            const int s_channel = m.lno ? 2*spin : 0;
            const int h_channel = m.lno ? s_channel+1 : spin+1;
            if (s) add(element(s,n,b.row,b.col),n,v+s_channel*plane,nullptr,b.rows,b.cols,p[0],p[1]);
            add(up,order,v+h_channel*plane,nullptr,b.rows,b.cols,p[0],p[1]);
        }
    }
    if (m.noncollinear) {
        for (int j=0; j<n; ++j) {
            for (int i=0; i<n; ++i) {
                const double* upper = element(h,order,j,i+n);
                double* below = element(h,order,i+n,j);
                below[0] = upper[0];
                below[1] = -upper[1];
            }
        }
    }
}

/* A successful factorization of the shifted overlap certifies that no
   overlap eigenvalue is near the cutoff. Leave room for factorization
   roundoff and for differences between the two orthogonal bases. */
bool factor_overlap(int n, double cutoff, const double* s, double* factor)
{
    double diagonal_max = 0.0;
    for (int i=0; i<n; ++i) diagonal_max = std::max(diagonal_max,s[2*(size_t(i)*n+i)]);
    const double relative_margin = std::max(1.0e-7,64.0*n*std::numeric_limits<double>::epsilon());
    const double shift = std::max(2*cutoff,relative_margin*diagonal_max);
    const size_t entries = 2*size_t(n)*n;
    std::copy_n(s,entries,factor);
    for (int i=0; i<n; ++i) element(factor,n,i,i)[0] -= shift;
    int info;
    F77_NAME(zpotrf,ZPOTRF)(&lower,&n,factor,&n,&info);
    if (info) return false;
    std::copy_n(s,entries,factor);
    F77_NAME(zpotrf,ZPOTRF)(&lower,&n,factor,&n,&info);
    return info == 0;
}

bool reduce_cholesky(int n, int order, const double* factor, double* h)
{
    const int itype = 1;
    int info;
    F77_NAME(zhegst,ZHEGST)(&itype,&lower,&n,h,&order,factor,&n,&info);
    if (info || n==order) return info==0;

    F77_NAME(zhegst,ZHEGST)(&itype,&lower,&n,element(h,order,n,n),&order,factor,&n,&info);
    if (info) return false;
    const char left='L', right='R', plain='N', adjoint='C', nonunit='N';
    const double one[2] = {1.0,0.0};
    double* off = element(h,order,n,0);
    F77_NAME(ztrsm,ZTRSM)(&left,&lower,&plain,&nonunit,&n,&n,one,factor,&n,off,&order);
    F77_NAME(ztrsm,ZTRSM)(&right,&lower,&adjoint,&nonunit,&n,&n,one,factor,&n,off,&order);
    return true;
}

/* Both orthogonalizations produce unitarily equivalent Hermitian matrices.
   Bound every possible diagonal by the Frobenius norm to guard the legacy
   1000 Ha rejection when the overlap itself needed no penalty. */
bool no_high_diagonal(int order, const double* h)
{
    double sum = 0.0;
    for (int j=0; j<order; ++j) {
        const double* diagonal = h+2*(size_t(j)*order+j);
        double off = 0.0;
#pragma omp simd reduction(+:off)
        for (int i=2; i<2*(order-j); ++i) off += diagonal[i]*diagonal[i];
        sum += diagonal[0]*diagonal[0]+2*off;
    }
    return sum < 999.0*999.0;
}

bool sorted_spectrum(int n, const double* eigen)
{
    for (int i=0; i<n; ++i)
        if (!std::isfinite(eigen[i]) || (i>0 && eigen[i]<eigen[i-1])) return false;
    return true;
}

struct LapackPlan {
    using Diagonalize = decltype(&F77_NAME(zheevd,ZHEEVD));
    Diagonalize two_stage = nullptr;
    int complex_size = 0, real_size = 0, int_size = 0;

    bool query_values(Diagonalize diagonalize, int order, const char* job)
    {
        int requested=-1, info=0, integers=0;
        double matrix[2]{}, scratch[2]{}, eigen=0.0, real=0.0;
        diagonalize(job,&lower,&order,matrix,&order,&eigen,scratch,&requested,&real,&requested,
                    &integers,&requested,&info);
        if (info || !valid_work(scratch[0]) || !valid_work(real) || integers<1) return false;
        complex_size = std::max(complex_size,int(scratch[0]));
        real_size = std::max(real_size,int(real));
        int_size = std::max(int_size,integers);
        return true;
    }

    bool query(int n, int order, bool want_vectors)
    {
        if (!query_values(F77_NAME(zheevd,ZHEEVD),order,want_vectors ? &vectors : &values)) return false;
#if defined(__ELF__)
        const auto candidate = F77_NAME(zheevd_2stage,ZHEEVD_2STAGE);
        if (!want_vectors && order>=128 && candidate && query_values(candidate,order,&values)) two_stage = candidate;
#endif
        int requested=-1, info=0;
        double matrix[2]{}, scratch[2]{}, eigen=0.0, real=0.0;
        F77_NAME(zheev,ZHEEV)(&vectors,&lower,&n,matrix,&n,&eigen,scratch,&requested,&real,&info);
        if (info || !valid_work(scratch[0])) return false;
        complex_size = std::max(complex_size,int(scratch[0]));
        real_size = std::max(real_size,3*n);
        return true;
    }

    static bool valid_work(double size)
    {
        return size>=1.0 && size<=std::numeric_limits<int>::max();
    }
};

struct Workspace {
    Buffer<double> s, h, factor, product, eig_s, work, rwork;
    Buffer<double> original_overlap, ao_slice;
    Buffer<int> iwork;
    Buffer<Phase> phases;
    bool allow_two_stage = true;

    bool prepare(int n, int order, size_t phase_count, const LapackPlan& lapack, int projected_bands)
    {
        s = allocate<double>(2*size_t(n)*n);
        factor = allocate<double>(2*size_t(n)*n);
        h = allocate<double>(2*size_t(order)*order);
        eig_s = allocate<double>(n);
        work = allocate<double>(2*size_t(lapack.complex_size));
        rwork = allocate<double>(lapack.real_size);
        iwork = allocate<int>(lapack.int_size);
        phases = allocate<Phase>(phase_count);
        if (projected_bands) {
            original_overlap = allocate<double>(2*size_t(n)*n);
            ao_slice = allocate<double>(2*size_t(order)*projected_bands);
            if (!original_overlap || !ao_slice) return false;
        }
        return s && factor && h && eig_s && work && rwork && iwork && phases;
    }

    bool spectral_overlap(int n, const LapackPlan& lapack)
    {
        int info;
        F77_NAME(zheev,ZHEEV)(&vectors,&lower,&n,s.get(),&n,eig_s.get(),work.get(),&lapack.complex_size,rwork.get(),&info);
        if (info || !sorted_spectrum(n,eig_s.get())) return false;
        for (int j=0; j<n; ++j) {
            if (eig_s.get()[j]<0.0) eig_s.get()[j] = 1.0e-14;
            if (eig_s.get()[j]==0.0) return false;
            const double inverse_root = 1.0/std::sqrt(eig_s.get()[j]);
            double* col = element(s.get(),n,0,j);
#pragma omp simd
            for (int i=0; i<2*n; ++i) col[i] *= inverse_root;
        }
        return true;
    }

    bool spectral_transform(int n, int order, double cutoff, bool penalized)
    {
        if (!product) product = allocate<double>(2*size_t(order)*order);
        if (!product) return false;
        for (int j=0; j<order; j+=n) {
            if (!openmx_dense_cpu_gemm('N','N',1,order,n,n,element(h.get(),order,0,j),order,s.get(),n,
                                      element(product.get(),order,0,j),order)) return false;
        }
        for (int i=0; i<order; i+=n) {
            if (!openmx_dense_cpu_gemm('C','N',1,n,order,n,s.get(),n,element(product.get(),order,i,0),order,
                                      element(h.get(),order,i,0),order)) return false;
        }
        if (penalized) {
            for (int i=0; i<n; ++i) {
                const double e = eig_s.get()[i];
                const double penalty = e<cutoff ? std::pow(e/cutoff,-2.0)-1.0 : 0.0;
                for (int j=i; j<order; j+=n) element(h.get(),order,j,j)[0] += penalty;
                if (element(h.get(),order,i,i)[0]>1000.0) {
                    for (int j=i; j<order; j+=n) {
                        for (int k=0; k<order; ++k) {
                            std::fill_n(element(h.get(),order,j,k),2,0.0);
                            std::fill_n(element(h.get(),order,k,j),2,0.0);
                        }
                        element(h.get(),order,j,j)[0] = 10000.0;
                    }
                }
            }
        }
        return true;
    }

    bool diagonalize(LapackPlan::Diagonalize solver, const LapackPlan& lapack, int order,
                     double* eigen, bool want_vectors = false)
    {
        int info;
        solver(want_vectors ? &vectors : &values,&lower,&order,h.get(),&order,eigen,work.get(),&lapack.complex_size,
               rwork.get(),&lapack.real_size,iwork.get(),&lapack.int_size,&info);
        return info==0 && sorted_spectrum(order,eigen);
    }

    bool coefficients(int n, int order, int first, int count, bool cholesky, double* ao)
    {
        double* selected = element(h.get(),order,0,first);
        if (cholesky) {
            const char left='L', adjoint='C', nonunit='N';
            const double one[2]={1.0,0.0};
            for (int i=0; i<order; i+=n)
                F77_NAME(ztrsm,ZTRSM)(&left,&lower,&adjoint,&nonunit,&n,&count,one,
                                      factor.get(),&n,selected+2*i,&order);
            std::copy_n(selected,2*size_t(order)*count,ao);
        } else {
            for (int i=0; i<order; i+=n)
                if (!openmx_dense_cpu_gemm('N','N',1,n,count,n,s.get(),n,selected+2*i,
                                          order,element(ao,order,i,0),order)) return false;
        }
        return true;
    }

    bool project(int n, int order, bool cholesky, const OpenMXBandDosProjection& dos, float* weights)
    {
        if (!coefficients(n,order,dos.first,dos.count,cholesky,ao_slice.get())) return false;
        // Apply S to the AO coefficients once for every selected band. S is
        // the same for both spin components of a noncollinear eigensystem.
        for (int row=0; row<order; row+=n)
            if (!openmx_dense_cpu_gemm('N','N',1,n,dos.count,n,original_overlap.get(),n,
                                      ao_slice.get()+2*row,order,h.get()+2*row,order)) return false;
        for (int band=0; band<dos.count; ++band) {
            const double* c = element(ao_slice.get(),order,0,band);
            const double* sc = element(h.get(),order,0,band);
            float* out = weights+size_t(band)*order;
            if (!dos.spin) {
#pragma omp simd
                for (int j=0; j<n; ++j)
                    out[j] = (float)openmx_dos_weight(c[2*j],c[2*j+1],sc[2*j],sc[2*j+1]);
            } else {
                for (int j=0; j<n; ++j) {
                    const int a = 2*j, b = a+2*n;
                    openmx_dos_spin_weights(dos.spin+j,c[a],c[a+1],c[b],c[b+1],sc[a],sc[a+1],sc[b],sc[b+1],out);
                }
            }
        }
        return true;
    }

    bool solve(const OpenMXKPathMatrices& m, const PhasePlan& plan, const LapackPlan& lapack,
               OpenMXHostBlochAdd add, const OpenMXKPathPoint& point, int order, double* eigen, double* ao,
               const OpenMXBandDosProjection* dos, float* weights)
    {
        const int n = m.order;
        const bool penalized = !m.lno && !(dos && dos->unregularized_overlap);
        plan.evaluate(point,phases.get());
        bool cholesky = false;
        for (int spin=0; spin<m.spin_channels; ++spin) {
            const bool new_overlap = spin==0 || m.lno;
            assemble(m,plan,phases.get(),add,spin,new_overlap ? s.get() : nullptr,h.get(),order);
            if (new_overlap) {
                if (weights) std::copy_n(s.get(),2*size_t(n)*n,original_overlap.get());
                cholesky = factor_overlap(n,m.overlap_cutoff,s.get(),factor.get());
                if (!cholesky && !spectral_overlap(n,lapack)) return false;
            }
            if (cholesky && (!reduce_cholesky(n,order,factor.get(),h.get()) ||
                             (penalized && !no_high_diagonal(order,h.get())))) {
                cholesky = false;
                assemble(m,plan,phases.get(),add,spin,nullptr,h.get(),order);
                if (!spectral_overlap(n,lapack)) return false;
            }
            if (!cholesky && !spectral_transform(n,order,m.overlap_cutoff,penalized)) return false;

            double* output = eigen+size_t(spin)*order;
            if (allow_two_stage && lapack.two_stage) {
                if (diagonalize(lapack.two_stage,lapack,order,output)) continue;
                // LAPACK can overwrite H on failure. Rebuild it for the one-stage solver.
                allow_two_stage = false;
                assemble(m,plan,phases.get(),add,spin,nullptr,h.get(),order);
                const bool transformed = cholesky ? reduce_cholesky(n,order,factor.get(),h.get())
                    : spectral_transform(n,order,m.overlap_cutoff,penalized);
                if (!transformed) return false;
            }
            if (!diagonalize(F77_NAME(zheevd,ZHEEVD),lapack,order,output,ao || weights)) return false;
            if (ao && !coefficients(n,order,0,order,cholesky,ao+2*size_t(spin)*order*order)) return false;
            if (weights && !project(n,order,cholesky,*dos,weights+size_t(spin)*dos->count*order)) return false;
        }
        return true;
    }
};
int solve_cpu(const OpenMXKPathMatrices* m, int point_count, const OpenMXKPathPoint* points,
    double* eigenvalues, double* ao, const OpenMXBandDosProjection* dos, float* weights,
    int simd_mode, size_t workspace_limit, int* threads_used)
try {
    if (!m->planar || m->order<=0 || point_count<=0 || m->block_count<=0 ||
        m->order>std::numeric_limits<int>::max()/4) return 0;
    const int order = m->noncollinear ? 2*m->order : m->order;
    const auto size_max = std::numeric_limits<size_t>::max();
    if (size_t(order)>size_max/order/(16*sizeof(double))) return 0;
    if ((ao || weights) && 1+5*size_t(order)+2*size_t(order)*order>std::numeric_limits<int>::max()) return 0;

    const size_t phase_plan_size = size_t(m->block_count)*(sizeof(Translation)+sizeof(int));
    if (workspace_limit<phase_plan_size) return 0;
    const PhasePlan plan(*m);
    const int max_workers = std::min(point_count,omp_get_max_threads());
    LapackPlan lapack;
    if (!lapack.query(m->order,order,ao || weights)) return 0;
    const int projected_bands = weights ? dos->count : 0;
    const size_t projection_bytes = weights ? 2*(size_t(m->order)*m->order+size_t(projected_bands)*order)*sizeof(double) : 0;
    // Reserve the matrix product too, in case a worker needs the spectral path.
    const size_t worker_bytes = 4*(size_t(m->order)*m->order+size_t(order)*order)*sizeof(double) +
        (m->order+2*size_t(lapack.complex_size)+lapack.real_size)*sizeof(double) +
        size_t(lapack.int_size)*sizeof(int)+plan.shifts.size()*sizeof(Phase)+projection_bytes;
    const size_t capacity = (workspace_limit-phase_plan_size)/worker_bytes;
    const int workers = int(std::min(size_t(max_workers),capacity));
    if (!workers) return 0;

    const BlasThreads blas(workers>1);
    bool success = true;
#pragma omp parallel num_threads(workers) if(workers>1) reduction(&&:success)
    {
        const LocalBlasThreads worker_blas(workers>1);
        OpenMXHostBlochAdd add = openmx_core_host_bloch_add(simd_mode);
        if (!add) add = openmx_core_host_bloch_add(OPENMX_GRID_SCALAR);
        Workspace workspace;
        success = add && workspace.prepare(m->order,order,plan.shifts.size(),lapack,projected_bands);
#pragma omp master
        *threads_used = omp_get_num_threads();
#pragma omp for schedule(dynamic,1)
        for (int k=0; k<point_count; ++k) {
            const size_t row = size_t(k)*order*m->spin_channels;
            if (success) success = workspace.solve(*m,plan,lapack,add,points[k],order,eigenvalues+row,
                ao ? ao+2*row*order : nullptr,dos,weights ? weights+row*projected_bands : nullptr);
        }
    }
    return success;
} catch (const std::bad_alloc&) {
    return 0;
} catch (const std::length_error&) {
    return 0;
}
}

extern "C" int openmx_band_kpath_solve_cpu(const OpenMXKPathMatrices* m, int point_count,
    const OpenMXKPathPoint* points, double* eigenvalues, int simd_mode, size_t workspace_limit, int* threads_used)
{
    return solve_cpu(m,point_count,points,eigenvalues,nullptr,nullptr,nullptr,simd_mode,workspace_limit,threads_used);
}

extern "C" int openmx_band_mo_vectors_cpu(const OpenMXKPathMatrices* m, int point_count,
    const OpenMXKPathPoint* points, double* eigenvalues, double* ao, int simd_mode, size_t workspace_limit, int* threads_used)
{
    return solve_cpu(m,point_count,points,eigenvalues,ao,nullptr,nullptr,simd_mode,workspace_limit,threads_used);
}

extern "C" int openmx_band_dos_cpu(const OpenMXKPathMatrices* m, int point_count,
    const OpenMXKPathPoint* points, const OpenMXBandDosProjection* projection, double* eigen, float* weights,
    int simd_mode, size_t workspace_limit, int* threads_used)
{
    return solve_cpu(m,point_count,points,eigen,nullptr,projection,weights,simd_mode,workspace_limit,threads_used);
}
