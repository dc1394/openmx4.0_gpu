/* Optional MAGMA multi-GPU host adapter. MAGMA 2.10.0 is BSD licensed;
   see makefile and third_party/magma.openmx.make.inc for the pinned dependency.
   No MAGMA implementation is copied into this file. */
#include "openmx_magma_mgpu.h"
#include "openmx_magma_lock.h"
#include <cstdio>
#include <cstring>
#include <cstdint>

extern "C" size_t openmx_magma_mgpu_device_bytes(int ngpu, int n, int is_complex)
{
    if (ngpu < 2 || ngpu > 8 || n < 1 || n > 32766) return SIZE_MAX;
    const size_t rows = size_t(n), columns = (rows + ngpu - 1)/ngpu;
    const size_t elements = 2*rows*columns + 8192*rows;
    const size_t width = is_complex ? 16 : 8, reserve = size_t(256)<<20;
    if (elements > (SIZE_MAX-reserve)/width) return SIZE_MAX;
    return width*elements + reserve;
}

static thread_local char magma_error[256] = "";

static int fail(int status, const char *message)
{
    std::snprintf(magma_error, sizeof(magma_error), "%s", message);
    return status;
}

extern "C" const char *openmx_magma_mgpu_last_error(void) { return magma_error; }

#ifdef OPENMX_HAVE_MAGMA
#include <magma_v2.h>
#include "hip_runtime_compat.h"
extern "C" __attribute__((weak)) hipError_t OpenMX_GpuMemGetInfo(size_t *, size_t *);
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cerrno>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <vector>

/* A private, versioned extension of the pinned dependency. Weak binding
   preserves prefix-only use of ordinary external MAGMA installations. */
extern "C" int openmx_magma_device_map_v1(int,const int*) __attribute__((weak));

namespace {
std::mutex &magma_mutex = openmx_magma_runtime_mutex();
using Complex = std::complex<double>;
static_assert(sizeof(magma_int_t) == sizeof(int), "MAGMA must use LP64 integer indices");
static_assert(sizeof(Complex) == sizeof(magmaDoubleComplex), "complex storage mismatch");

template<class T> T conjugate(T x) { return x; }
template<> Complex conjugate(Complex x) { return std::conj(x); }
template<class T> double real_part(T x) { return x; }
template<> double real_part(Complex x) { return x.real(); }
template<class T> bool finite(T x) { return std::isfinite(x); }
template<> bool finite(Complex x) { return std::isfinite(x.real()) && std::isfinite(x.imag()); }
template<class T> double abs_bound(T x) { return std::abs(x); }
template<> double abs_bound(Complex x) { return std::abs(x.real())+std::abs(x.imag()); }
template<class T> T load(const void *p, size_t index)
{
    T value;
    std::memcpy(&value, static_cast<const unsigned char *>(p) + index*sizeof(T), sizeof(T));
    return value;
}
template<class T> T matrix(const void *p, int lda, int i, int j, bool lower)
{
    if (i == j) return T(real_part(load<T>(p, size_t(j)*lda+i)));
    if ((i > j) == lower) return load<T>(p, size_t(j)*lda+i);
    return conjugate(load<T>(p, size_t(i)*lda+j));
}

/* Like the AMD adapter, keep one successful initialization across solves.
   Unlike call_once, balance and retry failed initialization, which can be
   transient. MAGMA caches host device properties, not dense GPU scratch. */
struct MagmaLibrary {
    bool initialized=false;
    void release() {
        if (initialized) (void)magma_finalize();
        initialized=false;
    }
    ~MagmaLibrary() { release(); }
    int ensure() {
        if (initialized) return OPENMX_MAGMA_SUCCESS;
        const int result=magma_init();
        if (result!=MAGMA_SUCCESS) {
            /* Upstream increments its initialization count even on error. */
            (void)magma_finalize();
            return fail(OPENMX_MAGMA_LIBRARY_ERROR,"magma_init failed");
        }
        initialized=true;
        return OPENMX_MAGMA_SUCCESS;
    }
} magma_library;

template<class T> struct HostWorkspace {
    std::vector<T> matrix, work;
    std::vector<double> eigenvalues, rwork;
    std::vector<magma_int_t> iwork;
    void clear() {
        HostWorkspace empty;
        matrix.swap(empty.matrix); work.swap(empty.work);
        eigenvalues.swap(empty.eigenvalues); rwork.swap(empty.rwork); iwork.swap(empty.iwork);
    }
};
HostWorkspace<double> real_workspace;
HostWorkspace<Complex> complex_workspace;
template<class T> HostWorkspace<T> &cached_workspace();
template<> HostWorkspace<double> &cached_workspace<double>() { return real_workspace; }
template<> HostWorkspace<Complex> &cached_workspace<Complex>() { return complex_workspace; }

size_t host_cache_limit()
{
    const char *s=std::getenv("OPENMX_MAGMA_HOST_CACHE_MB");
    const size_t fallback=size_t(64)<<20;
    if (!s || !*s || *s=='-') return fallback;
    char *end=nullptr;
    errno=0;
    const unsigned long long mb=std::strtoull(s,&end,10);
    return errno || end==s || *end || mb>SIZE_MAX/(size_t(1)<<20) ? fallback : size_t(mb)<<20;
}

/* Avoid vector's geometric capacity growth exceeding the retention budget.
   Reusing an existing capacity does not clear untouched workspace pages. */
template<class T> void resize_exact(std::vector<T> &v,size_t count)
{
    if (count>v.capacity()) { std::vector<T> replacement(count); v.swap(replacement); }
    else v.resize(count);
}

/* Every solve still restores the caller's device, including all failures. */
struct Session {
    int previous = -1;
    bool mapped = false;
    ~Session() {
        if (mapped) (void)openmx_magma_device_map_v1(0,nullptr);
        if (previous >= 0) (void)hipSetDevice(previous);
    }
};

int device_preflight(int ngpu, const int *devices, int n, size_t width, Session &session)
{
    int count = 0;
    if (hipGetDevice(&session.previous) != hipSuccess || hipGetDeviceCount(&count) != hipSuccess)
        return fail(OPENMX_MAGMA_HIP_ERROR, "HIP device query failed");
    if (count < ngpu) return fail(OPENMX_MAGMA_UNAVAILABLE, "fewer visible GPUs than requested");
    bool prefix=true;
    for (int j=0;j<ngpu;++j) {
        if (devices[j]<0 || devices[j]>=count)
            return fail(OPENMX_MAGMA_INVALID,"MAGMA device ordinal is not visible");
        for (int i=0;i<j;++i) if (devices[i]==devices[j])
            return fail(OPENMX_MAGMA_INVALID,"MAGMA device ordinals must be distinct");
        prefix=prefix && devices[j]==j;
    }
    if (!prefix && !openmx_magma_device_map_v1)
        return fail(OPENMX_MAGMA_UNAVAILABLE,"MAGMA dependency lacks selected-device support");
    /* Conservative bounds for the one-stage reduction, divide-and-conquer
       and backtransform scratch, including block-cyclic padding. They are
       not all live at once. Use the largest bound, plus a context reserve.
       Parent node-wide admission prevents competing dense owners racing
       these checks; unrelated external processes can still consume memory. */
    const size_t required = openmx_magma_mgpu_device_bytes(ngpu,n,width==16);
    for (int dev = 0; dev < ngpu; ++dev) {
        size_t free_bytes = 0, total = 0;
        if (hipSetDevice(devices[dev]) != hipSuccess || (OpenMX_GpuMemGetInfo ? OpenMX_GpuMemGetInfo(&free_bytes,&total) : hipMemGetInfo(&free_bytes,&total)) != hipSuccess)
            return fail(OPENMX_MAGMA_HIP_ERROR, "HIP memory preflight failed");
        if (free_bytes < required) return fail(OPENMX_MAGMA_RESOURCES, "insufficient per-GPU MAGMA scratch/reserve");
    }
    const int status=magma_library.ensure();
    if (status) return status;
    if (!prefix) {
        /* Main-thread MAGMA save/restore operations must begin inside the
           group. OpenMP workers observe the same immutable map. */
        if (hipSetDevice(devices[0])!=hipSuccess)
            return fail(OPENMX_MAGMA_HIP_ERROR,"cannot select MAGMA group owner");
        if (openmx_magma_device_map_v1(ngpu,devices))
            return fail(OPENMX_MAGMA_LIBRARY_ERROR,"MAGMA device map setup failed");
        session.mapped=true;
    } else if (hipSetDevice(session.previous)!=hipSuccess)
        return fail(OPENMX_MAGMA_HIP_ERROR,"cannot restore caller HIP device");
    return OPENMX_MAGMA_SUCCESS;
}

template<class T>
void order_eigenpairs(int n,int count,bool vectors,std::vector<T> &a,std::vector<double> &w)
{
    bool ordered=true;
    for (int j=0;j<count;++j) {
        /* Leave nonfinite results for validation; NaNs cannot participate
           in a strict weak ordering. */
        if (!std::isfinite(w[j])) return;
        if (j && w[j]<w[j-1]) ordered=false;
    }
    if (ordered) return;
    /* MAGMA's divide-and-conquer merge can return nearly degenerate values
       a few ulps out of order. Preserve each value/vector pair, then apply
       exactly the same residual and orthogonality validation as before.
       Permutation cycles require only one temporary column, not n*n. */
    if (vectors) {
        std::vector<int> order(count);
        for (int j=0;j<count;++j) order[j]=j;
        std::stable_sort(order.begin(),order.end(),[&](int i,int j) { return w[i]<w[j]; });
        std::vector<T> column(n);
        for (int first=0;first<count;++first) {
            if (order[first]==first) continue;
            std::copy_n(a.data()+size_t(first)*n,n,column.data());
            int to=first;
            while (order[to]!=first) {
                const int from=order[to];
                std::copy_n(a.data()+size_t(from)*n,n,a.data()+size_t(to)*n);
                order[to]=to;
                to=from;
            }
            std::copy_n(column.data(),n,a.data()+size_t(to)*n);
            order[to]=to;
        }
    }
    std::sort(w.begin(),w.begin()+count);
    std::fprintf(stderr,"<OpenMX_MAGMA> sorted returned eigenpairs n=%d count=%d before validation\n",n,count);
}

template<class T>
bool validate(const void *original, int lda, bool lower, int n, int count,
              const std::vector<T> &a, const std::vector<double> &w, bool vectors,
              double norm)
{
    for (int j = 0; j < count; ++j) {
        if (!std::isfinite(w[j]) || (j && w[j] < w[j-1])) {
            std::snprintf(magma_error,sizeof(magma_error),
                "MAGMA eigenvalue validation failed: index=%d value=%.17g previous=%.17g",
                j,w[j],j?w[j-1]:w[j]);
            return false;
        }
    }
    if (!vectors) return true;
    for (size_t i = 0; i < size_t(n)*count; ++i) if (!finite(a[i])) {
        std::snprintf(magma_error,sizeof(magma_error),"MAGMA nonfinite eigenvector entry: offset=%zu",i);
        return false;
    }

    /* Four evenly spaced eigenvectors and 32 spread rows, independent of
       MAGMA's reduction. Check mutual orthogonality over every vector row.
       This bounds normal runtime validation by O(n), besides the full finite
       scan, so a CPU oracle does not dominate a large GPU solve. The optional
       full-row check and independent analytic tests provide deeper coverage. */
    const int samples = std::min(count, 4);
    const char *full = std::getenv("OPENMX_MAGMA_VALIDATE_FULL");
    const int rows = full && std::strcmp(full,"1") == 0 ? n : std::min(n,32);
    int column[4];
    for (int s = 0; s < samples; ++s)
        column[s] = samples == 1 ? 0 : int(size_t(s)*(count-1)/(samples-1));
    std::vector<T> residual(size_t(rows)*samples, T(0));
    for (int j = 0; j < n; ++j) {
        T v[4];
        for (int s = 0; s < samples; ++s) v[s] = a[size_t(column[s])*n+j];
        for (int r = 0; r < rows; ++r) {
            const int i = rows == 1 ? 0 : int(size_t(r)*(n-1)/(rows-1));
            const T value = matrix<T>(original, lda, i, j, lower);
            for (int s = 0; s < samples; ++s) residual[size_t(s)*rows+r] += value*v[s];
        }
    }
    for (int s = 0; s < samples; ++s) {
        double rnorm = 0, vnorm = 0;
        for (int i = 0; i < n; ++i) {
            const T v = a[size_t(column[s])*n+i];
            vnorm = std::hypot(vnorm, std::abs(v));
        }
        for (int r = 0; r < rows; ++r) {
            const int i = rows == 1 ? 0 : int(size_t(r)*(n-1)/(rows-1));
            rnorm = std::hypot(rnorm, std::abs(residual[size_t(s)*rows+r] - w[column[s]]*a[size_t(column[s])*n+i]));
        }
        rnorm *= std::sqrt(double(n)/rows);
        const double denominator = (norm + std::abs(w[column[s]]))*vnorm;
        if (!std::isfinite(rnorm) || !std::isfinite(vnorm) || vnorm < 0.5 ||
            rnorm > 1e-10*std::max(denominator, std::numeric_limits<double>::min())) {
            std::snprintf(magma_error,sizeof(magma_error),
                "MAGMA residual validation failed: column=%d residual=%.17g scale=%.17g vector_norm=%.17g tolerance=1e-10",
                column[s],rnorm,denominator,vnorm);
            return false;
        }
        for (int t = 0; t <= s; ++t) {
            T dot = T(0);
            for (int i = 0; i < n; ++i)
                dot += conjugate(a[size_t(column[t])*n+i])*a[size_t(column[s])*n+i];
            if (std::abs(dot-T(s==t ? 1 : 0)) > 1e-10) {
                std::snprintf(magma_error,sizeof(magma_error),
                    "MAGMA orthogonality validation failed: columns=%d,%d error=%.17g tolerance=1e-10",
                    column[t],column[s],std::abs(dot-T(s==t ? 1 : 0)));
                return false;
            }
        }
    }
    return true;
}

template<class T>
int solve(int ngpu, const int *devices, int vectors, int lower, int n, int first, int last, void *A, int lda, double *W)
{
    Session session;
    int status = device_preflight(ngpu, devices, n, sizeof(T), session);
    if (status) return status;
    const int count = last-first+1;
    const magma_vec_t jobz = vectors ? MagmaVec : MagmaNoVec;
    const magma_range_t range = first == 1 && last == n ? MagmaRangeAll : MagmaRangeI;
    /* MAGMA's n==1 quick return writes the eigenpair but leaves m unset. */
    magma_int_t found = n == 1 ? 1 : 0, info = 0, liwork = -1;
    double real_query = 0, work_query = 0;
    magmaDoubleComplex complex_query{};
    const bool complex = sizeof(T) != sizeof(double);
    int result;
    /* n is checked before this query because upstream formulas use int. */
    if (complex) {
        result = magma_zheevdx_m(ngpu, jobz, range, MagmaLower, n, nullptr, n,
            0, 0, first, last, &found, nullptr, &complex_query, -1,
            &real_query, -1, &liwork, -1, &info);
        work_query = MAGMA_Z_REAL(complex_query);
    } else {
        result = magma_dsyevdx_m(ngpu, jobz, range, MagmaLower, n, nullptr, n,
            0, 0, first, last, &found, nullptr, &work_query, -1, &liwork, -1, &info);
    }
    const double imax = std::numeric_limits<magma_int_t>::max();
    if (result || info || !std::isfinite(work_query) || work_query < 1 || work_query > imax ||
        (complex && (!std::isfinite(real_query) || real_query < 1 || real_query > imax)) || liwork < 1)
        return fail(OPENMX_MAGMA_RESOURCES, "MAGMA workspace query failed or exceeded LP64 range");
    const int lwork = int(std::ceil(work_query));
    const int lrwork = complex ? int(std::ceil(real_query)) : 0;
    const size_t needed=sizeof(T)*(size_t(n)*n+size_t(lwork))+
        sizeof(double)*(size_t(n)+size_t(lrwork))+sizeof(magma_int_t)*size_t(liwork);
    const size_t limit=host_cache_limit();
    const bool retain=needed<=limit;
    /* Only one scalar type is retained, so the cap is per process, not per
       type/rank/device. Large solves use transient buffers and release any
       smaller cache first to avoid an unnecessary memory peak. */
    if (complex || !retain) real_workspace.clear();
    if (!complex || !retain) complex_workspace.clear();
    HostWorkspace<T> transient;
    HostWorkspace<T> &storage=retain?cached_workspace<T>():transient;
    const size_t retained=sizeof(T)*(storage.matrix.capacity()+storage.work.capacity())+
        sizeof(double)*(storage.eigenvalues.capacity()+storage.rwork.capacity())+
        sizeof(magma_int_t)*storage.iwork.capacity();
    const size_t after_growth=sizeof(T)*(std::max(storage.matrix.capacity(),size_t(n)*n)+
        std::max(storage.work.capacity(),size_t(lwork)))+
        sizeof(double)*(std::max(storage.eigenvalues.capacity(),size_t(n))+
        std::max(storage.rwork.capacity(),size_t(lrwork)))+
        sizeof(magma_int_t)*std::max(storage.iwork.capacity(),size_t(liwork));
    if (retain && (retained>limit || after_growth>limit)) storage.clear();
    resize_exact(storage.matrix,size_t(n)*n); resize_exact(storage.work,size_t(lwork));
    resize_exact(storage.eigenvalues,size_t(n)); resize_exact(storage.rwork,size_t(lrwork));
    resize_exact(storage.iwork,size_t(liwork));
    auto &copy=storage.matrix; auto &work=storage.work;
    auto &eigenvalues=storage.eigenvalues; auto &rwork=storage.rwork; auto &iwork=storage.iwork;
    std::fill(eigenvalues.begin(),eigenvalues.end(),std::numeric_limits<double>::quiet_NaN());
    double norm = 0;
    for (int j = 0; j < n; ++j) {
        double column_sum = 0;
        for (int i = 0; i < n; ++i) {
            const T value = matrix<T>(A, lda, i, j, lower != 0);
            if (!finite(value)) return fail(OPENMX_MAGMA_INVALID, "nonfinite input triangle");
            copy[size_t(j)*n+i] = value;
            /* A cheap upper bound avoids a hypot per dense complex entry;
               this only scales the independent residual tolerance. */
            column_sum += abs_bound(value);
        }
        norm = std::max(norm, column_sum);
    }
    if (!std::isfinite(norm)) return fail(OPENMX_MAGMA_INVALID, "input norm overflow");
    if (complex) {
        result = magma_zheevdx_m(ngpu, jobz, range, MagmaLower, n,
            reinterpret_cast<magmaDoubleComplex *>(copy.data()), n, 0, 0, first, last,
            &found, eigenvalues.data(), reinterpret_cast<magmaDoubleComplex *>(work.data()), lwork,
            rwork.data(), lrwork, iwork.data(), liwork, &info);
    } else {
        result = magma_dsyevdx_m(ngpu, jobz, range, MagmaLower, n,
            reinterpret_cast<double *>(copy.data()), n, 0, 0, first, last,
            &found, eigenvalues.data(), reinterpret_cast<double *>(work.data()), lwork,
            iwork.data(), liwork, &info);
    }
    /* All streams used internally must have finished before host results are
       accepted. Check every participating device, not whichever one MAGMA
       happened to leave selected after an early return. */
    for (int dev = 0; dev < ngpu; ++dev) {
        if (hipSetDevice(devices[dev]) != hipSuccess || hipDeviceSynchronize() != hipSuccess)
            return fail(OPENMX_MAGMA_HIP_ERROR, "MAGMA device execution failed");
    }
    if (result || info || found != count) {
        std::snprintf(magma_error, sizeof(magma_error), "MAGMA solve failed: status=%d info=%d eigenpairs=%d expected=%d",
                      result, int(info), int(found), count);
        return result==MAGMA_ERR_DEVICE_ALLOC || result==MAGMA_ERR_HOST_ALLOC ||
               info==MAGMA_ERR_DEVICE_ALLOC || info==MAGMA_ERR_HOST_ALLOC ?
               OPENMX_MAGMA_RESOURCES : OPENMX_MAGMA_LIBRARY_ERROR;
    }
    order_eigenpairs(n,count,vectors!=0,copy,eigenvalues);
    if (!validate<T>(A, lda, lower != 0, n, count, copy, eigenvalues, vectors != 0, norm)) {
        /* Optional failure evidence: six uint64 fields (magic, scalar bytes,
           n, eigenpair count, lower, vectors), then contiguous column-major
           input, returned vectors and eigenvalues. The failed solve never
           publishes either vectors or eigenvalues to its caller. */
        const char *dump=std::getenv("OPENMX_MAGMA_FAILURE_DUMP");
        if (dump && *dump) {
            FILE *fp=std::fopen(dump,"wb");
            bool ok=fp!=nullptr;
            const uint64_t header[]={0x4f4d584d41474d41ULL,sizeof(T),uint64_t(n),uint64_t(count),uint64_t(lower),uint64_t(vectors)};
            if (ok) ok=std::fwrite(header,sizeof(header),1,fp)==1;
            for (int j=0;j<n && ok;++j)
                ok=std::fwrite(static_cast<const unsigned char *>(A)+size_t(j)*lda*sizeof(T),sizeof(T),n,fp)==size_t(n);
            if (ok && vectors) ok=std::fwrite(copy.data(),sizeof(T),size_t(n)*count,fp)==size_t(n)*count;
            if (ok) ok=std::fwrite(eigenvalues.data(),sizeof(double),count,fp)==size_t(count);
            if (fp && std::fclose(fp)!=0) ok=false;
            std::fprintf(stderr,"<OpenMX_MAGMA> failure dump %s: %s\n",ok?"saved":"failed",dump);
        }
        return OPENMX_MAGMA_VALIDATION_ERROR;
    }
    if (session.mapped) {
        if (openmx_magma_device_map_v1(0,nullptr))
            return fail(OPENMX_MAGMA_LIBRARY_ERROR,"MAGMA device map cleanup failed");
        session.mapped=false;
    }
    if (hipSetDevice(session.previous) != hipSuccess)
        return fail(OPENMX_MAGMA_HIP_ERROR, "cannot restore caller HIP device after MAGMA");
    /* No fallible operations after publication. */
    if (vectors) for (int j = 0; j < count; ++j)
        std::memcpy(static_cast<unsigned char *>(A)+size_t(j)*lda*sizeof(T),
                    copy.data()+size_t(j)*n, size_t(n)*sizeof(T));
    std::memcpy(W, eigenvalues.data(), size_t(count)*sizeof(double));
    return OPENMX_MAGMA_SUCCESS;
}
} // namespace

extern "C" int openmx_magma_mgpu_available(void) { return 1; }
extern "C" int openmx_magma_mgpu_available_devices(int ngpu,const int *devices)
{
    if (!devices || ngpu<2 || ngpu>MagmaMaxGPUs) return 0;
    for (int i=0;i<ngpu;++i) if (devices[i]!=i) return openmx_magma_device_map_v1!=nullptr;
    return 1;
}
extern "C" void openmx_magma_mgpu_release_cache(void)
{
    std::lock_guard<std::mutex> lock(magma_mutex);
    real_workspace.clear(); complex_workspace.clear();
}
extern "C" int openmx_magma_mgpu_solve_host_devices(int ngpu, const int *devices, int is_complex, int want_vectors,
    int lower, int n, int first, int last, void *A, int lda, double *W)
{
    magma_error[0] = '\0';
    if (n == 0 && first == 1 && last == 0) return OPENMX_MAGMA_SUCCESS;
    /* Both real/complex drivers form 2*n*n+6*n+1 in LP64 arithmetic. */
    if (ngpu < 2 || ngpu > MagmaMaxGPUs || n < 1 || lda < n || first < 1 || last < first || last > n ||
        !A || !W || !devices || (is_complex != 0 && is_complex != 1) ||
        (want_vectors != 0 && want_vectors != 1) || (lower != 0 && lower != 1))
        return fail(OPENMX_MAGMA_INVALID, "invalid MAGMA host solve arguments");
    if (n > 32766)
        return fail(OPENMX_MAGMA_RESOURCES, "matrix exceeds MAGMA LP64 workspace range");
    try {
        std::lock_guard<std::mutex> lock(magma_mutex);
        return is_complex ? solve<Complex>(ngpu,devices,want_vectors,lower,n,first,last,A,lda,W)
                          : solve<double>(ngpu,devices,want_vectors,lower,n,first,last,A,lda,W);
    } catch (const std::bad_alloc &) {
        return fail(OPENMX_MAGMA_RESOURCES, "MAGMA host workspace allocation failed");
    } catch (...) {
        return fail(OPENMX_MAGMA_LIBRARY_ERROR, "unexpected MAGMA adapter exception");
    }
}
#else
extern "C" int openmx_magma_mgpu_available(void) { return 0; }
extern "C" int openmx_magma_mgpu_available_devices(int,const int*) { return 0; }
extern "C" void openmx_magma_mgpu_release_cache(void) {}
extern "C" int openmx_magma_mgpu_solve_host_devices(int, const int*, int, int, int, int n, int first, int last,
    void *, int, double *)
{
    magma_error[0] = '\0';
    if (n == 0 && first == 1 && last == 0) return OPENMX_MAGMA_SUCCESS;
    return fail(OPENMX_MAGMA_UNAVAILABLE, "OpenMX was built without MAGMA support");
}
#endif

extern "C" int openmx_magma_mgpu_solve_host(int ngpu,int is_complex,int want_vectors,
    int lower,int n,int first,int last,void *A,int lda,double *W)
{
    int devices[8];
    for (int i=0;i<8;++i) devices[i]=i;
    return openmx_magma_mgpu_solve_host_devices(ngpu,devices,is_complex,want_vectors,lower,n,first,last,A,lda,W);
}

extern "C" int openmx_magma_mgpu_dsyevdx_host(int ngpu,int n,int maxn,double *A,int lda,double *W)
{ return openmx_magma_mgpu_solve_host(ngpu,0,1,1,n,1,maxn,A,lda,W); }
extern "C" int openmx_magma_mgpu_zheevdx_host(int ngpu,int n,int maxn,void *A,int lda,double *W)
{ return openmx_magma_mgpu_solve_host(ngpu,1,1,1,n,1,maxn,A,lda,W); }
