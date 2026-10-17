/*
 * Minimal explicit-instantiation translation unit of GEMMul8 for OpenMX.
 *
 * Upstream GEMMul8 builds one object per (routine, type, backend)
 * combination from the cu_recipe sources via its own make system.
 * OpenMX only needs the INT8-backend gemm for double and hipDoubleComplex
 * plus the matching workSize queries, so this single TU includes the
 * template definitions and instantiates just those four entry points.
 *
 * Compile with -I<GEMMul8>/include -I<GEMMul8>/src.
 */
#include "gemm/gemm_impl.hpp"
#include "worksize/worksize_impl.hpp"
/* the MAGMA Ozaki-II adapter (third_party/magma, interface_hip/ozaki2.cpp)
   additionally routes symm/hemm, syr2k/her2k and trmm */
#include "symm/symm_impl.hpp"
#include "hemm/hemm_impl.hpp"
#include "syr2k/syr2k_impl.hpp"
#include "her2k/her2k_impl.hpp"
#include "trmm/trmm_impl.hpp"

/*
 * The include chain above only declares the device-kernel launchers
 * (see *_declaration.hpp); upstream compiles their definitions as
 * separate per-instantiation objects.  Include the definition headers
 * here so the explicit instantiations below reach them implicitly.
 */
#include "oz2/mod/mod_hi2mid.hpp"
#include "oz2/mod/mod_reduce_matprod.hpp"
#include "oz2/scaling/fast/scaling.hpp"
#include "oz2/scaling/accu/extract.hpp"
#include "oz2/scaling/accu/scaling.hpp"
#include "oz2/scaling/general/scaling_rowwise.hpp"
#include "oz2/scaling/general/scaling_colwise.hpp"
#include "oz2/scaling/accu/extract_symm_hemm.hpp"
#include "oz2/scaling/accu/scaling_symm_hemm.hpp"
#include "oz2/scaling/accu/scaling_syrk_herk.hpp"
#include "oz2/scaling/fast/scaling_symm_hemm.hpp"
#include "oz2/scaling/general/scaling_symm_hemm.hpp"
#include "oz2/undo_scaling/undo_scaling.hpp"
#include "oz2/undo_scaling/undo_scaling_syr2k_her2k.hpp"

namespace gemmul8 {

template std::vector<double> gemm<double, Backend::INT8, double, double>(
    hipblasHandle_t,
    hipblasOperation_t, hipblasOperation_t,
    size_t, size_t, size_t,
    const double *,
    const double *const, size_t,
    const double *const, size_t,
    const double *,
    double *const, size_t,
    int, bool,
    void *const,
    void *const,
    void *const,
    bool, bool, bool, bool);

template std::vector<double> gemm<hipDoubleComplex, Backend::INT8, hipDoubleComplex, hipDoubleComplex>(
    hipblasHandle_t,
    hipblasOperation_t, hipblasOperation_t,
    size_t, size_t, size_t,
    const hipDoubleComplex *,
    const hipDoubleComplex *const, size_t,
    const hipDoubleComplex *const, size_t,
    const hipDoubleComplex *,
    hipDoubleComplex *const, size_t,
    int, bool,
    void *const,
    void *const,
    void *const,
    bool, bool, bool, bool);

template std::vector<double> symm<double, Backend::INT8, double, double>(
    hipblasHandle_t,
    hipblasSideMode_t, hipblasFillMode_t,
    size_t, size_t,
    const double *,
    const double *const, size_t,
    const double *const, size_t,
    const double *,
    double *const, size_t,
    int, bool,
    void *const, void *const, void *const,
    bool, bool, bool, bool);

template std::vector<double> hemm<hipDoubleComplex, Backend::INT8, hipDoubleComplex, hipDoubleComplex>(
    hipblasHandle_t,
    hipblasSideMode_t, hipblasFillMode_t,
    size_t, size_t,
    const hipDoubleComplex *,
    const hipDoubleComplex *const, size_t,
    const hipDoubleComplex *const, size_t,
    const hipDoubleComplex *,
    hipDoubleComplex *const, size_t,
    int, bool,
    void *const, void *const, void *const,
    bool, bool, bool, bool);

template std::vector<double> syr2k<double, Backend::INT8, double, double>(
    hipblasHandle_t,
    hipblasFillMode_t, hipblasOperation_t,
    size_t, size_t,
    const double *,
    const double *const, size_t,
    const double *const, size_t,
    const double *,
    double *const, size_t,
    int, bool,
    void *const, void *const, void *const,
    bool, bool, bool, bool);

template std::vector<double> her2k<hipDoubleComplex, Backend::INT8, hipDoubleComplex, hipDoubleComplex>(
    hipblasHandle_t,
    hipblasFillMode_t, hipblasOperation_t,
    size_t, size_t,
    const hipDoubleComplex *,
    const hipDoubleComplex *const, size_t,
    const hipDoubleComplex *const, size_t,
    const double *,
    hipDoubleComplex *const, size_t,
    int, bool,
    void *const, void *const, void *const,
    bool, bool, bool, bool);

template std::vector<double> trmm<double, Backend::INT8, double, double>(
    hipblasHandle_t,
    hipblasSideMode_t, hipblasFillMode_t,
    hipblasOperation_t, hipblasDiagType_t,
    size_t, size_t,
    const double *,
    const double *const, size_t,
    const double *const, size_t,
    double *const, size_t,
    int, bool,
    void *const, void *const, void *const,
    bool, bool, bool, bool);

template std::vector<double> trmm<hipDoubleComplex, Backend::INT8, hipDoubleComplex, hipDoubleComplex>(
    hipblasHandle_t,
    hipblasSideMode_t, hipblasFillMode_t,
    hipblasOperation_t, hipblasDiagType_t,
    size_t, size_t,
    const hipDoubleComplex *,
    const hipDoubleComplex *const, size_t,
    const hipDoubleComplex *const, size_t,
    hipDoubleComplex *const, size_t,
    int, bool,
    void *const, void *const, void *const,
    bool, bool, bool, bool);

template size_t workSize<false, Backend::INT8, Func::gemm>(
    size_t, size_t, size_t, int, bool, bool, size_t *, size_t *, bool);

template size_t workSize<true, Backend::INT8, Func::gemm>(
    size_t, size_t, size_t, int, bool, bool, size_t *, size_t *, bool);

template size_t workSize<false, Backend::INT8, Func::symm>(
    size_t, size_t, size_t, int, bool, bool, size_t *, size_t *, bool);

template size_t workSize<true, Backend::INT8, Func::hemm>(
    size_t, size_t, size_t, int, bool, bool, size_t *, size_t *, bool);

template size_t workSize<false, Backend::INT8, Func::syr2k>(
    size_t, size_t, size_t, int, bool, bool, size_t *, size_t *, bool);

template size_t workSize<true, Backend::INT8, Func::her2k>(
    size_t, size_t, size_t, int, bool, bool, size_t *, size_t *, bool);

template size_t workSize<false, Backend::INT8, Func::trmm>(
    size_t, size_t, size_t, int, bool, bool, size_t *, size_t *, bool);

template size_t workSize<true, Backend::INT8, Func::trmm>(
    size_t, size_t, size_t, int, bool, bool, size_t *, size_t *, bool);

} // namespace gemmul8
