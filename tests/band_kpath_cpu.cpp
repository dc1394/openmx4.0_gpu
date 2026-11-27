#include "openmx_band_kpath.h"
#include "openmx_core_host_simd.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <omp.h>
#include <vector>

namespace {
constexpr auto unlimited = std::numeric_limits<size_t>::max();
constexpr double pi = 3.14159265358979323846;
using Complex = std::complex<double>;

bool close(double a, double b)
{
    const bool ok = std::isfinite(a) && std::abs(a-b)<2e-11;
    if (!ok) std::fprintf(stderr,"CPU band spectrum %.17g, expected %.17g\n",a,b);
    return ok;
}

bool two_bands(const double* eigen, double a, double b, double off_diagonal)
{
    const double center = (a+b)/2, gap = std::hypot((a-b)/2,off_diagonal);
    return close(eigen[0],center-gap) && close(eigen[1],center+gap);
}

bool generalized_bands(const double* eigen, double a, double b, Complex h, Complex s)
{
    // det(H - lambda*S) for an overlap with diagonals 1 and 4.
    const double quadratic = 4-std::norm(s);
    const double linear = 4*a+b-2*std::real(h*std::conj(s));
    const double constant = a*b-std::norm(h);
    const double center = linear/(2*quadratic);
    const double gap = std::sqrt(center*center-constant/quadratic);
    return close(eigen[0],center-gap) && close(eigen[1],center+gap);
}

bool generalized_vectors(const double* eigen, const double* vectors, double a, double b, Complex h, Complex s)
{
    for (int state=0; state<2; ++state) {
        const double* column=vectors+4*state;
        const Complex x(column[0],column[1]), y(column[2],column[3]);
        const Complex sx=x+s*y, sy=std::conj(s)*x+4.0*y;
        const Complex hx=a*x+h*y, hy=std::conj(h)*x+b*y;
        if (!close(std::abs(hx-eigen[state]*sx),0.0) || !close(std::abs(hy-eigen[state]*sy),0.0))
            return false;
        for (int other=0; other<2; ++other) {
            const double* q=vectors+4*other;
            const Complex product=std::conj(Complex(q[0],q[1]))*sx+std::conj(Complex(q[2],q[3]))*sy;
            if (!close(std::abs(product-double(other==state)),0.0)) return false;
        }
    }
    return true;
}

bool spin_and_exchange(int mode, int expected_workers)
{
    const OpenMXKPathBlock blocks[] = {
        {0,0,2,2,{0,0,0},0}, {0,1,1,1,{1,0,0},12}, {1,0,1,1,{-1,0,0},15},
        {0,1,1,1,{1,0,0},18}, {1,0,1,1,{-1,0,0},20}
    };
    // S(0), H_up(0), H_down(0), two hopping images, then two K(R) images.
    const double input[] = {1,.2,.2,4, 2,.5,.5,12, -2,.5,.5,20, .1,.25,.25, .1,.25,.25,
                            .5,-.125, .5,-.125};
    const OpenMXKPathMatrices m = {2,2,0,0,0,1,5,22,18,1e-9,blocks,input};
    OpenMXKPathPoint points[7]{};
    for (int k=0; k<7; ++k) points[k].xyz[0] = double(k)/7;
    double eigen[7][4];
    double vectors[7][16];
    int workers=0;
    if (!openmx_band_mo_vectors_cpu(&m,7,points,eigen[0],vectors[0],mode,unlimited,&workers) || workers!=expected_workers)
        return false;
    for (int k=0; k<7; ++k) {
        const auto phase = std::polar(1.0,2*pi*points[k].xyz[0]);
        const auto overlap = .2+.1*phase;
        const auto up=.5+.75*phase, down=.5+.125*phase;
        if (!generalized_bands(eigen[k],2,12,up,overlap) || !generalized_bands(eigen[k]+2,-2,20,down,overlap) ||
            !generalized_vectors(eigen[k],vectors[k],2,12,up,overlap) ||
            !generalized_vectors(eigen[k]+2,vectors[k]+8,-2,20,down,overlap)) return false;
    }
    return true;
}

bool noncollinear(int mode)
{
    const OpenMXKPathBlock blocks[] = {{0,0,1,1,{0,0,0},0}, {0,0,1,1,{1,0,0},7}, {0,0,1,1,{-1,0,0},14}};
    const double input[] = {2,2,8,-.8,.2,0,0, 0,0,0,.4,.3,.25,-.5, 0,0,0,0,0,-.25,.5};
    const OpenMXKPathMatrices m = {1,1,1,0,1,1,3,21,21,1e-9,blocks,input};
    const OpenMXKPathPoint point = {{.25,0,0},0};
    double eigen[2];
    int workers;
    return openmx_band_kpath_solve_cpu(&m,1,&point,eigen,mode,unlimited,&workers) &&
           two_bands(eigen,.75,4.5,std::abs(Complex(-1.1,.6))/2);
}

bool overlap_penalty(double s, double h, bool lno, double low, double high)
{
    const OpenMXKPathBlock block = {0,0,2,2,{0,0,0},0};
    const double input[] = {s,0,0,1, h,0,0,3};
    const OpenMXKPathMatrices m = {2,1,0,int(lno),0,1,1,8,8,1e-9,&block,input};
    const OpenMXKPathPoint point{};
    double eigen[2];
    int workers;
    return openmx_band_kpath_solve_cpu(&m,1,&point,eigen,OPENMX_GRID_SCALAR,unlimited,&workers) &&
           close(eigen[0],low) && close(eigen[1],high);
}

bool large_diagonal()
{
    constexpr int n = 129;
    const OpenMXKPathBlock block = {0,0,n,n,{0,0,0},0};
    std::vector<double> input(2*n*n,0.0);
    for (int i=0; i<n; ++i) {
        input[i*(n+1)] = 2.0;
        input[n*n+i*(n+1)] = 2.0*i/n;
    }
    const OpenMXKPathMatrices m = {n,1,0,0,0,1,1,2*n*n,2*n*n,1e-9,&block,input.data()};
    const OpenMXKPathPoint point{};
    std::array<double,n> eigen;
    int workers;
    if (openmx_band_kpath_solve_cpu(&m,1,&point,eigen.data(),OPENMX_GRID_SCALAR,1,&workers) ||
        !openmx_band_kpath_solve_cpu(&m,1,&point,eigen.data(),OPENMX_GRID_SCALAR,unlimited,&workers)) return false;
    for (int i=0; i<n; ++i)
        if (!close(eigen[i],double(i)/n)) return false;
    return true;
}
}

extern "C" int openmx_test_cpu_kpath(void)
{
    const int saved_threads = omp_get_max_threads(), saved_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    omp_set_num_threads(2);
    bool ok = true;
    for (int mode : {OPENMX_GRID_SCALAR,OPENMX_GRID_AVX2,OPENMX_GRID_AVX512}) {
        if (openmx_core_host_simd_usable(mode))
            ok = spin_and_exchange(mode,2) && noncollinear(mode) && ok;
    }
    omp_set_num_threads(saved_threads);
    omp_set_dynamic(saved_dynamic);
    return ok && overlap_penalty(5e-10,1e-9,false,3,5) && overlap_penalty(1e-11,2e-11,false,3,10000) &&
           overlap_penalty(-1e-6,2e-14,false,3,10000) && overlap_penalty(0.5,1000,false,3,10000) &&
           overlap_penalty(5e-10,1e-9,true,2,3) && large_diagonal();
}
