#if defined(nosse) || !defined(__SSE2__)
#include "mimic_sse.h"

/* Preserve the Krylov kernel's two-lane arithmetic on non-SSE2 hosts.
   These definitions were missing from the original nosse implementation. */
__m128d _mm_setzero_pd(void) { return (__m128d){{0.0,0.0}}; }
__m128d _mm_add_pd(__m128d a, __m128d b) {
  return (__m128d){{a.m128d_f64[0]+b.m128d_f64[0],a.m128d_f64[1]+b.m128d_f64[1]}};
}
__m128d _mm_mul_pd(__m128d a, __m128d b) {
  return (__m128d){{a.m128d_f64[0]*b.m128d_f64[0],a.m128d_f64[1]*b.m128d_f64[1]}};
}
__m128d _mm_loadu_pd(const double* p) { return (__m128d){{p[0],p[1]}}; }
__m128d _mm_load_pd(const double* p) { return _mm_loadu_pd(p); }
void _mm_storeu_pd(double* p, __m128d a) { p[0]=a.m128d_f64[0];p[1]=a.m128d_f64[1]; }
void _mm_store_pd(double* p, __m128d a) { _mm_storeu_pd(p,a); }

#endif
