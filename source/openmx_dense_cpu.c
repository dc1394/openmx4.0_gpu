#include "openmx_dense_cpu.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { double r, i; } Z;
extern void dsyevr_(const char*,const char*,const char*,const int*,double*,const int*,
    const double*,const double*,const int*,const int*,const double*,int*,double*,double*,
    const int*,int*,double*,const int*,int*,const int*,int*);
extern void zheevr_(const char*,const char*,const char*,const int*,Z*,const int*,
    const double*,const double*,const int*,const int*,const double*,int*,double*,Z*,
    const int*,int*,Z*,const int*,double*,const int*,int*,const int*,int*);
extern void dgemm_(const char*,const char*,const int*,const int*,const int*,const double*,
    const double*,const int*,const double*,const int*,const double*,double*,const int*);
extern void zgemm_(const char*,const char*,const int*,const int*,const int*,const Z*,
    const Z*,const int*,const Z*,const int*,const Z*,Z*,const int*);

static void *allocate(size_t count, size_t width)
{
    return count <= SIZE_MAX/width ? malloc((count ? count : 1)*width) : NULL;
}
static int work_length(double query)
{
    return isfinite(query) && query>=1.0 && query<=INT_MAX ? (int)ceil(query) : 0;
}

int openmx_dense_cpu_eigen(int n,int nev,int is_complex,void *a,int lda,double *w)
{
    const char job='V',range='I',lower='L';
    const int first=1;
    const double zero=0;
    const size_t width=is_complex ? sizeof(Z) : sizeof(double);
    int found=0,info=0,lwork=-1,lrwork=-1,liwork=-1,iq=0,ok=0;
    Z query={0,0}; double rq=0;
    void *copy=NULL,*vectors=NULL,*work=NULL;
    double *values=NULL,*rwork=NULL;
    int *support=NULL,*iwork=NULL;
    if (n<0 || nev<0 || nev>n || lda<(n ? n : 1)) return 0;
    if (n==0 || nev==0) return 1;
    if (!a || !w || (size_t)n>SIZE_MAX/(size_t)n/width || n>INT_MAX/26) return 0;
    /* Workspace query does not access matrices. Keep a real scalar in each
       position rather than a null pointer for vendor LAPACK implementations. */
    if (is_complex)
        zheevr_(&job,&range,&lower,&n,&query,&n,&zero,&zero,&first,&nev,&zero,&found,
            &rq,&query,&n,&iq,&query,&lwork,&rq,&lrwork,&iq,&liwork,&info);
    else
        dsyevr_(&job,&range,&lower,&n,&query.r,&n,&zero,&zero,&first,&nev,&zero,&found,
            &rq,&query.r,&n,&iq,&query.r,&lwork,&iq,&liwork,&info);
    lwork=work_length(query.r); lrwork=is_complex ? work_length(rq) : 1; liwork=iq;
    if (info || !lwork || !lrwork || liwork<1) return 0;
    copy=allocate((size_t)n*n,width); vectors=allocate((size_t)n*nev,width);
    values=allocate(n,sizeof(double)); support=allocate((size_t)2*n,sizeof(int));
    work=allocate(lwork,width); rwork=allocate(lrwork,sizeof(double));
    iwork=allocate(liwork,sizeof(int));
    if (!copy || !vectors || !values || !support || !work || !rwork || !iwork) goto done;
    /* Only the lower triangle is initialized in several callers. */
    for (int j=0;j<n;j++)
        memcpy((char*)copy+((size_t)j*n+j)*width,
               (const char*)a+((size_t)j*lda+j)*width,(size_t)(n-j)*width);
    if (is_complex)
        zheevr_(&job,&range,&lower,&n,copy,&n,&zero,&zero,&first,&nev,&zero,&found,
            values,vectors,&n,support,work,&lwork,rwork,&lrwork,iwork,&liwork,&info);
    else
        dsyevr_(&job,&range,&lower,&n,copy,&n,&zero,&zero,&first,&nev,&zero,&found,
            values,vectors,&n,support,work,&lwork,iwork,&liwork,&info);
    if (info || found!=nev) goto done;
    for (int j=0;j<nev;j++) {
        if (!isfinite(values[j]) || (j && values[j]<values[j-1])) goto done;
    }
    for (size_t i=0;i<(size_t)n*nev*(is_complex ? 2 : 1);i++)
        if (!isfinite(((double*)vectors)[i])) goto done;
    for (int j=0;j<nev;j++)
        memcpy((char*)a+(size_t)j*lda*width,(char*)vectors+(size_t)j*n*width,(size_t)n*width);
    memcpy(w,values,(size_t)nev*sizeof(double));
    ok=1;
done:
    free(copy); free(vectors); free(values); free(support); free(work); free(rwork); free(iwork);
    return ok;
}

int openmx_dense_cpu_gemm(char ta,char tb,int is_complex,int m,int n,int k,
    const void *a,int lda,const void *b,int ldb,void *c,int ldc)
{
    const Z one={1,0},zero={0,0};
    if ((ta!='N' && ta!='T' && ta!='C') || (tb!='N' && tb!='T' && tb!='C') ||
        m<0 || n<0 || k<0 || lda<1 || ldb<1 || ldc<1 ||
        lda<(ta=='N'?m:k) || ldb<(tb=='N'?k:n) || ldc<m) return 0;
    if (!m || !n) return 1;
    if (!c || (k && (!a || !b))) return 0;
    /* Avoid passing null inputs to a BLAS implementation even for k==0. */
    if (!k) {
        for (int j=0;j<n;j++) memset((char*)c+(size_t)j*ldc*(is_complex?sizeof(Z):sizeof(double)),
                                    0,(size_t)m*(is_complex?sizeof(Z):sizeof(double)));
        return 1;
    }
    if (is_complex) zgemm_(&ta,&tb,&m,&n,&k,&one,a,&lda,b,&ldb,&zero,c,&ldc);
    else dgemm_(&ta,&tb,&m,&n,&k,&one.r,a,&lda,b,&ldb,&zero.r,c,&ldc);
    return 1;
}
