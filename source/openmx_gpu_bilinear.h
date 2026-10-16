#ifndef OPENMX_GPU_BILINEAR_H
#define OPENMX_GPU_BILINEAR_H
#include <mpi.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Host-only descriptors. A span can describe either a contiguous/strided
   vector or one column of a jagged orbital table. No pointer crosses MPI. */
typedef struct {
    const void *data,*rows;
    size_t offset,stride;
    int count,width;
} OpenMXBilinearSpan;
typedef struct {
    OpenMXBilinearSpan left,right;
    const double *matrix;
    const void *matrix_rows[4];
    int scaled;
    double scale[4];
} OpenMXBilinearTerm;
typedef int (*OpenMXBilinearEmit)(void *,const OpenMXBilinearTerm *);
typedef struct {
    size_t rows;
    int spins,negate_fourth;
    void *context;
    int (*visit)(void *,size_t,OpenMXBilinearEmit,void *);
    void (*commit)(void *,size_t,size_t,const double *);
} OpenMXBilinearSource;

/* All ranks enter, including ones with completed work. Returns 0 with
   untouched outputs when capacity is exhausted; the caller runs its
   existing CPU quadrature. local_done < 0 explicitly disables GPU work. */
int openmx_gpu_bilinear_execute(MPI_Comm,OpenMXBilinearSource *,int local_done,int enabled,const char *label);

/* Internal wire format: offsets refer only to packed scalar arrays. */
typedef struct {uint64_t left,right,matrix;int n0,n1;double scale[4];} OpenMXBilinearIndex;
typedef struct {uint64_t rows,terms,orbitals,matrices;int spins,negate_fourth;} OpenMXBilinearHeader;
int openmx_gpu_bilinear_pack(const OpenMXBilinearSource *,size_t,size_t,void **,size_t *);

static inline double openmx_bilinear_value(const OpenMXBilinearSpan *s,int index)
{
    const void *p=s->data;size_t at=s->offset+(size_t)index*s->stride;
    if (s->rows) {
        memcpy(&p,(const unsigned char*)s->rows+(size_t)index*sizeof(void*),sizeof(p));
        at=s->offset;
    }
    if (s->width==4) {float v;memcpy(&v,(const unsigned char*)p+at*4,4);return v;}
    double v;memcpy(&v,(const unsigned char*)p+at*8,8);return v;
}
static inline double openmx_bilinear_matrix(const OpenMXBilinearTerm *t,int spin,int i,int j)
{
    if (t->matrix) return t->matrix[((size_t)spin*t->left.count+i)*t->right.count+j];
    const double *row;
    memcpy(&row,(const unsigned char*)t->matrix_rows[spin]+(size_t)i*sizeof(row),sizeof(row));
    return row[j];
}
#ifdef __cplusplus
}
#endif
#endif
