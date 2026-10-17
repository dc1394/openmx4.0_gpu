#include "openmx_gpu_bilinear.h"
#include "openmx_gpu_collective.h"
#include "openmx_gpu_workers.h"
#include "set_hip_default_device_from_local_rank.h"
#include "hip_runtime_compat.h"
#include <stdlib.h>

typedef struct {
    OpenMXBilinearHeader h;
    const uint64_t *ptr;
    const OpenMXBilinearIndex *terms;
    const double *orbitals,*matrices;
    double *result;
} PackedSource;

static int packed_open(const void *packet,size_t bytes,PackedSource *p)
{
    if (bytes<sizeof(p->h)) return 0;
    memcpy(&p->h,packet,sizeof(p->h));
    const OpenMXBilinearHeader *h=&p->h;
    if ((h->spins!=1 && h->spins!=2 && h->spins!=4) || h->rows>SIZE_MAX-1 ||
        h->rows>SIZE_MAX/(4*sizeof(double))) return 0;
    const unsigned char *base=packet;size_t off=sizeof(*h);
#define FIELD(member,count,type) do { \
    if ((count)>(SIZE_MAX-off)/sizeof(type)) return 0; \
    size_t next=off+(size_t)(count)*sizeof(type);if (next>bytes) return 0; \
    p->member=(const type*)(base+off);off=next; \
} while (0)
    FIELD(ptr,h->rows+1,uint64_t);FIELD(terms,h->terms,OpenMXBilinearIndex);
    FIELD(orbitals,h->orbitals,double);FIELD(matrices,h->matrices,double);
#undef FIELD
    if (off!=bytes || p->ptr[0] || p->ptr[h->rows]!=h->terms) return 0;
    for (size_t i=0;i<h->rows;i++) if (p->ptr[i]>p->ptr[i+1]) return 0;
    for (size_t i=0;i<h->terms;i++) {
        const OpenMXBilinearIndex *t=p->terms+i;
        if (t->n0<1 || t->n1<1 || t->left>h->orbitals || (uint64_t)t->n0>h->orbitals-t->left ||
            t->right>h->orbitals || (uint64_t)t->n1>h->orbitals-t->right ||
            t->matrix>h->matrices || (uint64_t)t->n0>(h->matrices-t->matrix)/(unsigned)t->n1/(unsigned)h->spins) return 0;
    }
    return 1;
}

static void bilinear_kernel(const PackedSource *host,const void *device,double *output)
{
    const unsigned char *base=device;const size_t n=host->h.rows,nt=host->h.terms;
    const int spins=host->h.spins,negative=host->h.negate_fourth;
    const uint64_t *ptr=(const uint64_t*)(base+sizeof(OpenMXBilinearHeader));
    const OpenMXBilinearIndex *terms=(const OpenMXBilinearIndex*)(ptr+n+1);
    const double *orb=(const double*)(terms+nt),*dm=orb+host->h.orbitals;
#pragma omp target teams distribute parallel for thread_limit(128) is_device_ptr(ptr,terms,orb,dm,output)
    for (size_t row=0;row<n;row++) {
        double sums[4]={0,0,0,0};

        for (uint64_t k=ptr[row];k<ptr[row+1];k++) {
            OpenMXBilinearIndex t=terms[k];

            for (int spin=0;spin<spins;spin++) {
                double value=0;

                for (int i=0;i<t.n0;i++) {
                    double inner=0;

                    for (int j=0;j<t.n1;j++) inner+=orb[t.right+j]*dm[t.matrix+((size_t)spin*t.n0+i)*t.n1+j];
                    value+=orb[t.left+i]*inner;
                }
                sums[spin]+=value*t.scale[spin];
            }
        }

        for (int spin=0;spin<spins;spin++) output[row*(size_t)spins+spin]=(negative && spin==3)?-sums[spin]:sums[spin];
    }
}

typedef struct {
    OpenMXBilinearSource *source;
    OpenMXGpuWorkers pool;
    int planned;
    void *packet[OPENMX_GPU_WORKERS_MAX],*device[OPENMX_GPU_WORKERS_MAX];
    size_t bytes[OPENMX_GPU_WORKERS_MAX];
    double *result[OPENMX_GPU_WORKERS_MAX];
} BilinearWork;

static size_t bilinear_cost(void *context,int worker,int count)
{
    BilinearWork *w=context;
    if (w->planned!=count) {
        for (int i=0;i<OPENMX_GPU_WORKERS_MAX;i++) {free(w->packet[i]);w->packet[i]=NULL;w->bytes[i]=0;}
        w->planned=count;
    }
    size_t lo=openmx_gpu_worker_cut(w->source->rows,worker,count),hi=openmx_gpu_worker_cut(w->source->rows,worker+1,count);
    if (!w->packet[worker] && !openmx_gpu_bilinear_pack(w->source,lo,hi,&w->packet[worker],&w->bytes[worker])) return SIZE_MAX;
    size_t nb=w->bytes[worker];
    if (hi-lo>(SIZE_MAX-nb)/sizeof(double)/(unsigned)w->source->spins) return SIZE_MAX;
    return nb+(hi-lo)*(size_t)w->source->spins*sizeof(double);
}
static int bilinear_worker(void *context,int worker,int phase)
{
    BilinearWork *w=context;const size_t bytes=w->bytes[worker];
    PackedSource p={0};
    if (!packed_open(w->packet[worker],bytes,&p)) return 0;
    const size_t outputs=(size_t)p.h.rows*p.h.spins*sizeof(double);
    if (phase==OPENMX_GPU_WORKER_PREPARE) {
        w->result[worker]=malloc(outputs?outputs:1);
        if (!w->result[worker] || hipMalloc(&w->device[worker],bytes+outputs)!=hipSuccess) return 0;
        return hipMemcpy(w->device[worker],w->packet[worker],bytes,hipMemcpyHostToDevice)==hipSuccess;
    }
    if (phase==OPENMX_GPU_WORKER_EXECUTE)
        bilinear_kernel(&p,w->device[worker],(double*)((unsigned char*)w->device[worker]+bytes));
    else if (phase==OPENMX_GPU_WORKER_COMMIT)
        return hipMemcpy(w->result[worker],(unsigned char*)w->device[worker]+bytes,outputs,hipMemcpyDeviceToHost)==hipSuccess;
    else if (phase==OPENMX_GPU_WORKER_RELEASE) {
        if (w->device[worker] && hipFree(w->device[worker])!=hipSuccess) return 0;
    }
    return 1;
}
static int bilinear_local(OpenMXBilinearSource *source,const char *label)
{
    if (!source->rows) return 1;
    BilinearWork w={0};w.source=source;
    OpenMXGpuWorkerRetry retry={bilinear_cost,&w,{0}};
    int ok=0;
    while (openmx_gpu_workers_capacity(&w.pool,source->rows,1,openmx_gpu_worker_retry_cost,&retry)) {
        ok=1;
        /* A rejected larger forced group can leave a different cached layout.
           Reconfirm the final count before launching any worker. */
        if (ok) for (int i=0;i<w.pool.count;i++) if (bilinear_cost(&w,i,w.pool.count)==SIZE_MAX) ok=0;
        if (ok) ok=openmx_gpu_workers_run(&w.pool,bilinear_worker,&w,label);
        if (ok) for (int i=0;i<w.pool.count;i++) source->commit(source->context,
            openmx_gpu_worker_cut(source->rows,i,w.pool.count),openmx_gpu_worker_cut(source->rows,i+1,w.pool.count),w.result[i]);
        for (int i=0;i<OPENMX_GPU_WORKERS_MAX;i++) {free(w.result[i]);w.result[i]=NULL;w.device[i]=NULL;}
        if (ok) break;
        retry.rejected[w.pool.count]=1;
    }
    for (int i=0;i<OPENMX_GPU_WORKERS_MAX;i++) free(w.packet[i]);
    return ok;
}

static int packed_visit(void *context,size_t row,OpenMXBilinearEmit emit,void *target)
{
    PackedSource *p=context;
    for (uint64_t k=p->ptr[row];k<p->ptr[row+1];k++) {
        const OpenMXBilinearIndex *i=p->terms+k;OpenMXBilinearTerm t={0};
        t.left=(OpenMXBilinearSpan){p->orbitals,NULL,i->left,1,i->n0,8};
        t.right=(OpenMXBilinearSpan){p->orbitals,NULL,i->right,1,i->n1,8};
        t.matrix=p->matrices+i->matrix;t.scaled=1;
        memcpy(t.scale,i->scale,sizeof(t.scale));
        if (!emit(target,&t)) return 0;
    }
    return 1;
}
static void packed_commit(void *context,size_t first,size_t last,const double *result)
{
    PackedSource *p=context;memcpy(p->result+first*(size_t)p->h.spins,result,(last-first)*(size_t)p->h.spins*sizeof(double));
}
static int bilinear_pack(void *context,size_t first,size_t last,void **packet,size_t *bytes)
{return openmx_gpu_bilinear_pack(context,first,last,packet,bytes);}
static int bilinear_remote(void *context,const void *packet,size_t bytes,void **result,size_t *result_bytes)
{
    (void)context;PackedSource p={0};if (!packed_open(packet,bytes,&p)) return 0;
    *result_bytes=(size_t)p.h.rows*p.h.spins*sizeof(double);
    *result=malloc(*result_bytes?*result_bytes:1);if (!*result) return 0;
    p.result=*result;
    OpenMXBilinearSource source={p.h.rows,p.h.spins,p.h.negate_fourth,&p,packed_visit,packed_commit};
    return bilinear_local(&source,"density-remote-shard");
}
static void bilinear_commit(void *context,size_t first,size_t last,const void *result,size_t bytes)
{
    (void)bytes;OpenMXBilinearSource *source=context;source->commit(source->context,first,last,result);
}
int openmx_gpu_bilinear_execute(MPI_Comm comm,OpenMXBilinearSource *source,int local_done,int enabled,const char *label)
{
    int borrow=enabled && openmx_gpu_workers_capacity_enabled();
    int usable=enabled && (gpu_rank_device_usable() || borrow);
    int done=local_done>0 || !source->rows || (local_done==0 && usable && bilinear_local(source,label));
    static const OpenMXGpuCollectiveOps ops={bilinear_pack,bilinear_remote,bilinear_commit};
    return openmx_gpu_collective_retry(comm,&ops,source,source->rows,
        done?1:local_done<0 || !borrow?-1:0,borrow && usable,label);
}
