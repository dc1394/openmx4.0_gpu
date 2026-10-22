#ifndef OPENMX_FORCE_RECOVERY_H
#define OPENMX_FORCE_RECOVERY_H
#include "openmx_core_force_recovery.h"

/* The SDK kernels keep their backend-specific reduction policies. Packing and
   MPI participation are shared; each synchronous dispatch borrows one C++
   packet whose storage is destroyed before the next recovery collective. */
static void ForceRecovery_Failure(const char *label,const char *error)
{
    char message[512];
    snprintf(message,sizeof(message),"%s: cannot allocate or execute a bounded host recovery packet: %s",label,error);
    Force4B_gpu_abort(message);
}

static void Force4B_capacity_copy(float *dest,const float *host,const float *device,size_t offset,size_t count)
{
    if(!count)return;
    if(host)memcpy(dest,host+offset,count*sizeof(float));
    else if(OpenMXGpuMemcpy(dest,device+offset,count*sizeof(float),OPENMX_GPU_COPY_DEVICE_TO_HOST)!=OPENMX_GPU_SUCCESS)
        Force4B_gpu_abort("Force4B: cannot read an existing projector archive.");
}

typedef struct {
    int kind;
    Force4BGpuContext *g;
    size_t first;
    Force4BGpuItem *items;
    size_t *a,*b;
    int *halo;
    double *cdm,*output;
    MPI_Comm comm;
    Force4BCapacityParameters parameters;
} ForceVnaRecoveryWork;

static int Force4B_RecoveryPacket(const OpenMXForceVnaRecoveryView *packet,void *opaque)
{
    ForceVnaRecoveryWork *work=(ForceVnaRecoveryWork*)opaque;
    const int kind=work->kind;Force4BGpuContext *g=work->g;
    size_t *offset=packet->offset,*coff=packet->density_offset;
    float *data=packet->projectors;double *weights=packet->density;
    size_t at=0,ct=0;offset[0]=coff[0]=0;
    for(size_t i=0;i<packet->items;i++){
        size_t q=work->first+i;Force4BGpuItem it=work->items[q];size_t mn=(size_t)it.ian*it.jan;
        memcpy(weights+ct,work->cdm+it.cdm_off,mn*sizeof(double));ct+=mn;
        if(kind==1)for(int k=it.k_off;k<it.k_off+it.k_n;k++){
            for(int dir=1;dir<4;dir++){
                size_t num=(size_t)it.ian*g->num_proj;
                Force4B_capacity_copy(data+at,g->flat_host,g->flat_dev,dir*g->flat_stride+work->a[k],num);at+=num;
            }
            size_t num=(size_t)it.jan*g->num_proj;
            Force4B_capacity_copy(data+at,work->halo[k]?g->halo_host:g->flat_host,
                work->halo[k]?g->halo_dev:g->flat_dev,work->b[k],num);at+=num;
        }else{
            for(int dir=0;dir<4;dir++){
                size_t num=(size_t)it.ian*g->num_proj;
                Force4B_capacity_copy(data+at,g->c2_host,g->c2_dev,dir*g->c2_stride+work->a[q],num);at+=num;
            }
            for(int dir=0;dir<4;dir++){
                size_t num=(size_t)it.jan*g->num_proj;
                Force4B_capacity_copy(data+at,g->c2_host,g->c2_dev,dir*g->c2_stride+work->b[q],num);at+=num;
            }
        }
        offset[i+1]=at;coff[i+1]=ct;
    }
    if(at!=packet->values || ct!=packet->density_values)return 0;
    OpenMXGpuTask task={0};task.count=packet->items;task.inputs=3;task.outputs=1;
    task.kernel=Force4B_capacity_kernel;task.parameters=&work->parameters;task.parameter_bytes=sizeof(work->parameters);
    task.view[0]=(OpenMXGpuTaskView){work->items+work->first,0,sizeof(*work->items)};
    task.view[1]=(OpenMXGpuTaskView){data,0,0,offset,sizeof(float)};
    task.view[2]=(OpenMXGpuTaskView){weights,0,0,coff,sizeof(double)};
    task.view[3]=(OpenMXGpuTaskView){work->output+3*work->first,0,3*sizeof(double)};
    (void)openmx_gpu_task_execute(work->comm,&task,1,kind==1?"force-VNA1-capacity":"force-VNA2-capacity");
    return 1; /* execute also returns zero after successful CPU completion. */
}

static void Force4B_capacity(int kind,Force4BGpuContext *g,int nitems,Force4BGpuItem *items,
    size_t *a,size_t *b,int *halo,double *cdm,double *output,int done)
{
    size_t first=done?(size_t)nitems:0;
    ForceVnaRecoveryWork work={kind,g,first,items,a,b,halo,cdm,output,mpi_comm_level1,{kind,g->num_proj,openmx_fp32_enabled()}};
    for(;;){
        int remaining=first<(size_t)nitems,any;
        MPI_Allreduce(&remaining,&any,1,MPI_INT,MPI_MAX,mpi_comm_level1);if(!any)break;
        size_t n=0,values=0,cdms=0;
        while(first+n<(size_t)nitems && n<OPENMX_FORCE_RECOVERY_MAX_ITEMS){
            Force4BGpuItem it=items[first+n];
            size_t add=(size_t)g->num_proj*(kind==1?(size_t)it.k_n*(3*it.ian+it.jan):4*(size_t)(it.ian+it.jan));
            if(n && values+add>16777216)break;
            values+=add;cdms+=(size_t)it.ian*it.jan;n++;
        }
        char error[256];work.first=first;
        if(!openmx_core_force_vna_recovery_run(n,values,cdms,Force4B_RecoveryPacket,&work,error,sizeof(error)))
            ForceRecovery_Failure("Force4B",error);
        first+=n;
    }
}

typedef struct {
    int kind;
    ForceHNLGpuContext *g;
    size_t first;
    const Force4BGpuItem *items;
    const size_t *a,*b;
    const int *halo,*ene,*nlp,*h0;
    const double *cdm;
    double *output;
    MPI_Comm comm;
} ForceHnlRecoveryWork;

static int Force_HNL_RecoveryPacket(const OpenMXForceHnlRecoveryView *packet,void *opaque)
{
    ForceHnlRecoveryWork *work=(ForceHnlRecoveryWork*)opaque;
    const int kind=work->kind;ForceHNLGpuContext *g=work->g;
    ForceHNLCapacityItem *prs=packet->item;size_t *offset=packet->offset,*nlp_offset=packet->length_offset;
    double *data=packet->data;int *length=packet->length;
    size_t at=0,ki=0;offset[0]=nlp_offset[0]=0;
    for(size_t i=0;i<packet->items;i++){
        size_t p=work->first+i;const Force4BGpuItem *it=work->items+p;
        prs[i]=(ForceHNLCapacityItem){kind,it->ian,it->jan,it->k_n,kind==2?work->h0[p]:0,kind==2?work->nlp[p]:0};
        size_t mn=(size_t)it->ian*it->jan;memcpy(data+at,work->cdm+it->cdm_off,mn*sizeof(double));at+=mn;
        if(kind==1){
            for(int k=it->k_off;k<it->k_off+it->k_n;k++){
                int np=work->nlp[k];length[ki++]=np;
                memcpy(data+at,g->ene+work->ene[k],(size_t)np*sizeof(double));at+=np;
                for(int dir=1;dir<=3;dir++){
                    size_t num=(size_t)it->ian*np;
                    memcpy(data+at,g->flat+dir*g->flat_stride+work->a[k],num*sizeof(double));at+=num;
                }
                size_t num=(size_t)it->jan*np;
                memcpy(data+at,(work->halo[k]?g->halo:g->flat)+work->b[k],num*sizeof(double));at+=num;
            }
        }else{
            int np=work->nlp[p];memcpy(data+at,g->ene+work->ene[p],(size_t)np*sizeof(double));at+=np;
            for(int dir=0;dir<4;dir++){
                size_t num=(size_t)it->ian*np;
                if(work->h0[p]==1){if(dir==0)memcpy(data+at,g->flat+work->a[p],num*sizeof(double));else memset(data+at,0,num*sizeof(double));}
                else memcpy(data+at,g->c2+dir*g->c2_stride+work->a[p],num*sizeof(double));
                at+=num;
            }
            for(int dir=0;dir<4;dir++){
                size_t num=(size_t)it->jan*np;
                memcpy(data+at,g->c2+dir*g->c2_stride+work->b[p],num*sizeof(double));at+=num;
            }
        }
        offset[i+1]=at;nlp_offset[i+1]=ki;
    }
    if(at!=packet->values || ki!=packet->lengths)return 0;
    OpenMXGpuTask task={0};task.count=packet->items;task.inputs=3;task.outputs=1;task.kernel=Force_HNL_capacity_kernel;
    task.view[0]=(OpenMXGpuTaskView){prs,0,sizeof(*prs)};
    task.view[1]=(OpenMXGpuTaskView){data,0,0,offset,sizeof(double)};
    task.view[2]=(OpenMXGpuTaskView){length,0,0,nlp_offset,sizeof(int)};
    task.view[3]=(OpenMXGpuTaskView){work->output+3*work->first,0,3*sizeof(double)};
    (void)openmx_gpu_task_execute(work->comm,&task,1,kind==1?"force-nonlocal1-capacity":"force-nonlocal2-capacity");
    return 1;
}

static void Force_HNL_capacity(int kind,ForceHNLGpuContext *g,int nitems,const Force4BGpuItem *items,
    const size_t *a,const size_t *b,const int *halo,const int *ene,const int *nlp,const int *h0,
    const double *cdm,double *output,int done)
{
    size_t first=done?(size_t)nitems:0;
    ForceHnlRecoveryWork work={kind,g,first,items,a,b,halo,ene,nlp,h0,cdm,output,mpi_comm_level1};
    for(;;){
        int remaining=first<(size_t)nitems,any;
        MPI_Allreduce(&remaining,&any,1,MPI_INT,MPI_MAX,mpi_comm_level1);if(!any)break;
        size_t n=0,values=0,nk=0;
        while(first+n<(size_t)nitems && n<OPENMX_FORCE_RECOVERY_MAX_ITEMS){
            const Force4BGpuItem *it=items+first+n;size_t add=(size_t)it->ian*it->jan;
            if(kind==1)for(int k=it->k_off;k<it->k_off+it->k_n;k++)add+=(size_t)nlp[k]*(1+3*it->ian+it->jan);
            else add+=(size_t)nlp[first+n]*(1+4*it->ian+4*it->jan);
            if(n && values+add>8388608)break;
            values+=add;nk+=kind==1?it->k_n:0;n++;
        }
        char error[256];work.first=first;
        if(!openmx_core_force_hnl_recovery_run(n,values,nk,Force_HNL_RecoveryPacket,&work,error,sizeof(error)))
            ForceRecovery_Failure("Force_HNL",error);
        first+=n;
    }
}
#endif
