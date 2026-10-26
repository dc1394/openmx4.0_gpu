#ifndef OPENMX_FORCE_RECOVERY_H
#define OPENMX_FORCE_RECOVERY_H
#include "openmx_core_force_recovery.h"
#include "openmx_host_simd.h"
#include "openmx_core_force_simd.h"

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

/* Host SIMD body of the recovery items (OPENMX_HOST_SIMD): the quantity of
   Force4B_capacity_kernel with the Core's projector block kernel, in that
   kernel's sequential order (per element the k blocks ascending, then the
   weighted sum over the elements row-major). Rows are read in place from the
   host archives, else staged from the device archives. Local to the rank. */
static void Force4B_capacity_rows(const float *base,int rows,int np,const float **table)
{
    for(int r=0;r<rows;r++)table[r]=base+(size_t)r*(size_t)np;
}
static void Force4B_capacity_host(int kind,Force4BGpuContext *g,int nitems,const Force4BGpuItem *items,
    const size_t *a,const size_t *b,const int *halo,const double *cdm,double *output)
{
    const int np=g->num_proj,fp32=openmx_fp32_enabled(),mode=OpenMX_HostSimdMode();
    int max_rows=1,p;
    for(p=0;p<nitems;p++){
        if(max_rows<items[p].ian)max_rows=items[p].ian;
        if(max_rows<items[p].jan)max_rows=items[p].jan;
    }
#pragma omp parallel if(!omp_in_parallel())
    {
        const size_t plane=(size_t)max_rows*(size_t)max_rows;
        const size_t scratch_bytes=openmx_core_force_vna_scratch(max_rows,np,2,0);
        const float **tables=(const float**)malloc(sizeof(float*)*8*(size_t)max_rows);
        float *stage=(float*)malloc(sizeof(float)*8*(size_t)max_rows*(size_t)(np?np:1));
        double *block=(double*)malloc(sizeof(double)*6*plane);
        void *scratch=malloc(scratch_bytes?scratch_bytes:1);
        if(!tables || !stage || !block || !scratch)Force4B_gpu_abort("Force4B: cannot allocate the host recovery workspace.");
        double *sum=block+3*plane;
        int q;
#pragma omp for schedule(dynamic)
        for(q=0;q<nitems;q++){
            const Force4BGpuItem it=items[q];
            const int ian=it.ian,jan=it.jan,mn=ian*jan;
            const size_t rows_a=(size_t)ian*(size_t)np,rows_b=(size_t)jan*(size_t)np;
            const float **t[8];
            OpenMXForceVnaBlock blk;
            int e,c;
            for(c=0;c<8;c++)t[c]=tables+(size_t)c*(size_t)max_rows;
            memset(&blk,0,sizeof(blk));
            blk.rows=ian;blk.columns=jan;blk.projectors=np;blk.fp32=fp32;blk.lanes_rows=0;blk.output_stride=(size_t)jan;
            for(c=0;c<3;c++)blk.output[c]=block+(size_t)c*plane;
            if(kind==1){
                for(e=0;e<3*mn;e++)sum[(size_t)(e/mn)*plane+(size_t)(e%mn)]=0.0;
                for(int k=it.k_off;k<it.k_off+it.k_n;k++){
                    const float *right_host=halo[k]?g->halo_host:g->flat_host;
                    const float *right_dev=halo[k]?g->halo_dev:g->flat_dev;
                    for(int dir=1;dir<4;dir++){
                        const size_t at=(size_t)dir*g->flat_stride+a[k];
                        const float *base=g->flat_host?g->flat_host+at:stage+(size_t)(dir-1)*rows_a;
                        if(!g->flat_host)Force4B_capacity_copy(stage+(size_t)(dir-1)*rows_a,NULL,g->flat_dev,at,rows_a);
                        Force4B_capacity_rows(base,ian,np,t[dir-1]);
                    }
                    {
                        const float *base=right_host?right_host+b[k]:stage+3*rows_a;
                        if(!right_host)Force4B_capacity_copy(stage+3*rows_a,NULL,right_dev,b[k],rows_b);
                        Force4B_capacity_rows(base,jan,np,t[3]);
                    }
                    blk.terms=1;
                    for(c=0;c<3;c++)blk.derivative[c]=t[c];
                    blk.value=t[3];
                    if(!openmx_core_force_vna_block(&blk,scratch,scratch_bytes,mode))
                        Force4B_gpu_abort("Force4B: the host recovery kernel refused a block.");
                    for(c=0;c<3;c++)for(e=0;e<mn;e++)sum[(size_t)c*plane+(size_t)e]+=block[(size_t)c*plane+(size_t)e];
                }
            }else{
                for(int dir=0;dir<4;dir++){
                    const size_t at_a=(size_t)dir*g->c2_stride+a[q],at_b=(size_t)dir*g->c2_stride+b[q];
                    const float *base_a=g->c2_host?g->c2_host+at_a:stage+(size_t)dir*rows_a;
                    const float *base_b=g->c2_host?g->c2_host+at_b:stage+4*rows_a+(size_t)dir*rows_b;
                    if(!g->c2_host){
                        Force4B_capacity_copy(stage+(size_t)dir*rows_a,NULL,g->c2_dev,at_a,rows_a);
                        Force4B_capacity_copy(stage+4*rows_a+(size_t)dir*rows_b,NULL,g->c2_dev,at_b,rows_b);
                    }
                    Force4B_capacity_rows(base_a,ian,np,t[dir]);
                    Force4B_capacity_rows(base_b,jan,np,t[4+dir]);
                }
                blk.terms=2;
                for(c=0;c<3;c++){blk.derivative[c]=t[1+c];blk.column_derivative[c]=t[5+c];}
                blk.row_value=t[0];blk.value=t[4];
                blk.output[0]=sum;blk.output[1]=sum+plane;blk.output[2]=sum+2*plane;
                if(!openmx_core_force_vna_block(&blk,scratch,scratch_bytes,mode))
                    Force4B_gpu_abort("Force4B: the host recovery kernel refused a block.");
            }
            {
                double fx=0.0,fy=0.0,fz=0.0;
                for(e=0;e<mn;e++){
                    const double w=cdm[it.cdm_off+(size_t)e];
                    fx+=w*sum[(size_t)e];fy+=w*sum[plane+(size_t)e];fz+=w*sum[2*plane+(size_t)e];
                }
                output[3*(size_t)q]=fx;output[3*(size_t)q+1]=fy;output[3*(size_t)q+2]=fz;
            }
        }
        free(scratch);free(block);free(stage);free((void*)tables);
    }
}

static void Force4B_capacity(int kind,Force4BGpuContext *g,int nitems,Force4BGpuItem *items,
    size_t *a,size_t *b,int *halo,double *cdm,double *output,int done)
{
    /* Host SIMD policy (uniform on every rank): no collective recovery; a
       rank whose batch did not complete evaluates its items itself. */
    if(OpenMX_HostSimdMode()!=OPENMX_HOST_SIMD_LEGACY){
        if(!done)Force4B_capacity_host(kind,g,nitems,items,a,b,halo,cdm,output);
        return;
    }
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

/* Host SIMD body of the nonlocal recovery items: the quantity of
   Force_HNL_capacity_kernel through the Core's nonlocal block kernel, one
   channel per projector entry l carrying ene[l], so that every product is
   (ene[l]*a[l])*b[l] as in that kernel and the terms run in its order (the
   derivative-row loop before the value-row loop of the second case). Then
   the k blocks ascending and the weighted sum over the elements row-major.
   The archives are host arrays. Local to the rank. */
static void Force_HNL_capacity_host(int kind,ForceHNLGpuContext *g,int nitems,const Force4BGpuItem *items,
    const size_t *a,const size_t *b,const int *halo,const int *ene,const int *nlp,const int *h0,
    const double *cdm,double *output)
{
    const int mode=OpenMX_HostSimdMode();
    int max_rows=1,max_np=1,p;
    for(p=0;p<nitems;p++){
        const Force4BGpuItem *it=items+p;
        if(max_rows<it->ian)max_rows=it->ian;
        if(max_rows<it->jan)max_rows=it->jan;
        if(kind==1){for(int k=it->k_off;k<it->k_off+it->k_n;k++)if(max_np<nlp[k])max_np=nlp[k];}
        else if(max_np<nlp[p])max_np=nlp[p];
    }
#pragma omp parallel if(!omp_in_parallel())
    {
        const size_t plane=(size_t)max_rows*(size_t)max_rows;
        const size_t scratch_bytes=openmx_core_force_nl_scratch(max_rows,(size_t)max_np,2);
        OpenMXForceProjectorChannel *channels=(OpenMXForceProjectorChannel*)calloc((size_t)max_np,sizeof(*channels));
        dcomplex *planes=(dcomplex*)malloc(sizeof(dcomplex)*9*plane);
        double *sum=(double*)malloc(sizeof(double)*3*plane);
        void *scratch=malloc(scratch_bytes?scratch_bytes:1);
        if(!channels || !planes || !sum || !scratch)Force4B_gpu_abort("Force_HNL: cannot allocate the host recovery workspace.");
        int q;
#pragma omp for schedule(dynamic)
        for(q=0;q<nitems;q++){
            const Force4BGpuItem *it=items+q;
            const int ian=it->ian,jan=it->jan,mn=ian*jan;
            OpenMXForceProjectorSpecies species;
            OpenMXProjectorForceTerm terms[2];
            OpenMXProjectorForceBlock blk;
            int e,c,kfirst=kind==1?it->k_off:q,klast=kind==1?it->k_off+it->k_n:q+1;
            for(c=0;c<3;c++)for(e=0;e<mn;e++)sum[(size_t)c*plane+(size_t)e]=0.0;
            for(int k=kfirst;k<klast;k++){
                const int np=nlp[k];
                const double *energy=g->ene+ene[k];
                size_t count=0;
                for(int l=0;l<np;l++){
                    channels[l].twice_angular=0;channels[l].offset=(size_t)l;
                    channels[l].energy[0]=energy[l];channels[l].energy[1]=0.0;
                    channels[l].weight[0]=channels[l].weight[1]=1.0;
                }
                memset(&species,0,sizeof(species));
                species.relativistic=0;species.count=(size_t)np;species.components=(size_t)np;species.channels=channels;
                memset(terms,0,sizeof(terms));
                if(kind==1){
                    /* tx += ene[l]*a_dir[m][l]*b[n][l] */
                    for(c=0;c<3;c++)terms[0].derivative[0][c]=g->flat+(size_t)(c+1)*g->flat_stride+a[k];
                    terms[0].value[0]=(halo[k]?g->halo:g->flat)+b[k];
                    terms[0].derivative_rows=(size_t)ian;terms[0].value_rows=(size_t)jan;
                    terms[0].sign=1.0;
                    count=1;
                }else{
                    const double *a0=h0[k]==1?g->flat+a[k]:g->c2+a[k];
                    if(h0[k]!=1){
                        /* sx -= ene[l]*a_dir[m][l]*b0[n][l] */
                        for(c=0;c<3;c++)terms[count].derivative[0][c]=g->c2+(size_t)(c+1)*g->c2_stride+a[k];
                        terms[count].value[0]=g->c2+b[k];
                        terms[count].derivative_rows=(size_t)ian;terms[count].value_rows=(size_t)jan;
                        terms[count].sign=-1.0;
                        count++;
                    }
                    if(h0[k]==0 || h0[k]==1){
                        /* sx -= ene[l]*a0[m][l]*b_dir[n][l]: value rows m, derivative rows n */
                        for(c=0;c<3;c++)terms[count].derivative[0][c]=g->c2+(size_t)(c+1)*g->c2_stride+b[k];
                        terms[count].value[0]=a0;
                        terms[count].derivative_rows=(size_t)jan;terms[count].value_rows=(size_t)ian;
                        terms[count].transpose=1;terms[count].value_first=1;
                        terms[count].sign=-1.0;
                        count++;
                    }
                }
                for(size_t t=0;t<count;t++){terms[t].derivative_stride=(size_t)np;terms[t].value_stride=(size_t)np;}
                memset(&blk,0,sizeof(blk));
                blk.rows=(size_t)ian;blk.columns=(size_t)jan;blk.output_rows=(size_t)ian;blk.output_stride=(size_t)jan;
                blk.term_count=count;blk.terms=terms;blk.accumulate=0;
                for(c=0;c<3;c++)blk.output[c]=planes+(size_t)c*3*plane;
                if(count && !openmx_core_force_nl_block(&species,&blk,scratch,scratch_bytes,mode))
                    Force4B_gpu_abort("Force_HNL: the host recovery kernel refused a block.");
                if(count)for(c=0;c<3;c++)for(e=0;e<mn;e++)sum[(size_t)c*plane+(size_t)e]+=planes[(size_t)c*3*plane+(size_t)e].r;
            }
            {
                const double *w=cdm+it->cdm_off;
                double fx=0.0,fy=0.0,fz=0.0;
                for(e=0;e<mn;e++){fx+=w[e]*sum[(size_t)e];fy+=w[e]*sum[plane+(size_t)e];fz+=w[e]*sum[2*plane+(size_t)e];}
                output[3*(size_t)q]=fx;output[3*(size_t)q+1]=fy;output[3*(size_t)q+2]=fz;
            }
        }
        free(scratch);free(sum);free(planes);free(channels);
    }
}

static void Force_HNL_capacity(int kind,ForceHNLGpuContext *g,int nitems,const Force4BGpuItem *items,
    const size_t *a,const size_t *b,const int *halo,const int *ene,const int *nlp,const int *h0,
    const double *cdm,double *output,int done)
{
    /* Host SIMD policy (uniform on every rank): no collective recovery; a
       rank whose batch did not complete evaluates its items itself. */
    if(OpenMX_HostSimdMode()!=OPENMX_HOST_SIMD_LEGACY){
        if(!done)Force_HNL_capacity_host(kind,g,nitems,items,a,b,halo,ene,nlp,h0,cdm,output);
        return;
    }
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
