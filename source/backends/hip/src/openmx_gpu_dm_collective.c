#include "openmx_gpu_dm_collective.h"
#include "openmx_gpu_dm_entry.h"
#include "openmx_gpu_task.h"
#include "openmx_gpu_workers.h"
#include "openmx_mgpu_runtime.h"
#include "hip_runtime_compat.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {int kind,states,phased,edm,pdm;size_t row_bytes;} DMParameters;
static void dm_kernel(size_t first,size_t count,const void *parameters,void *const v[],int gpu)
{
    const DMParameters *p=parameters;
    int kind=p->kind,states=p->states,phased=p->phased,edm=p->edm,pdm=p->pdm;
    size_t row_bytes=p->row_bytes;
    unsigned char *vectors=v[0];double *phase=v[1],*weights=v[2],*out=v[3];
    (void)first;
#pragma omp target teams distribute parallel for if(gpu) is_device_ptr(vectors,phase,weights,out)
    for(size_t i=0;i<count;i++){
        int indices[2]={0,1};double result[10]={0};
        openmx_gpu_dm_entry(kind,2,states,states,1,vectors+i*row_bytes,indices,weights,
                            phase+2*i,result,phased,edm,pdm,0);
        for(int c=0;c<10;c++)out[10*i+c]=result[c];
    }
}
static void dm_abort(MPI_Comm comm,const char *message)
{
    fprintf(stderr,"density matrix recovery: %s\n",message);fflush(stderr);
    if(comm!=MPI_COMM_NULL)MPI_Abort(comm,1);
    abort();
}
static void dm_copy(MPI_Comm comm,void *dst,const void *src,size_t bytes,int host)
{
    if(host)memcpy(dst,src,bytes);
    else if(hipMemcpy(dst,src,bytes,hipMemcpyDeviceToHost)!=hipSuccess)
        dm_abort(comm,"cannot read resident eigenvector rows");
}
void openmx_gpu_dm_collective(MPI_Comm comm,int kind,int n,int states,int stride,int entries,
    const void *vectors,int host_vectors,const int *basis0,const int *basis1,
    const int *phase_index,int phase_count,const double *phase_r,const double *phase_i,
    const double *occupation,const double *energy_weight,const double *partial_weight,
    double *const outputs[10],int accumulate)
{
    if(entries<0 || (entries && (kind<OPENMX_GPU_DM_REAL || kind>OPENMX_GPU_EDM_GATHER ||
       n<1 || n>INT32_MAX/2 || !vectors || !basis0 || !basis1 || !outputs ||
       (kind<=OPENMX_GPU_DM_NC && (states<1 || stride<states || !occupation)) ||
       (phase_index && (phase_count<1 || !phase_r || !phase_i)))))
        dm_abort(comm,"invalid dimensions or operands");
    for(int i=0;i<entries;i++)if(basis0[i]<0 || basis0[i]>=n || basis1[i]<0 || basis1[i]>=n ||
        (phase_index && (phase_index[i]<0 || phase_index[i]>=phase_count)))
        dm_abort(comm,"invalid sparse entry index");
    int done=!entries;
    if(!done && !openmx_gpu_capacity_test_recovery())
        done=openmx_gpu_dm_try_entries(kind,n,states,stride,entries,vectors,host_vectors,
            basis0,basis1,phase_index,phase_count,phase_r,phase_i,occupation,energy_weight,
            partial_weight,outputs,accumulate);
    size_t width=kind==OPENMX_GPU_DM_REAL?sizeof(double):sizeof(OpenMXDMComplex);
    int gather=kind>=OPENMX_GPU_DM_GATHER,spins=kind==OPENMX_GPU_DM_NC?2:1;
    if(gather)states=0;
    size_t row_bytes=gather?16*sizeof(OpenMXDMComplex):(size_t)2*spins*states*width;
    size_t chunk=row_bytes?((size_t)32*1024*1024)/(row_bytes+12*sizeof(double)):1;
    if(chunk<1)chunk=1;if(chunk>1024)chunk=1024;
    int next=done?entries:0;
    double *weights=NULL,*phase=NULL,*out=NULL;unsigned char *rows=NULL;
    if(!done){
        weights=calloc(states?3*(size_t)states:1,sizeof(double));
        if(!weights)dm_abort(comm,"host memory exhausted for state weights");
        if(states){memcpy(weights,occupation,states*sizeof(double));
            if(energy_weight)memcpy(weights+states,energy_weight,states*sizeof(double));
            if(partial_weight)memcpy(weights+2*(size_t)states,partial_weight,states*sizeof(double));}
        for(;;){
            rows=malloc(chunk*row_bytes);phase=malloc(2*chunk*sizeof(double));out=malloc(10*chunk*sizeof(double));
            if(rows && phase && out)break;
            free(rows);free(phase);free(out);
            if(chunk==1)dm_abort(comm,"host memory exhausted for one density entry");
            chunk=(chunk+1)/2;
        }
    }
    for(;;){
        int pending=next<entries,any=pending;
        if(comm!=MPI_COMM_NULL)MPI_Allreduce(MPI_IN_PLACE,&any,1,MPI_INT,MPI_MAX,comm);
        if(!any)break;
        size_t count=pending?(size_t)(entries-next):0;if(count>chunk)count=chunk;
        for(size_t i=0;i<count;i++){
            int k=next+(int)i,a=basis0[k],b=basis1[k];unsigned char *row=rows+i*row_bytes;
            phase[2*i]=phase_index?phase_r[phase_index[k]]:1;
            phase[2*i+1]=phase_index?phase_i[phase_index[k]]:0;
            if(gather){
                memset(row,0,row_bytes);
                size_t src[3]={(size_t)a+(size_t)b*2*n,(size_t)(a+n)+(size_t)(b+n)*2*n,(size_t)a+(size_t)(b+n)*2*n};
                int dst[3]={4,14,12};
                for(int c=0;c<3;c++)dm_copy(comm,row+dst[c]*width,(const unsigned char*)vectors+src[c]*width,width,host_vectors);
            }else for(int spin=0;spin<spins;spin++){
                dm_copy(comm,row+(size_t)2*spin*states*width,(const unsigned char*)vectors+(size_t)(a+spin*n)*stride*width,(size_t)states*width,host_vectors);
                dm_copy(comm,row+(size_t)(2*spin+1)*states*width,(const unsigned char*)vectors+(size_t)(b+spin*n)*stride*width,(size_t)states*width,host_vectors);
            }
        }
        DMParameters p={kind,states,phase_index!=NULL,energy_weight!=NULL,partial_weight!=NULL,row_bytes};
        OpenMXGpuTask t={0};t.count=count;
        /* Exact k-point groups are a single-node Band policy. Cluster,
           ELPA/COSMA and multi-node recovery can use the auxiliary node
           pool even when device-resident dense cooperation is disabled. */
        t.scope_kind=phase_index && !openmx_mgpu_multi_node()?2:1;
        t.inputs=3;t.outputs=1;t.kernel=dm_kernel;t.parameters=&p;t.parameter_bytes=sizeof(p);
        t.view[0]=(OpenMXGpuTaskView){rows,0,row_bytes};t.view[1]=(OpenMXGpuTaskView){phase,0,2*sizeof(double)};
        t.view[2]=(OpenMXGpuTaskView){weights,3*(size_t)states*sizeof(double),0};t.view[3]=(OpenMXGpuTaskView){out,0,10*sizeof(double)};
        openmx_gpu_task_execute(comm,&t,1,"density-matrix-capacity");
        for(int c=0;c<10;c++)if(outputs && outputs[c])for(size_t i=0;i<count;i++){
            if(accumulate)outputs[c][next+i]+=out[10*i+c];else outputs[c][next+i]=out[10*i+c];
        }
        next+=(int)count;
    }
    free(rows);free(phase);free(out);free(weights);
}
