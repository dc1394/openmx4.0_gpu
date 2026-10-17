#include "openmx_gpu_collective.h"
#include "openmx_gpu_workers.h"
#include "openmx_stdout_level.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Private communicator and bounded messages keep these transfers separate
   from the surrounding SCF traffic, without relying on MPI_THREAD_MULTIPLE. */
enum { HEADER=0, READY=1, PAYLOAD=2 };

static void send_bytes(const void *data,size_t bytes,int dest,MPI_Comm comm)
{
    const unsigned char *p=data;
    while (bytes) {
        int chunk=bytes>16777216 ? 16777216 : (int)bytes;
        MPI_Send(p,chunk,MPI_BYTE,dest,PAYLOAD,comm);
        p+=chunk; bytes-=(size_t)chunk;
    }
}
static void recv_bytes(void *data,size_t bytes,int src,MPI_Comm comm)
{
    unsigned char *p=data;
    while (bytes) {
        int chunk=bytes>16777216 ? 16777216 : (int)bytes;
        MPI_Recv(p,chunk,MPI_BYTE,src,PAYLOAD,comm,MPI_STATUS_IGNORE);
        p+=chunk; bytes-=(size_t)chunk;
    }
}

/* Receiver allocation is acknowledged before any payload is sent. Thus a
   failed host allocation cannot leave a peer blocked sending a large slab. */
static int send_buffer(const void *data,size_t bytes,int valid,int dest,MPI_Comm comm)
{
    uint64_t header[2]={(uint64_t)bytes,(uint64_t)(valid!=0)};
    int ready=0;
    MPI_Send(header,2,MPI_UINT64_T,dest,HEADER,comm);
    MPI_Recv(&ready,1,MPI_INT,dest,READY,comm,MPI_STATUS_IGNORE);
    if (ready && bytes) send_bytes(data,bytes,dest,comm);
    return ready;
}
static int recv_buffer(void **data,size_t *bytes,int src,MPI_Comm comm)
{
    uint64_t header[2]; int ready=0;
    *data=NULL; *bytes=0;
    MPI_Recv(header,2,MPI_UINT64_T,src,HEADER,comm,MPI_STATUS_IGNORE);
    if (header[1] && header[0]<=SIZE_MAX) {
        *bytes=(size_t)header[0];
        *data=malloc(*bytes ? *bytes : 1);
        ready=(*data!=NULL);
    }
    MPI_Send(&ready,1,MPI_INT,src,READY,comm);
    if (ready && *bytes) recv_bytes(*data,*bytes,src,comm);
    return ready;
}

int openmx_gpu_collective_retry(MPI_Comm comm,const OpenMXGpuCollectiveOps *ops,
    void *context,size_t items,int local_done,int can_help,const char *label)
{
    int rank,size,any_need,need=(local_done==0 && items>0);
    MPI_Comm_rank(comm,&rank); MPI_Comm_size(comm,&size);
    MPI_Allreduce(&need,&any_need,1,MPI_INT,MPI_MAX,comm);
    if (!any_need || size<2) return local_done>0;
    MPI_Comm wire,node;
    MPI_Comm_dup(comm,&wire);
    MPI_Comm_split_type(wire,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&node);
    int leader=rank;
    MPI_Allreduce(MPI_IN_PLACE,&leader,1,MPI_INT,MPI_MIN,node);
    MPI_Comm_free(&node);
    /* triples: physical-node representative, eligibility, pending work */
    int *census=calloc((size_t)size*3,sizeof(int));
    int *dest=malloc((size_t)size*sizeof(int));
    int *failed=calloc((size_t)size,sizeof(int));
    void **result=calloc((size_t)size,sizeof(void*));
    size_t *result_bytes=calloc((size_t)size,sizeof(size_t));
    int ok=census && dest && failed && result && result_bytes && ops && ops->pack && ops->run && ops->commit;
    MPI_Allreduce(MPI_IN_PLACE,&ok,1,MPI_INT,MPI_MIN,wire);
    int local[3]={leader,can_help!=0,need};
    if (!ok) goto done;
    MPI_Allgather(local,3,MPI_INT,census,3,MPI_INT,wire);
    for (int origin=0;origin<size;origin++) {
        if (!census[3*origin+2]) continue;
        int nodes=0;
        /* Pick one eligible rank per physical node, preferring the origin
           on its own node. A rank never exports its HIP device ordinal. */
        for (int peer=0;peer<size;peer++) {
            if (!census[3*peer+1]) continue;
            int seen=0;
            for (int i=0;i<nodes;i++) if (census[3*dest[i]]==census[3*peer]) seen=1;
            if (!seen) dest[nodes++]=peer;
        }
        for (int i=0;i<nodes;i++) if (census[3*dest[i]]==census[3*origin]) {
            if (census[3*origin+1]) dest[i]=origin;
        }
        /* Prefer another node for indivisible work. On a failed capacity
           attempt remove the declining nodes and redistribute privately;
           a fully occupied owner must not veto spare capacity elsewhere. */
        for (int i=0;i<nodes;i++) if (census[3*dest[i]]==census[3*origin]) {
            int own=dest[i];
            for (int j=i+1;j<nodes;j++) dest[j-1]=dest[j];
            dest[nodes-1]=own;break;
        }
        uint64_t count=rank==origin ? (uint64_t)items : 0;
        MPI_Bcast(&count,1,MPI_UINT64_T,origin,wire);
        int eligible=nodes;
        while (eligible>0) {
            nodes=count<(uint64_t)eligible?(int)count:eligible;
            if (!nodes || (nodes==1 && census[3*dest[0]]==census[3*origin])) break;
            int own=-1;
            for (int i=0;i<nodes;i++) {failed[i]=0;if (dest[i]==rank) own=i;}
            int success=1;
            void *packet=NULL,*output=NULL;size_t bytes=0,output_bytes=0;
            for (int i=0;i<nodes;i++) {
                if (dest[i]==origin) continue;
                if (rank==origin) {
                    size_t first=openmx_gpu_worker_cut((size_t)count,i,nodes);
                    size_t last=openmx_gpu_worker_cut((size_t)count,i+1,nodes);
                    void *p=NULL;size_t nb=0;
                    int packed=ops->pack(context,first,last,&p,&nb);
                    if (!send_buffer(p,nb,packed && (p || !nb),dest[i],wire)) failed[i]=1;
                    free(p);
                }
                else if (rank==dest[i]) {
                    success=recv_buffer(&packet,&bytes,origin,wire);
                    if (success) success=ops->run(context,packet,bytes,&output,&output_bytes);
                    if (!success) failed[i]=1;
                    free(packet);packet=NULL;
                }
            }
            if (rank==origin && own>=0) {
                size_t first=openmx_gpu_worker_cut((size_t)count,own,nodes);
                size_t last=openmx_gpu_worker_cut((size_t)count,own+1,nodes);
                int packed=ops->pack(context,first,last,&packet,&bytes);
                if (!(packed && ops->run(context,packet,bytes,&result[own],&result_bytes[own]))) failed[own]=1;
                free(packet);packet=NULL;
            }
            for (int i=0;i<nodes;i++) {
                if (dest[i]==origin) continue;
                if (rank==origin) {
                    if (!recv_buffer(&result[i],&result_bytes[i],dest[i],wire)) failed[i]=1;
                }
                else if (rank==dest[i]) {
                    if (!send_buffer(output,output_bytes,success && (output || !output_bytes),origin,wire)) failed[i]=1;
                    free(output);output=NULL;
                }
            }
            MPI_Allreduce(MPI_IN_PLACE,failed,nodes,MPI_INT,MPI_MAX,wire);
            success=1;for (int i=0;i<nodes;i++) if (failed[i]) success=0;
            if (rank==origin) {
                if (success) {
                    for (int i=0;i<nodes;i++)
                        ops->commit(context,openmx_gpu_worker_cut((size_t)count,i,nodes),
                            openmx_gpu_worker_cut((size_t)count,i+1,nodes),result[i],result_bytes[i]);
                    local_done=1;
                }
                if (OPENMX_STDOUT_LEVEL(1)) {
                    fprintf(stderr,"<OpenMX_MGPU> work name=%s origin=%d nodes=%d items=%zu tier=remote completed=%d\n",
                        label?label:"unnamed",origin,nodes,items,success);
                }
            }
            for (int i=0;i<nodes;i++) {free(result[i]);result[i]=NULL;result_bytes[i]=0;}
            if (success) break;
            int remaining=0;
            for (int i=0;i<eligible;i++) if (i>=nodes || !failed[i]) dest[remaining++]=dest[i];
            eligible=remaining;
        }

    }
done:
    free(census); free(dest); free(failed); free(result); free(result_bytes);
    MPI_Comm_free(&wire);
    return local_done>0;
}
