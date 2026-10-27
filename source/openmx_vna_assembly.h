#ifndef OPENMX_VNA_ASSEMBLY_H
#define OPENMX_VNA_ASSEMBLY_H
#include "openmx_core_vna_invocation.h"
#include "openmx_vna_status.h"
#include "openmx_vna_cpu.h"

/* The persistent ABI keeps the application's existing float projector
   format. Standalone CPU evaluators remain generic in Type_DS_VNA. */
#ifdef __cplusplus
static_assert(sizeof(Type_DS_VNA)==sizeof(float),"VNA storage ABI requires float projectors");
#else
_Static_assert(sizeof(Type_DS_VNA)==sizeof(float),"VNA storage ABI requires float projectors");
#endif

/* The synchronous assembly is shared. Native dispatch adapters keep device
   admission, rank turns, capacity recovery and architecture policies local
   to the CUDA/HIP implementations. */

/* One projector block's rows to (pack) or from (unpack) a contiguous
   message, in the legacy h/i/j element order: a single copy when the rows
   are contiguous with the component stride, else one copy per row. Either
   way the same values move as in the element loops, bit for bit. */
static size_t SetPro_RingRows(Type_DS_VNA *message,Type_DS_VNA **block,int rows,int components,int unpack)
{
  const size_t row_values=(size_t)components;
  int contiguous=1;
  for (int i=1;i<rows && contiguous;++i) contiguous=block[i]==block[0]+(size_t)i*row_values;
  if (rows>0 && contiguous) {
    if (unpack) memcpy(block[0],message,sizeof(Type_DS_VNA)*row_values*(size_t)rows);
    else memcpy(message,block[0],sizeof(Type_DS_VNA)*row_values*(size_t)rows);
  }
  else for (int i=0;i<rows;++i) {
    if (unpack) memcpy(block[i],message+(size_t)i*row_values,sizeof(Type_DS_VNA)*row_values);
    else memcpy(message+(size_t)i*row_values,block[i],sizeof(Type_DS_VNA)*row_values);
  }
  return row_values*(size_t)(rows>0?rows:0);
}

/* The pairs [first, count) of a pair list, for the host SIMD assembly of the
   pairs a device batch left unfinished. The CPU kernels read only the per-pair
   arrays; the per-atom CSR offsets do not describe a pair range. */
static void SetPro_PairTail(const OpenMXAtomPairBuffers *all,int first,OpenMXAtomPairBuffers *tail)
{
  *tail=*all;
  tail->count=all->count-first;
  tail->local=all->local+first;
  tail->neighbor=all->neighbor+first;
  tail->central_global=all->central_global+first;
  tail->central_species=all->central_species+first;
  tail->neighbor_species=all->neighbor_species+first;
  tail->displacements=all->displacements+(size_t)first*6;
  tail->offsets=NULL;
}

/* OPENMX_SETPRO_VERBOSE=1: where the projector ring spends its time (per rank
   and call). Waiting phases are the size handshake, the blocking receive and
   the completion of each send; copies are the message pack and unpack. */
typedef struct { double sizes, pack, send, receive, unpack, halo, wait, bytes; long messages; } SetProRingTimes;
static inline void SetPro_RingTick(int on,double *clock,double *phase)
{
  double now;
  if (!on) return;
  dtime(&now);
  *phase+=now-*clock;
  *clock=now;
}
static double SetPro_Separable(const OpenMXVnaMatrixViews *matrices,const OpenMXReciprocalViews *reciprocal,
    const OpenMXVnaInvocationBuffers *invocation,const SetProVnaTables *tables,int *used_gpu)
{
  double ****HVNA=matrices->HVNA;
  Type_DS_VNA *****DS_VNA=matrices->DS_VNA;
  const OpenMXProjectorMatrix projectors_matrix={matrices->projectors.layout,matrices->projectors.values};
  double begin,end,communication_begin,communication_end;
  double time1=0,time2=0,time3=0,time4=0,time5=0;
  dtime(&begin);
  int peers,rank;
  MPI_Comm_size(mpi_comm_level1,&peers);
  MPI_Comm_rank(mpi_comm_level1,&rank);
  OpenMXProjectorTransfer transfer=OpenMX_TransferBegin(mpi_comm_level1,peers,sizeof(Type_DS_VNA));
  int *sent=transfer.buffers.sent,*received=transfer.buffers.received;
  int *send_sizes=transfer.buffers.send_sizes,*receive_sizes=transfer.buffers.receive_sizes;
  OpenMXVnaPreparationBuffers preparation=invocation->preparation;
  const OpenMXVnaPreparation *preparation_owner=invocation->preparation_owner;
  const OpenMXAtomPairBuffers *pairs=&invocation->pairs;
  double *wavevectors=invocation->wavevectors;
  const int components=preparation.projector_components,fp32=openmx_fp32_enabled();
  SetPro_PrepareRadials(reciprocal, &preparation,wavevectors,tables->weights,
      tables->maximum_wavevector-tables->minimum_wavevector);
  SetProGpu2Context gpu2={0};
  const int phases_timed=SetPro_verbose();
  double phase_clock=0.0,phase_dsvna=0.0,phase_barrier=0.0,phase_ring=0.0,phase_local=0.0;
  if (phases_timed) dtime(&phase_clock);
  if (SetPro_DSVNA_GpuBegin(reciprocal, tables->conversion,tables->conversion_width, &gpu2,&preparation)) {
    SetPro_DSVNA_GpuRun(DS_VNA,pairs->count,pairs->local,pairs->neighbor,wavevectors,&gpu2);
    const int host_route=gpu2.host_route,host_first=gpu2.host_first;
    SetPro_DSVNA_GpuEnd(&gpu2);
    /* a late device failure leaves this rank's remaining pairs to the host
       SIMD assembly (rank-local; the failure verdict was collective) */
    if (host_route && host_first<pairs->count) {
      OpenMXAtomPairBuffers rest;
      SetPro_PairTail(pairs,host_first,&rest);
      SetPro_DSVNA_CPU(&projectors_matrix,preparation_owner,&preparation,wavevectors,&rest,tables,&time1,&time2,&time3);
    }
  }
  else
    SetPro_DSVNA_CPU(&projectors_matrix,preparation_owner,&preparation,wavevectors,pairs,tables,&time1,&time2,&time3);

  SetPro_RingTick(phases_timed,&phase_clock,&phase_dsvna);
  MPI_Barrier(mpi_comm_level1);
  SetPro_RingTick(phases_timed,&phase_clock,&phase_barrier);
  if (measure_time) dtime(&communication_begin);
  SetProHvnaDispatch hvna={0};
  const int gpu=SetPro_HVNA_DispatchBegin(reciprocal, &hvna,DS_VNA,preparation_owner,preparation.projectors);
  const SetProHvnaGeometry hvna_geometry={
      Matomnum,M2G,F_G2M,WhatSpecies,FNAN,natn,RMI1,Spe_Atom_Cut1,Dis,tables->atom_seconds};
  OpenMXHvnaProjectorBuffers projectors;
  OpenMXHvnaProjectors *projector_owner=NULL;
  OpenMXHvnaWorkspaceBuffers halo_workspace;
  OpenMXHvnaWorkspace *halo_owner=NULL;
  SetProHvnaHost halo_cache;
  memset(&halo_cache,0,sizeof(halo_cache));
  if (!gpu) {
    char error[256];
    projector_owner=openmx_core_hvna_projectors_create(preparation_owner,reciprocal->vps.neutral_energies,&projectors,error,sizeof(error));
    if (!projector_owner) { SetPro_gpu_abort(error);abort(); }
    halo_owner=SetPro_HVNA_CPU_Workspace(projector_owner,&halo_workspace);
    SetPro_HVNA_CPU_HaloCache(&halo_cache,&hvna_geometry,&projectors,atomnum,fp32);
  }
  const int ring_timed=SetPro_verbose();
  SetProRingTimes ring;
  double ring_clock=0.0;
  memset(&ring,0,sizeof(ring));
  if (ring_timed) dtime(&ring_clock);

  /* Preserve the two-pass ring protocol: announce the next atom's size to
     every peer, then transfer one atom per peer and consume its halo before
     reusing the receive slot. Each nonblocking send is waited for before the
     send buffer is reused. The two directions have independent owners. */
  int pending;
  do {
    for (int id=0;id<peers;++id) {
      const int destination=(rank+id)%peers,source=(rank-id+peers)%peers;
      MPI_Request request;MPI_Status status;
      const int sending=sent[destination]<F_Snd_Num[destination];
      int count=0;
      if (sending) {
        const int gc=Snd_GAN[destination][sent[destination]];
        const int rows=preparation.orbitals[WhatSpecies[gc]];
        count=OpenMX_TransferCount(&transfer,FNAN[gc]+1,rows,components);
        MPI_Isend(&count,1,MPI_INT,destination,999,mpi_comm_level1,&request);
      }
      send_sizes[destination]=count;
      receive_sizes[source]=0;
      if (received[source]<F_Rcv_Num[source])
        MPI_Recv(receive_sizes+source,1,MPI_INT,source,999,mpi_comm_level1,&status);
      if (sending) MPI_Wait(&request,&status);
    }
    SetPro_RingTick(ring_timed,&ring_clock,&ring.sizes);
    for (int id=0;id<peers;++id) {
      const int destination=(rank+id)%peers,source=(rank-id+peers)%peers;
      MPI_Request request;MPI_Status status;
      const int sending=sent[destination]<F_Snd_Num[destination];
      if (sending) {
        const int mc=Snd_MAN[destination][sent[destination]],gc=Snd_GAN[destination][sent[destination]];
        const int rows=preparation.orbitals[WhatSpecies[gc]];
        Type_DS_VNA *buffer=OpenMX_TransferBuffer(&transfer,OPENMX_EXCHANGE_SEND,send_sizes[destination]);
        size_t offset=0;
        for (int h=0;h<=FNAN[gc];++h) offset+=SetPro_RingRows(buffer+offset,DS_VNA[0][mc][h],rows,components,0);
        SetPro_RingTick(ring_timed,&ring_clock,&ring.pack);
        MPI_Isend(buffer,send_sizes[destination],MPI_Type_DS_VNA,destination,999,mpi_comm_level1,&request);
        SetPro_RingTick(ring_timed,&ring_clock,&ring.send);
        ring.messages++;ring.bytes+=(double)send_sizes[destination]*sizeof(Type_DS_VNA);
      }
      if (received[source]<F_Rcv_Num[source]) {
        const int gc=Rcv_GAN[source][received[source]],original_mc=F_TopMAN[source]+received[source];
        const int rows=preparation.orbitals[WhatSpecies[gc]];
        const int count=OpenMX_TransferCount(&transfer,FNAN[gc]+1,rows,components);
        if (receive_sizes[source]!=count) OpenMX_TransferFail(mpi_comm_level1,"inconsistent VNA receive count");
        Type_DS_VNA *buffer=OpenMX_TransferBuffer(&transfer,OPENMX_EXCHANGE_RECEIVE,count);
        SetPro_RingTick(ring_timed,&ring_clock,&ring.unpack);
        MPI_Recv(buffer,count,MPI_Type_DS_VNA,source,999,mpi_comm_level1,&status);
        SetPro_RingTick(ring_timed,&ring_clock,&ring.receive);
        size_t offset=0;
        for (int h=0;h<=FNAN[gc];++h) offset+=SetPro_RingRows(buffer+offset,DS_VNA[0][Matomnum+1][h],rows,components,1);
        SetPro_RingTick(ring_timed,&ring_clock,&ring.unpack);
        if (gpu) SetPro_HVNA_DispatchArchive(&hvna,DS_VNA,original_mc,gc);
        else SetPro_HVNA_CPU_Halo(&hvna_geometry,&projectors,&halo_workspace,&projectors_matrix,&matrices->potential,original_mc,gc,fp32,&halo_cache);
        SetPro_RingTick(ring_timed,&ring_clock,&ring.halo);
        ++received[source];
      }
      if (sending) { MPI_Wait(&request,&status);++sent[destination]; }
      SetPro_RingTick(ring_timed,&ring_clock,&ring.wait);
    }
    pending=0;
    for (int id=0;id<peers;++id)
      pending|=sent[id]<F_Snd_Num[id] || received[id]<F_Rcv_Num[id];
  } while (pending);
  openmx_core_hvna_workspace_destroy(halo_owner);
  SetPro_RingTick(phases_timed,&phase_clock,&phase_ring);
  SetPro_HvnaHost_Free(&halo_cache);
  if (ring_timed)
    SetPro_note("HVNA ring","size handshake %.3f s, pack %.3f s, send %.3f s, receive wait %.3f s, unpack %.3f s, "
        "halo %s %.3f s, send wait %.3f s; %ld messages, %.1f MiB sent",
        ring.sizes,ring.pack,ring.send,ring.receive,ring.unpack,gpu?"archive":"compute",ring.halo,ring.wait,
        ring.messages,ring.bytes/1048576.0);
  if (measure_time) { dtime(&communication_end);time4=communication_end-communication_begin;dtime(&communication_begin); }
  if (gpu) *used_gpu|=SetPro_HVNA_DispatchRun(&hvna,HVNA);
  else SetPro_HVNA_CPU_Local(&hvna_geometry,projector_owner,&projectors,&projectors_matrix,&matrices->potential,fp32);
  SetPro_RingTick(phases_timed,&phase_clock,&phase_local);
  if (phases_timed)
    SetPro_note("Separable phases","DS_VNA %.3f s, barrier wait %.3f s, ring %.3f s (incl. halo HVNA), local HVNA %.3f s",
        phase_dsvna,phase_barrier,phase_ring,phase_local);
  openmx_core_hvna_projectors_destroy(projector_owner);
  if (measure_time) {
    dtime(&communication_end);time5=communication_end-communication_begin;
    printf("Set_ProExpn_VNA myid=%2d time1=%7.3f time2=%7.3f time3=%7.3f time4=%7.3f time5=%7.3f\n",
        rank,time1,time2,time3,time4,time5);fflush(stdout);
  }
  OpenMX_TransferEnd(&transfer);
  dtime(&end);
  return end-begin;
}

static double SetPro_OneCenter(const OpenMXVnaMatrixViews *matrices,int mirror,
    const OpenMXVnaInvocationBuffers *invocation,const SetProVnaTables *tables,SetProGpu3Context *gpu)
{
  double *****matrix=mirror ? matrices->HVNA3 : matrices->HVNA2;
  const OpenMXDoubleBlockMatrix *flat=mirror ? &matrices->mirrored : &matrices->one_center;
  double begin,end;
  dtime(&begin);
  const OpenMXAtomPairBuffers *pairs=&invocation->pairs;
  const OpenMXVnaPreparationBuffers *basis=&invocation->preparation;
  if (gpu->enabled && !gpu->host_route) {
    SetPro_VNA23_GpuRun(matrix,mirror,pairs->count,pairs->local,pairs->neighbor,invocation->wavevectors,tables->weights,gpu);
    /* a late device failure (collective verdict) leaves this rank's remaining
       pairs, and the other orientation, to the host SIMD assembly */
    if (gpu->host_route && gpu->host_first<pairs->count) {
      OpenMXAtomPairBuffers rest;
      SetPro_PairTail(pairs,gpu->host_first,&rest);
      SetPro_VNA23_CPU(flat,mirror,invocation->wavevectors,basis,&rest,tables);
    }
  }
  else
    SetPro_VNA23_CPU(flat,mirror,invocation->wavevectors,basis,pairs,tables);
  if (!mirror) {
    /* Preserve h/i/j accumulation order for the on-site contribution. */
    for (int mc=1;mc<=pairs->local_atoms;++mc) {
      const size_t first=pairs->offsets[mc-1],last=pairs->offsets[mc];
      const int orbitals=basis->orbitals[pairs->central_species[first]];
      for (int i=0;i<orbitals;++i)
        for (int j=0;j<orbitals;++j) openmx_double_block_row(&matrices->potential,0,mc,0,i)[j]=0.0;
      for (size_t p=first;p<last;++p)
        for (int i=0;i<orbitals;++i)
          for (int j=0;j<orbitals;++j) openmx_double_block_row(&matrices->potential,0,mc,0,i)[j]+=openmx_double_block_row(flat,0,mc,pairs->neighbor[p],i)[j];
    }
  }
  dtime(&end);
  return end-begin;
}

/* This boundary borrows the legacy application state explicitly. Everything
   exported by invocation (basis, task maps, geometry, wavevectors) is owned. */
double Set_ProExpn_VNA_WithStatus(const OpenMXRealHarmonicData *conversion,const OpenMXGaussRuleData *radial,const OpenMXVnaMatrixViews *vna,const OpenMXReciprocalViews *reciprocal,int *used_gpu)
{
  const int conversion_max=List_YOUSO[25]>List_YOUSO[35]?List_YOUSO[25]:List_YOUSO[35];
  if (!conversion || !conversion->coefficients || conversion_max<0 ||
      conversion->orders<=(size_t)conversion_max || conversion->width<2*(size_t)conversion_max+1) {
    SetPro_gpu_abort("invalid real harmonic conversion table"); abort();
  }
  if (!radial || radial->count!=GL_Mesh || !radial->nodes || !radial->weights) {
    SetPro_gpu_abort("invalid VNA radial quadrature"); abort();
  }
  const OpenMXVnaInvocationInput input={
    {List_YOUSO[25],List_YOUSO[24],List_YOUSO[35],List_YOUSO[34],GL_Mesh,
     SpeciesNum,Spe_MaxL_Basis,Spe_Num_Basis},
    {atomnum,Matomnum,SpeciesNum,TCpyCell+1,M2G,WhatSpecies,FNAN,natn,ncn,Gxyz,atv},
    Radial_kmin,PAO_Nkmax,radial->nodes
  };
  OpenMXVnaInvocationBuffers invocation;
  char error[256];
  OpenMXVnaInvocation *owner=openmx_core_vna_invocation_create(&input,&invocation,error,sizeof(error));
  if (!owner) { SetPro_gpu_abort(error); abort(); }
  const SetProVnaTables tables={reciprocal->neutral_transform,reciprocal->product_transform,reciprocal->potential_transform,
      conversion->coefficients,conversion->width,
      radial->weights,Radial_kmin,PAO_Nkmax,time_per_atom};
  int hvna_gpu_used=0;
  if (used_gpu) *used_gpu=0;
  SetPro_note("entry","");
  const double time1=SetPro_Separable(vna,reciprocal,&invocation,&tables,&hvna_gpu_used);
  SetProGpu3Context gpu3={0};
  const int vna23_gpu=SetPro_VNA23_GpuBegin(reciprocal, conversion->coefficients,conversion->width, &gpu3);
  const double time2=SetPro_OneCenter(vna,0,&invocation,&tables,&gpu3);
  const double time3=SetPro_OneCenter(vna,1,&invocation,&tables,&gpu3);
  const int vna23_gpu_used=gpu3.used_gpu;
  if (vna23_gpu) SetPro_VNA23_GpuEnd(&gpu3);
  openmx_core_vna_invocation_destroy(owner);
  if (used_gpu) *used_gpu=hvna_gpu_used || vna23_gpu_used;
  SetPro_note("exit","Set_ProExpn %.2f s, Set_VNA2 %.2f s, Set_VNA3 %.2f s",time1,time2,time3);
  if (measure_time)
    printf("Time Set_ProExpn=%7.3f Set_VNA2=%7.3f Set_VNA3=%7.3f\n",time1,time2,time3);
  return time1+time2+time3;
}

double Set_ProExpn_VNA(const OpenMXRealHarmonicData *conversion,const OpenMXGaussRuleData *radial,const OpenMXVnaMatrixViews *vna,const OpenMXReciprocalViews *reciprocal)
{
  return Set_ProExpn_VNA_WithStatus(conversion,radial,vna,reciprocal,NULL);
}
#endif
