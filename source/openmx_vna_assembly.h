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
  if (SetPro_DSVNA_GpuBegin(reciprocal, &gpu2,&preparation)) {
    SetPro_DSVNA_GpuRun(DS_VNA,pairs->count,pairs->local,pairs->neighbor,wavevectors,&gpu2);
    SetPro_DSVNA_GpuEnd(&gpu2);
  }
  else
    SetPro_DSVNA_CPU(&projectors_matrix,preparation_owner,&preparation,wavevectors,pairs,tables,&time1,&time2,&time3);

  MPI_Barrier(mpi_comm_level1);
  if (measure_time) dtime(&communication_begin);
  SetProHvnaDispatch hvna={0};
  const int gpu=SetPro_HVNA_DispatchBegin(reciprocal, &hvna,DS_VNA,preparation_owner,preparation.projectors);
  const SetProHvnaGeometry hvna_geometry={
      Matomnum,M2G,F_G2M,WhatSpecies,FNAN,natn,RMI1,Spe_Atom_Cut1,Dis,tables->atom_seconds};
  OpenMXHvnaProjectorBuffers projectors;
  OpenMXHvnaProjectors *projector_owner=NULL;
  OpenMXHvnaWorkspaceBuffers halo_workspace;
  OpenMXHvnaWorkspace *halo_owner=NULL;
  if (!gpu) {
    char error[256];
    projector_owner=openmx_core_hvna_projectors_create(preparation_owner,reciprocal->vps.neutral_energies,&projectors,error,sizeof(error));
    if (!projector_owner) { SetPro_gpu_abort(error);abort(); }
    halo_owner=SetPro_HVNA_CPU_Workspace(projector_owner,&halo_workspace);
  }

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
    for (int id=0;id<peers;++id) {
      const int destination=(rank+id)%peers,source=(rank-id+peers)%peers;
      MPI_Request request;MPI_Status status;
      const int sending=sent[destination]<F_Snd_Num[destination];
      if (sending) {
        const int mc=Snd_MAN[destination][sent[destination]],gc=Snd_GAN[destination][sent[destination]];
        const int rows=preparation.orbitals[WhatSpecies[gc]];
        Type_DS_VNA *buffer=OpenMX_TransferBuffer(&transfer,OPENMX_EXCHANGE_SEND,send_sizes[destination]);
        int offset=0;
        for (int h=0;h<=FNAN[gc];++h)
          for (int i=0;i<rows;++i)
            for (int j=0;j<components;++j) buffer[offset++]=DS_VNA[0][mc][h][i][j];
        MPI_Isend(buffer,send_sizes[destination],MPI_Type_DS_VNA,destination,999,mpi_comm_level1,&request);
      }
      if (received[source]<F_Rcv_Num[source]) {
        const int gc=Rcv_GAN[source][received[source]],original_mc=F_TopMAN[source]+received[source];
        const int rows=preparation.orbitals[WhatSpecies[gc]];
        const int count=OpenMX_TransferCount(&transfer,FNAN[gc]+1,rows,components);
        if (receive_sizes[source]!=count) OpenMX_TransferFail(mpi_comm_level1,"inconsistent VNA receive count");
        Type_DS_VNA *buffer=OpenMX_TransferBuffer(&transfer,OPENMX_EXCHANGE_RECEIVE,count);
        MPI_Recv(buffer,count,MPI_Type_DS_VNA,source,999,mpi_comm_level1,&status);
        int offset=0;
        for (int h=0;h<=FNAN[gc];++h)
          for (int i=0;i<rows;++i)
            for (int j=0;j<components;++j) DS_VNA[0][Matomnum+1][h][i][j]=buffer[offset++];
        if (gpu) SetPro_HVNA_DispatchArchive(&hvna,DS_VNA,original_mc,gc);
        else SetPro_HVNA_CPU_Halo(&hvna_geometry,&projectors,&halo_workspace,&projectors_matrix,&matrices->potential,original_mc,gc,fp32);
        ++received[source];
      }
      if (sending) { MPI_Wait(&request,&status);++sent[destination]; }
    }
    pending=0;
    for (int id=0;id<peers;++id)
      pending|=sent[id]<F_Snd_Num[id] || received[id]<F_Rcv_Num[id];
  } while (pending);
  openmx_core_hvna_workspace_destroy(halo_owner);
  if (measure_time) { dtime(&communication_end);time4=communication_end-communication_begin;dtime(&communication_begin); }
  if (gpu) *used_gpu|=SetPro_HVNA_DispatchRun(&hvna,HVNA);
  else SetPro_HVNA_CPU_Local(&hvna_geometry,projector_owner,&projectors,&projectors_matrix,&matrices->potential,fp32);
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
  if (gpu->enabled)
    SetPro_VNA23_GpuRun(matrix,mirror,pairs->count,pairs->local,pairs->neighbor,invocation->wavevectors,gpu);
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
double Set_ProExpn_VNA_WithStatus(const OpenMXVnaMatrixViews *vna,const OpenMXReciprocalViews *reciprocal,int *used_gpu)
{
  const OpenMXVnaInvocationInput input={
    {List_YOUSO[25],List_YOUSO[24],List_YOUSO[35],List_YOUSO[34],GL_Mesh,
     SpeciesNum,Spe_MaxL_Basis,Spe_Num_Basis},
    {atomnum,Matomnum,SpeciesNum,TCpyCell+1,M2G,WhatSpecies,FNAN,natn,ncn,Gxyz,atv},
    Radial_kmin,PAO_Nkmax,GL_Abscissae
  };
  OpenMXVnaInvocationBuffers invocation;
  char error[256];
  OpenMXVnaInvocation *owner=openmx_core_vna_invocation_create(&input,&invocation,error,sizeof(error));
  if (!owner) { SetPro_gpu_abort(error); abort(); }
  const SetProVnaTables tables={reciprocal->neutral_transform,reciprocal->product_transform,reciprocal->potential_transform,
      &Comp2Real[0][0][0],sizeof(Comp2Real[0][0])/sizeof(Comp2Real[0][0][0]),
      GL_Weight,Radial_kmin,PAO_Nkmax,time_per_atom};
  int hvna_gpu_used=0;
  if (used_gpu) *used_gpu=0;
  SetPro_note("entry","");
  const double time1=SetPro_Separable(vna,reciprocal,&invocation,&tables,&hvna_gpu_used);
  SetProGpu3Context gpu3={0};
  const int vna23_gpu=SetPro_VNA23_GpuBegin(reciprocal, &gpu3);
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

double Set_ProExpn_VNA(const OpenMXVnaMatrixViews *vna,const OpenMXReciprocalViews *reciprocal)
{
  return Set_ProExpn_VNA_WithStatus(vna,reciprocal,NULL);
}
#endif
