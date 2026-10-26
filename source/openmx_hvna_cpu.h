#ifndef OPENMX_HVNA_CPU_H
#define OPENMX_HVNA_CPU_H
#include "openmx_core_hvna_workspace.h"
#include "openmx_projector_matrix.h"
#include "openmx_host_simd.h"
#include "openmx_core_vna_simd.h"
#include "openmx_setpro_host.h"
#include <omp.h>

/* Transposed projector blocks for the Core's host HVNA kernel
   (openmx_core_vna_hvna_block): the local pass keeps the left blocks
   (mc, k) of the atom being assembled and runs lanes over rows, the halo pass
   keeps the right blocks (received atom, kl) and runs lanes over columns. A
   block is transposed on first use and reused by every pair of that atom;
   the halo cache lives for the whole ring (one allocation, reset per received
   atom). Under fp32 the cache holds the float lanes; without fp32 it holds
   the lanes widened to double, and each call also widens the broadcast block
   once, so the kernel's inner loop needs no conversion. Double projector
   storage takes the double path under both settings: its product
   (double)(left*right) is already the double product. Padding lanes are
   never read (the kernel masks its tail lanes) and stay uninitialized.
   Legacy mode allocates nothing, and the legacy rows below stay in use. */
typedef struct {
  int mode, right_transposed, blocks, atom, fp64;
  size_t lane_stride, block_values, components;
  float *values;        /* fp32: [block][component][lane_stride] */
  double *dvalues;      /* fp64: the same blocks widened to double */
  double *broadcast;    /* fp64: one legacy block widened to double, [row][component] */
  int *ready;
} SetProHvnaHost;

/* Double operands: the products are double ones. */
static inline int SetPro_HvnaHost_Fp64(int fp32) { return !fp32 || sizeof(Type_DS_VNA)!=sizeof(float); }

static void SetPro_HvnaHost_Init(SetProHvnaHost *h,int right_transposed,int blocks,int components,
    int max_orbitals,int fp32)
{
  memset(h,0,sizeof(*h));
  h->mode=OpenMX_HostSimdMode();h->right_transposed=right_transposed;h->blocks=blocks;h->atom=-1;
  h->fp64=SetPro_HvnaHost_Fp64(fp32);h->components=(size_t)(components>0?components:0);
  h->lane_stride=((size_t)max_orbitals+15)/16*16;h->block_values=h->components*h->lane_stride;
  if (h->mode==OPENMX_HOST_SIMD_LEGACY || blocks<=0 || components<=0 || max_orbitals<=0) return;
  if (h->fp64) {
    h->dvalues=(double*)SetPro_HostAlloc(sizeof(double)*(h->block_values*(size_t)blocks+1));
    h->broadcast=(double*)SetPro_HostAlloc(sizeof(double)*((size_t)max_orbitals*h->components+1));
  }
  else h->values=(float*)SetPro_HostAlloc(sizeof(float)*(h->block_values*(size_t)blocks+1));
  h->ready=(int*)SetPro_HostAlloc(sizeof(int)*(size_t)blocks);
  memset(h->ready,0,sizeof(int)*(size_t)blocks);
}
static void SetPro_HvnaHost_Reset(SetProHvnaHost *h,int atom)
{
  if (h->ready && h->atom!=atom) { memset(h->ready,0,sizeof(int)*(size_t)h->blocks); h->atom=atom; }
}
static void SetPro_HvnaHost_Free(SetProHvnaHost *h)
{
  free(h->values); free(h->dvalues); free(h->broadcast); free(h->ready);
  h->values=NULL; h->dvalues=NULL; h->broadcast=NULL; h->ready=NULL;
}
/* The [component][lane] copy of a legacy [row][stride] block, written one
   contiguous lane row at a time. */
static const float *SetPro_HvnaHost_Lanes(SetProHvnaHost *h,int index,const Type_DS_VNA *legacy,
    size_t stride,int rows)
{
  float *block=h->values+(size_t)index*h->block_values;
  if (!h->ready[index]) {
    size_t L;
    int r;
    for (L=0; L<h->components; ++L) {
      float *lane=block+L*h->lane_stride;
      for (r=0; r<rows; ++r) lane[r]=(float)legacy[(size_t)r*stride+L];
    }
    h->ready[index]=1;
  }
  return block;
}
static const double *SetPro_HvnaHost_LanesDouble(SetProHvnaHost *h,int index,const Type_DS_VNA *legacy,
    size_t stride,int rows)
{
  double *block=h->dvalues+(size_t)index*h->block_values;
  if (!h->ready[index]) {
    size_t L;
    int r;
    for (L=0; L<h->components; ++L) {
      double *lane=block+L*h->lane_stride;
      for (r=0; r<rows; ++r) lane[r]=(double)legacy[(size_t)r*stride+L];
    }
    h->ready[index]=1;
  }
  return block;
}
/* The broadcast block of one call widened to double, rows [row][components]. */
static const double *SetPro_HvnaHost_Broadcast(SetProHvnaHost *h,const Type_DS_VNA *legacy,size_t stride,int rows)
{
  int r;
  size_t L;
  for (r=0; r<rows; ++r) {
    const Type_DS_VNA *in=legacy+(size_t)r*stride;
    double *out=h->broadcast+(size_t)r*h->components;
    for (L=0; L<h->components; ++L) out[L]=(double)in[L];
  }
  return h->broadcast;
}

/* Borrowed geometry for synchronous CPU assembly. Matrix values and shape
   metadata have calculation-scoped C++ owners. Atom/neighbor maps retain
   their legacy bases; no global lookup occurs inside the contraction. */
typedef struct {
  int local_atoms;
  const int *local_to_global, *global_to_local, *species, *neighbor_counts;
  int *const *neighbors;
  int **const *intersections;
  const double *cutoffs;
  double *const *distances;
  double *atom_seconds;
} SetProHvnaGeometry;

static OpenMXHvnaWorkspace *SetPro_HVNA_CPU_Workspace(const OpenMXHvnaProjectors *projectors,
    OpenMXHvnaWorkspaceBuffers *buffers)
{
  char error[256];
  OpenMXHvnaWorkspace *owner = openmx_core_hvna_workspace_create(projectors,buffers,error,sizeof(error));
  if (!owner) {
    SetPro_gpu_abort(error);
    abort();
  }
  return owner;
}

/* Preserve group/member summation order and the float-product rounding point.
   Each row writes a disjoint part of the worker's flat matrix. */
static inline void SetPro_HVNA_CPU_Row(const OpenMXHvnaProjectorBuffers *projectors,
    const Type_DS_VNA *left, size_t left_stride, const Type_DS_VNA *right, size_t right_stride,
    int species, int columns, int m,
    double *matrix, size_t stride, int first, int fp32)
{
  double *row = matrix+(size_t)m*stride;
  for (int n=0; n<columns; ++n) {
    double sum = 0.0;
    int L = 0;
    for (int group=0; group<projectors->groups; ++group) {
      const double ene = projectors->energies[(size_t)species*projectors->groups+group];
      double tmp0 = 0.0;
      for (int member=0; member<projectors->lengths[group]; ++member) {
        tmp0 += fp32 ? (double)(left[(size_t)m*left_stride+L]*right[(size_t)n*right_stride+L])
                     : (double)left[(size_t)m*left_stride+L]*(double)right[(size_t)n*right_stride+L];
        ++L;
      }
      sum += ene*tmp0;
    }
    if (first) row[n] = sum;
    else row[n] += sum;
  }
}

/* One (mc, k) x (partner, kl) block through the Core kernel; zero when the
   kernel declines (nothing written), so the caller keeps the legacy rows. */
static int SetPro_HVNA_Host_Block(SetProHvnaHost *host,const OpenMXHvnaProjectorBuffers *projectors,
    const OpenMXHvnaWorkspaceBuffers *workspace,int rows,int columns,int projector_species,
    const Type_DS_VNA *left,size_t left_stride,int k,const Type_DS_VNA *right,size_t right_stride,int kl,
    int first,int parallel_rows,int fp32)
{
  OpenMXVnaHvnaBlock b;
  if (!host || !host->ready || (host->right_transposed?kl:k)>=host->blocks ||
      host->fp64!=SetPro_HvnaHost_Fp64(fp32)) return 0;
  memset(&b,0,sizeof(b));
  b.rows=rows;b.columns=columns;b.groups=projectors->groups;b.fp32=!host->fp64;b.first=first;
  b.lengths=projectors->lengths;b.energies=projectors->energies+(size_t)projector_species*projectors->groups;
  b.matrix=workspace->matrix;b.stride=workspace->orbital_capacity;
  if (host->right_transposed) {
    b.right_transposed=1;
    if (host->fp64) {
      b.right_double=SetPro_HvnaHost_LanesDouble(host,kl,right,right_stride,columns);
      b.left_double=SetPro_HvnaHost_Broadcast(host,left,left_stride,rows);
      b.left_stride=host->components;
    }
    else {
      b.right=SetPro_HvnaHost_Lanes(host,kl,right,right_stride,columns);
      b.left=(const float*)(const void*)left;b.left_stride=left_stride;   /* float storage here */
    }
    b.right_stride=host->lane_stride;
    if (parallel_rows && rows>8 && omp_get_max_threads()>1) {
      int ok=1,m0;
#pragma omp parallel for reduction(&&:ok)
      for (m0=0; m0<rows; m0+=4) {
        OpenMXVnaHvnaBlock part=b;
        part.rows=rows-m0<4?rows-m0:4;
        if (part.left) part.left=b.left+(size_t)m0*b.left_stride;
        if (part.left_double) part.left_double=b.left_double+(size_t)m0*b.left_stride;
        part.matrix=b.matrix+(size_t)m0*b.stride;
        ok=ok && openmx_core_vna_hvna_block(&part,host->mode);
      }
      return ok;
    }
    return openmx_core_vna_hvna_block(&b,host->mode);
  }
  b.right_transposed=0;
  if (host->fp64) {
    b.left_double=SetPro_HvnaHost_LanesDouble(host,k,left,left_stride,rows);
    b.right_double=SetPro_HvnaHost_Broadcast(host,right,right_stride,columns);
    b.right_stride=host->components;
  }
  else {
    b.left=SetPro_HvnaHost_Lanes(host,k,left,left_stride,rows);
    b.right=(const float*)(const void*)right;b.right_stride=right_stride;   /* float storage here */
  }
  b.left_stride=host->lane_stride;
  return openmx_core_vna_hvna_block(&b,host->mode);
}

static void SetPro_HVNA_CPU_Pair(const SetProHvnaGeometry *geometry,
    const OpenMXHvnaProjectorBuffers *projectors, const OpenMXHvnaWorkspaceBuffers *workspace,
    const OpenMXProjectorMatrix *ds, const OpenMXDoubleBlockMatrix *hvna,
    int mc, int j, int partner, int partner_species,
    int parallel_rows, int fp32, SetProHvnaHost *host)
{
  const int gc = geometry->local_to_global[mc];
  const int species = geometry->species[gc];
  const int rows = projectors->orbitals[species], columns = projectors->orbitals[partner_species];
  int initialized = 0;
  for (int k=0; k<=geometry->neighbor_counts[gc]; ++k) {
    const int kl = geometry->intersections[mc][j][k];
    if (kl<0) continue;
    const int projector_species = geometry->species[geometry->neighbors[gc][k]];
    const Type_DS_VNA *left=openmx_projector_block_row(ds,0,mc,k,0);
    const Type_DS_VNA *right=openmx_projector_block_row(ds,0,partner,kl,0);
    const size_t left_stride=(size_t)ds->layout.shapes[2*openmx_block_index(&ds->layout,mc,k)+1];
    const size_t right_stride=(size_t)ds->layout.shapes[2*openmx_block_index(&ds->layout,partner,kl)+1];
    if (SetPro_HVNA_Host_Block(host,projectors,workspace,rows,columns,projector_species,
            left,left_stride,k,right,right_stride,kl,!initialized,parallel_rows,fp32)) {
      /* the Core kernel wrote this block's contribution */
    }
    else if (parallel_rows && rows>8) {
#pragma omp parallel for
      for (int m=0; m<rows; ++m)
        SetPro_HVNA_CPU_Row(projectors,left,left_stride,right,right_stride,projector_species,
            columns,m,workspace->matrix,workspace->orbital_capacity,!initialized,fp32);
    }
    else {
      for (int m=0; m<rows; ++m)
        SetPro_HVNA_CPU_Row(projectors,left,left_stride,right,right_stride,projector_species,
            columns,m,workspace->matrix,workspace->orbital_capacity,!initialized,fp32);
    }
    initialized = 1;
  }
  const double rcut = geometry->cutoffs[species]+geometry->cutoffs[partner_species];
  const double dmp = dampingF(rcut,geometry->distances[gc][j]);
  for (int m=0; m<rows; ++m)
    for (int n=0; n<columns; ++n)
      openmx_double_block_row(hvna,0,mc,j,m)[n] = dmp*(initialized ? workspace->matrix[(size_t)m*workspace->orbital_capacity+n] : 0.0);
}

static int SetPro_HVNA_MaxOrbitals(const OpenMXHvnaProjectorBuffers *projectors)
{
  int s,max_orbitals=0;
  for (s=0; s<projectors->species; ++s)
    if (max_orbitals<projectors->orbitals[s]) max_orbitals=projectors->orbitals[s];
  return max_orbitals;
}

/* The ring's halo cache: blocks for the largest neighbour list of any atom
   (a received atom can be any atom), allocated once per Set_ProExpn_VNA. */
static void SetPro_HVNA_CPU_HaloCache(SetProHvnaHost *host,const SetProHvnaGeometry *geometry,
    const OpenMXHvnaProjectorBuffers *projectors,int atoms,int fp32)
{
  int max_blocks=0;
  for (int gc=1; gc<=atoms; ++gc)
    if (max_blocks<geometry->neighbor_counts[gc]+1) max_blocks=geometry->neighbor_counts[gc]+1;
  SetPro_HvnaHost_Init(host,1,max_blocks,projectors->components,SetPro_HVNA_MaxOrbitals(projectors),fp32);
}

static void SetPro_HVNA_CPU_Halo(const SetProHvnaGeometry *geometry,
    const OpenMXHvnaProjectorBuffers *projectors, const OpenMXHvnaWorkspaceBuffers *workspace,
    const OpenMXProjectorMatrix *ds, const OpenMXDoubleBlockMatrix *hvna,
    int original_mc, int received_global, int fp32, SetProHvnaHost *host)
{
  /* the received atom's blocks (Matomnum+1, kl) serve every local pair that points at it */
  if (host->ready && geometry->neighbor_counts[received_global]+1>host->blocks) host=NULL;
  else SetPro_HvnaHost_Reset(host,received_global);
  for (int mc=1; mc<=geometry->local_atoms; ++mc) {
    double begin, end;
    dtime(&begin);
    const int gc = geometry->local_to_global[mc];
    for (int j=0; j<=geometry->neighbor_counts[gc]; ++j) {
      const int global = geometry->neighbors[gc][j];
      if (geometry->global_to_local[global]==original_mc)
        SetPro_HVNA_CPU_Pair(geometry,projectors,workspace,ds,hvna,mc,j,
            geometry->local_atoms+1,geometry->species[received_global],1,fp32,host);
    }
    dtime(&end);
    geometry->atom_seconds[gc] += end-begin;
  }
}

static void SetPro_HVNA_CPU_Local(const SetProHvnaGeometry *geometry,
    const OpenMXHvnaProjectors *owner, const OpenMXHvnaProjectorBuffers *projectors,
    const OpenMXProjectorMatrix *ds, const OpenMXDoubleBlockMatrix *hvna, int fp32)
{
  int max_blocks=0;
  for (int mc=1; mc<=geometry->local_atoms; ++mc)
    if (max_blocks<geometry->neighbor_counts[geometry->local_to_global[mc]]+1)
      max_blocks=geometry->neighbor_counts[geometry->local_to_global[mc]]+1;
#pragma omp parallel
  {
    OpenMXHvnaWorkspaceBuffers workspace;
    OpenMXHvnaWorkspace *worker = SetPro_HVNA_CPU_Workspace(owner,&workspace);
    const int thread = omp_get_thread_num(), threads = omp_get_num_threads();
    /* this worker's atom: its left blocks (mc, k) serve all of its pairs */
    SetProHvnaHost host;
    SetPro_HvnaHost_Init(&host,0,max_blocks,projectors->components,SetPro_HVNA_MaxOrbitals(projectors),fp32);
    for (int mc=thread*geometry->local_atoms/threads+1;
         mc<(thread+1)*geometry->local_atoms/threads+1; ++mc) {
      double begin, end;
      dtime(&begin);
      const int gc = geometry->local_to_global[mc];
      SetPro_HvnaHost_Reset(&host,mc);
      for (int j=0; j<=geometry->neighbor_counts[gc]; ++j) {
        const int global = geometry->neighbors[gc][j];
        const int partner = geometry->global_to_local[global];
        if (partner<=geometry->local_atoms)
          SetPro_HVNA_CPU_Pair(geometry,projectors,&workspace,ds,hvna,mc,j,
              partner,geometry->species[global],0,fp32,&host);
      }
      dtime(&end);
      geometry->atom_seconds[gc] += end-begin;
    }
    SetPro_HvnaHost_Free(&host);
    openmx_core_hvna_workspace_destroy(worker);
  }
}
#endif
