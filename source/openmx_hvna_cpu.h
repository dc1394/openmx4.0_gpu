#ifndef OPENMX_HVNA_CPU_H
#define OPENMX_HVNA_CPU_H
#include "openmx_core_hvna_workspace.h"
#include "openmx_projector_matrix.h"
#include "openmx_host_simd.h"
#include "openmx_core_vna_simd.h"
#include <omp.h>

/* Transposed projector blocks for the Core's host HVNA kernel
   (openmx_core_vna_hvna_block): the local pass keeps the left blocks
   (mc, k) of the atom being assembled and runs lanes over rows, the halo pass
   keeps the right blocks (received atom, kl) and runs lanes over columns. A
   block is transposed on first use and reused by every pair of that atom.
   Legacy mode allocates nothing, and the legacy rows below stay in use. */
typedef struct {
  int mode, right_transposed, blocks, atom;
  size_t lane_stride, block_floats;
  float *values;
  int *ready;
} SetProHvnaHost;

static void SetPro_HvnaHost_Init(SetProHvnaHost *h,int right_transposed,int blocks,int components,int max_orbitals)
{
  h->mode=OpenMX_HostSimdMode();h->right_transposed=right_transposed;h->blocks=blocks;h->atom=-1;
  h->lane_stride=((size_t)max_orbitals+15)/16*16;h->block_floats=(size_t)components*h->lane_stride;
  h->values=NULL;h->ready=NULL;
  if (h->mode==OPENMX_HOST_SIMD_LEGACY || blocks<=0 || components<=0 || max_orbitals<=0) return;
  h->values=(float*)SetPro_checked_malloc(sizeof(float)*(h->block_floats*(size_t)blocks+1));
  h->ready=(int*)SetPro_checked_malloc(sizeof(int)*(size_t)blocks);
  memset(h->ready,0,sizeof(int)*(size_t)blocks);
}
static void SetPro_HvnaHost_Reset(SetProHvnaHost *h,int atom)
{
  if (h->ready && h->atom!=atom) { memset(h->ready,0,sizeof(int)*(size_t)h->blocks); h->atom=atom; }
}
static void SetPro_HvnaHost_Free(SetProHvnaHost *h) { free(h->values); free(h->ready); h->values=NULL; h->ready=NULL; }
/* The [component][lane] copy of a legacy [row][stride] block (zero padding lanes). */
static const float *SetPro_HvnaHost_Block(SetProHvnaHost *h,int index,const Type_DS_VNA *legacy,
    size_t stride,int rows,int components)
{
  float *block=h->values+(size_t)index*h->block_floats;
  if (!h->ready[index]) {
    int r,L;
    memset(block,0,sizeof(float)*h->block_floats);
    for (r=0; r<rows; ++r)
      for (L=0; L<components; ++L) block[(size_t)L*h->lane_stride+r]=(float)legacy[(size_t)r*stride+L];
    h->ready[index]=1;
  }
  return block;
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
  if (!host || !host->values || (host->right_transposed?kl:k)>=host->blocks) return 0;
  b.rows=rows;b.columns=columns;b.groups=projectors->groups;b.fp32=fp32;b.first=first;
  b.lengths=projectors->lengths;b.energies=projectors->energies+(size_t)projector_species*projectors->groups;
  b.matrix=workspace->matrix;b.stride=workspace->orbital_capacity;
  if (host->right_transposed) {
    b.right_transposed=1;b.left=left;b.left_stride=left_stride;
    b.right=SetPro_HvnaHost_Block(host,kl,right,right_stride,columns,projectors->components);
    b.right_stride=host->lane_stride;
    if (parallel_rows && rows>8 && omp_get_max_threads()>1) {
      int ok=1,m0;
#pragma omp parallel for reduction(&&:ok)
      for (m0=0; m0<rows; m0+=4) {
        OpenMXVnaHvnaBlock part=b;
        part.rows=rows-m0<4?rows-m0:4;
        part.left=b.left+(size_t)m0*b.left_stride;
        part.matrix=b.matrix+(size_t)m0*b.stride;
        ok=ok && openmx_core_vna_hvna_block(&part,host->mode);
      }
      return ok;
    }
    return openmx_core_vna_hvna_block(&b,host->mode);
  }
  b.right_transposed=0;b.right=right;b.right_stride=right_stride;
  b.left=SetPro_HvnaHost_Block(host,k,left,left_stride,rows,projectors->components);
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

static void SetPro_HVNA_CPU_Halo(const SetProHvnaGeometry *geometry,
    const OpenMXHvnaProjectorBuffers *projectors, const OpenMXHvnaWorkspaceBuffers *workspace,
    const OpenMXProjectorMatrix *ds, const OpenMXDoubleBlockMatrix *hvna,
    int original_mc, int received_global, int fp32)
{
  /* the received atom's blocks (Matomnum+1, kl) serve every local pair that points at it */
  SetProHvnaHost host;
  SetPro_HvnaHost_Init(&host,1,geometry->neighbor_counts[received_global]+1,projectors->components,
      SetPro_HVNA_MaxOrbitals(projectors));
  SetPro_HvnaHost_Reset(&host,received_global);
  for (int mc=1; mc<=geometry->local_atoms; ++mc) {
    double begin, end;
    dtime(&begin);
    const int gc = geometry->local_to_global[mc];
    for (int j=0; j<=geometry->neighbor_counts[gc]; ++j) {
      const int global = geometry->neighbors[gc][j];
      if (geometry->global_to_local[global]==original_mc)
        SetPro_HVNA_CPU_Pair(geometry,projectors,workspace,ds,hvna,mc,j,
            geometry->local_atoms+1,geometry->species[received_global],1,fp32,&host);
    }
    dtime(&end);
    geometry->atom_seconds[gc] += end-begin;
  }
  SetPro_HvnaHost_Free(&host);
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
    SetPro_HvnaHost_Init(&host,0,max_blocks,projectors->components,SetPro_HVNA_MaxOrbitals(projectors));
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
