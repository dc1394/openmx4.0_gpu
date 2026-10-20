#ifndef OPENMX_HVNA_CPU_H
#define OPENMX_HVNA_CPU_H
#include "openmx_core_hvna_workspace.h"
#include "openmx_projector_matrix.h"
#include <omp.h>

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

static void SetPro_HVNA_CPU_Pair(const SetProHvnaGeometry *geometry,
    const OpenMXHvnaProjectorBuffers *projectors, const OpenMXHvnaWorkspaceBuffers *workspace,
    const OpenMXProjectorMatrix *ds, const OpenMXDoubleBlockMatrix *hvna,
    int mc, int j, int partner, int partner_species,
    int parallel_rows, int fp32)
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
    if (parallel_rows && rows>8) {
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

static void SetPro_HVNA_CPU_Halo(const SetProHvnaGeometry *geometry,
    const OpenMXHvnaProjectorBuffers *projectors, const OpenMXHvnaWorkspaceBuffers *workspace,
    const OpenMXProjectorMatrix *ds, const OpenMXDoubleBlockMatrix *hvna,
    int original_mc, int received_global, int fp32)
{
  for (int mc=1; mc<=geometry->local_atoms; ++mc) {
    double begin, end;
    dtime(&begin);
    const int gc = geometry->local_to_global[mc];
    for (int j=0; j<=geometry->neighbor_counts[gc]; ++j) {
      const int global = geometry->neighbors[gc][j];
      if (geometry->global_to_local[global]==original_mc)
        SetPro_HVNA_CPU_Pair(geometry,projectors,workspace,ds,hvna,mc,j,
            geometry->local_atoms+1,geometry->species[received_global],1,fp32);
    }
    dtime(&end);
    geometry->atom_seconds[gc] += end-begin;
  }
}

static void SetPro_HVNA_CPU_Local(const SetProHvnaGeometry *geometry,
    const OpenMXHvnaProjectors *owner, const OpenMXHvnaProjectorBuffers *projectors,
    const OpenMXProjectorMatrix *ds, const OpenMXDoubleBlockMatrix *hvna, int fp32)
{
#pragma omp parallel
  {
    OpenMXHvnaWorkspaceBuffers workspace;
    OpenMXHvnaWorkspace *worker = SetPro_HVNA_CPU_Workspace(owner,&workspace);
    const int thread = omp_get_thread_num(), threads = omp_get_num_threads();
    for (int mc=thread*geometry->local_atoms/threads+1;
         mc<(thread+1)*geometry->local_atoms/threads+1; ++mc) {
      double begin, end;
      dtime(&begin);
      const int gc = geometry->local_to_global[mc];
      for (int j=0; j<=geometry->neighbor_counts[gc]; ++j) {
        const int global = geometry->neighbors[gc][j];
        const int partner = geometry->global_to_local[global];
        if (partner<=geometry->local_atoms)
          SetPro_HVNA_CPU_Pair(geometry,projectors,&workspace,ds,hvna,mc,j,
              partner,geometry->species[global],0,fp32);
      }
      dtime(&end);
      geometry->atom_seconds[gc] += end-begin;
    }
    openmx_core_hvna_workspace_destroy(worker);
  }
}
#endif
