/**********************************************************************
  Set_Density_Grid_GPU.c:

  A single-owner GPU service for Set_Density_Grid.  The owner gathers
  the unique atom-centred orbital grids once per geometry, gathers only
  CDM on subsequent SCF iterations, constructs the density on the full
  regular grid, and scatters the contiguous B partitions to all ranks.
***********************************************************************/

#include "mpi.h"
#include "openmx_common.h"
#include "openmx_env.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_views.h"
#include "openmx_host_simd.h"
#include "grid_precision.h"
#include "set_cuda_default_device_from_local_rank.h"
#include "openmx_gpu_bilinear.h"
#include <accel.h>
#include <limits.h>
#include <openacc.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(uint32_t) == sizeof(unsigned int), "uint32_t must match MPI_UNSIGNED");
_Static_assert(sizeof(uint16_t) == sizeof(unsigned short), "uint16_t must match MPI_UNSIGNED_SHORT");

enum {
  SDG_COUNT_ATOMS = 0,
  SDG_COUNT_ORBS,
  SDG_COUNT_OUTPUTS,
  SDG_COUNT_PAIRS,
  SDG_COUNT_TERMS,
  SDG_COUNT_CDM,
  SDG_COUNT_FIELDS
};

typedef struct {
  int ready;
  int unavailable;
  int owner;
  int spin_count;
  int nh_bits;
  uint32_t nh_mask;
  int local_cdm_count;
  int device_resident;

  size_t atom_count;
  size_t orbital_count;
  size_t output_count;
  size_t pair_count;
  size_t term_count;
  size_t cdm_count;
  size_t global_grid_count;
  size_t device_bytes;

  uint32_t *atom_out_offset;
  uint32_t *atom_orb_offset;
  uint32_t *atom_pair_offset;
  int *atom_no;

  int *pair_neighbor;
  uint32_t *pair_cdm_offset;

  uint16_t *out_atom;
  uint32_t *out_ptr;
  uint32_t *terms;
  uint32_t *grid_ptr;
  uint32_t *grid_out;
  Type_Orbs_Grid *orbs;
  double *cdm;
  double *rho;

  int *cdm_counts;
  int *cdm_displs;
  int *scatter_counts;
  int *scatter_displs;
} SetDensityGpuCache;

static SetDensityGpuCache SDG_cache = {0};

typedef struct {int mc,h,point;} SDGRecoveryTerm;
typedef struct {
  int cnt_kind,rank,spins;
  size_t rows,*atom_offset,*ptr;
  SDGRecoveryTerm *terms;
  double *****dm,***density;
} SDGRecovery;

static int SDG_recovery_visit(void *context,size_t row,OpenMXBilinearEmit emit,void *target)
{
  SDGRecovery *c=context;
  for (size_t k=c->ptr[row];k<c->ptr[row+1];k++) {
    SDGRecoveryTerm t=c->terms[k];
    int gc=M2G[t.mc],gh=natn[gc][t.h],mh=F_G2M[gh];
    int nc=GListTAtoms1[t.mc][t.h][t.point],nh=GListTAtoms2[t.mc][t.h][t.point];
    int no0=Spe_Total_CNO[WhatSpecies[gc]],no1=Spe_Total_CNO[WhatSpecies[gh]];
    OpenMXBilinearTerm term={0};
    term.left=(OpenMXBilinearSpan){NULL,NULL,0,1,no0,sizeof(Type_Orbs_Grid)};
    term.right=(OpenMXBilinearSpan){NULL,NULL,0,1,no1,sizeof(Type_Orbs_Grid)};
    if (c->cnt_kind==0 && Cnt_switch==1) {
      term.left.rows=COrbs_Grid[t.mc];term.left.offset=nc;
      term.right.rows=COrbs_Grid[mh];term.right.offset=nh;
    } else {
      term.left.data=Orbs_Grid[t.mc][nc];
      term.right.data=G2ID[gh]==c->rank?Orbs_Grid[mh][nh]:Orbs_Grid_FNAN[t.mc][t.h][t.point];
    }
    for (int spin=0;spin<c->spins;spin++) term.matrix_rows[spin]=c->dm[spin][t.mc][t.h];
    if (!emit(target,&term)) return 0;
  }
  return 1;
}

static void SDG_recovery_commit(void *context,size_t first,size_t last,const double *result)
{
  SDGRecovery *c=context;
  for (int mc=1;mc<=Matomnum;mc++) {
    size_t lo=c->atom_offset[mc-1],hi=c->atom_offset[mc];
    if (lo<first) lo=first;
    if (hi>last) hi=last;
    for (size_t row=lo;row<hi;row++) for (int spin=0;spin<c->spins;spin++)
      c->density[spin][mc][row-c->atom_offset[mc-1]]=result[(row-first)*(size_t)c->spins+spin];
  }
}

/* Recovery uses the host orbital tables already required by the CPU path.
   Only CSR indices are added. Packing deduplicates the orbital vectors and
   DM blocks needed by a shard, so no GPU holds the complete rank's tables. */
int Set_Density_Grid_GPU_Capacity(int cnt_kind,int local_done,double *****dm,double ***density)
{
  SDGRecovery c={0};c.cnt_kind=cnt_kind;c.spins=SpinP_switch+1;c.dm=dm;c.density=density;
  MPI_Comm_rank(mpi_comm_level1,&c.rank);
  int enabled=scf_eigen_lib_flag==GPUSOLVER &&
    openmx_env_flag_at(OPENMX_ENV_DENSITY_GRID_GPU, openmx_env_flag_at(OPENMX_ENV_SETDENSITY_GPU, 1));
  int pending=enabled && !local_done,any=0;
  MPI_Allreduce(&pending,&any,1,MPI_INT,MPI_MAX,mpi_comm_level1);
  if (!any) return local_done;
  int ready=1;
  if (pending) {
    c.atom_offset=calloc((size_t)Matomnum+1,sizeof(size_t));
    ready=c.atom_offset!=NULL;
    for (int mc=1;ready && mc<=Matomnum;mc++) {
      size_t n=(size_t)GridN_Atom[M2G[mc]];
      if (c.atom_offset[mc-1]>SIZE_MAX-n) ready=0;
      else c.atom_offset[mc]=c.atom_offset[mc-1]+n;
    }
    if (ready) {
      c.rows=c.atom_offset[Matomnum];
      if (c.rows>SIZE_MAX/sizeof(size_t)-1) ready=0;
      else {c.ptr=calloc(c.rows+1,sizeof(size_t));ready=c.ptr!=NULL;}
    }
    size_t terms=0;
    for (int mc=1;ready && mc<=Matomnum;mc++) for (int h=0;ready && h<=FNAN[M2G[mc]];h++) {
      size_t n=(size_t)NumOLG[mc][h];
      if (terms>SIZE_MAX/sizeof(SDGRecoveryTerm)-n) {ready=0;break;}
      terms+=n;
      for (int point=0;point<NumOLG[mc][h];point++) {
        size_t row=c.atom_offset[mc-1]+(size_t)GListTAtoms1[mc][h][point];
        if (row>=c.atom_offset[mc]) {ready=0;break;}
        c.ptr[row+1]++;
      }
    }
    if (ready) {c.terms=malloc((terms?terms:1)*sizeof(SDGRecoveryTerm));ready=c.terms!=NULL;}
    if (ready) {
      for (size_t row=1;row<=c.rows;row++) c.ptr[row]+=c.ptr[row-1];
      for (int mc=1;mc<=Matomnum;mc++) for (int h=0;h<=FNAN[M2G[mc]];h++)
        for (int point=0;point<NumOLG[mc][h];point++) {
          size_t row=c.atom_offset[mc-1]+(size_t)GListTAtoms1[mc][h][point];
          c.terms[c.ptr[row]++]=(SDGRecoveryTerm){mc,h,point};
        }
      for (size_t row=c.rows;row>0;row--) c.ptr[row]=c.ptr[row-1];
      c.ptr[0]=0;
    }
  }
  OpenMXBilinearSource source={ready?c.rows:0,c.spins,0,&c,SDG_recovery_visit,SDG_recovery_commit};
  int done=openmx_gpu_bilinear_execute(mpi_comm_level1,&source,
      local_done?1:!enabled || !ready?-1:0,enabled,"density-capacity");
  free(c.atom_offset);free(c.ptr);free(c.terms);
  return local_done || (pending && ready && done);
}

static int SDG_bits_for(unsigned int max_value)
{
  int bits = 0;
  while (max_value != 0U) {
    bits++;
    max_value >>= 1;
  }
  return (bits == 0) ? 1 : bits;
}

static int SDG_add_bytes(size_t *total, size_t count, size_t element_size)
{
  size_t bytes;
  if (count != 0 && element_size > SIZE_MAX / count) return 0;
  bytes = count * element_size;
  if (bytes > SIZE_MAX - *total) return 0;
  *total += bytes;
  return 1;
}

static void *SDG_malloc(size_t count, size_t element_size)
{
  size_t bytes;
  if (count != 0 && element_size > SIZE_MAX / count) return NULL;
  bytes = count * element_size;
  return malloc(bytes == 0 ? 1 : bytes);
}

/* Recoverable allocations for both retained CSR data and transient tables.
   The OpenACC present table borrows CUDA storage; no implicit allocator is
   allowed to abort a capacity retry. */
static void SDG_acc_delete_if_present(void *host_ptr, size_t bytes)
{
  void *device = host_ptr && bytes ? acc_deviceptr(host_ptr) : NULL;
  if (device) {
    acc_unmap_data(host_ptr);
    if (cudaFree(device) != cudaSuccess) {
      fprintf(stderr,"density-grid CUDA buffer release failed.\n");
      MPI_Abort(mpi_comm_level1,1);
      abort();
    }
  }
}

static void *SDG_map(void *host, size_t bytes, int copy)
{
  void *device;
  if (!bytes) return host ? host : (void *)1;
  if (!host) return NULL;
  device=acc_deviceptr(host);
  if (device) return device;
  if (cudaMalloc(&device,bytes)!=cudaSuccess) {cudaGetLastError();return NULL;}
  if (copy && cudaMemcpy(device,host,bytes,cudaMemcpyHostToDevice)!=cudaSuccess) {
    cudaFree(device);cudaGetLastError();return NULL;
  }
  acc_map_data(host,device,bytes);
  return device;
}
static void *SDG_copyin(void *host,size_t bytes) {return SDG_map(host,bytes,1);}
static void *SDG_create(void *host,size_t bytes) {return SDG_map(host,bytes,0);}

static void SDG_delete_device(SetDensityGpuCache *cache)
{
  if (!cache->device_resident) return;

  /* Upload can fail after only a prefix of these mappings was created.
     Query the present table so the same cleanup handles full and partial
     uploads without attempting to delete an absent mapping. */
  SDG_acc_delete_if_present(cache->orbs, cache->orbital_count * sizeof(Type_Orbs_Grid));
  SDG_acc_delete_if_present(cache->atom_out_offset, (cache->atom_count + 1) * sizeof(uint32_t));
  SDG_acc_delete_if_present(cache->atom_orb_offset, (cache->atom_count + 1) * sizeof(uint32_t));
  SDG_acc_delete_if_present(cache->atom_pair_offset, (cache->atom_count + 1) * sizeof(uint32_t));
  SDG_acc_delete_if_present(cache->atom_no, cache->atom_count * sizeof(int));
  SDG_acc_delete_if_present(cache->pair_neighbor, cache->pair_count * sizeof(int));
  SDG_acc_delete_if_present(cache->pair_cdm_offset, cache->pair_count * sizeof(uint32_t));
  SDG_acc_delete_if_present(cache->out_atom, cache->output_count * sizeof(uint16_t));
  SDG_acc_delete_if_present(cache->out_ptr, (cache->output_count + 1) * sizeof(uint32_t));
  SDG_acc_delete_if_present(cache->terms, cache->term_count * sizeof(uint32_t));
  SDG_acc_delete_if_present(cache->grid_ptr, (cache->global_grid_count + 1) * sizeof(uint32_t));
  SDG_acc_delete_if_present(cache->grid_out, cache->output_count * sizeof(uint32_t));
  SDG_acc_delete_if_present(cache->cdm, cache->cdm_count * sizeof(double));
  SDG_acc_delete_if_present(cache->rho, cache->global_grid_count * (size_t)cache->spin_count * sizeof(double));
  acc_wait_all();

  /* NVHPC keeps storage released by acc_delete in an OpenACC freelist.  The
     following Band/Cluster eigensolver uses cudaMemGetInfo() to decide whether
     GEMMul8 can allocate its workspace, so leaving the roughly multi-GiB
     density epoch in that freelist makes the memory appear unavailable and
     can force a much slower native-cuBLAS path.  Return the cached blocks to CUDA
     before the next SCF diagonalization. */
  acc_clear_freelists();
  cache->device_resident = 0;
}

static void SDG_free_host(SetDensityGpuCache *cache)
{
  SDG_delete_device(cache);
  free(cache->atom_out_offset);
  free(cache->atom_orb_offset);
  free(cache->atom_pair_offset);
  free(cache->atom_no);
  free(cache->pair_neighbor);
  free(cache->pair_cdm_offset);
  free(cache->out_atom);
  free(cache->out_ptr);
  free(cache->terms);
  free(cache->grid_ptr);
  free(cache->grid_out);
  free(cache->orbs);
  free(cache->cdm);
  free(cache->rho);
  free(cache->cdm_counts);
  free(cache->cdm_displs);
  free(cache->scatter_counts);
  free(cache->scatter_displs);
  memset(cache, 0, sizeof(*cache));
}

static void SDG_local_free(void);

void Set_Density_Grid_GPU_Invalidate(void)
{
  SDG_free_host(&SDG_cache);
  SDG_local_free();
}

static void SDG_disable_current_epoch(SetDensityGpuCache *cache)
{
  SDG_free_host(cache);
  cache->unavailable = 1;
}

static int SDG_make_counts(const unsigned long long *rank_counts, int numprocs, int field,
                           int **counts_out, int **displs_out, size_t *total_out)
{
  int *counts = NULL;
  int *displs = NULL;
  size_t total = 0;
  int rank;

  counts = (int *)SDG_malloc((size_t)numprocs, sizeof(int));
  displs = (int *)SDG_malloc((size_t)numprocs, sizeof(int));
  if (counts == NULL || displs == NULL) {
    free(counts);
    free(displs);
    return 0;
  }

  for (rank = 0; rank < numprocs; rank++) {
    unsigned long long count = rank_counts[(size_t)rank * SDG_COUNT_FIELDS + (size_t)field];
    if (count > (unsigned long long)INT_MAX || total > (size_t)INT_MAX || count > (unsigned long long)((size_t)INT_MAX - total)) {
      free(counts);
      free(displs);
      return 0;
    }
    counts[rank] = (int)count;
    displs[rank] = (int)total;
    total += (size_t)count;
  }

  *counts_out = counts;
  *displs_out = displs;
  *total_out = total;
  return 1;
}

static int SDG_compute_local_counts(unsigned long long counts[SDG_COUNT_FIELDS], int spin_count,
                                    int *max_grid, int *max_h)
{
  int Mc_AN, h_AN, Gc_AN, Gh_AN, Cwan, Hwan;
  size_t atoms = 0, orbs = 0, outputs = 0, pairs = 0, terms = 0, cdm = 0;

  *max_grid = 0;
  *max_h = 0;

  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
    int NO0;
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    NO0 = Spe_Total_CNO[Cwan];
    atoms++;
    outputs += (size_t)GridN_Atom[Gc_AN];
    orbs += (size_t)GridN_Atom[Gc_AN] * (size_t)NO0;
    if (*max_grid < GridN_Atom[Gc_AN]) *max_grid = GridN_Atom[Gc_AN];
    if (*max_h < FNAN[Gc_AN] + 1) *max_h = FNAN[Gc_AN] + 1;

    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
      int NO1;
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      NO1 = Spe_Total_CNO[Hwan];
      pairs++;
      terms += (size_t)NumOLG[Mc_AN][h_AN];
      cdm += (size_t)spin_count * (size_t)NO0 * (size_t)NO1;
    }
  }

  if (atoms > ULLONG_MAX || orbs > ULLONG_MAX || outputs > ULLONG_MAX || pairs > ULLONG_MAX ||
      terms > ULLONG_MAX || cdm > ULLONG_MAX) return 0;
  counts[SDG_COUNT_ATOMS] = (unsigned long long)atoms;
  counts[SDG_COUNT_ORBS] = (unsigned long long)orbs;
  counts[SDG_COUNT_OUTPUTS] = (unsigned long long)outputs;
  counts[SDG_COUNT_PAIRS] = (unsigned long long)pairs;
  counts[SDG_COUNT_TERMS] = (unsigned long long)terms;
  counts[SDG_COUNT_CDM] = (unsigned long long)cdm;
  return 1;
}

static int SDG_build_cache(int spin_count, int owner, int myid, int numprocs)
{
  SetDensityGpuCache *cache = &SDG_cache;
  unsigned long long local_counts[SDG_COUNT_FIELDS] = {0};
  unsigned long long *rank_counts = NULL;
  int local_max_grid = 0, local_max_h = 0, max_grid = 0, max_h = 0;
  int *atom_counts = NULL, *atom_displs = NULL;
  int *orb_counts = NULL, *orb_displs = NULL;
  int *out_counts = NULL, *out_displs = NULL;
  int *term_counts = NULL, *term_displs = NULL;
  int *pair_counts = NULL, *pair_displs = NULL;
  int *cdm_counts = NULL, *cdm_displs = NULL;
  size_t total_atoms = 0, total_orbs = 0, total_outputs = 0;
  size_t total_pairs = 0, total_terms = 0, total_cdm = 0;
  int ok = 1, all_ok = 1;
  int *local_atom_ids = NULL, *atom_ids = NULL;
  Type_Orbs_Grid *local_orbs = NULL;
  uint32_t *local_out_gn = NULL, *out_gn = NULL;
  uint16_t *local_degree = NULL, *degree = NULL;
  uint32_t *local_ptr = NULL, *local_cursor = NULL, *local_terms = NULL;
  int nh_bits = 0, h_bits = 0;
  uint32_t nh_mask = 0;
  int Mc_AN, h_AN, Gc_AN, Gh_AN, Nc, Nog, i;
  size_t pos;

  if (!SDG_compute_local_counts(local_counts, spin_count, &local_max_grid, &local_max_h)) ok = 0;
  MPI_Allreduce(&local_max_grid, &max_grid, 1, MPI_INT, MPI_MAX, mpi_comm_level1);
  MPI_Allreduce(&local_max_h, &max_h, 1, MPI_INT, MPI_MAX, mpi_comm_level1);

  if (myid == owner) rank_counts = (unsigned long long *)SDG_malloc((size_t)numprocs * SDG_COUNT_FIELDS,
                                                                    sizeof(unsigned long long));
  if (myid == owner && rank_counts == NULL) ok = 0;
  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
  if (!all_ok) {
    free(rank_counts);
    return 0;
  }

  MPI_Gather(local_counts, SDG_COUNT_FIELDS, MPI_UNSIGNED_LONG_LONG,
             rank_counts, SDG_COUNT_FIELDS, MPI_UNSIGNED_LONG_LONG, owner, mpi_comm_level1);

  if (myid == owner) {
    ok = SDG_make_counts(rank_counts, numprocs, SDG_COUNT_ATOMS, &atom_counts, &atom_displs, &total_atoms) &&
         SDG_make_counts(rank_counts, numprocs, SDG_COUNT_ORBS, &orb_counts, &orb_displs, &total_orbs) &&
         SDG_make_counts(rank_counts, numprocs, SDG_COUNT_OUTPUTS, &out_counts, &out_displs, &total_outputs) &&
         SDG_make_counts(rank_counts, numprocs, SDG_COUNT_PAIRS, &pair_counts, &pair_displs, &total_pairs) &&
         SDG_make_counts(rank_counts, numprocs, SDG_COUNT_TERMS, &term_counts, &term_displs, &total_terms) &&
         SDG_make_counts(rank_counts, numprocs, SDG_COUNT_CDM, &cdm_counts, &cdm_displs, &total_cdm);

    nh_bits = SDG_bits_for((max_grid <= 1) ? 0U : (unsigned int)(max_grid - 1));
    h_bits = SDG_bits_for((max_h <= 1) ? 0U : (unsigned int)(max_h - 1));
    if (nh_bits + h_bits > 32 || total_atoms != (size_t)atomnum || total_atoms > (size_t)UINT16_MAX ||
        total_outputs > (size_t)UINT32_MAX || total_pairs > (size_t)UINT32_MAX ||
        total_terms > (size_t)UINT32_MAX || total_cdm > (size_t)UINT32_MAX) ok = 0;
    if (nh_bits == 32) ok = 0;
    if (ok) nh_mask = (1U << nh_bits) - 1U;

    if (ok) {
      atom_ids = (int *)SDG_malloc(total_atoms, sizeof(int));
      cache->orbs = (Type_Orbs_Grid *)SDG_malloc(total_orbs, sizeof(Type_Orbs_Grid));
      out_gn = (uint32_t *)SDG_malloc(total_outputs, sizeof(uint32_t));
      degree = (uint16_t *)SDG_malloc(total_outputs, sizeof(uint16_t));
      cache->terms = (uint32_t *)SDG_malloc(total_terms, sizeof(uint32_t));
      cache->cdm = (double *)SDG_malloc(total_cdm, sizeof(double));
      cache->rho = (double *)SDG_malloc((size_t)spin_count * (size_t)Ngrid1 * (size_t)Ngrid2 * (size_t)Ngrid3,
                                        sizeof(double));
      if (atom_ids == NULL || cache->orbs == NULL || out_gn == NULL || degree == NULL || cache->terms == NULL ||
          cache->cdm == NULL || cache->rho == NULL) ok = 0;
    }
  }

  MPI_Bcast(&ok, 1, MPI_INT, owner, mpi_comm_level1);
  if (!ok) goto fail;
  MPI_Bcast(&nh_bits, 1, MPI_INT, owner, mpi_comm_level1);
  MPI_Bcast(&nh_mask, 1, MPI_UNSIGNED, owner, mpi_comm_level1);

  local_atom_ids = (int *)SDG_malloc((size_t)local_counts[SDG_COUNT_ATOMS], sizeof(int));
  local_orbs = (Type_Orbs_Grid *)SDG_malloc((size_t)local_counts[SDG_COUNT_ORBS], sizeof(Type_Orbs_Grid));
  local_out_gn = (uint32_t *)SDG_malloc((size_t)local_counts[SDG_COUNT_OUTPUTS], sizeof(uint32_t));
  local_degree = (uint16_t *)calloc((size_t)(local_counts[SDG_COUNT_OUTPUTS] == 0 ? 1 : local_counts[SDG_COUNT_OUTPUTS]),
                                    sizeof(uint16_t));
  local_ptr = (uint32_t *)SDG_malloc((size_t)local_counts[SDG_COUNT_OUTPUTS] + 1, sizeof(uint32_t));
  local_cursor = (uint32_t *)SDG_malloc((size_t)local_counts[SDG_COUNT_OUTPUTS], sizeof(uint32_t));
  local_terms = (uint32_t *)SDG_malloc((size_t)local_counts[SDG_COUNT_TERMS], sizeof(uint32_t));
  ok = (local_atom_ids != NULL && local_orbs != NULL && local_out_gn != NULL && local_degree != NULL &&
        local_ptr != NULL && local_cursor != NULL && local_terms != NULL);
  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
  if (!all_ok) goto fail;

  pos = 0;
  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) local_atom_ids[Mc_AN - 1] = M2G[Mc_AN];

  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
    int NO0;
    Gc_AN = M2G[Mc_AN];
    NO0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];
    for (Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++) {
      for (i = 0; i < NO0; i++) local_orbs[pos++] = Orbs_Grid[Mc_AN][Nc][i];
    }
  }
  if (pos != (size_t)local_counts[SDG_COUNT_ORBS]) ok = 0;

  pos = 0;
  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
    Gc_AN = M2G[Mc_AN];
    for (Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++) {
      int GN = GridListAtom[Mc_AN][Nc];
      if (GN < 0 || (size_t)GN >= (size_t)Ngrid1 * (size_t)Ngrid2 * (size_t)Ngrid3) ok = 0;
      local_out_gn[pos++] = (uint32_t)GN;
    }
  }

  pos = 0;
  {
    size_t out_base = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
      Gc_AN = M2G[Mc_AN];
      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
        for (Nog = 0; Nog < NumOLG[Mc_AN][h_AN]; Nog++) {
          uint16_t *d;
          Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
          if (Nc < 0 || Nc >= GridN_Atom[Gc_AN]) {
            ok = 0;
            continue;
          }
          d = &local_degree[out_base + (size_t)Nc];
          if (*d == UINT16_MAX) ok = 0;
          else (*d)++;
          pos++;
        }
      }
      out_base += (size_t)GridN_Atom[Gc_AN];
    }
  }
  if (pos != (size_t)local_counts[SDG_COUNT_TERMS]) ok = 0;

  local_ptr[0] = 0;
  for (pos = 0; pos < (size_t)local_counts[SDG_COUNT_OUTPUTS]; pos++) {
    uint64_t next = (uint64_t)local_ptr[pos] + (uint64_t)local_degree[pos];
    if (next > UINT32_MAX) ok = 0;
    local_ptr[pos + 1] = (uint32_t)next;
    local_cursor[pos] = local_ptr[pos];
  }
  if ((size_t)local_ptr[local_counts[SDG_COUNT_OUTPUTS]] != (size_t)local_counts[SDG_COUNT_TERMS]) ok = 0;

  {
    size_t out_base = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
      Gc_AN = M2G[Mc_AN];
      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
        for (Nog = 0; Nog < NumOLG[Mc_AN][h_AN]; Nog++) {
          uint32_t slot, code;
          int Nh;
          Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
          Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
          Gh_AN = natn[Gc_AN][h_AN];
          if (Nc < 0 || Nc >= GridN_Atom[Gc_AN] || Nh < 0 || Nh >= GridN_Atom[Gh_AN] ||
              (uint32_t)Nh > nh_mask || ((uint64_t)(unsigned int)h_AN << nh_bits) > UINT32_MAX) {
            ok = 0;
            continue;
          }
          slot = local_cursor[out_base + (size_t)Nc]++;
          code = ((uint32_t)h_AN << nh_bits) | (uint32_t)Nh;
          if (slot >= (uint32_t)local_counts[SDG_COUNT_TERMS]) ok = 0;
          else local_terms[slot] = code;
        }
      }
      out_base += (size_t)GridN_Atom[Gc_AN];
    }
  }

  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
  if (!all_ok) goto fail;

  MPI_Gatherv(local_atom_ids, (int)local_counts[SDG_COUNT_ATOMS], MPI_INT,
              atom_ids, atom_counts, atom_displs, MPI_INT, owner, mpi_comm_level1);
  MPI_Gatherv(local_orbs, (int)local_counts[SDG_COUNT_ORBS], MPI_Type_Orbs_Grid,
              cache->orbs, orb_counts, orb_displs, MPI_Type_Orbs_Grid, owner, mpi_comm_level1);
  MPI_Gatherv(local_out_gn, (int)local_counts[SDG_COUNT_OUTPUTS], MPI_UNSIGNED,
              out_gn, out_counts, out_displs, MPI_UNSIGNED, owner, mpi_comm_level1);
  MPI_Gatherv(local_degree, (int)local_counts[SDG_COUNT_OUTPUTS], MPI_UNSIGNED_SHORT,
              degree, out_counts, out_displs, MPI_UNSIGNED_SHORT, owner, mpi_comm_level1);
  MPI_Gatherv(local_terms, (int)local_counts[SDG_COUNT_TERMS], MPI_UNSIGNED,
              cache->terms, term_counts, term_displs, MPI_UNSIGNED, owner, mpi_comm_level1);

  free(local_atom_ids); local_atom_ids = NULL;
  free(local_orbs); local_orbs = NULL;
  free(local_out_gn); local_out_gn = NULL;
  free(local_degree); local_degree = NULL;
  free(local_ptr); local_ptr = NULL;
  free(local_cursor); local_cursor = NULL;
  free(local_terms); local_terms = NULL;

  if (myid == owner) {
    int *atom_index = NULL;
    uint32_t *grid_degree = NULL, *grid_cursor = NULL;
    size_t a, p, out, grid_count;
    uint64_t out_off = 0, orb_off = 0, pair_off = 0, cdm_off = 0, term_off = 0;

    cache->atom_out_offset = (uint32_t *)SDG_malloc(total_atoms + 1, sizeof(uint32_t));
    cache->atom_orb_offset = (uint32_t *)SDG_malloc(total_atoms + 1, sizeof(uint32_t));
    cache->atom_pair_offset = (uint32_t *)SDG_malloc(total_atoms + 1, sizeof(uint32_t));
    cache->atom_no = (int *)SDG_malloc(total_atoms, sizeof(int));
    cache->pair_neighbor = (int *)SDG_malloc(total_pairs, sizeof(int));
    cache->pair_cdm_offset = (uint32_t *)SDG_malloc(total_pairs, sizeof(uint32_t));
    cache->out_atom = (uint16_t *)SDG_malloc(total_outputs, sizeof(uint16_t));
    cache->out_ptr = (uint32_t *)SDG_malloc(total_outputs + 1, sizeof(uint32_t));
    atom_index = (int *)SDG_malloc((size_t)atomnum + 1, sizeof(int));
    grid_count = (size_t)Ngrid1 * (size_t)Ngrid2 * (size_t)Ngrid3;
    grid_degree = (uint32_t *)calloc(grid_count == 0 ? 1 : grid_count, sizeof(uint32_t));
    grid_cursor = (uint32_t *)SDG_malloc(grid_count, sizeof(uint32_t));
    cache->grid_ptr = (uint32_t *)SDG_malloc(grid_count + 1, sizeof(uint32_t));
    cache->grid_out = (uint32_t *)SDG_malloc(total_outputs, sizeof(uint32_t));
    cache->scatter_counts = (int *)SDG_malloc((size_t)numprocs, sizeof(int));
    cache->scatter_displs = (int *)SDG_malloc((size_t)numprocs, sizeof(int));

    if (cache->atom_out_offset == NULL || cache->atom_orb_offset == NULL || cache->atom_pair_offset == NULL ||
        cache->atom_no == NULL || cache->pair_neighbor == NULL || cache->pair_cdm_offset == NULL ||
        cache->out_atom == NULL || cache->out_ptr == NULL || atom_index == NULL || grid_degree == NULL ||
        grid_cursor == NULL || cache->grid_ptr == NULL || cache->grid_out == NULL ||
        cache->scatter_counts == NULL || cache->scatter_displs == NULL) ok = 0;

    if (ok) {
      for (i = 0; i <= atomnum; i++) atom_index[i] = -1;
      for (a = 0; a < total_atoms; a++) {
        int ga = atom_ids[a];
        if (ga < 1 || ga > atomnum || atom_index[ga] != -1) ok = 0;
        else atom_index[ga] = (int)a;
      }
      for (i = 1; i <= atomnum; i++) if (atom_index[i] < 0) ok = 0;
    }

    for (a = 0; ok && a < total_atoms; a++) {
      int ga = atom_ids[a];
      int no0 = Spe_Total_CNO[WhatSpecies[ga]];
      cache->atom_out_offset[a] = (uint32_t)out_off;
      cache->atom_orb_offset[a] = (uint32_t)orb_off;
      cache->atom_pair_offset[a] = (uint32_t)pair_off;
      cache->atom_no[a] = no0;
      for (Nc = 0; Nc < GridN_Atom[ga]; Nc++) cache->out_atom[out_off + (size_t)Nc] = (uint16_t)a;
      out_off += (uint64_t)GridN_Atom[ga];
      orb_off += (uint64_t)GridN_Atom[ga] * (uint64_t)no0;
      for (h_AN = 0; h_AN <= FNAN[ga]; h_AN++) {
        int gb = natn[ga][h_AN];
        int b = (gb >= 1 && gb <= atomnum) ? atom_index[gb] : -1;
        int no1 = Spe_Total_CNO[WhatSpecies[gb]];
        if (b < 0 || cdm_off > UINT32_MAX) {
          ok = 0;
          break;
        }
        cache->pair_neighbor[pair_off] = b;
        cache->pair_cdm_offset[pair_off] = (uint32_t)cdm_off;
        cdm_off += (uint64_t)spin_count * (uint64_t)no0 * (uint64_t)no1;
        pair_off++;
      }
    }
    if (out_off != total_outputs || orb_off != total_orbs || pair_off != total_pairs || cdm_off != total_cdm) ok = 0;
    if (ok) {
      cache->atom_out_offset[total_atoms] = (uint32_t)out_off;
      cache->atom_orb_offset[total_atoms] = (uint32_t)orb_off;
      cache->atom_pair_offset[total_atoms] = (uint32_t)pair_off;
    }

    cache->out_ptr[0] = 0;
    for (out = 0; ok && out < total_outputs; out++) {
      term_off += (uint64_t)degree[out];
      if (term_off > UINT32_MAX) ok = 0;
      cache->out_ptr[out + 1] = (uint32_t)term_off;
    }
    if (term_off != total_terms) ok = 0;

    for (out = 0; ok && out < total_outputs; out++) {
      uint32_t gn = out_gn[out];
      if ((size_t)gn >= grid_count || grid_degree[gn] == UINT32_MAX) ok = 0;
      else grid_degree[gn]++;
    }

    cache->grid_ptr[0] = 0;
    for (p = 0; ok && p < grid_count; p++) {
      uint64_t next = (uint64_t)cache->grid_ptr[p] + (uint64_t)grid_degree[p];
      if (next > UINT32_MAX) ok = 0;
      cache->grid_ptr[p + 1] = (uint32_t)next;
      grid_cursor[p] = cache->grid_ptr[p];
    }
    if ((size_t)cache->grid_ptr[grid_count] != total_outputs) ok = 0;

    for (out = 0; ok && out < total_outputs; out++) {
      uint32_t gn = out_gn[out];
      uint32_t slot = grid_cursor[gn]++;
      if ((size_t)slot >= total_outputs) ok = 0;
      else cache->grid_out[slot] = (uint32_t)out;
    }

    if (ok) {
      unsigned long long n2d = (unsigned long long)Ngrid1 * (unsigned long long)Ngrid2;
      int rank;
      for (rank = 0; rank < numprocs; rank++) {
        unsigned long long b0 = ((unsigned long long)rank * n2d + (unsigned long long)numprocs - 1ULL) /
                                (unsigned long long)numprocs;
        unsigned long long b1 = ((unsigned long long)(rank + 1) * n2d + (unsigned long long)numprocs - 1ULL) /
                                (unsigned long long)numprocs;
        unsigned long long disp = b0 * (unsigned long long)Ngrid3;
        unsigned long long count = (b1 - b0) * (unsigned long long)Ngrid3;
        if (disp > (unsigned long long)INT_MAX || count > (unsigned long long)INT_MAX) ok = 0;
        else {
          cache->scatter_displs[rank] = (int)disp;
          cache->scatter_counts[rank] = (int)count;
        }
      }
    }

    cache->device_bytes = 0;
    if (ok) {
      ok = SDG_add_bytes(&cache->device_bytes, total_orbs, sizeof(Type_Orbs_Grid)) &&
           SDG_add_bytes(&cache->device_bytes, total_atoms + 1, sizeof(uint32_t)) &&
           SDG_add_bytes(&cache->device_bytes, total_atoms + 1, sizeof(uint32_t)) &&
           SDG_add_bytes(&cache->device_bytes, total_atoms + 1, sizeof(uint32_t)) &&
           SDG_add_bytes(&cache->device_bytes, total_atoms, sizeof(int)) &&
           SDG_add_bytes(&cache->device_bytes, total_pairs, sizeof(int)) &&
           SDG_add_bytes(&cache->device_bytes, total_pairs, sizeof(uint32_t)) &&
           SDG_add_bytes(&cache->device_bytes, total_outputs, sizeof(uint16_t)) &&
           SDG_add_bytes(&cache->device_bytes, total_outputs + 1, sizeof(uint32_t)) &&
           SDG_add_bytes(&cache->device_bytes, total_terms, sizeof(uint32_t)) &&
           SDG_add_bytes(&cache->device_bytes, grid_count + 1, sizeof(uint32_t)) &&
           SDG_add_bytes(&cache->device_bytes, total_outputs, sizeof(uint32_t)) &&
           SDG_add_bytes(&cache->device_bytes, total_cdm, sizeof(double)) &&
           SDG_add_bytes(&cache->device_bytes, (size_t)spin_count * grid_count, sizeof(double));
    }

    free(atom_index);
    free(grid_degree);
    free(grid_cursor);
  }

  MPI_Bcast(&ok, 1, MPI_INT, owner, mpi_comm_level1);
  if (!ok) goto fail;

  cache->ready = 1;
  cache->owner = owner;
  cache->spin_count = spin_count;
  cache->nh_bits = nh_bits;
  cache->nh_mask = nh_mask;
  cache->local_cdm_count = (int)local_counts[SDG_COUNT_CDM];
  cache->atom_count = total_atoms;
  cache->orbital_count = total_orbs;
  cache->output_count = total_outputs;
  cache->pair_count = total_pairs;
  cache->term_count = total_terms;
  cache->cdm_count = total_cdm;
  cache->global_grid_count = (size_t)Ngrid1 * (size_t)Ngrid2 * (size_t)Ngrid3;
  if (myid == owner) {
    cache->cdm_counts = cdm_counts; cdm_counts = NULL;
    cache->cdm_displs = cdm_displs; cdm_displs = NULL;
    if (2 <= level_stdout) {
      printf("<Set_Density_Grid> one-rank GPU service cache: owner=%d atoms=%zu atom-grids=%zu terms=%zu device=%.3f GiB\n",
             owner, total_atoms, total_outputs, total_terms,
             (double)cache->device_bytes / (1024.0 * 1024.0 * 1024.0));
      fflush(stdout);
    }
  }

  free(rank_counts);
  free(atom_ids);
  free(out_gn);
  free(degree);
  free(atom_counts); free(atom_displs);
  free(orb_counts); free(orb_displs);
  free(out_counts); free(out_displs);
  free(term_counts); free(term_displs);
  free(pair_counts); free(pair_displs);
  free(cdm_counts); free(cdm_displs);
  return 1;

fail:
  free(local_atom_ids);
  free(local_orbs);
  free(local_out_gn);
  free(local_degree);
  free(local_ptr);
  free(local_cursor);
  free(local_terms);
  free(rank_counts);
  free(atom_ids);
  free(out_gn);
  free(degree);
  free(atom_counts); free(atom_displs);
  free(orb_counts); free(orb_displs);
  free(out_counts); free(out_displs);
  free(term_counts); free(term_displs);
  free(pair_counts); free(pair_displs);
  free(cdm_counts); free(cdm_displs);
  SDG_free_host(cache);
  cache->unavailable = 1;
  return 0;
}

static int SDG_pack_local_cdm(double *****CDM, double *buffer, size_t expected, int spin_count)
{
  size_t p = 0;
  int Mc_AN, h_AN, Gc_AN, Gh_AN, NO0, NO1, spin, i, j;

  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
    Gc_AN = M2G[Mc_AN];
    NO0 = Spe_Total_CNO[WhatSpecies[Gc_AN]];
    for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {
      Gh_AN = natn[Gc_AN][h_AN];
      NO1 = Spe_Total_CNO[WhatSpecies[Gh_AN]];
      for (spin = 0; spin < spin_count; spin++) {
        for (i = 0; i < NO0; i++) {
          for (j = 0; j < NO1; j++) {
            if (p >= expected) return 0;
            buffer[p++] = CDM[spin][Mc_AN][h_AN][i][j];
          }
        }
      }
    }
  }
  return p == expected;
}

static int SDG_upload_device(SetDensityGpuCache *cache)
{
  if (cache->device_resident) {
    acc_update_device(cache->cdm, cache->cdm_count * sizeof(double));
    return 1;
  }

  /* Mark the upload as resident before the first allocation so a partial
     failure is rolled back by SDG_delete_device(). */
  cache->device_resident = 1;
  if (SDG_copyin(cache->orbs, cache->orbital_count * sizeof(Type_Orbs_Grid)) == NULL ||
      SDG_copyin(cache->atom_out_offset, (cache->atom_count + 1) * sizeof(uint32_t)) == NULL ||
      SDG_copyin(cache->atom_orb_offset, (cache->atom_count + 1) * sizeof(uint32_t)) == NULL ||
      SDG_copyin(cache->atom_pair_offset, (cache->atom_count + 1) * sizeof(uint32_t)) == NULL ||
      SDG_copyin(cache->atom_no, cache->atom_count * sizeof(int)) == NULL ||
      SDG_copyin(cache->pair_neighbor, cache->pair_count * sizeof(int)) == NULL ||
      SDG_copyin(cache->pair_cdm_offset, cache->pair_count * sizeof(uint32_t)) == NULL ||
      SDG_copyin(cache->out_atom, cache->output_count * sizeof(uint16_t)) == NULL ||
      SDG_copyin(cache->out_ptr, (cache->output_count + 1) * sizeof(uint32_t)) == NULL ||
      SDG_copyin(cache->terms, cache->term_count * sizeof(uint32_t)) == NULL ||
      SDG_copyin(cache->grid_ptr, (cache->global_grid_count + 1) * sizeof(uint32_t)) == NULL ||
      SDG_copyin(cache->grid_out, cache->output_count * sizeof(uint32_t)) == NULL ||
      SDG_copyin(cache->cdm, cache->cdm_count * sizeof(double)) == NULL ||
      SDG_create(cache->rho, cache->global_grid_count * (size_t)cache->spin_count * sizeof(double)) == NULL) {
    SDG_delete_device(cache);
    return 0;
  }
  return 1;
}

static void SDG_run_kernel_single(SetDensityGpuCache *cache)
{
  Type_Orbs_Grid *orbs = cache->orbs;
  uint32_t *atom_out_offset = cache->atom_out_offset;
  uint32_t *atom_orb_offset = cache->atom_orb_offset;
  uint32_t *atom_pair_offset = cache->atom_pair_offset;
  int *atom_no = cache->atom_no;
  int *pair_neighbor = cache->pair_neighbor;
  uint32_t *pair_cdm_offset = cache->pair_cdm_offset;
  uint16_t *out_atom = cache->out_atom;
  uint32_t *out_ptr = cache->out_ptr;
  uint32_t *terms = cache->terms;
  uint32_t *grid_ptr = cache->grid_ptr;
  uint32_t *grid_out = cache->grid_out;
  double *cdm = cache->cdm;
  double *rho = cache->rho;
  size_t atom_count = cache->atom_count;
  size_t orbital_count = cache->orbital_count;
  size_t output_count = cache->output_count;
  size_t pair_count = cache->pair_count;
  size_t term_count = cache->term_count;
  size_t cdm_count = cache->cdm_count;
  size_t grid_count = cache->global_grid_count;
  int spin_count = cache->spin_count;
  int nh_bits = cache->nh_bits;
  uint32_t nh_mask = cache->nh_mask;

#pragma acc parallel loop gang vector_length(128) \
  present(orbs[0:orbital_count], atom_out_offset[0:atom_count+1], atom_orb_offset[0:atom_count+1], \
          atom_pair_offset[0:atom_count+1], atom_no[0:atom_count], pair_neighbor[0:pair_count], \
          pair_cdm_offset[0:pair_count], out_atom[0:output_count], out_ptr[0:output_count+1], \
          terms[0:term_count], grid_ptr[0:grid_count+1], grid_out[0:output_count], \
          cdm[0:cdm_count], rho[0:grid_count*spin_count])
  for (size_t gn = 0; gn < grid_count; gn++) {
    double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0;

#pragma acc loop seq
    for (uint32_t q = grid_ptr[gn]; q < grid_ptr[gn + 1]; q++) {
      uint32_t out = grid_out[q];
      int a = (int)out_atom[out];
      uint32_t Nc0 = out - atom_out_offset[a];
      int NO0 = atom_no[a];
      size_t orb0 = (size_t)atom_orb_offset[a] + (size_t)Nc0 * (size_t)NO0;
      double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;

#pragma acc loop seq
      for (uint32_t t = out_ptr[out]; t < out_ptr[out + 1]; t++) {
        uint32_t code = terms[t];
        uint32_t Nh = code & nh_mask;
        uint32_t h = code >> nh_bits;
        uint32_t pair = atom_pair_offset[a] + h;
        int b = pair_neighbor[pair];
        int NO1 = atom_no[b];
        size_t orb1 = (size_t)atom_orb_offset[b] + (size_t)Nh * (size_t)NO1;
        size_t mat = (size_t)NO0 * (size_t)NO1;
        size_t base = (size_t)pair_cdm_offset[pair];
        double e0 = 0.0, e1 = 0.0, e2 = 0.0, e3 = 0.0;

#pragma acc loop seq
        for (int ii = 0; ii < NO0; ii++) {
          double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
#pragma acc loop seq
          for (int jj = 0; jj < NO1; jj++) {
            double phi1 = (double)orbs[orb1 + (size_t)jj];
            size_t ij = (size_t)ii * (size_t)NO1 + (size_t)jj;
            t0 += phi1 * cdm[base + ij];
            if (spin_count >= 2) t1 += phi1 * cdm[base + mat + ij];
            if (spin_count == 4) {
              t2 += phi1 * cdm[base + 2U * mat + ij];
              t3 += phi1 * cdm[base + 3U * mat + ij];
            }
          }
          {
            double phi0 = (double)orbs[orb0 + (size_t)ii];
            e0 += phi0 * t0;
            if (spin_count >= 2) e1 += phi0 * t1;
            if (spin_count == 4) {
              e2 += phi0 * t2;
              e3 += phi0 * t3;
            }
          }
        }
        a0 += e0;
        if (spin_count >= 2) a1 += e1;
        if (spin_count == 4) {
          a2 += e2;
          a3 += e3;
        }
      }
      g0 += a0;
      if (spin_count >= 2) g1 += a1;
      if (spin_count == 4) {
        g2 += a2;
        g3 += a3;
      }
    }

    rho[gn] = g0;
    if (spin_count >= 2) rho[grid_count + gn] = g1;
    if (spin_count == 4) {
      rho[2U * grid_count + gn] = g2;
      rho[3U * grid_count + gn] = -g3;
    }
  }
}

static void SDG_run_kernel_part(SetDensityGpuCache *cache,size_t first,size_t last)
{
  Type_Orbs_Grid *orbs = cache->orbs;
  uint32_t *atom_out_offset = cache->atom_out_offset;
  uint32_t *atom_orb_offset = cache->atom_orb_offset;
  uint32_t *atom_pair_offset = cache->atom_pair_offset;
  int *atom_no = cache->atom_no;
  int *pair_neighbor = cache->pair_neighbor;
  uint32_t *pair_cdm_offset = cache->pair_cdm_offset;
  uint16_t *out_atom = cache->out_atom;
  uint32_t *out_ptr = cache->out_ptr;
  uint32_t *terms = cache->terms;
  uint32_t *grid_ptr = cache->grid_ptr;
  uint32_t *grid_out = cache->grid_out;
  double *cdm = cache->cdm;
  double *rho = cache->rho;
  size_t atom_count = cache->atom_count;
  size_t orbital_count = cache->orbital_count;
  size_t output_count = cache->output_count;
  size_t pair_count = cache->pair_count;
  size_t term_count = cache->term_count;
  size_t cdm_count = cache->cdm_count;
  size_t grid_count = cache->global_grid_count;
  int spin_count = cache->spin_count;
  int nh_bits = cache->nh_bits;
  uint32_t nh_mask = cache->nh_mask;

#pragma acc parallel loop gang vector_length(128) deviceptr(orbs,atom_out_offset,atom_orb_offset,atom_pair_offset,atom_no,pair_neighbor,pair_cdm_offset,out_atom,out_ptr,terms,grid_ptr,grid_out,cdm,rho)
  for (size_t gn = first; gn < last; gn++) {
    double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0;

#pragma acc loop seq
    for (uint32_t q = grid_ptr[gn]; q < grid_ptr[gn + 1]; q++) {
      uint32_t out = grid_out[q];
      int a = (int)out_atom[out];
      uint32_t Nc0 = out - atom_out_offset[a];
      int NO0 = atom_no[a];
      size_t orb0 = (size_t)atom_orb_offset[a] + (size_t)Nc0 * (size_t)NO0;
      double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;

#pragma acc loop seq
      for (uint32_t t = out_ptr[out]; t < out_ptr[out + 1]; t++) {
        uint32_t code = terms[t];
        uint32_t Nh = code & nh_mask;
        uint32_t h = code >> nh_bits;
        uint32_t pair = atom_pair_offset[a] + h;
        int b = pair_neighbor[pair];
        int NO1 = atom_no[b];
        size_t orb1 = (size_t)atom_orb_offset[b] + (size_t)Nh * (size_t)NO1;
        size_t mat = (size_t)NO0 * (size_t)NO1;
        size_t base = (size_t)pair_cdm_offset[pair];
        double e0 = 0.0, e1 = 0.0, e2 = 0.0, e3 = 0.0;

#pragma acc loop seq
        for (int ii = 0; ii < NO0; ii++) {
          double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
#pragma acc loop seq
          for (int jj = 0; jj < NO1; jj++) {
            double phi1 = (double)orbs[orb1 + (size_t)jj];
            size_t ij = (size_t)ii * (size_t)NO1 + (size_t)jj;
            t0 += phi1 * cdm[base + ij];
            if (spin_count >= 2) t1 += phi1 * cdm[base + mat + ij];
            if (spin_count == 4) {
              t2 += phi1 * cdm[base + 2U * mat + ij];
              t3 += phi1 * cdm[base + 3U * mat + ij];
            }
          }
          {
            double phi0 = (double)orbs[orb0 + (size_t)ii];
            e0 += phi0 * t0;
            if (spin_count >= 2) e1 += phi0 * t1;
            if (spin_count == 4) {
              e2 += phi0 * t2;
              e3 += phi0 * t3;
            }
          }
        }
        a0 += e0;
        if (spin_count >= 2) a1 += e1;
        if (spin_count == 4) {
          a2 += e2;
          a3 += e3;
        }
      }
      g0 += a0;
      if (spin_count >= 2) g1 += a1;
      if (spin_count == 4) {
        g2 += a2;
        g3 += a3;
      }
    }

    rho[gn-first] = g0;
    if (spin_count >= 2) rho[(last-first) + gn-first] = g1;
    if (spin_count == 4) {
      rho[2U * (last-first) + gn-first] = g2;
      rho[3U * (last-first) + gn-first] = -g3;
    }
  }
}


typedef struct {
  OpenMXGpuWorkers workers;
  SetDensityGpuCache *source,device[OPENMX_GPU_WORKERS_MAX];
  OpenMXGpuReadView input[OPENMX_GPU_WORKERS_MAX][13];
  const void *source_device[13];
  double *target;
} SDGServiceWork;

static int SDG_service_worker(void *opaque,int worker,int phase)
{
  SDGServiceWork *w=(SDGServiceWork*)opaque;
  SetDensityGpuCache *c=w->source,*d=w->device+worker;
  size_t first=openmx_gpu_worker_cut(w->workers.items,worker,w->workers.count);
  size_t last=openmx_gpu_worker_cut(w->workers.items,worker+1,w->workers.count);
  if (phase==OPENMX_GPU_WORKER_PREPARE) {
    *d=*c;d->rho=NULL;
    if (!openmx_gpu_read_view(&w->input[worker][0],w->source_device[0],(c->orbital_count)*sizeof(Type_Orbs_Grid))) return 0;
    d->orbs=(Type_Orbs_Grid*)w->input[worker][0].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][1],w->source_device[1],(c->atom_count+1)*sizeof(uint32_t))) return 0;
    d->atom_out_offset=(uint32_t*)w->input[worker][1].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][2],w->source_device[2],(c->atom_count+1)*sizeof(uint32_t))) return 0;
    d->atom_orb_offset=(uint32_t*)w->input[worker][2].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][3],w->source_device[3],(c->atom_count+1)*sizeof(uint32_t))) return 0;
    d->atom_pair_offset=(uint32_t*)w->input[worker][3].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][4],w->source_device[4],(c->atom_count)*sizeof(int))) return 0;
    d->atom_no=(int*)w->input[worker][4].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][5],w->source_device[5],(c->pair_count)*sizeof(int))) return 0;
    d->pair_neighbor=(int*)w->input[worker][5].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][6],w->source_device[6],(c->pair_count)*sizeof(uint32_t))) return 0;
    d->pair_cdm_offset=(uint32_t*)w->input[worker][6].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][7],w->source_device[7],(c->output_count)*sizeof(uint16_t))) return 0;
    d->out_atom=(uint16_t*)w->input[worker][7].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][8],w->source_device[8],(c->output_count+1)*sizeof(uint32_t))) return 0;
    d->out_ptr=(uint32_t*)w->input[worker][8].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][9],w->source_device[9],(c->term_count)*sizeof(uint32_t))) return 0;
    d->terms=(uint32_t*)w->input[worker][9].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][10],w->source_device[10],(c->global_grid_count+1)*sizeof(uint32_t))) return 0;
    d->grid_ptr=(uint32_t*)w->input[worker][10].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][11],w->source_device[11],(c->output_count)*sizeof(uint32_t))) return 0;
    d->grid_out=(uint32_t*)w->input[worker][11].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][12],w->source_device[12],(c->cdm_count)*sizeof(double))) return 0;
    d->cdm=(double*)w->input[worker][12].pointer;
    return cudaMalloc((void**)&d->rho,(last-first)*(size_t)c->spin_count*sizeof(double))==cudaSuccess;
  }
  if (phase==OPENMX_GPU_WORKER_EXECUTE) SDG_run_kernel_part(d,first,last);
  else if (phase==OPENMX_GPU_WORKER_COMMIT) {
    for (int spin=0;spin<c->spin_count;spin++)
      if (!openmx_gpu_copy_device(w->target+(size_t)spin*c->global_grid_count+first,
          d->rho+(size_t)spin*(last-first),(last-first)*sizeof(double))) return 0;
  } else if (phase==OPENMX_GPU_WORKER_RELEASE) {
    if (d->rho) cudaFree(d->rho);
    for (int i=0;i<13;i++) openmx_gpu_read_view_release(&w->input[worker][i]);
  }
  return 1;
}

static void SDG_run_kernel(SetDensityGpuCache *cache)
{
  SDGServiceWork w={0};w.source=cache;
  openmx_gpu_workers_owner(&w.workers,cache->global_grid_count);
  if (w.workers.count>1) {
    w.source_device[0]=acc_deviceptr(cache->orbs);
    w.source_device[1]=acc_deviceptr(cache->atom_out_offset);
    w.source_device[2]=acc_deviceptr(cache->atom_orb_offset);
    w.source_device[3]=acc_deviceptr(cache->atom_pair_offset);
    w.source_device[4]=acc_deviceptr(cache->atom_no);
    w.source_device[5]=acc_deviceptr(cache->pair_neighbor);
    w.source_device[6]=acc_deviceptr(cache->pair_cdm_offset);
    w.source_device[7]=acc_deviceptr(cache->out_atom);
    w.source_device[8]=acc_deviceptr(cache->out_ptr);
    w.source_device[9]=acc_deviceptr(cache->terms);
    w.source_device[10]=acc_deviceptr(cache->grid_ptr);
    w.source_device[11]=acc_deviceptr(cache->grid_out);
    w.source_device[12]=acc_deviceptr(cache->cdm);
    w.target=(double*)acc_deviceptr(cache->rho);
    if (w.target && openmx_gpu_workers_run(&w.workers,SDG_service_worker,&w,"density-owner-grid")) return;
  }
  SDG_run_kernel_single(cache);
}

/* Eigensolvers whose density matrix the device paths integrate.  Every
   solver hands Set_Density_Grid the same DM[spin][Mc_AN][h_AN][i][j] over
   the FNAN neighbours, so the quadrature itself is solver-independent.
   The distributed local mode (each rank its own atoms, Set_Hamiltonian's
   resident tables or the on-the-fly tiles) takes the cluster and band
   solvers and, since 2026-10-09, DC (5), Krylov (8) and DC-LNO (11); the
   single-owner service, which builds the whole grid's orbital tables on
   one device, stays with the cluster and band solvers: at the O(N)
   solvers' sizes those tables fit neither the device nor comfortably the
   host. */
static int SDG_solver_supported(int local_mode)
{
  if (Solver == 2 || Solver == 3) return 1;
  return local_mode && (Solver == 5 || Solver == 8 || Solver == 11);
}

int Set_Density_Grid_GPU_Service(const OpenMXDensityCacheHooks *cache_hooks, int Cnt_kind, int Calc_CntOrbital_ON, double *****CDM,
                                 double **Density_Grid_B0, double *elapsed)
{
  SetDensityGpuCache *cache = &SDG_cache;
  int numprocs, myid, owner = Host_ID;
  int enabled = 0, device_count = 0, device = 0;
  int spin_count = SpinP_switch + 1;
  int ok = 1, all_ok = 1;
  double start, end;
  double *local_cdm = NULL;
  size_t reserve_bytes;
  size_t free_bytes = 0, total_bytes = 0;
  int need_release = 0;
  int persistent;

  (void)Calc_CntOrbital_ON;
  dtime(&start);
  MPI_Comm_size(mpi_comm_level1, &numprocs);
  MPI_Comm_rank(mpi_comm_level1, &myid);

  if (myid == owner) {
    int requested = openmx_env_flag_at(OPENMX_ENV_DENSITY_GRID_GPU,
                                openmx_env_flag_at(OPENMX_ENV_SETDENSITY_GPU, 1));
    enabled = requested && scf_eigen_lib_flag == GPUSOLVER && SDG_solver_supported(0) &&
              Cnt_switch == 0 && (Cnt_kind == 0 || Cnt_kind == 1) &&
              (SpinP_switch == 0 || SpinP_switch == 1 || SpinP_switch == 3);
    if (enabled) {
      cudaError_t status = cudaGetDeviceCount(&device_count);
      if (status != cudaSuccess || device_count <= 0 || acc_get_num_devices(acc_device_nvidia) <= 0 ||
          !gpu_rank_device_usable()) enabled = 0;
    }
  }
  MPI_Bcast(&enabled, 1, MPI_INT, owner, mpi_comm_level1);
  if (!enabled || cache->unavailable) return 0;

  if (!cache->ready) {
    if (!SDG_build_cache(spin_count, owner, myid, numprocs)) return 0;
  }
  if (cache->spin_count != spin_count) {
    Set_Density_Grid_GPU_Invalidate();
    if (!SDG_build_cache(spin_count, owner, myid, numprocs)) return 0;
  }

  reserve_bytes = openmx_env_mib_bytes_at(OPENMX_ENV_DENSITY_GRID_GPU_RESERVE_MB, 256);
  if (myid == owner && !cache->device_resident) {
    cudaGetDevice(&device);
    acc_set_device_num(device, acc_device_nvidia);
    if (OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
        free_bytes < cache->device_bytes || reserve_bytes > free_bytes - cache->device_bytes) need_release = 1;
  }
  MPI_Bcast(&need_release, 1, MPI_INT, owner, mpi_comm_level1);

  if (need_release) {
    /* First return only disposable storage.  In particular, do not destroy
       the collinear Cluster solver context here: it owns the transformed
       overlap matrix, and dropping it makes the next SCF diagonalize the
       overlap again.  OpenACC freelists are per MPI process, so every rank
       returns its deleted blocks before the owner retries cudaMemGetInfo(). */
    acc_wait_all();
    openmx_gemmul8ReleaseWorkspaces();
    acc_clear_freelists();
    MPI_Barrier(mpi_comm_level1);

    if (myid == owner) {
      need_release =
          (OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
           free_bytes < cache->device_bytes || reserve_bytes > free_bytes - cache->device_bytes);
    }
    MPI_Bcast(&need_release, 1, MPI_INT, owner, mpi_comm_level1);
  }

  if (need_release && Solver == 2 && SpinP_switch == 3) {
    /* The non-collinear Cluster path keeps its final dense eigenvectors for a
       later scatter.  Preserve them on the host, then return only their device
       copy and its OpenACC freelist allocation. */
    if (cache_hooks && cache_hooks->demote_eigenvectors)
      cache_hooks->demote_eigenvectors(cache_hooks->context);
    acc_wait_all();
    acc_clear_freelists();
    MPI_Barrier(mpi_comm_level1);
  }

  if (myid == owner && !cache->device_resident) {
    if (OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
        free_bytes < cache->device_bytes || reserve_bytes > free_bytes - cache->device_bytes) {
      ok = 0;
      if (2 <= level_stdout) {
        fprintf(stderr,
                "Set_Density_Grid GPU owner needs %.3f GiB plus %.3f GiB reserve, but only %.3f GiB is free; CPU fallback.\n",
                (double)cache->device_bytes / (1024.0 * 1024.0 * 1024.0),
                (double)reserve_bytes / (1024.0 * 1024.0 * 1024.0),
                (double)free_bytes / (1024.0 * 1024.0 * 1024.0));
        fflush(stderr);
      }
    }
  }
  MPI_Bcast(&ok, 1, MPI_INT, owner, mpi_comm_level1);
  if (!ok) {
    SDG_disable_current_epoch(cache);
    return 0;
  }

  local_cdm = (double *)SDG_malloc((size_t)cache->local_cdm_count, sizeof(double));
  ok = (local_cdm != NULL) && SDG_pack_local_cdm(CDM, local_cdm, (size_t)cache->local_cdm_count, spin_count);
  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, mpi_comm_level1);
  if (!all_ok) {
    free(local_cdm);
    SDG_disable_current_epoch(cache);
    return 0;
  }

  MPI_Gatherv(local_cdm, cache->local_cdm_count, MPI_DOUBLE,
              cache->cdm, cache->cdm_counts, cache->cdm_displs, MPI_DOUBLE, owner, mpi_comm_level1);
  free(local_cdm);

  if (myid == owner) {
    if (!SDG_upload_device(cache)) ok = 0;
    if (ok) {
      SDG_run_kernel(cache);
      acc_update_self(cache->rho, cache->global_grid_count * (size_t)spin_count * sizeof(double));
      acc_wait_all();
    }
  }
  MPI_Bcast(&ok, 1, MPI_INT, owner, mpi_comm_level1);
  if (!ok) {
    SDG_disable_current_epoch(cache);
    return 0;
  }

  /* Reuse small geometry tables across SCF steps. Large epochs still yield
     their storage to the dense eigensolver after each density contraction. */
  persistent = openmx_env_flag_at(OPENMX_ENV_DENSITY_GRID_GPU_PERSISTENT,
      cache->device_bytes <= openmx_env_mib_bytes_at(OPENMX_ENV_DENSITY_GRID_GPU_CACHE_MB, 64));
  if (myid == owner && !persistent) SDG_delete_device(cache);

  for (int spin = 0; spin < spin_count; spin++) {
    const double *sendbuf = (myid == owner) ? cache->rho + (size_t)spin * cache->global_grid_count : NULL;
    MPI_Scatterv(sendbuf, cache->scatter_counts, cache->scatter_displs, MPI_DOUBLE,
                 Density_Grid_B0[spin], My_NumGridB_AB, MPI_DOUBLE, owner, mpi_comm_level1);
  }

  Density_Grid_Copy_B2D(Density_Grid_B0);
  dtime(&end);
  *elapsed = end - start;
  return 1;
}

/**********************************************************************
   Distributed (per-rank) GPU quadrature for Set_Density_Grid.

   Every rank evaluates the atomic-sphere density of its own atoms on
   the device, reusing the Set_Hamiltonian matrix-elements tables (the
   central full-sphere orbitals, the FNAN-layout neighbour orbitals and
   the overlap index lists).  When Set_Hamiltonian keeps those tables
   device-resident the present_or_copyin staging below only attaches,
   so a single copy serves both routines.  The kernel reproduces the
   CPU loop exactly: per output point the pair contributions are summed
   in (h_AN, Nog) order and each contribution is
   sum_i orbs0[i] * (sum_j orbs1[j] * CDM[i][j]).  The result fills
   Tmp_Den_Grid and the ordinary A-to-B communication of
   Set_Density_Grid.c distributes it afterwards.
***********************************************************************/

typedef struct {
  int ready;
  int unavailable;
  int spin_count;
  int cnt_kind;
  int node_ranks;
  size_t output_count;     /* sum of GridN_Atom over the rank's atoms */
  size_t term_count;       /* overlap points of the rank's pairs */
  size_t dm_count;         /* packed density-matrix doubles */
  size_t extra_bytes;      /* device bytes owned by this mode alone */
  uint32_t *atom_out_base; /* Matomnum+2, host only */
  uint32_t *out_ptr;       /* CSR over the output points */
  uint32_t *term_pt;       /* CSR payload: global overlap-point index */
  uint32_t *pt_pair;       /* overlap point -> pair */
  double *dm;
  double *tmpden;
  int device_resident;
  int device_id;
  SetHamiltonianMETables t;
  /* the on-the-fly mode of a rank without resident tables (orbs_grid_gpu.h):
     no CSR, the density from per-pair tiles evaluated on the device */
  int otf;
  size_t otf_dm_count;
  size_t otf_output_count;
  uint32_t *otf_out_base;
} SDGLocalContext;

static SDGLocalContext SDG_local = {0};

/* Set_Hamiltonian.c: the planned on-the-fly density call's figures */
int Set_Hamiltonian_OnTheFly_DensityStage(int otf_ranks, size_t *need_bytes, size_t *budget_bytes);

static void SDG_local_delete_device(SDGLocalContext *c)
{
  int current_device;

  if (!c->device_resident) return;
  current_device = acc_get_device_num(acc_device_nvidia);
  if (current_device != c->device_id)
    acc_set_device_num(c->device_id, acc_device_nvidia);
  SDG_acc_delete_if_present(c->out_ptr, (c->output_count + 1) * sizeof(uint32_t));
  SDG_acc_delete_if_present(c->term_pt, c->term_count * sizeof(uint32_t));
  SDG_acc_delete_if_present(c->pt_pair, c->term_count * sizeof(uint32_t));
  SDG_acc_delete_if_present(c->dm, c->dm_count * sizeof(double));
  SDG_acc_delete_if_present(c->tmpden, (size_t)c->spin_count * c->output_count * sizeof(double));
  c->device_resident = 0;
  if (current_device != c->device_id)
    acc_set_device_num(current_device, acc_device_nvidia);
}

static int SDG_local_upload(SDGLocalContext *c)
{
  if (c->device_resident) return 1;
  c->device_id = acc_get_device_num(acc_device_nvidia);
  c->device_resident = 1; /* include partially uploaded tables in cleanup */
  if (SDG_copyin(c->out_ptr, (c->output_count + 1) * sizeof(uint32_t)) == NULL ||
      SDG_copyin(c->term_pt, c->term_count * sizeof(uint32_t)) == NULL ||
      SDG_copyin(c->pt_pair, c->term_count * sizeof(uint32_t)) == NULL ||
      SDG_create(c->dm, c->dm_count * sizeof(double)) == NULL ||
      SDG_create(c->tmpden, (size_t)c->spin_count * c->output_count * sizeof(double)) == NULL) {
    SDG_local_delete_device(c);
    return 0;
  }
  return 1;
}

static void SDG_local_free(void)
{
  free(SDG_local.otf_out_base);
  SDG_local.otf_out_base = NULL;
  SDG_local.otf = 0;
  SDGLocalContext *c = &SDG_local;

  SDG_local_delete_device(c);
  free(c->atom_out_base);
  free(c->out_ptr);
  free(c->term_pt);
  free(c->pt_pair);
  free(c->dm);
  free(c->tmpden);
  memset(c, 0, sizeof(*c));
}

static int SDG_local_build(int Cnt_kind, int spin_count)
{
  SDGLocalContext *c = &SDG_local;
  uint32_t *degree = NULL;
  int ok = 1;
  int Mc_AN, p;
  size_t out, pos;

  {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0 ||
        acc_get_num_devices(acc_device_nvidia) <= 0) ok = 0;
  }

  if (ok && !Set_Hamiltonian_GetMatrixElementsTables(Cnt_kind, &c->t)) ok = 0;

  if (ok) {
    c->spin_count = spin_count;
    c->cnt_kind = Cnt_kind;
    c->term_count = c->t.total_nolg;
    c->dm_count = c->t.total_h;
    if (c->term_count > (size_t)UINT32_MAX) ok = 0;
  }

  if (ok) {
    c->atom_out_base = (uint32_t *)SDG_malloc((size_t)Matomnum + 2, sizeof(uint32_t));
    ok = (c->atom_out_base != NULL);
  }
  if (ok) {
    size_t base = 0;
    c->atom_out_base[0] = 0;
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
      c->atom_out_base[Mc_AN] = (uint32_t)base;
      base += (size_t)GridN_Atom[M2G[Mc_AN]];
      if (base > (size_t)UINT32_MAX) {
        ok = 0;
        break;
      }
    }
    if (ok) {
      c->atom_out_base[Matomnum + 1] = (uint32_t)base;
      c->output_count = base;
    }
  }

  if (ok) {
    degree = (uint32_t *)calloc(c->output_count == 0 ? 1 : c->output_count, sizeof(uint32_t));
    c->out_ptr = (uint32_t *)SDG_malloc(c->output_count + 1, sizeof(uint32_t));
    c->term_pt = (uint32_t *)SDG_malloc(c->term_count == 0 ? 1 : c->term_count, sizeof(uint32_t));
    c->pt_pair = (uint32_t *)SDG_malloc(c->term_count == 0 ? 1 : c->term_count, sizeof(uint32_t));
    c->dm = (double *)SDG_malloc(c->dm_count == 0 ? 1 : c->dm_count, sizeof(double));
    c->tmpden = (double *)SDG_malloc((size_t)spin_count *
                                     (c->output_count == 0 ? 1 : c->output_count), sizeof(double));
    ok = (degree != NULL && c->out_ptr != NULL && c->term_pt != NULL && c->pt_pair != NULL &&
          c->dm != NULL && c->tmpden != NULL);
  }

  if (ok) {
    for (p = 0; p < c->t.pair_count && ok; p++) {
      size_t base = c->atom_out_base[c->t.pair_Mc_AN[p]];
      size_t lo = c->t.pair_nolg_offset[p];
      size_t n = (size_t)c->t.pair_NOLG[p];

      for (pos = 0; pos < n; pos++) {
        size_t o = base + (size_t)c->t.nolg_Nc[lo + pos];
        if (o >= c->output_count) {
          ok = 0;
          break;
        }
        degree[o]++;
        c->pt_pair[lo + pos] = (uint32_t)p;
      }
    }
  }

  if (ok) {
    uint64_t running = 0;
    c->out_ptr[0] = 0;
    for (out = 0; out < c->output_count; out++) {
      running += (uint64_t)degree[out];
      if (running > UINT32_MAX) {
        ok = 0;
        break;
      }
      c->out_ptr[out + 1] = (uint32_t)running;
      degree[out] = c->out_ptr[out]; /* reuse as insertion cursor */
    }
    if (ok && (size_t)c->out_ptr[c->output_count] != c->term_count) ok = 0;
  }

  if (ok) {
    for (p = 0; p < c->t.pair_count; p++) {
      size_t base = c->atom_out_base[c->t.pair_Mc_AN[p]];
      size_t lo = c->t.pair_nolg_offset[p];
      size_t n = (size_t)c->t.pair_NOLG[p];

      for (pos = 0; pos < n; pos++) {
        size_t o = base + (size_t)c->t.nolg_Nc[lo + pos];
        c->term_pt[degree[o]++] = (uint32_t)(lo + pos);
      }
    }
  }

  free(degree);
  degree = NULL;

  if (ok) {
    c->extra_bytes = 0;
    ok = SDG_add_bytes(&c->extra_bytes, c->output_count + 1, sizeof(uint32_t)) &&
         SDG_add_bytes(&c->extra_bytes, c->term_count, sizeof(uint32_t)) &&
         SDG_add_bytes(&c->extra_bytes, c->term_count, sizeof(uint32_t)) &&
         SDG_add_bytes(&c->extra_bytes, c->dm_count, sizeof(double)) &&
         SDG_add_bytes(&c->extra_bytes, (size_t)spin_count * c->output_count, sizeof(double));
  }

  /* This builder is rank-local: OTF, cached, empty and failed ranks do
     not enter it.  Collective decisions belong to the public Prepare. */
  if (!ok) {
    SDG_local_free();
    SDG_local.unavailable = 1;
    return 0;
  }

  c->ready = 1;
  return 1;
}

/* Peak storage owned by local density, excluding shared orbital tables. */
static size_t SDG_local_workspace_bytes(const SDGLocalContext *c)
{
  size_t bytes = c->extra_bytes;
  const int mode = Grid_Precision_Kernel();

  /* The FP32/DF contraction stages an additional density-matrix image
     even when all CSR buffers are resident.  Include it in the rank's
     admission check and in the workspace budget shared with OTF ranks. */
  if ((mode == 1 || mode == 2) &&
      !SDG_add_bytes(&bytes, c->dm_count, (mode == 2 ? 2 : 1) * sizeof(float))) return SIZE_MAX;
  return bytes;
}

/* Device bytes this call would stage on top of what is already resident. */
static size_t SDG_local_call_bytes(const SDGLocalContext *c)
{
  size_t bytes = SDG_local_workspace_bytes(c);

  if (bytes == SIZE_MAX) return SIZE_MAX;
  if (c->device_resident) bytes -= c->extra_bytes;
  if (!c->t.orbs0_resident && !SDG_add_bytes(&bytes, c->t.total_orbs0, sizeof(Type_Orbs_Grid))) return SIZE_MAX;
  if (!c->t.orbs1_resident && !SDG_add_bytes(&bytes, c->t.total_orbs1, sizeof(Type_Orbs_Grid))) return SIZE_MAX;
  if (!c->t.nolg_resident && !SDG_add_bytes(&bytes, c->t.total_nolg, sizeof(int))) return SIZE_MAX;
  if (!c->t.meta_resident) {
    if (!SDG_add_bytes(&bytes, (size_t)c->t.pair_count, 2 * sizeof(int)) ||
        !SDG_add_bytes(&bytes, (size_t)c->t.pair_count, 4 * sizeof(size_t))) return SIZE_MAX;
  }
  return bytes;
}

/* the end of an SCF cycle: the resident device buffers of the local mode go
   (the next cycle stages them again) */
void Set_Density_Grid_GPU_EndCycle(void)
{
  SDG_local_delete_device(&SDG_local);
}

/* *vote_need / *vote_budget (zero on entry): this rank's figures for the
   stage vote of Set_Density_Grid_GPU_Local_Prepare, set by the CSR
   admission; zero need means no CSR staging of its own. */
static int SDG_local_prepare(const OpenMXAtomicViews *atomic, int Cnt_kind, int Calc_CntOrbital_ON, int node_ranks,
                             size_t *vote_need, size_t *vote_budget)
{
  SDGLocalContext *c = &SDG_local;
  int spin_count = SpinP_switch + 1;
  /* Without orbital optimization the contracted orbitals are the primitive
     ones (CNO == NO), so the final density of an MD step (Cnt_kind 1, for
     the forces) reads the kind-0 tables and tiles of Set_Hamiltonian too;
     it used to leave the quadrature to the one-rank service, 11.6 s per
     step on rank 0 for C1000 (RTX 5080, 6 ranks). */
  const int table_kind = (Cnt_switch == 0) ? 0 : Cnt_kind;
  int enabled, mode = 1;
  size_t reserve_bytes, need;
  size_t free_bytes = 0, total_bytes = 0;

  (void)Calc_CntOrbital_ON;

  /* 0 = off, 1 = auto (only when the shared Set_Hamiltonian tables are
     device-resident, so each call stages just the CSR/DM/density buffers),
     2 = always (stage the shared tables per call too) */
  mode = (int)openmx_env_long_at(OPENMX_ENV_DENSITY_GRID_GPU_LOCAL, mode);

  enabled = (mode != 0) &&
            openmx_env_flag_at(OPENMX_ENV_DENSITY_GRID_GPU, openmx_env_flag_at(OPENMX_ENV_SETDENSITY_GPU, 1)) &&
            scf_eigen_lib_flag == GPUSOLVER && SDG_solver_supported(1) &&
            Cnt_switch == 0 && (Cnt_kind == 0 || Cnt_kind == 1) &&
            (SpinP_switch == 0 || SpinP_switch == 1 || SpinP_switch == 3) &&
            (Matomnum == 0 || gpu_rank_device_usable());
  if (!enabled) return 0;
  /* No quadrature work, but this rank must not veto the other ranks. */
  if (Matomnum == 0) return 1;

  /* Mode 1 rides on the tables Set_Hamiltonian has already built.  Asking
     Set_Hamiltonian_GetMatrixElementsTables below would lazily BUILD the
     multi-GiB host tables on every rank whose Set_Hamiltonian ran on the
     CPU (a crowded device leaves most ranks without tables), and at
     1000-atom scale that simultaneous build alone can exhaust the host
     memory.  Probe for existing tables first.  All preparation here is
     rank-local; the public wrapper performs collectives on every rank. */
  if (mode == 1) {
    /* a rank with resident tables takes the CSR kernel, one without them the
       on-the-fly tiles (its Set_Hamiltonian evaluated its orbitals on the
       device as well); other ranks may keep their CPU quadrature */
    int myid = 0;
    int my_ready, my_otf = 0;

    MPI_Comm_rank(mpi_comm_level1, &myid);
    my_ready = Set_Hamiltonian_MatrixElementsTables_Ready(table_kind);
    if (my_ready) {
      SetHamiltonianMETables probe;
      if (!Set_Hamiltonian_GetMatrixElementsTables(table_kind, &probe) ||
          !(probe.orbs0_resident && probe.orbs1_resident && probe.nolg_resident && probe.meta_resident))
        my_ready = 0;
    }
    if (!my_ready) my_otf = Set_Hamiltonian_OnTheFly_DensityPossible(atomic, table_kind, myid);
    if (!(my_ready || my_otf)) return 0;
    if (my_otf) {
      if (c->ready && !c->otf) SDG_local_free();
      if (!c->otf || c->spin_count != spin_count || c->cnt_kind != table_kind) {
        int Mc_AN;
        size_t base = 0;
        SDG_local_free();
        c->otf_out_base = (uint32_t *)SDG_malloc((size_t)Matomnum + 2, sizeof(uint32_t));
        if (c->otf_out_base == NULL) return 0;
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
          c->otf_out_base[Mc_AN] = (uint32_t)base;
          const int n = GridN_Atom[M2G[Mc_AN]];
          if (n < 0 || (size_t)n > (size_t)UINT32_MAX - base) {
            SDG_local_free();
            return 0;
          }
          base += (size_t)n;
        }
        c->otf_out_base[Matomnum + 1] = (uint32_t)base;
        c->otf_output_count = base;
        c->otf_dm_count = Set_Hamiltonian_MatrixElements_TotalH(table_kind, myid);
        c->spin_count = spin_count;
        c->cnt_kind = table_kind;
        c->otf = 1;
      }
      return c->otf_output_count != 0;
    }
    if (c->otf) SDG_local_free();
  }

  if (c->otf) SDG_local_free(); /* switching auto OTF to forced CSR */
  if (c->unavailable) return 0;

  if (c->ready && (c->spin_count != spin_count || c->cnt_kind != table_kind)) {
    SDG_local_free();
  }
  if (!c->ready) {
    if (!SDG_local_build(table_kind, spin_count)) return 0;
  }

  /* refresh the shared-table view: the pointers and the residency flags may
     have changed since the epoch was built */
  if (!Set_Hamiltonian_GetMatrixElementsTables(table_kind, &c->t) ||
      c->t.total_nolg != c->term_count || c->t.total_h != c->dm_count) {
    SDG_local_free();
    SDG_local.unavailable = 1;
    return 0;
  }

  if (c->output_count == 0) return 0;

  /* In auto mode the per-call staging of the multi-GiB orbital tables is
     slower than the single-owner service, so engage only when the shared
     tables already live on the device (Set_Hamiltonian residency). */
  if (mode == 1 &&
      !(c->t.orbs0_resident && c->t.orbs1_resident && c->t.nolg_resident && c->t.meta_resident)) {
    return 0;
  }

  c->node_ranks = node_ranks > 0 ? node_ranks : 1;
  reserve_bytes = openmx_env_mib_bytes_at(OPENMX_ENV_DENSITY_GRID_GPU_RESERVE_MB, 256);
  if (c->device_resident && acc_get_device_num(acc_device_nvidia) != c->device_id)
    SDG_local_delete_device(c);
  if (OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
    *vote_need = SIZE_MAX;  /* the stage cannot be planned on this device */
    return 0;
  }
  need = SDG_local_call_bytes(c);
  /* the stage vote: this call's staging against the rank's share of the
     device, the figures of the admission below */
  *vote_need = need;
  *vote_budget = reserve_bytes <= free_bytes ? (free_bytes - reserve_bytes) / (size_t)c->node_ranks : 0;
  if (free_bytes < reserve_bytes) return 0;
  if ((free_bytes - reserve_bytes) / (size_t)c->node_ranks < need) return 0;
  return 1;
}

/* what the last Set_Density_Grid_GPU_Local_Prepare found, the same on
   every rank: the ranks with atoms, those that declined the distributed
   quadrature, and those among them without space for the minimum compact
   on-the-fly chunk (the largest such need and the smallest payload); host:
   the stage vote sent every rank to the host SIMD quadrature */
static struct {
  int atom_ranks;
  int declined;
  int ws_declined;
  unsigned long long ws_need;
  unsigned long long ws_avail;
  int host;
} SDG_census = {0};

/* Every rank participates, including OTF ranks, empty partitions and
   ranks whose local preparation failed.  A collective inside the CSR
   builder deadlocks when any other rank reuses a cache or chooses OTF. */
int Set_Density_Grid_GPU_Local_Prepare(const OpenMXAtomicViews *atomic, int Cnt_kind, int Calc_CntOrbital_ON)
{
  MPI_Comm node_comm = MPI_COMM_NULL;
  int node_ranks = 1, node_otf = 0, my_otf, ok, myid = 0, grow = 0, ws_declined = 0;
  size_t ws_need = 0, ws_avail = 0, vote_need = 0, vote_budget = 0;
  unsigned long long mine[2] = {0ULL, 0ULL}, node[2] = {0ULL, 0ULL};
  unsigned long long counts[4], totals[4], extremes[2], worst[2];

  /* Input and DFT_GPU_DeviceInit set this flag identically on every rank
     (device initialization failures are reduced collectively).  CPU runs
     need no local-GPU communicator.  Per-rank environment/device/cache
     decisions below must still participate in every collective. */
  if (scf_eigen_lib_flag != GPUSOLVER) {
    Set_Hamiltonian_OnTheFly_SetDensityRanks(0);
    OpenMX_GpuPhaseNeed_Register("density_grid_local", 0);
    memset(&SDG_census, 0, sizeof(SDG_census));
    return 0;
  }

  MPI_Comm_rank(mpi_comm_level1, &myid);
  MPI_Comm_split_type(mpi_comm_level1, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
  MPI_Comm_size(node_comm, &node_ranks);
  ok = SDG_local_prepare(atomic, Cnt_kind, Calc_CntOrbital_ON, node_ranks, &vote_need, &vote_budget);
  my_otf = ok && Matomnum > 0 && SDG_local.otf;
  MPI_Allreduce(&my_otf, &node_otf, 1, MPI_INT, MPI_SUM, node_comm);
  Set_Hamiltonian_OnTheFly_SetDensityRanks(node_otf);

  /* Plan the node's on-the-fly workspaces before any quadrature starts.
     Prefer full atom tiles when they fit; otherwise compact overlap-point
     chunks fit the physical share or an explicit workspace cap.  Without
     an explicit cap, ranks whose full tiles exceed the default may grow
     into space left by bounded OTF workspaces and resident-table CSR
     staging.  A rank declines only when even its minimum compact chunk
     cannot fit, or its memory budget is unavailable.  It then takes its
     CPU loop while the other ranks retain their distributed work. */
  if (my_otf) {
    mine[0] = (unsigned long long)Set_Hamiltonian_OnTheFly_DensityBudget(
        SDG_local.spin_count, SDG_local.otf_dm_count, SDG_local.otf_output_count, node_otf,
        &grow, &ws_need, &ws_avail, myid);
    if (0 < grow) mine[1] = 1ULL;
  }
  else if (ok && Matomnum > 0) {
    mine[0] = (unsigned long long)SDG_local_call_bytes(&SDG_local);
  }
  MPI_Allreduce(mine, node, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, node_comm);
  MPI_Comm_free(&node_comm);
  if (0 < grow && !Set_Hamiltonian_OnTheFly_DensityGrow((size_t)node[0], (int)node[1], &ws_avail)) grow = -1;
  /* the stage vote of an on-the-fly rank: the workspace the call prefers
     against the rank's share of the device (no valid plan: the memory query
     failed, the stage cannot be planned) */
  if (my_otf && !Set_Hamiltonian_OnTheFly_DensityStage(node_otf, &vote_need, &vote_budget)) vote_need = SIZE_MAX;
  if (grow < 0) {
    ok = 0;
    ws_declined = 1;
    if (3 <= level_stdout) {
      fprintf(stderr,
              "<Set_Density_Grid> rank %d: a minimum on-the-fly chunk needs %.3f MiB, %.3f MiB "
              "available on the device; the rank declines the distributed GPU quadrature\n",
              myid, (double)ws_need / (1024.0 * 1024.0), (double)ws_avail / (1024.0 * 1024.0));
      fflush(stderr);
    }
  }

  /* Publish CSR needs on the same communicator as the original policy;
     ranks without CSR buffers contribute zero.  The census rides along. */
  /* Include the transient precision image in the next-phase hint too, so
     Set_Hamiltonian can release enough resident storage before density. */
  counts[0] = SDG_local.ready && !SDG_local.otf ?
      (unsigned long long)SDG_local_workspace_bytes(&SDG_local) : 0ULL;
  counts[1] = Matomnum > 0 ? 1ULL : 0ULL;
  counts[2] = (Matomnum > 0 && !ok) ? 1ULL : 0ULL;
  counts[3] = (unsigned long long)ws_declined;
  MPI_Allreduce(counts, totals, 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM, mpi_comm_level1);
  worst[0] = 0ULL;
  worst[1] = ULLONG_MAX;
  if (0ULL < totals[3]) {
    /* the same count on every rank: the extremes of the declined ranks */
    extremes[0] = ws_declined ? (unsigned long long)ws_need : 0ULL;
    extremes[1] = ws_declined ? ULLONG_MAX - (unsigned long long)ws_avail : 0ULL;
    MPI_Allreduce(extremes, worst, 2, MPI_UNSIGNED_LONG_LONG, MPI_MAX, mpi_comm_level1);
  }
  OpenMX_GpuPhaseNeed_Register("density_grid_local", (size_t)totals[0]);
  SDG_census.atom_ranks = (int)totals[1];
  SDG_census.declined = (int)totals[2];
  SDG_census.ws_declined = (int)totals[3];
  SDG_census.ws_need = worst[0];
  SDG_census.ws_avail = totals[3] ? ULLONG_MAX - worst[1] : 0ULL;

  /* The stage vote, the last collective of the preparation (every rank of a
     GPU run reaches it): the distributed quadrature's device working set on
     this rank (the CSR staging of the call, or the workspace an on-the-fly
     rank prefers) against the rank's share of free memory less the reserve.
     After a host decision every rank declines, and Set_Density_Grid skips
     the one-rank service and the capacity recovery as well
     (Set_Density_Grid_GPU_Local_StageOnHost). */
  SDG_census.host = OpenMX_HostSimdStageOnHost("Set_Density_Grid", vote_need, vote_budget);
  if (SDG_census.host) ok = 0;
  return ok;
}

/* The stage vote of the last Set_Density_Grid_GPU_Local_Prepare, the same on
   every rank: 1 = the whole call on the host SIMD quadrature. No collective. */
int Set_Density_Grid_GPU_Local_StageOnHost(void)
{
  return SDG_census.host;
}

/* The census was reduced by Prepare, so every rank makes the same choice
   without another collective.  Memory pressure must not funnel all atoms
   through a single-owner service, nor discard useful distributed GPU work
   merely because another rank cannot use it.  Empty partitions do not
   count as capable atom ranks.  With the local mode explicitly disabled,
   or without a distributed implementation for this call, both conditions
   are false and the existing service remains available. */
int Set_Density_Grid_GPU_Local_KeepDistributed(void)
{
  return SDG_census.ws_declined > 0 || SDG_census.declined < SDG_census.atom_ranks;
}

/* The level-2 line of a Set_Density_Grid call in which ranks with atoms
   could not hold a minimum on-the-fly chunk or failed on the
   device, on Host_ID when the situation changes.  path 1: the one-rank
   service integrated the density.  path 0 or 2: the distributed quadrature
   ran (2: without using the service, so every declined rank took the CPU loop),
   and failed is the number of ranks with atoms whose quadrature failed on
   the device and took the CPU loop (summed by Set_Density_Grid).  Ranks
   without a device path for a call (orbital optimization, an unsupported
   solver) take the service or the CPU loop silently, as designed.  No
   collectives; every rank keeps the same state. */
void Set_Density_Grid_GPU_Local_Report(int path, int failed)
{
  static int last_path = -1, last_cpu = -1, last_ws = -1, last_failed = -1;
  int myid, cpu, other;
  const int ws = SDG_census.ws_declined;
  char what[192] = "";

  if (ws == 0 && failed == 0) {
    /* a later departure is announced again */
    last_path = -1;
    return;
  }
  cpu = (path == 2 ? SDG_census.declined : 0) + failed;
  if (path == last_path && cpu == last_cpu && ws == last_ws && failed == last_failed) return;
  last_path = path;
  last_cpu = cpu;
  last_ws = ws;
  last_failed = failed;
  MPI_Comm_rank(mpi_comm_level1, &myid);
  if (myid != Host_ID || level_stdout < 2) return;

  if (0 < ws) {
    snprintf(what, sizeof(what),
             "cannot hold a minimum on-the-fly chunk (up to %.3f MiB needed, %.3f MiB available)",
             (double)SDG_census.ws_need / (1024.0 * 1024.0), (double)SDG_census.ws_avail / (1024.0 * 1024.0));
  }
  if (path == 1) {
    printf("<Set_Density_Grid> %d of %d rank(s) with atoms %s; the one-rank GPU service integrates the density\n",
           ws, SDG_census.atom_ranks, what);
  }
  else {
    const char *sep = ": ";

    other = cpu - failed - (path == 2 ? ws : 0);
    printf("<Set_Density_Grid> CPU loop on %d of %d rank(s) with atoms", cpu, SDG_census.atom_ranks);
    if (path == 2 && 0 < ws) {
      printf("%s%d %s; keeping distributed quadrature", sep, ws, what);
      sep = "; ";
    }
    if (0 < failed) {
      printf("%s%d failed on the device", sep, failed);
      sep = "; ";
    }
    if (0 < other) printf("%s%d without a device path for this call", sep, other);
    printf("\n");
  }
  fflush(stdout);
}


typedef struct {
  int mode,spin_count,pair_count;
  size_t output_count,term_count,dm_count,dm_total,total_nolg,total_orbs0,total_orbs1;
  int *pair_NO0;
  int *pair_NO1;
  size_t *pair_h_offset;
  size_t *pair_nolg_offset;
  size_t *pair_orbs0_offset;
  size_t *pair_orbs1_offset;
  int *nolg_Nc;
  Type_Orbs_Grid *orbs0buf;
  Type_Orbs_Grid *orbs1buf;
  uint32_t *out_ptr;
  uint32_t *term_pt;
  uint32_t *pt_pair;
  double *dm;
  float *dmf;
  double *tmpden;
} SDGKernelInput;

static void SDG_local_range(SDGKernelInput *w,size_t first,size_t last)
{
  const int mode=w->mode,fp32=(mode==1),spin_count=w->spin_count;
  const size_t dm_count=w->dm_count;
  int *pair_NO0=w->pair_NO0;
  int *pair_NO1=w->pair_NO1;
  size_t *pair_h_offset=w->pair_h_offset;
  size_t *pair_nolg_offset=w->pair_nolg_offset;
  size_t *pair_orbs0_offset=w->pair_orbs0_offset;
  size_t *pair_orbs1_offset=w->pair_orbs1_offset;
  int *nolg_Nc=w->nolg_Nc;
  Type_Orbs_Grid *orbs0buf=w->orbs0buf;
  Type_Orbs_Grid *orbs1buf=w->orbs1buf;
  uint32_t *out_ptr=w->out_ptr;
  uint32_t *term_pt=w->term_pt;
  uint32_t *pt_pair=w->pt_pair;
  double *dm=w->dm;
  float *dmf=w->dmf;
  double *tmpden=w->tmpden;
    if (mode == 2) {
      /* double-float: the density matrix as hi + lo floats (dmf, dmf + dm_count),
         every product exact through an FMA, the sum over j compensated, the
         sums over i and over the pairs in FP64 */
      {
#pragma acc parallel loop gang vector_length(128) deviceptr(pair_NO0,pair_NO1,pair_h_offset,pair_nolg_offset,pair_orbs0_offset,pair_orbs1_offset,nolg_Nc,orbs0buf,orbs1buf,out_ptr,term_pt,pt_pair,dm,dmf,tmpden)
    for (size_t out = first; out < last; out++) {
      double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0;

#pragma acc loop seq
      for (uint32_t q = out_ptr[out]; q < out_ptr[out + 1]; q++) {
        uint32_t pt = term_pt[q];
        uint32_t pair = pt_pair[pt];
        int NO0 = pair_NO0[pair];
        int NO1 = pair_NO1[pair];
        size_t mat = (size_t)NO0 * (size_t)NO1;
        size_t base = pair_h_offset[pair];
        size_t o0 = pair_orbs0_offset[pair] + (size_t)nolg_Nc[pt] * (size_t)NO0;
        size_t o1 = pair_orbs1_offset[pair] + ((size_t)pt - pair_nolg_offset[pair]) * (size_t)NO1;
        double e0 = 0.0, e1 = 0.0, e2 = 0.0, e3 = 0.0;

#pragma acc loop seq
        for (int ii = 0; ii < NO0; ii++) {
          const double f0 = (double)orbs0buf[o0 + (size_t)ii];
          int sp;
#pragma acc loop seq
          for (sp = 0; sp < spin_count; sp++) {
            const size_t row = base + (size_t)sp * mat + (size_t)ii * (size_t)NO1;
            float ts = 0.0f, tc = 0.0f;
            double t;
#pragma acc loop seq
            for (int jj = 0; jj < NO1; jj++) {
              const float f = orbs1buf[o1 + (size_t)jj];
              const float hi = dmf[row + (size_t)jj];
              const float lo = dmf[dm_count + row + (size_t)jj];
              const float pr = f * hi;
              const float er = fmaf(f, hi, -pr);
              const float qq = fmaf(f, lo, er);
              float u = ts + pr;
              if (fabsf(ts) >= fabsf(pr)) tc += (ts - u) + pr; else tc += (pr - u) + ts;
              ts = u;
              u = ts + qq;
              if (fabsf(ts) >= fabsf(qq)) tc += (ts - u) + qq; else tc += (qq - u) + ts;
              ts = u;
            }
            t = (double)ts + (double)tc;
            if (sp == 0) e0 += f0 * t;
            else if (sp == 1) e1 += f0 * t;
            else if (sp == 2) e2 += f0 * t;
            else e3 += f0 * t;
          }
        }
        g0 += e0;
        if (spin_count >= 2) g1 += e1;
        if (spin_count == 4) {
          g2 += e2;
          g3 += e3;
        }
      }

      tmpden[out-first] = g0;
      if (spin_count >= 2) tmpden[(last-first) + out-first] = g1;
      if (spin_count == 4) {
        tmpden[2U * (last-first) + out-first] = g2;
        tmpden[3U * (last-first) + out-first] = g3;
      }
    }
      }
    }
    else if (fp32) {
      {
#pragma acc parallel loop gang vector_length(128) deviceptr(pair_NO0,pair_NO1,pair_h_offset,pair_nolg_offset,pair_orbs0_offset,pair_orbs1_offset,nolg_Nc,orbs0buf,orbs1buf,out_ptr,term_pt,pt_pair,dm,dmf,tmpden)
    for (size_t out = first; out < last; out++) {
      double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0;

#pragma acc loop seq
      for (uint32_t q = out_ptr[out]; q < out_ptr[out + 1]; q++) {
        uint32_t pt = term_pt[q];
        uint32_t pair = pt_pair[pt];
        int NO0 = pair_NO0[pair];
        int NO1 = pair_NO1[pair];
        size_t mat = (size_t)NO0 * (size_t)NO1;
        size_t base = pair_h_offset[pair];
        size_t o0 = pair_orbs0_offset[pair] + (size_t)nolg_Nc[pt] * (size_t)NO0;
        size_t o1 = pair_orbs1_offset[pair] + ((size_t)pt - pair_nolg_offset[pair]) * (size_t)NO1;
        float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f, e3 = 0.0f;

#pragma acc loop seq
        for (int ii = 0; ii < NO0; ii++) {
          float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
#pragma acc loop seq
          for (int jj = 0; jj < NO1; jj++) {
            float phi1 = orbs1buf[o1 + (size_t)jj];
            size_t ij = (size_t)ii * (size_t)NO1 + (size_t)jj;
            t0 += phi1 * dmf[base + ij];
            if (spin_count >= 2) t1 += phi1 * dmf[base + mat + ij];
            if (spin_count == 4) {
              t2 += phi1 * dmf[base + 2U * mat + ij];
              t3 += phi1 * dmf[base + 3U * mat + ij];
            }
          }
          {
            float phi0 = orbs0buf[o0 + (size_t)ii];
            e0 += phi0 * t0;
            if (spin_count >= 2) e1 += phi0 * t1;
            if (spin_count == 4) {
              e2 += phi0 * t2;
              e3 += phi0 * t3;
            }
          }
        }
        g0 += (double)e0;
        if (spin_count >= 2) g1 += (double)e1;
        if (spin_count == 4) {
          g2 += (double)e2;
          g3 += (double)e3;
        }
      }

      tmpden[out-first] = g0;
      if (spin_count >= 2) tmpden[(last-first) + out-first] = g1;
      if (spin_count == 4) {
        tmpden[2U * (last-first) + out-first] = g2;
        tmpden[3U * (last-first) + out-first] = g3;
      }
    }
      }
    }
    else {
#pragma acc parallel loop gang vector_length(128) deviceptr(pair_NO0,pair_NO1,pair_h_offset,pair_nolg_offset,pair_orbs0_offset,pair_orbs1_offset,nolg_Nc,orbs0buf,orbs1buf,out_ptr,term_pt,pt_pair,dm,dmf,tmpden)
    for (size_t out = first; out < last; out++) {
      double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0;

#pragma acc loop seq
      for (uint32_t q = out_ptr[out]; q < out_ptr[out + 1]; q++) {
        uint32_t pt = term_pt[q];
        uint32_t pair = pt_pair[pt];
        int NO0 = pair_NO0[pair];
        int NO1 = pair_NO1[pair];
        size_t mat = (size_t)NO0 * (size_t)NO1;
        size_t base = pair_h_offset[pair];
        size_t o0 = pair_orbs0_offset[pair] + (size_t)nolg_Nc[pt] * (size_t)NO0;
        size_t o1 = pair_orbs1_offset[pair] + ((size_t)pt - pair_nolg_offset[pair]) * (size_t)NO1;
        double e0 = 0.0, e1 = 0.0, e2 = 0.0, e3 = 0.0;

#pragma acc loop seq
        for (int ii = 0; ii < NO0; ii++) {
          double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
#pragma acc loop seq
          for (int jj = 0; jj < NO1; jj++) {
            double phi1 = (double)orbs1buf[o1 + (size_t)jj];
            size_t ij = (size_t)ii * (size_t)NO1 + (size_t)jj;
            t0 += phi1 * dm[base + ij];
            if (spin_count >= 2) t1 += phi1 * dm[base + mat + ij];
            if (spin_count == 4) {
              t2 += phi1 * dm[base + 2U * mat + ij];
              t3 += phi1 * dm[base + 3U * mat + ij];
            }
          }
          {
            double phi0 = (double)orbs0buf[o0 + (size_t)ii];
            e0 += phi0 * t0;
            if (spin_count >= 2) e1 += phi0 * t1;
            if (spin_count == 4) {
              e2 += phi0 * t2;
              e3 += phi0 * t3;
            }
          }
        }
        g0 += e0;
        if (spin_count >= 2) g1 += e1;
        if (spin_count == 4) {
          g2 += e2;
          g3 += e3;
        }
      }

      tmpden[out-first] = g0;
      if (spin_count >= 2) tmpden[(last-first) + out-first] = g1;
      if (spin_count == 4) {
        tmpden[2U * (last-first) + out-first] = g2;
        tmpden[3U * (last-first) + out-first] = g3;
      }
    }
    }

}

typedef struct {
  OpenMXGpuWorkers workers;
  SDGKernelInput source,device[OPENMX_GPU_WORKERS_MAX];
  OpenMXGpuReadView input[OPENMX_GPU_WORKERS_MAX][14];
} SDGLocalWork;

static int SDG_local_worker(void *opaque,int worker,int phase)
{
  SDGLocalWork *w=(SDGLocalWork*)opaque;
  SDGKernelInput *s=&w->source,*d=w->device+worker;
  size_t first=openmx_gpu_worker_cut(w->workers.items,worker,w->workers.count);
  size_t last=openmx_gpu_worker_cut(w->workers.items,worker+1,w->workers.count);
  if (phase==OPENMX_GPU_WORKER_PREPARE) {
    *d=*s;d->tmpden=NULL;
    if (!openmx_gpu_read_view(&w->input[worker][0],s->pair_NO0,(s->pair_count)*sizeof(int))) return 0;
    d->pair_NO0=(int*)w->input[worker][0].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][1],s->pair_NO1,(s->pair_count)*sizeof(int))) return 0;
    d->pair_NO1=(int*)w->input[worker][1].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][2],s->pair_h_offset,(s->pair_count)*sizeof(size_t))) return 0;
    d->pair_h_offset=(size_t*)w->input[worker][2].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][3],s->pair_nolg_offset,(s->pair_count)*sizeof(size_t))) return 0;
    d->pair_nolg_offset=(size_t*)w->input[worker][3].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][4],s->pair_orbs0_offset,(s->pair_count)*sizeof(size_t))) return 0;
    d->pair_orbs0_offset=(size_t*)w->input[worker][4].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][5],s->pair_orbs1_offset,(s->pair_count)*sizeof(size_t))) return 0;
    d->pair_orbs1_offset=(size_t*)w->input[worker][5].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][6],s->nolg_Nc,(s->total_nolg)*sizeof(int))) return 0;
    d->nolg_Nc=(int*)w->input[worker][6].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][7],s->orbs0buf,(s->total_orbs0)*sizeof(Type_Orbs_Grid))) return 0;
    d->orbs0buf=(Type_Orbs_Grid*)w->input[worker][7].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][8],s->orbs1buf,(s->total_orbs1)*sizeof(Type_Orbs_Grid))) return 0;
    d->orbs1buf=(Type_Orbs_Grid*)w->input[worker][8].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][9],s->out_ptr,(s->output_count+1)*sizeof(uint32_t))) return 0;
    d->out_ptr=(uint32_t*)w->input[worker][9].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][10],s->term_pt,(s->term_count)*sizeof(uint32_t))) return 0;
    d->term_pt=(uint32_t*)w->input[worker][10].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][11],s->pt_pair,(s->term_count)*sizeof(uint32_t))) return 0;
    d->pt_pair=(uint32_t*)w->input[worker][11].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][12],s->dm,(s->dm_count)*sizeof(double))) return 0;
    d->dm=(double*)w->input[worker][12].pointer;
    if (!openmx_gpu_read_view(&w->input[worker][13],s->dmf,(s->dm_total)*sizeof(float))) return 0;
    d->dmf=(float*)w->input[worker][13].pointer;
    return cudaMalloc((void**)&d->tmpden,(last-first)*(size_t)s->spin_count*sizeof(double))==cudaSuccess;
  }
  if (phase==OPENMX_GPU_WORKER_EXECUTE) SDG_local_range(d,first,last);
  else if (phase==OPENMX_GPU_WORKER_COMMIT) {
    for (int spin=0;spin<s->spin_count;spin++)
      if (!openmx_gpu_copy_device(s->tmpden+(size_t)spin*s->output_count+first,
          d->tmpden+(size_t)spin*(last-first),(last-first)*sizeof(double))) return 0;
  } else if (phase==OPENMX_GPU_WORKER_RELEASE) {
    if (d->tmpden) cudaFree(d->tmpden);
    for (int i=0;i<14;i++) openmx_gpu_read_view_release(&w->input[worker][i]);
  }
  return 1;
}

static int SDG_local_distribute(SDGKernelInput *input)
{
  SDGLocalWork w={0};w.source=*input;
  openmx_gpu_workers_local(&w.workers,input->output_count);
  if (w.workers.count<=1) return 0;
  w.source.pair_NO0=(int*)acc_deviceptr((void*)input->pair_NO0);
  w.source.pair_NO1=(int*)acc_deviceptr((void*)input->pair_NO1);
  w.source.pair_h_offset=(size_t*)acc_deviceptr((void*)input->pair_h_offset);
  w.source.pair_nolg_offset=(size_t*)acc_deviceptr((void*)input->pair_nolg_offset);
  w.source.pair_orbs0_offset=(size_t*)acc_deviceptr((void*)input->pair_orbs0_offset);
  w.source.pair_orbs1_offset=(size_t*)acc_deviceptr((void*)input->pair_orbs1_offset);
  w.source.nolg_Nc=(int*)acc_deviceptr((void*)input->nolg_Nc);
  w.source.orbs0buf=(Type_Orbs_Grid*)acc_deviceptr((void*)input->orbs0buf);
  w.source.orbs1buf=(Type_Orbs_Grid*)acc_deviceptr((void*)input->orbs1buf);
  w.source.out_ptr=(uint32_t*)acc_deviceptr((void*)input->out_ptr);
  w.source.term_pt=(uint32_t*)acc_deviceptr((void*)input->term_pt);
  w.source.pt_pair=(uint32_t*)acc_deviceptr((void*)input->pt_pair);
  w.source.dm=(double*)acc_deviceptr((void*)input->dm);
  w.source.tmpden=(double*)acc_deviceptr(input->tmpden);
  /* The caller owns the transient float mapping for this attempt. */
  w.source.dmf=input->dm_total?(float*)acc_deviceptr(input->dmf):NULL;
  int ok=openmx_gpu_workers_run(&w.workers,SDG_local_worker,&w,"density-local-CSR");
  return ok;
}

int Set_Density_Grid_GPU_Local_Run(const OpenMXAtomicViews *atomic, double *****CDM, double ***Tmp_Den_Grid)
{
  SDGLocalContext *c = &SDG_local;
  const int spin_count = c->spin_count;
  const int pair_count = c->t.pair_count;
  const size_t output_count = c->output_count;
  const size_t term_count = c->term_count;
  const size_t dm_count = c->dm_count;
  const size_t total_nolg = c->t.total_nolg;
  const size_t total_orbs0 = c->t.total_orbs0;
  const size_t total_orbs1 = c->t.total_orbs1;
  const int *pair_NO0 = c->t.pair_NO0;
  const int *pair_NO1 = c->t.pair_NO1;
  const int *nolg_Nc = c->t.nolg_Nc;
  const size_t *pair_h_offset = c->t.pair_h_offset;
  const size_t *pair_nolg_offset = c->t.pair_nolg_offset;
  const size_t *pair_orbs0_offset = c->t.pair_orbs0_offset;
  const size_t *pair_orbs1_offset = c->t.pair_orbs1_offset;
  const Type_Orbs_Grid *orbs0buf = c->t.orbs0buf;
  const Type_Orbs_Grid *orbs1buf = c->t.orbs1buf;
  uint32_t *out_ptr = c->out_ptr;
  uint32_t *term_pt = c->term_pt;
  uint32_t *pt_pair = c->pt_pair;
  double *dm = c->dm;
  double *tmpden = c->tmpden;
  float *dmf = NULL;
  int fp32 = 0, mode = 0;
  size_t dm_total = 0;
  int Mc_AN, spin;

  if (Matomnum == 0) return 1;
  if (c->otf) {
    /* the on-the-fly mode: the packed density matrix and the density in
       transient host buffers, the tiles on the device */
    const size_t n_out = c->otf_output_count;
    /* persistent, pinned when possible: dm_otf goes up and den_otf comes
       back every step (tens of MiB; a pageable transfer runs at a fraction
       of the PCIe rate), so they are kept across calls (grow-only) */
    static double *dm_otf = NULL, *den_otf = NULL;
    static size_t dm_cap = 0, den_cap = 0;
    static int dm_pageable = 0, den_pageable = 0;
    const size_t dm_need = c->otf_dm_count ? c->otf_dm_count : 1;
    const size_t den_need = (size_t)spin_count * (n_out ? n_out : 1);
    int ok;
    if (dm_cap < dm_need) {
      if (dm_otf != NULL) { if (dm_pageable) free(dm_otf); else (void)cudaFreeHost(dm_otf); }
      dm_otf = NULL; dm_cap = 0; dm_pageable = 0;
      if (cudaMallocHost((void **)&dm_otf, sizeof(double) * dm_need) != cudaSuccess) {
        (void)cudaGetLastError();
        dm_otf = (double *)SDG_malloc(dm_need, sizeof(double));
        dm_pageable = 1;
      }
      if (dm_otf != NULL) dm_cap = dm_need;
    }
    if (den_cap < den_need) {
      if (den_otf != NULL) { if (den_pageable) free(den_otf); else (void)cudaFreeHost(den_otf); }
      den_otf = NULL; den_cap = 0; den_pageable = 0;
      if (cudaMallocHost((void **)&den_otf, sizeof(double) * den_need) != cudaSuccess) {
        (void)cudaGetLastError();
        den_otf = (double *)SDG_malloc(den_need, sizeof(double));
        den_pageable = 1;
      }
      if (den_otf != NULL) den_cap = den_need;
    }
    ok = dm_otf != NULL && den_otf != NULL && n_out != 0;
    int myid = 0;

    MPI_Comm_rank(mpi_comm_level1, &myid);
    if (ok) ok = SDG_pack_local_cdm(CDM, dm_otf, c->otf_dm_count, spin_count);
    if (ok) ok = Set_Hamiltonian_OnTheFly_Density(atomic, c->cnt_kind, spin_count, dm_otf, c->otf_dm_count, den_otf, n_out,
                                                  c->otf_out_base, myid);
    if (ok) {
      for (spin = 0; spin < spin_count; spin++) {
        for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
          size_t base = c->otf_out_base[Mc_AN];
          size_t n = (size_t)GridN_Atom[M2G[Mc_AN]];

          memcpy(Tmp_Den_Grid[spin][Mc_AN], den_otf + (size_t)spin * n_out + base, sizeof(double) * n);
        }
      }
    }
    /* dm_otf / den_otf persist */
    return ok;
  }

  if (!c->ready || output_count == 0) return 0;
  if (!SDG_pack_local_cdm(CDM, dm, dm_count, spin_count)) return 0;

  /* This CSR and its buffers are invariant in size until geometry changes.
     Bound all ranks' retained storage to a small fraction of free VRAM, so
     a crowded device still leaves room for the next dense solve. */
  {
    size_t cap = openmx_env_mib_bytes_at(OPENMX_ENV_DENSITY_GRID_GPU_LOCAL_CACHE_MB, 32);
    size_t free_bytes = 0, total_bytes = 0;
    int keep_resident = cap != 0 && c->extra_bytes <= cap;

    if (keep_resident && !c->device_resident)
      keep_resident = OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
                     c->extra_bytes <= free_bytes / 32U / (size_t)c->node_ranks;
    /* Nonempty output with no terms takes the original transient path. */
    keep_resident = keep_resident && term_count != 0 && dm_count != 0;
    if (!keep_resident) SDG_local_delete_device(c);
    else if (!SDG_local_upload(c)) return 0;
    if (c->device_resident) acc_update_device(dm, dm_count * sizeof(double));
  }

  /* the FP32 stage of the grid integrals (grid_precision.h): the products
     and the short sums over the orbitals of a pair in FP32 from a float
     copy of the density matrix, the sum over the pairs in FP64 */
  mode = Grid_Precision_Kernel();
  fp32 = (mode == 1);
  if (mode == 1 || mode == 2) {
    dmf = (float *)malloc(sizeof(float) * dm_count * (mode == 2 ? 2 : 1));
    if (dmf == NULL) { fp32 = 0; mode = 0; }
    else {
      for (size_t q = 0; q < dm_count; q++) dmf[q] = (float)dm[q];
      if (mode == 2) for (size_t q = 0; q < dm_count; q++) dmf[dm_count + q] = (float)(dm[q] - (double)dmf[q]);
    }
  }
  dm_total = dm_count * (mode == 2 ? 2 : 1);

  void *map_host[15]={(void*)pair_NO0,(void*)pair_NO1,(void*)pair_h_offset,
    (void*)pair_nolg_offset,(void*)pair_orbs0_offset,(void*)pair_orbs1_offset,
    (void*)nolg_Nc,(void*)orbs0buf,(void*)orbs1buf,out_ptr,term_pt,pt_pair,dm,dmf,tmpden};
  size_t map_bytes[15]={pair_count*sizeof(int),pair_count*sizeof(int),pair_count*sizeof(size_t),
    pair_count*sizeof(size_t),pair_count*sizeof(size_t),pair_count*sizeof(size_t),
    total_nolg*sizeof(int),total_orbs0*sizeof(Type_Orbs_Grid),total_orbs1*sizeof(Type_Orbs_Grid),
    (output_count+1)*sizeof(uint32_t),term_count*sizeof(uint32_t),term_count*sizeof(uint32_t),
    dm_count*sizeof(double),dmf?dm_total*sizeof(float):0,(size_t)spin_count*output_count*sizeof(double)};
  int created[15]={0};
  for (int i=0;i<15;i++) {
    if (!map_bytes[i] || acc_deviceptr(map_host[i])) continue;
    if (!SDG_map(map_host[i],map_bytes[i],i!=14)) goto sdg_map_failure;
    created[i]=1;
  }

#pragma acc data copyin(pair_NO0[0:pair_count], pair_NO1[0:pair_count],                            \
                        pair_h_offset[0:pair_count], pair_nolg_offset[0:pair_count],               \
                        pair_orbs0_offset[0:pair_count], pair_orbs1_offset[0:pair_count],          \
                        nolg_Nc[0:total_nolg], orbs0buf[0:total_orbs0], orbs1buf[0:total_orbs1],   \
                        out_ptr[0:output_count+1], term_pt[0:term_count], pt_pair[0:term_count],   \
                        dm[0:dm_count])                                                            \
                 copyout(tmpden[0:(size_t)spin_count*output_count])
  {
    SDGKernelInput distributed={0};
    distributed.mode=mode;
    distributed.spin_count=spin_count;
    distributed.pair_count=pair_count;
    distributed.output_count=output_count;
    distributed.term_count=term_count;
    distributed.total_nolg=total_nolg;
    distributed.total_orbs0=total_orbs0;
    distributed.total_orbs1=total_orbs1;
    distributed.tmpden=tmpden;
    distributed.pair_NO0=(void*)pair_NO0;
    distributed.pair_NO1=(void*)pair_NO1;
    distributed.pair_h_offset=(void*)pair_h_offset;
    distributed.pair_nolg_offset=(void*)pair_nolg_offset;
    distributed.pair_orbs0_offset=(void*)pair_orbs0_offset;
    distributed.pair_orbs1_offset=(void*)pair_orbs1_offset;
    distributed.nolg_Nc=(void*)nolg_Nc;
    distributed.orbs0buf=(void*)orbs0buf;
    distributed.orbs1buf=(void*)orbs1buf;
    distributed.out_ptr=(void*)out_ptr;
    distributed.term_pt=(void*)term_pt;
    distributed.pt_pair=(void*)pt_pair;
    distributed.dm=(void*)dm;
    distributed.dmf=(void*)dmf;
    distributed.dm_count=dm_count;distributed.dm_total=dmf?dm_total:0;
    if (!SDG_local_distribute(&distributed)) {
    if (mode == 2) {
      /* double-float: the density matrix as hi + lo floats (dmf, dmf + dm_count),
         every product exact through an FMA, the sum over j compensated, the
         sums over i and over the pairs in FP64 */
#pragma acc data copyin(dmf[0:dm_total])
      {
#pragma acc parallel loop gang vector_length(128)                                                  \
    present(pair_NO0[0:pair_count], pair_NO1[0:pair_count], pair_h_offset[0:pair_count],           \
            pair_nolg_offset[0:pair_count], pair_orbs0_offset[0:pair_count],                       \
            pair_orbs1_offset[0:pair_count], nolg_Nc[0:total_nolg], orbs0buf[0:total_orbs0],       \
            orbs1buf[0:total_orbs1], out_ptr[0:output_count+1], term_pt[0:term_count],             \
            pt_pair[0:term_count], dmf[0:dm_total], tmpden[0:(size_t)spin_count*output_count])
    for (size_t out = 0; out < output_count; out++) {
      double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0;

#pragma acc loop seq
      for (uint32_t q = out_ptr[out]; q < out_ptr[out + 1]; q++) {
        uint32_t pt = term_pt[q];
        uint32_t pair = pt_pair[pt];
        int NO0 = pair_NO0[pair];
        int NO1 = pair_NO1[pair];
        size_t mat = (size_t)NO0 * (size_t)NO1;
        size_t base = pair_h_offset[pair];
        size_t o0 = pair_orbs0_offset[pair] + (size_t)nolg_Nc[pt] * (size_t)NO0;
        size_t o1 = pair_orbs1_offset[pair] + ((size_t)pt - pair_nolg_offset[pair]) * (size_t)NO1;
        double e0 = 0.0, e1 = 0.0, e2 = 0.0, e3 = 0.0;

#pragma acc loop seq
        for (int ii = 0; ii < NO0; ii++) {
          const double f0 = (double)orbs0buf[o0 + (size_t)ii];
          int sp;
#pragma acc loop seq
          for (sp = 0; sp < spin_count; sp++) {
            const size_t row = base + (size_t)sp * mat + (size_t)ii * (size_t)NO1;
            float ts = 0.0f, tc = 0.0f;
            double t;
#pragma acc loop seq
            for (int jj = 0; jj < NO1; jj++) {
              const float f = orbs1buf[o1 + (size_t)jj];
              const float hi = dmf[row + (size_t)jj];
              const float lo = dmf[dm_count + row + (size_t)jj];
              const float pr = f * hi;
              const float er = fmaf(f, hi, -pr);
              const float qq = fmaf(f, lo, er);
              float u = ts + pr;
              if (fabsf(ts) >= fabsf(pr)) tc += (ts - u) + pr; else tc += (pr - u) + ts;
              ts = u;
              u = ts + qq;
              if (fabsf(ts) >= fabsf(qq)) tc += (ts - u) + qq; else tc += (qq - u) + ts;
              ts = u;
            }
            t = (double)ts + (double)tc;
            if (sp == 0) e0 += f0 * t;
            else if (sp == 1) e1 += f0 * t;
            else if (sp == 2) e2 += f0 * t;
            else e3 += f0 * t;
          }
        }
        g0 += e0;
        if (spin_count >= 2) g1 += e1;
        if (spin_count == 4) {
          g2 += e2;
          g3 += e3;
        }
      }

      tmpden[out] = g0;
      if (spin_count >= 2) tmpden[output_count + out] = g1;
      if (spin_count == 4) {
        tmpden[2U * output_count + out] = g2;
        tmpden[3U * output_count + out] = g3;
      }
    }
      }
    }
    else if (fp32) {
#pragma acc data copyin(dmf[0:dm_count])
      {
#pragma acc parallel loop gang vector_length(128)                                                  \
    present(pair_NO0[0:pair_count], pair_NO1[0:pair_count], pair_h_offset[0:pair_count],           \
            pair_nolg_offset[0:pair_count], pair_orbs0_offset[0:pair_count],                       \
            pair_orbs1_offset[0:pair_count], nolg_Nc[0:total_nolg], orbs0buf[0:total_orbs0],       \
            orbs1buf[0:total_orbs1], out_ptr[0:output_count+1], term_pt[0:term_count],             \
            pt_pair[0:term_count], dmf[0:dm_count], tmpden[0:(size_t)spin_count*output_count])
    for (size_t out = 0; out < output_count; out++) {
      double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0;

#pragma acc loop seq
      for (uint32_t q = out_ptr[out]; q < out_ptr[out + 1]; q++) {
        uint32_t pt = term_pt[q];
        uint32_t pair = pt_pair[pt];
        int NO0 = pair_NO0[pair];
        int NO1 = pair_NO1[pair];
        size_t mat = (size_t)NO0 * (size_t)NO1;
        size_t base = pair_h_offset[pair];
        size_t o0 = pair_orbs0_offset[pair] + (size_t)nolg_Nc[pt] * (size_t)NO0;
        size_t o1 = pair_orbs1_offset[pair] + ((size_t)pt - pair_nolg_offset[pair]) * (size_t)NO1;
        float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f, e3 = 0.0f;

#pragma acc loop seq
        for (int ii = 0; ii < NO0; ii++) {
          float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
#pragma acc loop seq
          for (int jj = 0; jj < NO1; jj++) {
            float phi1 = orbs1buf[o1 + (size_t)jj];
            size_t ij = (size_t)ii * (size_t)NO1 + (size_t)jj;
            t0 += phi1 * dmf[base + ij];
            if (spin_count >= 2) t1 += phi1 * dmf[base + mat + ij];
            if (spin_count == 4) {
              t2 += phi1 * dmf[base + 2U * mat + ij];
              t3 += phi1 * dmf[base + 3U * mat + ij];
            }
          }
          {
            float phi0 = orbs0buf[o0 + (size_t)ii];
            e0 += phi0 * t0;
            if (spin_count >= 2) e1 += phi0 * t1;
            if (spin_count == 4) {
              e2 += phi0 * t2;
              e3 += phi0 * t3;
            }
          }
        }
        g0 += (double)e0;
        if (spin_count >= 2) g1 += (double)e1;
        if (spin_count == 4) {
          g2 += (double)e2;
          g3 += (double)e3;
        }
      }

      tmpden[out] = g0;
      if (spin_count >= 2) tmpden[output_count + out] = g1;
      if (spin_count == 4) {
        tmpden[2U * output_count + out] = g2;
        tmpden[3U * output_count + out] = g3;
      }
    }
      }
    }
    else {
#pragma acc parallel loop gang vector_length(128)                                                  \
    present(pair_NO0[0:pair_count], pair_NO1[0:pair_count], pair_h_offset[0:pair_count],           \
            pair_nolg_offset[0:pair_count], pair_orbs0_offset[0:pair_count],                       \
            pair_orbs1_offset[0:pair_count], nolg_Nc[0:total_nolg], orbs0buf[0:total_orbs0],       \
            orbs1buf[0:total_orbs1], out_ptr[0:output_count+1], term_pt[0:term_count],             \
            pt_pair[0:term_count], dm[0:dm_count], tmpden[0:(size_t)spin_count*output_count])
    for (size_t out = 0; out < output_count; out++) {
      double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0;

#pragma acc loop seq
      for (uint32_t q = out_ptr[out]; q < out_ptr[out + 1]; q++) {
        uint32_t pt = term_pt[q];
        uint32_t pair = pt_pair[pt];
        int NO0 = pair_NO0[pair];
        int NO1 = pair_NO1[pair];
        size_t mat = (size_t)NO0 * (size_t)NO1;
        size_t base = pair_h_offset[pair];
        size_t o0 = pair_orbs0_offset[pair] + (size_t)nolg_Nc[pt] * (size_t)NO0;
        size_t o1 = pair_orbs1_offset[pair] + ((size_t)pt - pair_nolg_offset[pair]) * (size_t)NO1;
        double e0 = 0.0, e1 = 0.0, e2 = 0.0, e3 = 0.0;

#pragma acc loop seq
        for (int ii = 0; ii < NO0; ii++) {
          double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
#pragma acc loop seq
          for (int jj = 0; jj < NO1; jj++) {
            double phi1 = (double)orbs1buf[o1 + (size_t)jj];
            size_t ij = (size_t)ii * (size_t)NO1 + (size_t)jj;
            t0 += phi1 * dm[base + ij];
            if (spin_count >= 2) t1 += phi1 * dm[base + mat + ij];
            if (spin_count == 4) {
              t2 += phi1 * dm[base + 2U * mat + ij];
              t3 += phi1 * dm[base + 3U * mat + ij];
            }
          }
          {
            double phi0 = (double)orbs0buf[o0 + (size_t)ii];
            e0 += phi0 * t0;
            if (spin_count >= 2) e1 += phi0 * t1;
            if (spin_count == 4) {
              e2 += phi0 * t2;
              e3 += phi0 * t3;
            }
          }
        }
        g0 += e0;
        if (spin_count >= 2) g1 += e1;
        if (spin_count == 4) {
          g2 += e2;
          g3 += e3;
        }
      }

      tmpden[out] = g0;
      if (spin_count >= 2) tmpden[output_count + out] = g1;
      if (spin_count == 4) {
        tmpden[2U * output_count + out] = g2;
        tmpden[3U * output_count + out] = g3;
      }
    }
    }
    }
    /* copyout on an already-present mapping does not transfer its contents. */
    acc_update_self(tmpden, (size_t)spin_count * output_count * sizeof(double));
  }
  for (int i=0;i<15;i++) if (created[i]) SDG_acc_delete_if_present(map_host[i],map_bytes[i]);
  free(dmf);

  for (spin = 0; spin < spin_count; spin++) {
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
      size_t base = c->atom_out_base[Mc_AN];
      size_t n = (size_t)GridN_Atom[M2G[Mc_AN]];

      memcpy(Tmp_Den_Grid[spin][Mc_AN], tmpden + (size_t)spin * output_count + base,
             sizeof(double) * n);
    }
  }
  return 1;

sdg_map_failure:
  for (int i=0;i<15;i++) if (created[i]) SDG_acc_delete_if_present(map_host[i],map_bytes[i]);
  free(dmf);
  return 0;
}
