/**********************************************************************
  Set_Density_Grid_GPU.c:

  A single-owner GPU service for Set_Density_Grid.  The owner gathers
  the unique atom-centred orbital grids once per geometry, gathers only
  CDM on subsequent SCF iterations, constructs the density on the full
  regular grid, and scatters the contiguous B partitions to all ranks.
***********************************************************************/

#include "mpi.h"
#include "openmx_common.h"
#include "openmx_gpu_bilinear.h"
#include "openmx_gpu_workers.h"
#include "set_hip_default_device_from_local_rank.h"
#include <limits.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int openmx_gpu_is_apu(void);

_Static_assert(sizeof(uint32_t) == sizeof(unsigned int), "uint32_t must match MPI_UNSIGNED");
_Static_assert(sizeof(uint16_t) == sizeof(unsigned short), "uint16_t must match MPI_UNSIGNED_SHORT");

/* Persistent host-to-device associations backing the NVIDIA OpenACC cache
   model with HIP allocations visible to OpenMP target regions. */
static void *acc_copyin(void *h, size_t n)
{
  void *d=NULL; int dev=omp_get_default_device();
  if (hipMalloc(&d,n)!=hipSuccess) return NULL;
  if (hipMemcpy(d,h,n,hipMemcpyHostToDevice)!=hipSuccess ||
      omp_target_associate_ptr(h,d,n,0,dev)!=0){ hipFree(d); return NULL; }
  return d;
}
static void *acc_create(void *h, size_t n)
{
  void *d=NULL; int dev=omp_get_default_device();
  if (hipMalloc(&d,n)!=hipSuccess) return NULL;
  if (omp_target_associate_ptr(h,d,n,0,dev)!=0){ hipFree(d); return NULL; }
  return d;
}
static int acc_is_present(void *h,size_t n)
{ (void)n; return omp_target_is_present(h,omp_get_default_device()); }
static void acc_delete(void *h,size_t n)
{
  int dev=omp_get_default_device(); void *d=omp_get_mapped_ptr(h,dev); (void)n;
  if (d){ omp_target_disassociate_ptr(h,dev); hipFree(d); }
}
static void acc_update_device(void *h,size_t n)
{ void *d=omp_get_mapped_ptr(h,omp_get_default_device()); if(d) hipMemcpy(d,h,n,hipMemcpyHostToDevice); }
static void acc_update_self(void *h,size_t n)
{ void *d=omp_get_mapped_ptr(h,omp_get_default_device()); if(d) hipMemcpy(h,d,n,hipMemcpyDeviceToHost); }
#define acc_wait_all() hipDeviceSynchronize()
#define acc_clear_freelists() ((void)0)

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
  const char *control=getenv("OPENMX_DENSITY_GRID_GPU");
  if (!control) control=getenv("OPENMX_SETDENSITY_GPU");
  const char *gpu=getenv("OPENMX_GPU");
  int enabled=(!gpu || atoi(gpu)!=0) && scf_eigen_lib_flag==GPUSOLVER && (!control || !*control || atoi(control)!=0);
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

static int SDG_env_bool(const char *name, int fallback)
{
  const char *value = getenv(name);
  if (value == NULL || value[0] == '\0') return fallback;
  return atoi(value) != 0;
}

static size_t SDG_env_mib(const char *name, size_t fallback_mib)
{
  const char *value = getenv(name);
  char *endp = NULL;
  unsigned long long mib;

  if (value == NULL || value[0] == '\0') return fallback_mib * 1024ULL * 1024ULL;
  mib = strtoull(value, &endp, 10);
  if (endp == value || *endp != '\0') return fallback_mib * 1024ULL * 1024ULL;
  if (mib > (unsigned long long)SIZE_MAX / (1024ULL * 1024ULL)) return SIZE_MAX;
  return (size_t)mib * 1024ULL * 1024ULL;
}

/* The single-owner and per-rank target kernels are effective on discrete
   GPUs.  On an APU shared by many MPI ranks, the distributed host quadrature
   avoids target-runtime serialization and is substantially faster. */
static int SDG_gpu_default_enabled(void)
{
  return openmx_gpu_is_apu() ? 0 : 1;
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

static void SDG_acc_delete_if_present(void *host_ptr, size_t bytes)
{
  if (host_ptr != NULL && bytes != 0 && acc_is_present(host_ptr, bytes)) {
    acc_delete(host_ptr, bytes);
  }
}

static void SDG_delete_device(SetDensityGpuCache *cache)
{
  if (!cache->device_resident) return;

  /* acc_copyin can fail after only a prefix of these mappings was created.
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
     following Band/Cluster eigensolver uses hipMemGetInfo() to decide whether
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
  if (acc_copyin(cache->orbs, cache->orbital_count * sizeof(Type_Orbs_Grid)) == NULL ||
      acc_copyin(cache->atom_out_offset, (cache->atom_count + 1) * sizeof(uint32_t)) == NULL ||
      acc_copyin(cache->atom_orb_offset, (cache->atom_count + 1) * sizeof(uint32_t)) == NULL ||
      acc_copyin(cache->atom_pair_offset, (cache->atom_count + 1) * sizeof(uint32_t)) == NULL ||
      acc_copyin(cache->atom_no, cache->atom_count * sizeof(int)) == NULL ||
      acc_copyin(cache->pair_neighbor, cache->pair_count * sizeof(int)) == NULL ||
      acc_copyin(cache->pair_cdm_offset, cache->pair_count * sizeof(uint32_t)) == NULL ||
      acc_copyin(cache->out_atom, cache->output_count * sizeof(uint16_t)) == NULL ||
      acc_copyin(cache->out_ptr, (cache->output_count + 1) * sizeof(uint32_t)) == NULL ||
      acc_copyin(cache->terms, cache->term_count * sizeof(uint32_t)) == NULL ||
      acc_copyin(cache->grid_ptr, (cache->global_grid_count + 1) * sizeof(uint32_t)) == NULL ||
      acc_copyin(cache->grid_out, cache->output_count * sizeof(uint32_t)) == NULL ||
      acc_copyin(cache->cdm, cache->cdm_count * sizeof(double)) == NULL ||
      acc_create(cache->rho, cache->global_grid_count * (size_t)cache->spin_count * sizeof(double)) == NULL) {
    SDG_delete_device(cache);
    return 0;
  }
  return 1;
}

static void SDG_run_kernel(SetDensityGpuCache *cache)
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

#pragma omp target teams distribute parallel for map(present, alloc:orbs[0:orbital_count], atom_out_offset[0:atom_count+1], atom_orb_offset[0:atom_count+1], \
          atom_pair_offset[0:atom_count+1], atom_no[0:atom_count], pair_neighbor[0:pair_count], \
          pair_cdm_offset[0:pair_count], out_atom[0:output_count], out_ptr[0:output_count+1], \
          terms[0:term_count], grid_ptr[0:grid_count+1], grid_out[0:output_count], \
          cdm[0:cdm_count], rho[0:grid_count*spin_count])
  for (size_t gn = 0; gn < grid_count; gn++) {
    double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0;

    for (uint32_t q = grid_ptr[gn]; q < grid_ptr[gn + 1]; q++) {
      uint32_t out = grid_out[q];
      int a = (int)out_atom[out];
      uint32_t Nc0 = out - atom_out_offset[a];
      int NO0 = atom_no[a];
      size_t orb0 = (size_t)atom_orb_offset[a] + (size_t)Nc0 * (size_t)NO0;
      double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;

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

        for (int ii = 0; ii < NO0; ii++) {
          double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
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

/* Eigensolvers whose density matrix the device paths integrate.  Every
   solver hands Set_Density_Grid the same DM[spin][Mc_AN][h_AN][i][j] over
   the FNAN neighbours, so the quadrature itself is solver-independent.
   The distributed local mode (each rank its own atoms on Set_Hamiltonian's
   resident tables; the on-the-fly tiles of Set_Hamiltonian.c have no
   solver gate) takes the cluster and band solvers and DC (5), Krylov (8)
   and DC-LNO (11); the single-owner service, which builds the whole grid's
   orbital tables on one device, stays with the cluster and band solvers:
   at the O(N) solvers' sizes those tables fit neither the device nor
   comfortably the host (as in the NVIDIA version). */
static int SDG_solver_supported(int local_mode)
{
  if (Solver == 2 || Solver == 3) return 1;
  return local_mode && (Solver == 5 || Solver == 8 || Solver == 11);
}

int Set_Density_Grid_GPU_Service(int Cnt_kind, int Calc_CntOrbital_ON, double *****CDM,
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
    int requested = SDG_env_bool("OPENMX_DENSITY_GRID_GPU",
                                SDG_env_bool("OPENMX_SETDENSITY_GPU", SDG_gpu_default_enabled()));
    enabled = requested && scf_eigen_lib_flag == GPUSOLVER && SDG_solver_supported(0) &&
              Cnt_switch == 0 && (Cnt_kind == 0 || Cnt_kind == 1) &&
              (SpinP_switch == 0 || SpinP_switch == 1 || SpinP_switch == 3);
    if (enabled) {
      hipError_t status = hipGetDeviceCount(&device_count);
      if (status != hipSuccess || device_count <= 0 || omp_get_num_devices() <= 0 ||
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

  reserve_bytes = SDG_env_mib("OPENMX_DENSITY_GRID_GPU_RESERVE_MB", 256);
  if (myid == owner && !cache->device_resident) {
    hipGetDevice(&device);
    omp_set_default_device(device);
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess ||
        free_bytes < cache->device_bytes || reserve_bytes > free_bytes - cache->device_bytes) need_release = 1;
  }
  MPI_Bcast(&need_release, 1, MPI_INT, owner, mpi_comm_level1);

  if (need_release) {
    /* First return only disposable storage.  In particular, do not destroy
       the collinear Cluster solver context here: it owns the transformed
       overlap matrix, and dropping it makes the next SCF diagonalize the
       overlap again.  OpenACC freelists are per MPI process, so every rank
       returns its deleted blocks before the owner retries hipMemGetInfo(). */
    acc_wait_all();
    openmx_gemmul8ReleaseWorkspaces();
    acc_clear_freelists();
    MPI_Barrier(mpi_comm_level1);

    if (myid == owner) {
      need_release =
          (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess ||
           free_bytes < cache->device_bytes || reserve_bytes > free_bytes - cache->device_bytes);
    }
    MPI_Bcast(&need_release, 1, MPI_INT, owner, mpi_comm_level1);
  }

  if (need_release && Solver == 2 && SpinP_switch == 3) {
    /* The non-collinear Cluster path keeps its final dense eigenvectors for a
       later scatter.  Preserve them on the host, then return only their device
       copy and its OpenACC freelist allocation. */
    Cluster_DFT_NonCol_DemoteGpuSolverCachedEVec();
    acc_wait_all();
    acc_clear_freelists();
    MPI_Barrier(mpi_comm_level1);
  }

  if (myid == owner && !cache->device_resident) {
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess ||
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

  /* Small geometry tables are cheap to retain and otherwise get allocated
     and uploaded on every SCF step. Keep the large epochs transient so
     the following dense eigensolve can still claim the device. */
  persistent = SDG_env_bool("OPENMX_DENSITY_GRID_GPU_PERSISTENT",
      cache->device_bytes <= SDG_env_mib("OPENMX_DENSITY_GRID_GPU_CACHE_MB", 64));
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
   the device.  A rank whose Set_Hamiltonian matrix-elements tables (the
   central full-sphere orbitals, the FNAN-layout neighbour orbitals and
   the overlap index lists) are device-resident runs the CSR kernel
   below on them, so a single copy serves both routines; in the auto
   mode a rank without resident tables integrates with the on-the-fly
   orbital tiles of Set_Hamiltonian.c (Set_Density_Grid_OTF_Run) in the
   same step.  The CSR kernel reproduces the CPU loop's order: per
   output point the pair contributions are summed in (h_AN, Nog) order
   and each contribution is sum_i orbs0[i] * (sum_j orbs1[j] * CDM[i][j]),
   in FP64 or, where the grid integrals run in double-float
   (openmx_grid_precision_df), with the double-float kernel of
   set_hamiltonian_hip.cu (FP64 between the pairs, a gather either way).
   The result fills Tmp_Den_Grid and the ordinary A-to-B communication
   of Set_Density_Grid.c distributes it afterwards.

   The preparation is rank-local (SDG_local_prepare); the public
   Set_Density_Grid_GPU_Local_Prepare holds its only collective, which
   every rank enters when the eigensolver is on the GPU (and none
   otherwise), and Set_Density_Grid.c agrees on the result.  Empty
   partitions, cached, on-the-fly and failed ranks therefore cannot
   desynchronize the collectives.
***********************************************************************/

/* set_hamiltonian_hip.cu */
extern int openmx_grid_precision_df(void);
extern int Set_Density_Grid_Hip_LocalDensity(size_t output_count, int spin_count, int max_no, size_t dm_count,
                                             const uint32_t *d_out_ptr, const uint32_t *d_term_pt,
                                             const uint32_t *d_pt_pair, const int *d_no0, const int *d_no1,
                                             const size_t *d_hoff, const size_t *d_noff, const size_t *d_o0off,
                                             const size_t *d_o1off, const int *d_nc, const float *d_o0,
                                             const float *d_o1, double *d_dm, double *d_den);
/* Set_Hamiltonian.c: the on-the-fly density of a rank (rank-local) */
extern int Set_Density_Grid_OTF_Possible(int Cnt_kind);
extern int Set_Density_Grid_OTF_Prepare(int Cnt_kind, size_t *need_bytes, size_t *available_bytes);
extern int Set_Density_Grid_OTF_Run(int Cnt_kind, double *****CDM, double ***Tmp_Den_Grid);

typedef struct {
  int ready;
  int unavailable;
  int spin_count;
  int cnt_kind;
  int node_ranks;          /* ranks sharing this rank's device: the budget's denominator */
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
  int joins_validated;
  int max_no1;             /* the largest NO1 the terms reach (validated with the joins) */
  int device_id;
  uint32_t *d_out_ptr, *d_term_pt, *d_pt_pair;
  double *d_dm, *d_tmpden;
  SetHamiltonianMETables t;
  /* the on-the-fly mode of a rank without resident tables: no CSR, the
     orbitals evaluated on the device batch by batch (Set_Hamiltonian.c) */
  int otf;
} SDGLocalContext;

static SDGLocalContext SDG_local = {0};

static void SDG_local_delete_device(SDGLocalContext *c)
{
  if (c->d_out_ptr != NULL) hipFree(c->d_out_ptr);
  if (c->d_term_pt != NULL) hipFree(c->d_term_pt);
  if (c->d_pt_pair != NULL) hipFree(c->d_pt_pair);
  if (c->d_dm != NULL) hipFree(c->d_dm);
  if (c->d_tmpden != NULL) hipFree(c->d_tmpden);
  c->d_out_ptr = c->d_term_pt = c->d_pt_pair = NULL;
  c->d_dm = c->d_tmpden = NULL;
}

static int SDG_local_upload(SDGLocalContext *c)
{
  int device;
  size_t terms = c->term_count ? c->term_count : 1;
  size_t dm_count = c->dm_count ? c->dm_count : 1;
  if (hipGetDevice(&device) != hipSuccess) return 0;
  if (c->d_out_ptr != NULL && c->device_id != device) SDG_local_delete_device(c);
  if (c->d_out_ptr != NULL) return 1;
  c->device_id = device;
  if (hipMalloc((void**)&c->d_out_ptr, sizeof(uint32_t)*(c->output_count+1)) != hipSuccess ||
      hipMalloc((void**)&c->d_term_pt, sizeof(uint32_t)*terms) != hipSuccess ||
      hipMalloc((void**)&c->d_pt_pair, sizeof(uint32_t)*terms) != hipSuccess ||
      hipMalloc((void**)&c->d_dm, sizeof(double)*dm_count) != hipSuccess ||
      hipMalloc((void**)&c->d_tmpden, sizeof(double)*c->spin_count*c->output_count) != hipSuccess ||
      hipMemcpy(c->d_out_ptr, c->out_ptr, sizeof(uint32_t)*(c->output_count+1), hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(c->d_term_pt, c->term_pt, sizeof(uint32_t)*c->term_count, hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(c->d_pt_pair, c->pt_pair, sizeof(uint32_t)*c->term_count, hipMemcpyHostToDevice) != hipSuccess) {
    SDG_local_delete_device(c);
    return 0;
  }
  return 1;
}

static void SDG_local_free(void)
{
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

/* Yield only reproducible device buffers to a following dense eigensolve;
   geometry tables and their validated host layout remain available. */
void Set_Density_Grid_GPU_Release_Device_Caches(void)
{
  SDG_delete_device(&SDG_cache);
  SDG_local_delete_device(&SDG_local);
}

/* The end of an SCF cycle (DFT.c, before the forces): only the local
   mode's device buffers go -- the next cycle stages them again from the
   host CSR, which stays -- while the single-owner service keeps its
   small persistent tables (OPENMX_DENSITY_GRID_GPU_CACHE_MB), as in the
   NVIDIA version.  Rank-local; no HIP call when nothing is held (a CPU
   run). */
void Set_Density_Grid_GPU_EndCycle(void)
{
  SDG_local_delete_device(&SDG_local);
}

/* Rank-local: OTF, cached, empty and failed ranks do not enter it, so it
   holds no collective; a failure marks only this rank unavailable for
   the geometry. */
static int SDG_local_build(int Cnt_kind, int spin_count)
{
  SDGLocalContext *c = &SDG_local;
  uint32_t *degree = NULL;
  int ok = 1;
  int Mc_AN, p;
  size_t out, pos;

  {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 0 ||
        omp_get_num_devices() <= 0) ok = 0;
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
      const int n = GridN_Atom[M2G[Mc_AN]];
      c->atom_out_base[Mc_AN] = (uint32_t)base;
      if (n < 0 || (size_t)n > (size_t)UINT32_MAX - base) {
        ok = 0;
        break;
      }
      base += (size_t)n;
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

  if (!ok) {
    SDG_local_free();
    SDG_local.unavailable = 1;
    return 0;
  }

  c->ready = 1;
  return 1;
}

/* Device bytes this call would stage on top of what is already resident. */
static size_t SDG_local_call_bytes(const SDGLocalContext *c)
{
  /* Unlike the CUDA implementation, the HIP double-float kernel splits
     the existing device DM in place, so no precision image is added. */
  size_t bytes = c->d_out_ptr != NULL ? 0 : c->extra_bytes;

  if (!c->t.orbs0_resident && !SDG_add_bytes(&bytes, c->t.total_orbs0, sizeof(Type_Orbs_Grid))) return SIZE_MAX;
  if (!c->t.orbs1_resident && !SDG_add_bytes(&bytes, c->t.total_orbs1, sizeof(Type_Orbs_Grid))) return SIZE_MAX;
  if (!c->t.nolg_resident && !SDG_add_bytes(&bytes, c->t.total_nolg, sizeof(int))) return SIZE_MAX;
  if (!c->t.meta_resident) {
    if (!SDG_add_bytes(&bytes, (size_t)c->t.pair_count, 2 * sizeof(int)) ||
        !SDG_add_bytes(&bytes, (size_t)c->t.pair_count, 4 * sizeof(size_t))) return SIZE_MAX;
  }
  return bytes;
}

/* This rank's choice, without communication: 1 when it can integrate its
   atoms in the local mode (the CSR kernel, or the on-the-fly tiles:
   SDG_local.otf), 0 when it cannot.  node_ranks: the ranks sharing the
   device. */
static int SDG_local_prepare(int Cnt_kind, int Calc_CntOrbital_ON, int node_ranks)
{
  SDGLocalContext *c = &SDG_local;
  int spin_count = SpinP_switch + 1;
  /* Without orbital optimization, the final density (Cnt_kind 1) uses
     the same primitive orbitals and resident tables as the SCF density. */
  const int table_kind = (Cnt_switch == 0) ? 0 : Cnt_kind;
  int enabled, mode = 1, my_ready;
  size_t reserve_bytes, need;
  size_t free_bytes = 0, total_bytes = 0;

  (void)Calc_CntOrbital_ON;

  /* 0 = off; 1 = auto: the CSR kernel on a rank whose Set_Hamiltonian
     tables are device-resident, the on-the-fly tiles on a rank without
     them; any other value = the CSR kernel on every rank.  The CSR kernel
     needs the tables device-resident: unlike the NVIDIA version this one
     neither builds missing tables (Set_Hamiltonian_GetMatrixElementsTables
     only returns them -- on every rank at once the multi-GiB host build
     can exhaust the host memory) nor stages host-only ones, so with a
     value other than 0 and 1 a rank without resident tables declines and
     the group falls back (to the all-rank on-the-fly mode, the
     single-owner service or the CPU loop of Set_Density_Grid.c). */
  {
    const char *value = getenv("OPENMX_DENSITY_GRID_GPU_LOCAL");
    if (value != NULL && value[0] != '\0') mode = atoi(value);
  }

  enabled = (mode != 0) &&
            SDG_env_bool("OPENMX_DENSITY_GRID_GPU",
                         SDG_env_bool("OPENMX_SETDENSITY_GPU", SDG_gpu_default_enabled())) &&
            scf_eigen_lib_flag == GPUSOLVER && SDG_solver_supported(1) &&
            Cnt_switch == 0 && (Cnt_kind == 0 || Cnt_kind == 1) &&
            (SpinP_switch == 0 || SpinP_switch == 1 || SpinP_switch == 3) &&
            (Matomnum == 0 || gpu_rank_device_usable());
  if (!enabled) return 0;
  /* No quadrature work, but this rank must not veto the other ranks. */
  if (Matomnum == 0) return 1;

  /* Ride on the tables Set_Hamiltonian has already built: probe for
     them without building (Set_Hamiltonian_MatrixElementsTables_Ready),
     then for their residency. */
  my_ready = Set_Hamiltonian_MatrixElementsTables_Ready(table_kind);
  if (my_ready) {
    SetHamiltonianMETables probe;
    if (!Set_Hamiltonian_GetMatrixElementsTables(table_kind, &probe) ||
        !(probe.orbs0_resident && probe.orbs1_resident && probe.nolg_resident && probe.meta_resident))
      my_ready = 0;
  }
  if (!my_ready) {
    /* the on-the-fly tiles (auto mode only): this rank's Set_Hamiltonian
       evaluated its orbitals the same way, and Set_Density_Grid_OTF_Possible
       keeps its gates (the matrix-elements GPU path, at most
       OPENMX_BAND_GPU_MAX_DEVICE_RANKS ranks per device,
       OPENMX_DENSITY_GRID_ONTHEFLY) */
    if (mode != 1 || !Set_Density_Grid_OTF_Possible(table_kind)) return 0;
    if (!c->otf) {
      SDG_local_free();
      c->otf = 1;
    }
    c->spin_count = spin_count;
    c->cnt_kind = table_kind;
    return 1;
  }

  if (c->otf) SDG_local_free(); /* back from the on-the-fly mode to the CSR */
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

  /* the per-call staging of the multi-GiB orbital tables is not done here:
     engage only when the shared tables live on the device */
  if (!(c->t.orbs0_resident && c->t.orbs1_resident && c->t.nolg_resident && c->t.meta_resident)) {
    return 0;
  }

  c->node_ranks = node_ranks > 0 ? node_ranks : 1;
  reserve_bytes = SDG_env_mib("OPENMX_DENSITY_GRID_GPU_RESERVE_MB", 256);
  if (c->d_out_ptr != NULL) {
    int device;
    if (hipGetDevice(&device) != hipSuccess) return 0;
    if (device != c->device_id) SDG_local_delete_device(c);
  }
  if (OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) return 0;
  need = SDG_local_call_bytes(c);
  if (free_bytes < reserve_bytes) return 0;
  if ((free_bytes - reserve_bytes) / (size_t)c->node_ranks < need) return 0;
  return 1;
}

/* Every rank enters this when the eigensolver runs on the GPU -- the
   ranks that take the CSR kernel, the on-the-fly ranks, empty partitions
   and the ranks whose preparation failed alike -- and none enters a
   collective otherwise: Input and DFT_GPU_DeviceInit set
   scf_eigen_lib_flag identically on every rank (a device failure on any
   rank demotes all of them).  The budget's denominator is the number of
   ranks sharing the device (openmx_gpu_ranks_per_device_noncollective,
   no communication), not of the node as in the NVIDIA version.  The
   CSR ranks' transient device need is published for compatibility
   (OpenMX_GpuPhaseNeed_Register; this version's Set_Hamiltonian demotes
   its resident tables on the eigensolvers' memory failures instead of
   reading it). */
static struct {
  int atom_ranks, declined, workspace_declined;
} SDG_census = {0};

int Set_Density_Grid_GPU_Local_Prepare(int Cnt_kind, int Calc_CntOrbital_ON)
{
  unsigned long long counts[4], totals[4];
  size_t need = 0, available = 0;
  int ok, workspace_declined = 0;

  if (scf_eigen_lib_flag != GPUSOLVER) {
    OpenMX_GpuPhaseNeed_Register("density_grid_local", 0);
    memset(&SDG_census, 0, sizeof(SDG_census));
    return 0;
  }

  ok = SDG_local_prepare(Cnt_kind, Calc_CntOrbital_ON, openmx_gpu_ranks_per_device_noncollective());
  if (ok && Matomnum > 0 && SDG_local.otf) {
    ok = Set_Density_Grid_OTF_Prepare(SDG_local.cnt_kind, &need, &available);
    workspace_declined = !ok && need > available;
    if (workspace_declined && 3 <= level_stdout) {
      int myid;
      MPI_Comm_rank(mpi_comm_level1, &myid);
      fprintf(stderr, "<Set_Density_Grid> rank %d: %.1f MiB of on-the-fly workspace needed, "
              "%.1f MiB available; declining distributed GPU quadrature\n",
              myid, (double)need / 1048576.0, (double)available / 1048576.0);
      fflush(stderr);
    }
  }

  /* The census rides along with the existing phase-need reduction.  All
     ranks participate regardless of their local path or stdout level. */
  counts[0] = (SDG_local.ready && !SDG_local.otf) ? (unsigned long long)SDG_local.extra_bytes : 0ULL;
  counts[1] = Matomnum > 0 ? 1ULL : 0ULL;
  counts[2] = Matomnum > 0 && !ok ? 1ULL : 0ULL;
  counts[3] = (unsigned long long)workspace_declined;
  MPI_Allreduce(counts, totals, 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM, mpi_comm_level1);
  OpenMX_GpuPhaseNeed_Register("density_grid_local", (size_t)totals[0]);
  SDG_census.atom_ranks = (int)totals[1];
  SDG_census.declined = (int)totals[2];
  SDG_census.workspace_declined = (int)totals[3];
  return ok;
}

/* No collectives here.  path 1 is the single-owner service; path 2 lets
   capable ranks use the GPU while declined ranks use the original CPU loop.
   failed counts runtime GPU failures on ranks with atoms. */
void Set_Density_Grid_GPU_Local_Report(int path, int failed)
{
  static int last_path = -1, last_cpu = -1, last_workspace = -1, last_failed = -1;
  const int workspace = SDG_census.workspace_declined;
  const int cpu = (path == 2 ? SDG_census.declined : 0) + failed;
  int myid;

  if (workspace == 0 && cpu == 0) {
    last_path = -1;
    return;
  }
  if (level_stdout < 2) return;
  if (path == last_path && cpu == last_cpu && workspace == last_workspace && failed == last_failed) return;
  last_path = path;
  last_cpu = cpu;
  last_workspace = workspace;
  last_failed = failed;
  MPI_Comm_rank(mpi_comm_level1, &myid);
  if (myid != Host_ID) return;

  if (path == 1) {
    printf("<Set_Density_Grid> %d of %d rank(s) with atoms lack on-the-fly workspace; "
           "the one-rank GPU service integrates the density\n", workspace, SDG_census.atom_ranks);
  }
  else {
    printf("<Set_Density_Grid> CPU loop on %d of %d rank(s) with atoms "
           "(%d declined local GPU preparation, %d lacked on-the-fly workspace, %d failed on the device)\n",
           cpu, SDG_census.atom_ranks, path == 2 ? SDG_census.declined : 0, workspace, failed);
  }
  fflush(stdout);
}

/* 1 when the last local preparation of this rank chose the on-the-fly
   tiles, 0 for the CSR kernel (for the stage profile of Set_Density_Grid.c) */
int Set_Density_Grid_GPU_Local_OnTheFly(void)
{
  return SDG_local.otf;
}

int Set_Density_Grid_GPU_Local_Run(double *****CDM, double ***Tmp_Den_Grid)
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
  int Mc_AN, spin;

  if (Matomnum == 0) return 1;
  /* the on-the-fly tiles fill Tmp_Den_Grid themselves (and leave it as
     they found it when they fail) */
  if (c->otf) return Set_Density_Grid_OTF_Run(c->cnt_kind, CDM, Tmp_Den_Grid);

  if (!c->ready || output_count == 0) return 0;
  if (!SDG_pack_local_cdm(CDM, dm, dm_count, spin_count)) return 0;

  /* Validate once per geometry; the shared-table invalidation resets this
     cache together with its CSR. Rewalking every join on the CPU in every
     SCF step can cost more than the GPU contraction itself. */
  if (!c->joins_validated) {
    int max_no1 = 0;
    for (size_t out=0; out<output_count; out++) {
      if (out_ptr[out]>out_ptr[out+1] || out_ptr[out+1]>term_count) return 0;
      for (uint32_t q=out_ptr[out]; q<out_ptr[out+1]; q++) {
        uint32_t pt=term_pt[q], pair;
        size_t o0,o1,mat;
        if (pt>=total_nolg) return 0;
        pair=pt_pair[pt];
        if (pair>=(uint32_t)pair_count) return 0;
        mat=(size_t)pair_NO0[pair]*(size_t)pair_NO1[pair];
        o0=pair_orbs0_offset[pair]+(size_t)c->t.nolg_Nc[pt]*(size_t)pair_NO0[pair];
        o1=pair_orbs1_offset[pair]+((size_t)pt-pair_nolg_offset[pair])*(size_t)pair_NO1[pair];
        if (pair_h_offset[pair]+(size_t)spin_count*mat>dm_count ||
            o0+(size_t)pair_NO0[pair]>total_orbs0 ||
            o1+(size_t)pair_NO1[pair]>total_orbs1) return 0;
        if (max_no1<pair_NO1[pair]) max_no1=pair_NO1[pair];
      }
    }
    c->max_no1 = max_no1;
    c->joins_validated = 1;
  }

  {
    double *dm_host=dm, *tmpden_host=tmpden;
    int dev=omp_get_default_device();
    int keep_resident = 0;
    int status = 1;   /* 0: the double-float kernel ran, 1: the FP64 kernel runs, 2: failure */
    size_t free_bytes = 0, total_bytes = 0;
    size_t cap = SDG_env_mib("OPENMX_DENSITY_GRID_GPU_LOCAL_CACHE_MB", 32);

    if (c->extra_bytes <= cap && cap != 0) {
      if (c->d_out_ptr != NULL) keep_resident = 1;
      else if (OpenMX_GpuMemGetInfo(&free_bytes, &total_bytes) == hipSuccess)
        keep_resident = c->extra_bytes <= free_bytes / 32U / (size_t)c->node_ranks;
    }

    pair_NO0=(const int*)omp_get_mapped_ptr(pair_NO0,dev);
    pair_NO1=(const int*)omp_get_mapped_ptr(pair_NO1,dev);
    pair_h_offset=(const size_t*)omp_get_mapped_ptr(pair_h_offset,dev);
    pair_nolg_offset=(const size_t*)omp_get_mapped_ptr(pair_nolg_offset,dev);
    pair_orbs0_offset=(const size_t*)omp_get_mapped_ptr(pair_orbs0_offset,dev);
    pair_orbs1_offset=(const size_t*)omp_get_mapped_ptr(pair_orbs1_offset,dev);
    nolg_Nc=(const int*)omp_get_mapped_ptr(nolg_Nc,dev);
    orbs0buf=(const Type_Orbs_Grid*)omp_get_mapped_ptr(orbs0buf,dev);
    orbs1buf=(const Type_Orbs_Grid*)omp_get_mapped_ptr(orbs1buf,dev);
    if (!pair_NO0 || !pair_NO1 || !pair_h_offset || !pair_nolg_offset ||
        !pair_orbs0_offset || !pair_orbs1_offset || !nolg_Nc || !orbs0buf || !orbs1buf) return 0;

    if (!SDG_local_upload(c)) return 0;
    out_ptr = c->d_out_ptr;
    term_pt = c->d_term_pt;
    pt_pair = c->d_pt_pair;
    dm = c->d_dm;
    tmpden = c->d_tmpden;
    if (hipMemcpy(dm, dm_host, sizeof(double)*dm_count, hipMemcpyHostToDevice) != hipSuccess) {
      SDG_local_delete_device(c);
      return 0;
    }

    /* the double-float kernel splits the device copy of the density
       matrix in place (the next call uploads it again) */
    if (sizeof(Type_Orbs_Grid) == sizeof(float) && openmx_grid_precision_df()) {
      status = Set_Density_Grid_Hip_LocalDensity(output_count, spin_count, c->max_no1, dm_count,
                                                 out_ptr, term_pt, pt_pair, pair_NO0, pair_NO1,
                                                 pair_h_offset, pair_nolg_offset, pair_orbs0_offset,
                                                 pair_orbs1_offset, nolg_Nc, (const float *)orbs0buf,
                                                 (const float *)orbs1buf, dm, tmpden);
      if (status == 2) {
        SDG_local_delete_device(c);
        return 0;
      }
    }

    if (status == 1) {
#pragma omp target teams distribute parallel for is_device_ptr(pair_NO0,pair_NO1,pair_h_offset,pair_nolg_offset,pair_orbs0_offset,pair_orbs1_offset,nolg_Nc,orbs0buf,orbs1buf,out_ptr,term_pt,pt_pair,dm,tmpden)
      for (size_t out = 0; out < output_count; out++) {
        double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0;

        for (uint32_t q = out_ptr[out]; q < out_ptr[out + 1]; q++) {
          uint32_t pt = term_pt[q];
          uint32_t pair = pt_pair[pt];
          int NO0 = pair_NO0[pair];
          int NO1 = pair_NO1[pair];
          size_t mat = (size_t)NO0 * (size_t)NO1;
          size_t base = pair_h_offset[pair];
          /* orbs0: the central atom's sphere, selected through nolg_Nc */
          size_t o0 = pair_orbs0_offset[pair] + (size_t)nolg_Nc[pt] * (size_t)NO0;
          size_t o1 = pair_orbs1_offset[pair] + ((size_t)pt - pair_nolg_offset[pair]) * (size_t)NO1;
          double e0 = 0.0, e1 = 0.0, e2 = 0.0, e3 = 0.0;

          for (int ii = 0; ii < NO0; ii++) {
            double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
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

    if (hipMemcpy(tmpden_host,tmpden,sizeof(double)*(size_t)spin_count*output_count,hipMemcpyDeviceToHost) != hipSuccess) {
      SDG_local_delete_device(c);
      return 0;
    }
    if (!keep_resident) SDG_local_delete_device(c);
    tmpden=tmpden_host;
  }

  for (spin = 0; spin < spin_count; spin++) {
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++) {
      size_t base = c->atom_out_base[Mc_AN];
      size_t n = (size_t)GridN_Atom[M2G[Mc_AN]];

      memcpy(Tmp_Den_Grid[spin][Mc_AN], tmpden + (size_t)spin * output_count + base,
             sizeof(double) * n);
    }
  }
  return 1;
}
