/**********************************************************************
  Exact_Exchange.c:

     Exact exchange of the hybrid functionals in the SCF of the main
     program: the layer between the solvers and modules/exx.

     See docs/EXX_IMPLEMENTATION_GUIDE.md, sections 6 and 7.4, for the
     algorithm, and openmx_exact_exchange.h for the interface. The
     functions of this file take every input as an argument and read no
     global variable.
***********************************************************************/

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "openmx_core_native_resources.h"
#include "openmx_core_radial_views.h"
#include "openmx_exact_exchange.h"
#include "openmx_exx.h"
#include "lapack_prototypes.h"

/* Persistent metadata belongs to the explicit calculation owner. */
struct OpenMXExactExchangeState {
  OpenMXNativeResources *resources; /* borrowed from this calculation */
  int initialized;
  OpenMXExactExchangeInput input;   /* values of the input keywords */
  OpenMXExactExchangeStructure structure; /* arrays of the main program, borrowed */
  OpenMXExxState *module;           /* tables of modules/exx, released by the reset */
  int prepared;                     /* the module holds the present structure */
  MPI_Comm comm;                    /* communicator of the last preparation or collection */
  int orbitals;                     /* N of the present structure */
  int allocated_channels;           /* spin channels the matrices below have storage for */
  int spin_channels;                /* spin channels of the last collection: 0, 1 or 2 */
  size_t elements;                  /* N*N of the matrices below */
  double *density[2];               /* D of every spin channel, N x N */
  double *exchange[2];              /* K of the last build */
  double *previous[2];              /* K of the build before */
  int collecting;                   /* the solver hands vectors and occupations over */
  int exchange_ready;               /* K exists: it is added to the Hamiltonian */
  int outer_iterations;             /* builds of K for the present structure */
  int outer_converged;              /* the last build met scf.exx.outer.criterion */
  double energy;                    /* E_x of the last build, Hartree */
  double change;                    /* delta of the last build, Hartree */
  double residual_reference;        /* residual norm of the first step with the K of the last build; negative: not seen yet */
  double time_prepare,time_collect,time_build,time_add; /* seconds on this process */
};

static void ExactExchange_Reset(OpenMXNativeResources *owner, void *metadata)
{
  OpenMXExactExchangeState *state=(OpenMXExactExchangeState*)metadata;
  (void)owner;
  /* The matrices are allocations of the owner, which releases them. */
  if (state->module!=NULL) openmx_exx_destroy(state->module);
  state->module = NULL;
}

OpenMXExactExchangeState *Exact_Exchange_Bind_Resources(OpenMXCoreContext *core)
{
  OpenMXNativeResources *owner=NULL;
  void *metadata=NULL;
  char error[256];
  if (!openmx_core_native_resources_state(core,"Exact_Exchange",sizeof(OpenMXExactExchangeState),
                                          ExactExchange_Reset,&owner,&metadata,error,sizeof(error))) {
    fprintf(stderr,"OpenMX exact exchange state: %s\n",error);
    abort();
  }
  OpenMXExactExchangeState *state=(OpenMXExactExchangeState*)metadata;
  if (!state->initialized) {
    state->resources = owner;
    state->initialized = 1;
  }
  return state;
}

void Exact_Exchange_Configure(OpenMXExactExchangeState *state, const OpenMXExactExchangeInput *input)
{
  state->input = *input;
}

void Exact_Exchange_Bind_Structure(OpenMXExactExchangeState *state, const OpenMXExactExchangeStructure *structure)
{
  state->structure = *structure;
}

int Exact_Exchange_Enabled(const OpenMXExactExchangeState *state)
{
  return state!=NULL && state->input.functional!=OPENMX_EXACT_EXCHANGE_NONE;
}

int Exact_Exchange_Active(const OpenMXExactExchangeState *state)
{
  return Exact_Exchange_Enabled(state) && state->exchange_ready;
}

double Exact_Exchange_Semilocal_Alpha(const OpenMXExactExchangeState *state)
{
  return Exact_Exchange_Active(state) ? state->input.alpha : 0.0;
}

double Exact_Exchange_Semilocal_Omega(const OpenMXExactExchangeState *state)
{
  return Exact_Exchange_Enabled(state) ? state->input.omega : 0.0;
}

int Exact_Exchange_Outer_Iterations(const OpenMXExactExchangeState *state)
{
  return state->outer_iterations;
}

int Exact_Exchange_Outer_Converged(const OpenMXExactExchangeState *state)
{
  return state->outer_converged;
}

double Exact_Exchange_Energy(const OpenMXExactExchangeState *state)
{
  return state->energy;
}

double Exact_Exchange_Change(const OpenMXExactExchangeState *state)
{
  return state->change;
}

void Exact_Exchange_Times(const OpenMXExactExchangeState *state, double times[4])
{
  times[0] = state->time_prepare;
  times[1] = state->time_collect;
  times[2] = state->time_build;
  times[3] = state->time_add;
}

/* Ends the run of every process: the state of the SCF cannot be continued. */
static void ExactExchange_Abort(MPI_Comm comm, const char *where, const char *message)
{
  int rank=0;
  MPI_Comm_rank(comm,&rank);
  fprintf(stderr,"Exact exchange (%s), process %d: %s\n",where,rank,message);
  fflush(stderr);
  MPI_Abort(comm,1);
  abort();
}

static void *ExactExchange_Allocate(OpenMXExactExchangeState *state, size_t count, size_t bytes,
                                    MPI_Comm comm, const char *where)
{
  void *pointer=openmx_core_native_malloc(state->resources,count,bytes,1);
  if (pointer==NULL) ExactExchange_Abort(comm,where,"out of memory");
  return pointer;
}

/* The same matrix on every process: the sum is formed once, on process 0,
   and sent back. Decisions taken from it then agree on all processes. One
   call of MPI moves at most 2^27 elements, which is within int. */
static void ExactExchange_Sum(double *values, size_t count, MPI_Comm comm)
{
  const size_t chunk=(size_t)1<<27;
  int rank=0;
  MPI_Comm_rank(comm,&rank);
  for (size_t first=0; first<count; first+=chunk) {
    const int part=(int)(count-first<chunk ? count-first : chunk);
    if (rank==0) MPI_Reduce(MPI_IN_PLACE,values+first,part,MPI_DOUBLE,MPI_SUM,0,comm);
    else         MPI_Reduce(values+first,NULL,part,MPI_DOUBLE,MPI_SUM,0,comm);
    MPI_Bcast(values+first,part,MPI_DOUBLE,0,comm);
  }
}

/* (M + M^T)/2 in place. */
static void ExactExchange_Symmetrize(double *matrix, int n)
{
  for (int i=0; i<n; i++) {
    for (int j=i+1; j<n; j++) {
      const double value=0.5*(matrix[(size_t)i*n+j]+matrix[(size_t)j*n+i]);
      matrix[(size_t)i*n+j] = value;
      matrix[(size_t)j*n+i] = value;
    }
  }
}

/* The matrix products of modules/exx go to the BLAS of the main program. */
static int ExactExchange_Dgemm(void *context, char transa, char transb, int m, int n, int k, double alpha,
                               const double *a, int lda, const double *b, int ldb, double beta, double *c,
                               int ldc)
{
  (void)context;
  F77_NAME(dgemm,DGEMM)(&transa,&transb,&m,&n,&k,&alpha,(double*)a,&lda,(double*)b,&ldb,&beta,c,&ldc);
  return 0;
}

/* Storage of the density and exchange matrices of spin_channels channels. */
static void ExactExchange_Reserve(OpenMXExactExchangeState *state, int spin_channels, MPI_Comm comm)
{
  const size_t elements=(size_t)state->orbitals*(size_t)state->orbitals;
  if (state->elements!=elements) {
    for (int spin=0; spin<state->allocated_channels; spin++) {
      openmx_core_native_free(state->resources,state->density[spin]);
      openmx_core_native_free(state->resources,state->exchange[spin]);
      openmx_core_native_free(state->resources,state->previous[spin]);
      state->density[spin] = NULL;
      state->exchange[spin] = NULL;
      state->previous[spin] = NULL;
    }
    state->allocated_channels = 0;
    state->elements = elements;
  }
  for (int spin=state->allocated_channels; spin<spin_channels; spin++) {
    state->density[spin] = (double*)ExactExchange_Allocate(state,elements,sizeof(double),comm,"density matrix");
    state->exchange[spin] = (double*)ExactExchange_Allocate(state,elements,sizeof(double),comm,"exchange matrix");
    state->previous[spin] = (double*)ExactExchange_Allocate(state,elements,sizeof(double),comm,"exchange matrix");
  }
  if (state->allocated_channels<spin_channels) state->allocated_channels = spin_channels;
}

void Exact_Exchange_Prepare(OpenMXExactExchangeState *state, const OpenMXAtomicViews *atomic,
    MPI_Comm comm, int verbosity)
{
  if (!Exact_Exchange_Enabled(state)) return;

  const int atom_count=state->structure.atom_count;
  const int *species_of_atom=state->structure.species_of_atom;
  double *const *positions=state->structure.positions;
  double (*lattice)[4]=state->structure.lattice;
  if (atom_count<1 || species_of_atom==NULL || positions==NULL || lattice==NULL)
    ExactExchange_Abort(comm,"preparation","the structure of the calculation is not bound");

  const double start=MPI_Wtime();
  const int species_count=(int)atomic->species.species;
  int rank=0;
  MPI_Comm_rank(comm,&rank);

  /* The exchange matrix of an earlier structure is not carried over. */
  state->comm = comm;
  state->prepared = 0;
  state->spin_channels = 0;
  state->collecting = 0;
  state->exchange_ready = 0;
  state->outer_iterations = 0;
  state->outer_converged = 0;
  state->residual_reference = -1.0;
  state->energy = 0.0;
  state->change = 0.0;

  if (state->module==NULL) {
    state->module = openmx_exx_create();
    if (state->module==NULL) ExactExchange_Abort(comm,"preparation","out of memory");
  }

  /* The PAOs of every species: the radial functions in the order of the
     solvers (by angular momentum, then by multiplicity) on the mesh of the
     PAO file. */
  OpenMXExxSpecies *species=(OpenMXExxSpecies*)ExactExchange_Allocate(state,(size_t)species_count,
                                                                      sizeof(OpenMXExxSpecies),comm,"preparation");
  int **angular=(int**)ExactExchange_Allocate(state,(size_t)species_count,sizeof(int*),comm,"preparation");
  double **values=(double**)ExactExchange_Allocate(state,(size_t)species_count,sizeof(double*),comm,"preparation");
  for (int s=0; s<species_count; s++) {
    const int mesh=atomic->species.pao_mesh[s];
    const int lmax=atomic->species.basis_lmax[s];
    int shells=0;
    for (int l=0; l<=lmax; l++) shells += atomic->species.basis_rows[s][l];
    if (mesh<2 || shells<1) ExactExchange_Abort(comm,"preparation","a species has no pseudo-atomic orbitals");
    angular[s] = (int*)ExactExchange_Allocate(state,(size_t)shells,sizeof(int),comm,"preparation");
    values[s] = (double*)ExactExchange_Allocate(state,(size_t)shells*(size_t)mesh,sizeof(double),comm,"preparation");
    int shell=0;
    for (int l=0; l<=lmax; l++) {
      for (int multiplicity=0; multiplicity<atomic->species.basis_rows[s][l]; multiplicity++) {
        angular[s][shell] = l;
        memcpy(values[s]+(size_t)shell*(size_t)mesh,
               openmx_radial_row4(&atomic->orbitals,(size_t)s,(size_t)l,(size_t)multiplicity),
               sizeof(double)*(size_t)mesh);
        shell++;
      }
    }
    species[s].cutoff = atomic->species.cutoff[s];
    species[s].mesh_size = mesh;
    species[s].mesh = openmx_radial_row2(&atomic->pao_radii,(size_t)s);
    species[s].shell_count = shells;
    species[s].angular = angular[s];
    species[s].values = values[s];
  }

  double *coordinates=(double*)ExactExchange_Allocate(state,3*(size_t)atom_count,sizeof(double),comm,"preparation");
  int *kinds=(int*)ExactExchange_Allocate(state,(size_t)atom_count,sizeof(int),comm,"preparation");
  for (int a=0; a<atom_count; a++) {
    for (int x=0; x<3; x++) coordinates[3*a+x] = positions[a+1][x+1];
    kinds[a] = species_of_atom[a+1];
  }

  OpenMXExxGeometry geometry;
  for (int i=0; i<3; i++) {
    for (int x=0; x<3; x++) geometry.lattice[i][x] = lattice[i+1][x+1];
  }
  geometry.atom_count = atom_count;
  geometry.positions = coordinates;
  geometry.species = kinds;
  geometry.species_count = species_count;
  geometry.species_data = species;

  OpenMXExxOptions options;
  openmx_exx_default_options(&options);
  options.kernel = OPENMX_EXX_KERNEL_SHORT_RANGE;
  options.kernel_parameter = state->input.omega;
  options.abf_threshold = state->input.abf_threshold;
  options.v_cutoff = state->input.v_cutoff;
  options.radial_intervals = state->input.radial_intervals;
  options.reciprocal_maximum = state->input.reciprocal_maximum;
  options.reciprocal_intervals = state->input.radial_intervals;

  char error[512];
  if (openmx_exx_prepare(state->module,&options,&geometry,error,sizeof(error))!=0)
    ExactExchange_Abort(comm,"preparation",error);

  openmx_core_native_free(state->resources,kinds);
  openmx_core_native_free(state->resources,coordinates);
  for (int s=0; s<species_count; s++) {
    openmx_core_native_free(state->resources,values[s]);
    openmx_core_native_free(state->resources,angular[s]);
  }
  openmx_core_native_free(state->resources,values);
  openmx_core_native_free(state->resources,angular);
  openmx_core_native_free(state->resources,species);

  /* The solvers count the contracted orbitals; the two numbers agree as long
     as the orbital optimization is not combined with a hybrid functional. */
  int orbitals=0;
  for (int a=0; a<atom_count; a++) orbitals += atomic->species.contracted_orbitals[species_of_atom[a+1]];
  if (openmx_exx_orbital_count(state->module)!=orbitals || openmx_exx_atom_count(state->module)!=atom_count)
    ExactExchange_Abort(comm,"preparation","the orbitals of the structure differ from those of the solver");
  state->orbitals = orbitals;
  state->prepared = 1;
  state->time_prepare += MPI_Wtime()-start;

  if (rank==0 && 0<verbosity) {
    long abfs=0,c_sites=0,v_sites=0;
    OpenMXExxReport report;
    for (int a=0; a<atom_count; a++) {
      abfs += openmx_exx_atom_abfs(state->module,a);
      c_sites += openmx_exx_c_neighbours(state->module,a);
      v_sites += openmx_exx_v_neighbours(state->module,a);
    }
    printf("<EXX>  %d atoms, %d orbitals, %ld ABFs, %ld C sites, %ld V sites (%.2f s)\n",
           atom_count,orbitals,abfs,c_sites,v_sites,MPI_Wtime()-start);
    if (2<=verbosity && openmx_exx_report(state->module,&report)==0) {
      printf("<EXX>  local RI: %ld systems, %ld by eigen-decomposition (%ld vectors left out), smallest pivot %.3e\n",
             report.systems,report.eigen_decompositions,report.dropped,report.smallest_pivot);
    }
    fflush(stdout);
  }
}

void Exact_Exchange_Collect_Begin(OpenMXExactExchangeState *state, int spin_channels, MPI_Comm comm)
{
  if (!Exact_Exchange_Enabled(state)) return;
  if (!state->prepared) ExactExchange_Abort(comm,"density matrix","no structure is prepared");
  if (spin_channels<1 || 2<spin_channels) ExactExchange_Abort(comm,"density matrix","unsupported number of spin channels");
  /* The exchange matrices of a collinear calculation keep their number of
     spin channels during the SCF of a structure. */
  if (state->exchange_ready && state->spin_channels!=spin_channels)
    ExactExchange_Abort(comm,"density matrix","the number of spin channels changed");
  ExactExchange_Reserve(state,spin_channels,comm);
  for (int spin=0; spin<spin_channels; spin++)
    memset(state->density[spin],0,sizeof(double)*state->elements);
  state->comm = comm;
  state->spin_channels = spin_channels;
  state->collecting = 1;
}

void Exact_Exchange_Collect_Gamma(OpenMXExactExchangeState *state, int spin, int basis_count,
    int state_count, const double *occupation, const double *vectors)
{
  if (state==NULL || !state->collecting) return;

  const double start=MPI_Wtime();
  const int n=state->orbitals;
  if (basis_count!=n || spin<0 || state->spin_channels<=spin)
    ExactExchange_Abort(state->comm,"density matrix","the vectors of the solver have an unexpected size or spin");

  /* The states beyond the last occupied one do not contribute. */
  int occupied=0;
  for (int s=0; s<state_count; s++) if (occupation[s]!=0.0) occupied = s+1;
  if (0<occupied) {
    /* vectors is the matrix C of occupied x N elements with the leading
       dimension state_count (states in the rows). D += C^T (F C). */
    double *scaled=(double*)openmx_core_native_malloc(state->resources,(size_t)occupied*(size_t)n,sizeof(double),0);
    if (scaled==NULL) ExactExchange_Abort(state->comm,"density matrix","out of memory");
    for (int basis=0; basis<n; basis++) {
      const double *column=vectors+(size_t)basis*(size_t)state_count;
      double *target=scaled+(size_t)basis*(size_t)occupied;
      for (int s=0; s<occupied; s++) target[s] = occupation[s]*column[s];
    }
    char transposed='T',plain='N';
    int rows=n,states=occupied,stride=state_count;
    double one=1.0;
    F77_NAME(dgemm,DGEMM)(&transposed,&plain,&rows,&rows,&states,&one,(double*)vectors,&stride,
                          scaled,&states,&one,state->density[spin],&rows);
    openmx_core_native_free(state->resources,scaled);
  }
  state->time_collect += MPI_Wtime()-start;
}

void Exact_Exchange_Collect_End(OpenMXExactExchangeState *state, MPI_Comm comm)
{
  if (state==NULL || !state->collecting) return;

  const double start=MPI_Wtime();
  for (int spin=0; spin<state->spin_channels; spin++) {
    ExactExchange_Sum(state->density[spin],state->elements,comm);
    ExactExchange_Symmetrize(state->density[spin],state->orbitals);
  }
  state->collecting = 0;
  state->time_collect += MPI_Wtime()-start;
}

double Exact_Exchange_Density_Deviation(const OpenMXExactExchangeState *state, int spin_channels,
    int local_atoms, const int *local_to_global, const int *neighbour_count, int *const *neighbour_atom,
    const int *species_of_atom, const int *orbitals_of_species, const int *first_orbital,
    double *****density, MPI_Comm comm)
{
  const size_t n=(size_t)state->orbitals;
  double deviation=0.0,largest=0.0;
  for (int spin=0; spin<spin_channels && spin<state->spin_channels; spin++) {
    const double *dense=state->density[spin];
    for (int local=1; local<=local_atoms; local++) {
      const int atom=local_to_global[local];
      const int rows=orbitals_of_species[species_of_atom[atom]];
      const size_t row0=(size_t)first_orbital[atom]-1;
      for (int neighbour=0; neighbour<=neighbour_count[atom]; neighbour++) {
        const int partner=neighbour_atom[atom][neighbour];
        const int columns=orbitals_of_species[species_of_atom[partner]];
        const size_t column0=(size_t)first_orbital[partner]-1;
        for (int i=0; i<rows; i++) {
          for (int j=0; j<columns; j++) {
            const double difference=fabs(density[spin][local][neighbour][i][j]-dense[(row0+i)*n+column0+j]);
            if (deviation<difference) deviation = difference;
          }
        }
      }
    }
  }
  MPI_Allreduce(&deviation,&largest,1,MPI_DOUBLE,MPI_MAX,comm);
  return largest;
}

void Exact_Exchange_Build(OpenMXExactExchangeState *state, MPI_Comm comm, int verbosity)
{
  if (!Exact_Exchange_Enabled(state)) return;
  if (!state->prepared || state->spin_channels<1 || state->collecting)
    ExactExchange_Abort(comm,"exchange matrix","no density matrix is collected");

  const double start=MPI_Wtime();
  const int n=state->orbitals;
  const int atoms=openmx_exx_atom_count(state->module);
  int rank=0,processes=1;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&processes);

  /* The centre atoms of this process: a contiguous range, chosen so that
     the ranges carry about the same number of ABFs times V neighbours. */
  double total=0.0;
  for (int a=0; a<atoms; a++)
    total += (double)openmx_exx_atom_abfs(state->module,a)*(double)openmx_exx_v_neighbours(state->module,a);
  int first=atoms,last=atoms;
  {
    const double lower=total*(double)rank/(double)processes;
    const double upper=total*(double)(rank+1)/(double)processes;
    double before=0.0;
    for (int a=0; a<atoms; a++) {
      const double middle=before+0.5*(double)openmx_exx_atom_abfs(state->module,a)
                                     *(double)openmx_exx_v_neighbours(state->module,a);
      if (lower<=middle && middle<upper) {
        if (first==atoms) first = a;
        last = a+1;
      }
      before += (double)openmx_exx_atom_abfs(state->module,a)*(double)openmx_exx_v_neighbours(state->module,a);
    }
    if (first==atoms) last = atoms;
  }

  const OpenMXExxDenseBackend backend={ExactExchange_Dgemm,NULL};
  for (int spin=0; spin<state->spin_channels; spin++) {
    /* The matrix of the build before stays for delta; without one it is 0. */
    double *swap=state->previous[spin];
    state->previous[spin] = state->exchange[spin];
    state->exchange[spin] = swap;
    if (!state->exchange_ready) memset(state->previous[spin],0,sizeof(double)*state->elements);
    memset(state->exchange[spin],0,sizeof(double)*state->elements);

    if (state->input.alpha!=0.0) {
      char error[512];
      if (first<last && openmx_exx_add_gamma(state->module,first,last,-state->input.alpha,state->density[spin],
                                             state->exchange[spin],&backend,error,sizeof(error))!=0)
        ExactExchange_Abort(comm,"exchange matrix",error);
      ExactExchange_Sum(state->exchange[spin],state->elements,comm);
      ExactExchange_Symmetrize(state->exchange[spin],n);
    }
  }

  /* One spin channel stands for both without spin polarization. Process 0
     decides for all: the numbers that steer the outer iteration must not
     differ by a rounding error between the processes. */
  const double degeneracy=state->spin_channels==1 ? 2.0 : 1.0;
  double results[2]={0.0,0.0};
  for (int spin=0; spin<state->spin_channels; spin++) {
    results[0] += degeneracy*openmx_exx_energy_gamma(state->elements,state->density[spin],state->exchange[spin]);
    results[1] += degeneracy*openmx_exx_change_gamma(state->elements,state->density[spin],state->exchange[spin],
                                                     state->previous[spin]);
  }
  MPI_Bcast(results,2,MPI_DOUBLE,0,comm);
  state->energy = results[0];
  state->change = results[1];
  state->exchange_ready = 1;
  state->outer_iterations++;
  state->outer_converged = 0;
  state->residual_reference = -1.0;
  state->time_build += MPI_Wtime()-start;

  if (rank==0 && 2<=verbosity) {
    printf("<EXX>  exchange matrix of %d orbitals built in %.2f s (centre atoms %d to %d of %d on process 0)\n",
           n,MPI_Wtime()-start,first+1,last,atoms);
    fflush(stdout);
  }
}

int Exact_Exchange_Outer_Continue(OpenMXExactExchangeState *state)
{
  if (2<=state->outer_iterations && fabs(state->change)<state->input.outer_criterion) {
    state->outer_converged = 1;
    return 0;
  }
  return state->outer_iterations<state->input.outer_max_iterations;
}

int Exact_Exchange_Mixing_Offset(int mixing_type, int start_pulay, int local_step)
{
  int first;  /* step count of the first mixing after the build */
  int before; /* 1 when the mixing of a step comes before its solution */
  int offset;

  switch (mixing_type) {
  case 1: /* RMM-DIIS, */
  case 2: /* GR-Pulay, */
  case 6: /* RMM-ADIIS: the history starts at the count start_pulay */
    first = start_pulay-2;
    before = 0;
    break;
  case 4: /* RMM-DIISK: from the count 7 on, the extrapolation at the count
             start_pulay takes two residuals; before that it takes every
             residual since the count 2 */
    first = 7<=start_pulay ? start_pulay-2 : 1;
    before = 0;
    break;
  case 7: /* RMM-DIISV: the counts of RMM-DIISK */
    first = 7<=start_pulay ? start_pulay-2 : 1;
    before = 1;
    break;
  case 5: /* RMM-DIISH: every residual since the count 2 */
    first = 1;
    before = 1;
    break;
  default: /* Simple, Kerker: no history of residuals */
    return -1;
  }
  if (first<1) return -1;
  offset = local_step+before-first;
  return 0<=offset ? offset : -1;
}

int Exact_Exchange_Inner_Stop_Allowed(OpenMXExactExchangeState *state, int step, double residual_norm,
    double criterion)
{
  double bound;
  if (!Exact_Exchange_Active(state)) return 1;
  if (step<2) return 0;
  if (step==2 || state->residual_reference<0.0) {
    state->residual_reference = residual_norm;
    return 0;
  }
  bound = 0.1*state->residual_reference;
  if (bound<criterion) bound = criterion;
  return residual_norm<=bound;
}

void Exact_Exchange_Add_Cluster_Hs(OpenMXExactExchangeState *state, int spin, int n, double *Hs,
    int na_rows, int na_cols, int nblk, int np_rows, int np_cols, int my_prow, int my_pcol)
{
  if (!Exact_Exchange_Active(state)) return;
  if (n!=state->orbitals || spin<0 || state->spin_channels<=spin)
    ExactExchange_Abort(state->comm,"Hamiltonian","the matrix of the solver has an unexpected size or spin");

  const double start=MPI_Wtime();
  const double *exchange=state->exchange[spin];
  for (int j=0; j<na_cols; j++) {
    const int column=np_cols*nblk*(j/nblk)+j%nblk+((np_cols+my_pcol)%np_cols)*nblk;
    if (n<=column) continue;
    for (int i=0; i<na_rows; i++) {
      const int row=np_rows*nblk*(i/nblk)+i%nblk+((np_rows+my_prow)%np_rows)*nblk;
      if (row<n) Hs[(size_t)j*(size_t)na_rows+i] += exchange[(size_t)row*(size_t)n+column];
    }
  }
  state->time_add += MPI_Wtime()-start;
}
