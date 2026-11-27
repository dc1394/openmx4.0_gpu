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
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "openmx_core_native_resources.h"
#include "openmx_core_radial_views.h"
#include "openmx_exact_exchange.h"
#include "openmx_exact_exchange_device.h"
#include "openmx_exact_exchange_host.h"
#include "openmx_gpu_backend.h"
#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
#include "openmx_gpu_architecture.h"
#include "openmx_gpu_collective.h"
#include "openmx_gpu_device.h"
#include "openmx_mgpu_runtime.h"
#endif
#include "openmx_host_simd.h"
#include "openmx_exx.h"
#include "lapack_prototypes.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/* K(k) of one spin channel and k point in the local array of a solver
   (Exact_Exchange_Add_Band_Hs): the sum of equation (14), kept from the
   first addition until K changes. The layout of the local array belongs to
   the key: the solvers on a device pass the whole matrix (a 1 x 1 grid),
   the others their block-cyclic part. */
typedef struct {
  int spin,n,na_rows,na_cols,nblk,np_rows,np_cols,my_prow,my_pcol;
  double k[3];
  dcomplex *values;                 /* na_rows x na_cols, in the order of the local array */
} ExactExchangeBandMatrix;

enum { EXX_GPU_PRODUCTS = 1, EXX_GPU_ADDITION = 2 };

/* Persistent metadata belongs to the explicit calculation owner. */
struct OpenMXExactExchangeState {
  OpenMXNativeResources *resources; /* borrowed from this calculation */
  int initialized;
  OpenMXExactExchangeInput input;   /* values of the input keywords */
  FILE *scf_log;                    /* rank 0's append stream for the current MD step */
  int stdout_level;
  int gpu_dispatch_enabled;         /* rank 0's result for all ranks */
  int gpu_work;                     /* EXX_GPU_* completed on this rank in the current MD step */
  OpenMXExactExchangeStructure structure; /* arrays of the main program, borrowed */
  OpenMXExxState *module;           /* tables of modules/exx, released by the reset */
  int prepared;                     /* the module holds the present structure */
  MPI_Comm comm;                    /* communicator of the last preparation or collection */
  int orbitals;                     /* N of the present structure */
  int sites;                        /* form of the matrices below: 0 N x N (Gamma point), 1 blocks of the site form */
  int folded;                       /* the site form is folded by input.fold_period, its V neighbours and columns */
  long block_count;                 /* site form: the blocks M_xy(R) of the list of modules/exx */
  int *block_atoms;                 /* x and y of every block, counted from 0 */
  int *block_cell;                  /* R of every block, three integers */
  size_t *block_offset;             /* first element of every block; block_count+1 entries */
  int *first_orbital;               /* first orbital of every atom, counted from 0; N at the end */
  int allocated_channels;           /* spin channels the matrices below have storage for */
  int spin_channels;                /* spin channels of the last collection: 0, 1 or 2 */
  size_t elements;                  /* elements of each matrix below: N*N, or those of all blocks */
  double *density[2];               /* D of every spin channel */
  double *exchange[2];              /* K of the last build */
  double *previous[2];              /* K of the build before */
  int collecting;                   /* the solver hands vectors and occupations over */
  int exchange_ready;               /* K exists: it is added to the Hamiltonian */
  int outer_iterations;             /* builds of K for the present structure */
  int outer_converged;              /* the last build met scf.exx.outer.criterion */
  double energy;                    /* E_x of the last build, Hartree */
  double change;                    /* delta of the last build, Hartree */
  double residual_reference;        /* residual norm of the first step with the K of the last build; negative: not seen yet */
  /* The builds that Anderson's method combines (Exact_Exchange_Extrapolate):
     room+1 slots of every spin channel, the oldest first; entry 2*slot+spin */
  int history_room;                 /* scf.exx.outer.extrapolation that the slots were made for; 0: none */
  int history_channels;             /* spin channels that the slots have room for */
  int history_count;                /* builds stored in the first slots for the present structure */
  double **history_built;           /* K of a build as it was built, X */
  double **history_residual;        /* the residual of that build, F = X - Y */
  double time_prepare,time_collect,time_build,time_add; /* seconds on this process */
  OpenMXExactExchangeDevice device; /* matrix products and whole calls of the build on the device of this process */
  double time_device;               /* seconds in the products, with the transfers; part of time_build */
  double time_device_calls;         /* seconds in the calls that the device computed; part of time_build */
  long long device_additions;       /* dense Hamiltonians of a solver on its device that K was added to */
  ExactExchangeBandMatrix *band_matrices; /* K(k) of the additions since the last build */
  int band_matrix_count;            /* entries of band_matrices in use */
  int band_matrix_room;             /* entries that band_matrices has room for */
  size_t band_matrix_bytes;         /* bytes of the values of the entries in use */
  long long band_sums;              /* sums of equation (14) since the binding */
  long long band_reuses;            /* additions since the binding that took K(k) from an entry */
  double *gradient;                 /* dE_x/dtau of the last call of Exact_Exchange_Gradient, 3 per atom */
  int gradient_atoms;               /* atoms that gradient has room for */
  /* K(R) for the outputs after the SCF (Exact_Exchange_Prepare_Outputs):
     the blocks of the site form without folding, in the arrays of the
     outputs, or the blocks above when the site form is not folded */
  int outputs;                      /* the matrices of the outputs exist */
  int output_owned;                 /* the arrays below are allocations of the outputs */
  int output_spin_channels;
  long output_block_count;
  int *output_block_atoms;          /* x and y of every block, counted from 0 */
  int *output_block_cell;           /* R of every block, three integers */
  size_t *output_block_offset;      /* first element of every block; output_block_count+1 entries */
  double *output_exchange[2];       /* K(R) of every spin channel */
};

static void ExactExchange_Reset(OpenMXNativeResources *owner, void *metadata)
{
  OpenMXExactExchangeState *state=(OpenMXExactExchangeState*)metadata;
  (void)owner;
  /* The matrices and the lists of the blocks are allocations of the owner,
     which releases them. */
  if (state->module!=NULL) openmx_exx_destroy(state->module);
  state->module = NULL;
  if (state->scf_log!=NULL) fclose(state->scf_log);
  state->scf_log = NULL;
  Exact_Exchange_Device_Release(&state->device);
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
    Exact_Exchange_Device_Initialize(&state->device);
    state->initialized = 1;
  }
  return state;
}

void Exact_Exchange_Configure(OpenMXExactExchangeState *state, const OpenMXExactExchangeInput *input)
{
  state->input = *input;
  /* The fault injection of the GPU paths: no matrix product fits. The device
     is still asked for its free memory, and every product goes to the host. */
  state->device.reserve_bytes = input->device_capacity_test ? (size_t)-1 : OPENMX_EXACT_EXCHANGE_DEVICE_RESERVE;
  /* GEMMul8 only where the input asks for it: the default is FP64. */
  state->device.gemmul8 = input->gemmul8 ? 1 : 0;
}

void Exact_Exchange_Log(const OpenMXExactExchangeState *state, int level, const char *format, ...)
{
  va_list args;
  if (level<=state->stdout_level) {
    va_start(args,format);
    vfprintf(stdout,format,args);
    va_end(args);
    fflush(stdout);
  }
  if (state->scf_log!=NULL && (level==1 || level<=state->input.fileout_level)) {
    va_start(args,format);
    vfprintf(state->scf_log,format,args);
    va_end(args);
    fflush(state->scf_log);
  }
}

static int ExactExchange_Detailed(const OpenMXExactExchangeState *state)
{
  return 2<=state->stdout_level || 2<=state->input.fileout_level;
}

static void ExactExchange_Report_Device_Failure(const OpenMXExactExchangeState *state, int rank, int disabled_before)
{
  if (state->device.disabled && !disabled_before) {
    Exact_Exchange_Log(state,2,"<EXX>  WARNING: process %d: %s failed with status %d.\n",rank,
                       Exact_Exchange_Device_Failure_Name(state->device.failure),state->device.failure_status);
  }
}

static int ExactExchange_Configure_Dispatch(OpenMXExactExchangeState *state, MPI_Comm comm, int rank)
{
  const double configured=state->input.gpu_min_work;
  /* Batches amortize transfer and launch overhead even when no single
     product would pay for its own copies. The total-work floor still applies. */
  state->device.dispatch_work = configured>=0.0 ? configured : 1.0e8;
  state->device.sharing_ranks = 1;
  state->device.retain_bytes = 0;
#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
  unsigned char uuid[16];
  int device=-1;
  const int available=!OpenMX_HostSimdPreferred("EXX") && gpu_rank_device_usable();
  const int usable=available && OpenMXGpuGetDevice(&device)==OPENMX_GPU_SUCCESS &&
                   openmx_gpu_physical_uuid(device,uuid);
  const int sharing_ranks=openmx_gpu_local_sharers(comm,usable ? uuid : NULL,NULL);
  const int enabled=available && !state->device.disabled;
  if (!usable) return enabled;
  if (sharing_ranks>0) state->device.sharing_ranks = sharing_ranks;
  size_t free_bytes=0,total_bytes=0;
  /* Cap retained storage across ranks, leaving room for the rest of SCF.
     Larger requests can still use temporary memory during EXX. */
  if (OpenMX_GpuMemGetInfo(&free_bytes,&total_bytes)==OPENMX_GPU_SUCCESS)
    state->device.retain_bytes = total_bytes/3/(size_t)state->device.sharing_ranks;
  else
    (void)OpenMXGpuGetLastError();
  if (configured<0.0) {
#if OPENMX_GPU_HIP
    hipDeviceProp_t properties;
    if (OpenMXGpuGetDeviceProperties(&properties,device)!=OPENMX_GPU_SUCCESS)
      (void)OpenMXGpuGetLastError();
    /* Consumer AMD GPUs share their limited native FP64 throughput across
       the active ranks. Use a higher cutoff for transfers and batched work. */
    else if (!openmx_amd_high_throughput_fp64(properties.gcnArchName))
      state->device.dispatch_work = 1.0e10*state->device.sharing_ranks;
#else
    int ratio=0;
    if (cudaDeviceGetAttribute(&ratio,cudaDevAttrSingleToDoublePrecisionPerfRatio,device)!=cudaSuccess)
      (void)OpenMXGpuGetLastError();
    else if (ratio>=8) {
      /* Batching amortizes launches even on a heavily shared GeForce.
         Increase the cutoff gradually once more than 16 ranks share it. */
      state->device.dispatch_work *= fmax(1.0,state->device.sharing_ranks/16.0);
    }
#endif
  }
  if (rank==0 && enabled)
    Exact_Exchange_Log(state,2,"<EXX>  GPU dispatch: at least %.3g multiplications in one product (%d ranks on this GPU%s)\n",
                       state->device.dispatch_work,state->device.sharing_ranks,configured<0.0 ? ", automatic" : "");
  return enabled;
#else
  (void)comm;
  (void)rank;
  return 0;
#endif
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

void Exact_Exchange_End_Device_Phase(OpenMXExactExchangeState *state)
{
  if (state!=NULL) Exact_Exchange_Device_Release(&state->device);
}

double Exact_Exchange_Semilocal_Alpha(const OpenMXExactExchangeState *state)
{
  return Exact_Exchange_Active(state) ? state->input.alpha : 0.0;
}

double Exact_Exchange_Semilocal_Omega(const OpenMXExactExchangeState *state)
{
  return (Exact_Exchange_Enabled(state) && state->input.functional==OPENMX_EXACT_EXCHANGE_HSE06)
         ? state->input.omega : 0.0;
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

void Exact_Exchange_Report_Scf(const OpenMXExactExchangeState *state, MPI_Comm comm,
    int inner_converged, int scf_limit)
{
  int rank;
  MPI_Comm_rank(comm,&rank);
  if (rank==0) {
    const int builds=state->outer_iterations;
    if (builds==0) {
      Exact_Exchange_Log(state,1,"<EXX>  WARNING: the GGA-PBE SCF did not converge within scf.maxIter=%d. "
                         "The energies are from GGA-PBE.\n",scf_limit);
    }
    else if (!inner_converged) {
      Exact_Exchange_Log(state,1,"<EXX>  WARNING: the SCF with exchange matrix %d did not converge "
                         "within scf.maxIter=%d.\n",builds,scf_limit);
    }
    else if (!state->outer_converged) {
      Exact_Exchange_Log(state,1,"<EXX>  WARNING: the outer iteration did not converge in %d exchange builds "
                         "(scf.exx.outer.maxIter, delta=%.3e Hartree).\n",builds,state->change);
    }
    else {
      Exact_Exchange_Log(state,1,"<EXX>  The outer iteration converged with %d builds of the exchange matrix: "
                         "delta=%.3e Hartree.\n",builds,state->change);
    }
  }
  if (!ExactExchange_Detailed(state)) return;

  const OpenMXExactExchangeDevice *device=&state->device;
  long long local_use[4]={device->device_products,device->declined_products,device->device_products>0,
                          device->gemmul8_products};
  long long local_calls[6]={device->device_calls,device->declined_calls,device->device_calls>0,
                            device->call_products,device->call_large_products,device->call_gemmul8_products};
  long long local_add[3]={state->band_sums,state->band_reuses,state->device_additions};
  double local_seconds[2]={state->time_device,state->time_device_calls};
  long long use[4],calls[6],add[3];
  double seconds[2];
  MPI_Reduce(local_use,use,4,MPI_LONG_LONG,MPI_SUM,0,comm);
  MPI_Reduce(local_calls,calls,6,MPI_LONG_LONG,MPI_SUM,0,comm);
  MPI_Reduce(local_add,add,3,MPI_LONG_LONG,MPI_SUM,0,comm);
  MPI_Reduce(local_seconds,seconds,2,MPI_DOUBLE,MPI_MAX,0,comm);
  if (rank!=0) return;

  Exact_Exchange_Log(state,2,"<EXX>  time (s): preparation %.2f, density matrices %.2f, exchange matrix %.2f, addition %.2f\n",
                     state->time_prepare,state->time_collect,state->time_build,state->time_add);
  if (add[0]>0 || add[1]>0) {
    Exact_Exchange_Log(state,2,"<EXX>  K(k) of the k points: %lld sums over the blocks of K, and %lld additions of a sum"
                       " kept since the same build (all processes)\n",add[0],add[1]);
  }
  if (use[0]>0 || use[1]>0) {
    Exact_Exchange_Log(state,2,"<EXX>  GPU: %lld matrix products of the exchange matrix on the device, in %lld processes\n"
                       "<EXX>  (at most %.2f s in one process); %lld left to the host for lack of device memory.\n",
                       use[0],use[2],seconds[0],use[1]);
    if (state->input.gemmul8 && use[0]>0) {
      Exact_Exchange_Log(state,2,"<EXX>  GPU: GEMMul8 computed %lld of the matrix products on the device, FP64 the other %lld"
                         " (scf.gemmul8.enable on).\n",use[3],use[0]-use[3]);
    }
  }
  if (calls[0]>0 || calls[1]>0) {
    Exact_Exchange_Log(state,2,"<EXX>  GPU: %lld calls of the exchange build on the device, in %lld processes, with %lld matrix products,"
                       " %lld of them large\n<EXX>  (at most %.2f s in one process); %lld calls left to the host for lack of device memory.\n",
                       calls[0],calls[2],calls[3],calls[4],seconds[1],calls[1]);
    if (state->input.gemmul8 && calls[0]>0) {
      Exact_Exchange_Log(state,2,"<EXX>  GPU: GEMMul8 computed %lld of the %lld large matrix products of these calls, FP64 the other"
                         " %lld products (scf.gemmul8.enable on).\n",calls[5],calls[4],calls[3]-calls[5]);
    }
  }
  if (add[2]>0) {
    Exact_Exchange_Log(state,2,"<EXX>  GPU: the exchange matrix was added to %lld dense Hamiltonians on the device.\n",add[2]);
  }
}

void Exact_Exchange_Finish_Report(OpenMXExactExchangeState *state, MPI_Comm comm)
{
  if (!Exact_Exchange_Enabled(state) || !state->prepared) return;
  int rank;
  MPI_Comm_rank(comm,&rank);
  int local[3]={!!(state->gpu_work&EXX_GPU_PRODUCTS),!!(state->gpu_work&EXX_GPU_ADDITION),state->device.disabled};
  int totals[3];
  MPI_Reduce(local,totals,3,MPI_INT,MPI_SUM,0,comm);
  if (rank!=0) return;

  if (totals[0]>0) {
    Exact_Exchange_Log(state,1,"<EXX>  GPU-accelerated exact exchange (%s matrix products)\n",OPENMX_GPU_RUNTIME_NAME);
  }
  else if (totals[1]>0) {
    Exact_Exchange_Log(state,1,"<EXX>  Exact exchange added to the Hamiltonian on GPU (%s)\n",OPENMX_GPU_RUNTIME_NAME);
  }
  else if (state->gpu_dispatch_enabled && state->outer_iterations>0) {
    Exact_Exchange_Log(state,1,"<EXX>  Exact exchange matrix products ran on CPU (GPU dispatch enabled, %s)\n",
                       OPENMX_GPU_RUNTIME_NAME);
  }
  if (totals[2]>0) {
    Exact_Exchange_Log(state,1,"<EXX>  WARNING: %d processes stopped using their device for the exchange matrix after a failed call.\n",
                       totals[2]);
  }
  if (state->scf_log!=NULL) fclose(state->scf_log);
  state->scf_log = NULL;
}

void Exact_Exchange_Device_Addition(OpenMXExactExchangeState *state, double seconds)
{
  if (state==NULL) return;
  state->gpu_work |= EXX_GPU_ADDITION;
  state->device_additions++;
  state->time_add += seconds;
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

/* (M + M^T)/2 in place. Returns the largest |M_ij - M_ji| before. */
static double ExactExchange_Symmetrize(double *matrix, int n)
{
  double largest=0.0;
  for (int i=0; i<n; i++) {
    for (int j=i+1; j<n; j++) {
      const double upper=matrix[(size_t)i*n+j],lower=matrix[(size_t)j*n+i];
      const double value=0.5*(upper+lower);
      if (largest<fabs(upper-lower)) largest = fabs(upper-lower);
      matrix[(size_t)i*n+j] = value;
      matrix[(size_t)j*n+i] = value;
    }
  }
  return largest;
}

/* The same for one spin channel in the form of the state: the means of
   M_xy(R) and the transpose of M_yx(-R) for the blocks of the site form. */
static double ExactExchange_Symmetrize_Channel(const OpenMXExactExchangeState *state, double *matrix, MPI_Comm comm,
                                               const char *where)
{
  double largest=0.0;
  if (state->sites) {
    char error[512];
    if (openmx_exx_symmetrize_blocks(state->module,matrix,&largest,error,sizeof(error))!=0)
      ExactExchange_Abort(comm,where,error);
  }
  else largest = ExactExchange_Symmetrize(matrix,state->orbitals);
  return largest;
}

/* X + X^T in place for one spin channel in the form of the state: the half
   build gives X (section 6.4), and K = X + X^T is symmetric to the last bit,
   K_xy(R) = K_yx(-R)^T in the site form. */
static void ExactExchange_Mirror_Channel(const OpenMXExactExchangeState *state, double *matrix, MPI_Comm comm,
                                         const char *where)
{
  char error[512];
  const int failed=state->sites ? openmx_exx_mirror_blocks(state->module,matrix,error,sizeof(error))
                                : openmx_exx_mirror_gamma(state->module,matrix,error,sizeof(error));
  if (failed!=0) ExactExchange_Abort(comm,where,error);
}

/* What the table of functions carries to the matrix products of one build. */
typedef struct {
  OpenMXExactExchangeDevice *device;
  double device_seconds; /* in the products that the device took, with the transfers */
  OpenMXExactExchangeHostProducts host; /* the counts of the products, and the loops of the build */
} ExactExchangeProducts;

static ExactExchangeProducts ExactExchange_Products(OpenMXExactExchangeState *state, int threads)
{
  OpenMXExactExchangeDevice *device=OpenMX_HostSimdPreferred("EXX") ? NULL : &state->device;
  const ExactExchangeProducts products={device,0.0,{0,0.0,{0,0,0},threads,NULL,0}};
  return products;
}

/* The matrix products of modules/exx. The device of this process takes the
   large ones when it can (openmx_exact_exchange_device.h); the others, and
   all of them in the CPU build, go to the BLAS of the main program, divided
   among the threads when they are large enough
   (openmx_exact_exchange_host.h). The products of a loop
   (ExactExchange_Parallel_For) are called by several threads at the same
   time: they are small, and each goes to the BLAS on the thread that
   calls. */
static int ExactExchange_Dgemm(void *context, char transa, char transb, int m, int n, int k, double alpha,
                               const double *a, int lda, const double *b, int ldb, double beta, double *c,
                               int ldc)
{
  ExactExchangeProducts *products=(ExactExchangeProducts*)context;
  if (products->host.in_loop) {
    Exact_Exchange_Host_Loop_Dgemm(&products->host,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
    return 0;
  }
  Exact_Exchange_Host_Note_Product(&products->host,m,n,k);
  if (products->device!=NULL) {
    const double start=MPI_Wtime();
    if (Exact_Exchange_Device_Dgemm(products->device,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc)) {
      products->device_seconds += MPI_Wtime()-start;
      return 0;
    }
  }
  Exact_Exchange_Host_Dgemm(products->host.threads,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
  return 0;
}

/* The loop of the table of functions: the threads of OpenMP of the main
   program run the pieces into which modules/exx divides the ABFs of a
   centre atom (openmx_exact_exchange_host.h). */
static int ExactExchange_Parallel_For(void *context, int count, void (*body)(void *argument, int index),
                                      void *argument)
{
  ExactExchangeProducts *products=(ExactExchangeProducts*)context;
  return Exact_Exchange_Host_Parallel_For(&products->host,count,body,argument);
}

/* The counts of the device that the builds report. The weights of
   the forces use the device as well and are counted aside: the counts are
   taken before the weights, the two largest sizes set to zero, and the
   counts put back afterwards. */
typedef struct {
  long long device_products,gemmul8_products,declined_products;
  size_t largest_bytes;
  long long device_calls,declined_calls,call_products,call_large_products,call_gemmul8_products;
  size_t largest_call_bytes;
} ExactExchangeDeviceCounts;

static ExactExchangeDeviceCounts ExactExchange_Device_Counts(const OpenMXExactExchangeDevice *device)
{
  const ExactExchangeDeviceCounts counts={device->device_products,device->gemmul8_products,
                                          device->declined_products,device->largest_bytes,device->device_calls,
                                          device->declined_calls,device->call_products,device->call_large_products,
                                          device->call_gemmul8_products,device->largest_call_bytes};
  return counts;
}

static void ExactExchange_Device_Counts_Back(OpenMXExactExchangeDevice *device, const ExactExchangeDeviceCounts *counts)
{
  device->device_products = counts->device_products;
  device->gemmul8_products = counts->gemmul8_products;
  device->declined_products = counts->declined_products;
  device->largest_bytes = counts->largest_bytes;
  device->device_calls = counts->device_calls;
  device->declined_calls = counts->declined_calls;
  device->call_products = counts->call_products;
  device->call_large_products = counts->call_large_products;
  device->call_gemmul8_products = counts->call_gemmul8_products;
  device->largest_call_bytes = counts->largest_call_bytes;
}

/* K is about to change, or the structure: the K(k) kept for the additions
   are dropped, and the next addition at every k point sums again. */
static void ExactExchange_Forget_Band_Matrices(OpenMXExactExchangeState *state)
{
  for (int e=0; e<state->band_matrix_count; e++) {
    openmx_core_native_free(state->resources,state->band_matrices[e].values);
    state->band_matrices[e].values = NULL;
  }
  state->band_matrix_count = 0;
  state->band_matrix_bytes = 0;
}

/* The list of the blocks of the site form, as the module gives it for the
   prepared structure. */
static void ExactExchange_Release_Blocks(OpenMXExactExchangeState *state)
{
  openmx_core_native_free(state->resources,state->block_atoms);
  openmx_core_native_free(state->resources,state->block_cell);
  openmx_core_native_free(state->resources,state->block_offset);
  openmx_core_native_free(state->resources,state->first_orbital);
  state->block_atoms = NULL;
  state->block_cell = NULL;
  state->block_offset = NULL;
  state->first_orbital = NULL;
  state->block_count = 0;
}

/* The list of the blocks of module: atoms, cells and offsets in arrays of
   the state. Returns the number of blocks. */
static long ExactExchange_Read_Blocks(OpenMXExactExchangeState *state, OpenMXExxState *module, MPI_Comm comm,
                                      int **block_atoms, int **block_cell, size_t **block_offset)
{
  char error[512];
  if (openmx_exx_prepare_blocks(module,error,sizeof(error))!=0) ExactExchange_Abort(comm,"preparation",error);
  const long blocks=openmx_exx_block_count(module);
  if (blocks<1) ExactExchange_Abort(comm,"preparation","the structure has no block of the exchange matrix");

  int *atoms=(int*)ExactExchange_Allocate(state,2*(size_t)blocks,sizeof(int),comm,"preparation");
  int *cell=(int*)ExactExchange_Allocate(state,3*(size_t)blocks,sizeof(int),comm,"preparation");
  size_t *offset=(size_t*)ExactExchange_Allocate(state,(size_t)blocks+1,sizeof(size_t),comm,"preparation");
  for (long block=0; block<blocks; block++) {
    if (openmx_exx_block(module,block,&atoms[2*block],&atoms[2*block+1],&cell[3*block],&offset[block])!=0)
      ExactExchange_Abort(comm,"preparation","the list of the blocks of the exchange matrix is incomplete");
  }
  offset[blocks] = openmx_exx_block_elements(module);
  *block_atoms = atoms;
  *block_cell = cell;
  *block_offset = offset;
  return blocks;
}

static void ExactExchange_List_Blocks(OpenMXExactExchangeState *state, MPI_Comm comm)
{
  const int atoms=openmx_exx_atom_count(state->module);
  state->block_count = ExactExchange_Read_Blocks(state,state->module,comm,&state->block_atoms,&state->block_cell,
                                                 &state->block_offset);
  state->first_orbital = (int*)ExactExchange_Allocate(state,(size_t)atoms+1,sizeof(int),comm,"preparation");
  state->first_orbital[0] = 0;
  for (int a=0; a<atoms; a++) state->first_orbital[a+1] = state->first_orbital[a]+openmx_exx_atom_orbitals(state->module,a);
}

/* The slots of Anderson's method (Exact_Exchange_Extrapolate). */
static void ExactExchange_Release_History(OpenMXExactExchangeState *state)
{
  for (int slot=0; slot<=state->history_room && state->history_built!=NULL; slot++) {
    for (int spin=0; spin<state->history_channels; spin++) {
      openmx_core_native_free(state->resources,state->history_built[2*slot+spin]);
      openmx_core_native_free(state->resources,state->history_residual[2*slot+spin]);
    }
  }
  openmx_core_native_free(state->resources,state->history_built);
  openmx_core_native_free(state->resources,state->history_residual);
  state->history_built = NULL;
  state->history_residual = NULL;
  state->history_room = 0;
  state->history_channels = 0;
  state->history_count = 0;
}

/* Storage of the density and exchange matrices of spin_channels channels. */
static void ExactExchange_Reserve(OpenMXExactExchangeState *state, int spin_channels, MPI_Comm comm)
{
  const size_t elements=state->sites ? state->block_offset[state->block_count]
                                     : (size_t)state->orbitals*(size_t)state->orbitals;
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
    ExactExchange_Release_History(state);
  }
  for (int spin=state->allocated_channels; spin<spin_channels; spin++) {
    state->density[spin] = (double*)ExactExchange_Allocate(state,elements,sizeof(double),comm,"density matrix");
    state->exchange[spin] = (double*)ExactExchange_Allocate(state,elements,sizeof(double),comm,"exchange matrix");
    state->previous[spin] = (double*)ExactExchange_Allocate(state,elements,sizeof(double),comm,"exchange matrix");
  }
  if (state->allocated_channels<spin_channels) state->allocated_channels = spin_channels;
}

/* The T and V blocks after a preparation that the processes divided
   (section 6.7): every block was computed by one process and holds zeros on
   the others, so the sum over the processes is the block of the whole
   preparation, and every process stores it. The blocks of one atom travel
   at a time. */
static void ExactExchange_Add_Up_Blocks(OpenMXExactExchangeState *state, OpenMXExxState *module, MPI_Comm comm)
{
  const int atoms=openmx_exx_atom_count(module);
  size_t largest=0;
  for (int a=0; a<atoms; a++) {
    const size_t elements=openmx_exx_atom_table_elements(module,a);
    if (largest<elements) largest = elements;
  }
  if (largest==0) ExactExchange_Abort(comm,"preparation","the structure has no T and V blocks");
  double *values=(double*)ExactExchange_Allocate(state,largest,sizeof(double),comm,"preparation");
  for (int a=0; a<atoms; a++) {
    const size_t elements=openmx_exx_atom_table_elements(module,a);
    if (openmx_exx_get_atom_tables(module,a,values)!=0)
      ExactExchange_Abort(comm,"preparation","the T and V blocks of an atom cannot be read");
    ExactExchange_Sum(values,elements,comm);
    if (openmx_exx_set_atom_tables(module,a,values)!=0)
      ExactExchange_Abort(comm,"preparation","the T and V blocks of an atom cannot be stored");
  }
  openmx_core_native_free(state->resources,values);
}

/* The radius of the sphere whose volume is that of the cell times the number
   of k points (section 5.6, [13]). */
double Exact_Exchange_Truncation_Limit(int kpoint_count, double (*lattice)[4])
{
  const double *a=lattice[1], *b=lattice[2], *c=lattice[3];
  const double volume=fabs(a[1]*(b[2]*c[3]-b[3]*c[2])+a[2]*(b[3]*c[1]-b[1]*c[3])+a[3]*(b[1]*c[2]-b[2]*c[1]));
  return cbrt(3.0*(double)kpoint_count*volume/(4.0*3.14159265358979323846));
}

/* The radius R_c of the truncated Coulomb interaction of PBE0 (bohr): the
   value of scf.exx.coulomb.cutoff, or else the radius of the sphere whose
   volume is that of the cell times the number of k points. The cell is that
   of the structure at the preparation, so that the radius follows the cell
   where MD.Type changes it (EvsLC and DF scale it). A value beyond the
   radius of the sphere ends the run (section 13, item 72): the function
   that reads the input refuses it for the cell of the input, and this check
   for a later cell. */
static double ExactExchange_Truncation_Radius(const OpenMXExactExchangeInput *input, double (*lattice)[4],
                                              MPI_Comm comm)
{
  const double limit=Exact_Exchange_Truncation_Limit(input->kpoint_count,lattice);
  if (!(0.0<input->coulomb_cutoff)) return limit;
  if (limit+OPENMX_EXACT_EXCHANGE_TRUNCATION_TOLERANCE<input->coulomb_cutoff) {
    char message[256];
    snprintf(message,sizeof(message),"scf.exx.coulomb.cutoff=%.6f (bohr) is larger than %.6f (bohr), the "
             "radius of the sphere of the volume of the cell times the number of k points",
             input->coulomb_cutoff,limit);
    ExactExchange_Abort(comm,"preparation",message);
  }
  return input->coulomb_cutoff;
}

/* The ABFs, neighbour lists, T blocks and V blocks of the bound structure as
   it is at the call, in module (section 6.2). Every process makes the
   tables of the species and the neighbour lists, and the processes of
   comm divide the pairs of sites of the T and V blocks and add the blocks
   up (section 6.7). prepared_options gets the options of the preparation.
   Returns the number of orbitals of the structure. */
static int ExactExchange_Prepare_Module(OpenMXExactExchangeState *state, OpenMXExxState *module,
                                        const OpenMXAtomicViews *atomic, MPI_Comm comm,
                                        OpenMXExxOptions *prepared_options)
{
  const int atom_count=state->structure.atom_count;
  const int *species_of_atom=state->structure.species_of_atom;
  double *const *positions=state->structure.positions;
  double (*lattice)[4]=state->structure.lattice;
  const int species_count=(int)atomic->species.species;
  int rank=0,processes=1;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&processes);

  /* The workers can share an x86 kernel regardless of their thread-local
     state. Fixed-width SVE would need to resolve its width per worker. */
  const int mode=OpenMX_HostSimdMode();
  openmx_exx_set_device_minimum_work(module,state->device.dispatch_work);
  openmx_exx_set_host_add_rows(module,mode==OPENMX_GRID_AVX2 || mode==OPENMX_GRID_AVX512
                                     ? openmx_core_host_add_rows(mode) : NULL);

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
  if (state->input.functional==OPENMX_EXACT_EXCHANGE_PBE0) {
    options.kernel = OPENMX_EXX_KERNEL_TRUNCATED_COULOMB;
    options.kernel_parameter = ExactExchange_Truncation_Radius(&state->input,lattice,comm);
  }
  else {
    options.kernel = OPENMX_EXX_KERNEL_SHORT_RANGE;
    options.kernel_parameter = state->input.omega;
  }
  options.abf_threshold = state->input.abf_threshold;
  options.v_cutoff = state->input.v_cutoff;
  double largest=0.0; /* the largest ABF radius: that of the PAOs */
  for (int s=0; s<species_count; s++) {
    if (largest<species[s].cutoff) largest = species[s].cutoff;
  }
  if (state->input.functional==OPENMX_EXACT_EXCHANGE_PBE0 && state->input.v_cutoff<0.0) {
    /* (P|v|Q) of the truncated interaction is 0 beyond r_P + r_Q + R_c, and
       the V list of two atoms reaches the larger of their ABF radii plus
       v_cutoff: R_c plus the largest radius keeps every block that is not 0 */
    options.v_cutoff = options.kernel_parameter+largest;
  }
  options.radial_intervals = state->input.radial_intervals;
  options.reciprocal_maximum = state->input.reciprocal_maximum;
  options.reciprocal_intervals = state->input.radial_intervals;
  if (state->input.functional==OPENMX_EXACT_EXCHANGE_PBE0) {
    /* The trapezoid rule in kappa gives (P|v|Q)_d of the truncated interaction
       without aliasing while r_P + r_Q + R_c + d < 2 pi/(step of kappa): the
       integrand of equation (12) is an entire function of that exponential
       type (section 13, item 19 of the guide). The V blocks reach d = (the
       larger ABF radius) + v_cutoff, so PBE0 takes more intervals than
       1DFFT.NumGridR where R_c + 3 r_max + v_cutoff needs them. */
    const double length=options.kernel_parameter+3.0*largest+options.v_cutoff;
    const double needed=floor(options.reciprocal_maximum*length/(2.0*3.14159265358979323846))+1.0;
    if (!(needed<(double)INT_MAX))
      ExactExchange_Abort(comm,"preparation","the truncated Coulomb interaction needs too many reciprocal intervals");
    if ((double)options.reciprocal_intervals<needed) options.reciprocal_intervals = (int)needed;
  }

  /* Every process makes the tables of the species and the neighbour lists,
     and the processes divide the pairs of sites of the T and V blocks
     (section 6.7). One process computes all pairs with the operations and
     in the order of the undivided preparation. The pairs of a process run
     on the threads of the main program (-nt): a pair writes its own blocks,
     so the blocks are the same for every number of threads. */
  int threads=1;
#ifdef _OPENMP
  threads = omp_get_max_threads();
#endif
  ExactExchangeProducts products={NULL,0.0,{0,0.0,{0,0,0},threads,NULL,0}};
  if (1<threads)
    products.host.loop = (OpenMXExactExchangeLoopProducts*)ExactExchange_Allocate(
        state,(size_t)threads,sizeof(OpenMXExactExchangeLoopProducts),comm,"preparation");
  const OpenMXExxDenseBackend pieces={NULL,&products,1<threads ? ExactExchange_Parallel_For : NULL,threads,NULL};
  char error[512];
  if (openmx_exx_prepare_part_pieces(module,&options,&geometry,rank,processes,&pieces,error,
                                     sizeof(error))!=0)
    ExactExchange_Abort(comm,"preparation",error);
  openmx_core_native_free(state->resources,products.host.loop);

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
  if (openmx_exx_orbital_count(module)!=orbitals || openmx_exx_atom_count(module)!=atom_count)
    ExactExchange_Abort(comm,"preparation","the orbitals of the structure differ from those of the solver");
  if (1<processes) ExactExchange_Add_Up_Blocks(state,module,comm);
  *prepared_options = options;
  return orbitals;
}

void Exact_Exchange_Prepare(OpenMXExactExchangeState *state, const OpenMXAtomicViews *atomic,
    int sites, MPI_Comm comm, int stdout_level, const char *scf_log)
{
  if (!Exact_Exchange_Enabled(state)) return;

  const int atom_count=state->structure.atom_count;
  const int *species_of_atom=state->structure.species_of_atom;
  double *const *positions=state->structure.positions;
  double (*lattice)[4]=state->structure.lattice;
  if (atom_count<1 || species_of_atom==NULL || positions==NULL || lattice==NULL)
    ExactExchange_Abort(comm,"preparation","the structure of the calculation is not bound");

  const double start=MPI_Wtime();
  int rank=0,processes=1;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&processes);
  state->stdout_level = stdout_level;
  if (rank==0) {
    state->scf_log = fopen(scf_log,"a");
    if (state->scf_log==NULL) ExactExchange_Abort(comm,"preparation","cannot open the SCF log");
  }

  /* The exchange matrix of an earlier structure is not carried over. */
  int gpu_enabled=ExactExchange_Configure_Dispatch(state,comm,rank);
  MPI_Reduce(&gpu_enabled,&state->gpu_dispatch_enabled,1,MPI_INT,MPI_MAX,0,comm);
  state->gpu_work = 0;
  if (rank==0) {
    const char *functional=state->input.functional==OPENMX_EXACT_EXCHANGE_HSE06 ? "HSE06" : "PBE0";
    if (state->gpu_dispatch_enabled)
      Exact_Exchange_Log(state,1,"<EXX>  Preparing %s exact exchange (GPU dispatch enabled, %s)\n",
                         functional,OPENMX_GPU_RUNTIME_NAME);
    else
      Exact_Exchange_Log(state,1,"<EXX>  Preparing %s exact exchange (CPU matrix products)\n",functional);
  }
  state->comm = comm;
  state->prepared = 0;
  state->folded = 0;
  state->spin_channels = 0;
  state->collecting = 0;
  state->exchange_ready = 0;
  ExactExchange_Forget_Band_Matrices(state);
  state->outer_iterations = 0;
  state->outer_converged = 0;
  state->residual_reference = -1.0;
  state->history_count = 0;
  state->energy = 0.0;
  state->change = 0.0;
  Exact_Exchange_Release_Outputs(state);
  ExactExchange_Release_Blocks(state);

  if (state->module==NULL) {
    state->module = openmx_exx_create();
    if (state->module==NULL) ExactExchange_Abort(comm,"preparation","out of memory");
  }

  OpenMXExxOptions options;
  const int orbitals=ExactExchange_Prepare_Module(state,state->module,atomic,comm,&options);
  state->orbitals = orbitals;
  state->sites = sites!=0;
  /* The site form folds the V neighbours by the period of a regular k mesh
     (section 6.4): the V neighbours whose cells agree modulo the period
     become one site with the sum of their V blocks. K(k) at the points of
     the mesh, E_x and its derivative keep their values within rounding,
     and a build goes through one V neighbour of every class. Every process
     folds the added blocks in the same way. The columns of the builds fold
     by the same period: the columns of D of the sites of one class differ
     by the sign of the mesh only, and the products with the T blocks of a
     centre take one site of every class. */
  const int *period=state->input.fold_period;
  const int folded=state->sites && 0<period[0] && 0<period[1] && 0<period[2];
  char error[512];
  if (folded && openmx_exx_fold_v_neighbours(state->module,period,error,sizeof(error))!=0)
    ExactExchange_Abort(comm,"preparation",error);
  if (folded && openmx_exx_fold_columns(state->module,state->input.fold_antiperiodic,error,sizeof(error))!=0)
    ExactExchange_Abort(comm,"preparation",error);
  state->folded = folded;
  if (state->sites) ExactExchange_List_Blocks(state,comm);
  state->prepared = 1;
  state->time_prepare += MPI_Wtime()-start;

  /* Every process solved the systems of equation (7) of its share. */
  OpenMXExxReport report={0,0,0,1.0};
  if (ExactExchange_Detailed(state)) {
    OpenMXExxReport mine;
    if (openmx_exx_report(state->module,&mine)!=0)
      ExactExchange_Abort(comm,"preparation","the local RI has no report");
    long counts[3]={mine.systems,mine.eigen_decompositions,mine.dropped},sums[3]={0,0,0};
    MPI_Reduce(counts,sums,3,MPI_LONG,MPI_SUM,0,comm);
    MPI_Reduce(&mine.smallest_pivot,&report.smallest_pivot,1,MPI_DOUBLE,MPI_MIN,0,comm);
    report.systems = sums[0];
    report.eigen_decompositions = sums[1];
    report.dropped = sums[2];
  }

  if (rank==0 && ExactExchange_Detailed(state)) {
    long abfs=0,c_sites=0,v_sites=0,folded_sites=0;
    for (int a=0; a<atom_count; a++) {
      abfs += openmx_exx_atom_abfs(state->module,a);
      c_sites += openmx_exx_c_neighbours(state->module,a);
      v_sites += openmx_exx_prepared_v_neighbours(state->module,a);
      folded_sites += openmx_exx_v_neighbours(state->module,a);
    }
    Exact_Exchange_Log(state,2,"<EXX>  %d atoms, %d orbitals, %ld ABFs, %ld C sites, %ld V sites (%.2f s)\n",
                       atom_count,orbitals,abfs,c_sites,v_sites,MPI_Wtime()-start);
    if (OpenMX_HostSimdPreferred("EXX"))
      Exact_Exchange_Log(state,2,"<EXX>  exchange products on the host (OPENMX_HOST_SIMD_PREFER)\n");
    if (folded) {
      const int *sign=state->input.fold_antiperiodic;
      Exact_Exchange_Log(state,2,"<EXX>  V sites folded by the period of the k mesh, %d x %d x %d cells: %ld of %ld\n",
                         period[0],period[1],period[2],folded_sites,v_sites);
      Exact_Exchange_Log(state,2,"<EXX>  columns of D folded by the same period with signs %c %c %c along a1, a2, a3\n",
                         sign[0] ? '-' : '+',sign[1] ? '-' : '+',sign[2] ? '-' : '+');
    }
    Exact_Exchange_Log(state,2,"<EXX>  Exact exchange mixing fraction %.4f\n",state->input.alpha);
    if (state->input.functional==OPENMX_EXACT_EXCHANGE_PBE0) {
      Exact_Exchange_Log(state,2,"<EXX>  truncated Coulomb interaction: R_c %.6f bohr; V sites up to the larger ABF radius plus %.6f bohr;"
                         " %d reciprocal intervals up to %.6f/bohr\n",
                         options.kernel_parameter,options.v_cutoff,options.reciprocal_intervals,options.reciprocal_maximum);
    }
    else {
      Exact_Exchange_Log(state,2,"<EXX>  HSE06 screening parameter %.4f bohr^-1\n",state->input.omega);
    }
    if (state->sites) {
      Exact_Exchange_Log(state,2,"<EXX>  site form: %ld blocks of the density and exchange matrices, %.1f MB each per spin channel\n",
                         state->block_count,(double)state->block_offset[state->block_count]*sizeof(double)/1048576.0);
    }
    Exact_Exchange_Log(state,2,"<EXX>  local RI: %ld systems, %ld by eigen-decomposition (%ld vectors left out), smallest pivot %.3e\n",
                       report.systems,report.eigen_decompositions,report.dropped,report.smallest_pivot);
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

int Exact_Exchange_Collecting(const OpenMXExactExchangeState *state)
{
  return state!=NULL && state->collecting;
}

void Exact_Exchange_Collect_Gamma(OpenMXExactExchangeState *state, int spin, int basis_count,
    int state_count, const double *occupation, const double *vectors)
{
  if (state==NULL || !state->collecting) return;

  const double start=MPI_Wtime();
  const int n=state->orbitals;
  if (state->sites) ExactExchange_Abort(state->comm,"density matrix","the solver of the Gamma point met the site form");
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

void Exact_Exchange_Collect_K(OpenMXExactExchangeState *state, int spin, double k1, double k2, double k3,
    double weight_sum, int basis_count, int state_count, int stride, const double *vectors,
    const double *amplitude)
{
  if (state==NULL || !state->collecting) return;

  const double start=MPI_Wtime();
  const int n=state->orbitals;
  if (!state->sites) ExactExchange_Abort(state->comm,"density matrix","the solver of a k mesh met the Gamma form");
  if (basis_count!=n || spin<0 || state->spin_channels<=spin || state_count<0 || stride<state_count || !(0.0<weight_sum))
    ExactExchange_Abort(state->comm,"density matrix","the vectors of the solver have an unexpected size or spin");

  /* A folded site form holds for the points of its mesh only: k_i n_i is
     an integer, or half an integer along an antiperiodic direction, up to
     the shift of the solver (Shift_K_Point). */
  if (state->folded) {
    const double k[3]={k1,k2,k3};
    for (int i=0; i<3; i++) {
      const double turns=k[i]*(double)state->input.fold_period[i]-(state->input.fold_antiperiodic[i] ? 0.5 : 0.0);
      if (1.0e-8<fabs(turns-floor(turns+0.5)))
        ExactExchange_Abort(state->comm,"density matrix","a k point of the solver is not a point of the folded mesh");
    }
  }

  /* The states beyond the last one with an amplitude do not contribute. */
  int occupied=state_count;
  if (amplitude!=NULL) {
    occupied = 0;
    for (int s=0; s<state_count; s++) if (amplitude[s]!=0.0) occupied = s+1;
  }
  if (0<occupied) {
    /* vectors is the matrix C of the states (rows) and the orbitals
       (columns) with the leading dimension stride. Q = C^H A^2 C has the
       elements sum_s a_s^2 conj(c_mu,s) c_nu,s. */
    const dcomplex *c=(const dcomplex*)vectors;
    dcomplex *q=(dcomplex*)openmx_core_native_malloc(state->resources,(size_t)n*(size_t)n,sizeof(dcomplex),0);
    dcomplex *scaled=NULL;
    if (q==NULL) ExactExchange_Abort(state->comm,"density matrix","out of memory");
    if (amplitude!=NULL) {
      scaled = (dcomplex*)openmx_core_native_malloc(state->resources,(size_t)occupied*(size_t)n,sizeof(dcomplex),0);
      if (scaled==NULL) ExactExchange_Abort(state->comm,"density matrix","out of memory");
      for (int basis=0; basis<n; basis++) {
        const dcomplex *column=c+(size_t)basis*(size_t)stride;
        dcomplex *target=scaled+(size_t)basis*(size_t)occupied;
        for (int s=0; s<occupied; s++) {
          const double factor=amplitude[s]*amplitude[s];
          target[s].r = factor*column[s].r;
          target[s].i = factor*column[s].i;
        }
      }
    }
    {
      char adjoint='C',plain='N';
      int rows=n,states=occupied,left=stride,right=amplitude!=NULL ? occupied : stride;
      dcomplex one={1.0,0.0},zero={0.0,0.0};
      F77_NAME(zgemm,ZGEMM)(&adjoint,&plain,&rows,&rows,&states,&one,(dcomplex*)c,&left,
                            amplitude!=NULL ? scaled : (dcomplex*)c,&right,&zero,q,&rows);
    }

    /* Equation (5): D_xy(R) += Re[e^{2 pi i k.R} Q_xy]/W for every block. A
       k point that stands for itself and for -k carries the weight 2, and
       the real part is the sum of the two contributions. */
    const double two_pi=6.283185307179586476925286766559;
    double *density=state->density[spin];
    for (long block=0; block<state->block_count; block++) {
      const int *cell=&state->block_cell[3*block];
      const double phase=two_pi*(k1*(double)cell[0]+k2*(double)cell[1]+k3*(double)cell[2]);
      const double co=cos(phase)/weight_sum,si=sin(phase)/weight_sum;
      const int row0=state->first_orbital[state->block_atoms[2*block]];
      const int column0=state->first_orbital[state->block_atoms[2*block+1]];
      const int rows=state->first_orbital[state->block_atoms[2*block]+1]-row0;
      const int columns=state->first_orbital[state->block_atoms[2*block+1]+1]-column0;
      double *target=density+state->block_offset[block];
      for (int j=0; j<columns; j++) {
        const dcomplex *source=q+(size_t)(column0+j)*(size_t)n+row0;
        for (int i=0; i<rows; i++) target[(size_t)i*columns+j] += co*source[i].r-si*source[i].i;
      }
    }
    openmx_core_native_free(state->resources,scaled);
    openmx_core_native_free(state->resources,q);
  }
  state->time_collect += MPI_Wtime()-start;
}

void Exact_Exchange_Collect_End(OpenMXExactExchangeState *state, MPI_Comm comm)
{
  if (state==NULL || !state->collecting) return;

  const double start=MPI_Wtime();
  for (int spin=0; spin<state->spin_channels; spin++) {
    ExactExchange_Sum(state->density[spin],state->elements,comm);
    ExactExchange_Symmetrize_Channel(state,state->density[spin],comm,"density matrix");
  }
  state->collecting = 0;
  state->time_collect += MPI_Wtime()-start;
}

double Exact_Exchange_Density_Deviation(const OpenMXExactExchangeState *state, int spin_channels,
    int local_atoms, const int *local_to_global, const int *neighbour_count, int *const *neighbour_atom,
    int *const *neighbour_cell, int *const *cell_index,
    const int *species_of_atom, const int *orbitals_of_species, const int *first_orbital,
    double *****density, MPI_Comm comm)
{
  const size_t n=(size_t)state->orbitals;
  double deviation=0.0,largest=0.0;
  if (state->sites && (neighbour_cell==NULL || cell_index==NULL))
    ExactExchange_Abort(comm,"density matrix","the comparison of the site form needs the cells of the neighbours");
  for (int spin=0; spin<spin_channels && spin<state->spin_channels; spin++) {
    const double *collected=state->density[spin];
    for (int local=1; local<=local_atoms; local++) {
      const int atom=local_to_global[local];
      const int rows=orbitals_of_species[species_of_atom[atom]];
      const size_t row0=(size_t)first_orbital[atom]-1;
      for (int neighbour=0; neighbour<=neighbour_count[atom]; neighbour++) {
        const int partner=neighbour_atom[atom][neighbour];
        const int columns=orbitals_of_species[species_of_atom[partner]];
        const size_t column0=(size_t)first_orbital[partner]-1;
        const double *block=NULL;
        if (state->sites) {
          /* the block of the two atoms and the cell of the neighbour; the
             solver counts the Cartesian components of a cell from 1 */
          const int *cell=cell_index[neighbour_cell[atom][neighbour]];
          const int shift[3]={cell[1],cell[2],cell[3]};
          const long index=openmx_exx_find_block(state->module,atom-1,partner-1,shift);
          if (index<0) continue; /* outside the list: the build does not read it */
          block = collected+state->block_offset[index];
        }
        for (int i=0; i<rows; i++) {
          for (int j=0; j<columns; j++) {
            const double mine=block!=NULL ? block[(size_t)i*columns+j] : collected[(row0+i)*n+column0+j];
            const double difference=fabs(density[spin][local][neighbour][i][j]-mine);
            if (!isfinite(difference)) deviation = INFINITY;
            else if (deviation<difference) deviation = difference;
          }
        }
      }
    }
  }
  MPI_Allreduce(&deviation,&largest,1,MPI_DOUBLE,MPI_MAX,comm);
  return largest;
}

/* The share of this process in a build (section 6.7): a contiguous range
   first <= P < last of the ABFs of the unit cell, counted atom after atom
   (openmx_exx_add_gamma_abfs). The contribution of a centre atom is a sum
   over its ABFs, so a range may end inside an atom, and processes share an
   atom when they are more than the atoms or when the atoms differ in size.
   openmx_exx_divide_abfs makes the ranges. It estimates the time of a range
   from the products of its ABFs, weighed by their speed, and from what
   every part of a centre atom repeats: the products with the V blocks read
   all T blocks of the V neighbours, and the part gathers the density
   matrix and adds its result. Of all divisions into contiguous ranges, it
   takes one with the shortest longest estimate. A process without work
   gets first = last. Every process computes the same division from the
   same numbers. The build and the weights of the forces divide the ABFs
   by the estimate of the half build (half): both go through the same
   groups of V neighbours. */
static void ExactExchange_Abfs(OpenMXExactExchangeState *state, const OpenMXExxState *module, int sites,
                               MPI_Comm comm, int half, long *first_abf, long *last_abf)
{
  int rank=0,processes=1;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&processes);
  long *boundaries=(long*)ExactExchange_Allocate(state,(size_t)processes+1,sizeof(long),comm,"exchange matrix");
  if (openmx_exx_divide_abfs(module,sites,half,processes,boundaries)!=0)
    ExactExchange_Abort(comm,"exchange matrix","the ABFs cannot be divided among the processes");
  *first_abf = boundaries[rank];
  *last_abf = boundaries[rank+1];
  openmx_core_native_free(state->resources,boundaries);
}

void Exact_Exchange_Build(OpenMXExactExchangeState *state, MPI_Comm comm)
{
  if (!Exact_Exchange_Enabled(state)) return;
  if (!state->prepared || state->spin_channels<1 || state->collecting)
    ExactExchange_Abort(comm,"exchange matrix","no density matrix is collected");

  const double start=MPI_Wtime();
  const int n=state->orbitals;
  int rank=0,processes=1;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&processes);

  long first,last;
  ExactExchange_Abfs(state,state->module,state->sites,comm,1,&first,&last);

  /* The threads of the main program (-nt) run the loops of the build and
     share its large products. A BLAS with threads of its own computes on
     the calling thread until modules/exx has added the last atom. */
  int threads=1;
#ifdef _OPENMP
  threads = omp_get_max_threads();
#endif
  const int blas_threads=1<threads ? Exact_Exchange_Host_Blas_Single() : 0;
  ExactExchangeProducts products=ExactExchange_Products(state,threads);
  if (1<threads)
    products.host.loop = (OpenMXExactExchangeLoopProducts*)ExactExchange_Allocate(
        state,(size_t)threads,sizeof(OpenMXExactExchangeLoopProducts),comm,"exchange matrix");
  /* A whole call accepted by CUDA or HIP uses this table for its products
     and data movement (openmx_exact_exchange_device.h). */
  OpenMXExxDevice device_table;
  const OpenMXExxDevice *device_calls=Exact_Exchange_Device_Table(products.device,&device_table);
  const OpenMXExxDenseBackend backend={ExactExchange_Dgemm,&products,1<threads ? ExactExchange_Parallel_For : NULL,
                                       threads,device_calls};
  const long long device_before=state->device.device_products;
  const long long declined_before=state->device.declined_products;
  const long long calls_before=state->device.device_calls;
  const long long declined_calls_before=state->device.declined_calls;
  const long long call_products_before=state->device.call_products;
  const int disabled_before=state->device.disabled;
  double call_seconds=0.0,share_seconds=0.0;
  ExactExchange_Forget_Band_Matrices(state);
  for (int spin=0; spin<state->spin_channels; spin++) {
    /* The matrix of the build before stays for delta; without one it is 0. */
    double *swap=state->previous[spin];
    state->previous[spin] = state->exchange[spin];
    state->exchange[spin] = swap;
    if (!state->exchange_ready) memset(state->previous[spin],0,sizeof(double)*state->elements);
    memset(state->exchange[spin],0,sizeof(double)*state->elements);

    if (state->input.alpha!=0.0) {
      char error[512];
      if (first<last) {
        const double call_start=MPI_Wtime();
        const int failed=state->sites
          ? openmx_exx_add_blocks_half_abfs(state->module,first,last,-state->input.alpha,state->density[spin],
                                            state->exchange[spin],&backend,error,sizeof(error))
          : openmx_exx_add_gamma_half_abfs(state->module,first,last,-state->input.alpha,state->density[spin],
                                           state->exchange[spin],&backend,error,sizeof(error));
        if (Exact_Exchange_Device_Call_End(&state->device)) call_seconds += MPI_Wtime()-call_start;
        share_seconds += MPI_Wtime()-call_start;
        if (failed!=0) ExactExchange_Abort(comm,"exchange matrix",error);
      }
      ExactExchange_Sum(state->exchange[spin],state->elements,comm);
      /* The sum over the ABFs of all processes is X of the half build, and
         K = X + X^T: K(k) is Hermitian to the last bit. */
      ExactExchange_Mirror_Channel(state,state->exchange[spin],comm,"exchange matrix");
    }
  }
  openmx_core_native_free(state->resources,products.host.loop);
  Exact_Exchange_Host_Blas_Restore(blas_threads);

  /* One spin channel stands for both without spin polarization. Process 0
     decides for all: the numbers that steer the outer iteration must not
     differ by a rounding error between the processes. */
  const double degeneracy=state->spin_channels==1 ? 2.0 : 1.0;
  double results[2]={0.0,0.0};
  if (rank==0) {
    for (int spin=0; spin<state->spin_channels; spin++) {
      results[0] += degeneracy*openmx_exx_energy_gamma(state->elements,state->density[spin],state->exchange[spin]);
      results[1] += degeneracy*openmx_exx_change_gamma(state->elements,state->density[spin],state->exchange[spin],
                                                       state->previous[spin]);
    }
    if (!isfinite(results[0]) || !isfinite(results[1]))
      ExactExchange_Abort(comm,"exchange matrix","nonfinite exchange energy or convergence measure");
  }
  MPI_Bcast(results,2,MPI_DOUBLE,0,comm);
  state->energy = results[0];
  state->change = results[1];
  state->exchange_ready = 1;
  state->outer_iterations++;
  state->outer_converged = 0;
  state->residual_reference = -1.0;
  state->time_build += MPI_Wtime()-start;
  state->time_device += products.device_seconds;
  state->time_device_calls += call_seconds;

  if (device_before<state->device.device_products || calls_before<state->device.device_calls)
    state->gpu_work |= EXX_GPU_PRODUCTS;
  ExactExchange_Report_Device_Failure(state,rank,disabled_before);
  const int detailed=ExactExchange_Detailed(state);
  if (rank==0 && detailed
      && (calls_before<state->device.device_calls || declined_calls_before<state->device.declined_calls))
    Exact_Exchange_Log(state,2,"<EXX>  device calls of process 0: %lld in %.2f s with %lld matrix products, at most %.1f MB for one call; left to the host for lack of device memory: %lld\n",
                       state->device.device_calls-calls_before,call_seconds,state->device.call_products-call_products_before,
                       (double)state->device.largest_call_bytes/1048576.0,state->device.declined_calls-declined_calls_before);
  if (rank==0 && detailed && 0<products.host.count) {
    Exact_Exchange_Log(state,2,"<EXX>  matrix products of process 0: %lld, the largest %d x %d with inner dimension %d\n",
                       products.host.count,products.host.largest[0],products.host.largest[1],products.host.largest[2]);
    if (device_before<state->device.device_products || declined_before<state->device.declined_products)
      Exact_Exchange_Log(state,2,"<EXX>  on the device: %lld in %.2f s, at most %.1f MB for one product; left to the host for lack of device memory: %lld\n",
                         state->device.device_products-device_before,products.device_seconds,
                         (double)state->device.largest_bytes/1048576.0,state->device.declined_products-declined_before);
  }

  /* The time that the processes spent on their shares: the division is even
     when the two numbers are close. */
  double shares[2]={share_seconds,-share_seconds},extremes[2]={0.0,0.0};
  if (detailed && 1<processes) MPI_Reduce(shares,extremes,2,MPI_DOUBLE,MPI_MAX,0,comm);

  if (rank==0 && detailed) {
    const long abfs=openmx_exx_abf_count(state->module);
    if (first<last) {
      Exact_Exchange_Log(state,2,"<EXX>  exchange matrix of %d orbitals built in %.2f s (ABFs %ld to %ld of %ld on process 0)\n",
                         n,MPI_Wtime()-start,first+1,last,abfs);
    }
    else {
      /* no ABF has its middle in the part of this process */
      Exact_Exchange_Log(state,2,"<EXX>  exchange matrix of %d orbitals built in %.2f s (none of the %ld ABFs on process 0)\n",
                         n,MPI_Wtime()-start,abfs);
    }
    if (1<processes) {
      Exact_Exchange_Log(state,2,"<EXX>  shares of the %d processes in the build: %.2f s to %.2f s\n",
                         processes,-extremes[1],extremes[0]);
    }
  }
}

const double *Exact_Exchange_Gradient(OpenMXExactExchangeState *state, MPI_Comm comm)
{
  if (!Exact_Exchange_Active(state) || !state->prepared || state->spin_channels<1 || state->collecting)
    return NULL;

  const double start=MPI_Wtime();
  const int atoms=openmx_exx_atom_count(state->module);
  int rank=0,processes=1;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&processes);
  double weight_seconds=0.0,call_seconds=0.0,device_seconds=0.0;
  /* what the weights of process 0 did on the device; zero without */
  ExactExchangeDeviceCounts weight_use;
  memset(&weight_use,0,sizeof(weight_use));
  if (state->gradient_atoms!=atoms) {
    openmx_core_native_free(state->resources,state->gradient);
    state->gradient = (double*)ExactExchange_Allocate(state,3*(size_t)atoms,sizeof(double),comm,"forces");
    state->gradient_atoms = atoms;
  }
  memset(state->gradient,0,sizeof(double)*3*(size_t)atoms);

  if (state->input.alpha!=0.0) {
    /* E_x depends on the positions through the T and V blocks, and the
       derivative takes two steps (section 6.7). The first one makes the
       weights of the blocks, the derivatives of E_x with respect to their
       elements, from the density matrices. They come in the half form: as
       in a half build, a centre atom goes through about half of its V
       neighbours, and a V block that it goes through also stands for the
       mirrored block of the other atom. Every process computes the weights
       of its ABFs, its share of a half build, and the arrays are added up.
       The W part of a group adds to the T blocks of other atoms, so the
       sum has rounding. */
    long first_abf=0,last_abf=0;
    ExactExchange_Abfs(state,state->module,state->sites,comm,1,&first_abf,&last_abf);
    double **weights=(double**)ExactExchange_Allocate(state,(size_t)atoms,sizeof(double*),comm,"forces");
    for (int a=0; a<atoms; a++)
      weights[a] = (double*)ExactExchange_Allocate(state,openmx_exx_atom_table_elements(state->module,a),
                                                   sizeof(double),comm,"forces");
    int threads=1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const int blas_threads=1<threads ? Exact_Exchange_Host_Blas_Single() : 0;
    /* The device of the builds computes the weights too: one whole call for
       every spin channel on CUDA and HIP (Exact_Exchange_Device_Table),
       and the large products one at a time when a call does not fit, as in
       a build. The counts of the builds stay as they were. */
    const ExactExchangeDeviceCounts counts=ExactExchange_Device_Counts(&state->device);
    const int disabled_before=state->device.disabled;
    state->device.largest_bytes = 0;
    state->device.largest_call_bytes = 0;
    ExactExchangeProducts products=ExactExchange_Products(state,threads);
    if (1<threads)
      products.host.loop = (OpenMXExactExchangeLoopProducts*)ExactExchange_Allocate(
          state,(size_t)threads,sizeof(OpenMXExactExchangeLoopProducts),comm,"forces");
    OpenMXExxDevice device_table;
    const OpenMXExxDevice *device_calls=Exact_Exchange_Device_Table(products.device,&device_table);
    const OpenMXExxDenseBackend backend={ExactExchange_Dgemm,&products,1<threads ? ExactExchange_Parallel_For : NULL,
                                         threads,device_calls};
    /* E_x has the degeneracy of the spin channels as a factor (one channel
       stands for both without spin polarization), and so has its
       derivative. */
    const double degeneracy=state->spin_channels==1 ? 2.0 : 1.0;
    char error[512];
    for (int spin=0; spin<state->spin_channels && first_abf<last_abf; spin++) {
      const double call_start=MPI_Wtime();
      const int failed=state->sites
        ? openmx_exx_add_block_weights_blocks_half_abfs(state->module,first_abf,last_abf,
                                                        -state->input.alpha*degeneracy,state->density[spin],weights,
                                                        &backend,error,sizeof(error))
        : openmx_exx_add_block_weights_gamma_half_abfs(state->module,first_abf,last_abf,
                                                       -state->input.alpha*degeneracy,state->density[spin],weights,
                                                       &backend,error,sizeof(error));
      if (Exact_Exchange_Device_Call_End(&state->device)) call_seconds += MPI_Wtime()-call_start;
      if (failed!=0) ExactExchange_Abort(comm,"forces",error);
    }
    Exact_Exchange_Host_Blas_Restore(blas_threads);
    device_seconds = products.device_seconds;
    const ExactExchangeDeviceCounts after=ExactExchange_Device_Counts(&state->device);
    weight_use.device_products = after.device_products-counts.device_products;
    weight_use.declined_products = after.declined_products-counts.declined_products;
    weight_use.largest_bytes = after.largest_bytes;
    weight_use.device_calls = after.device_calls-counts.device_calls;
    weight_use.declined_calls = after.declined_calls-counts.declined_calls;
    weight_use.call_products = after.call_products-counts.call_products;
    weight_use.largest_call_bytes = after.largest_call_bytes;
    if (weight_use.device_products>0 || weight_use.device_calls>0) state->gpu_work |= EXX_GPU_PRODUCTS;
    ExactExchange_Device_Counts_Back(&state->device,&counts);
    ExactExchange_Report_Device_Failure(state,rank,disabled_before);
    weight_seconds = MPI_Wtime()-start;
    if (1<processes) {
      for (int a=0; a<atoms; a++)
        ExactExchange_Sum(weights[a],openmx_exx_atom_table_elements(state->module,a),comm);
    }

    /* The second step goes through the pairs of sites, which the processes
       divide: a pair is visited once, with the weights of the blocks of
       both of its atoms. The pairs of a process run on the threads of the
       weights; the vectors of the pairs are added in their order, so the
       derivative is the same for every number of threads. */
    if (openmx_exx_add_position_gradient_pairs_pieces(state->module,(const double *const*)weights,rank,processes,
                                                      &backend,state->gradient,error,sizeof(error))!=0)
      ExactExchange_Abort(comm,"forces",error);
    openmx_core_native_free(state->resources,products.host.loop);
    for (int a=0; a<atoms; a++) openmx_core_native_free(state->resources,weights[a]);
    openmx_core_native_free(state->resources,weights);
    ExactExchange_Sum(state->gradient,3*(size_t)atoms,comm);
  }
  if (rank==0 && ExactExchange_Detailed(state)) {
    /* the forces are the negative of the derivative; their sum vanishes */
    double largest=0.0,sum[3]={0.0,0.0,0.0};
    for (int a=0; a<atoms; a++) {
      for (int x=0; x<3; x++) {
        const double value=state->gradient[3*(size_t)a+(size_t)x];
        if (largest<fabs(value)) largest = fabs(value);
        sum[x] -= value;
      }
    }
    Exact_Exchange_Log(state,2,"<EXX>  forces of the exact exchange in %.2f s: largest component %.3e, sum %.1e %.1e %.1e (Hartree/bohr)\n",
                       MPI_Wtime()-start,largest,sum[0],sum[1],sum[2]);
    Exact_Exchange_Log(state,2,"<EXX>  of that, %.2f s for the weights of the T and V blocks on process 0\n",weight_seconds);
    if (0<weight_use.device_calls || 0<weight_use.declined_calls)
      Exact_Exchange_Log(state,2,"<EXX>  the weights on the device of process 0: %lld calls in %.2f s with %lld matrix products, at most %.1f MB for one call; left to the host for lack of device memory: %lld\n",
                         weight_use.device_calls,call_seconds,weight_use.call_products,
                         (double)weight_use.largest_call_bytes/1048576.0,weight_use.declined_calls);
    if (0<weight_use.device_products || 0<weight_use.declined_products)
      Exact_Exchange_Log(state,2,"<EXX>  matrix products of the weights on the device of process 0: %lld in %.2f s, at most %.1f MB for one product; left to the host for lack of device memory: %lld\n",
                         weight_use.device_products,device_seconds,(double)weight_use.largest_bytes/1048576.0,
                         weight_use.declined_products);
  }
  Exact_Exchange_End_Device_Phase(state);
  return state->gradient;
}

int Exact_Exchange_Outer_Continue(OpenMXExactExchangeState *state, int loose)
{
  /* The density of an inner SCF that stopped on a larger criterion than
     scf.criterion has not answered its matrix within scf.criterion: the
     outer iteration does not end on it. */
  if (2<=state->outer_iterations && fabs(state->change)<state->input.outer_criterion && !loose) {
    state->outer_converged = 1;
    return 0;
  }
  return state->outer_iterations<state->input.outer_max_iterations;
}

/* Room for room+1 builds of every spin channel of the present matrices. */
static void ExactExchange_Reserve_History(OpenMXExactExchangeState *state, int room, MPI_Comm comm)
{
  if (state->history_built!=NULL && state->history_room==room && state->spin_channels<=state->history_channels)
    return;
  ExactExchange_Release_History(state);
  const size_t slots=2*((size_t)room+1);
  state->history_built = (double**)ExactExchange_Allocate(state,slots,sizeof(double*),comm,"extrapolation");
  state->history_residual = (double**)ExactExchange_Allocate(state,slots,sizeof(double*),comm,"extrapolation");
  state->history_room = room;
  state->history_channels = state->spin_channels;
  for (int slot=0; slot<=room; slot++) {
    for (int spin=0; spin<state->spin_channels; spin++) {
      state->history_built[2*slot+spin] =
        (double*)ExactExchange_Allocate(state,state->elements,sizeof(double),comm,"extrapolation");
      state->history_residual[2*slot+spin] =
        (double*)ExactExchange_Allocate(state,state->elements,sizeof(double),comm,"extrapolation");
    }
  }
}

/* Solves the d x d system a g = b in place by elimination with partial
   pivoting. Returns 0 when a pivot is not above tiny times the largest
   diagonal element: the residuals of the builds are then nearly dependent. */
static int ExactExchange_Solve_Small(int d, double *a, double *b)
{
  double largest=0.0;
  for (int i=0; i<d; i++) if (largest<fabs(a[i*d+i])) largest = fabs(a[i*d+i]);
  if (!(0.0<largest)) return 0;
  for (int k=0; k<d; k++) {
    int pivot=k;
    for (int i=k+1; i<d; i++) if (fabs(a[pivot*d+k])<fabs(a[i*d+k])) pivot = i;
    if (!(1.0e-12*largest<fabs(a[pivot*d+k]))) return 0;
    if (pivot!=k) {
      for (int j=0; j<d; j++) {
        const double t=a[k*d+j];
        a[k*d+j] = a[pivot*d+j];
        a[pivot*d+j] = t;
      }
      const double t=b[k];
      b[k] = b[pivot];
      b[pivot] = t;
    }
    for (int i=k+1; i<d; i++) {
      const double f=a[i*d+k]/a[k*d+k];
      for (int j=k; j<d; j++) a[i*d+j] -= f*a[k*d+j];
      b[i] -= f*b[k];
    }
  }
  for (int k=d-1; 0<=k; k--) {
    double sum=b[k];
    for (int j=k+1; j<d; j++) sum -= a[k*d+j]*b[j];
    b[k] = sum/a[k*d+k];
  }
  for (int k=0; k<d; k++) if (!isfinite(b[k])) return 0;
  return 1;
}

int Exact_Exchange_Extrapolate(OpenMXExactExchangeState *state, MPI_Comm comm)
{
  const int room=state!=NULL ? state->input.outer_extrapolation : 0;
  if (!Exact_Exchange_Active(state) || room<=0 || state->outer_iterations<2) return 0;
  enum { largest_room=8 };          /* the largest value that the input allows */
  if (!state->prepared || state->spin_channels<1 || state->collecting)
    ExactExchange_Abort(comm,"extrapolation","no exchange matrix is built");
  if (largest_room<room) ExactExchange_Abort(comm,"extrapolation","scf.exx.outer.extrapolation is larger than 8");

  const double start=MPI_Wtime();
  int rank=0;
  MPI_Comm_rank(comm,&rank);
  ExactExchange_Reserve_History(state,room,comm);
  const int channels=state->spin_channels;
  const size_t elements=state->elements;

  /* The new build in the slot after the stored ones: X_n and F_n = X_n - Y_{n-1}. */
  const int stored=state->history_count;
  for (int spin=0; spin<channels; spin++) {
    double *built=state->history_built[2*stored+spin];
    double *residual=state->history_residual[2*stored+spin];
    const double *x=state->exchange[spin],*y=state->previous[spin];
    memcpy(built,x,sizeof(double)*elements);
    for (size_t e=0; e<elements; e++) residual[e] = x[e]-y[e];
  }

  /* The differences between the slots first-1+j and first+j, j = 1..d, with
     the new build in slot stored. Process 0 forms and solves the normal
     equations and sends the coefficients; a nearly dependent set loses its
     oldest difference. */
  int d=stored<room ? stored : room;
  double coefficient[largest_room];
  if (rank==0) {
    while (0<d) {
      const int first=stored-d+1;
      double a[largest_room*largest_room],b[largest_room];
      for (int j=0; j<d; j++) {
        b[j] = 0.0;
        for (int k=0; k<d; k++) a[j*d+k] = 0.0;
      }
      for (int spin=0; spin<channels; spin++) {
        const double *f=state->history_residual[2*stored+spin];
        for (int j=0; j<d; j++) {
          const double *fj1=state->history_residual[2*(first+j)+spin];
          const double *fj0=state->history_residual[2*(first+j-1)+spin];
          double sum=0.0;
          for (size_t e=0; e<elements; e++) sum += (fj1[e]-fj0[e])*f[e];
          b[j] += sum;
          for (int k=0; k<=j; k++) {
            const double *fk1=state->history_residual[2*(first+k)+spin];
            const double *fk0=state->history_residual[2*(first+k-1)+spin];
            double product=0.0;
            for (size_t e=0; e<elements; e++) product += (fj1[e]-fj0[e])*(fk1[e]-fk0[e]);
            a[j*d+k] += product;
          }
        }
      }
      for (int j=0; j<d; j++)
        for (int k=j+1; k<d; k++) a[j*d+k] = a[k*d+j];
      if (ExactExchange_Solve_Small(d,a,b)) {
        for (int j=0; j<d; j++) coefficient[j] = b[j];
        break;
      }
      d--;
    }
  }
  MPI_Bcast(&d,1,MPI_INT,0,comm);
  if (0<d) MPI_Bcast(coefficient,d,MPI_DOUBLE,0,comm);

  /* Y_n = X_n - sum_j g_j (X_{first+j} - X_{first+j-1}) */
  if (0<d) {
    const int first=stored-d+1;
    for (int spin=0; spin<channels; spin++) {
      double *target=state->exchange[spin];
      for (int j=0; j<d; j++) {
        const double g=coefficient[j];
        const double *x1=state->history_built[2*(first+j)+spin];
        const double *x0=state->history_built[2*(first+j-1)+spin];
        for (size_t e=0; e<elements; e++) target[e] -= g*(x1[e]-x0[e]);
      }
    }
    /* K(k) of the solvers comes from the new matrix */
    ExactExchange_Forget_Band_Matrices(state);
  }

  /* The new build joins the stored ones; with all slots full the oldest
     leaves, and its arrays take the place after the others. */
  if (stored<room) {
    state->history_count = stored+1;
  }
  else {
    for (int spin=0; spin<channels; spin++) {
      double *built=state->history_built[spin],*residual=state->history_residual[spin];
      for (int slot=0; slot<room; slot++) {
        state->history_built[2*slot+spin] = state->history_built[2*(slot+1)+spin];
        state->history_residual[2*slot+spin] = state->history_residual[2*(slot+1)+spin];
      }
      state->history_built[2*room+spin] = built;
      state->history_residual[2*room+spin] = residual;
    }
    state->history_count = room;
  }

  if (rank==0 && ExactExchange_Detailed(state)) {
    if (0<d) {
      Exact_Exchange_Log(state,2,"<EXX>  the next inner SCF takes K from %d builds (Anderson, scf.exx.outer.extrapolation): coefficients",d+1);
      for (int j=0; j<d; j++) Exact_Exchange_Log(state,2," %.4f",coefficient[j]);
      Exact_Exchange_Log(state,2,"\n");
    }
    else if (stored==0) {
      Exact_Exchange_Log(state,2,"<EXX>  the next inner SCF takes K of the build: no earlier build for the extrapolation\n");
    }
    else {
      Exact_Exchange_Log(state,2,"<EXX>  the next inner SCF takes K of the build: the residuals of the builds are nearly dependent\n");
    }
    Exact_Exchange_Log(state,2,"<EXX>  extrapolation in %.2f s\n",MPI_Wtime()-start);
  }
  return 0<d;
}

int Exact_Exchange_Mixing_Offset(int mixing_type, int start_pulay, int local_step)
{
  int first;  /* step count of the first mixing after the build */
  int before; /* 1 when the mixing of a step comes before its solution */
  int offset;

  switch (mixing_type) {
  case 1: /* RMM-DIIS, */
  case 2: /* GR-Pulay, */
  case 6: /* RMM-ADIIS: the history starts at the count start_pulay, the
             first step with the new matrix. The counts before mix simply,
             with the weight of OPENMX_MIXING_DENSITY_RATIO, which the
             converged residual of the build (below the bound 1e-9 of that
             rule) and the large first residual of the new matrix take down
             to scf.Min.Mixing.Weight: a simple mixing with the new matrix
             would hardly move the density. */
    first = start_pulay-1;
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

int Exact_Exchange_Inner_Stop_Allowed(OpenMXExactExchangeState *state, int mixing_type, int step,
    double residual_norm, double criterion)
{
  /* Simple and Kerker mixing take their weight from the policy
     OPENMX_MIXING_DENSITY_RATIO (core/src/mixing_policy.cpp). It divides by
     the change of the squared residual norm and bounds that change from
     below by 1e-9, so a squared norm under 1e-9 can only reduce the weight,
     down to the lower limit of the input. */
  const double plain_mixing_floor = sqrt(1.0e-9);
  double bound;
  if (!Exact_Exchange_Active(state)) return 1;
  if (step<2) return 0;
  if (step==2 || state->residual_reference<0.0) {
    state->residual_reference = residual_norm;
    return 0;
  }
  bound = 0.1*state->residual_reference;
  if (bound<criterion) bound = criterion;
  if ((mixing_type==0 || mixing_type==3) && bound<plain_mixing_floor) bound = plain_mixing_floor;
  return residual_norm<=bound;
}

double Exact_Exchange_Inner_Criterion(const OpenMXExactExchangeState *state, double criterion)
{
  /* After a build that meets scf.exx.outer.criterion, only an inner SCF
     that stopped on scf.criterion lets the next build end the outer
     iteration (Exact_Exchange_Outer_Continue): that inner SCF keeps it. */
  if (Exact_Exchange_Active(state) && 0.0<state->input.inner_ratio
      && state->input.outer_criterion<=fabs(state->change)) {
    const double loose=state->input.inner_ratio*fabs(state->change);
    if (criterion<loose) return loose;
  }
  return criterion;
}

void Exact_Exchange_Add_Cluster_Hs(OpenMXExactExchangeState *state, int spin, int n, double *Hs,
    int na_rows, int na_cols, int nblk, int np_rows, int np_cols, int my_prow, int my_pcol)
{
  if (!Exact_Exchange_Active(state)) return;
  if (state->sites) ExactExchange_Abort(state->comm,"Hamiltonian","the solver of the Gamma point met the site form");
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

/* Equation (14): adds K(k) = sum_R e^{2 pi i k.R} K(R) of one spin channel
   to the local elements of target, with the rows on the orbitals of the
   cell at the origin, as the solver builds H(k). local_row and
   local_column give the local row and column of every orbital, -1 for
   those of other processes; a block of an atom pair without a local row or
   column adds nothing and is skipped. Every element receives its terms in
   the order of the blocks. */
static void ExactExchange_Band_Sum(const OpenMXExactExchangeState *state, int spin, double k1, double k2,
    double k3, const int *local_row, const int *local_column, const char *row_atoms, const char *column_atoms,
    int na_rows, dcomplex *target)
{
  const double two_pi=6.283185307179586476925286766559;
  const double *exchange=state->exchange[spin];
  for (long block=0; block<state->block_count; block++) {
    const int x=state->block_atoms[2*block],y=state->block_atoms[2*block+1];
    if (!row_atoms[x] || !column_atoms[y]) continue;
    const int *cell=&state->block_cell[3*block];
    const double phase=two_pi*(k1*(double)cell[0]+k2*(double)cell[1]+k3*(double)cell[2]);
    const double co=cos(phase),si=sin(phase);
    const int row0=state->first_orbital[x];
    const int column0=state->first_orbital[y];
    const int rows=state->first_orbital[x+1]-row0;
    const int columns=state->first_orbital[y+1]-column0;
    const double *source=exchange+state->block_offset[block];
    for (int j=0; j<columns; j++) {
      const int column=local_column[column0+j];
      if (column<0) continue;
      dcomplex *values=target+(size_t)column*(size_t)na_rows;
      for (int i=0; i<rows; i++) {
        const int row=local_row[row0+i];
        if (row<0) continue;
        const double value=source[(size_t)i*columns+j];
        values[row].r += value*co;
        values[row].i += value*si;
      }
    }
  }
}

/* The entry of K(k) for this spin channel, k point and local array, or NULL. */
static ExactExchangeBandMatrix *ExactExchange_Band_Kept(OpenMXExactExchangeState *state, int spin, double k1,
    double k2, double k3, int n, int na_rows, int na_cols, int nblk, int np_rows, int np_cols, int my_prow,
    int my_pcol)
{
  for (int e=0; e<state->band_matrix_count; e++) {
    ExactExchangeBandMatrix *entry=&state->band_matrices[e];
    if (entry->spin==spin && entry->k[0]==k1 && entry->k[1]==k2 && entry->k[2]==k3 && entry->n==n
        && entry->na_rows==na_rows && entry->na_cols==na_cols && entry->nblk==nblk && entry->np_rows==np_rows
        && entry->np_cols==np_cols && entry->my_prow==my_prow && entry->my_pcol==my_pcol)
      return entry;
  }
  return NULL;
}

/* A new entry with zeros in its values, or NULL when the entries would
   hold more than the two exchange matrices of every spin channel (K and the
   K of the build before) or the memory is short. */
static ExactExchangeBandMatrix *ExactExchange_Band_Keep(OpenMXExactExchangeState *state, int spin, double k1,
    double k2, double k3, int n, int na_rows, int na_cols, int nblk, int np_rows, int np_cols, int my_prow,
    int my_pcol)
{
  const size_t bytes=(size_t)na_rows*(size_t)na_cols*sizeof(dcomplex);
  const size_t limit=2*(size_t)state->spin_channels*state->elements*sizeof(double);
  if (limit<state->band_matrix_bytes || limit-state->band_matrix_bytes<bytes) return NULL;
  if (state->band_matrix_count==state->band_matrix_room) {
    const int room=state->band_matrix_room<8 ? 8 : 2*state->band_matrix_room;
    ExactExchangeBandMatrix *grown=(ExactExchangeBandMatrix*)openmx_core_native_realloc(
        state->resources,state->band_matrices,(size_t)room*sizeof(ExactExchangeBandMatrix));
    if (grown==NULL) return NULL;
    state->band_matrices = grown;
    state->band_matrix_room = room;
  }
  dcomplex *values=(dcomplex*)openmx_core_native_malloc(state->resources,(size_t)na_rows*(size_t)na_cols,
                                                        sizeof(dcomplex),1);
  if (values==NULL) return NULL;
  ExactExchangeBandMatrix *entry=&state->band_matrices[state->band_matrix_count++];
  entry->spin = spin;
  entry->n = n;
  entry->na_rows = na_rows;
  entry->na_cols = na_cols;
  entry->nblk = nblk;
  entry->np_rows = np_rows;
  entry->np_cols = np_cols;
  entry->my_prow = my_prow;
  entry->my_pcol = my_pcol;
  entry->k[0] = k1;
  entry->k[1] = k2;
  entry->k[2] = k3;
  entry->values = values;
  state->band_matrix_bytes += bytes;
  return entry;
}

void Exact_Exchange_Add_Band_Hs(OpenMXExactExchangeState *state, int spin, double k1, double k2, double k3,
    int n, double *Hs, int na_rows, int na_cols, int nblk, int np_rows, int np_cols, int my_prow, int my_pcol)
{
  if (!Exact_Exchange_Active(state)) return;
  if (!state->sites) ExactExchange_Abort(state->comm,"Hamiltonian","the solver of a k mesh met the Gamma form");
  if (n!=state->orbitals || spin<0 || state->spin_channels<=spin)
    ExactExchange_Abort(state->comm,"Hamiltonian","the matrix of the solver has an unexpected size or spin");

  const double start=MPI_Wtime();
  dcomplex *matrix=(dcomplex*)Hs;

  /* The local rows and columns of the n x n matrix come first in the local
     array: the row of the matrix grows with the local row. */
  int rows=0,columns=0;
  for (int i=0; i<na_rows; i++)
    if (np_rows*nblk*(i/nblk)+i%nblk+((np_rows+my_prow)%np_rows)*nblk<n) rows = i+1;
  for (int j=0; j<na_cols; j++)
    if (np_cols*nblk*(j/nblk)+j%nblk+((np_cols+my_pcol)%np_cols)*nblk<n) columns = j+1;

  /* K is the same until the next build, and so is K(k): the first addition
     at a k point sums equation (14) into an entry, and the later ones add
     the entry. Without room for an entry the sum goes to an array that is
     dropped afterwards, so that K(k) has the same rounding at every k
     point. */
  const dcomplex *values=NULL;
  dcomplex *scratch=NULL;
  ExactExchangeBandMatrix *entry=ExactExchange_Band_Kept(state,spin,k1,k2,k3,n,na_rows,na_cols,nblk,np_rows,
                                                         np_cols,my_prow,my_pcol);
  if (entry!=NULL) {
    values = entry->values;
    state->band_reuses++;
  }
  else {
    /* Where the rows and columns of the n x n matrix are in the local array
       of this process, -1 for those of other processes, and the atoms with
       a local row or column. */
    const int atoms=openmx_exx_atom_count(state->module);
    int *local=(int*)openmx_core_native_malloc(state->resources,2*(size_t)n,sizeof(int),0);
    char *reached=(char*)openmx_core_native_malloc(state->resources,2*(size_t)atoms,sizeof(char),1);
    if (local==NULL || reached==NULL) ExactExchange_Abort(state->comm,"Hamiltonian","out of memory");
    int *local_row=local,*local_column=local+n;
    char *row_atoms=reached,*column_atoms=reached+atoms;
    for (int g=0; g<2*n; g++) local[g] = -1;
    for (int i=0; i<na_rows; i++) {
      const int row=np_rows*nblk*(i/nblk)+i%nblk+((np_rows+my_prow)%np_rows)*nblk;
      if (row<n) local_row[row] = i;
    }
    for (int j=0; j<na_cols; j++) {
      const int column=np_cols*nblk*(j/nblk)+j%nblk+((np_cols+my_pcol)%np_cols)*nblk;
      if (column<n) local_column[column] = j;
    }
    for (int a=0; a<atoms; a++)
      for (int g=state->first_orbital[a]; g<state->first_orbital[a+1]; g++) {
        if (0<=local_row[g]) row_atoms[a] = 1;
        if (0<=local_column[g]) column_atoms[a] = 1;
      }

    entry = ExactExchange_Band_Keep(state,spin,k1,k2,k3,n,na_rows,na_cols,nblk,np_rows,np_cols,my_prow,my_pcol);
    dcomplex *target=entry!=NULL ? entry->values : NULL;
    if (target==NULL)
      target = scratch = (dcomplex*)openmx_core_native_malloc(state->resources,(size_t)na_rows*(size_t)na_cols,
                                                              sizeof(dcomplex),1);
    /* Without memory for that array the terms go to Hs one by one. */
    ExactExchange_Band_Sum(state,spin,k1,k2,k3,local_row,local_column,row_atoms,column_atoms,na_rows,
                           target!=NULL ? target : matrix);
    values = target;
    state->band_sums++;
    openmx_core_native_free(state->resources,reached);
    openmx_core_native_free(state->resources,local);
  }

  if (values!=NULL) {
    for (int j=0; j<columns; j++) {
      dcomplex *to=matrix+(size_t)j*(size_t)na_rows;
      const dcomplex *from=values+(size_t)j*(size_t)na_rows;
      for (int i=0; i<rows; i++) {
        to[i].r += from[i].r;
        to[i].i += from[i].i;
      }
    }
  }
  openmx_core_native_free(state->resources,scratch);
  state->time_add += MPI_Wtime()-start;
}

/* ---- outputs after the SCF (section 6.5) ---------------------------------- */

/* A block of the folded site form with the class of its cell: key[0] and
   key[1] are the atoms, key[2] to key[4] the components of the cell
   modulo the period, from 0 to the period minus 1. */
typedef struct {
  int key[5];
  long block;
} ExactExchangeFoldedBlock;

static int ExactExchange_Compare_Keys(const int *first, const int *second)
{
  for (int i=0; i<5; i++)
    if (first[i]!=second[i]) return first[i]<second[i] ? -1 : 1;
  return 0;
}

static int ExactExchange_Compare_Folded(const void *a, const void *b)
{
  const ExactExchangeFoldedBlock *first=(const ExactExchangeFoldedBlock*)a;
  const ExactExchangeFoldedBlock *second=(const ExactExchangeFoldedBlock*)b;
  const int order=ExactExchange_Compare_Keys(first->key,second->key);
  if (order!=0) return order;
  return first->block<second->block ? -1 : (second->block<first->block ? 1 : 0);
}

static void ExactExchange_Folded_Key(int x, int y, const int cell[3], const int period[3], int key[5])
{
  key[0] = x;
  key[1] = y;
  for (int i=0; i<3; i++) {
    int component=cell[i]%period[i];
    if (component<0) component += period[i];
    key[2+i] = component;
  }
}

/* The first of the sorted blocks with this key, or -1. */
static long ExactExchange_Find_Class(const ExactExchangeFoldedBlock *classes, long count, const int key[5])
{
  long low=0,high=count;
  while (low<high) {
    const long middle=low+(high-low)/2;
    if (ExactExchange_Compare_Keys(classes[middle].key,key)<0) low = middle+1;
    else high = middle;
  }
  return (low<count && ExactExchange_Compare_Keys(classes[low].key,key)==0) ? low : -1;
}

/* D of the folded site form on the blocks of the outputs (atoms, cells and
   offsets of blocks blocks). A block of the folded list keeps its values.
   Any other block takes those of the first block of its class in the
   folded list, with the same atoms and a cell that agrees modulo the
   period, times the sign of the mesh: -1 for every period between the two
   cells along an antiperiodic direction (section 6.4). D of the points of
   the mesh repeats in this way up to the rounding of its phases and the
   phase of the shift of the k points from the mesh (Shift_K_Point, section
   6.5), 1e-11 of D in the tests. moved gets the number of blocks of the
   second kind. */
static void ExactExchange_Unfold_Density(OpenMXExactExchangeState *state, long blocks, const int *atoms,
                                         const int *cells, const size_t *offsets, double *const *unfolded,
                                         MPI_Comm comm, long *moved)
{
  const int *period=state->input.fold_period;
  const int *antiperiodic=state->input.fold_antiperiodic;
  const long folded=state->block_count;
  ExactExchangeFoldedBlock *classes=(ExactExchangeFoldedBlock*)ExactExchange_Allocate(
      state,(size_t)folded,sizeof(ExactExchangeFoldedBlock),comm,"outputs");
  for (long f=0; f<folded; f++) {
    ExactExchange_Folded_Key(state->block_atoms[2*f],state->block_atoms[2*f+1],&state->block_cell[3*f],period,
                             classes[f].key);
    classes[f].block = f;
  }
  qsort(classes,(size_t)folded,sizeof(ExactExchangeFoldedBlock),ExactExchange_Compare_Folded);
  long count=0;
  for (long b=0; b<blocks; b++) {
    const int x=atoms[2*b],y=atoms[2*b+1];
    const int *cell=&cells[3*b];
    long source=openmx_exx_find_block(state->module,x,y,cell);
    double sign=1.0;
    if (source<0) {
      int key[5];
      ExactExchange_Folded_Key(x,y,cell,period,key);
      const long entry=ExactExchange_Find_Class(classes,folded,key);
      if (entry<0)
        ExactExchange_Abort(comm,"outputs","a block of the outputs has no block of its class in the folded site form");
      source = classes[entry].block;
      const int *from=&state->block_cell[3*source];
      for (int i=0; i<3; i++)
        if (antiperiodic[i] && ((cell[i]-from[i])/period[i])%2!=0) sign = -sign;
      count++;
    }
    const size_t elements=offsets[b+1]-offsets[b];
    if (elements!=state->block_offset[source+1]-state->block_offset[source])
      ExactExchange_Abort(comm,"outputs","two blocks of the same atoms differ in size");
    for (int spin=0; spin<state->spin_channels; spin++) {
      const double *values=state->density[spin]+state->block_offset[source];
      double *target=unfolded[spin]+offsets[b];
      for (size_t e=0; e<elements; e++) target[e] = sign*values[e];
    }
  }
  openmx_core_native_free(state->resources,classes);
  *moved = count;
}

/* Equation (14) with K(R) of the outputs: adds K(k) of one spin channel to
   H[1..N][1..N], with the phase of the H(k) of the outputs. Every element
   receives its terms in the order of the blocks. */
static void ExactExchange_Output_Sum(const OpenMXExactExchangeState *state, int spin, double k1, double k2,
                                     double k3, dcomplex **H)
{
  const double two_pi=6.283185307179586476925286766559;
  const double *exchange=state->output_exchange[spin];
  for (long block=0; block<state->output_block_count; block++) {
    const int x=state->output_block_atoms[2*block],y=state->output_block_atoms[2*block+1];
    const int *cell=&state->output_block_cell[3*block];
    const double phase=two_pi*(k1*(double)cell[0]+k2*(double)cell[1]+k3*(double)cell[2]);
    const double co=cos(phase),si=sin(phase);
    const int row0=state->first_orbital[x]+1;
    const int column0=state->first_orbital[y]+1;
    const int rows=state->first_orbital[x+1]-state->first_orbital[x];
    const int columns=state->first_orbital[y+1]-state->first_orbital[y];
    const double *source=exchange+state->output_block_offset[block];
    for (int i=0; i<rows; i++) {
      dcomplex *row=H[row0+i]+column0;
      const double *values=source+(size_t)i*(size_t)columns;
      for (int j=0; j<columns; j++) {
        row[j].r += values[j]*co;
        row[j].i += values[j]*si;
      }
    }
  }
}

/* K(k) of the outputs and K(k) of the SCF at the points of the folded
   mesh, where the two agree up to rounding and the phase of Shift_K_Point
   (section 6.5), 1e-12 of the largest element in the tests. results[0]
   gets the largest difference of an element and results[1] the largest
   element, over all points, spin channels and processes of comm, which
   divide the points among them. Collective. */
static void ExactExchange_Check_Outputs(OpenMXExactExchangeState *state, MPI_Comm comm, double results[2])
{
  const int n=state->orbitals;
  const int atoms=openmx_exx_atom_count(state->module);
  const int *period=state->input.fold_period;
  const int *antiperiodic=state->input.fold_antiperiodic;
  int rank=0,processes=1;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&processes);

  /* the whole matrix on this process: every orbital is local */
  int *local=(int*)ExactExchange_Allocate(state,(size_t)n,sizeof(int),comm,"outputs");
  char *reached=(char*)ExactExchange_Allocate(state,(size_t)atoms,sizeof(char),comm,"outputs");
  for (int g=0; g<n; g++) local[g] = g;
  for (int a=0; a<atoms; a++) reached[a] = 1;
  const size_t side=(size_t)n+1;
  dcomplex *scf=(dcomplex*)ExactExchange_Allocate(state,(size_t)n*(size_t)n,sizeof(dcomplex),comm,"outputs");
  dcomplex *buffer=(dcomplex*)ExactExchange_Allocate(state,side*side,sizeof(dcomplex),comm,"outputs");
  dcomplex **outputs=(dcomplex**)ExactExchange_Allocate(state,side,sizeof(dcomplex*),comm,"outputs");
  for (size_t i=0; i<side; i++) outputs[i] = buffer+i*side;

  double mine[2]={0.0,0.0};
  const long points=(long)period[0]*(long)period[1]*(long)period[2];
  for (long p=rank; p<points; p+=processes) {
    const long m[3]={p/((long)period[1]*(long)period[2]),(p/(long)period[2])%(long)period[1],p%(long)period[2]};
    double k[3];
    for (int i=0; i<3; i++) k[i] = ((double)m[i]+(antiperiodic[i] ? 0.5 : 0.0))/(double)period[i];
    for (int spin=0; spin<state->spin_channels; spin++) {
      memset(scf,0,sizeof(dcomplex)*(size_t)n*(size_t)n);
      memset(buffer,0,sizeof(dcomplex)*side*side);
      ExactExchange_Band_Sum(state,spin,k[0],k[1],k[2],local,local,reached,reached,n,scf);
      ExactExchange_Output_Sum(state,spin,k[0],k[1],k[2],outputs);
      for (int j=0; j<n; j++) {
        for (int i=0; i<n; i++) {
          const dcomplex a=scf[(size_t)j*(size_t)n+(size_t)i],b=outputs[i+1][j+1];
          const double difference=hypot(a.r-b.r,a.i-b.i),size=hypot(a.r,a.i);
          if (mine[0]<difference) mine[0] = difference;
          if (mine[1]<size) mine[1] = size;
        }
      }
    }
  }
  MPI_Allreduce(mine,results,2,MPI_DOUBLE,MPI_MAX,comm);
  openmx_core_native_free(state->resources,outputs);
  openmx_core_native_free(state->resources,buffer);
  openmx_core_native_free(state->resources,scf);
  openmx_core_native_free(state->resources,reached);
  openmx_core_native_free(state->resources,local);
}

void Exact_Exchange_Prepare_Outputs(OpenMXExactExchangeState *state, const OpenMXAtomicViews *atomic,
    MPI_Comm comm)
{
  if (!Exact_Exchange_Enabled(state)) return;
  Exact_Exchange_Release_Outputs(state);
  if (!Exact_Exchange_Active(state)) return;
  if (!state->prepared || state->spin_channels<1 || state->collecting)
    ExactExchange_Abort(comm,"outputs","the SCF left no density and exchange matrices");

  const double start=MPI_Wtime();
  int rank=0,processes=1;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&processes);
  /* the K(k) that the solver kept are not used after the SCF */
  ExactExchange_Forget_Band_Matrices(state);

  if (state->sites && !state->folded) {
    /* K(R) of the last build holds at every k point */
    state->output_owned = 0;
    state->output_spin_channels = state->spin_channels;
    state->output_block_count = state->block_count;
    state->output_block_atoms = state->block_atoms;
    state->output_block_cell = state->block_cell;
    state->output_block_offset = state->block_offset;
    for (int spin=0; spin<state->spin_channels; spin++) state->output_exchange[spin] = state->exchange[spin];
    state->outputs = 1;
    if (rank==0)
      Exact_Exchange_Log(state,2,"<EXX>  outputs after the SCF: K(R) of the last build with unfolded V sites\n");
    return;
  }

  /* The folded mesh needs new, unfolded integrals. The Gamma solver's
     module already holds the unfolded integrals; D(R) is its dense D at
     every translation of the one-point mesh. */
  OpenMXExxState *module=state->module;
  if (state->sites) {
    module=openmx_exx_create();
    if (module==NULL) ExactExchange_Abort(comm,"outputs","out of memory");
    OpenMXExxOptions options;
    (void)ExactExchange_Prepare_Module(state,module,atomic,comm,&options);
  }
  else if (!state->first_orbital) {
    const int atoms=openmx_exx_atom_count(module);
    state->first_orbital=(int*)ExactExchange_Allocate(state,(size_t)atoms+1,sizeof(int),comm,"outputs");
    state->first_orbital[0]=0;
    for (int atom=0; atom<atoms; ++atom)
      state->first_orbital[atom+1]=state->first_orbital[atom]+openmx_exx_atom_orbitals(module,atom);
  }
  int *block_atoms=NULL,*block_cell=NULL;
  size_t *block_offset=NULL;
  const long blocks=ExactExchange_Read_Blocks(state,module,comm,&block_atoms,&block_cell,&block_offset);
  const size_t elements=block_offset[blocks];
  long v_sites=0;
  for (int a=0; a<openmx_exx_atom_count(module); a++) v_sites += openmx_exx_v_neighbours(module,a);
  const double prepared=MPI_Wtime();

  state->output_owned = 1;
  state->output_spin_channels = state->spin_channels;
  state->output_block_count = blocks;
  state->output_block_atoms = block_atoms;
  state->output_block_cell = block_cell;
  state->output_block_offset = block_offset;
  double *density[2]={NULL,NULL};
  for (int spin=0; spin<state->spin_channels; spin++) {
    density[spin] = (double*)ExactExchange_Allocate(state,elements,sizeof(double),comm,"outputs");
    state->output_exchange[spin] = (double*)ExactExchange_Allocate(state,elements,sizeof(double),comm,"outputs");
  }
  long moved=0;
  if (state->sites)
    ExactExchange_Unfold_Density(state,blocks,block_atoms,block_cell,block_offset,density,comm,&moved);
  else {
    const int *first=state->first_orbital;
#pragma omp parallel for schedule(static) if (elements>=65536)
    for (long b=0; b<blocks; ++b) {
      const int a=block_atoms[2*b], c=block_atoms[2*b+1], cols=first[c+1]-first[c];
      for (int spin=0; spin<state->spin_channels; ++spin) {
        double *target=density[spin]+block_offset[b];
        const double *source=state->density[spin];
        for (int row=first[a]; row<first[a+1]; ++row) {
          memcpy(target,source+(size_t)row*state->orbitals+first[c],(size_t)cols*sizeof(double));
          target+=cols;
        }
      }
    }
  }
  /* the half build needs D_xy(R) = D_yx(-R)^T to the last bit */
  double asymmetry=0.0;
  char error[512];
  for (int spin=0; spin<state->spin_channels; spin++) {
    double largest=0.0;
    if (openmx_exx_symmetrize_blocks(module,density[spin],&largest,error,sizeof(error))!=0)
      ExactExchange_Abort(comm,"outputs",error);
    if (asymmetry<largest) asymmetry = largest;
  }

  /* One half build, as in the SCF (Exact_Exchange_Build), on the device of
     the builds where it takes the work. The counts of the device stay
     those of the SCF. */
  long first=0,last=0;
  ExactExchange_Abfs(state,module,1,comm,1,&first,&last);
  int threads=1;
#ifdef _OPENMP
  threads = omp_get_max_threads();
#endif
  const int blas_threads=1<threads ? Exact_Exchange_Host_Blas_Single() : 0;
  const ExactExchangeDeviceCounts counts=ExactExchange_Device_Counts(&state->device);
  const int disabled_before=state->device.disabled;
  state->device.largest_bytes = 0;
  state->device.largest_call_bytes = 0;
  ExactExchangeProducts products=ExactExchange_Products(state,threads);
  if (1<threads)
    products.host.loop = (OpenMXExactExchangeLoopProducts*)ExactExchange_Allocate(
        state,(size_t)threads,sizeof(OpenMXExactExchangeLoopProducts),comm,"outputs");
  OpenMXExxDevice device_table;
  const OpenMXExxDevice *device_calls=Exact_Exchange_Device_Table(products.device,&device_table);
  const OpenMXExxDenseBackend backend={ExactExchange_Dgemm,&products,1<threads ? ExactExchange_Parallel_For : NULL,
                                       threads,device_calls};
  double call_seconds=0.0;
  for (int spin=0; spin<state->spin_channels && state->input.alpha!=0.0; spin++) {
    double *exchange=state->output_exchange[spin];
    if (first<last) {
      const double call_start=MPI_Wtime();
      const int failed=openmx_exx_add_blocks_half_abfs(module,first,last,-state->input.alpha,density[spin],exchange,
                                                       &backend,error,sizeof(error));
      if (Exact_Exchange_Device_Call_End(&state->device)) call_seconds += MPI_Wtime()-call_start;
      if (failed!=0) ExactExchange_Abort(comm,"outputs",error);
    }
    ExactExchange_Sum(exchange,elements,comm);
    if (openmx_exx_mirror_blocks(module,exchange,error,sizeof(error))!=0) ExactExchange_Abort(comm,"outputs",error);
  }
  openmx_core_native_free(state->resources,products.host.loop);
  Exact_Exchange_Host_Blas_Restore(blas_threads);
  const ExactExchangeDeviceCounts after=ExactExchange_Device_Counts(&state->device);
  if (after.device_products>counts.device_products || after.device_calls>counts.device_calls)
    state->gpu_work |= EXX_GPU_PRODUCTS;
  ExactExchange_Device_Counts_Back(&state->device,&counts);
  for (int spin=0; spin<state->spin_channels; spin++) openmx_core_native_free(state->resources,density[spin]);
  if (state->sites) openmx_exx_destroy(module);
  state->outputs = 1;
  const double built=MPI_Wtime();

  ExactExchange_Report_Device_Failure(state,rank,disabled_before);
  const int detailed=ExactExchange_Detailed(state);
  double check[2]={0.0,0.0};
  if (detailed && state->sites) ExactExchange_Check_Outputs(state,comm,check);

  if (rank==0 && detailed) {
    const int *period=state->input.fold_period;
    Exact_Exchange_Log(state,2,"<EXX>  outputs after the SCF: K(R) without folding from D of the last build, %ld V sites,"
                       " %ld blocks, %.1f MB per spin channel (%.2f s)\n",
                       v_sites,blocks,(double)elements*sizeof(double)/1048576.0,built-start);
    Exact_Exchange_Log(state,2,"<EXX>  outputs: preparation %.2f s, build %.2f s; %ld blocks from another cell of their class,"
                       " largest asymmetry of D %.3e\n",prepared-start,built-prepared,moved,asymmetry);
    if (0<after.device_calls-counts.device_calls || 0<after.declined_calls-counts.declined_calls)
      Exact_Exchange_Log(state,2,"<EXX>  the outputs on the device of process 0: %lld calls in %.2f s, at most %.1f MB for one call;"
                         " left to the host for lack of device memory: %lld\n",
                         after.device_calls-counts.device_calls,call_seconds,(double)after.largest_call_bytes/1048576.0,
                         after.declined_calls-counts.declined_calls);
    if (0<after.device_products-counts.device_products || 0<after.declined_products-counts.declined_products)
      Exact_Exchange_Log(state,2,"<EXX>  matrix products of the outputs on the device of process 0: %lld in %.2f s, at most %.1f MB"
                         " for one product; left to the host for lack of device memory: %lld\n",
                         after.device_products-counts.device_products,products.device_seconds,
                         (double)after.largest_bytes/1048576.0,after.declined_products-counts.declined_products);
    if (state->sites)
      Exact_Exchange_Log(state,2,"<EXX>  K(k) of the outputs at the %d points of the mesh: largest difference from K(k) of the SCF %.3e,"
                         " largest element %.3e\n",period[0]*period[1]*period[2],check[0],check[1]);
  }
}

int Exact_Exchange_Output_Blocks(const OpenMXExactExchangeState *state, OpenMXExactExchangeOutputView *view)
{
  if (!Exact_Exchange_Active(state)) return 0;
  if (!state->outputs) return -1;
  *view = (OpenMXExactExchangeOutputView){
    .atoms = openmx_exx_atom_count(state->module),
    .orbitals = state->orbitals,
    .spin_channels = state->output_spin_channels,
    .blocks = state->output_block_count,
    .block_atoms = state->output_block_atoms,
    .block_cell = state->output_block_cell,
    .first_orbital = state->first_orbital,
    .block_offset = state->output_block_offset,
    .values = {state->output_exchange[0], state->output_exchange[1]}
  };
  return 1;
}

void Exact_Exchange_Add_Output_Hk(const OpenMXExactExchangeState *state, int spin, double k1, double k2,
    double k3, int n, dcomplex **H)
{
  if (!Exact_Exchange_Active(state)) return;
  if (!state->outputs)
    ExactExchange_Abort(state->comm,"outputs","the exchange matrix of the outputs is not prepared");
  if (n!=state->orbitals || spin<0 || state->output_spin_channels<=spin)
    ExactExchange_Abort(state->comm,"outputs","the matrix of the outputs has an unexpected size or spin");
  ExactExchange_Output_Sum(state,spin,k1,k2,k3,H);
}

void Exact_Exchange_Release_Outputs(OpenMXExactExchangeState *state)
{
  if (state==NULL) return;
  if (state->output_owned) {
    openmx_core_native_free(state->resources,state->output_block_atoms);
    openmx_core_native_free(state->resources,state->output_block_cell);
    openmx_core_native_free(state->resources,state->output_block_offset);
    for (int spin=0; spin<2; spin++) openmx_core_native_free(state->resources,state->output_exchange[spin]);
  }
  state->outputs = 0;
  state->output_owned = 0;
  state->output_spin_channels = 0;
  state->output_block_count = 0;
  state->output_block_atoms = NULL;
  state->output_block_cell = NULL;
  state->output_block_offset = NULL;
  state->output_exchange[0] = NULL;
  state->output_exchange[1] = NULL;
}
