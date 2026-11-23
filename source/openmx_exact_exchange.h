#ifndef OPENMX_EXACT_EXCHANGE_H
#define OPENMX_EXACT_EXCHANGE_H
#include "mpi.h"
#include "openmx_core.h"
#include "openmx_core_atomic.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Exact exchange of the hybrid functionals: the layer between the SCF of the
   main program and modules/exx (docs/EXX_IMPLEMENTATION_GUIDE.md, sections 6
   and 7.4).

   The state belongs to one calculation, like the states of the solvers. The
   functions take every input as an argument and read no global variable.
   MPI is used in this layer only: modules/exx works within one process.

   The orbitals of the unit cell, N in number, are ordered by atom and,
   within an atom, as in the solvers: orbital i of atom A (both counted from
   1) has the index MP[A]-1+i.

   The density and exchange matrices have one of two forms, chosen at the
   preparation of a structure. The Gamma form serves the cluster solver: N x N
   matrices, with all lattice images of an orbital folded onto it. The site
   form serves the band solver and any k mesh: the blocks M_xy(R) between the
   orbitals of atom x in the cell at the origin and those of atom y in the
   cell R, for the list of blocks that modules/exx gives (exx.h).

   Borrowed calculation state. Bind, clear and finish require quiescent
   users; clear and finish invalidate the view. */
typedef struct OpenMXExactExchangeState OpenMXExactExchangeState;
OpenMXExactExchangeState *Exact_Exchange_Bind_Resources(OpenMXCoreContext *core);

#define OPENMX_EXACT_EXCHANGE_NONE 0
#define OPENMX_EXACT_EXCHANGE_HSE06 1
#define OPENMX_EXACT_EXCHANGE_PBE0 2

/* The values of the input keywords, and the fault injection of the GPU paths. */
typedef struct {
  int functional;            /* OPENMX_EXACT_EXCHANGE_... */
  double alpha;              /* scf.exx.alpha */
  double omega;              /* scf.exx.omega, 1/bohr */
  double abf_threshold;      /* scf.exx.abf.pca */
  double v_cutoff;           /* scf.exx.v.cutoff, bohr */
  int outer_max_iterations;  /* scf.exx.outer.maxIter */
  double outer_criterion;    /* scf.exx.outer.criterion, Hartree */
  int radial_intervals;      /* 1DFFT.NumGridR */
  double reciprocal_maximum; /* square root of 1DFFT.EnergyCutoff, 1/bohr */
  int device_capacity_test;  /* OPENMX_GPU_CAPACITY_TEST_RECOVERY: the GPU leaves every matrix
                                product of the builds to the host, as when its memory is short */
} OpenMXExactExchangeInput;

void Exact_Exchange_Configure(OpenMXExactExchangeState *state, const OpenMXExactExchangeInput *input);

/* The structure of the calculation, in the arrays of the main program: atoms
   and Cartesian components are counted from 1, lengths are in bohr, and
   lattice[i] is the lattice vector a_i. The layer keeps the pointers and
   reads the arrays at every preparation, since the main program moves the
   atoms in place. The arrays have to outlive the calculation. The function
   that reads the input binds them: it is the one that fills them. */
typedef struct {
  int atom_count;
  const int *species_of_atom;
  double *const *positions;
  double (*lattice)[4];
} OpenMXExactExchangeStructure;

void Exact_Exchange_Bind_Structure(OpenMXExactExchangeState *state, const OpenMXExactExchangeStructure *structure);

/* 1 when the input asks for a hybrid functional. NULL is allowed. */
int Exact_Exchange_Enabled(const OpenMXExactExchangeState *state);

/* ---- one structure (section 6.2) ---------------------------------------- */

/* ABFs, neighbour lists, T blocks and V blocks of the bound structure as it
   is at the call, on every process. sites selects the form of the density
   and exchange matrices of this structure: 0 for the Gamma form, 1 for the
   site form, which also gets its list of blocks here. The exchange matrix
   of an earlier structure is dropped: the SCF of the new structure starts
   with the semilocal functional. Does nothing unless a hybrid functional is
   enabled. A failure ends the run through MPI_Abort. verbosity is
   level.of.stdout. */
void Exact_Exchange_Prepare(OpenMXExactExchangeState *state, const OpenMXAtomicViews *atomic,
    int sites, MPI_Comm comm, int verbosity);

/* ---- semilocal part ------------------------------------------------------ */

/* The mixing ratio and the screening parameter that the semilocal exchange
   takes (Set_XC_Grid). The ratio is 0 until the first exchange matrix of
   the structure exists: the first inner SCF is a PBE one (section 6.1). */
double Exact_Exchange_Semilocal_Alpha(const OpenMXExactExchangeState *state);
double Exact_Exchange_Semilocal_Omega(const OpenMXExactExchangeState *state);

/* ---- density matrix (section 6.3) ----------------------------------------- */

/* Starts a collection: the next solution of the eigenvalue problem hands its
   vectors and occupations over. spin_channels is 1 without and 2 with spin
   polarization. */
void Exact_Exchange_Collect_Begin(OpenMXExactExchangeState *state, int spin_channels, MPI_Comm comm);

/* Gamma form. Adds sum_s occupation[s] c_s c_s^T of the states that this
   process holds to the density matrix of one spin channel.
   vectors[basis*state_count + s] is the component of state s on the orbital
   basis (the layout of EVec1 of the cluster solver); occupations are those
   of one spin channel, between 0 and 1. Does nothing outside a collection. */
void Exact_Exchange_Collect_Gamma(OpenMXExactExchangeState *state, int spin, int basis_count,
    int state_count, const double *occupation, const double *vectors);

/* Site form, equation (5). Adds the contribution of the states that this
   process holds at one k point to the blocks of one spin channel:

     D_{mu nu}(R) += Re[ e^{2 pi i k.R} sum_s a_s^2 conj(c_{mu s}) c_{nu s} ] / weight_sum

   k1, k2, k3 are the coordinates of the k point in units of the reciprocal
   lattice vectors, and weight_sum is the sum of the weights of all k points
   of the mesh. vectors holds complex numbers as pairs of a real and an
   imaginary part: the pair at 2*(basis*stride + s) is the component of state
   s on the orbital basis (the transposed vectors of the band solver).
   amplitude[s]^2 is the weight of the k point times the occupation of the
   state in one spin channel (the occupation is between 0 and 1); NULL means
   that the vectors are multiplied by their amplitudes already. A k point
   that the solver treats in place of the pair k and -k carries twice the
   weight: the real part is the contribution of both. Does nothing outside a
   collection. */
void Exact_Exchange_Collect_K(OpenMXExactExchangeState *state, int spin, double k1, double k2, double k3,
    double weight_sum, int basis_count, int state_count, int stride, const double *vectors,
    const double *amplitude);

/* Ends the collection: sums the contributions of the processes, after which
   every process holds the same density matrices. */
void Exact_Exchange_Collect_End(OpenMXExactExchangeState *state, MPI_Comm comm);

/* The largest deviation of the collected density matrices from the density
   matrix of the solver, over the elements that the solver stores:
   density[spin][MA][LB][i][j] for the atoms 1 .. local_atoms of this process
   and their neighbours 0 .. neighbour_count[GA]. first_orbital is MP. The
   site form compares the block of the two atoms and the cell of the
   neighbour, cell_index[neighbour_cell[GA][LB]][1..3] (ncn and atv_ijk of
   the main program); the Gamma form does not read these two arguments. The
   result is the maximum over the processes. */
double Exact_Exchange_Density_Deviation(const OpenMXExactExchangeState *state, int spin_channels,
    int local_atoms, const int *local_to_global, const int *neighbour_count, int *const *neighbour_atom,
    int *const *neighbour_cell, int *const *cell_index,
    const int *species_of_atom, const int *orbitals_of_species, const int *first_orbital,
    double *****density, MPI_Comm comm);

/* ---- exchange matrix (sections 6.4 and 6.6) ------------------------------ */

/* Builds the exchange matrix K of every spin channel from the collected
   density matrices: the centre atoms are distributed over the processes and
   the contributions summed. Keeps the matrix of the build before, and
   computes E_x (equation (4)) and delta (equation (15)). From here on K is
   added to the Hamiltonian and the semilocal part is the hybrid one. */
void Exact_Exchange_Build(OpenMXExactExchangeState *state, MPI_Comm comm, int verbosity);

/* After a build: 1 when the outer iteration goes on with a new inner SCF, 0
   when it ends. It ends when |delta| is below scf.exx.outer.criterion (never
   after the first build) or when scf.exx.outer.maxIter builds are done. */
int Exact_Exchange_Outer_Continue(OpenMXExactExchangeState *state);

int Exact_Exchange_Outer_Iterations(const OpenMXExactExchangeState *state); /* builds for this structure */
int Exact_Exchange_Outer_Converged(const OpenMXExactExchangeState *state);  /* the criterion was met */
double Exact_Exchange_Energy(const OpenMXExactExchangeState *state);        /* E_x of the last build, Hartree */
double Exact_Exchange_Change(const OpenMXExactExchangeState *state);        /* delta of the last build, Hartree */
/* The largest difference between an element of K and the element that
   equals it by symmetry, K_ij and K_ji or K_xy(R) and K_yx(-R)^T, in the last
   build before the two were replaced by their mean. K(k) is Hermitian when
   it vanishes. */
double Exact_Exchange_Asymmetry(const OpenMXExactExchangeState *state);

/* ---- inner SCF with an exchange matrix (section 6.6) --------------------- */

/* A build changes the problem that the inner SCF solves. The mixers that
   extrapolate keep the residuals of earlier steps, and a residual measured
   with the exchange matrix before reports the converged density of that
   matrix as the solution: it must not enter an extrapolation. The mixers
   choose their action from a step count, so the count is set back.

   Returns the offset to subtract from the step count of the mixing
   (SCF_iter_shift of DFT()), or -1 when the count stays as it is.
   mixing_type is scf.Mixing.Type as the main program codes it (0 Simple,
   1 RMM-DIIS, 2 GR-Pulay, 3 Kerker, 4 RMM-DIISK, 5 RMM-DIISH, 6 RMM-ADIIS,
   7 RMM-DIISV), start_pulay is scf.Mixing.StartPulay, and local_step is the
   count of the step of the build.

   The first mixing after the build still sees the residual of the old
   matrix: it is the mixing at the end of the step of the build for the
   mixers of the density (1, 2, 4, 6), and the one before the solution of
   the next step for those of the Hamiltonian and of the potential (5, 7).
   The offset gives it the count from which the extrapolations of the
   mixer never reach back to it. */
int Exact_Exchange_Mixing_Offset(int mixing_type, int start_pulay, int local_step);

/* Whether the inner SCF may stop at a step. The stop test of the main
   program compares the band energies of two steps, and passes as well when
   the mixing has hardly moved the density, which then has not answered the
   exchange matrix. step counts the steps since the build (1: the first
   solution with the matrix), and residual_norm is the residual norm that
   the stop test of the step sees (NormRD). At step 2 this is the first
   residual measured with the matrix; it is kept as the reference. A later
   step may stop when the norm has fallen to a tenth of the reference, or
   below criterion (scf.criterion), the level under which the matrix has not
   disturbed the density.

   Simple and Kerker mixing (mixing_type 0 and 3 in the coding of
   Exact_Exchange_Mixing_Offset) have a third level, sqrt(1e-9). Under it
   their rule for the mixing weight only reduces the weight, and the
   residual stands still, with or without an exchange matrix: the SCF of
   the semilocal functional ends in this state as well. A tenth of the
   reference can lie below what these two mixers reach.

   To be called at every step. Returns 1 without an exchange matrix. NULL
   is allowed. */
int Exact_Exchange_Inner_Stop_Allowed(OpenMXExactExchangeState *state, int mixing_type, int step,
    double residual_norm, double criterion);

/* ---- addition to the Hamiltonian (section 6.5) --------------------------- */

/* 1 when an exchange matrix exists and has to be added. NULL is allowed. */
int Exact_Exchange_Active(const OpenMXExactExchangeState *state);

/* Gamma form. Adds K of one spin channel to the dense Hamiltonian of the
   cluster solver in its block-cyclic distribution: Hs[j*na_rows+i] is the
   local element (i,j) of the n x n matrix, and the remaining arguments are
   the values of the BLACS grid that the solver uses. Does nothing without
   an exchange matrix. */
void Exact_Exchange_Add_Cluster_Hs(OpenMXExactExchangeState *state, int spin, int n, double *Hs,
    int na_rows, int na_cols, int nblk, int np_rows, int np_cols, int my_prow, int my_pcol);

/* Site form, equation (14). Adds K(k) = sum_R e^{2 pi i k.R} K(R) of one
   spin channel to the dense complex Hamiltonian of the band solver at the k
   point (k1, k2, k3): the pair of doubles at 2*(j*na_rows+i) is the local
   element (i,j) of the n x n matrix in the block-cyclic distribution of the
   solver. The phase is the one with which the solver builds H(k) from the
   elements of the neighbour list. Does nothing without an exchange matrix. */
void Exact_Exchange_Add_Band_Hs(OpenMXExactExchangeState *state, int spin, double k1, double k2, double k3,
    int n, double *Hs, int na_rows, int na_cols, int nblk, int np_rows, int np_cols, int my_prow, int my_pcol);

/* ---- times --------------------------------------------------------------- */

/* Seconds spent on this process since the binding: preparation, collection
   of the density matrices, builds, additions to the Hamiltonian. */
void Exact_Exchange_Times(const OpenMXExactExchangeState *state, double times[4]);

/* How the builds used the GPUs (openmx_exact_exchange_device.h), over the
   processes of comm: use[0] is the number of matrix products computed on a
   device, use[1] the number left to the host because a device had no room
   for them, use[2] the number of processes with at least one product on
   their device, and use[3] the number of processes whose device was
   switched off after a failed call. seconds is the longest time that one
   process spent in its products on the device, transfers included; it is a
   part of the time of the builds. All are 0 in the CPU build. Collective. */
void Exact_Exchange_Device_Use(const OpenMXExactExchangeState *state, MPI_Comm comm, long long use[4],
    double *seconds);

#ifdef __cplusplus
}
#endif
#endif
