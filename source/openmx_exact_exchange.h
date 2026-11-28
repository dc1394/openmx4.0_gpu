#ifndef OPENMX_EXACT_EXCHANGE_H
#define OPENMX_EXACT_EXCHANGE_H
#include "mpi.h"
#include "openmx_core.h"
#include "openmx_core_atomic.h"
#include "openmx_core_electronic.h"
#include "openmx_complex.h"
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
   form serves the band and local solvers: the blocks M_xy(R) between the
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
  double omega;              /* scf.exx.omega, 1/bohr; HSE06 only */
  double coulomb_cutoff;     /* scf.exx.coulomb.cutoff, bohr; PBE0 only. 0 takes the radius of the
                                sphere whose volume is kpoint_count times that of the cell */
  int kpoint_count;          /* points of the k mesh of the band solver, 1 for the cluster solver */
  int fold_period[3];        /* the numbers of points of a regular k mesh of the band solver along
                                b_1, b_2 and b_3 (scf.Kgrid with GCenter or REGULAR): the site form
                                folds its V neighbours and its columns by this period in cells
                                (section 6.4 of the guide). 0 for the cluster solver and other meshes */
  int fold_antiperiodic[3];  /* 1 along b_i where the points of that mesh lie half a point away from
                                the Gamma point (REGULAR with an even number of points): the density
                                matrix changes its sign over the period there. 0 otherwise */
  double abf_threshold;      /* scf.exx.abf.pca */
  double v_cutoff;           /* scf.exx.v.cutoff, bohr. Negative for PBE0 without the keyword: R_c
                                plus the largest ABF radius, which keeps every V block that the
                                truncated interaction does not make zero */
  int outer_max_iterations;  /* scf.exx.outer.maxIter */
  double outer_criterion;    /* scf.exx.outer.criterion, Hartree */
  int outer_extrapolation;   /* scf.exx.outer.extrapolation: the number of earlier builds that
                                Anderson's method combines with a new one for the next inner SCF
                                (Exact_Exchange_Extrapolate); 0 takes the new matrix as it is */
  double inner_ratio;        /* scf.exx.inner.criterion.ratio: an inner SCF after a build stops at
                                the larger of scf.criterion and this ratio times |delta| of the
                                build (Exact_Exchange_Inner_Criterion); 0 keeps scf.criterion */
  int radial_intervals;      /* 1DFFT.NumGridR */
  double reciprocal_maximum; /* square root of 1DFFT.EnergyCutoff, 1/bohr */
  int device_capacity_test;  /* OPENMX_GPU_CAPACITY_TEST_RECOVERY: the GPU leaves every matrix
                                product of the builds to the host, as when its memory is short */
  int gemmul8;               /* scf.gemmul8.enable on: the GPU computes its matrix products of the
                                builds through the GEMMul8 bridge; otherwise in FP64 */
  double gpu_min_work;       /* scf.exx.gpu.min.work: m*n*k per product, -1 for device-aware selection */
  int fileout_level;         /* level.of.fileout for the EXX sections of the SCF log */
} OpenMXExactExchangeInput;

void Exact_Exchange_Configure(OpenMXExactExchangeState *state, const OpenMXExactExchangeInput *input);

/* The largest radius R_c of the truncated Coulomb interaction of PBE0 (bohr)
   for a cell (lattice[i] is a_i, counted from 1, in bohr) and the points of
   the k mesh: that of the sphere whose volume is kpoint_count times that of
   the cell. PBE0 takes it without scf.exx.coulomb.cutoff. A larger radius
   counts the exchange with the periodic images of the density matrix, whose
   period is the supercell of the k mesh, and ends the run. The tolerance
   (bohr) admits the radius as printed with six decimals. */
#define OPENMX_EXACT_EXCHANGE_TRUNCATION_TOLERANCE 1.0e-6
double Exact_Exchange_Truncation_Limit(int kpoint_count, double (*lattice)[4]);

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

static inline int Exact_Exchange_Local_Solver(int solver)
{
  return solver==5 || solver==8 || solver==11; /* DC, Krylov, DC-LNO */
}

/* The owned atoms and unfolded PAO neighbour list of a local solver. */
typedef struct {
  int local_atoms;
  const int *local_to_global, *neighbour_count;
  int *const *neighbour_atom, *const *neighbour_cell, *const *cell_index;
  const OpenMXElectronicViews *electronic;
} OpenMXExactExchangeLocalTopology;

/* ---- one structure (section 6.2) ---------------------------------------- */

/* ABFs, neighbour lists, T blocks and V blocks of the bound structure as it
   is at the call, on every process. The processes of comm divide the pairs
   of sites of the T and V blocks, and every process gets all blocks
   (section 6.7). sites selects the form of the density
   and exchange matrices of this structure: 0 for the Gamma form, 1 for the
   site form, which also gets its list of blocks here. Pass local for a
   projection onto the unfolded PAO neighbour list. The exchange matrix
   of an earlier structure is dropped: the SCF of the new structure starts
   with the semilocal functional. Does nothing unless a hybrid functional is
   enabled. A failure ends the run through MPI_Abort. Rank 0 appends to
   the SCF log that is merged into .out. stdout_level is level.of.stdout. */
void Exact_Exchange_Prepare(OpenMXExactExchangeState *state, const OpenMXAtomicViews *atomic,
    int sites, const OpenMXExactExchangeLocalTopology *local, MPI_Comm comm,
    int stdout_level, const char *scf_log);

/* Local solvers already return real-space D on the same list used for H.
   Collect the owned blocks before SCF mixing changes them. */
void Exact_Exchange_Collect_Local(OpenMXExactExchangeState *state,
    const OpenMXElectronicViews *electronic, MPI_Comm comm);

/* Projected H+K in calculation-owned storage with room for solver halos.
   Before the first exchange build, returns the ordinary Hamiltonian. */
double *****Exact_Exchange_Local_Hamiltonian(OpenMXExactExchangeState *state,
    const OpenMXElectronicViews *electronic);

/* Write to each output enabled for the message's level. Ordinary messages
   are part of the .out summary even at level.of.fileout 0. */
void Exact_Exchange_Log(const OpenMXExactExchangeState *state, int level, const char *format, ...);

/* Report SCF convergence and optional counters, then record actual device
   use and close the SCF log after the force and post-SCF steps. Collective. */
void Exact_Exchange_Report_Scf(const OpenMXExactExchangeState *state, MPI_Comm comm,
    int inner_converged, int scf_limit);
void Exact_Exchange_Report_Iteration(const OpenMXExactExchangeState *state, int md, int scf,
    int inner_converged);
void Exact_Exchange_Finish_Report(OpenMXExactExchangeState *state, MPI_Comm comm);

/* ---- semilocal part ------------------------------------------------------ */

/* The mixing ratio and the screening parameter that the semilocal exchange
   takes (Set_XC_Grid). The ratio is 0 until the first exchange matrix of
   the structure exists: the first inner SCF is a PBE one (section 6.1).
   The screening parameter is 0 for PBE0, whose semilocal exchange is the
   PBE exchange times 1 - alpha. */
double Exact_Exchange_Semilocal_Alpha(const OpenMXExactExchangeState *state);
double Exact_Exchange_Semilocal_Omega(const OpenMXExactExchangeState *state);

/* ---- density matrix (section 6.3) ----------------------------------------- */

/* Starts a collection: the next solution of the eigenvalue problem hands its
   vectors and occupations over. spin_channels is 1 without and 2 with spin
   polarization. */
void Exact_Exchange_Collect_Begin(OpenMXExactExchangeState *state, int spin_channels, MPI_Comm comm);

/* 1 between Exact_Exchange_Collect_Begin and Exact_Exchange_Collect_End: a
   solver that has to fetch its vectors from a device asks before it does.
   NULL is allowed. */
int Exact_Exchange_Collecting(const OpenMXExactExchangeState *state);

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
   density matrices: the ABFs of the centre atoms are distributed over the
   processes, which may share an atom (section 6.7), and the contributions
   summed. The build is the half build of section 6.4, which goes through
   one of the two V neighbours of every mirror pair: its sum X gives
   K = X + X^T, symmetric to the last bit. Keeps the matrix of the build
   before, and computes E_x (equation (4)) and delta (equation (15)). From
   here on K is added to the Hamiltonian and the semilocal part is the hybrid
   one. */
void Exact_Exchange_Build(OpenMXExactExchangeState *state, MPI_Comm comm);

/* After a build: 1 when the outer iteration goes on with a new inner SCF, 0
   when it ends. It ends when |delta| is below scf.exx.outer.criterion or when
   scf.exx.outer.maxIter builds are done. The criterion does not end it after
   the first build, nor with loose = 1: the inner SCF of the build stopped on
   the criterion of Exact_Exchange_Inner_Criterion, larger than scf.criterion,
   which the step did not meet. */
int Exact_Exchange_Outer_Continue(OpenMXExactExchangeState *state, int loose);

/* Anderson's method for the outer iteration (scf.exx.outer.extrapolation m,
   section 6.6 of the guide), after a build that the outer iteration goes on
   from. Build n makes X_n from the density of the inner SCF that used the
   matrix Y_{n-1}, and F_n = X_n - Y_{n-1} is its residual. Without the
   method, the next inner SCF uses Y_n = X_n. With it, Y_n = X_n - sum_j g_j
   (X_j - X_{j-1}) over the last m differences of the stored builds, where
   the coefficients g_j minimize |F_n - sum_j g_j (F_j - F_{j-1})| (sums over
   all elements of all spin channels). The first build is not stored: its
   inner SCF had no exchange matrix and the semilocal functional without the
   hybrid part, so its residual belongs to another iteration. E_x and delta
   stay those of the build, and the outer iteration ends on a built matrix:
   to be called only when Exact_Exchange_Outer_Continue returned 1. Process
   0 computes the coefficients for all. Returns 1 when the matrix of the next
   inner SCF is not X_n. A failure ends the run through MPI_Abort. */
int Exact_Exchange_Extrapolate(OpenMXExactExchangeState *state, MPI_Comm comm);

int Exact_Exchange_Outer_Iterations(const OpenMXExactExchangeState *state); /* builds for this structure */
int Exact_Exchange_Outer_Converged(const OpenMXExactExchangeState *state);  /* the criterion was met */
double Exact_Exchange_Energy(const OpenMXExactExchangeState *state);        /* E_x of the last build, Hartree */
double Exact_Exchange_Change(const OpenMXExactExchangeState *state);        /* delta of the last build, Hartree */

/* ---- forces (section 5.9) ------------------------------------------------- */

/* The derivative of E_x of the last build with respect to the positions of
   the atoms, with the collected density matrices fixed: gradient[3*a + x] is
   dE_x/dtau along the Cartesian axis x (0, 1, 2) of atom a (counted from 0),
   in Hartree/bohr, as the main program keeps dE/dtau of its total energy
   (Gxyz[a+1][17+x]). The force of the exact exchange is its negative. The
   response of the orbitals is not in it: the Pulay term of the main program
   has it, with the energy density matrix of the Hamiltonian that contains
   K. The ABFs of the centre atoms are distributed over the processes for
   the derivatives of E_x with respect to the T and V blocks, as in the
   build, and the pairs of sites for the derivatives of the blocks (section
   6.7); every process gets the sum. The device may take matrix products
   while forming the weights. The SCF build counters exclude them.
   The optional strain array receives dE_x/dF in Hartree, for x' = F x.
   The Cartesian gradient belongs to the state and holds until
   the next call or preparation. Returns NULL without an exchange matrix (no
   hybrid functional, or no build for the present structure). A failure
   ends the run through MPI_Abort. */
const double *Exact_Exchange_Gradient(OpenMXExactExchangeState *state, MPI_Comm comm, double strain[9]);

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
   mixer never reach back to it: the last count before a new history for
   RMM-DIIS, GR-Pulay and RMM-ADIIS (start_pulay - 1), whose history then
   begins with the first residual of the new matrix. */
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

/* The criterion of the present inner SCF for the stop test of the main
   program, in place of scf.criterion (criterion). With
   scf.exx.inner.criterion.ratio c > 0 and an exchange matrix, it is the
   larger of criterion and c |delta| of the last build: an inner SCF from a
   matrix that is still far from its fixed point stops sooner. After a
   build with |delta| below scf.exx.outer.criterion it is criterion, since
   the next build may end the outer iteration.
   Exact_Exchange_Inner_Stop_Allowed keeps scf.criterion, so the residual
   still has to fall to a tenth of its reference. A step that stops only on
   the larger value is reported to Exact_Exchange_Outer_Continue. Returns
   criterion without an exchange matrix or with c = 0. NULL is allowed. */
double Exact_Exchange_Inner_Criterion(const OpenMXExactExchangeState *state, double criterion);

/* ---- addition to the Hamiltonian (section 6.5) --------------------------- */

/* 1 when an exchange matrix exists and has to be added. NULL is allowed. */
int Exact_Exchange_Active(const OpenMXExactExchangeState *state);

/* Return device resources after a phase that can reuse them has completed. */
void Exact_Exchange_End_Device_Phase(OpenMXExactExchangeState *state);

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
   elements of the neighbour list. Does nothing without an exchange matrix.
   K(k) is summed once for every spin channel, k point and local array
   until the next build, and kept while the kept matrices of this process
   take no more memory than K and the K of the build before; the later
   additions add the kept K(k). Without room the sum goes to an array that
   is dropped afterwards, so every k point has the same rounding: the sum
   is formed first and then added to Hs. Only when the memory for that
   array is short do the terms go to Hs one at a time. */
void Exact_Exchange_Add_Band_Hs(OpenMXExactExchangeState *state, int spin, double k1, double k2, double k3,
    int n, double *Hs, int na_rows, int na_cols, int nblk, int np_rows, int np_cols, int my_prow, int my_pcol);

/* A solver that keeps its dense Hamiltonian on a device builds the matrix
   to add with the two functions above in a buffer of the host (one process
   holds the whole matrix: a 1 x 1 grid with one block of n), copies it to
   the device and adds it there. It then reports the addition here, with the
   seconds of the copy and of the addition on the device. NULL is allowed. */
void Exact_Exchange_Device_Addition(OpenMXExactExchangeState *state, double seconds);

/* ---- outputs after the SCF (section 6.5) ---------------------------------- */

/* The band dispersion and the DOS of the band solver build H(k) from the
   Hamiltonian of the neighbour list at k points of their own, and add K(k)
   with the function below. Exact_Exchange_Prepare_Outputs makes K(R) for
   them after the SCF: K(R) of the last build when the site form is not
   folded, which holds at every k point. A folded site form holds at the
   points of its mesh only: the function then prepares the structure again
   without folding, takes D of the last build for every block of that
   preparation (a block outside the folded list takes the block of its
   class times the sign of the mesh), and builds K(R) once, divided among
   the processes of comm as a build of the SCF is. The tables of the SCF
   stay. Detailed logging compares K(k) of the outputs with K(k) of the SCF
   at the points of the mesh. Does nothing without an exchange
   matrix or for the Gamma form. Collective. */
void Exact_Exchange_Prepare_Outputs(OpenMXExactExchangeState *state, const OpenMXAtomicViews *atomic,
    MPI_Comm comm);

/* Read-only view of the post-SCF (unfolded) exchange, valid through
   Exact_Exchange_Release_Outputs. Atoms and first orbitals are zero-based.
   Each spin holds row-major blocks at the same offsets, including an
   end offset after the last block. */
typedef struct {
    int atoms, orbitals, spin_channels;
    long blocks;
    const int *block_atoms, *block_cell, *first_orbital;
    const size_t *block_offset;
    const double *values[2];
} OpenMXExactExchangeOutputView;

/* 1: view available, 0: no active exchange, -1: outputs not prepared. */
int Exact_Exchange_Output_Blocks(const OpenMXExactExchangeState *state, OpenMXExactExchangeOutputView *view);

/* Adds K(k) = sum_R e^{2 pi i k.R} K(R) of the outputs for one spin channel
   to H[1..n][1..n], the dense H(k) of the outputs at the k point (k1, k2,
   k3), with the orbitals counted from 1 and the phase of the H(k) that
   Hamiltonian_Band builds. Does nothing without an exchange matrix; NULL is
   allowed. Ends the run when K(R) of the outputs is not prepared. */
void Exact_Exchange_Add_Output_Hk(const OpenMXExactExchangeState *state, int spin, double k1, double k2,
    double k3, int n, dcomplex **H);

/* Gives the memory of K(R) of the outputs back. NULL is allowed. */
void Exact_Exchange_Release_Outputs(OpenMXExactExchangeState *state);

#ifdef __cplusplus
}
#endif
#endif
