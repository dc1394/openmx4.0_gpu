/* Test of the C interface, compiled as C: the header must be valid C, and a C
   program must be able to prepare a structure, build the exchange matrix of
   the Gamma point, and pass its own matrix product.  That the numbers are
   those of the C++ interface is checked by exx-exchange-system. */
#include <openmx/exx/exx.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

enum { POINTS = 160, ATOMS = 3, N = 7 };

/* Always active, including Release/NDEBUG. */
#define CHECK(expression) do { if (!(expression)) { \
  fprintf(stderr,"line %d: %s\n",__LINE__,#expression); openmx_exx_destroy(state); return 1; } } while (0)

static int close_to(double a, double b, double relative)
{
  const double scale = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
  return fabs(a - b) <= relative * (scale > 1e-300 ? scale : 1.0);
}

/* The product of the caller: it counts its calls in the context and leaves
   the work to the product of the library. */
static int counting_dgemm(void *context, char transa, char transb, int m, int n, int k, double alpha,
                          const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc)
{
  ++*(long *)context;
  return openmx_exx_reference_dgemm(NULL, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static int failing_dgemm(void *context, char transa, char transb, int m, int n, int k, double alpha,
                         const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc)
{
  (void)context; (void)transa; (void)transb; (void)m; (void)n; (void)k; (void)alpha;
  (void)a; (void)lda; (void)b; (void)ldb; (void)beta; (void)c; (void)ldc;
  return 3;
}

/* A radial mesh like that of a .pao file, dense near the nucleus, and smooth
   functions that vanish at the cutoff radius: the test needs no file. */
static void radial_mesh(double cutoff, double *mesh)
{
  int i;
  for (i = 0; i < POINTS; ++i) mesh[i] = cutoff * exp(-6.0 * (1.0 - (double)i / (POINTS - 1)));
}

static double radial_function(int kind, double x)
{
  const double envelope = (1.0 - x * x) * (1.0 - x * x);
  if (x >= 1.0) return 0.0;
  switch (kind) {
    case 0: return envelope;                      /* s */
    case 1: return envelope * (1.0 - 3.0 * x * x); /* s with a node */
    default: return envelope * x;                 /* p */
  }
}

static double largest(const double *values, int count)
{
  double result = 0.0;
  int i;
  for (i = 0; i < count; ++i) if (fabs(values[i]) > result) result = fabs(values[i]);
  return result;
}

int main(void)
{
  OpenMXExxState *state = NULL;
  OpenMXExxOptions options;
  OpenMXExxGeometry geometry;
  OpenMXExxSpecies species[2];
  OpenMXExxReport report;
  OpenMXExxDenseBackend table;
  static double mesh_a[POINTS], mesh_b[POINTS], values_a[3 * POINTS], values_b[POINTS];
  const int angular_a[3] = {0, 0, 1}, angular_b[1] = {0};
  const double cutoff_a = 4.2, cutoff_b = 3.6;
  const double positions[3 * ATOMS] = {0.2, 0.1, 0.3, 2.5, 0.4, 0.1, 0.6, 2.9, 3.1};
  const int atom_species[ATOMS] = {0, 1, 1};
  const int wrong_species[ATOMS] = {0, 1, 2};
  const double lattice[3][3] = {{6.3, 0.0, 0.0}, {0.4, 6.8, 0.0}, {0.2, -0.3, 7.1}};
  double d[N * N], k[N * N], other[N * N], truncated[N * N], zero[N * N];
  char message[160], tiny[4];
  double sum, scale;
  long calls = 0;
  int i, j, a;

  /* NULL is harmless where the header says so. */
  openmx_exx_destroy(NULL);
  openmx_exx_default_options(NULL);
  CHECK(openmx_exx_atom_count(NULL) == -1 && openmx_exx_orbital_count(NULL) == -1);
  CHECK(openmx_exx_report(NULL, &report) != 0);

  /* A density matrix of two orbitals with the occupations 1 and 0.6. */
  for (i = 0; i < N; ++i)
    for (j = 0; j < N; ++j)
      d[i * N + j] = cos(0.9 * i + 0.3) * cos(0.9 * j + 0.3) + 0.6 * sin(1.7 * i + 0.1) * sin(1.7 * j + 0.1);
  memset(zero, 0, sizeof zero);

  state = openmx_exx_create();
  CHECK(state != NULL);

  /* Nothing is prepared yet. */
  CHECK(openmx_exx_atom_count(state) == -1 && openmx_exx_orbital_count(state) == -1);
  CHECK(openmx_exx_atom_orbitals(state, 0) == -1 && openmx_exx_atom_abfs(state, 0) == -1);
  CHECK(openmx_exx_c_neighbours(state, 0) == -1 && openmx_exx_v_neighbours(state, 0) == -1);
  CHECK(openmx_exx_report(state, &report) != 0);
  strcpy(message, "untouched");
  memset(k, 0, sizeof k);
  CHECK(openmx_exx_add_gamma(state, 0, 1, -0.25, d, k, NULL, message, sizeof message) != 0);
  CHECK(strlen(message) > 0 && strcmp(message, "untouched") != 0);

  openmx_exx_default_options(&options);
  CHECK(options.kernel == OPENMX_EXX_KERNEL_SHORT_RANGE && options.kernel_parameter == 0.11);
  CHECK(options.abf_threshold == 1.0e-4 && options.v_cutoff == 8.0 && options.radial_intervals == 900);
  CHECK(options.reciprocal_maximum == 60.0 && options.reciprocal_intervals == 900);
  /* Coarse grids keep the test short. */
  options.abf_threshold = 1.0e-3;
  options.v_cutoff = 2.5;
  options.radial_intervals = 100;
  options.reciprocal_maximum = 14.0;
  options.reciprocal_intervals = 100;

  radial_mesh(cutoff_a, mesh_a);
  radial_mesh(cutoff_b, mesh_b);
  for (i = 0; i < POINTS; ++i) {
    for (j = 0; j < 3; ++j) values_a[j * POINTS + i] = radial_function(j, mesh_a[i] / cutoff_a);
    values_b[i] = radial_function(0, mesh_b[i] / cutoff_b);
  }
  species[0].cutoff = cutoff_a;
  species[0].mesh_size = POINTS;
  species[0].mesh = mesh_a;
  species[0].shell_count = 3;
  species[0].angular = angular_a;
  species[0].values = values_a;
  species[1].cutoff = cutoff_b;
  species[1].mesh_size = POINTS;
  species[1].mesh = mesh_b;
  species[1].shell_count = 1;
  species[1].angular = angular_b;
  species[1].values = values_b;
  memcpy(geometry.lattice, lattice, sizeof lattice);
  geometry.atom_count = ATOMS;
  geometry.positions = positions;
  geometry.species = atom_species;
  geometry.species_count = 2;
  geometry.species_data = species;

  /* Missing arguments of the preparation; the message is optional. */
  CHECK(openmx_exx_prepare(NULL, &options, &geometry, message, sizeof message) != 0 && strlen(message) > 0);
  CHECK(openmx_exx_prepare(state, NULL, &geometry, message, sizeof message) != 0);
  CHECK(openmx_exx_prepare(state, &options, NULL, NULL, 0) != 0);
  CHECK(openmx_exx_atom_count(state) == -1);

  CHECK(openmx_exx_prepare(state, &options, &geometry, message, sizeof message) == 0);
  CHECK(message[0] == '\0');
  CHECK(openmx_exx_atom_count(state) == ATOMS && openmx_exx_orbital_count(state) == N);
  CHECK(openmx_exx_atom_orbitals(state, 0) == 5 && openmx_exx_atom_orbitals(state, 1) == 1);
  CHECK(openmx_exx_atom_orbitals(state, 2) == 1);
  /* One s function has one product, so its atom has one ABF. */
  CHECK(openmx_exx_atom_abfs(state, 0) > 1 && openmx_exx_atom_abfs(state, 1) == 1 && openmx_exx_atom_abfs(state, 2) == 1);
  /* The cell is smaller than the cutoffs: every atom has neighbours. */
  for (a = 0; a < ATOMS; ++a) CHECK(openmx_exx_c_neighbours(state, a) > 1 && openmx_exx_v_neighbours(state, a) > 1);
  CHECK(openmx_exx_atom_orbitals(state, ATOMS) == -1 && openmx_exx_atom_abfs(state, -1) == -1);
  CHECK(openmx_exx_c_neighbours(state, ATOMS) == -1 && openmx_exx_v_neighbours(state, -1) == -1);
  CHECK(openmx_exx_report(state, &report) == 0 && report.systems > 0);
  CHECK(report.eigen_decompositions >= 0 && report.dropped >= 0);
  CHECK(report.smallest_pivot > 0.0 && report.smallest_pivot <= 1.0);
  CHECK(openmx_exx_report(state, NULL) != 0);

  /* The exchange matrix of a symmetric density matrix is symmetric. */
  CHECK(openmx_exx_add_gamma(state, 0, ATOMS, -0.25, d, k, NULL, message, sizeof message) == 0);
  CHECK(message[0] == '\0');
  scale = largest(k, N * N);
  CHECK(scale > 1.0e-6);
  for (i = 0; i < N; ++i)
    for (j = 0; j < i; ++j) CHECK(fabs(k[i * N + j] - k[j * N + i]) <= 1.0e-12 * scale);

  /* The atoms one by one, as several processes would compute them. */
  memset(other, 0, sizeof other);
  for (a = 0; a < ATOMS; ++a) CHECK(openmx_exx_add_gamma(state, a, a + 1, -0.25, d, other, NULL, NULL, 0) == 0);
  CHECK(memcmp(k, other, sizeof k) == 0);
  CHECK(openmx_exx_add_gamma(state, 2, 2, -0.25, d, other, NULL, NULL, 0) == 0);
  CHECK(memcmp(k, other, sizeof k) == 0);

  /* The product of the caller. */
  table.dgemm = counting_dgemm;
  table.context = &calls;
  memset(other, 0, sizeof other);
  CHECK(openmx_exx_add_gamma(state, 0, ATOMS, -0.25, d, other, &table, message, sizeof message) == 0);
  CHECK(calls > 0 && memcmp(k, other, sizeof k) == 0);
  table.dgemm = failing_dgemm;
  table.context = NULL;
  CHECK(openmx_exx_add_gamma(state, 0, ATOMS, -0.25, d, other, &table, message, sizeof message) != 0);
  CHECK(strlen(message) > 0);

  /* Equations (4) and (15).  The exchange energy of occupied orbitals is
     negative. */
  sum = 0.0;
  for (i = 0; i < N * N; ++i) sum += d[i] * k[i];
  CHECK(sum < 0.0);
  CHECK(close_to(openmx_exx_energy_gamma(N * N, d, k), 0.5 * sum, 1.0e-14));
  CHECK(close_to(openmx_exx_change_gamma(N * N, d, k, zero), sum, 1.0e-14));
  CHECK(openmx_exx_change_gamma(N * N, d, k, k) == 0.0);
  CHECK(openmx_exx_energy_gamma(N * N, NULL, k) == 0.0 && openmx_exx_change_gamma(N * N, d, k, NULL) == 0.0);

  /* Rejected calls add nothing. */
  memcpy(other, k, sizeof k);
  CHECK(openmx_exx_add_gamma(state, -1, ATOMS, -0.25, d, other, NULL, message, sizeof message) != 0);
  CHECK(openmx_exx_add_gamma(state, 0, ATOMS + 1, -0.25, d, other, NULL, message, sizeof message) != 0);
  CHECK(openmx_exx_add_gamma(state, 2, 1, -0.25, d, other, NULL, message, sizeof message) != 0);
  CHECK(openmx_exx_add_gamma(state, 0, ATOMS, -0.25, NULL, other, NULL, message, sizeof message) != 0);
  CHECK(openmx_exx_add_gamma(state, 0, ATOMS, -0.25, d, NULL, NULL, message, sizeof message) != 0);
  CHECK(memcmp(k, other, sizeof k) == 0);
  /* A short buffer receives the beginning of the message, terminated. */
  memset(tiny, 'x', sizeof tiny);
  CHECK(openmx_exx_add_gamma(state, 0, ATOMS + 1, -0.25, d, other, NULL, tiny, sizeof tiny) != 0);
  CHECK(tiny[3] == '\0' && strlen(tiny) == 3);

  /* A preparation that fails leaves the prepared structure in place. */
  options.kernel = 99;
  CHECK(openmx_exx_prepare(state, &options, &geometry, message, sizeof message) != 0 && strlen(message) > 0);
  options.kernel = OPENMX_EXX_KERNEL_SHORT_RANGE;
  options.kernel_parameter = -1.0;
  CHECK(openmx_exx_prepare(state, &options, &geometry, message, sizeof message) != 0 && strlen(message) > 0);
  options.kernel_parameter = 0.11;
  geometry.species = wrong_species;
  CHECK(openmx_exx_prepare(state, &options, &geometry, message, sizeof message) != 0 && strlen(message) > 0);
  geometry.species = atom_species;
  geometry.atom_count = 0;
  CHECK(openmx_exx_prepare(state, &options, &geometry, message, sizeof message) != 0 && strlen(message) > 0);
  geometry.atom_count = ATOMS;
  species[1].mesh_size = 1;
  CHECK(openmx_exx_prepare(state, &options, &geometry, message, sizeof message) != 0 && strlen(message) > 0);
  species[1].mesh_size = POINTS;
  CHECK(openmx_exx_orbital_count(state) == N);
  memset(other, 0, sizeof other);
  CHECK(openmx_exx_add_gamma(state, 0, ATOMS, -0.25, d, other, NULL, message, sizeof message) == 0);
  CHECK(memcmp(k, other, sizeof k) == 0);

  /* The kernel of PBE0 gives another matrix for the same structure. */
  options.kernel = OPENMX_EXX_KERNEL_TRUNCATED_COULOMB;
  options.kernel_parameter = 10.0;
  CHECK(openmx_exx_prepare(state, &options, &geometry, message, sizeof message) == 0);
  memset(truncated, 0, sizeof truncated);
  CHECK(openmx_exx_add_gamma(state, 0, ATOMS, -0.25, d, truncated, NULL, message, sizeof message) == 0);
  for (i = 0; i < N * N; ++i) other[i] = truncated[i] - k[i];
  CHECK(largest(other, N * N) > 1.0e-3 * scale);
  for (i = 0; i < N; ++i)
    for (j = 0; j < i; ++j) CHECK(fabs(truncated[i * N + j] - truncated[j * N + i]) <= 1.0e-12 * largest(truncated, N * N));

  openmx_exx_destroy(state);
  printf("C interface of the exact exchange: %d orbitals, largest element of K %.3e (HSE06 kernel), %ld products\n",
         N, scale, calls);
  return 0;
}
