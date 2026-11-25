/* Test of the C interface, compiled as C: the header must be valid C, and a C
   program must be able to prepare a structure, build the exchange matrix of
   the Gamma point and that of the site form, take the derivative of E_x with
   respect to the positions of the atoms, and pass its own matrix product,
   its own loop for the pieces of the build and its own device, and divide
   the builds, the preparation and the derivative as processes do.  That the
   numbers are those of the C++ interface is checked by exx-exchange-system
   and exx-exchange-forces.  The folding of the V neighbours by the period of
   a k mesh is checked with the period 1 against the Gamma form. */
#include <openmx/exx/exx.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { POINTS = 160, ATOMS = 3, N = 7 };

/* Always active, including Release/NDEBUG. */
#define CHECK(expression) do { if (!(expression)) { \
  fprintf(stderr,"line %d: %s\n",__LINE__,#expression); openmx_exx_destroy(state); free(site_d); return 1; } } while (0)

static int close_to(double a, double b, double relative)
{
  const double scale = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
  return fabs(a - b) <= relative * (scale > 1e-300 ? scale : 1.0);
}

/* The context of the product and the loops of the caller: the products, the
   loops, and the smallest and the largest number of calls that a loop was
   asked for.  The library calls a loop from the thread that called the
   library, one loop at a time, so the loops count without protection. */
typedef struct {
  long products;
  long loops;
  int smallest, largest;
} Calls;

static void forget_loops(Calls *calls)
{
  calls->loops = 0;
  calls->smallest = 0;
  calls->largest = 0;
}

/* The product of the caller: it counts its calls in the context and leaves
   the work to the product of the library. */
static int counting_dgemm(void *context, char transa, char transb, int m, int n, int k, double alpha,
                          const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc)
{
  ++((Calls *)context)->products;
  return openmx_exx_reference_dgemm(NULL, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static int failing_dgemm(void *context, char transa, char transb, int m, int n, int k, double alpha,
                         const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc)
{
  (void)context; (void)transa; (void)transb; (void)m; (void)n; (void)k; (void)alpha;
  (void)a; (void)lda; (void)b; (void)ldb; (void)beta; (void)c; (void)ldc;
  return 3;
}

/* The loops of the caller: they note their calls in the context and run the
   pieces on the calling thread, in order or from the last to the first. */
static void note_loop(void *context, int count)
{
  Calls *calls = (Calls *)context;
  if (calls->loops == 0 || count < calls->smallest) calls->smallest = count;
  if (calls->loops == 0 || count > calls->largest) calls->largest = count;
  ++calls->loops;
}

static int ordered_loop(void *context, int count, void (*body)(void *argument, int index), void *argument)
{
  int index;
  note_loop(context, count);
  for (index = 0; index < count; ++index) body(argument, index);
  return 0;
}

static int reversed_loop(void *context, int count, void (*body)(void *argument, int index), void *argument)
{
  int index;
  note_loop(context, count);
  for (index = count - 1; index >= 0; --index) body(argument, index);
  return 0;
}

static int failing_loop(void *context, int count, void (*body)(void *argument, int index), void *argument)
{
  reversed_loop(context, count, body, argument);
  return 2;
}

/* A device of the caller in host memory (OpenMXExxDevice): the blocks come
   from malloc, dgemm is the product of the library, and move follows exx.h.
   It counts its calls and lets one of them fail: the call "failing", counted
   from 0 after accept, release not counted, or none for -1. */
typedef struct {
  int answer;              /* of accept */
  long failing;
  long accepted, calls, allocations, releases;
} Device;

static void forget_device(Device *device, int answer, long failing)
{
  device->answer = answer;
  device->failing = failing;
  device->accepted = 0;
  device->calls = 0;
  device->allocations = 0;
  device->releases = 0;
}

static int device_fails(void *context)
{
  Device *device = (Device *)context;
  return device->calls++ == device->failing;
}

static int device_accept(void *context, double multiplications, size_t bytes)
{
  Device *device = (Device *)context;
  ++device->accepted;
  return multiplications > 0.0 && bytes > 0 ? device->answer : 0;
}

static void *device_allocate(void *context, size_t bytes)
{
  void *pointer;
  if (device_fails(context)) return NULL;
  pointer = malloc(bytes > 0 ? bytes : 1);
  if (pointer != NULL) ++((Device *)context)->allocations;
  return pointer;
}

static void device_release(void *context, void *pointer)
{
  ++((Device *)context)->releases;
  free(pointer);
}

static int device_copy(void *context, void *target, const void *source, size_t bytes)
{
  if (device_fails(context)) return 1;
  memcpy(target, source, bytes);
  return 0;
}

static int device_dgemm(void *context, char transa, char transb, int m, int n, int k, double alpha,
                        const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc)
{
  if (device_fails(context)) return 1;
  return openmx_exx_reference_dgemm(NULL, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static long long device_offset(const OpenMXExxDeviceIndex *index, int r, int c)
{
  return (index->rows != NULL ? index->rows[r] : r * index->stride) +
         (index->columns != NULL ? index->columns[c] : c * index->step);
}

static int device_move(void *context, int rows, int columns, const int *lengths, double alpha, int add,
                       double *target, const OpenMXExxDeviceIndex *target_index, const double *source,
                       const OpenMXExxDeviceIndex *source_index)
{
  int r, c;
  if (device_fails(context)) return 1;
  for (r = 0; r < rows; ++r)
    for (c = 0; c < (lengths != NULL ? lengths[r] : columns); ++c) {
      double *element = target + device_offset(target_index, r, c);
      const double value = alpha * source[device_offset(source_index, r, c)];
      *element = add ? *element + value : value;
    }
  return 0;
}

static int device_finish(void *context)
{
  return device_fails(context) ? 1 : 0;
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

/* The entries for processes that divide the work (section 6.7 of the guide):
   builds over ranges of ABFs, the preparation in shares, and the derivative
   of E_x by the weights of the blocks and the pairs of sites.  whole is a
   prepared structure with the list of its blocks; k, site_k and gradient are
   what the calls for all atoms gave with the factor -0.25.  Returns 0 when
   every check holds and the line of the first one that fails otherwise. */
static int divided_work(const OpenMXExxState *whole, const OpenMXExxOptions *options,
                        const OpenMXExxGeometry *geometry, const double *d, const double *k, const double *site_d,
                        const double *site_k, size_t elements, const double *gradient)
{
#define EXPECT(expression) do { if (!(expression)) { \
  fprintf(stderr,"line %d: %s\n",__LINE__,#expression); failed = __LINE__; goto done; } } while (0)
  const int share_counts[3] = {1, 2, 5}, pair_counts[4] = {1, 2, 3, 7};
  const double scale = largest(k, N * N), site_scale = largest(site_k, (int)elements);
  const double steepest = largest(gradient, 3 * ATOMS);
  OpenMXExxState *share = NULL;
  OpenMXExxReport report, share_report;
  OpenMXExxDenseBackend table = {0};
  Calls calls = {0, 0, 0, 0};
  double *tables[ATOMS] = {NULL, NULL, NULL}, *weights[ATOMS] = {NULL, NULL, NULL};
  double *sums[ATOMS], *piece[ATOMS], *site_weights[ATOMS], *holed[ATOMS];
  const double *read[ATOMS], *site_read[ATOMS], *missing[ATOMS];
  double *site_other = NULL;
  double other[N * N], paired[3 * ATOMS], once[3 * ATOMS], again[3 * ATOMS];
  size_t size[ATOMS], e;
  char message[160];
  long total, count, cut, p, systems, first_abf[ATOMS + 1], boundaries[8];
  int failed = 0, a, i, j, trial, part, parts;

  /* The ABFs of the unit cell are those of its atoms.  Up to six processes
     divide them into contiguous ranges that cover them in order, in either
     form and for the whole and the half build, and one process takes them
     all.  The division writes processes + 1 numbers and no more. */
  total = openmx_exx_abf_count(whole);
  first_abf[0] = 0;
  for (a = 0; a < ATOMS; ++a) first_abf[a + 1] = first_abf[a] + openmx_exx_atom_abfs(whole, a);
  EXPECT(total == first_abf[ATOMS] && total > ATOMS && openmx_exx_abf_count(NULL) == -1);
  for (i = 0; i < 4; ++i) {
    for (parts = 1; parts <= 6; ++parts) {
      for (j = 0; j < 8; ++j) boundaries[j] = -1;
      EXPECT(openmx_exx_divide_abfs(whole, i & 1, i >> 1, parts, boundaries) == 0);
      EXPECT(boundaries[0] == 0 && boundaries[parts] == total && boundaries[parts + 1] == -1);
      for (j = 0; j < parts; ++j) EXPECT(boundaries[j] <= boundaries[j + 1]);
      if (parts == 1) EXPECT(boundaries[1] == total);
      if (parts == 2) EXPECT(0 < boundaries[1] && boundaries[1] < total);
    }
  }
  EXPECT(openmx_exx_divide_abfs(NULL, 0, 0, 2, boundaries) != 0 && openmx_exx_divide_abfs(whole, 0, 0, 2, NULL) != 0);
  EXPECT(openmx_exx_divide_abfs(whole, 0, 0, 0, boundaries) != 0 && openmx_exx_divide_abfs(whole, 1, 1, -1, boundaries) != 0);

  /* The range of all ABFs is the range of all atoms, bit for bit. */
  site_other = (double *)malloc(elements * sizeof *site_other);
  EXPECT(site_other != NULL);
  memset(other, 0, sizeof other);
  EXPECT(openmx_exx_add_gamma_abfs(whole, 0, total, -0.25, d, other, NULL, message, sizeof message) == 0);
  EXPECT(message[0] == '\0' && memcmp(other, k, sizeof other) == 0);
  memset(site_other, 0, elements * sizeof *site_other);
  EXPECT(openmx_exx_add_blocks_abfs(whole, 0, total, -0.25, site_d, site_other, NULL, message, sizeof message) == 0);
  EXPECT(message[0] == '\0' && memcmp(site_other, site_k, elements * sizeof *site_other) == 0);
  /* An empty range adds nothing.  Ranges that are not inside the unit cell
     and missing arrays are rejected, and add nothing either. */
  EXPECT(openmx_exx_add_gamma_abfs(whole, 1, 1, -0.25, d, other, NULL, message, sizeof message) == 0);
  EXPECT(openmx_exx_add_gamma_abfs(whole, total, total, -0.25, d, other, NULL, message, sizeof message) == 0);
  EXPECT(openmx_exx_add_blocks_abfs(whole, 1, 1, -0.25, site_d, site_other, NULL, message, sizeof message) == 0);
  EXPECT(message[0] == '\0');
  EXPECT(openmx_exx_add_gamma_abfs(whole, -1, 2, -0.25, d, other, NULL, message, sizeof message) != 0);
  EXPECT(strlen(message) > 0);
  EXPECT(openmx_exx_add_gamma_abfs(whole, 2, 1, -0.25, d, other, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_gamma_abfs(whole, 0, total + 1, -0.25, d, other, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_gamma_abfs(NULL, 0, total, -0.25, d, other, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_gamma_abfs(whole, 0, total, -0.25, NULL, other, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_gamma_abfs(whole, 0, total, -0.25, d, NULL, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_blocks_abfs(whole, -1, 2, -0.25, site_d, site_other, NULL, message, sizeof message) != 0);
  EXPECT(strlen(message) > 0);
  EXPECT(openmx_exx_add_blocks_abfs(whole, 2, 1, -0.25, site_d, site_other, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_blocks_abfs(whole, 0, total + 1, -0.25, site_d, site_other, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_blocks_abfs(NULL, 0, total, -0.25, site_d, site_other, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_blocks_abfs(whole, 0, total, -0.25, NULL, site_other, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_blocks_abfs(whole, 0, total, -0.25, site_d, NULL, NULL, NULL, 0) != 0);
  EXPECT(memcmp(other, k, sizeof other) == 0);
  EXPECT(memcmp(site_other, site_k, elements * sizeof *site_other) == 0);
  /* Two ranges that meet anywhere, inside an atom as well, add up to K
     within rounding. */
  for (cut = 1; cut < total; ++cut) {
    memset(other, 0, sizeof other);
    EXPECT(openmx_exx_add_gamma_abfs(whole, cut, total, -0.25, d, other, NULL, NULL, 0) == 0);
    EXPECT(openmx_exx_add_gamma_abfs(whole, 0, cut, -0.25, d, other, NULL, NULL, 0) == 0);
    for (i = 0; i < N * N; ++i) EXPECT(fabs(other[i] - k[i]) <= 1.0e-12 * scale);
    memset(site_other, 0, elements * sizeof *site_other);
    EXPECT(openmx_exx_add_blocks_abfs(whole, 0, cut, -0.25, site_d, site_other, NULL, NULL, 0) == 0);
    EXPECT(openmx_exx_add_blocks_abfs(whole, cut, total, -0.25, site_d, site_other, NULL, NULL, 0) == 0);
    for (e = 0; e < elements; ++e) EXPECT(fabs(site_other[e] - site_k[e]) <= 1.0e-12 * site_scale);
  }
  /* So do the ABFs one by one, with the product of the caller. */
  table.context = &calls;
  table.dgemm = counting_dgemm;
  memset(other, 0, sizeof other);
  memset(site_other, 0, elements * sizeof *site_other);
  for (p = 0; p < total; ++p) {
    EXPECT(openmx_exx_add_gamma_abfs(whole, p, p + 1, -0.25, d, other, &table, NULL, 0) == 0);
    EXPECT(openmx_exx_add_blocks_abfs(whole, p, p + 1, -0.25, site_d, site_other, &table, NULL, 0) == 0);
  }
  EXPECT(calls.products > 0);
  for (i = 0; i < N * N; ++i) EXPECT(fabs(other[i] - k[i]) <= 1.0e-12 * scale);
  for (e = 0; e < elements; ++e) EXPECT(fabs(site_other[e] - site_k[e]) <= 1.0e-12 * site_scale);

  /* The half build (section 6.4 of the guide): its sum X over two ranges
     gives X + X^T, which is K within rounding and symmetric to the last bit;
     d and site_d are symmetric. */
  cut = total / 2;
  memset(other, 0, sizeof other);
  EXPECT(openmx_exx_add_gamma_half_abfs(whole, cut, total, -0.25, d, other, NULL, message, sizeof message) == 0);
  EXPECT(openmx_exx_add_gamma_half_abfs(whole, 0, cut, -0.25, d, other, NULL, message, sizeof message) == 0);
  EXPECT(openmx_exx_mirror_gamma(whole, other, message, sizeof message) == 0 && message[0] == '\0');
  for (i = 0; i < N * N; ++i) EXPECT(fabs(other[i] - k[i]) <= 1.0e-12 * scale);
  for (i = 0; i < N; ++i)
    for (j = i + 1; j < N; ++j) EXPECT(other[i * N + j] == other[j * N + i]);
  memset(site_other, 0, elements * sizeof *site_other);
  EXPECT(openmx_exx_add_blocks_half_abfs(whole, 0, cut, -0.25, site_d, site_other, NULL, message, sizeof message) == 0);
  EXPECT(openmx_exx_add_blocks_half_abfs(whole, cut, total, -0.25, site_d, site_other, NULL, message, sizeof message) == 0);
  EXPECT(openmx_exx_mirror_blocks(whole, site_other, message, sizeof message) == 0 && message[0] == '\0');
  for (e = 0; e < elements; ++e) EXPECT(fabs(site_other[e] - site_k[e]) <= 1.0e-12 * site_scale);
  /* Rejected calls leave a message. */
  EXPECT(openmx_exx_add_gamma_half_abfs(whole, 2, 1, -0.25, d, other, NULL, message, sizeof message) != 0);
  EXPECT(strlen(message) > 0);
  EXPECT(openmx_exx_add_gamma_half_abfs(NULL, 0, total, -0.25, d, other, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_blocks_half_abfs(whole, 0, total + 1, -0.25, site_d, site_other, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_blocks_half_abfs(whole, 0, total, -0.25, NULL, site_other, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_mirror_gamma(whole, NULL, message, sizeof message) != 0 && strlen(message) > 0);
  EXPECT(openmx_exx_mirror_gamma(NULL, other, NULL, 0) != 0 && openmx_exx_mirror_blocks(NULL, site_other, NULL, 0) != 0);
  EXPECT(openmx_exx_mirror_blocks(whole, NULL, NULL, 0) != 0);

  /* The preparation in shares.  The T and V blocks of an atom are one array
     of a size that is known without a share. */
  share = openmx_exx_create();
  EXPECT(share != NULL);
  EXPECT(openmx_exx_atom_table_elements(NULL, 0) == 0 && openmx_exx_atom_table_elements(share, 0) == 0);
  EXPECT(openmx_exx_atom_table_elements(whole, -1) == 0 && openmx_exx_atom_table_elements(whole, ATOMS) == 0);
  for (a = 0; a < ATOMS; ++a) {
    size[a] = openmx_exx_atom_table_elements(whole, a);
    EXPECT(size[a] > 0);
    tables[a] = (double *)malloc(3 * size[a] * sizeof **tables);
    EXPECT(tables[a] != NULL);
    sums[a] = tables[a] + size[a];
    piece[a] = sums[a] + size[a];
    EXPECT(openmx_exx_get_atom_tables(whole, a, tables[a]) == 0);
  }
  EXPECT(openmx_exx_get_atom_tables(NULL, 0, piece[0]) != 0 && openmx_exx_get_atom_tables(share, 0, piece[0]) != 0);
  EXPECT(openmx_exx_get_atom_tables(whole, ATOMS, piece[0]) != 0 && openmx_exx_get_atom_tables(whole, 0, NULL) != 0);
  EXPECT(openmx_exx_set_atom_tables(NULL, 0, tables[0]) != 0 && openmx_exx_set_atom_tables(share, 0, tables[0]) != 0);
  EXPECT(openmx_exx_report(whole, &report) == 0);
  for (trial = 0; trial < 3; ++trial) {
    parts = share_counts[trial];
    systems = 0;
    for (a = 0; a < ATOMS; ++a) memset(sums[a], 0, size[a] * sizeof **sums);
    for (part = 0; part < parts; ++part) {
      EXPECT(openmx_exx_prepare_part(share, options, geometry, part, parts, message, sizeof message) == 0);
      EXPECT(message[0] == '\0');
      EXPECT(openmx_exx_report(share, &share_report) == 0);
      systems += share_report.systems;
      for (a = 0; a < ATOMS; ++a) {
        EXPECT(openmx_exx_atom_table_elements(share, a) == size[a]);
        EXPECT(openmx_exx_get_atom_tables(share, a, piece[a]) == 0);
        for (e = 0; e < size[a]; ++e) sums[a][e] += piece[a][e];
      }
    }
    /* Every pair of atoms belongs to one share, so the sums over the shares
       are the blocks of the whole preparation, without rounding, and the
       shares solved its systems between them. */
    EXPECT(systems == report.systems);
    for (a = 0; a < ATOMS; ++a)
      for (e = 0; e < size[a]; ++e) EXPECT(sums[a][e] == tables[a][e]);
    /* The state of the last share knows the sizes.  With several shares it
       refuses the builds until the sums of all atoms are stored. */
    EXPECT(openmx_exx_abf_count(share) == total && openmx_exx_orbital_count(share) == N);
    for (a = 0; a < ATOMS && parts > 1; ++a) {
      memcpy(other, k, sizeof other);
      EXPECT(openmx_exx_add_gamma(share, 0, ATOMS, -0.25, d, other, NULL, message, sizeof message) != 0);
      EXPECT(strlen(message) > 0);
      EXPECT(openmx_exx_add_gamma_abfs(share, 0, total, -0.25, d, other, NULL, NULL, 0) != 0);
      EXPECT(memcmp(other, k, sizeof other) == 0);
      EXPECT(openmx_exx_set_atom_tables(share, a, sums[a]) == 0);
    }
    memset(other, 0, sizeof other);
    EXPECT(openmx_exx_add_gamma(share, 0, ATOMS, -0.25, d, other, NULL, message, sizeof message) == 0);
    for (i = 0; i < N * N; ++i) EXPECT(other[i] == k[i]);
  }
  /* The pairs of a share in the pieces of a loop of the caller: the blocks
     and the report of the share without pieces, bit for bit.  A share that
     does not exist is rejected.  The sums stored afterwards make the state
     the whole preparation again. */
  {
    Calls loop_calls = {0, 0, 0, 0};
    OpenMXExxDenseBackend loops = {NULL, NULL, reversed_loop, 3, NULL};
    OpenMXExxReport plain_report, pieces_report;
    loops.context = &loop_calls;
    for (part = 0; part < 2; ++part) {
      EXPECT(openmx_exx_prepare_part(share, options, geometry, part, 2, NULL, 0) == 0);
      EXPECT(openmx_exx_report(share, &plain_report) == 0);
      for (a = 0; a < ATOMS; ++a) EXPECT(openmx_exx_get_atom_tables(share, a, sums[a]) == 0);
      EXPECT(openmx_exx_prepare_part_pieces(share, options, geometry, part, 2, &loops, message, sizeof message) == 0);
      EXPECT(message[0] == '\0');
      EXPECT(openmx_exx_report(share, &pieces_report) == 0);
      EXPECT(pieces_report.systems == plain_report.systems &&
             pieces_report.eigen_decompositions == plain_report.eigen_decompositions &&
             pieces_report.dropped == plain_report.dropped &&
             pieces_report.smallest_pivot == plain_report.smallest_pivot);
      for (a = 0; a < ATOMS; ++a) {
        EXPECT(openmx_exx_get_atom_tables(share, a, piece[a]) == 0);
        EXPECT(memcmp(piece[a], sums[a], size[a] * sizeof **piece) == 0);
      }
    }
    EXPECT(loop_calls.loops > 0 && loop_calls.largest <= 3);
    EXPECT(openmx_exx_prepare_part_pieces(share, options, geometry, 2, 2, &loops, message, sizeof message) != 0);
    EXPECT(strlen(message) > 0);
    EXPECT(openmx_exx_prepare_part_pieces(share, options, geometry, 0, 1, NULL, message, sizeof message) == 0);
    EXPECT(message[0] == '\0');
    for (a = 0; a < ATOMS; ++a) {
      EXPECT(openmx_exx_get_atom_tables(share, a, piece[a]) == 0);
      EXPECT(memcmp(piece[a], tables[a], size[a] * sizeof **piece) == 0);
    }
  }
  /* The site form of the state that holds the sums. */
  EXPECT(openmx_exx_prepare_blocks(share, message, sizeof message) == 0);
  memset(site_other, 0, elements * sizeof *site_other);
  EXPECT(openmx_exx_add_blocks(share, 0, ATOMS, -0.25, site_d, site_other, NULL, message, sizeof message) == 0);
  for (e = 0; e < elements; ++e) EXPECT(site_other[e] == site_k[e]);
  /* Shares and arrays that do not exist are rejected, and the state holds
     what it held. */
  EXPECT(openmx_exx_prepare_part(share, options, geometry, 2, 2, message, sizeof message) != 0);
  EXPECT(strlen(message) > 0);
  EXPECT(openmx_exx_prepare_part(share, options, geometry, -1, 2, NULL, 0) != 0);
  EXPECT(openmx_exx_prepare_part(share, options, geometry, 0, 0, NULL, 0) != 0);
  EXPECT(openmx_exx_prepare_part(NULL, options, geometry, 0, 2, NULL, 0) != 0);
  EXPECT(openmx_exx_set_atom_tables(share, ATOMS, sums[0]) != 0 && openmx_exx_set_atom_tables(share, 0, NULL) != 0);
  memset(site_other, 0, elements * sizeof *site_other);
  EXPECT(openmx_exx_add_blocks(share, 0, ATOMS, -0.25, site_d, site_other, NULL, message, sizeof message) == 0);
  for (e = 0; e < elements; ++e) EXPECT(site_other[e] == site_k[e]);

  /* The derivative of E_x in two steps: the weights of the T and V blocks
     from the density matrix, for all ABFs here, then the pairs of sites, in
     one share or in several. */
  for (a = 0; a < ATOMS; ++a) {
    weights[a] = (double *)calloc(2 * size[a], sizeof **weights);
    EXPECT(weights[a] != NULL);
    site_weights[a] = weights[a] + size[a];
    read[a] = weights[a];
    site_read[a] = site_weights[a];
  }
  EXPECT(openmx_exx_add_block_weights_gamma_abfs(whole, 0, total, -0.25, d, weights, NULL, message, sizeof message) == 0);
  EXPECT(message[0] == '\0');
  for (trial = 0; trial < 4; ++trial) {
    parts = pair_counts[trial];
    memset(paired, 0, sizeof paired);
    for (part = 0; part < parts; ++part)
      EXPECT(openmx_exx_add_position_gradient_pairs(whole, read, part, parts, paired, message, sizeof message) == 0);
    EXPECT(message[0] == '\0');
    for (i = 0; i < 3 * ATOMS; ++i) EXPECT(fabs(paired[i] - gradient[i]) <= 1.0e-11 * steepest);
    if (parts == 1) memcpy(once, paired, sizeof once);
  }
  /* The weights of the site form with the product of the caller, in two
     ranges of ABFs that meet inside the first atom. */
  count = calls.products;
  EXPECT(first_abf[1] > 1);
  EXPECT(openmx_exx_add_block_weights_blocks_abfs(whole, 1, total, -0.25, site_d, site_weights, &table, NULL, 0) == 0);
  EXPECT(openmx_exx_add_block_weights_blocks_abfs(whole, 0, 1, -0.25, site_d, site_weights, &table, NULL, 0) == 0);
  EXPECT(calls.products > count);
  memset(paired, 0, sizeof paired);
  EXPECT(openmx_exx_add_position_gradient_pairs(whole, site_read, 0, 1, paired, NULL, 0) == 0);
  for (i = 0; i < 3 * ATOMS; ++i) EXPECT(fabs(paired[i] - gradient[i]) <= 1.0e-11 * steepest);
  /* The pairs in the pieces of a loop of the caller give the bits of no
     pieces, and a share that does not exist adds nothing. */
  {
    Calls loop_calls = {0, 0, 0, 0};
    OpenMXExxDenseBackend loops = {NULL, NULL, ordered_loop, 2, NULL};
    loops.context = &loop_calls;
    memset(paired, 0, sizeof paired);
    memset(again, 0, sizeof again);
    for (part = 0; part < 2; ++part) {
      EXPECT(openmx_exx_add_position_gradient_pairs(whole, site_read, part, 2, paired, NULL, 0) == 0);
      EXPECT(openmx_exx_add_position_gradient_pairs_pieces(whole, site_read, part, 2, &loops, again, message,
                                                           sizeof message) == 0);
      EXPECT(message[0] == '\0');
    }
    EXPECT(memcmp(again, paired, sizeof again) == 0);
    EXPECT(loop_calls.loops > 0 && loop_calls.largest <= 2);
    EXPECT(openmx_exx_add_position_gradient_pairs_pieces(whole, site_read, 2, 2, &loops, again, message,
                                                         sizeof message) != 0);
    EXPECT(strlen(message) > 0);
    EXPECT(memcmp(again, paired, sizeof again) == 0);
  }
  /* Rejected calls add nothing, neither to the derivative nor to the
     weights. */
  for (a = 0; a < ATOMS; ++a) {
    missing[a] = a == 1 ? NULL : read[a];
    holed[a] = a == 1 ? NULL : weights[a];
  }
  memcpy(paired, once, sizeof paired);
  EXPECT(openmx_exx_add_position_gradient_pairs(NULL, read, 0, 1, paired, message, sizeof message) != 0);
  EXPECT(strlen(message) > 0);
  EXPECT(openmx_exx_add_position_gradient_pairs(whole, NULL, 0, 1, paired, NULL, 0) != 0);
  EXPECT(openmx_exx_add_position_gradient_pairs(whole, missing, 0, 1, paired, NULL, 0) != 0);
  EXPECT(openmx_exx_add_position_gradient_pairs(whole, read, 0, 1, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_position_gradient_pairs(whole, read, 1, 1, paired, NULL, 0) != 0);
  EXPECT(openmx_exx_add_position_gradient_pairs(whole, read, -1, 2, paired, NULL, 0) != 0);
  EXPECT(openmx_exx_add_position_gradient_pairs(whole, read, 0, 0, paired, NULL, 0) != 0);
  EXPECT(memcmp(paired, once, sizeof paired) == 0);
  EXPECT(openmx_exx_add_block_weights_gamma_abfs(NULL, 0, total, -0.25, d, weights, NULL, message, sizeof message) != 0);
  EXPECT(strlen(message) > 0);
  EXPECT(openmx_exx_add_block_weights_gamma_abfs(whole, 0, total + 1, -0.25, d, weights, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_gamma_abfs(whole, -1, 2, -0.25, d, weights, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_gamma_abfs(whole, 2, 1, -0.25, d, weights, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_gamma_abfs(whole, 0, total, -0.25, NULL, weights, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_gamma_abfs(whole, 0, total, -0.25, d, NULL, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_gamma_abfs(whole, 0, total, -0.25, d, holed, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_blocks_abfs(NULL, 0, total, -0.25, site_d, weights, NULL, message, sizeof message) != 0);
  EXPECT(strlen(message) > 0);
  EXPECT(openmx_exx_add_block_weights_blocks_abfs(whole, 0, total + 1, -0.25, site_d, weights, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_blocks_abfs(whole, -1, 2, -0.25, site_d, weights, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_blocks_abfs(whole, 2, 1, -0.25, site_d, weights, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_blocks_abfs(whole, 0, total, -0.25, NULL, weights, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_blocks_abfs(whole, 0, total, -0.25, site_d, NULL, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_blocks_abfs(whole, 0, total, -0.25, site_d, holed, NULL, NULL, 0) != 0);
  /* An empty range adds nothing. */
  EXPECT(openmx_exx_add_block_weights_gamma_abfs(whole, 1, 1, -0.25, d, weights, NULL, message, sizeof message) == 0);
  EXPECT(openmx_exx_add_block_weights_blocks_abfs(whole, total, total, -0.25, site_d, weights, NULL, message, sizeof message) == 0);
  EXPECT(message[0] == '\0');
  memset(again, 0, sizeof again);
  EXPECT(openmx_exx_add_position_gradient_pairs(whole, read, 0, 1, again, NULL, 0) == 0);
  EXPECT(memcmp(again, once, sizeof again) == 0);
  /* The arrays of the atoms that a range does not touch are not read, and
     the weights of several calls add up: for every atom two calls with its
     ABFs and half the factor, as for two spin channels with the same density
     matrix. */
  for (a = 0; a < ATOMS; ++a) {
    memset(site_weights[a], 0, size[a] * sizeof **site_weights);
    for (i = 0; i < ATOMS; ++i) holed[i] = i == a ? site_weights[i] : NULL;
    EXPECT(openmx_exx_add_block_weights_gamma_abfs(whole, first_abf[a], first_abf[a + 1], -0.125, d, holed, NULL, NULL,
                                                   0) == 0);
    EXPECT(openmx_exx_add_block_weights_gamma_abfs(whole, first_abf[a], first_abf[a + 1], -0.125, d, holed, &table,
                                                   NULL, 0) == 0);
  }
  memset(paired, 0, sizeof paired);
  EXPECT(openmx_exx_add_position_gradient_pairs(whole, site_read, 0, 1, paired, NULL, 0) == 0);
  for (i = 0; i < 3 * ATOMS; ++i) EXPECT(fabs(paired[i] - gradient[i]) <= 1.0e-11 * steepest);
  /* The half form of the weights: other arrays than those above, the same
     derivative through the pairs. The Gamma form comes in two ranges that
     meet inside the first atom, one of them with the product of the
     caller. */
  for (a = 0; a < ATOMS; ++a) memset(weights[a], 0, 2 * size[a] * sizeof **weights);
  EXPECT(openmx_exx_add_block_weights_gamma_abfs(whole, 0, total, -0.25, d, site_weights, NULL, NULL, 0) == 0);
  EXPECT(openmx_exx_add_block_weights_gamma_half_abfs(whole, 0, 1, -0.25, d, weights, NULL, message,
                                                      sizeof message) == 0);
  EXPECT(message[0] == '\0');
  count = calls.products;
  EXPECT(openmx_exx_add_block_weights_gamma_half_abfs(whole, 1, total, -0.25, d, weights, &table, NULL, 0) == 0);
  EXPECT(calls.products > count);
  count = 0;
  for (a = 0; a < ATOMS; ++a)
    for (e = 0; e < size[a]; ++e) count += fabs(weights[a][e] - site_weights[a][e]) > 1.0e-6 ? 1 : 0;
  EXPECT(count > 0);
  for (a = 0; a < ATOMS; ++a) memset(site_weights[a], 0, size[a] * sizeof **site_weights);
  EXPECT(openmx_exx_add_block_weights_blocks_half_abfs(whole, 0, total, -0.25, site_d, site_weights, NULL, message,
                                                       sizeof message) == 0);
  EXPECT(message[0] == '\0');
  for (trial = 0; trial < 2; ++trial) {
    memset(paired, 0, sizeof paired);
    EXPECT(openmx_exx_add_position_gradient_pairs(whole, trial == 0 ? read : site_read, 0, 1, paired, NULL, 0) == 0);
    for (i = 0; i < 3 * ATOMS; ++i) EXPECT(fabs(paired[i] - gradient[i]) <= 1.0e-11 * steepest);
    if (trial == 0) memcpy(again, paired, sizeof again);
  }
  /* A range that is not empty needs the arrays of all atoms, also of those
     whose ABFs it does not hold; an empty range needs none. Rejected calls
     add nothing. */
  for (a = 0; a < ATOMS; ++a) holed[a] = a == 0 ? weights[a] : NULL;
  EXPECT(openmx_exx_add_block_weights_gamma_half_abfs(whole, 0, first_abf[1], -0.25, d, holed, NULL, message,
                                                      sizeof message) != 0);
  EXPECT(strlen(message) > 0);
  EXPECT(openmx_exx_add_block_weights_blocks_half_abfs(whole, 0, first_abf[1], -0.25, site_d, holed, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_gamma_half_abfs(NULL, 0, total, -0.25, d, weights, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_gamma_half_abfs(whole, 0, total + 1, -0.25, d, weights, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_gamma_half_abfs(whole, 0, total, -0.25, NULL, weights, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_gamma_half_abfs(whole, 0, total, -0.25, d, NULL, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_blocks_half_abfs(NULL, 0, total, -0.25, site_d, weights, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_blocks_half_abfs(whole, 2, 1, -0.25, site_d, weights, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_blocks_half_abfs(whole, -1, 2, -0.25, site_d, weights, NULL, NULL, 0) != 0);
  EXPECT(openmx_exx_add_block_weights_blocks_half_abfs(whole, 0, total, -0.25, NULL, weights, NULL, NULL, 0) != 0);
  for (a = 0; a < ATOMS; ++a) holed[a] = NULL;
  EXPECT(openmx_exx_add_block_weights_gamma_half_abfs(whole, 1, 1, -0.25, d, holed, NULL, message, sizeof message) == 0);
  EXPECT(openmx_exx_add_block_weights_blocks_half_abfs(whole, total, total, -0.25, site_d, holed, NULL, message,
                                                       sizeof message) == 0);
  EXPECT(message[0] == '\0');
  memset(paired, 0, sizeof paired);
  EXPECT(openmx_exx_add_position_gradient_pairs(whole, read, 0, 1, paired, NULL, 0) == 0);
  EXPECT(memcmp(paired, again, sizeof paired) == 0);
  /* A product that fails. */
  table.dgemm = failing_dgemm;
  EXPECT(openmx_exx_add_block_weights_gamma_abfs(whole, 0, total, -0.25, d, site_weights, &table, message,
                                                 sizeof message) != 0);
  EXPECT(strlen(message) > 0);
  EXPECT(openmx_exx_add_block_weights_blocks_abfs(whole, 0, total, -0.25, site_d, site_weights, &table, message,
                                                  sizeof message) != 0);
  EXPECT(strlen(message) > 0);
  EXPECT(openmx_exx_add_block_weights_gamma_half_abfs(whole, 0, total, -0.25, d, site_weights, &table, message,
                                                      sizeof message) != 0);
  EXPECT(strlen(message) > 0);
  EXPECT(openmx_exx_add_block_weights_blocks_half_abfs(whole, 0, total, -0.25, site_d, site_weights, &table, message,
                                                       sizeof message) != 0);
  EXPECT(strlen(message) > 0);
  /* A share whose blocks have not been added up has no derivative. */
  EXPECT(openmx_exx_prepare_part(share, options, geometry, 1, 2, message, sizeof message) == 0);
  EXPECT(openmx_exx_prepare_blocks(share, message, sizeof message) == 0);
  EXPECT(openmx_exx_add_block_weights_gamma_abfs(share, 0, total, -0.25, d, site_weights, NULL, message,
                                                 sizeof message) != 0);
  EXPECT(strlen(message) > 0);
  EXPECT(openmx_exx_add_block_weights_blocks_abfs(share, 0, total, -0.25, site_d, site_weights, NULL, message,
                                                  sizeof message) != 0);
  EXPECT(strlen(message) > 0);
  EXPECT(openmx_exx_add_block_weights_gamma_half_abfs(share, 0, total, -0.25, d, site_weights, NULL, message,
                                                      sizeof message) != 0);
  EXPECT(strlen(message) > 0);
  EXPECT(openmx_exx_add_block_weights_blocks_half_abfs(share, 0, total, -0.25, site_d, site_weights, NULL, message,
                                                       sizeof message) != 0);
  EXPECT(strlen(message) > 0);
  memcpy(paired, once, sizeof paired);
  EXPECT(openmx_exx_add_position_gradient_pairs(share, read, 0, 1, paired, message, sizeof message) != 0);
  EXPECT(strlen(message) > 0 && memcmp(paired, once, sizeof paired) == 0);

done:
  openmx_exx_destroy(share);
  for (a = 0; a < ATOMS; ++a) {
    free(tables[a]);
    free(weights[a]);
  }
  free(site_other);
  return failed;
#undef EXPECT
}

/* The V neighbours folded by the period of a k mesh (section 6.4 of the
   guide).  state is a prepared structure with the list of its blocks; k and
   gradient are the matrix and the derivative of the Gamma form for all atoms
   with the factor -0.25.  With the period 1 the site form of the Gamma
   point, D_xy(R) = D_xy, gives them again, through the centre atoms and
   through the pairs of sites with the half weights.  The state is prepared
   again at the end, with the list of its blocks.  Returns 0 when every check
   holds and the line of the first one that fails otherwise. */
static int folded_work(OpenMXExxState *state, const OpenMXExxOptions *options, const OpenMXExxGeometry *geometry,
                       const double *d, const double *k, const double *gradient)
{
#define EXPECT(expression) do { if (!(expression)) { \
  fprintf(stderr,"line %d: %s\n",__LINE__,#expression); failed = __LINE__; goto done; } } while (0)
  const int gamma_period[3] = {1, 1, 1}, zero_period[3] = {1, 0, 1};
  const int first[ATOMS] = {0, 5, 6};   /* the first orbital of every atom */
  const double scale = largest(k, N * N), steepest = largest(gradient, 3 * ATOMS);
  double *folded_d = NULL, *folded_k = NULL, *weights[ATOMS] = {NULL, NULL, NULL};
  const double *read[ATOMS];
  double other[N * N], through[3 * ATOMS];
  char message[160];
  int failed = 0, a, i, j, row, column, prepared_count[ATOMS], prepared_sum = 0, folded_sum = 0;
  long blocks, block, total;
  size_t elements, offset, element;

  for (a = 0; a < ATOMS; ++a) {
    prepared_count[a] = openmx_exx_v_neighbours(state, a);
    prepared_sum += prepared_count[a];
    EXPECT(prepared_count[a] > 1 && openmx_exx_prepared_v_neighbours(state, a) == prepared_count[a]);
  }
  EXPECT(openmx_exx_prepared_v_neighbours(state, ATOMS) == -1 && openmx_exx_prepared_v_neighbours(NULL, 0) == -1);
  /* What the folding rejects leaves the state as it was, with the list of
     its blocks. */
  blocks = openmx_exx_block_count(state);
  EXPECT(blocks > 0);
  EXPECT(openmx_exx_fold_v_neighbours(NULL, gamma_period, message, sizeof message) != 0 && strlen(message) > 0);
  EXPECT(openmx_exx_fold_v_neighbours(state, NULL, message, sizeof message) != 0 && strlen(message) > 0);
  EXPECT(openmx_exx_fold_v_neighbours(state, zero_period, message, sizeof message) != 0 && strlen(message) > 0);
  EXPECT(openmx_exx_block_count(state) == blocks);
  for (a = 0; a < ATOMS; ++a) EXPECT(openmx_exx_v_neighbours(state, a) == prepared_count[a]);
  /* With the period 1 an atom keeps one site of every atom at most.  The
     folding discards the list of the blocks, and a folded structure is not
     folded again. */
  EXPECT(openmx_exx_fold_v_neighbours(state, gamma_period, message, sizeof message) == 0 && message[0] == '\0');
  EXPECT(openmx_exx_block_count(state) == -1);
  for (a = 0; a < ATOMS; ++a) {
    EXPECT(openmx_exx_prepared_v_neighbours(state, a) == prepared_count[a]);
    EXPECT(openmx_exx_v_neighbours(state, a) >= 1 && openmx_exx_v_neighbours(state, a) <= ATOMS);
    folded_sum += openmx_exx_v_neighbours(state, a);
  }
  EXPECT(folded_sum < prepared_sum);
  EXPECT(openmx_exx_fold_v_neighbours(state, gamma_period, message, sizeof message) != 0 && strlen(message) > 0);
  EXPECT(openmx_exx_prepare_blocks(state, message, sizeof message) == 0);
  blocks = openmx_exx_block_count(state);
  elements = openmx_exx_block_elements(state);
  EXPECT(blocks > 0 && elements > 0);
  folded_d = (double *)calloc(2 * elements, sizeof *folded_d);
  EXPECT(folded_d != NULL);
  folded_k = folded_d + elements;
  element = 0;
  for (block = 0; block < blocks; ++block) {
    EXPECT(openmx_exx_block(state, block, &row, &column, NULL, &offset) == 0 && offset == element);
    for (i = 0; i < openmx_exx_atom_orbitals(state, row); ++i)
      for (j = 0; j < openmx_exx_atom_orbitals(state, column); ++j)
        folded_d[element++] = d[(first[row] + i) * N + first[column] + j];
  }
  EXPECT(element == elements);
  EXPECT(openmx_exx_add_blocks(state, 0, ATOMS, -0.25, folded_d, folded_k, NULL, message, sizeof message) == 0);
  /* The sum over the cells is the matrix of the Gamma form. */
  memset(other, 0, sizeof other);
  for (block = 0; block < blocks; ++block) {
    EXPECT(openmx_exx_block(state, block, &row, &column, NULL, &offset) == 0);
    for (i = 0; i < openmx_exx_atom_orbitals(state, row); ++i)
      for (j = 0; j < openmx_exx_atom_orbitals(state, column); ++j)
        other[(first[row] + i) * N + first[column] + j] += folded_k[offset++];
  }
  for (i = 0; i < N * N; ++i) EXPECT(fabs(other[i] - k[i]) <= 1.0e-12 * scale);
  EXPECT(close_to(openmx_exx_energy_gamma(elements, folded_d, folded_k), openmx_exx_energy_gamma(N * N, d, k),
                  1.0e-12));
  /* The derivative through the centre atoms, and through the pairs of sites
     of the preparation with the half weights of the folded blocks. */
  memset(through, 0, sizeof through);
  EXPECT(openmx_exx_add_position_gradient_blocks(state, 0, ATOMS, -0.25, folded_d, through, NULL, message,
                                                 sizeof message) == 0);
  for (i = 0; i < 3 * ATOMS; ++i) EXPECT(fabs(through[i] - gradient[i]) <= 1.0e-11 * steepest);
  total = openmx_exx_abf_count(state);
  for (a = 0; a < ATOMS; ++a) {
    weights[a] = (double *)calloc(openmx_exx_atom_table_elements(state, a), sizeof **weights);
    EXPECT(weights[a] != NULL);
    read[a] = weights[a];
  }
  EXPECT(openmx_exx_add_block_weights_blocks_half_abfs(state, 0, total, -0.25, folded_d, weights, NULL, message,
                                                       sizeof message) == 0);
  memset(through, 0, sizeof through);
  EXPECT(openmx_exx_add_position_gradient_pairs(state, read, 0, 1, through, message, sizeof message) == 0);
  for (i = 0; i < 3 * ATOMS; ++i) EXPECT(fabs(through[i] - gradient[i]) <= 1.0e-11 * steepest);
  /* A new preparation discards the folding. */
  EXPECT(openmx_exx_prepare(state, options, geometry, message, sizeof message) == 0);
  for (a = 0; a < ATOMS; ++a)
    EXPECT(openmx_exx_v_neighbours(state, a) == prepared_count[a] &&
           openmx_exx_prepared_v_neighbours(state, a) == prepared_count[a]);
  EXPECT(openmx_exx_prepare_blocks(state, message, sizeof message) == 0);
done:
  for (a = 0; a < ATOMS; ++a) free(weights[a]);
  free(folded_d);
  return failed;
#undef EXPECT
}

int main(void)
{
  OpenMXExxState *state = NULL;
  OpenMXExxOptions options;
  OpenMXExxGeometry geometry;
  OpenMXExxSpecies species[2];
  OpenMXExxReport report;
  OpenMXExxDenseBackend table;
  Device device;
  OpenMXExxDevice device_table;
  static double mesh_a[POINTS], mesh_b[POINTS], values_a[3 * POINTS], values_b[POINTS];
  const int angular_a[3] = {0, 0, 1}, angular_b[1] = {0};
  const double cutoff_a = 4.2, cutoff_b = 3.6;
  const double positions[3 * ATOMS] = {0.2, 0.1, 0.3, 2.5, 0.4, 0.1, 0.6, 2.9, 3.1};
  const int atom_species[ATOMS] = {0, 1, 1};
  const int wrong_species[ATOMS] = {0, 1, 2};
  const double lattice[3][3] = {{6.3, 0.0, 0.0}, {0.4, 6.8, 0.0}, {0.2, -0.3, 7.1}};
  const int first[ATOMS] = {0, 5, 6};   /* the first orbital of every atom */
  const int far_cell[3] = {99, 0, 0};
  const int thread_counts[3] = {2, 3, 64};
  double d[N * N], k[N * N], other[N * N], pieced[N * N], truncated[N * N], zero[N * N];
  double *site_d = NULL, *site_k = NULL, *site_other = NULL, *site_pieced = NULL;
  double gradient[3 * ATOMS], site_gradient[3 * ATOMS], looped[3 * ATOMS], moved[3 * ATOMS], energy[4];
  const double step = 2.0e-3;
  int c_count[ATOMS], v_count[ATOMS];
  char message[160], tiny[4];
  double sum, scale, asymmetry, steepest, numeric;
  Calls calls = {0, 0, 0, 0};
  long before, products, site_products, blocks = 0, block;
  size_t elements = 0, offset, element;
  int i, j, a, row, column, cell[3], abfs, threads, pieces, groups, form;
  long device_calls, failing;

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

  /* The product of the caller.  The table has no loop: the ABFs of a centre
     atom are computed one after the other. */
  table.dgemm = counting_dgemm;
  table.context = &calls;
  table.parallel_for = NULL;
  table.threads = 0;
  table.device = NULL;
  memset(other, 0, sizeof other);
  CHECK(openmx_exx_add_gamma(state, 0, ATOMS, -0.25, d, other, &table, message, sizeof message) == 0);
  products = calls.products;
  CHECK(products > 0 && memcmp(k, other, sizeof k) == 0);
  table.dgemm = failing_dgemm;
  table.context = NULL;
  CHECK(openmx_exx_add_gamma(state, 0, ATOMS, -0.25, d, other, &table, message, sizeof message) != 0);
  CHECK(strlen(message) > 0);

  /* The loop of the caller: the ABFs of a centre atom in pieces, as many as
     the table has threads or, with fewer ABFs, one for every ABF.  A loop is
     asked for one call for every piece.  The first atom has more than three
     ABFs and fewer than 64: 2 and 3 threads give 2 and 3 pieces, 64 threads
     one piece for every ABF.  The other two atoms have one ABF; they are one
     piece, and the loop is not called for them.  A build hands the table
     four loops for the first atom and two for every group of its V
     neighbours.  The four gather the rows of the density matrix, turn two
     arrays and add the last result to K.  The Gamma form has one group for
     every atom among them, and these are all three: the atoms are closer to
     the first one than the cutoff of its V neighbours (4.2 + 2.5 bohr).

     The sums over the ABFs are added in another order than without pieces,
     so the numbers agree within rounding.  For one number of threads they
     are the same bit for bit in whatever order the pieces run, and also with
     the product of the caller, which is that of the library and is called as
     often as without pieces. */
  abfs = openmx_exx_atom_abfs(state, 0);
  CHECK(abfs > 3 && abfs < 64);
  for (i = 0; i < 3; ++i) {
    threads = thread_counts[i];
    pieces = threads < abfs ? threads : abfs;
    table.dgemm = NULL;
    table.context = &calls;
    table.parallel_for = ordered_loop;
    table.threads = threads;
    forget_loops(&calls);
    memset(pieced, 0, sizeof pieced);
    CHECK(openmx_exx_add_gamma(state, 0, ATOMS, -0.25, d, pieced, &table, message, sizeof message) == 0);
    CHECK(calls.loops == 4 + 2 * ATOMS && calls.smallest == pieces && calls.largest == pieces);
    for (j = 0; j < N * N; ++j) CHECK(fabs(pieced[j] - k[j]) <= 1.0e-12 * scale);
    table.parallel_for = reversed_loop;
    memset(other, 0, sizeof other);
    CHECK(openmx_exx_add_gamma(state, 0, ATOMS, -0.25, d, other, &table, message, sizeof message) == 0);
    CHECK(memcmp(pieced, other, sizeof pieced) == 0);
    table.dgemm = counting_dgemm;
    before = calls.products;
    forget_loops(&calls);
    memset(other, 0, sizeof other);
    CHECK(openmx_exx_add_gamma(state, 0, ATOMS, -0.25, d, other, &table, message, sizeof message) == 0);
    CHECK(calls.products - before == products && calls.loops == 4 + 2 * ATOMS);
    CHECK(calls.smallest == pieces && calls.largest == pieces && memcmp(pieced, other, sizeof pieced) == 0);
  }
  /* With one thread the loop is not called, and the numbers are those
     without a table. */
  table.dgemm = NULL;
  table.threads = 1;
  forget_loops(&calls);
  memset(other, 0, sizeof other);
  CHECK(openmx_exx_add_gamma(state, 0, ATOMS, -0.25, d, other, &table, message, sizeof message) == 0);
  CHECK(calls.loops == 0 && memcmp(k, other, sizeof k) == 0);
  /* A loop that reports a failure ends the build with a message. */
  table.threads = 2;
  table.parallel_for = failing_loop;
  CHECK(openmx_exx_add_gamma(state, 0, ATOMS, -0.25, d, other, &table, message, sizeof message) != 0);
  CHECK(strlen(message) > 0);
  table.parallel_for = NULL;
  table.threads = 0;

  /* Equations (4) and (15).  The exchange energy of occupied orbitals is
     negative. */
  sum = 0.0;
  for (i = 0; i < N * N; ++i) sum += d[i] * k[i];
  CHECK(sum < 0.0);
  CHECK(close_to(openmx_exx_energy_gamma(N * N, d, k), 0.5 * sum, 1.0e-14));
  CHECK(close_to(openmx_exx_change_gamma(N * N, d, k, zero), sum, 1.0e-14));
  CHECK(openmx_exx_change_gamma(N * N, d, k, k) == 0.0);
  CHECK(openmx_exx_energy_gamma(N * N, NULL, k) == 0.0 && openmx_exx_change_gamma(N * N, d, k, NULL) == 0.0);

  /* The site form.  The list of the blocks is made on request. */
  CHECK(openmx_exx_block_count(NULL) == -1 && openmx_exx_block_elements(NULL) == 0);
  CHECK(openmx_exx_block_count(state) == -1 && openmx_exx_block_elements(state) == 0);
  CHECK(openmx_exx_block(state, 0, &row, &column, cell, &offset) != 0);
  CHECK(openmx_exx_find_block(state, 0, 0, far_cell) == -1 && openmx_exx_find_block(NULL, 0, 0, far_cell) == -1);
  CHECK(openmx_exx_prepare_blocks(NULL, message, sizeof message) != 0 && strlen(message) > 0);
  CHECK(openmx_exx_prepare_blocks(state, message, sizeof message) == 0 && message[0] == '\0');
  blocks = openmx_exx_block_count(state);
  elements = openmx_exx_block_elements(state);
  CHECK(blocks >= ATOMS * ATOMS && elements >= (size_t)(N * N));
  CHECK(openmx_exx_block(state, blocks, &row, &column, cell, &offset) != 0);
  CHECK(openmx_exx_block(state, -1, &row, &column, cell, &offset) != 0);
  CHECK(openmx_exx_find_block(state, 0, 0, far_cell) == -1 && openmx_exx_find_block(state, 0, 0, NULL) == -1);
  CHECK(openmx_exx_find_block(state, ATOMS, 0, far_cell) == -1 && openmx_exx_find_block(state, 0, -1, far_cell) == -1);
  site_d = (double *)calloc(4 * elements, sizeof *site_d);
  CHECK(site_d != NULL);
  site_k = site_d + elements;
  site_other = site_k + elements;
  site_pieced = site_other + elements;
  /* D_xy(R) = D_xy for every cell R: the density matrix of the Gamma point. */
  element = 0;
  for (block = 0; block < blocks; ++block) {
    CHECK(openmx_exx_block(state, block, &row, &column, cell, &offset) == 0);
    CHECK(row >= 0 && row < ATOMS && column >= 0 && column < ATOMS && offset == element);
    CHECK(openmx_exx_find_block(state, row, column, cell) == block);
    for (i = 0; i < openmx_exx_atom_orbitals(state, row); ++i)
      for (j = 0; j < openmx_exx_atom_orbitals(state, column); ++j)
        site_d[element++] = d[(first[row] + i) * N + first[column] + j];
  }
  CHECK(element == elements);
  CHECK(openmx_exx_add_blocks(state, 0, ATOMS, -0.25, site_d, site_k, NULL, message, sizeof message) == 0);
  CHECK(message[0] == '\0');
  /* The sum over the cells is the matrix of the Gamma form. */
  memset(other, 0, sizeof other);
  for (block = 0; block < blocks; ++block) {
    CHECK(openmx_exx_block(state, block, &row, &column, NULL, &offset) == 0);
    for (i = 0; i < openmx_exx_atom_orbitals(state, row); ++i)
      for (j = 0; j < openmx_exx_atom_orbitals(state, column); ++j)
        other[(first[row] + i) * N + first[column] + j] += site_k[offset++];
  }
  for (i = 0; i < N * N; ++i) CHECK(fabs(other[i] - k[i]) <= 1.0e-12 * scale);
  CHECK(close_to(openmx_exx_energy_gamma(elements, site_d, site_k), openmx_exx_energy_gamma(N * N, d, k), 1.0e-12));
  /* The atoms one by one, with the product of the caller, give the same
     numbers. */
  before = calls.products;
  table.context = &calls;
  table.dgemm = counting_dgemm;
  for (a = 0; a < ATOMS; ++a)
    CHECK(openmx_exx_add_blocks(state, a, a + 1, -0.25, site_d, site_other, &table, NULL, 0) == 0);
  site_products = calls.products - before;
  CHECK(site_products > 0 && memcmp(site_k, site_other, elements * sizeof *site_k) == 0);
  /* The site form with the loop of the caller, three pieces.  Every V
     neighbour of the first atom is a group of its own.  As in the Gamma
     form, the numbers agree with those without pieces within rounding, and
     they are the same bit for bit in whatever order the pieces run and with
     the product of the caller. */
  groups = openmx_exx_v_neighbours(state, 0);
  table.dgemm = NULL;
  table.parallel_for = ordered_loop;
  table.threads = 3;
  forget_loops(&calls);
  CHECK(openmx_exx_add_blocks(state, 0, ATOMS, -0.25, site_d, site_pieced, &table, message, sizeof message) == 0);
  CHECK(calls.loops == 4 + 2 * groups && calls.smallest == 3 && calls.largest == 3);
  for (element = 0; element < elements; ++element)
    CHECK(fabs(site_pieced[element] - site_k[element]) <= 1.0e-12 * scale);
  table.dgemm = counting_dgemm;
  table.parallel_for = reversed_loop;
  before = calls.products;
  forget_loops(&calls);
  memset(site_other, 0, elements * sizeof *site_other);
  CHECK(openmx_exx_add_blocks(state, 0, ATOMS, -0.25, site_d, site_other, &table, message, sizeof message) == 0);
  CHECK(calls.products - before == site_products && calls.loops == 4 + 2 * groups);
  CHECK(calls.smallest == 3 && calls.largest == 3);
  CHECK(memcmp(site_pieced, site_other, elements * sizeof *site_pieced) == 0);
  table.parallel_for = NULL;
  table.threads = 0;
  memcpy(site_other, site_k, elements * sizeof *site_k);
  /* K_xy(R) is the transpose of K_yx(-R); the means keep it. */
  asymmetry = -1.0;
  CHECK(openmx_exx_symmetrize_blocks(state, site_other, &asymmetry, message, sizeof message) == 0);
  CHECK(asymmetry >= 0.0 && asymmetry <= 1.0e-12 * scale);
  for (element = 0; element < elements; ++element)
    CHECK(fabs(site_other[element] - site_k[element]) <= 1.0e-12 * scale);
  CHECK(openmx_exx_symmetrize_blocks(state, site_other, &asymmetry, NULL, 0) == 0 && asymmetry == 0.0);
  /* Rejected calls add nothing. */
  memcpy(site_other, site_k, elements * sizeof *site_k);
  CHECK(openmx_exx_add_blocks(state, -1, ATOMS, -0.25, site_d, site_other, NULL, message, sizeof message) != 0);
  CHECK(openmx_exx_add_blocks(state, 0, ATOMS + 1, -0.25, site_d, site_other, NULL, message, sizeof message) != 0);
  CHECK(openmx_exx_add_blocks(state, 2, 1, -0.25, site_d, site_other, NULL, message, sizeof message) != 0);
  CHECK(openmx_exx_add_blocks(state, 0, ATOMS, -0.25, NULL, site_other, NULL, message, sizeof message) != 0);
  CHECK(openmx_exx_add_blocks(state, 0, ATOMS, -0.25, site_d, NULL, NULL, message, sizeof message) != 0);
  CHECK(openmx_exx_add_blocks(NULL, 0, ATOMS, -0.25, site_d, site_other, NULL, message, sizeof message) != 0);
  CHECK(openmx_exx_symmetrize_blocks(state, NULL, &asymmetry, message, sizeof message) != 0 && strlen(message) > 0);
  CHECK(openmx_exx_symmetrize_blocks(NULL, site_other, &asymmetry, message, sizeof message) != 0);
  CHECK(memcmp(site_k, site_other, elements * sizeof *site_k) == 0);
  /* A product that fails ends the build with a message. */
  table.dgemm = failing_dgemm;
  CHECK(openmx_exx_add_blocks(state, 0, ATOMS, -0.25, site_d, site_other, &table, message, sizeof message) != 0);
  CHECK(strlen(message) > 0);
  table.dgemm = counting_dgemm;
  memcpy(site_other, site_k, elements * sizeof *site_k);

  /* The device of the caller computes both forms within rounding, and the
     product of the table is not used.  When it declines, or one of its calls
     fails, the library computes the call with the table on the host: the
     product of the caller is that of the library, so the bits are those
     without a table.  In every case the device gets back all its memory. */
  device_table.context = &device;
  device_table.accept = device_accept;
  device_table.allocate = device_allocate;
  device_table.release = device_release;
  device_table.upload = device_copy;
  device_table.download = device_copy;
  device_table.dgemm = device_dgemm;
  device_table.move = device_move;
  device_table.finish = device_finish;
  table.device = &device_table;
  for (form = 0; form < 2; ++form) {
    double *output = form == 0 ? other : site_other;
    const double *reference = form == 0 ? k : site_k;
    const size_t count = form == 0 ? (size_t)(N * N) : elements;
    forget_device(&device, 1, -1);
    before = calls.products;
    memset(output, 0, count * sizeof *output);
    if (form == 0) CHECK(openmx_exx_add_gamma(state, 0, ATOMS, -0.25, d, output, &table, message, sizeof message) == 0);
    else CHECK(openmx_exx_add_blocks(state, 0, ATOMS, -0.25, site_d, output, &table, message, sizeof message) == 0);
    CHECK(message[0] == '\0' && calls.products == before);
    CHECK(device.accepted == 1 && device.allocations > 0 && device.releases == device.allocations);
    for (element = 0; element < count; ++element) CHECK(fabs(output[element] - reference[element]) <= 1.0e-12 * scale);
    device_calls = device.calls;
    for (failing = -1; failing < device_calls; ++failing) {
      forget_device(&device, failing < 0 ? 0 : 1, failing);
      memset(output, 0, count * sizeof *output);
      if (form == 0) CHECK(openmx_exx_add_gamma(state, 0, ATOMS, -0.25, d, output, &table, message, sizeof message) == 0);
      else CHECK(openmx_exx_add_blocks(state, 0, ATOMS, -0.25, site_d, output, &table, message, sizeof message) == 0);
      CHECK(message[0] == '\0' && device.accepted == 1 && device.releases == device.allocations);
      CHECK(failing < 0 ? device.calls == 0 : device.calls > failing);
      CHECK(memcmp(output, reference, count * sizeof *output) == 0);
    }
  }
  table.device = NULL;
  memcpy(site_other, site_k, elements * sizeof *site_k);

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
  /* So does the list of its blocks. */
  CHECK(openmx_exx_block_count(state) == blocks && openmx_exx_block_elements(state) == elements);
  memset(site_other, 0, elements * sizeof *site_other);
  CHECK(openmx_exx_add_blocks(state, 0, ATOMS, -0.25, site_d, site_other, NULL, message, sizeof message) == 0);
  CHECK(memcmp(site_k, site_other, elements * sizeof *site_k) == 0);

  /* The derivative of E_x with respect to the positions of the atoms.  Its
     sum over the atoms vanishes, the two forms agree for the density matrix
     of the Gamma point, and the atoms one by one give what one call gives. */
  memset(gradient, 0, sizeof gradient);
  CHECK(openmx_exx_add_position_gradient_gamma(state, 0, ATOMS, -0.25, d, gradient, NULL, message, sizeof message) == 0);
  CHECK(message[0] == '\0');
  steepest = largest(gradient, 3 * ATOMS);
  CHECK(steepest > 1.0e-6);
  for (j = 0; j < 3; ++j) {
    sum = 0.0;
    for (a = 0; a < ATOMS; ++a) sum += gradient[3 * a + j];
    CHECK(fabs(sum) <= 1.0e-12 * steepest);
  }
  memset(site_gradient, 0, sizeof site_gradient);
  before = calls.products;
  for (a = 0; a < ATOMS; ++a)
    CHECK(openmx_exx_add_position_gradient_blocks(state, a, a + 1, -0.25, site_d, site_gradient, &table, NULL, 0) == 0);
  CHECK(calls.products > before);
  for (i = 0; i < 3 * ATOMS; ++i) CHECK(fabs(site_gradient[i] - gradient[i]) <= 1.0e-11 * steepest);
  /* The weights of the blocks are computed in pieces of the ABFs: with a
     loop in the table the loop is called, and the numbers are the same bit
     for bit, whatever the order of the pieces.  The product of the caller
     is that of the library, so the Gamma form gives the bits of the call
     without a table. */
  table.parallel_for = reversed_loop;
  table.threads = 3;
  forget_loops(&calls);
  memset(looped, 0, sizeof looped);
  for (a = 0; a < ATOMS; ++a)
    CHECK(openmx_exx_add_position_gradient_blocks(state, a, a + 1, -0.25, site_d, looped, &table, NULL, 0) == 0);
  CHECK(calls.loops > 0 && memcmp(looped, site_gradient, sizeof looped) == 0);
  forget_loops(&calls);
  memset(looped, 0, sizeof looped);
  CHECK(openmx_exx_add_position_gradient_gamma(state, 0, ATOMS, -0.25, d, looped, &table, NULL, 0) == 0);
  CHECK(calls.loops > 0 && memcmp(looped, gradient, sizeof looped) == 0);
  table.parallel_for = NULL;
  table.threads = 0;
  /* Rejected calls and a product that fails add nothing. */
  memcpy(site_gradient, gradient, sizeof gradient);
  CHECK(openmx_exx_add_position_gradient_gamma(NULL, 0, ATOMS, -0.25, d, site_gradient, NULL, message, sizeof message) != 0);
  CHECK(openmx_exx_add_position_gradient_gamma(state, 0, ATOMS + 1, -0.25, d, site_gradient, NULL, message, sizeof message) != 0);
  CHECK(openmx_exx_add_position_gradient_gamma(state, 0, ATOMS, -0.25, NULL, site_gradient, NULL, message, sizeof message) != 0);
  CHECK(openmx_exx_add_position_gradient_gamma(state, 0, ATOMS, -0.25, d, NULL, NULL, message, sizeof message) != 0);
  CHECK(openmx_exx_add_position_gradient_blocks(NULL, 0, ATOMS, -0.25, site_d, site_gradient, NULL, message, sizeof message) != 0);
  CHECK(openmx_exx_add_position_gradient_blocks(state, 2, 1, -0.25, site_d, site_gradient, NULL, message, sizeof message) != 0);
  CHECK(openmx_exx_add_position_gradient_blocks(state, 0, ATOMS, -0.25, NULL, site_gradient, NULL, message, sizeof message) != 0);
  CHECK(openmx_exx_add_position_gradient_blocks(state, 0, ATOMS, -0.25, site_d, NULL, NULL, message, sizeof message) != 0);
  table.dgemm = failing_dgemm;
  CHECK(openmx_exx_add_position_gradient_gamma(state, 0, ATOMS, -0.25, d, site_gradient, &table, message, sizeof message) != 0);
  CHECK(strlen(message) > 0);
  CHECK(openmx_exx_add_position_gradient_blocks(state, 0, ATOMS, -0.25, site_d, site_gradient, &table, message, sizeof message) != 0);
  CHECK(strlen(message) > 0);
  table.dgemm = counting_dgemm;
  CHECK(memcmp(site_gradient, gradient, sizeof gradient) == 0);

  /* A central difference of fourth order of E_x for the second atom moved
     along the y axis, with the density matrix as it is.  No pair of sites of
     this structure is closer than 0.017 bohr to a cutoff of the neighbour
     lists, so the moves leave the lists as they are. */
  for (a = 0; a < ATOMS; ++a) {
    c_count[a] = openmx_exx_c_neighbours(state, a);
    v_count[a] = openmx_exx_v_neighbours(state, a);
  }
  geometry.positions = moved;
  for (i = 0; i < 4; ++i) {
    memcpy(moved, positions, sizeof moved);
    moved[3 * 1 + 1] += (i < 2 ? i - 2 : i - 1) * step;
    CHECK(openmx_exx_prepare(state, &options, &geometry, message, sizeof message) == 0);
    for (a = 0; a < ATOMS; ++a)
      CHECK(openmx_exx_c_neighbours(state, a) == c_count[a] && openmx_exx_v_neighbours(state, a) == v_count[a]);
    memset(other, 0, sizeof other);
    CHECK(openmx_exx_add_gamma(state, 0, ATOMS, -0.25, d, other, NULL, message, sizeof message) == 0);
    energy[i] = openmx_exx_energy_gamma(N * N, d, other);
  }
  geometry.positions = positions;
  numeric = (8.0 * (energy[2] - energy[1]) - (energy[3] - energy[0])) / (12.0 * step);
  CHECK(fabs(numeric - gradient[3 * 1 + 1]) <= 1.0e-6 * steepest);
  /* The structure itself again, with the list of its blocks. */
  CHECK(openmx_exx_prepare(state, &options, &geometry, message, sizeof message) == 0);
  CHECK(openmx_exx_prepare_blocks(state, message, sizeof message) == 0);
  CHECK(openmx_exx_block_count(state) == blocks && openmx_exx_block_elements(state) == elements);

  /* The entries for processes that divide the work. */
  CHECK(divided_work(state, &options, &geometry, d, k, site_d, site_k, elements, gradient) == 0);
  /* The V neighbours folded by the period of a k mesh. */
  CHECK(folded_work(state, &options, &geometry, d, k, gradient) == 0);
  CHECK(openmx_exx_block_count(state) == blocks && openmx_exx_block_elements(state) == elements);

  /* The kernel of PBE0 gives another matrix for the same structure. */
  options.kernel = OPENMX_EXX_KERNEL_TRUNCATED_COULOMB;
  options.kernel_parameter = 10.0;
  CHECK(openmx_exx_prepare(state, &options, &geometry, message, sizeof message) == 0);
  /* A preparation discards the list of the blocks. */
  CHECK(openmx_exx_block_count(state) == -1 && openmx_exx_block_elements(state) == 0);
  CHECK(openmx_exx_add_blocks(state, 0, ATOMS, -0.25, site_d, site_other, NULL, message, sizeof message) != 0);
  CHECK(strlen(message) > 0 && memcmp(site_k, site_other, elements * sizeof *site_k) == 0);
  memset(truncated, 0, sizeof truncated);
  CHECK(openmx_exx_add_gamma(state, 0, ATOMS, -0.25, d, truncated, NULL, message, sizeof message) == 0);
  for (i = 0; i < N * N; ++i) other[i] = truncated[i] - k[i];
  CHECK(largest(other, N * N) > 1.0e-3 * scale);
  for (i = 0; i < N; ++i)
    for (j = 0; j < i; ++j) CHECK(fabs(truncated[i * N + j] - truncated[j * N + i]) <= 1.0e-12 * largest(truncated, N * N));

  openmx_exx_destroy(state);
  free(site_d);
  printf("C interface of the exact exchange: %d orbitals, largest element of K %.3e (HSE06 kernel), %ld blocks, "
         "%ld products, largest derivative of E_x %.3e (central difference: deviation %.1e)\n", N, scale, blocks,
         calls.products, steepest, fabs(numeric - gradient[3 * 1 + 1]) / steepest);
  return 0;
}
