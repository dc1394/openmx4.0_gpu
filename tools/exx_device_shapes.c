/* exx_device_shapes.c

   Times, on one CUDA device, the matrix products that the build of the
   exchange matrix would make if all its arrays stayed on the device, with
   the sizes of two benchmark systems (siz-p444 and diaz-p222 of
   tools/exx_gpu_checks.py).  Double precision, cuBLAS only.

   A measurement aid for the design of the device path of the build
   (docs/EXX_IMPLEMENTATION_GUIDE.md, section 11).  The build system does
   not compile it and no test runs it.  It needs the CUDA runtime and
   cuBLAS, for example
     cc -O2 tools/exx_device_shapes.c -I$CUDA/include -L$CUDA/lib64 -lcublas -lcudart -lm

   usage: exx_device_shapes si|diamond|small

   Sizes of a centre atom:
     np  ABFs of the centre atom           nq  ABFs of a V neighbour
     u   orbitals of an atom               l   orbitals of the C neighbours
     s   orbitals of the sites S           nv  V neighbours of the centre atom
   Every product is C(m x n) = A(m x k) B(k x n), column-major, without
   transposition, on arrays that are on the device before the clock starts.
   A line gives the time of one call and the multiplications m*n*k (times
   the batch) per second.  "queued" calls are issued one after another and
   the device is waited for once, as a sequence of V neighbours would be.
   A few elements of every product are compared with sums formed on the
   host; the last line counts the comparisons that failed.  "small" is a
   quick run of all lines with tiny sizes. */

#define _POSIX_C_SOURCE 200809L /* clock_gettime */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
  const char *name;
  int np, nq, u, l, s, nv, atoms;
} System;

static const System systems[] = {
  {"si", 69, 69, 13, 1131, 8229, 99, 2},
  {"diamond", 69, 69, 13, 2171, 20163, 293, 2},
  {"small", 5, 5, 3, 17, 61, 4, 1},
};

typedef struct {
  cublasHandle_t blas;
  int m, n, k, batch;
  const double *a, *b;
  double *c;
  long long stride_a, stride_b, stride_c;
  const double *const *pointer_a, *const *pointer_b;
  double *const *pointer_c;
  size_t bytes;
  void *target;
  const void *source;
  enum cudaMemcpyKind kind;
} Case;

typedef void (*Step)(const Case *);

static int failures;

static double now(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void cuda_ok(cudaError_t status, const char *what)
{
  if (status == cudaSuccess) return;
  fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(status));
  exit(2);
}

static void blas_ok(cublasStatus_t status, const char *what)
{
  if (status == CUBLAS_STATUS_SUCCESS) return;
  fprintf(stderr, "cuBLAS error %d in %s\n", (int)status, what);
  exit(2);
}

static void wait_device(void) { cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); }

/* n numbers between -0.5 and 0.5 that depend on the seed. */
static double *host_array(size_t n, unsigned seed)
{
  double *x = (double *)malloc((n ? n : 1) * sizeof(double));
  unsigned long long state = 0x9E3779B97F4A7C15ULL * (seed + 1ULL);
  if (x == NULL) {
    fprintf(stderr, "no host memory for %zu numbers\n", n);
    exit(2);
  }
  for (size_t i = 0; i < n; i++) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    x[i] = (double)(state >> 11) / 9007199254740992.0 - 0.5;
  }
  return x;
}

static double *device_array(size_t n)
{
  void *d = NULL;
  cuda_ok(cudaMalloc(&d, (n ? n : 1) * sizeof(double)), "cudaMalloc");
  return (double *)d;
}

static double *device_copy(const double *host, size_t n)
{
  double *d = device_array(n);
  cuda_ok(cudaMemcpy(d, host, n * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy to the device");
  return d;
}

static void step_gemm(const Case *c)
{
  const double one = 1.0, zero = 0.0;
  blas_ok(cublasDgemm(c->blas, CUBLAS_OP_N, CUBLAS_OP_N, c->m, c->n, c->k, &one, c->a, c->m, c->b, c->k, &zero, c->c,
                      c->m),
          "cublasDgemm");
}

static void step_strided(const Case *c)
{
  const double one = 1.0, zero = 0.0;
  blas_ok(cublasDgemmStridedBatched(c->blas, CUBLAS_OP_N, CUBLAS_OP_N, c->m, c->n, c->k, &one, c->a, c->m, c->stride_a,
                                    c->b, c->k, c->stride_b, &zero, c->c, c->m, c->stride_c, c->batch),
          "cublasDgemmStridedBatched");
}

static void step_pointers(const Case *c)
{
  const double one = 1.0, zero = 0.0;
  blas_ok(cublasDgemmBatched(c->blas, CUBLAS_OP_N, CUBLAS_OP_N, c->m, c->n, c->k, &one, c->pointer_a, c->m,
                             c->pointer_b, c->k, &zero, c->pointer_c, c->m, c->batch),
          "cublasDgemmBatched");
}

static void step_copy(const Case *c) { cuda_ok(cudaMemcpy(c->target, c->source, c->bytes, c->kind), "cudaMemcpy"); }

static void step_axpy(const Case *c)
{
  const double one = 1.0;
  blas_ok(cublasDaxpy(c->blas, (int)(c->bytes / sizeof(double)), &one, c->a, 1, c->c, 1), "cublasDaxpy");
}

static void step_allocate(const Case *c)
{
  void *d = NULL;
  cuda_ok(cudaMalloc(&d, c->bytes), "cudaMalloc");
  cuda_ok(cudaFree(d), "cudaFree");
}

/* Issues `queued` steps, waits for the device once, and repeats this until
   0.5 s and three rounds have passed.  Returns the seconds of one round. */
static double seconds_of(Step step, const Case *c, int queued)
{
  int rounds = 0;
  double start, elapsed;
  for (int i = 0; i < 2; i++) step(c);
  wait_device();
  start = now();
  do {
    for (int i = 0; i < queued; i++) step(c);
    wait_device();
    rounds++;
    elapsed = now() - start;
  } while (elapsed < 0.5 || rounds < 3);
  return elapsed / rounds;
}

static double product_line(const char *label, Step step, const Case *c, int queued)
{
  const double seconds = seconds_of(step, c, queued);
  const double work = (double)c->m * (double)c->n * (double)c->k * (double)c->batch;
  printf("%-44s %6d %6d %6d %5d %6d %11.4f %9.2f\n", label, c->m, c->n, c->k, c->batch, queued, 1e3 * seconds / queued,
         1e-9 * work * queued / seconds);
  fflush(stdout);
  return seconds;
}

/* Compares three elements of member `member` of a product on the device
   with sums formed on the host.  The bound is above the rounding error of
   a sum of k terms in any order. */
static void check_member(const char *label, const Case *c, const double *host_a, const double *host_b, int member)
{
  const int row[3] = {0, c->m - 1, c->m / 2}, column[3] = {0, c->n - 1, c->n / 3};
  const double *a = host_a + (size_t)member * (size_t)c->stride_a, *b = host_b + (size_t)member * (size_t)c->stride_b;
  for (int e = 0; e < 3; e++) {
    double sum = 0.0, scale = 0.0, device = 0.0;
    const int i = row[e], j = column[e];
    for (int t = 0; t < c->k; t++) {
      const double term = a[(size_t)i + (size_t)t * (size_t)c->m] * b[(size_t)t + (size_t)j * (size_t)c->k];
      sum += term;
      scale += fabs(term);
    }
    cuda_ok(cudaMemcpy(&device, c->c + (size_t)member * (size_t)c->stride_c + (size_t)i + (size_t)j * (size_t)c->m,
                       sizeof(double), cudaMemcpyDeviceToHost),
            "cudaMemcpy of an element");
    if (!(fabs(device - sum) <= (1e-13 + 4e-16 * (double)c->k) * (scale + 1.0))) {
      printf("CHECK FAILED: %s, member %d, element (%d,%d): device %.17g, host %.17g\n", label, member, i, j, device,
             sum);
      failures++;
    }
  }
}

/* One product of the given dimensions: arrays, time, check. */
static double product(cublasHandle_t blas, const char *label, int m, int n, int k, int queued, unsigned seed)
{
  Case c;
  double *host_a = host_array((size_t)m * (size_t)k, seed), *host_b = host_array((size_t)k * (size_t)n, seed + 1);
  double *a = device_copy(host_a, (size_t)m * (size_t)k), *b = device_copy(host_b, (size_t)k * (size_t)n);
  double seconds;
  memset(&c, 0, sizeof(c));
  c.blas = blas;
  c.m = m;
  c.n = n;
  c.k = k;
  c.batch = 1;
  c.a = a;
  c.b = b;
  c.c = device_array((size_t)m * (size_t)n);
  seconds = product_line(label, step_gemm, &c, queued);
  check_member(label, &c, host_a, host_b, 0);
  cuda_ok(cudaFree(c.c), "cudaFree");
  cuda_ok(cudaFree(a), "cudaFree");
  cuda_ok(cudaFree(b), "cudaFree");
  free(host_a);
  free(host_b);
  return seconds;
}

/* `batch` products of the given dimensions in one call: with strides, and
   with arrays of pointers.  Returns the seconds of the form with strides;
   *with_pointers gets those of the other form. */
static double batched(cublasHandle_t blas, const char *label, int m, int n, int k, int batch, int queued, unsigned seed,
                      double *with_pointers)
{
  Case c;
  char text[96];
  const size_t size_a = (size_t)m * (size_t)k, size_b = (size_t)k * (size_t)n, size_c = (size_t)m * (size_t)n;
  double *host_a = host_array(size_a * (size_t)batch, seed), *host_b = host_array(size_b * (size_t)batch, seed + 1);
  double *a = device_copy(host_a, size_a * (size_t)batch), *b = device_copy(host_b, size_b * (size_t)batch);
  double *result = device_array(size_c * (size_t)batch);
  const double **host_pointers = (const double **)malloc(3 * (size_t)batch * sizeof(double *));
  void *pointers = NULL;
  double seconds;
  if (host_pointers == NULL) {
    fprintf(stderr, "no host memory for the pointers\n");
    exit(2);
  }
  for (int p = 0; p < batch; p++) {
    host_pointers[p] = a + (size_t)p * size_a;
    host_pointers[batch + p] = b + (size_t)p * size_b;
    host_pointers[2 * batch + p] = result + (size_t)p * size_c;
  }
  cuda_ok(cudaMalloc(&pointers, 3 * (size_t)batch * sizeof(double *)), "cudaMalloc");
  cuda_ok(cudaMemcpy(pointers, host_pointers, 3 * (size_t)batch * sizeof(double *), cudaMemcpyHostToDevice),
          "cudaMemcpy of the pointers");
  memset(&c, 0, sizeof(c));
  c.blas = blas;
  c.m = m;
  c.n = n;
  c.k = k;
  c.batch = batch;
  c.a = a;
  c.b = b;
  c.c = result;
  c.stride_a = (long long)size_a;
  c.stride_b = (long long)size_b;
  c.stride_c = (long long)size_c;
  c.pointer_a = (const double *const *)pointers;
  c.pointer_b = (const double *const *)pointers + batch;
  c.pointer_c = (double *const *)pointers + 2 * batch;
  snprintf(text, sizeof(text), "%s, strides", label);
  seconds = product_line(text, step_strided, &c, queued);
  check_member(text, &c, host_a, host_b, 0);
  check_member(text, &c, host_a, host_b, batch - 1);
  cuda_ok(cudaMemset(result, 0, size_c * (size_t)batch * sizeof(double)), "cudaMemset");
  snprintf(text, sizeof(text), "%s, pointers", label);
  *with_pointers = product_line(text, step_pointers, &c, queued);
  check_member(text, &c, host_a, host_b, 0);
  check_member(text, &c, host_a, host_b, batch - 1);
  cuda_ok(cudaFree(pointers), "cudaFree");
  cuda_ok(cudaFree(result), "cudaFree");
  cuda_ok(cudaFree(a), "cudaFree");
  cuda_ok(cudaFree(b), "cudaFree");
  free(host_pointers);
  free(host_a);
  free(host_b);
  return seconds;
}

/* Copies of `bytes` bytes between the host and the device, and on the
   device; an addition of that many numbers on the device; an allocation. */
static void memory_lines(cublasHandle_t blas, const char *label, size_t bytes, int queued, double *copy_seconds,
                         double *add_seconds)
{
  Case c;
  const size_t n = bytes / sizeof(double);
  double *pageable = host_array(n, 77), *first = device_copy(pageable, n), *second = device_copy(pageable, n);
  void *pinned = NULL;
  double seconds;
  cuda_ok(cudaMallocHost(&pinned, n * sizeof(double)), "cudaMallocHost");
  memcpy(pinned, pageable, n * sizeof(double));
  memset(&c, 0, sizeof(c));
  c.blas = blas;
  c.bytes = n * sizeof(double);
  const struct {
    const char *text;
    void *target;
    const void *source;
    enum cudaMemcpyKind kind;
  } copies[5] = {{"host to device, pageable", first, pageable, cudaMemcpyHostToDevice},
                 {"host to device, pinned", first, pinned, cudaMemcpyHostToDevice},
                 {"device to host, pageable", pageable, first, cudaMemcpyDeviceToHost},
                 {"device to host, pinned", pinned, first, cudaMemcpyDeviceToHost},
                 {"device to device", second, first, cudaMemcpyDeviceToDevice}};
  for (int i = 0; i < 5; i++) {
    const int calls = i == 4 ? queued : 1;
    c.target = copies[i].target;
    c.source = copies[i].source;
    c.kind = copies[i].kind;
    seconds = seconds_of(step_copy, &c, calls);
    printf("%-18s %8.1f MiB  %-26s %6d %11.4f ms %9.2f GiB/s\n", label, (double)c.bytes / 1048576.0, copies[i].text,
           calls, 1e3 * seconds / calls, (double)c.bytes * calls / seconds / 1073741824.0);
    if (i == 4) *copy_seconds = seconds / calls;
  }
  c.a = first;
  c.c = second;
  if (n <= 2147483647u) {
    seconds = seconds_of(step_axpy, &c, queued);
    printf("%-18s %8.1f MiB  %-26s %6d %11.4f ms %9.2f GiB/s\n", label, (double)c.bytes / 1048576.0,
           "y += x on the device", queued, 1e3 * seconds / queued, (double)c.bytes * queued / seconds / 1073741824.0);
    *add_seconds = seconds / queued;
  }
  seconds = seconds_of(step_allocate, &c, 1);
  printf("%-18s %8.1f MiB  %-26s %6d %11.4f ms\n", label, (double)c.bytes / 1048576.0, "cudaMalloc and cudaFree", 1,
         1e3 * seconds);
  fflush(stdout);
  cuda_ok(cudaFreeHost(pinned), "cudaFreeHost");
  cuda_ok(cudaFree(first), "cudaFree");
  cuda_ok(cudaFree(second), "cudaFree");
  free(pageable);
}

int main(int argc, char **argv)
{
  const System *sy = NULL;
  cublasHandle_t blas;
  struct cudaDeviceProp device;
  size_t free_bytes = 0, total_bytes = 0;
  double with_pointers = 0.0, ignore = 0.0;
  for (size_t i = 0; argc == 2 && i < sizeof(systems) / sizeof(systems[0]); i++)
    if (strcmp(argv[1], systems[i].name) == 0) sy = &systems[i];
  if (sy == NULL) {
    fprintf(stderr, "usage: %s si|diamond|small\n", argv[0]);
    return 2;
  }
  cuda_ok(cudaGetDeviceProperties(&device, 0), "cudaGetDeviceProperties");
  cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  blas_ok(cublasCreate(&blas), "cublasCreate");
  const int np = sy->np, nq = sy->nq, u = sy->u, l = sy->l, s = sy->s, nv = sy->nv, rows = np * u;
  printf("device 0: %s, %.0f MiB free of %.0f MiB\n", device.name, (double)free_bytes / 1048576.0,
         (double)total_bytes / 1048576.0);
  printf("system %s: np %d, nq %d, u %d, l %d, s %d, V neighbours %d, centre atoms %d\n\n", sy->name, np, nq, u, l, s, nv,
         sy->atoms);
  printf("%-44s %6s %6s %6s %5s %6s %11s %9s\n", "product", "m", "n", "k", "batch", "queued", "ms per call",
         "1e9 mult/s");

  /* the two large products of a centre atom */
  const double g = product(blas, "G = T D (large)", s, rows, l, 1, 1);
  const double last = product(blas, "last = T^T bracket (large)", s, l, rows, 1, 3);

  /* per V neighbour: W, and the four terms of all ABFs of the centre atom */
  const double w = product(blas, "W = V T", u * l, np, nq, nv, 5) / nv;
  const double b13 = batched(blas, "(i),(iii) per ABF: u x u, inner l", u, u, l, np, nv, 7, &with_pointers) / nv;
  const double b13p = with_pointers / nv;
  const double b24 = batched(blas, "(ii),(iv) per ABF: l x u, inner u", l, u, u, np, nv, 9, &with_pointers) / nv;
  const double b24p = with_pointers / nv;
  const double b24t = batched(blas, "(ii),(iv) per ABF: u x l, inner u", u, l, u, np, nv, 11, &with_pointers) / nv;
  const double b24tp = with_pointers / nv;
  const double f1 = product(blas, "(i) all ABFs: u x u, inner np*l", u, u, np * l, nv, 13) / nv;
  const double f2 = product(blas, "(ii) all ABFs: l x u, inner np*u", l, u, rows, nv, 15) / nv;
  const double f2t = product(blas, "(ii) all ABFs: u x l, inner np*u", u, l, rows, nv, 17) / nv;
  const double f3 = product(blas, "(iii) all ABFs: np*u x u, inner l", rows, u, l, nv, 19) / nv;
  const double f3t = product(blas, "(iii) all ABFs: u x np*u, inner l", u, rows, l, nv, 21) / nv;
  const double f4 = product(blas, "(iv) all ABFs: np*l x u, inner u", np * l, u, u, nv, 23) / nv;
  const double f4t = product(blas, "(iv) all ABFs: u x np*l, inner u", u, np * l, u, nv, 25) / nv;

  /* the cost of a call: a very small product, waited for, and queued */
  product(blas, "13 x 13 x 13, waited for every time", 13, 13, 13, 1, 27);
  product(blas, "13 x 13 x 13, 1000 queued", 13, 13, 13, 1000, 29);

  /* memory: the arrays of one V neighbour and of one centre atom */
  double copy_v = 0.0, add_v = 0.0;
  printf("\n%-18s %12s  %-26s %6s %14s %15s\n", "array", "size", "operation", "queued", "per call", "rate");
  memory_lines(blas, "np*u x l (one V)", (size_t)rows * (size_t)l * sizeof(double), nv, &copy_v, &add_v);
  memory_lines(blas, "np*u x s (G)", (size_t)rows * (size_t)s * sizeof(double), 1, &ignore, &ignore);
  memory_lines(blas, "l x s (D, last)", (size_t)l * (size_t)s * sizeof(double), 1, &ignore, &ignore);

  /* One centre atom from the lines above: the two large products, and per V
     neighbour W, the four terms, two copies of np*u x l numbers (the columns
     of G, the first term's source) and two additions of that size (the
     additions to the bracket).  The best of the measured forms of a term
     is taken. */
  const double best2 = f2 < f2t ? f2 : f2t, best3 = f3 < f3t ? f3 : f3t, best4 = f4 < f4t ? f4 : f4t;
  const double fused = f1 + best2 + best3 + best4;
  const double strided = 2.0 * b13 + 2.0 * (b24 < b24t ? b24 : b24t);
  const double pointers = 2.0 * b13p + 2.0 * (b24p < b24tp ? b24p : b24tp);
  const double memory = 2.0 * copy_v + 2.0 * add_v;
  printf("\none centre atom of %s, seconds (products on the device, arrays already there):\n", sy->name);
  printf("  large products                      %9.4f\n", g + last);
  printf("  W of %d V neighbours               %9.4f\n", nv, nv * w);
  printf("  four terms, one call for all ABFs   %9.4f\n", nv * fused);
  printf("  four terms, batched with strides    %9.4f\n", nv * strided);
  printf("  four terms, batched with pointers   %9.4f\n", nv * pointers);
  printf("  copies and additions of np*u x l    %9.4f\n", nv * memory);
  printf("  sum with the fastest form of terms  %9.4f\n",
         g + last + nv * (w + (fused < strided ? (fused < pointers ? fused : pointers)
                                                : (strided < pointers ? strided : pointers)) + memory));
  printf("checks that failed: %d\n", failures);
  blas_ok(cublasDestroy(blas), "cublasDestroy");
  return failures ? 1 : 0;
}
