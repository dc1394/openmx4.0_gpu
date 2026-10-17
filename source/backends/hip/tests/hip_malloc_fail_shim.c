/* LD_PRELOAD shim for the device-memory failure paths of the GPU solvers:
   hipMalloc fails with hipErrorOutOfMemory for allocations of one exact size,
   so that a run takes the path of a full device without filling it.

     OPENMX_TEST_HIPMALLOC_FAIL_SIZE   bytes of the allocations to fail (unset or 0:
                                       the shim only passes the calls through)
     OPENMX_TEST_HIPMALLOC_FAIL_SKIP   the first so many allocations of that size
                                       succeed (default 0)
     OPENMX_TEST_HIPMALLOC_FAIL_COUNT  at most so many fail (default: every later one)
     OPENMX_TEST_HIPMALLOC_FAIL_RANK   only in this MPI_COMM_WORLD rank (from
                                       OMPI_COMM_WORLD_RANK or PMI_RANK; default every rank)

   Each failure prints "TEST_HIPMALLOC_FAIL: ..." on stderr.

   Build:
     gcc -shared -fPIC -O2 -D__HIP_PLATFORM_AMD__ -I/opt/rocm-7.2.0/include \
         -o hip_malloc_fail_shim.so tests/hip_malloc_fail_shim.c -ldl

   Example, the collinear cluster solver's admission (n = 2808: a dense panel is
   2808 * 2808 * 8 = 63078912 bytes): the dense owner's second panel fails on
   the first try and the retry fails again, so SCF step 1 falls back to ELPA
   collectively ("could not allocate its GPU matrices", "using ELPA fallback")
   and step 2 is admitted to the GPU again:
     mpirun -np 16 -x LD_PRELOAD=$PWD/hip_malloc_fail_shim.so \
       -x OPENMX_TEST_HIPMALLOC_FAIL_SIZE=63078912 -x OPENMX_TEST_HIPMALLOC_FAIL_SKIP=1 \
       -x OPENMX_TEST_HIPMALLOC_FAIL_COUNT=2 -x OPENMX_TEST_HIPMALLOC_FAIL_RANK=0 \
       openmx scol.dat -nt 1
*/
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <hip/hip_runtime_api.h>

typedef hipError_t (*hip_malloc_fn)(void **, size_t);

static pthread_mutex_t shim_lock = PTHREAD_MUTEX_INITIALIZER;
static int             shim_ready = 0;
static size_t          fail_size = 0;
static long            fail_skip = 0, fail_count = -1, matches = 0, failures = 0;
static hip_malloc_fn   real_hip_malloc = NULL;

static long env_long(const char *name, long fallback)
{
    const char *value = getenv(name);
    char       *end = NULL;
    long        parsed;

    if (value == NULL || value[0] == '\0') return fallback;
    parsed = strtol(value, &end, 10);
    return (end == value) ? fallback : parsed;
}

static void shim_init(void)
{
    long rank = env_long("OMPI_COMM_WORLD_RANK", env_long("PMI_RANK", -1));
    long only = env_long("OPENMX_TEST_HIPMALLOC_FAIL_RANK", -1);

    real_hip_malloc = (hip_malloc_fn)dlsym(RTLD_NEXT, "hipMalloc");
    fail_size = (size_t)env_long("OPENMX_TEST_HIPMALLOC_FAIL_SIZE", 0);
    fail_skip = env_long("OPENMX_TEST_HIPMALLOC_FAIL_SKIP", 0);
    fail_count = env_long("OPENMX_TEST_HIPMALLOC_FAIL_COUNT", -1);
    if (0 <= only && rank != only) fail_size = 0;
    shim_ready = 1;
}

hipError_t hipMalloc(void **ptr, size_t size)
{
    int  fail = 0;
    long match_no = 0, failure_no = 0;

    pthread_mutex_lock(&shim_lock);
    if (!shim_ready) shim_init();
    if (fail_size != 0 && size == fail_size) {
        if (fail_skip <= matches && (fail_count < 0 || failures < fail_count)) {
            fail = 1;
            failures++;
        }
        matches++;
        match_no = matches;
        failure_no = failures;
    }
    pthread_mutex_unlock(&shim_lock);

    if (fail) {
        fprintf(stderr, "TEST_HIPMALLOC_FAIL: hipMalloc(%zu) failed on purpose (match %ld, failure %ld)\n", size,
                match_no, failure_no);
        fflush(stderr);
        if (ptr != NULL) *ptr = NULL;
        return hipErrorOutOfMemory;
    }
    if (real_hip_malloc == NULL) return hipErrorNotInitialized;
    return real_hip_malloc(ptr, size);
}
