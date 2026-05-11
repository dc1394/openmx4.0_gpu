#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#undef LEAK_DETECT
#include "leakdetect.h"
#include "mpi.h"
#include <omp.h>

typedef struct MEM_IN_FILE_ {
  size_t size;
  const char *file;
  unsigned int line;
} MEM_IN_FILE;

typedef struct LeakNode_ {
  void *ptr;
  MEM_IN_FILE info;
  struct LeakNode_ *next;
} LeakNode;

static int num_alloc_mem = 0;          /* total number of tracked allocations */
static int num_live_mem  = 0;          /* number of currently live allocations */
static LeakNode **mem_table = NULL;
static size_t mem_table_size = 0;
static int leak_table_initialized = 0;
static int lock_initialized = 0;

static omp_lock_t lock;

/*
 * A modest hash table is sufficient for leak detection.
 * The table is allocated on initialization and reused until finalize.
 */
#define LEAK_HASH_TABLE_SIZE 65521u

static size_t leak_hash_ptr(const void *ptr)
{
  uintptr_t x = (uintptr_t)ptr;
  /* Remove low alignment bits and mix a little. */
  x >>= 4;
  x ^= (x >> 7);
  x ^= (x >> 17);
  return (size_t)(x % mem_table_size);
}

static int leak_table_setup(void)
{
  size_t i;

  if (mem_table != NULL) {
    return 1;
  }

  mem_table_size = (size_t)LEAK_HASH_TABLE_SIZE;
  mem_table = (LeakNode **)calloc(mem_table_size, sizeof(LeakNode *));
  if (mem_table == NULL) {
    mem_table_size = 0;
    return 0;
  }

  for (i = 0; i < mem_table_size; ++i) {
    mem_table[i] = NULL;
  }

  leak_table_initialized = 1;
  return 1;
}

static void leak_table_clear_no_lock(void)
{
  size_t i;

  if (mem_table == NULL) {
    num_live_mem = 0;
    num_alloc_mem = 0;
    return;
  }

  for (i = 0; i < mem_table_size; ++i) {
    LeakNode *node = mem_table[i];
    while (node != NULL) {
      LeakNode *next = node->next;
      free(node);
      node = next;
    }
    mem_table[i] = NULL;
  }

  num_live_mem = 0;
  num_alloc_mem = 0;
}

static void leak_table_finalize_no_lock(void)
{
  leak_table_clear_no_lock();

  if (mem_table != NULL) {
    free(mem_table);
    mem_table = NULL;
  }

  mem_table_size = 0;
  leak_table_initialized = 0;
}

static void leak_table_insert_no_lock(void *ptr, size_t size, const char *file, unsigned int line)
{
  size_t idx;
  LeakNode *node;

  if (ptr == NULL) {
    return;
  }

  if (!leak_table_initialized && !leak_table_setup()) {
    /*
     * Tracking failed because the internal table could not be allocated.
     * Keep the user allocation valid, but it will not be tracked.
     */
    return;
  }

  idx = leak_hash_ptr(ptr);

  node = (LeakNode *)malloc(sizeof(LeakNode));
  if (node == NULL) {
    /* Same policy as above: keep the allocation valid even if tracking fails. */
    return;
  }

  node->ptr = ptr;
  node->info.size = size;
  node->info.file = file;
  node->info.line = line;
  node->next = mem_table[idx];
  mem_table[idx] = node;

  ++num_alloc_mem;
  ++num_live_mem;
}

static void leak_table_erase_no_lock(void *ptr)
{
  size_t idx;
  LeakNode *node;
  LeakNode *prev;

  if (ptr == NULL || mem_table == NULL) {
    return;
  }

  idx = leak_hash_ptr(ptr);
  node = mem_table[idx];
  prev = NULL;

  while (node != NULL) {
    if (node->ptr == ptr) {
      if (prev == NULL) {
        mem_table[idx] = node->next;
      }
      else {
        prev->next = node->next;
      }
      free(node);
      --num_live_mem;
      return;
    }
    prev = node;
    node = node->next;
  }
}

/* initialize */
void leak_detect_init(void)
{
  int myid;

  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  if (myid == 0) {
    printf("LEAK_DETECT init---------------------------\n");
    fflush(stdout);
  }

  if (!lock_initialized) {
    omp_init_lock(&lock);
    lock_initialized = 1;
  }

  omp_set_lock(&lock);
  if (!leak_table_initialized) {
    (void)leak_table_setup();
  }
  leak_table_clear_no_lock();
  omp_unset_lock(&lock);
}

/* wrapper of malloc */
void *leak_detect_malloc(size_t size, const char *file, unsigned int line)
{
  void *ptr = malloc(size);

  if (ptr == NULL) {
    return NULL;
  }

  if (!lock_initialized) {
    /* Fail-safe: initialize lazily if the caller forgot leak_detect_init(). */
    omp_init_lock(&lock);
    lock_initialized = 1;
  }

  omp_set_lock(&lock);
  leak_table_insert_no_lock(ptr, size, file, line);
  omp_unset_lock(&lock);

  return ptr;
}

/* wrapper of free */
void leak_detect_free(void *ptr)
{
  if (lock_initialized) {
    omp_set_lock(&lock);
    leak_table_erase_no_lock(ptr);
    omp_unset_lock(&lock);
  }
  free(ptr);
}

/* show memory leak */
void leak_detect_check(void)
{
  int myid;
  size_t i;

  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  if (!lock_initialized) {
    omp_init_lock(&lock);
    lock_initialized = 1;
  }

  omp_set_lock(&lock);

  printf("ID=%2d LEAK_DETECT result (%d / %d)-----------------------\n",
         myid, num_live_mem, num_alloc_mem);
  fflush(stdout);

  if (mem_table != NULL) {
    for (i = 0; i < mem_table_size; ++i) {
      LeakNode *node = mem_table[i];
      while (node != NULL) {
        printf("ID=%2d memory leak!, addr:%p\n", myid, node->ptr);
        printf(" size:%u\n", (unsigned int)node->info.size);
        printf(" file,line:%s:%u\n", node->info.file, node->info.line);
        printf("\n");
        fflush(stdout);
        node = node->next;
      }
    }
  }

  omp_unset_lock(&lock);
}

/* finalize */
void leak_detect_finalize(void)
{
  if (!lock_initialized) {
    return;
  }

  omp_set_lock(&lock);
  leak_table_finalize_no_lock();
  omp_unset_lock(&lock);

  omp_destroy_lock(&lock);
  lock_initialized = 0;
}
