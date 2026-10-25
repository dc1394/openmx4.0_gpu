/* Verify the MPI DSO and communicator using the production native rpaths. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    Dl_info library;
    char actual[PATH_MAX], expected[PATH_MAX];
    if (argc > 1 && argv[1][0]) {
        if (!dladdr((void *)MPI_Init, &library) ||
            !realpath(library.dli_fname, actual) || !realpath(argv[1], expected)) {
            fprintf(stderr, "Cannot resolve the loaded MPI library\n");
            return 1;
        }
        if (strcmp(actual, expected)) {
            fprintf(stderr, "Wrong MPI runtime: %s (expected %s)\n", actual, expected);
            return 1;
        }
    }
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        fprintf(stderr, "Expected two communicating MPI ranks, got %d\n", size);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (!rank) puts("Native MPI runtime and two-rank communicator: PASS");
    MPI_Finalize();
    return 0;
}
