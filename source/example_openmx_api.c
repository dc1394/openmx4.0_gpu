/**********************************************************************
  example_openmx_api.c:

    example_openmx_api.c is a minimal external C program showing
    how to split MPI_COMM_WORLD into subcommunicators and call
    the OpenMX API.

 Log of example_openmx_api.h:

    15/April/2026  Released by T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <omp.h>
#include "openmx_api.h"


int main(int argc, char *argv[])
{
  int world_rank, world_size;
  int color, sub_rank, sub_size;
  int nsplit,Nth_max,Nth;
  int ierr;
  const char *omx_input_files[100];
  MPI_Comm subcomm;
 
  /* MPI initialization */

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  /* get the number of threads */

  Nth_max = omp_get_max_threads();
  Nth = 1;
  if (Nth_max<Nth){
    MPI_Finalize();
    return 1;
  }

  /* set nsplit and input files to be passed */

  nsplit = 2;
  omx_input_files[0] = "./Methane.dat";
  omx_input_files[1] =  "./Cdia.dat";

  if (nsplit > world_size) {
    if (world_rank == 0) {
      fprintf(stderr,
	      "Error: Nsplit = %d is larger than the number of MPI processes = %d.\n",
	      nsplit, world_size);
    }
    MPI_Finalize();
    return 1;
  }

  /* Create subgroup index. */
  color = (world_rank * nsplit) / world_size;

  /* Split MPI_COMM_WORLD into nsplit subcommunicators. */
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &subcomm);
  MPI_Comm_rank(subcomm, &sub_rank);
  MPI_Comm_size(subcomm, &sub_size);

  /* Call the OpenMX API from each subgroup. */
  ierr = openmx_api(subcomm, omx_input_files[color], Nth);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Clean up. */
  MPI_Comm_free(&subcomm);
  MPI_Finalize();

  return ierr;
}
