/**********************************************************************
  openmx_api.h:

    openmx_api.h is a header file declaring the public C API routines for OpenMX.
    When you create a C code which calls the API of OpenMX, please include
    the header file.

 Log of openmx_api.h:
                                                                                                                    
    15/April/2026  Released by T. Ozaki

***********************************************************************/

#ifndef OPENMX_API_H
#define OPENMX_API_H

#include "mpi.h"

int openmx_api(MPI_Comm mpi_commWD, const char *omp_input_file, int OMP_threadN);
int openmx_api_f(MPI_Fint fcomm, const char *omp_input_file, int OMP_threadN);

#endif
