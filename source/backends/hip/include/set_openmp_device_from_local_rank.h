#ifndef _SET_OPENMP_FROM_LOCAL_RANK_H_
#define _SET_OPENMP_FROM_LOCAL_RANK_H_

#include <mpi.h>

int set_openmp_device_from_local_rank(void);
int set_openmp_nvidia_device_from_local_rank(void);
int set_openmp_device_from_local_rank_noncollective(void);
int set_openmp_nvidia_device_from_local_rank_noncollective(void);

#endif // _SET_OPENMP_FROM_LOCAL_RANK_H_
