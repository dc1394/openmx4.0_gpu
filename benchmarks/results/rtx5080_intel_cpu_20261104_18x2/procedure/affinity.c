#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <sched.h>
#include <mpi.h>
#include <omp.h>
int main(int argc,char **argv) {
 int rank,provided; MPI_Init_thread(&argc,&argv,MPI_THREAD_SERIALIZED,&provided); MPI_Comm_rank(MPI_COMM_WORLD,&rank);
 #pragma omp parallel num_threads(2)
 {
  cpu_set_t mask; sched_getaffinity(0,sizeof(mask),&mask);
  #pragma omp critical
  {
   char buf[1024]; int used=snprintf(buf,sizeof(buf),"{\"rank\":%d,\"thread\":%d,\"team\":%d,\"cpu\":%d,\"allowed\":[",rank,omp_get_thread_num(),omp_get_num_threads(),sched_getcpu());
   int n=0; for(int i=0;i<CPU_SETSIZE;i++) if(CPU_ISSET(i,&mask))used+=snprintf(buf+used,sizeof(buf)-used,"%s%d",n++?",":"",i);
   used+=snprintf(buf+used,sizeof(buf)-used,"]}\n"); write(1,buf,used);
  }
 }
 MPI_Finalize(); return 0;
}
