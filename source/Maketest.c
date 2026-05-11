/**********************************************************************
  Maketest.c:

     Maketest.c is a subroutine to generate *.out files which will be
     used to check whether OpenMX runs normally on many platforms or not.

  Log of Maketest.c:

     25/Oct/2004  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
/*  stat section */
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
/*  end stat section */
#include "openmx_common.h"
#include "Inputtools.h"
#include "mpi.h"
 
typedef struct {
  char fn[YOUSO10];
} fname_type;

int run_main(int argc, char *argv[], int numprocs0, int myid0);
int stringcomp( const void *a, const void *b);


void Maketest(char *mode, int argc, char *argv[]) 
{
  FILE *fp0,*fp1,*fp2;
  static int Num_DatFiles,i;

  static char fname0[YOUSO10];
  static char fname1[YOUSO10];
  static char fname_dat[YOUSO10];
  static char fname_dat2[YOUSO10];
  static char operate[800];
  fname_type *fndat;
  char namemode[YOUSO10];
  char *dir;
  DIR *dp;
  struct dirent *entry;
  int numprocs,myid;

  MPI_Comm_size(mpi_comm_level1, &numprocs);
  MPI_Comm_rank(mpi_comm_level1, &myid);

  if (myid==Host_ID){
    printf("\n*******************************************************\n"); fflush(stdout);  
    printf("*******************************************************\n"); fflush(stdout);  
    printf(" Welcome to OpenMX  Ver. %s                            \n",Version_OpenMX); fflush(stdout);  
    printf(" Copyright (C), 2002-2026, T.Ozaki                     \n"); fflush(stdout);  
    printf(" OpenMX comes with ABSOLUTELY NO WARRANTY.             \n"); fflush(stdout);  
    printf(" This is free software, and you are welcome to         \n"); fflush(stdout);  
    printf(" redistribute it under the constitution of the GNU-GPL.\n");fflush(stdout);  
    printf("*******************************************************\n"); fflush(stdout);  
    printf("*******************************************************\n\n\n"); fflush(stdout);  
  }

  if (strcasecmp(mode,"S")==0){  
    dir = "input_example";
    sprintf(namemode,"runtest");
  }
  else if (strcasecmp(mode,"L")==0){  
    dir = "large_example";
    sprintf(namemode,"runtestL");
  }
  else if (strcasecmp(mode,"L2")==0){  
    dir = "large2_example";
    sprintf(namemode,"runtestL2");
  }
  else if (strcasecmp(mode,"L3")==0){  
    dir = "large3_example";
    sprintf(namemode,"runtestL3");
  }
  else if (strcasecmp(mode,"G")==0){  
    dir = "geoopt_example";
    sprintf(namemode,"runtestG");
  }
  else if (strcasecmp(mode,"C")==0){  
    dir = "cellopt_example";
    sprintf(namemode,"runtestC");
  }
  else if (strcasecmp(mode,"CWF")==0){  
    dir = "cwf_example";
    sprintf(namemode,"runtestCWF");
  }
  else if (strcasecmp(mode,"WF")==0){  
    dir = "wf_example";
    sprintf(namemode,"runtestWF");
  }
  else if (strcasecmp(mode,"NEGF")==0){  
    dir = "negf_example";
    sprintf(namemode,"runtestNEGF");
  }
  else if (strcasecmp(mode,"CDDF")==0){  
    dir = "cddf_example";
    sprintf(namemode,"runtestCDDF");
  }
  else if (strcasecmp(mode,"DCLNO")==0){  
    dir = "dclno_example";
    sprintf(namemode,"runtestDCLNO");
  }

  /* print std */

  if (myid==Host_ID){

    printf("\n");fflush(stdout);  
    printf(" OpenMX is now in the mode in making of reference files\n");fflush(stdout);  
    printf(" which will be used in the test mode '%s'.\n",namemode);fflush(stdout);  
    printf("\n");fflush(stdout);  

    /* count the number of dat files */

    if(( dp = opendir(dir) ) == NULL ){
      printf("could not find the directory '%s'\n",dir);
      MPI_Finalize();
      exit(0);
    }

    Num_DatFiles = 0;
    while((entry = readdir(dp)) != NULL){

      if ( strstr(entry->d_name,".dat")!=NULL ){ 
          
        Num_DatFiles++;
      }
    }
    closedir(dp);

    fndat = (fname_type*)malloc(sizeof(fname_type)*Num_DatFiles);

    /* store the name of dat files */

    if(( dp = opendir(dir) ) == NULL ){
      printf("could not find the directory '%s'\n",dir);
      MPI_Finalize();
      exit(0);
    }

    Num_DatFiles = 0;
    while((entry = readdir(dp)) != NULL){
 
      if ( strstr(entry->d_name,".dat")!=NULL ){ 

        sprintf(fndat[Num_DatFiles].fn,"%s/%s",dir,entry->d_name);  
        Num_DatFiles++;
      }
    }
    closedir(dp);

    /* sorting fndat */

    qsort(fndat, Num_DatFiles, sizeof(fname_type), stringcomp);  

    /*
    for (i=0; i<Num_DatFiles; i++){
      printf("i=%2d %s\n",i,fndat[i].fn);
    } 
    */

  } /* if (myid==Host_ID) */

  if (myid==Host_ID){
    printf(" %2d dat files are found in the directory '%s'.\n\n\n",Num_DatFiles,dir);
  }

  MPI_Bcast(&Num_DatFiles, 1, MPI_INT, Host_ID, mpi_comm_level1);

  /* start the calculations */

  for (i=0; i<Num_DatFiles; i++){

    if (myid==Host_ID){
      sprintf(fname_dat,"%s",fndat[i].fn);
    }  

    MPI_Bcast(&fname_dat, YOUSO10, MPI_CHAR, Host_ID, mpi_comm_level1);

    /* run openmx */

    argv[1] = fname_dat;
    run_main(argc, argv, numprocs, myid); 
    MPI_Barrier(mpi_comm_level1);

    if (myid==Host_ID){

      input_open(fname_dat);
      input_string("System.Name",fname_dat2,"default");
      input_close();

      sprintf(operate,"cp %s.out %s/",fname_dat2,dir);
      printf("%s\n",operate);
      system(operate);
    }
  }

  /* freeing of arrays */

  if (myid==Host_ID){
    free(fndat);
  }

  if (myid==Host_ID){
    printf("\n\n\n\n");
    printf("Generated out files are stored in the directory '%s'.\n\n\n",dir);
  }

  MPI_Barrier(mpi_comm_level1);
  MPI_Finalize();
  exit(0);
}

