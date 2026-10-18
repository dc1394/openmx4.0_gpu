/**********************************************************************
  FT_VNA.c:

     FT_VNA.c is a subroutine to Fourier transform VNA potentials

  Log of FT_VNA.c:

     18/May/2004  Released by T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
/*  stat section */
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
/*  end stat section */
#include "openmx_common.h"
#include "mpi.h"
#include "openmx_core_radial.h"





void FT_VNA(OpenMXCoreContext *core, int device)
{
  int numprocs,myid,ID,NumSpe;
  int Lspe,spe;
  int RestartRead_Succeed;
  double TStime,TEtime;

  char fileFT[YOUSO10];
  FILE *fp;
  size_t size; 

  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID && MYID_MPI_COMM_WORLD==Host_ID && 0<level_stdout){
    printf("<FT_VNA>          Fourier transform of VNA potentials\n");
  }

  RestartRead_Succeed = 0;

  /***********************************************************
   In case of Scf_RestartFromFile==1, read Spe_CrudeVNA_Bessel
  ***********************************************************/

  if (Scf_RestartFromFile==1){

    /***********************************************************
                        read Spe_VNA_Bessel
    ***********************************************************/

    sprintf(fileFT,"%s%s_rst/%s.ftCvna",filepath,restart_filename,restart_filename);

    if ((fp = fopen(fileFT,"rb")) != NULL){

      RestartRead_Succeed = 1;

      for (spe=0; spe<SpeciesNum; spe++){

	size = fread(&Spe_CrudeVNA_Bessel[spe][0],sizeof(double),GL_Mesh,fp);
	if (size!=GL_Mesh) RestartRead_Succeed = 0;
      }

      fclose(fp);
    }
    else{
      printf("Error1: Could not open a file %s in FT_VNA\n",fileFT);
    }
  }

  /***********************************************************
   if (RestartRead_Succeed==0), calculate Spe_CrudeVNA_Bessel
  ***********************************************************/

  if (RestartRead_Succeed==0){

    /* loop for Lspe */

    for (Lspe=0; Lspe<MSpeciesNum; Lspe++){

      spe = Species_Top[myid] + Lspe;

      {
        char error[256];
        const OpenMXNeutralTable table = {Spe_Num_Mesh_VPS[spe],Spe_VPS_RV[spe],
          Spe_Vna[spe],Spe_Atom_Cut1[spe],Spe_CrudeVNA_Bessel[spe]};
        const OpenMXSquaredRadialGrid grid = {OneD_Grid,GL_Mesh,Spe_PAO_RV[spe][0],
          Spe_Atom_Cut1[spe]+0.5,Radial_kmin,PAO_Nkmax,GL_Abscissae};
        if (!openmx_core_transform_neutral_potential(core,device,&table,&grid,error,sizeof(error))) {
          fprintf(stderr,"FT_VNA: %s\n",error);
          MPI_Abort(mpi_comm_level1,1);
          return;
        }
      }

    } /* Lspe */

    /***********************************************************
      sending and receiving of Spe_CrudeVNA_Bessel by MPI
    ***********************************************************/

    for (ID=0; ID<Num_Procs2; ID++){
      NumSpe = Species_End[ID] - Species_Top[ID] + 1;
      for (Lspe=0; Lspe<NumSpe; Lspe++){
	spe = Species_Top[ID] + Lspe;
	MPI_Bcast(&Spe_CrudeVNA_Bessel[spe][0],
		  GL_Mesh,MPI_DOUBLE,ID,mpi_comm_level1);
      }
    }

    /***********************************************************
                        save Spe_CrudeVNA_Bessel
    ***********************************************************/

    if (myid==Host_ID){

      sprintf(fileFT,"%s%s_rst/%s.ftCvna",filepath,filename,filename);

      if ((fp = fopen(fileFT,"wb")) != NULL){

	for (spe=0; spe<SpeciesNum; spe++){
	  fwrite(&Spe_CrudeVNA_Bessel[spe][0],sizeof(double),GL_Mesh,fp);
	}

	fclose(fp);
      }
      else{
	printf("Error2: Could not open a file %s in FT_VNA\n",fileFT);
      }
    }

  } /* if (RestartRead_Succeed==0) */

  /***********************************************************
                         elapsed time
  ***********************************************************/

  dtime(&TEtime);

  /*
  printf("FT_VNA: myid=%2d Elapsed Time (s) = %15.12f\n",myid,TEtime-TStime);
  MPI_Finalize();
  exit(0);
  */

}




