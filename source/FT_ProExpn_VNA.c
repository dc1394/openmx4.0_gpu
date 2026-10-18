/**********************************************************************
  FT_ProExpn_VNA.c:

     FT_ProExpn_VNA.c is a subroutine to Fourier transform
     VNA separable projectors.

  Log of FT_ProExpn_VNA.c:

     7/Apr/2004  Released by T.Ozaki

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
#include "openmx_core_neutral.h"



void FT_ProExpn_VNA(OpenMXCoreContext *core, int device)
{
  int numprocs,myid,ID,NumSpe;
  int L,Lspe,spe,Mul;
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
    printf("<FT_ProExpn_VNA>  Fourier transform of VNA separable projectors\n");
  }

  RestartRead_Succeed = 0;

  /***********************************************************
     In case of Scf_RestartFromFile==1, read Spe_VNA_Bessel
  ***********************************************************/

  if (Scf_RestartFromFile==1){

    /***********************************************************
                        read Spe_VNA_Bessel
    ***********************************************************/

    sprintf(fileFT,"%s%s_rst/%s.ftPEvna",filepath,restart_filename,restart_filename);

    if ((fp = fopen(fileFT,"rb")) != NULL){

      RestartRead_Succeed = 1;

      for (spe=0; spe<SpeciesNum; spe++){
        for (L=0; L<=List_YOUSO[35]; L++){
	  for (Mul=0; Mul<List_YOUSO[34]; Mul++){

	    size = fread(&Spe_VNA_Bessel[spe][L][Mul][0],sizeof(double),GL_Mesh,fp);
	    if (size!=GL_Mesh) RestartRead_Succeed = 0;
	  }
        }
      }

      fclose(fp);
    }
    else{
      printf("Could not open a file %s in FT_ProExpn_VNA\n",fileFT);
    }
  }

  /***********************************************************
     if (RestartRead_Succeed==0), calculate Spe_VNA_Bessel
  ***********************************************************/

  if (RestartRead_Succeed==0){

    for (Lspe=0; Lspe<MSpeciesNum; Lspe++){

      spe = Species_Top[myid] + Lspe;

      const OpenMXNeutralProjectorTable table = {List_YOUSO[35],List_YOUSO[34],
        Spe_Num_Mesh_VPS[spe],Spe_VPS_RV[spe],Projector_VNA[spe],Spe_VNA_Bessel[spe]};
      const OpenMXGaussRadialGrid grid = {GL_Mesh,GL_Mesh,Spe_VPS_RV[spe][0],
        Spe_Atom_Cut1[spe]+0.5,Radial_kmin,PAO_Nkmax,GL_Abscissae,GL_Weight,GL_Abscissae};
      char error[256];
      if (!openmx_core_transform_neutral_projectors(core,device,&table,&grid,error,sizeof(error))) {
        fprintf(stderr,"FT_ProExpn_VNA species %d: %s\n",spe,error);
        MPI_Abort(mpi_comm_level1,1);
        return;
      }

    } /* Lspe */

    /***********************************************************
        sending and receiving of Spe_VNA_Bessel by MPI
    ***********************************************************/

    for (ID=0; ID<Num_Procs2; ID++){
      NumSpe = Species_End[ID] - Species_Top[ID] + 1;
      for (Lspe=0; Lspe<NumSpe; Lspe++){
	spe = Species_Top[ID] + Lspe;
	for (L=0; L<=List_YOUSO[35]; L++){
	  for (Mul=0; Mul<List_YOUSO[34]; Mul++){
	    MPI_Bcast(&Spe_VNA_Bessel[spe][L][Mul][0],
		      GL_Mesh,MPI_DOUBLE,ID,mpi_comm_level1);
	  }
	}
      }
    }

    /***********************************************************
                      save Spe_VNA_Bessel
    ***********************************************************/

    if (myid==Host_ID){

      sprintf(fileFT,"%s%s_rst/%s.ftPEvna",filepath,filename,filename);

      if ((fp = fopen(fileFT,"wb")) != NULL){

	for (spe=0; spe<SpeciesNum; spe++){
	  for (L=0; L<=List_YOUSO[35]; L++){
	    for (Mul=0; Mul<List_YOUSO[34]; Mul++){
	      fwrite(&Spe_VNA_Bessel[spe][L][Mul][0],sizeof(double),GL_Mesh,fp);
	    }
	  }
	}

	fclose(fp);
      }
      else{
	printf("Could not open a file %s in FT_ProExpn_VNA\n",fileFT);
      }
    }

  } /* if (RestartRead_Succeed==0) */

  /***********************************************************
                         elapsed time
  ***********************************************************/

  dtime(&TEtime);

  /*
  printf("FT_ProExpn_VNA: myid=%2d Elapsed Time (s) = %15.12f\n",myid,TEtime-TStime);
  MPI_Finalize();
  exit(0);
  */

}




