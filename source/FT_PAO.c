/**********************************************************************
  FT_PAO.c:

     FT_PAO.c is a subroutine to Fourier transform pseudo atomic 
     orbitals.

  Log of FT_PAO.c:

     15/Sep/2002  Released by T. Ozaki

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



void FT_PAO(OpenMXCoreContext *core, int device)
{
  int numprocs,myid,ID,NumSpe;
  int num_k,Lspe,spe,GL,Mul;
  int RestartRead_Succeed;
  double dk,h,rmin,rmax;
  double TStime, TEtime;

  char fileFT[YOUSO10];
  char operate[300];
  FILE *fp;
  size_t size; 

  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID && MYID_MPI_COMM_WORLD==Host_ID && 0<level_stdout){
    printf("<FT_PAO>          Fourier transform of pseudo atomic orbitals\n");
  }

  RestartRead_Succeed = 0;

  /***********************************************************
      In case of Scf_RestartFromFile==1, read Spe_RF_Bessel
  ***********************************************************/

  if (Scf_RestartFromFile==1){

    /***********************************************************
                      read Spe_RF_Bessel
    ***********************************************************/

    sprintf(fileFT,"%s%s_rst/%s.ftpao",filepath,restart_filename,restart_filename);

    if ((fp = fopen(fileFT,"rb")) != NULL){

      RestartRead_Succeed = 1;

      for (spe=0; spe<SpeciesNum; spe++){
	for (GL=0; GL<=Spe_MaxL_Basis[spe]; GL++){
	  for (Mul=0; Mul<Spe_Num_Basis[spe][GL]; Mul++){

	    size = fread(&Spe_RF_Bessel[spe][GL][Mul][0],sizeof(double),List_YOUSO[15],fp);
            if (size!=List_YOUSO[15]) RestartRead_Succeed = 0;
	  }
	}
      }  

      fclose(fp);
    }
    else{
      printf("Could not open a file %s in FT_PAO\n",fileFT);
    }

  }

  /***********************************************************
     if (RestartRead_Succeed==0), calculate Spe_RF_Bessel
  ***********************************************************/

  if (RestartRead_Succeed==0){

    for (Lspe=0; Lspe<MSpeciesNum; Lspe++){

      spe = Species_Top[myid] + Lspe;

      num_k = Ngrid_NormK;
      dk = PAO_Nkmax/(double)num_k;
      rmin = Spe_PAO_RV[spe][0];
      rmax = Spe_Atom_Cut1[spe] + 0.5;
      h = (rmax - rmin)/(double)OneD_Grid;

      {
        char error[256];
        const OpenMXPaoTable table = {Spe_MaxL_Basis[spe],Spe_Num_Basis[spe],
          Spe_Num_Mesh_PAO[spe],Spe_PAO_RV[spe],Spe_PAO_RWF[spe],Spe_RF_Bessel[spe]};
        const OpenMXUniformRadialGrid grid = {OneD_Grid,num_k,rmin,h,dk};
        if (!openmx_core_transform_pao(core,device,&table,&grid,error,sizeof(error))) {
          fprintf(stderr,"FT_PAO: %s\n",error);
          MPI_Abort(mpi_comm_level1,1);
          return;
        }
      }

    } /* Lspe */

    /***********************************************************
        sending and receiving of Spe_RF_Bessel by MPI
    ***********************************************************/

    for (ID=0; ID<Num_Procs2; ID++){
      NumSpe = Species_End[ID] - Species_Top[ID] + 1;
      for (Lspe=0; Lspe<NumSpe; Lspe++){
	spe = Species_Top[ID] + Lspe;
	for (GL=0; GL<=Spe_MaxL_Basis[spe]; GL++){
	  for (Mul=0; Mul<Spe_Num_Basis[spe][GL]; Mul++){

	    MPI_Bcast(&Spe_RF_Bessel[spe][GL][Mul][0],
		      List_YOUSO[15],MPI_DOUBLE,ID, mpi_comm_level1);
            MPI_Barrier(mpi_comm_level1);
	  }
	}
      }
    }

    /***********************************************************
                      save Spe_RF_Bessel
    ***********************************************************/

    if (myid==Host_ID){

      sprintf(operate,"%s%s_rst",filepath,filename);
      mkdir(operate,0775); 
   
      sprintf(fileFT,"%s%s_rst/%s.ftpao",filepath,filename,filename);

      if ((fp = fopen(fileFT,"wb")) != NULL){

	for (spe=0; spe<SpeciesNum; spe++){
	  for (GL=0; GL<=Spe_MaxL_Basis[spe]; GL++){
	    for (Mul=0; Mul<Spe_Num_Basis[spe][GL]; Mul++){
	      fwrite(&Spe_RF_Bessel[spe][GL][Mul][0],sizeof(double),List_YOUSO[15],fp);
	    }
	  }
	}  

	fclose(fp);
      }
      else{
	printf("Could not open a file %s in FT_PAO\n",fileFT);
      }
    }

  } /* if (RestartRead_Succeed==0) */

  /***********************************************************
                         elapsed time
  ***********************************************************/

  dtime(&TEtime);

  /*
  printf("FT_PAO.c: Time=%15.12f\n",TEtime-TStime);
  MPI_Finalize();
  exit(0);
  */

}



