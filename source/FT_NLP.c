/**********************************************************************
  FT_NLP.c:

     FT_NLP.c is a subroutine to Fourier transform projectors
     of nonlocal potentials.

  Log of FT_NLP.c:

     15/Sep/2002  Released by T.Ozaki

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



void FT_NLP(OpenMXCoreContext *core, int device)
{
  int numprocs,myid,ID,NumSpe;
  int i,num_k,so,Lspe,spe,L;
  int RestartRead_Succeed;
  double dk,rmin,rmax,h;
  double TStime,TEtime;

  char fileFT[YOUSO10];
  FILE *fp;
  size_t size; 

  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
 
  if (myid==Host_ID && MYID_MPI_COMM_WORLD==Host_ID && 0<level_stdout){
    printf("<FT_NLP>          Fourier transform of non-local projectors\n");
  }

  RestartRead_Succeed = 0;

  /***********************************************************
     In case of Scf_RestartFromFile==1, read Spe_NLRF_Bessel
  ***********************************************************/

  if (Scf_RestartFromFile==1){

    /****************************************************
          generate radial grids in the k-space
    ****************************************************/

    dk = PAO_Nkmax/(double)Ngrid_NormK;
    for (i=0; i<Ngrid_NormK; i++){
      NormK[i] = (double)i*dk;
    }

    /***********************************************************
                      read Spe_NLRF_Bessel
    ***********************************************************/

    sprintf(fileFT,"%s%s_rst/%s.ftnlp",filepath,restart_filename,restart_filename);

    if ((fp = fopen(fileFT,"rb")) != NULL){

      RestartRead_Succeed = 1;

      for (spe=0; spe<SpeciesNum; spe++){
	for (so=0; so<=VPS_j_dependency[spe]; so++){
	  for (L=1; L<=Spe_Num_RVPS[spe]; L++){

	    size = fread(&Spe_NLRF_Bessel[so][spe][L][0],sizeof(double),List_YOUSO[15],fp);
	    if (size!=List_YOUSO[15]) RestartRead_Succeed = 0;
	  }
	}
      }

      fclose(fp);
    }
    else{
      printf("Could not open a file %s in FT_NLP\n",fileFT);
    }
  }

  /***********************************************************
     if (RestartRead_Succeed==0), calculate Spe_NLRF_Bessel
  ***********************************************************/

  if (RestartRead_Succeed==0){

    for (Lspe=0; Lspe<MSpeciesNum; Lspe++){

      spe = Species_Top[myid] + Lspe;

      num_k = Ngrid_NormK;
      dk = PAO_Nkmax/(double)num_k;
      rmin = Spe_VPS_RV[spe][0];
      rmax = Spe_Atom_Cut1[spe] + 0.5;
      h = (rmax - rmin)/(double)OneD_Grid;

      {
        char error[256];
        OpenMXProjectorTable table = {0};
        const OpenMXUniformRadialGrid grid = {OneD_Grid,num_k,rmin,h,dk};
        table.projector_count = Spe_Num_RVPS[spe];
        table.angular = Spe_VPS_List[spe]+1;
        table.spin_count = VPS_j_dependency[spe]+1;
        table.mesh_count = Spe_Num_Mesh_VPS[spe];
        table.radii = Spe_VPS_RV[spe];
        for (so=0; so<table.spin_count && so<2; so++) {
          table.values[so] = Spe_VNL[so][spe];
          table.output[so] = Spe_NLRF_Bessel[so][spe]+1;
        }
        if (!openmx_core_transform_projectors(core,device,&table,&grid,error,sizeof(error))) {
          fprintf(stderr,"FT_NLP: %s\n",error);
          MPI_Abort(mpi_comm_level1,1);
          return;
        }
      }

    } /* Lspe */

    /****************************************************
     Remedy for MSpeciesNum==0
     generate radial grids in the k-space
    ****************************************************/

    dk = PAO_Nkmax/(double)Ngrid_NormK;
    for (i=0; i<Ngrid_NormK; i++){
      NormK[i] = (double)i*dk;
    }

    /***********************************************************
        sending and receiving of Spe_RF_Bessel by MPI
    ***********************************************************/

    for (ID=0; ID<Num_Procs2; ID++){
      NumSpe = Species_End[ID] - Species_Top[ID] + 1;
      for (Lspe=0; Lspe<NumSpe; Lspe++){
	spe = Species_Top[ID] + Lspe;
	for (so=0; so<=VPS_j_dependency[spe]; so++){
	  for (L=1; L<=Spe_Num_RVPS[spe]; L++){
	    MPI_Bcast(&Spe_NLRF_Bessel[so][spe][L][0],
		      List_YOUSO[15],MPI_DOUBLE,ID,mpi_comm_level1);
	  }
	}
      }
    }

    /***********************************************************
                      save Spe_NLRF_Bessel
    ***********************************************************/

    if (myid==Host_ID){

      sprintf(fileFT,"%s%s_rst/%s.ftnlp",filepath,filename,filename);

      if ((fp = fopen(fileFT,"wb")) != NULL){

	for (spe=0; spe<SpeciesNum; spe++){
  	  for (so=0; so<=VPS_j_dependency[spe]; so++){
	    for (L=1; L<=Spe_Num_RVPS[spe]; L++){
	      fwrite(&Spe_NLRF_Bessel[so][spe][L][0],sizeof(double),List_YOUSO[15],fp);
	    }
	  }
	}  

	fclose(fp);
      }
      else{
	printf("Could not open a file %s in FT_NLP\n",fileFT);
      }
    }

  } /* if (RestartRead_Succeed==0) */

  /***********************************************************
                         elapsed time
  ***********************************************************/

  dtime(&TEtime);

  /*
  printf("FT_NLP: myid=%2d Elapsed Time (s) = %15.12f\n",myid,TEtime-TStime);
  MPI_Finalize();
  exit(0);
  */

}
