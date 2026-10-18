/**********************************************************************
  FT_ProductPAO.c:

     FT_ProductPAO.c is a subroutine to Fourier transform the
     product of two pseudo atomic orbitals.

  Log of FT_ProductPAO.c:

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
#include "openmx_core_product.h"




void FT_ProductPAO(OpenMXCoreContext *core, int device)
{
  int numprocs,myid,ID,NumSpe;
  int l,Lmax,Lspe,spe,GL1,Mul1,GL2,Mul2;
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
    printf("<FT_ProductPAO>   Fourier transform of product of PAOs\n");
  }

  RestartRead_Succeed = 0;

  /***********************************************************
   In case of Scf_RestartFromFile==1, read Spe_ProductRF_Bessel
  ***********************************************************/

  if (Scf_RestartFromFile==1){

    /***********************************************************
                        read Spe_ProductRF_Bessel
    ***********************************************************/

    sprintf(fileFT,"%s%s_rst/%s.ftProPAO",filepath,restart_filename,restart_filename);

    if ((fp = fopen(fileFT,"rb")) != NULL){

      RestartRead_Succeed = 1;

      for (spe=0; spe<SpeciesNum; spe++){
	for (GL1=0; GL1<=Spe_MaxL_Basis[spe]; GL1++){
	  for (Mul1=0; Mul1<Spe_Num_Basis[spe][GL1]; Mul1++){
	    for (GL2=0; GL2<=Spe_MaxL_Basis[spe]; GL2++){

	      if (GL1<=GL2){

		Lmax = 2*GL2;

		for (Mul2=0; Mul2<Spe_Num_Basis[spe][GL2]; Mul2++){
		  for(l=0; l<=Lmax; l++){

		    size = fread(&Spe_ProductRF_Bessel[spe][GL1][Mul1][GL2][Mul2][l][0],
       		                 sizeof(double),GL_Mesh,fp);
 
		    if (size!=GL_Mesh) RestartRead_Succeed = 0;

		  }
		}
	      }
	    }
	  }
	}
      }

      fclose(fp);
    }
    else{
      printf("Could not open a file %s in FT_ProductPAO\n",fileFT);
    }
  }

  /***********************************************************
   if (RestartRead_Succeed==0), calculate Spe_ProductRF_Bessel
  ***********************************************************/

  if (RestartRead_Succeed==0){

    for (Lspe=0; Lspe<MSpeciesNum; Lspe++){

      spe = Species_Top[myid] + Lspe;

      const OpenMXProductTable table = {Spe_MaxL_Basis[spe],Spe_Num_Mesh_PAO[spe],
        Spe_Num_Basis[spe],Spe_PAO_RV[spe],Spe_PAO_RWF[spe],Spe_ProductRF_Bessel[spe]};
      const OpenMXGaussRadialGrid grid = {GL_Mesh,GL_Mesh,Spe_VPS_RV[spe][0],
        Spe_Atom_Cut1[spe]+0.5,Radial_kmin,PAO_Nkmax,GL_Abscissae,GL_Weight,GL_Abscissae};
      char error[256];
      if (!openmx_core_transform_products(core,device,&table,&grid,error,sizeof(error))) {
        fprintf(stderr,"FT_ProductPAO species %d: %s\n",spe,error);
        MPI_Abort(mpi_comm_level1,1);
        return;
      }
    } /* Lspe */

    /***********************************************************
       sending and receiving of Spe_ProductRF_Bessel by MPI
    ***********************************************************/

    for (ID=0; ID<Num_Procs2; ID++) {
      NumSpe = Species_End[ID] - Species_Top[ID] + 1;
      for (Lspe=0; Lspe<NumSpe; Lspe++){
	spe = Species_Top[ID] + Lspe;

	for (GL1=0; GL1<=Spe_MaxL_Basis[spe]; GL1++){
	  for (Mul1=0; Mul1<Spe_Num_Basis[spe][GL1]; Mul1++){
	    for (GL2=0; GL2<=Spe_MaxL_Basis[spe]; GL2++){

	      if (GL1<=GL2){

		Lmax = 2*GL2;

		for (Mul2=0; Mul2<Spe_Num_Basis[spe][GL2]; Mul2++){
		  for(l=0; l<=Lmax; l++){
                    MPI_Barrier(mpi_comm_level1);
		    MPI_Bcast(&Spe_ProductRF_Bessel[spe][GL1][Mul1][GL2][Mul2][l][0],
			      GL_Mesh,MPI_DOUBLE,ID,mpi_comm_level1);
		  }
		}
	      }
	    }
	  }
	}
      }
    }

    /***********************************************************
                      save Spe_ProductRF_Bessel
    ***********************************************************/

    if (myid==Host_ID){

      sprintf(fileFT,"%s%s_rst/%s.ftProPAO",filepath,filename,filename);

      if ((fp = fopen(fileFT,"wb")) != NULL){

	for (spe=0; spe<SpeciesNum; spe++){
	  for (GL1=0; GL1<=Spe_MaxL_Basis[spe]; GL1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[spe][GL1]; Mul1++){
	      for (GL2=0; GL2<=Spe_MaxL_Basis[spe]; GL2++){

		if (GL1<=GL2){

		  Lmax = 2*GL2;

		  for (Mul2=0; Mul2<Spe_Num_Basis[spe][GL2]; Mul2++){
		    for(l=0; l<=Lmax; l++){
		      fwrite(&Spe_ProductRF_Bessel[spe][GL1][Mul1][GL2][Mul2][l][0],
			     sizeof(double),GL_Mesh,fp);
		    }
		  }
		}
	      }
	    }
	  }
	}

	fclose(fp);
      }
      else{
	printf("Could not open a file %s in FT_ProductPAO\n",fileFT);
      }
    }

  } /* if (RestartRead_Succeed==0) */

  /***********************************************************
                         elapsed time
  ***********************************************************/

  dtime(&TEtime);

  /*
  printf("FT_ProductPAO: myid=%2d Elapsed Time (s) = %15.12f\n",myid,TEtime-TStime);
  MPI_Finalize();
  exit(0);
  */

}
