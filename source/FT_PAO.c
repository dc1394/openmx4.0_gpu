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
#include "openmx_reciprocal.h"
#include "openmx_atomic.h"
#include "mpi.h"
#include "openmx_core_radial.h"



void FT_PAO(OpenMXCoreContext *core, int device)
{
  const OpenMXAtomicViews atomic_views=OpenMX_AtomicTables(core);
  const OpenMXAtomicViews *atomic=&atomic_views;
  OpenMXReciprocalRows archive_rows;
  OpenMXReciprocalArchive *archive=OpenMX_ReciprocalArchiveBegin(core,OPENMX_ARCHIVE_PAO,&archive_rows);
  char archive_error[256];
  const OpenMXReciprocalViews reciprocal=OpenMX_ReciprocalTables(core);
  double ****Spe_RF_Bessel=reciprocal.pao.transforms;

  int numprocs,myid,ID,NumSpe;
  int num_k,Lspe,spe;
  int RestartRead_Succeed;
  double dk,h,rmin,rmax;
  double TStime, TEtime;

  char fileFT[YOUSO10];

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
    sprintf(fileFT,"%s%s_rst/%s.ftpao",filepath,restart_filename,restart_filename);
    RestartRead_Succeed=openmx_core_reciprocal_archive_read(archive,fileFT,archive_error,sizeof(archive_error));
    if (!RestartRead_Succeed)
      fprintf(stderr,"FT_PAO: %s: %s\n",fileFT,archive_error);
    /* Every rank must take the same recompute/distribution branch. */
    MPI_Allreduce(MPI_IN_PLACE,&RestartRead_Succeed,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  }

  /***********************************************************
     if (RestartRead_Succeed==0), calculate Spe_RF_Bessel
  ***********************************************************/

  if (RestartRead_Succeed==0){

    for (Lspe=0; Lspe<MSpeciesNum; Lspe++){

      spe = Species_Top[myid] + Lspe;

      num_k = Ngrid_NormK;
      dk = PAO_Nkmax/(double)num_k;
      rmin = openmx_radial_row2(&atomic->pao_radii,spe)[0];
      rmax = atomic->species.cutoff[spe] + 0.5;
      h = (rmax - rmin)/(double)OneD_Grid;

      {
        char error[256];
        const OpenMXPaoTable table = {atomic->species.basis_lmax[spe],atomic->species.basis_rows[spe],
          atomic->species.pao_mesh[spe],openmx_radial_row2(&atomic->pao_radii,spe),reciprocal.pao.orbitals[spe],Spe_RF_Bessel[spe]};
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
      NumSpe=Species_End[ID]-Species_Top[ID]+1;
      for (Lspe=0;Lspe<NumSpe;++Lspe) {
        spe=Species_Top[ID]+Lspe;
        for (size_t row=archive_rows.species_offsets[spe];row<archive_rows.species_offsets[spe+1];++row) {
          MPI_Bcast(archive_rows.row[row].values,archive_rows.row[row].count,MPI_DOUBLE,ID,mpi_comm_level1);
          MPI_Barrier(mpi_comm_level1);
        }
      }
    }

    /***********************************************************
                      save Spe_RF_Bessel
    ***********************************************************/

    if (myid==Host_ID){
      char operate[300];
      sprintf(operate,"%s%s_rst",filepath,filename);
      mkdir(operate,0775);
      sprintf(fileFT,"%s%s_rst/%s.ftpao",filepath,filename,filename);
      if (!openmx_core_reciprocal_archive_write(archive,fileFT,archive_error,sizeof(archive_error)))
        fprintf(stderr,"FT_PAO: %s: %s\n",fileFT,archive_error);
    }

  } /* if (RestartRead_Succeed==0) */

  /***********************************************************
                         elapsed time
  ***********************************************************/

  openmx_core_reciprocal_archive_destroy(archive);
  dtime(&TEtime);

  /*
  printf("FT_PAO.c: Time=%15.12f\n",TEtime-TStime);
  MPI_Finalize();
  exit(0);
  */

}



