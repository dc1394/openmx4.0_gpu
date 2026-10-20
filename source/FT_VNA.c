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
#include "openmx_reciprocal.h"
#include "openmx_atomic.h"
#include "mpi.h"
#include "openmx_core_radial.h"





void FT_VNA(OpenMXCoreContext *core, int device)
{
  const OpenMXAtomicViews atomic_views=OpenMX_AtomicTables(core);
  const OpenMXAtomicViews *atomic=&atomic_views;
  OpenMXReciprocalRows archive_rows;
  OpenMXReciprocalArchive *archive=OpenMX_ReciprocalArchiveBegin(core,OPENMX_ARCHIVE_NEUTRAL_POTENTIAL,&archive_rows);
  char archive_error[256];
  const OpenMXReciprocalViews reciprocal=OpenMX_ReciprocalTables(core);
  double **Spe_CrudeVNA_Bessel=reciprocal.vps.potential_transform;

  int numprocs,myid,ID,NumSpe;
  int Lspe,spe;
  int RestartRead_Succeed;
  double TStime,TEtime;

  char fileFT[YOUSO10];

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
    sprintf(fileFT,"%s%s_rst/%s.ftCvna",filepath,restart_filename,restart_filename);
    RestartRead_Succeed=openmx_core_reciprocal_archive_read(archive,fileFT,archive_error,sizeof(archive_error));
    if (!RestartRead_Succeed)
      fprintf(stderr,"FT_VNA: %s: %s\n",fileFT,archive_error);
    /* Every rank must take the same recompute/distribution branch. */
    MPI_Allreduce(MPI_IN_PLACE,&RestartRead_Succeed,1,MPI_INT,MPI_MIN,mpi_comm_level1);
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
        const OpenMXNeutralTable table = {atomic->species.vps_mesh[spe],openmx_radial_row2(&atomic->vps_radii,spe),
          openmx_radial_row2(&atomic->neutral_potential,spe),atomic->species.cutoff[spe],Spe_CrudeVNA_Bessel[spe]};
        const OpenMXSquaredRadialGrid grid = {OneD_Grid,GL_Mesh,openmx_radial_row2(&atomic->pao_radii,spe)[0],
          atomic->species.cutoff[spe]+0.5,Radial_kmin,PAO_Nkmax,GL_Abscissae};
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
      NumSpe=Species_End[ID]-Species_Top[ID]+1;
      for (Lspe=0;Lspe<NumSpe;++Lspe) {
        spe=Species_Top[ID]+Lspe;
        for (size_t row=archive_rows.species_offsets[spe];row<archive_rows.species_offsets[spe+1];++row) {
          MPI_Bcast(archive_rows.row[row].values,archive_rows.row[row].count,MPI_DOUBLE,ID,mpi_comm_level1);
        }
      }
    }

    /***********************************************************
                        save Spe_CrudeVNA_Bessel
    ***********************************************************/

    if (myid==Host_ID){
      sprintf(fileFT,"%s%s_rst/%s.ftCvna",filepath,filename,filename);
      if (!openmx_core_reciprocal_archive_write(archive,fileFT,archive_error,sizeof(archive_error)))
        fprintf(stderr,"FT_VNA: %s: %s\n",fileFT,archive_error);
    }

  } /* if (RestartRead_Succeed==0) */

  /***********************************************************
                         elapsed time
  ***********************************************************/

  openmx_core_reciprocal_archive_destroy(archive);
  dtime(&TEtime);

  /*
  printf("FT_VNA: myid=%2d Elapsed Time (s) = %15.12f\n",myid,TEtime-TStime);
  MPI_Finalize();
  exit(0);
  */

}




