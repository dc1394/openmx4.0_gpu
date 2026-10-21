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
#include "openmx_reciprocal.h"
#include "openmx_atomic.h"
#include "mpi.h"
#include "openmx_core_radial.h"



void FT_NLP(OpenMXCoreContext *core, int device)
{
  const OpenMXAtomicViews atomic_views=OpenMX_AtomicTables(core);
  const OpenMXAtomicViews *atomic=&atomic_views;
  OpenMXReciprocalRows archive_rows;
  OpenMXReciprocalArchive *archive=OpenMX_ReciprocalArchiveBegin(core,OPENMX_ARCHIVE_NONLOCAL,&archive_rows);
  char archive_error[256];
  const OpenMXReciprocalViews reciprocal=OpenMX_ReciprocalTables(core);
  double ****Spe_NLRF_Bessel=reciprocal.vps.transforms;

  int numprocs,myid,ID,NumSpe;
  int num_k,so,Lspe,spe;
  int RestartRead_Succeed;
  double dk,rmin,rmax,h;
  double TStime,TEtime;

  char fileFT[YOUSO10];

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
    sprintf(fileFT,"%s%s_rst/%s.ftnlp",filepath,restart_filename,restart_filename);
    RestartRead_Succeed=openmx_core_reciprocal_archive_read(archive,fileFT,archive_error,sizeof(archive_error));
    if (!RestartRead_Succeed)
      fprintf(stderr,"FT_NLP: %s: %s\n",fileFT,archive_error);
    /* Every rank must take the same recompute/distribution branch. */
    MPI_Allreduce(MPI_IN_PLACE,&RestartRead_Succeed,1,MPI_INT,MPI_MIN,mpi_comm_level1);
  }

  /***********************************************************
     if (RestartRead_Succeed==0), calculate Spe_NLRF_Bessel
  ***********************************************************/

  if (RestartRead_Succeed==0){

    for (Lspe=0; Lspe<MSpeciesNum; Lspe++){

      spe = Species_Top[myid] + Lspe;

      num_k = Ngrid_NormK;
      dk = PAO_Nkmax/(double)num_k;
      rmin = openmx_radial_row2(&atomic->vps_radii,spe)[0];
      rmax = atomic->species.cutoff[spe] + 0.5;
      h = (rmax - rmin)/(double)OneD_Grid;

      {
        char error[256];
        OpenMXProjectorTable table = {0};
        const OpenMXUniformRadialGrid grid = {OneD_Grid,num_k,rmin,h,dk};
        table.projector_count = atomic->species.radial_projectors[spe];
        table.angular = openmx_atomic_projector_angular(atomic,spe)+1;
        table.spin_count = atomic->species.relativistic[spe]+1;
        table.mesh_count = atomic->species.vps_mesh[spe];
        table.radii = openmx_radial_row2(&atomic->vps_radii,spe);
        for (so=0; so<table.spin_count && so<2; so++) {
          table.values[so] = reciprocal.vps.projectors[so][spe];
          table.output[so] = Spe_NLRF_Bessel[so][spe]+1;
        }
        if (!openmx_core_transform_projectors(core,device,&table,&grid,error,sizeof(error))) {
          fprintf(stderr,"FT_NLP: %s\n",error);
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
        }
      }
    }

    /***********************************************************
                      save Spe_NLRF_Bessel
    ***********************************************************/

    if (myid==Host_ID){
      sprintf(fileFT,"%s%s_rst/%s.ftnlp",filepath,filename,filename);
      if (!openmx_core_reciprocal_archive_write(archive,fileFT,archive_error,sizeof(archive_error)))
        fprintf(stderr,"FT_NLP: %s: %s\n",fileFT,archive_error);
    }

  } /* if (RestartRead_Succeed==0) */

  /***********************************************************
                         elapsed time
  ***********************************************************/

  openmx_core_reciprocal_archive_destroy(archive);
  dtime(&TEtime);

  /*
  printf("FT_NLP: myid=%2d Elapsed Time (s) = %15.12f\n",myid,TEtime-TStime);
  MPI_Finalize();
  exit(0);
  */

}
