/**********************************************************************
  openmx_api.c:

    openmx_api.c is an API routine which is called from an external program.

 Log of openmx_api.c:

    15/April/2026  Released by T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>
#include "tran_prototypes.h"
#include "tran_variables.h"


int openmx_api(MPI_Comm mpi_commWD, const char *omp_input_file, int OMP_threadN) 
{
  int i,j,k,MD_iter,argc;
  int numprocs,myid;
  char *argv[2];
  double TStime,TEtime;
  static char fileMemory[YOUSO10]; 
  char input_file_buf[YOUSO10];

  /* MPI set-up */

  mpi_comm_level1 = mpi_commWD;
  MPI_COMM_WORLD1 = mpi_commWD;
  MPI_Comm_size(MPI_COMM_WORLD1,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD1,&myid);
  NUMPROCS_MPI_COMM_WORLD = numprocs;
  MYID_MPI_COMM_WORLD = myid;
  Num_Procs = numprocs;

  /* OMP set-up */

  openmp_threads_num = OMP_threadN;
  omp_set_num_threads(openmp_threads_num);  

  /* set argc and argv */

  argc = 2;
  strcpy(input_file_buf, omp_input_file);
  argv[0] = "openmx_api";
  argv[1] = input_file_buf;

  /* for measuring elapsed time */

  dtime(&TStime);

  /*******************************************************
    check the NEB calculation or not, and if yes, go to 
    the NEB calculation.
  *******************************************************/

  if (neb_check(argv)) neb(argc,argv);

  /*******************************************************
   allocation of CompTime and show the greeting message 
  *******************************************************/

  CompTime = (double**)malloc(sizeof(double*)*numprocs); 
  for (i=0; i<numprocs; i++){
    CompTime[i] = (double*)malloc(sizeof(double)*30); 
    for (j=0; j<30; j++) CompTime[i][j] = 0.0;
  }

  if (myid==Host_ID){  
    printf("\n*******************************************************\n"); 
    printf("*******************************************************\n"); 
    printf(" Welcome to OpenMX   Ver. %s                           \n",Version_OpenMX); 
    printf(" Copyright (C), 2002-2026, T. Ozaki                    \n"); 
    printf(" OpenMX comes with ABSOLUTELY NO WARRANTY.             \n"); 
    printf(" This is free software, and you are welcome to         \n"); 
    printf(" redistribute it under the constitution of the GNU-GPL.\n");
    printf("*******************************************************\n"); 
    printf("*******************************************************\n\n"); 
  } 

  Init_List_YOUSO();
  remake_headfile = 0;
  ScaleSize = 1.2; 

  /****************************************************
                   Read the input file
  ****************************************************/

  init_alloc_first();

  CompTime[myid][1] = readfile(argv);
  MPI_Barrier(MPI_COMM_WORLD1);

  /* initialize PrintMemory routine */

  sprintf(fileMemory,"%s%s.memory%i",filepath,filename,myid);
  PrintMemory(fileMemory,0,"init"); 
  PrintMemory_Fix();

  /* initialize */

  init();

  /****************************************************
      SCF-DFT calculations, MD and geometrical
      optimization.
  ****************************************************/

  MD_iter = 1;

  do {

    /* for DFT-D2 by okuno and DFT-D3 by Ellner, and modified by Ozaki */
    if (dftD_switch==1 && version_dftD==2) DFTDvdW_init(MD_iter);   /* DFT-D2 */
    if (dftD_switch==1 && version_dftD==3) DFTD3vdW_init(MD_iter);  /* DFT-D3 */

    if (MD_switch==12)
      CompTime[myid][2] += truncation(1,1);  /* EvsLC */
    else if (MD_cellopt_flag==1)
      CompTime[myid][2] += truncation(1,1);  /* cell optimization */
    else{ 
      CompTime[myid][2] += truncation(MD_iter,1);
    }

    if (ML_flag==1 && myid==Host_ID) Get_VSZ(MD_iter);

    if (Solver==4) {
      TRAN_Calc_GridBound( mpi_comm_level1, atomnum, WhatSpecies, Spe_Atom_Cut1,
                           Ngrid1, Grid_Origin, Gxyz, tv, gtv, rgtv, Left_tv, Right_tv );

      /* output: TRAN_region[], TRAN_grid_bound */
    }

    if (Solver!=4 || TRAN_SCF_skip==0){

      CompTime[myid][3] += DFT(MD_iter,(MD_iter-1)%orbitalOpt_per_MDIter+1);
      iterout(MD_iter+MD_Current_Iter,MD_TimeStep*(MD_iter+MD_Current_Iter-1),filepath,filename);

      /* MD or geometry optimization */

      if (ML_flag==0) CompTime[myid][4] += MD_pac(MD_iter,argv[1]);
    }
    else{
      MD_Opt_OK = 1;
    }

    MD_iter++;

  } while(MD_Opt_OK==0 && (MD_iter+MD_Current_Iter)<=MD_IterNumber);

  if ( TRAN_output_hks ) {
     /* left is dummy */
     TRAN_RestartFile(mpi_comm_level1, "write","left",filepath,TRAN_hksoutfilename);
  }

  /****************************************************
               calculate Voronoi charge
  ****************************************************/
 
  if (Voronoi_Charge_flag==1) Voronoi_Charge();

  /****************************************************
        calculate Voronoi orbital magnetic moment
  ****************************************************/
 
  if (Voronoi_OrbM_flag==1) Voronoi_Orbital_Moment();

  /****************************************************
        output analysis of decomposed energies
  ****************************************************/

  if (Energy_Decomposition_flag==1) Output_Energy_Decomposition();

  /****************************************************
  making of a file *.frac for the fractional coordinates
  ****************************************************/

  Make_FracCoord(argv[1]);

  /****************************************************
   generate Wannier functions added by Hongming Weng
  ****************************************************/

  /* hmweng */
  if(Wannier_Func_Calc){
    if (myid==Host_ID) printf("Calling Generate_Wannier...\n");fflush(0);

    Generate_Wannier(argv[1]);
  }

  /****************************************************
      population analysis based on atomic orbitals 
              resembling Wannier functions
  ****************************************************/

  if(pop_anal_aow_flag){
    if (myid==Host_ID) printf("Population analysis based on atomic orbitals resembling Wannier functions\n");fflush(0);

    Population_Analysis_Wannier2(argv);
  }

  /*********************************************************
     Electronic transport calculations based on NEGF:
     transmission, current, eigen channel analysis, and
     real space analysis of current
  *********************************************************/

  if (Solver==4 && TRAN_analysis==1) {

    /* if SCF is skipped, calculate values of basis functions on each grid */ 
    if (1<=TRAN_SCF_skip) i = Set_Orbitals_Grid(0); 

    if (SpinP_switch==3) {
      TRAN_Main_Analysis_NC( mpi_comm_level1, argc, argv, Matomnum, M2G,  
                             GridN_Atom, GridListAtom, CellListAtom, 
                             Orbs_Grid, TNumGrid );
    }
    else {
      TRAN_Main_Analysis( mpi_comm_level1, argc, argv, Matomnum, M2G,  
                          GridN_Atom, GridListAtom, CellListAtom, 
                          Orbs_Grid, TNumGrid );
    }
  }

  /****************************************************
                  Making of output files
  ****************************************************/

  CompTime[myid][20] = OutData(argv[1]);

  /****************************************************
    write connectivity, Hamiltonian, overlap, density
    matrices, and etc. to a file, filename.scfout 
  ****************************************************/

  if (HS_fileout==1) SCF2File("write",argv[1]);

  /* elapsed time */

  dtime(&TEtime);
  CompTime[myid][0] = TEtime - TStime;
  Output_CompTime();
  for (i=0; i<numprocs; i++){
    free(CompTime[i]);
  }
  free(CompTime);

  /* merge log files */
  Merge_LogFile(argv[1]);

  /* free arrays for NEGF */

  if (Solver==4){

    TRAN_Deallocate_Atoms();
    TRAN_Deallocate_RestartFile("left");
    TRAN_Deallocate_RestartFile("right");
  }

  /* free arrays */

  Free_Arrays(0);

  /* print memory */

  PrintMemory("total",0,"sum");

  MPI_Barrier(MPI_COMM_WORLD1);
  if (myid==Host_ID){
    printf("\nThe calculation was normally finished.\n");fflush(stdout);
  }

  return 0;
}


int openmx_api_f(MPI_Fint fcomm, const char *omp_input_file, int OMP_threadN)
{
  MPI_Comm ccomm;
  ccomm = MPI_Comm_f2c(fcomm);
  return openmx_api(ccomm, omp_input_file, OMP_threadN);
}
