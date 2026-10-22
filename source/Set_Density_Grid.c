/**********************************************************************
  Set_Density_Grid.c:

     Set_Density_Grid.c is a subroutine to calculate a charge density 
     on grid by one-particle wave functions.

  Log of Set_Density_Grid.c:

     22/Nov/2001  Released by T. Ozaki
     19/Apr/2013  Modified by A.M. Ito     

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "openmx_common.h"
#include "openmx_env.h"
#include "openmx_grid_assembly.h"
#include "mpi.h"
#include <omp.h>

#define  measure_time   0

int Set_Density_Grid_GPU_Capacity(int,int,double *****,double ***);


double Set_Density_Grid(OpenMXGridWorkspace *grid, const OpenMXAtomicViews *atomic, int Cnt_kind, int Calc_CntOrbital_ON, double *****CDM, double **Density_Grid_B0, const OpenMXLocalBasisViews *local_basis)
{

  static int firsttime=1;
  int al,L0,Mul0,M0,p,size1,size2;
  int Gc_AN,Mc_AN,LN,AN,BN;
  int N3[4];
  int Cwan,NO0,i,j,k,n;
  int NN_S,NN_R;
  unsigned long long int N2D,n2D,GN; 
  int Max_Size,My_Max;
  size_t size_Tmp_Den_Grid;
  size_t size_Den_Snd_Grid_A2B;
  size_t size_Den_Rcv_Grid_A2B;
  int spin,Nc,GRc;

  double tmp0,sum;

  double TStime,TEtime;
  double ***Tmp_Den_Grid;
  double **Den_Snd_Grid_A2B;
  double **Den_Rcv_Grid_A2B;
  double *tmp_array;
  double *tmp_array2;

  int *Snd_Size,*Rcv_Size;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Stime_atom, Etime_atom;
  double time0,time1,time2;
  int use_local_gpu = 0;
  int local_ok = 0, local_any = 0, local_path = 0, local_failed = 0, failed_ranks = 0;
  MPI_Request failed_request = MPI_REQUEST_NULL;

  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  /* for OpenMP */
  int OMPID,Nthrds;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  
  dtime(&TStime);

  /* stage profile: OPENMX_BAND_PROFILE=1 prints one SETDENPROF line per rank
     and call (pre = GPU-mode decision + allocation, quad = the atom-sphere
     quadrature, a2b = the atom-to-B communication, sup = the superposition
     into the B partition, b2d = the B-to-D copy and the rest) */
  double sd_t0, sd_t1, sd_pre = 0.0, sd_quad = 0.0, sd_a2b = 0.0, sd_sup = 0.0, sd_b2d = 0.0;
  const int sd_prof = openmx_env_flag_at(OPENMX_ENV_BAND_PROFILE, 0);
  sd_t0 = TStime;

  /* Preferred GPU mode: every rank evaluates its own atoms' density on the
     device (sharing Set_Hamiltonian's tables) and the ordinary A-to-B
     communication below distributes it.  Memory pressure keeps this
     distributed path, with the CPU/OpenMP loop on ranks without workspace.
     The single-owner service handles calls without a distributed path. */
  {
    int flags[2], reduced[2];

    local_ok = Set_Density_Grid_GPU_Local_Prepare(atomic, Cnt_kind,Calc_CntOrbital_ON);
    /* every rank capable (the minimum of ok) and any rank capable (that of -ok) */
    flags[0] = local_ok;
    flags[1] = -local_ok;
    MPI_Allreduce(flags,reduced,2,MPI_INT,MPI_MIN,mpi_comm_level1);
    use_local_gpu = reduced[0];
    local_any = -reduced[1];

    if (use_local_gpu && myid==Host_ID && 2<=level_stdout){
      static int announced = 0;
      if (!announced){
        printf("<Set_Density_Grid> distributed GPU quadrature: every rank integrates its own atoms\n");
        fflush(stdout);
        announced = 1;
      }
    }
  }

  /* A single GPU-owner rank can construct the complete B partition and
     bypass the per-rank atom-density/A-to-B path below.  A return value of
     zero keeps the original CPU/OpenMP implementation as the fallback. */
  if (!use_local_gpu){
    if (!Set_Density_Grid_GPU_Local_KeepDistributed() &&
        Set_Density_Grid_GPU_Service(Cnt_kind,Calc_CntOrbital_ON,CDM,Density_Grid_B0,&time0)){
      Set_Density_Grid_GPU_Local_Report(1,0);
      return time0;
    }
    /* Ranks that can integrate their own atoms on the device still do;
       the others take the parallel CPU loop. */
    use_local_gpu = local_ok;
    local_path = 2;
  }

  /* allocation of arrays */

  OpenMXGridDensity density_storage=OpenMX_Grid_Density_Storage(grid);
  Tmp_Den_Grid=density_storage.atoms;
  Den_Snd_Grid_A2B=density_storage.send;
  Den_Rcv_Grid_A2B=density_storage.receive;
  size_Tmp_Den_Grid=density_storage.atom_values;
  size_Den_Snd_Grid_A2B=density_storage.send_values;
  size_Den_Rcv_Grid_A2B=density_storage.receive_values;

  /* PrintMemory */

  if (firsttime==1){
    PrintMemory("Set_Density_Grid: AtomDen_Grid",    sizeof(double)*size_Tmp_Den_Grid, NULL);
    PrintMemory("Set_Density_Grid: Den_Snd_Grid_A2B",sizeof(double)*size_Den_Snd_Grid_A2B, NULL);
    PrintMemory("Set_Density_Grid: Den_Rcv_Grid_A2B",sizeof(double)*size_Den_Rcv_Grid_A2B, NULL);
    firsttime = 0;
  }

  /****************************************************
                when orbital optimization
  ****************************************************/

  if (Calc_CntOrbital_ON==1 && Cnt_kind==0 && Cnt_switch==1){
      
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
       
      dtime(&Stime_atom);
      
      /* COrbs_Grid */
 
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      NO0 = atomic->species.contracted_orbitals[Cwan];
      for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

        al = -1;
	for (L0=0; L0<=atomic->species.basis_lmax[Cwan]; L0++){
	  for (Mul0=0; Mul0<atomic->species.contracted_basis[(size_t)(Cwan)*atomic->species.angular_channels+(L0)]; Mul0++){
	    for (M0=0; M0<=2*L0; M0++){

	      al++;
	      tmp0 = 0.0;

	      for (p=0; p<Spe_Specified_Num[Cwan][al]; p++){
	        j = Spe_Trans_Orbital[Cwan][al][p];  
	        tmp0 += openmx_local_contraction_row(local_basis, Mc_AN, al)[p]*Orbs_Grid[Mc_AN][Nc][j];/* AITUNE */
	      }

	      COrbs_Grid[Mc_AN][al][Nc] = (Type_Orbs_Grid)tmp0;
	    }
	  }
        }
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

    /**********************************************
     MPI:

     COrbs_Grid    
    ***********************************************/

    /* allocation of arrays  */
    Snd_Size = (int*)malloc(sizeof(int)*numprocs); 
    Rcv_Size = (int*)malloc(sizeof(int)*numprocs); 

    /* find data size for sending and receiving */

    My_Max = -10000;
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID!=0){
        /*  sending size */
        if (F_Snd_Num[IDS]!=0){
          /* find data size */ 
          size1 = 0; 
          for (n=0; n<F_Snd_Num[IDS]; n++){
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN];
            size1 += GridN_Atom[Gc_AN]*atomic->species.contracted_orbitals[Cwan];
          }

          Snd_Size[IDS] = size1;
          MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
        }
        else{
          Snd_Size[IDS] = 0;
        }

        /* receiving size */
        if (F_Rcv_Num[IDR]!=0){
          MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
          Rcv_Size[IDR] = size2;
        }
        else{
          Rcv_Size[IDR] = 0;
        }
        if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);
      } 
      else{
        Snd_Size[IDS] = 0;
        Rcv_Size[IDR] = 0;
      }

      if (My_Max<Snd_Size[IDS]) My_Max = Snd_Size[IDS];
      if (My_Max<Rcv_Size[IDR]) My_Max = Rcv_Size[IDR];

    }  

    MPI_Allreduce(&My_Max, &Max_Size, 1, MPI_INT, MPI_MAX, mpi_comm_level1);
    /* allocation of arrays */ 
    tmp_array  = (double*)malloc(sizeof(double)*Max_Size);
    tmp_array2 = (double*)malloc(sizeof(double)*Max_Size);

    /* send and receive COrbs_Grid */

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID!=0){

        /* sending of data */ 

        if (F_Snd_Num[IDS]!=0){

          /* find data size */
          size1 = Snd_Size[IDS];

          /* multidimentional array to vector array */
          k = 0; 
          for (n=0; n<F_Snd_Num[IDS]; n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN];
            NO0 = atomic->species.contracted_orbitals[Cwan];
            for (i=0; i<NO0; i++){
              for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
                tmp_array[k] = COrbs_Grid[Mc_AN][i][Nc];
                k++;
              }          
            }
          } 

          /* MPI_Isend */
          MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS,
                    tag, mpi_comm_level1, &request);
        }

        /* receiving of block data */

        if (F_Rcv_Num[IDR]!=0){

          /* find data size */
          size2 = Rcv_Size[IDR]; 

          /* MPI_Recv */
          MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

          k = 0;
          Mc_AN = F_TopMAN[IDR] - 1;
          for (n=0; n<F_Rcv_Num[IDR]; n++){
            Mc_AN++;
            Gc_AN = Rcv_GAN[IDR][n];
            Cwan = WhatSpecies[Gc_AN];
            NO0 = atomic->species.contracted_orbitals[Cwan];

            for (i=0; i<NO0; i++){
              for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
                COrbs_Grid[Mc_AN][i][Nc] = tmp_array2[k];
                k++;
              }          
            }
          }
        }
        if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);
      } 
    }  

    /* freeing of arrays  */
    free(tmp_array);
    free(tmp_array2);
    free(Snd_Size);
    free(Rcv_Size);
  }

  /**********************************************
              calculate Tmp_Den_Grid
  ***********************************************/

  dtime(&time1);
  sd_pre = time1 - sd_t0; sd_t0 = time1;

  /* the distributed GPU quadrature fills Tmp_Den_Grid directly; a run-time
     failure on any rank simply falls back to the CPU loop on that rank */
  if (use_local_gpu && !Set_Density_Grid_GPU_Local_Run(atomic, CDM,Tmp_Den_Grid)){
    use_local_gpu = 0;
    local_failed = 1;
  }

  /* All ranks enter the recovery phase. Completed ranks can lend their
     node's permitted GPUs; declined ranks publish only a complete shard. */
  if (Set_Density_Grid_GPU_Capacity(Cnt_kind,use_local_gpu,CDM,Tmp_Den_Grid)) {
    use_local_gpu=1;
    local_failed=0;
  }

  if (!use_local_gpu){
    OpenMX_Grid_Density_CPU(grid,atomic,Cnt_kind,CDM);
  } /* !use_local_gpu */

  dtime(&time2);
  if(myid==0 && measure_time){
    printf("Time for Part1=%18.5f\n",(time2-time1));fflush(stdout);
  }
  sd_quad = time2 - sd_t0; sd_t0 = time2;

  /* the ranks with atoms whose GPU quadrature failed, for the level-2
     line at the end: a non-blocking sum, which the communication below
     completes without holding any rank back */
  if (local_any){
    local_failed = (0<Matomnum && local_failed);
    MPI_Iallreduce(&local_failed,&failed_ranks,1,MPI_INT,MPI_SUM,mpi_comm_level1,&failed_request);
  }

  /******************************************************
      MPI communication from the partitions A to B 
  ******************************************************/
  
  /* copy Tmp_Den_Grid to Den_Snd_Grid_A2B */

  for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_A2B[ID] = 0;
  
  N2D = Ngrid1*Ngrid2;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];

    for (AN=0; AN<GridN_Atom[Gc_AN]; AN++){

      GN = GridListAtom[Mc_AN][AN];
      GN2N(GN,N3);
      n2D = N3[1]*Ngrid2 + N3[2];
      ID = (int)(n2D*(unsigned long long int)numprocs/N2D);

      if (SpinP_switch==0){
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]] = Tmp_Den_Grid[0][Mc_AN][AN];
      }
      else if (SpinP_switch==1){
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*2+0] = Tmp_Den_Grid[0][Mc_AN][AN];
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*2+1] = Tmp_Den_Grid[1][Mc_AN][AN];
      }
      else if (SpinP_switch==3){
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*4+0] = Tmp_Den_Grid[0][Mc_AN][AN];
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*4+1] = Tmp_Den_Grid[1][Mc_AN][AN];
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*4+2] = Tmp_Den_Grid[2][Mc_AN][AN];
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*4+3] = Tmp_Den_Grid[3][Mc_AN][AN];
      }

      Num_Snd_Grid_A2B[ID]++;
    }
  }    

  /* MPI: A to B */  

  OpenMXGridExchange a2b=OpenMX_Grid_Exchange(grid,OPENMX_GRID_A2B,0,0,
      (size_t)numprocs,(size_t)numprocs);
  request_send=a2b.send_requests;
  request_recv=a2b.receive_requests;
  stat_send=a2b.send_status;
  stat_recv=a2b.receive_status;

  NN_S = 0;
  NN_R = 0;

  tag = 999;
  for (ID=1; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (Num_Snd_Grid_A2B[IDS]!=0){
      MPI_Isend( &Den_Snd_Grid_A2B[IDS][0], Num_Snd_Grid_A2B[IDS]*(SpinP_switch+1), 
	         MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }

    if (Num_Rcv_Grid_A2B[IDR]!=0){
      MPI_Irecv( &Den_Rcv_Grid_A2B[IDR][0], Num_Rcv_Grid_A2B[IDR]*(SpinP_switch+1), 
  	         MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }
  }

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  /* Requests are complete; their storage is reused next iteration. */

  /* for myid */
  for (i=0; i<Num_Rcv_Grid_A2B[myid]*(SpinP_switch+1); i++){
    Den_Rcv_Grid_A2B[myid][i] = Den_Snd_Grid_A2B[myid][i];
  }

  dtime(&sd_t1); sd_a2b = sd_t1 - sd_t0; sd_t0 = sd_t1;
  /******************************************************
   superposition of rho_i to calculate charge density 
   in the partition B.
  ******************************************************/

  /* initialize arrays */
  
  for (spin=0; spin<(SpinP_switch+1); spin++){
    for (BN=0; BN<My_NumGridB_AB; BN++){
      Density_Grid_B0[spin][BN] = 0.0;
    }
  }
  
  /* superposition of densities rho_i */

  for (ID=0; ID<numprocs; ID++){

    for (LN=0; LN<Num_Rcv_Grid_A2B[ID]; LN++){

      BN    = Index_Rcv_Grid_A2B[ID][3*LN+0];      
      Gc_AN = Index_Rcv_Grid_A2B[ID][3*LN+1];        
      GRc   = Index_Rcv_Grid_A2B[ID][3*LN+2]; 

      if (Solver!=4 || (Solver==4 && atv_ijk[GRc][1]==0 )){

	/* spin collinear non-polarization */
	if ( SpinP_switch==0 ){
	  Density_Grid_B0[0][BN] += Den_Rcv_Grid_A2B[ID][LN];
	}

	/* spin collinear polarization */
	else if ( SpinP_switch==1 ){
	  Density_Grid_B0[0][BN] += Den_Rcv_Grid_A2B[ID][LN*2  ];
	  Density_Grid_B0[1][BN] += Den_Rcv_Grid_A2B[ID][LN*2+1];
	} 

	/* spin non-collinear */
	else if ( SpinP_switch==3 ){
	  Density_Grid_B0[0][BN] += Den_Rcv_Grid_A2B[ID][LN*4  ];
	  Density_Grid_B0[1][BN] += Den_Rcv_Grid_A2B[ID][LN*4+1];
	  Density_Grid_B0[2][BN] += Den_Rcv_Grid_A2B[ID][LN*4+2];
	  Density_Grid_B0[3][BN] += Den_Rcv_Grid_A2B[ID][LN*4+3];
	} 

      } /* if (Solve!=4.....) */           

    } /* AN */ 
  } /* ID */  

  /****************************************************
   Conjugate complex of Density_Grid[3][MN] due to
   difference in the definition between density matrix
   and charge density
  ****************************************************/

  if (SpinP_switch==3){

    for (BN=0; BN<My_NumGridB_AB; BN++){
      Density_Grid_B0[3][BN] = -Density_Grid_B0[3][BN]; 
    }
  }

  dtime(&sd_t1); sd_sup = sd_t1 - sd_t0; sd_t0 = sd_t1;
  /******************************************************
             MPI: from the partitions B to D
  ******************************************************/

  Density_Grid_Copy_B2D(Density_Grid_B0);

  /* a level-2 line when ranks with atoms took the CPU loop beside the GPU
     quadrature */
  if (local_any){
    MPI_Wait(&failed_request,MPI_STATUS_IGNORE);
  }
  Set_Density_Grid_GPU_Local_Report(local_path,failed_ranks);

  /* Density and packet storage are owned by this calculation. */

  /* elapsed time */
  dtime(&TEtime);
  time0 = TEtime - TStime;
  sd_b2d = TEtime - sd_t0;
  if (sd_prof) {
    fprintf(stderr, "SETDENPROF id=%d mode=%s pre=%.3f quad=%.3f a2b=%.3f sup=%.3f b2d=%.3f total=%.3f\n", myid,
            use_local_gpu ? "gpu-local" : "cpu", sd_pre, sd_quad, sd_a2b, sd_sup, sd_b2d, time0);
  }
  if(myid==0 && measure_time) printf("time0=%18.5f\n",time0);

  return time0;
}



void Data_Grid_Copy_B2C_2(double **data_B, double **data_C)
{
  static int firsttime=1;
  int CN,BN,LN,spin,i,gp,NN_S,NN_R;
  double *Work_Array_Snd_Grid_B2C;
  double *Work_Array_Rcv_Grid_B2C;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */
  /* the work arrays and the request/status arrays persist across calls
     (grow-only): this copy runs every SCF step and the allocation and the
     first touch of the tens of MiB were a measurable part of its time */
  {
    static double *snd_keep = NULL, *rcv_keep = NULL;
    static size_t snd_cap = 0, rcv_cap = 0;
    const size_t snd_need = (size_t)GP_B2C_S[NN_B2C_S]*(size_t)(SpinP_switch+1);
    const size_t rcv_need = (size_t)GP_B2C_R[NN_B2C_R]*(size_t)(SpinP_switch+1);
    if (snd_cap < snd_need) { free(snd_keep); snd_keep = (double*)malloc(sizeof(double)*(snd_need ? snd_need : 1)); snd_cap = snd_need; }
    if (rcv_cap < rcv_need) { free(rcv_keep); rcv_keep = (double*)malloc(sizeof(double)*(rcv_need ? rcv_need : 1)); rcv_cap = rcv_need; }
    Work_Array_Snd_Grid_B2C = snd_keep;
    Work_Array_Rcv_Grid_B2C = rcv_keep;
  }
  if (firsttime==1){
    PrintMemory("Data_Grid_Copy_B2C_2: Work_Array_Snd_Grid_B2C",
		sizeof(double)*GP_B2C_S[NN_B2C_S]*(SpinP_switch+1), NULL);
    PrintMemory("Data_Grid_Copy_B2C_2: Work_Array_Rcv_Grid_B2C",
		sizeof(double)*GP_B2C_R[NN_B2C_R]*(SpinP_switch+1), NULL);
    firsttime = 0;
  }

  /******************************************************
             MPI: from the partitions B to C
  ******************************************************/

  request_send = malloc(sizeof(MPI_Request)*NN_B2C_S);
  request_recv = malloc(sizeof(MPI_Request)*NN_B2C_R);
  stat_send = malloc(sizeof(MPI_Status)*NN_B2C_S);
  stat_recv = malloc(sizeof(MPI_Status)*NN_B2C_R);

  NN_S = 0;
  NN_R = 0;

  /* MPI_Irecv */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];
    gp = GP_B2C_R[ID];

    if (IDR!=myid){ 
      MPI_Irecv( &Work_Array_Rcv_Grid_B2C[(SpinP_switch+1)*gp], Num_Rcv_Grid_B2C[IDR]*(SpinP_switch+1),
                 MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }

  }

  /* MPI_Isend */

  for (ID=0; ID<NN_B2C_S; ID++){

    IDS = ID_NN_B2C_S[ID];
    gp = GP_B2C_S[ID];

    /* copy Density_Grid_B to Work_Array_Snd_Grid_B2C */

    for (LN=0; LN<Num_Snd_Grid_B2C[IDS]; LN++){
      BN = Index_Snd_Grid_B2C[IDS][LN];

      if (SpinP_switch==0){
        Work_Array_Snd_Grid_B2C[gp+LN]       = data_B[0][BN];
      }
      else if (SpinP_switch==1){
        Work_Array_Snd_Grid_B2C[2*gp+2*LN+0] = data_B[0][BN];
        Work_Array_Snd_Grid_B2C[2*gp+2*LN+1] = data_B[1][BN];
      }
      else if (SpinP_switch==3){
        Work_Array_Snd_Grid_B2C[4*gp+4*LN+0] = data_B[0][BN];
        Work_Array_Snd_Grid_B2C[4*gp+4*LN+1] = data_B[1][BN];
        Work_Array_Snd_Grid_B2C[4*gp+4*LN+2] = data_B[2][BN];
        Work_Array_Snd_Grid_B2C[4*gp+4*LN+3] = data_B[3][BN];
      }
    } /* LN */        

    if (IDS!=myid){
      MPI_Isend( &Work_Array_Snd_Grid_B2C[(SpinP_switch+1)*gp], Num_Snd_Grid_B2C[IDS]*(SpinP_switch+1), 
		 MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }
  }

  /* MPI_Waitall */

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /* copy Work_Array_Rcv_Grid_B2C to data_C */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];

    if (IDR==myid){

      gp = GP_B2C_S[ID];

      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){

	CN = Index_Rcv_Grid_B2C[IDR][LN];

	if (SpinP_switch==0){
	  data_C[0][CN] = Work_Array_Snd_Grid_B2C[gp+LN];
	}     
	else if (SpinP_switch==1){
	  data_C[0][CN] = Work_Array_Snd_Grid_B2C[2*gp+2*LN+0];
	  data_C[1][CN] = Work_Array_Snd_Grid_B2C[2*gp+2*LN+1];
	}     
	else if (SpinP_switch==3){
	  data_C[0][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+0];
	  data_C[1][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+1];
	  data_C[2][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+2];
	  data_C[3][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+3];
	}
      } /* LN */   

    }
    else {

      gp = GP_B2C_R[ID];

      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){
	CN = Index_Rcv_Grid_B2C[IDR][LN];

	if (SpinP_switch==0){
	  data_C[0][CN] = Work_Array_Rcv_Grid_B2C[gp+LN];
	}
	else if (SpinP_switch==1){
	  data_C[0][CN] = Work_Array_Rcv_Grid_B2C[2*gp+2*LN+0];
	  data_C[1][CN] = Work_Array_Rcv_Grid_B2C[2*gp+2*LN+1];
	}     
	else if (SpinP_switch==3){
	  data_C[0][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+0];
	  data_C[1][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+1];
	  data_C[2][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+2];
	  data_C[3][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+3];
	}
      }
    }
  }

  /* if (SpinP_switch==0), 
     copy data_B[0] to data_B[1]
     copy data_C[0] to data_C[1]
  */

  if (SpinP_switch==0){
    for (BN=0; BN<My_NumGridB_AB; BN++){
      data_B[1][BN] = data_B[0][BN]; 
    }

    for (CN=0; CN<My_NumGridC; CN++){
      data_C[1][CN] = data_C[0][CN]; 
    }
  }

  /* freeing of arrays */
  /* Work_Array_Snd_Grid_B2C persists */
  /* Work_Array_Rcv_Grid_B2C persists */
}



void Data_Grid_Copy_B2C_1(double *data_B, double *data_C)
{
  static int firsttime=1;
  int CN,BN,LN,spin,i,gp,NN_S,NN_R;
  double *Work_Array_Snd_Grid_B2C;
  double *Work_Array_Rcv_Grid_B2C;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */
  
  Work_Array_Snd_Grid_B2C = (double*)malloc(sizeof(double)*GP_B2C_S[NN_B2C_S]); 
  Work_Array_Rcv_Grid_B2C = (double*)malloc(sizeof(double)*GP_B2C_R[NN_B2C_R]); 

  if (firsttime==1){
    PrintMemory("Data_Grid_Copy_B2C_1: Work_Array_Snd_Grid_B2C",
		sizeof(double)*GP_B2C_S[NN_B2C_S], NULL);
    PrintMemory("Data_Grid_Copy_B2C_1: Work_Array_Rcv_Grid_B2C",
		sizeof(double)*GP_B2C_R[NN_B2C_R], NULL);
    firsttime = 0;
  }

  /******************************************************
             MPI: from the partitions B to C
  ******************************************************/

  request_send = malloc(sizeof(MPI_Request)*NN_B2C_S);
  request_recv = malloc(sizeof(MPI_Request)*NN_B2C_R);
  stat_send = malloc(sizeof(MPI_Status)*NN_B2C_S);
  stat_recv = malloc(sizeof(MPI_Status)*NN_B2C_R);

  NN_S = 0;
  NN_R = 0;

  /* MPI_Irecv */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];
    gp = GP_B2C_R[ID];

    if (IDR!=myid){ 
      MPI_Irecv( &Work_Array_Rcv_Grid_B2C[gp], Num_Rcv_Grid_B2C[IDR],
                 MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }
  }
 
  /* MPI_Isend */

  for (ID=0; ID<NN_B2C_S; ID++){

    IDS = ID_NN_B2C_S[ID];
    gp = GP_B2C_S[ID];

    /* copy Density_Grid_B to Work_Array_Snd_Grid_B2C */

    for (LN=0; LN<Num_Snd_Grid_B2C[IDS]; LN++){
      BN = Index_Snd_Grid_B2C[IDS][LN];
      Work_Array_Snd_Grid_B2C[gp+LN] = data_B[BN];
    } 

    if (IDS!=myid){
      MPI_Isend( &Work_Array_Snd_Grid_B2C[gp], Num_Snd_Grid_B2C[IDS], 
		 MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }
  }

  /* MPI_Waitall */

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /* copy Work_Array_Rcv_Grid_B2C to data_C */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];

    if (IDR==myid){
      gp = GP_B2C_S[ID];
      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){
	CN = Index_Rcv_Grid_B2C[IDR][LN];
	data_C[CN] = Work_Array_Snd_Grid_B2C[gp+LN];
      } 
    }
    else{

      gp = GP_B2C_R[ID];
      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){
	CN = Index_Rcv_Grid_B2C[IDR][LN];
	data_C[CN] = Work_Array_Rcv_Grid_B2C[gp+LN];
      }
    }
  }

  /* freeing of arrays */
  free(Work_Array_Snd_Grid_B2C);
  free(Work_Array_Rcv_Grid_B2C);
}





void Density_Grid_Copy_B2D(double **Density_Grid_B0)
{
  static int firsttime=1;
  int DN,BN,LN,spin,i,gp,NN_S,NN_R;
  double *Work_Array_Snd_Grid_B2D;
  double *Work_Array_Rcv_Grid_B2D;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */
  
  Work_Array_Snd_Grid_B2D = (double*)malloc(sizeof(double)*GP_B2D_S[NN_B2D_S]*(SpinP_switch+1)); 
  Work_Array_Rcv_Grid_B2D = (double*)malloc(sizeof(double)*GP_B2D_R[NN_B2D_R]*(SpinP_switch+1)); 

  if (firsttime==1){
    PrintMemory("Set_Density_Grid: Work_Array_Snd_Grid_B2D",
		sizeof(double)*GP_B2D_S[NN_B2D_S]*(SpinP_switch+1), NULL);
    PrintMemory("Set_Density_Grid: Work_Array_Rcv_Grid_B2D",
		sizeof(double)*GP_B2D_R[NN_B2D_R]*(SpinP_switch+1), NULL);
    firsttime = 0;
  }

  /******************************************************
             MPI: from the partitions B to D
  ******************************************************/

  request_send = malloc(sizeof(MPI_Request)*NN_B2D_S);
  request_recv = malloc(sizeof(MPI_Request)*NN_B2D_R);
  stat_send = malloc(sizeof(MPI_Status)*NN_B2D_S);
  stat_recv = malloc(sizeof(MPI_Status)*NN_B2D_R);

  NN_S = 0;
  NN_R = 0;

  /* MPI_Irecv */

  for (ID=0; ID<NN_B2D_R; ID++){

    IDR = ID_NN_B2D_R[ID];
    gp = GP_B2D_R[ID];

    if (IDR!=myid){ 
      MPI_Irecv( &Work_Array_Rcv_Grid_B2D[(SpinP_switch+1)*gp], Num_Rcv_Grid_B2D[IDR]*(SpinP_switch+1),
                 MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }
  }

  /* MPI_Isend */

  for (ID=0; ID<NN_B2D_S; ID++){

    IDS = ID_NN_B2D_S[ID];
    gp = GP_B2D_S[ID];

    /* copy Density_Grid_B0 to Work_Array_Snd_Grid_B2D */

    for (LN=0; LN<Num_Snd_Grid_B2D[IDS]; LN++){

      BN = Index_Snd_Grid_B2D[IDS][LN];

      if (SpinP_switch==0){
        Work_Array_Snd_Grid_B2D[gp+LN]       = Density_Grid_B0[0][BN];
      }
      else if (SpinP_switch==1){
        Work_Array_Snd_Grid_B2D[2*gp+2*LN+0] = Density_Grid_B0[0][BN];
        Work_Array_Snd_Grid_B2D[2*gp+2*LN+1] = Density_Grid_B0[1][BN];
      }
      else if (SpinP_switch==3){
        Work_Array_Snd_Grid_B2D[4*gp+4*LN+0] = Density_Grid_B0[0][BN];
        Work_Array_Snd_Grid_B2D[4*gp+4*LN+1] = Density_Grid_B0[1][BN];
        Work_Array_Snd_Grid_B2D[4*gp+4*LN+2] = Density_Grid_B0[2][BN];
        Work_Array_Snd_Grid_B2D[4*gp+4*LN+3] = Density_Grid_B0[3][BN];
      }
    } /* LN */        

    if (IDS!=myid){
      MPI_Isend( &Work_Array_Snd_Grid_B2D[(SpinP_switch+1)*gp], Num_Snd_Grid_B2D[IDS]*(SpinP_switch+1), 
		 MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }
  }

  /* MPI_Waitall */

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /* copy Work_Array_Rcv_Grid_B2D to Density_Grid_D */

  for (ID=0; ID<NN_B2D_R; ID++){

    IDR = ID_NN_B2D_R[ID];

    if (IDR==myid){

      gp = GP_B2D_S[ID];

      for (LN=0; LN<Num_Rcv_Grid_B2D[IDR]; LN++){

	DN = Index_Rcv_Grid_B2D[IDR][LN];

	if (SpinP_switch==0){
	  Density_Grid_D[0][DN] = Work_Array_Snd_Grid_B2D[gp+LN];
	}     
	else if (SpinP_switch==1){
	  Density_Grid_D[0][DN] = Work_Array_Snd_Grid_B2D[2*gp+2*LN+0];
	  Density_Grid_D[1][DN] = Work_Array_Snd_Grid_B2D[2*gp+2*LN+1];
	}     
	else if (SpinP_switch==3){
	  Density_Grid_D[0][DN] = Work_Array_Snd_Grid_B2D[4*gp+4*LN+0];
	  Density_Grid_D[1][DN] = Work_Array_Snd_Grid_B2D[4*gp+4*LN+1];
	  Density_Grid_D[2][DN] = Work_Array_Snd_Grid_B2D[4*gp+4*LN+2];
	  Density_Grid_D[3][DN] = Work_Array_Snd_Grid_B2D[4*gp+4*LN+3];
	}
      } /* LN */   

    }

    else{

      gp = GP_B2D_R[ID];

      for (LN=0; LN<Num_Rcv_Grid_B2D[IDR]; LN++){

	DN = Index_Rcv_Grid_B2D[IDR][LN];

	if (SpinP_switch==0){
	  Density_Grid_D[0][DN] = Work_Array_Rcv_Grid_B2D[gp+LN];
	}     
	else if (SpinP_switch==1){
	  Density_Grid_D[0][DN] = Work_Array_Rcv_Grid_B2D[2*gp+2*LN+0];
	  Density_Grid_D[1][DN] = Work_Array_Rcv_Grid_B2D[2*gp+2*LN+1];
	}     
	else if (SpinP_switch==3){
	  Density_Grid_D[0][DN] = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+0];
	  Density_Grid_D[1][DN] = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+1];
	  Density_Grid_D[2][DN] = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+2];
	  Density_Grid_D[3][DN] = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+3];
	}
      }

    }
  }

  /* if (SpinP_switch==0), copy Density_Grid[0] to Density_Grid[1] */

  if (SpinP_switch==0){
    for (BN=0; BN<My_NumGridB_AB; BN++){
      Density_Grid_B0[1][BN] = Density_Grid_B0[0][BN]; 
    }

    for (DN=0; DN<My_NumGridD; DN++){
      Density_Grid_D[1][DN] = Density_Grid_D[0][DN]; 
    }
  }

  /* freeing of arrays */
  free(Work_Array_Snd_Grid_B2D);
  free(Work_Array_Rcv_Grid_B2D);
}


void diagonalize_nc_density(double **Density_Grid_B0)
{
  int BN,DN,Mc_AN,Gc_AN,Nog,GRc;
  double Re11,Re22,Re12,Im12;
  double phi[2],theta[2],sit,cot,sip,cop;
  double d1,d2,d3,x,y,z,Cxyz[4];
  double Nup[2],Ndown[2];
  /* for OpenMP */
  int OMPID,Nthrds;

  /************************************
     Density_Grid in the partition B
  ************************************/

#pragma omp parallel shared(Density_Grid_B0,My_NumGridB_AB) private(OMPID,Nthrds,BN,Re11,Re22,Re12,Im12,Nup,Ndown,theta,phi) default(none)
  {

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();

    for (BN=OMPID; BN<My_NumGridB_AB; BN+=Nthrds){

      Re11 = Density_Grid_B0[0][BN];
      Re22 = Density_Grid_B0[1][BN];
      Re12 = Density_Grid_B0[2][BN];
      Im12 = Density_Grid_B0[3][BN];

      EulerAngle_Spin( 1, Re11, Re22, Re12, Im12, Re12, -Im12, Nup, Ndown, theta, phi );

      Density_Grid_B0[0][BN] = Nup[0];
      Density_Grid_B0[1][BN] = Ndown[0];
      Density_Grid_B0[2][BN] = theta[0];
      Density_Grid_B0[3][BN] = phi[0];
    }

#pragma omp flush(Density_Grid_B)

  } /* #pragma omp parallel */

  /************************************
     Density_Grid in the partition D
  ************************************/

#pragma omp parallel shared(Density_Grid_D,My_NumGridD) private(OMPID,Nthrds,DN,Re11,Re22,Re12,Im12,Nup,Ndown,theta,phi) default(none)
  {

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();

    for (DN=OMPID; DN<My_NumGridD; DN+=Nthrds){

      Re11 = Density_Grid_D[0][DN];
      Re22 = Density_Grid_D[1][DN];
      Re12 = Density_Grid_D[2][DN];
      Im12 = Density_Grid_D[3][DN];

      EulerAngle_Spin( 1, Re11, Re22, Re12, Im12, Re12, -Im12, Nup, Ndown, theta, phi );

      Density_Grid_D[0][DN] = Nup[0];
      Density_Grid_D[1][DN] = Ndown[0];
      Density_Grid_D[2][DN] = theta[0];
      Density_Grid_D[3][DN] = phi[0];
    }

#pragma omp flush(Density_Grid_D)

  } /* #pragma omp parallel */

}
