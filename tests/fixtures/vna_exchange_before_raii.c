/* Frozen VNA ring exchange from 842eca7d; retained solely as a test oracle. */
  for (ID=0; ID<numprocs; ID++){
    F_Snd_Num_WK[ID] = 0;
    F_Rcv_Num_WK[ID] = 0;
  }

  do {

    /***********************************
          set the size of data
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /* find the data size to send the block data */

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ){

	size1 = 0;
	n = F_Snd_Num_WK[IDS];

	Mc_AN = Snd_MAN[IDS][n];
	Gc_AN = Snd_GAN[IDS][n];
	Cwan = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_NO[Cwan];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];
	  size1 += tno1*tno2;
	}

	Snd_DS_VNA_Size[IDS] = size1;
	MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
	Snd_DS_VNA_Size[IDS] = 0;
      }

      /* receiving of the size of the data */

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_DS_VNA_Size[IDR] = size2;
      }
      else{
	Rcv_DS_VNA_Size[IDR] = 0;
      }

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) )  MPI_Wait(&request,&stat);

    } /* ID */

    /***********************************
               data transfer
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /******************************
         sending of the data
      ******************************/

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ){

	size1 = Snd_DS_VNA_Size[IDS];

	/*
	printf("S myid=%2d ID=%2d IDS=%2d IDR=%2d  size1=%2d\n",myid,ID,IDS,IDR,size1);fflush(stdout);
	*/

	/* allocation of the array */

	tmp_array = (Type_DS_VNA*)malloc(sizeof(Type_DS_VNA)*size1);

	/* multidimentional array to the vector array */

	num = 0;
	n = F_Snd_Num_WK[IDS];

	Mc_AN = Snd_MAN[IDS][n];
	Gc_AN = Snd_GAN[IDS][n];
	Cwan = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_NO[Cwan];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];

	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){
	      tmp_array[num] = DS_VNA[0][Mc_AN][h_AN][i][j];
	      num++;
	    }
	  }
	}

	MPI_Isend(&tmp_array[0], size1, MPI_Type_DS_VNA, IDS, tag, mpi_comm_level1, &request);

      }

      /******************************
        receiving of the block data
      ******************************/

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){

	size2 = Rcv_DS_VNA_Size[IDR];

	/*
	printf("R myid=%2d ID=%2d IDS=%2d IDR=%2d  size2=%2d\n",myid,ID,IDS,IDR,size2);fflush(stdout);
	*/

	tmp_array2 = (Type_DS_VNA*)malloc(sizeof(Type_DS_VNA)*size2);
	MPI_Recv(&tmp_array2[0], size2, MPI_Type_DS_VNA, IDR, tag, mpi_comm_level1, &stat);

	/* store */

	num = 0;
	n = F_Rcv_Num_WK[IDR];
	Original_Mc_AN = F_TopMAN[IDR] + n;

	Gc_AN = Rcv_GAN[IDR][n];
	Cwan = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_NO[Cwan];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];
	  for (i=0; i<tno1; i++){
	    for (j=0; j<tno2; j++){
	      DS_VNA[0][Matomnum+1][h_AN][i][j] = tmp_array2[num];
	      num++;
	    }
	  }
	}

	/* free tmp_array2 */
	free(tmp_array2);

	if (setpro_gpu){
	  SetPro_HVNA_GpuArchiveHalo(&hvna,DS_VNA, Original_Mc_AN, Gc_AN);
	}

	/*****************************************
              multiplying overlap integrals
	*****************************************/

        if (!setpro_gpu) {
          SetPro_HVNA_CPU_Halo(&hvna_geometry,&hvna_projectors,&halo_workspace,
              DS_VNA,HVNA,Original_Mc_AN,Rcv_GAN[IDR][F_Rcv_Num_WK[IDR]],fp32);
        }

	/********************************************
            increment of F_Rcv_Num_WK[IDR]
	********************************************/

	F_Rcv_Num_WK[IDR]++;

      }

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) {
	MPI_Wait(&request,&stat);
	free(tmp_array);  /* freeing of array */

	/********************************************
             increment of F_Snd_Num_WK[IDS]
	********************************************/

	F_Snd_Num_WK[IDS]++;
      }

    } /* ID */

    /*****************************************************
      check whether all the communications have finished
    *****************************************************/

    po = 0;
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) po += F_Snd_Num[IDS]-F_Snd_Num_WK[IDS];
      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ) po += F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR];
    }

  } while (po!=0);
