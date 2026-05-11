!**********************************************************************
!  example_openmx_api_f08.f90:
!
!  example_openmx_api_f08.f90 is a minimal external Fortran 2008 program 
!  showing how to split MPI_COMM_WORLD into subcommunicators and call
!  the OpenMX API.
!
!  Log of example_openmx_api_f08.f90:
!
!  15/April/2026  Released by T. Ozaki
!
!***********************************************************************

program example_openmx_api_f08
  use mpi_f08
  use omp_lib
  use openmx_api_f08, only: openmx_run
  implicit none

  integer :: world_rank, world_size
  integer :: color, sub_rank, sub_size
  integer :: nsplit, nth_max, nth
  integer :: ierr, openmx_ierr
  character(len=256) :: omx_input_files(100)
  type(MPI_Comm) :: subcomm

  ! MPI initialization
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, world_rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, world_size, ierr)

  ! Get the number of threads
  nth_max = omp_get_max_threads()
  nth = 1
  if (nth_max < nth) then
    call MPI_Finalize(ierr)
    stop 1
  end if

  ! Set nsplit and input files to be passed
  nsplit = 2
  omx_input_files(1) = "./Methane.dat"
  omx_input_files(2) = "./Cdia.dat"

  if (nsplit > world_size) then
    if (world_rank == 0) then
      write(*,'(A,I0,A,I0,A)') &
        "Error: Nsplit = ", nsplit, &
        " is larger than the number of MPI processes = ", world_size, "."
    end if
    call MPI_Finalize(ierr)
    stop 1
  end if

  ! Create subgroup index
  color = (world_rank * nsplit) / world_size

  ! Split MPI_COMM_WORLD into nsplit subcommunicators
  call MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, subcomm, ierr)
  call MPI_Comm_rank(subcomm, sub_rank, ierr)
  call MPI_Comm_size(subcomm, sub_size, ierr)

  ! Call the OpenMX API from each subgroup
  ! Note: Fortran arrays are 1-based, so color+1 is used.
  openmx_ierr = openmx_run(subcomm, trim(omx_input_files(color + 1)), nth)

  call MPI_Barrier(MPI_COMM_WORLD, ierr)

  ! Clean up
  call MPI_Comm_free(subcomm, ierr)
  call MPI_Finalize(ierr)

  if (openmx_ierr /= 0) stop 1

end program example_openmx_api_f08
