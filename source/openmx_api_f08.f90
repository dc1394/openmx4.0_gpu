!**********************************************************************
!    openmx_api_f08.f90 provides a minimal Fortran 2008 wrapper
!    to call the OpenMX API from an external program through mpi_f08.
!
!    This file converts a Fortran MPI communicator and an input file
!    name into the form required by the C API routine openmx_api_f().
!
! Log of openmx_api_f08.f90:
!
!    15/April/2026  Released by T. Ozaki
!
!***********************************************************************

module openmx_api_f08
  use, intrinsic :: iso_c_binding, only: c_int, c_char, c_null_char
  use mpi_f08, only: MPI_Comm
  implicit none
  private
  public :: openmx_run

  interface
    function openmx_api_f_c(fcomm, input_file, nt) bind(C, name="openmx_api_f") result(ierr)
      import :: c_int, c_char
      integer(c_int), value :: fcomm
      character(kind=c_char), intent(in) :: input_file(*)
      integer(c_int), value :: nt
      integer(c_int) :: ierr
    end function openmx_api_f_c
  end interface

contains

  integer function openmx_run(comm, input_file, nt) result(ierr)
    type(MPI_Comm), intent(in) :: comm
    character(len=*), intent(in) :: input_file
    integer, intent(in) :: nt

    character(kind=c_char), allocatable :: c_input(:)
    integer :: n, i

    n = len_trim(input_file)
    allocate(c_input(n + 1))

    do i = 1, n
      c_input(i) = input_file(i:i)
    end do
    c_input(n + 1) = c_null_char

    ierr = openmx_api_f_c(int(comm%MPI_VAL, c_int), c_input, int(nt, c_int))

    deallocate(c_input)
  end function openmx_run

end module openmx_api_f08
