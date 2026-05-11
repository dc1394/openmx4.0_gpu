#**********************************************************************
#  example_openmx_api.py:
#
#   example_openmx_api.py is a minimal external Python program showing
#   how to split MPI_COMM_WORLD into subcommunicators by mpi4py and
#   call the OpenMX Python API module pyopenmx_api.
#
#
# Log of example_openmx_api.h:
#
#   15/April/2026  Released by T. Ozaki
#
#**********************************************************************

from mpi4py import MPI
import pyopenmx_api

comm = MPI.COMM_WORLD
world_rank = comm.Get_rank()
world_size = comm.Get_size()

nsplit = 2
nth = 1

if nsplit > world_size:
    if world_rank == 0:
        print("Error: nsplit is larger than world_size")
    raise SystemExit(1)

color = (world_rank * nsplit) // world_size
subcomm = comm.Split(color, world_rank)

if color == 0:
    input_file = "./Methane.dat"
else:
    input_file = "./Cdia.dat"

print(
    "before run:",
    "world_rank =", world_rank,
    "world_size =", world_size,
    "color =", color,
    "sub_rank =", subcomm.Get_rank(),
    "sub_size =", subcomm.Get_size(),
    "input =", input_file
)

ierr = pyopenmx_api.run(subcomm.py2f(), input_file, nth)

print(
    "after run:",
    "world_rank =", world_rank,
    "color =", color,
    "ierr =", ierr
)
