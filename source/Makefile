###################################################################
#                                                                 #
#  Please set a proper CC and LIB for the compilation.            #
#  Examples of CC and LIB on several platforms are shown below.   #
#                                                                 #
###################################################################

#
# Ubuntu 22.04 LTS on WSL on Windows 11 (64bit)
#
# CC = /opt/nvidia/hpc_sdk/Linux_x86_64/23.5/comm_libs/mpi/bin/mpicc -O3 -Minfo=accel -acc -mtune=native -march=native -fopenmp -I${MKLROOT}/include -I${MKLROOT}/include/fftw -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.5/math_libs/12.1/targets/x86_64-linux/include -I/usr/local/cuda-12.1/include
# CC2 = /opt/nvidia/hpc_sdk/Linux_x86_64/23.5/comm_libs/mpi/bin/mpicc -O1 -Minfo=accel -acc -mtune=native -march=native -fopenmp -I${MKLROOT}/include -I${MKLROOT}/include/fftw -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.5/math_libs/12.1/targets/x86_64-linux/include -I/usr/local/cuda-12.1/include
# CC3 = /opt/nvidia/hpc_sdk/Linux_x86_64/23.5/comm_libs/mpi/bin/mpicc -O2 -Minfo=accel -acc -mtune=native -march=native -fopenmp -I${MKLROOT}/include -I${MKLROOT}/include/fftw-I/opt/nvidia/hpc_sdk/Linux_x86_64/23.5/math_libs/12.1/targets/x86_64-linux/include -I/usr/local/cuda-12.1/include
# FC = /opt/nvidia/hpc_sdk/Linux_x86_64/23.5/comm_libs/mpi/bin/mpif90 -O3 -mtune=native -march=native -fopenmp -I${MKLROOT}/include
# LIB= -L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_pgi_thread -lmkl_core -lmkl_blacs_openmpi_lp64 -pgf90libs -mp -lpthread -lmpi_mpifh -lmpi -lgfortran -lm -ldl -L/usr/local/cuda-12.1/lib64 -lcudart -lcusolver -Mcuda

#
# System C (Kugui) at ISSP, Univ. of Tokyo
#
# module purge && module load PrgEnv-intel/8.3.3 nvhpc-nompi/22.2_cuda11.2 openmpi_nvhpc/4.1.2
# CC = mpicc -O3 -Minfo=accel -acc -std=gnu11 -mtune=native -march=native -fopenmp -I${MKLROOT}/include -I${MKLROOT}/include/fftw -I/home/app/hpc_sdk/Linux_x86_64/22.2/math_libs/11.2/targets/x86_64-linux/include/ -I/home/app/hpc_sdk/Linux_x86_64/22.2/cuda/11.2/include
# CC2 = mpicc -O1 -Minfo=accel -acc -std=gnu11 -mtune=native -march=native -fopenmp -I${MKLROOT}/include -I${MKLROOT}/include/fftw -I/home/app/hpc_sdk/Linux_x86_64/22.2/math_libs/11.2/targets/x86_64-linux/include/ -I/home/app/hpc_sdk/Linux_x86_64/22.2/cuda/11.2/include
# CC3 = mpicc -O2 -Minfo=accel -acc -std=gnu11 -mtune=native -march=native -fopenmp -I${MKLROOT}/include -I${MKLROOT}/include/fftw -I/home/app/hpc_sdk/Linux_x86_64/22.2/math_libs/11.2/targets/x86_64-linux/include/ -I/home/app/hpc_sdk/Linux_x86_64/22.2/cuda/11.2/include
# FC = mpif90 -O3 -mtune=native -march=native -fopenmp -I${MKLROOT}/include
# LIB= -L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_pgi_thread -lmkl_core -lmkl_blacs_openmpi_lp64 -pgf90libs -mp -lpthread -lmpi_mpifh -lmpi -lgfortran -lm -ldl -L/home/app/hpc_sdk/Linux_x86_64/22.2/cuda/11.2/lib64 -lcublas -lcudart -lcusolver -Mcuda

#
# Wisteria/BDEC-01 Wisteria-A (Aquarius) at Information Technology Center, Univ. of Tokyo
#
# module load cuda/11.8 nvidia/24.1 nvmpi/24.1
# MKLROOT=/work/opt/local/x86_64/cores/intel/2023.1.0/mkl/latest
# CC = mpicc -O3 -std=gnu11 -Minfo=accel -acc -mtune=native -march=native -fopenmp -I${MKLROOT}/include -I${MKLROOT}/include/fftw -I/work/opt/local/x86_64/cores/nvidia/22.7/Linux_x86_64/22.7/cuda/11.7/include -I/work/opt/local/x86_64/cores/nvidia/22.7/Linux_x86_64/22.7/math_libs/11.7/targets/x86_64-linux/include/
# CC2 = mpicc -O1 -std=gnu11 -Minfo=accel -acc -mtune=native -march=native -fopenmp -I${MKLROOT}/include -I${MKLROOT}/include/fftw -I/work/opt/local/x86_64/cores/nvidia/22.7/Linux_x86_64/22.7/cuda/11.7/include -I/work/opt/local/x86_64/cores/nvidia/22.7/Linux_x86_64/22.7/math_libs/11.7/targets/x86_64-linux/include/
# CC3 = mpicc -O2 -std=gnu11 -Minfo=accel -acc -mtune=native -march=native -fopenmp -I${MKLROOT}/include -I${MKLROOT}/include/fftw -I/work/opt/local/x86_64/cores/nvidia/22.7/Linux_x86_64/22.7/cuda/11.7/include -I/work/opt/local/x86_64/cores/nvidia/22.7/Linux_x86_64/22.7/math_libs/11.7/targets/x86_64-linux/include/
# FC = mpif90 -O3 -Minfo=accel -acc -mtune=native -march=native -fopenmp -I${MKLROOT}/include
# LIB= -L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_pgi_thread -lmkl_core -lmkl_blacs_openmpi_lp64 -pgf90libs -mp -lpthread -lmpi_mpifh -lmpi -lgfortran -lm -ldl -L/work/opt/local/x86_64/cores/nvidia/22.7/Linux_x86_64/22.7/math_libs/11.7/targets/x86_64-linux/lib -lcudart -lcuda -lcusolver -lcublas -Mcuda 

#
# Pegasus at Center for Computational Sciences, University of Tsukuba
#
# module load nvhpc-nompi/23.1 openmpi/4.1.6/nvhpc23.1-cuda12.0
# MKLROOT = /system/apps/ubuntu/22.04-202404/oneapi/2023.0.0/mkl/2023.0.0
# CC = mpicc -O3 -Minfo=accel -acc -std=gnu11 -mtune=native -march=native -fopenmp -I${MKLROOT}/include -I${MKLROOT}/include/fftw -I/system/apps/ubuntu/22.04-202404/nvhpc/23.1/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/include -I/system/apps/ubuntu/22.04-202404/nvhpc/23.1/Linux_x86_64/23.1/cuda/12.0/include
# CC2 = mpicc -O1 -Minfo=accel -acc -std=gnu11 -mtune=native -march=native -fopenmp -I${MKLROOT}/include -I${MKLROOT}/include/fftw -I/system/apps/ubuntu/22.04-202404/nvhpc/23.1/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/include -I/system/apps/ubuntu/22.04-202404/nvhpc/23.1/Linux_x86_64/23.1/cuda/12.0/include
# CC3 = mpicc -O2 -Minfo=accel -acc -std=gnu11 -mtune=native -march=native -fopenmp -I${MKLROOT}/include -I${MKLROOT}/include/fftw -I/system/apps/ubuntu/22.04-202404/nvhpc/23.1/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/include -I/system/apps/ubuntu/22.04-202404/nvhpc/23.1/Linux_x86_64/23.1/cuda/12.0/include
# FC = mpif90 -O2 -mtune=native -march=native -fopenmp -I${MKLROOT}/include
# LIB= -L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_pgi_thread -lmkl_core -lmkl_blacs_openmpi_lp64 -pgf90libs -mp -lpthread -lmpi_mpifh -lmpi -lm -ldl -L/system/apps/ubuntu/22.04-202404/nvhpc/23.1/Linux_x86_64/23.1/cuda/12.0/lib64 -L/system/apps/ubuntu/22.04-202404/nvhpc/23.1/Linux_x86_64/23.1/math_libs/12.0/targets/x86_64-linux/lib -lcudart -lcusolver -lcublas -cuda

#
# Miyabi at Center for Computational Sciences, University of Tsukuba
#
# module purge && module load nvidia ompi-cuda/4.1.6-13.1 && module load fftw/3.3.10
# CC = mpicc -O3 -Minfo=accel -acc -std=gnu11 -mtune=native -march=native -fopenmp -I/work/opt/local/aarch64/apps/nvidia/24.9/fftw/3.3.10/include/fftw -I/work/opt/local/aarch64/cores/nvidia/24.9/Linux_aarch64/24.9/math_libs/13.1/targets/sbsa-linux/include -I/work/opt/local/aarch64/cores/nvidia/24.9/Linux_aarch64/24.9/cuda/13.1/include
# CC2 = mpicc -O1 -Minfo=accel -acc -std=gnu11 -mtune=native -march=native -fopenmp -I/work/opt/local/aarch64/apps/nvidia/24.9/fftw/3.3.10/include/fftw -I/work/opt/local/aarch64/cores/nvidia/24.9/Linux_aarch64/24.9/math_libs/13.1/targets/sbsa-linux/include -I/work/opt/local/aarch64/cores/nvidia/24.9/Linux_aarch64/24.9/cuda/13.1/include
# CC3 = mpicc -O2 -Minfo=accel -acc -std=gnu11 -mtune=native -march=native -fopenmp -I/work/opt/local/aarch64/apps/nvidia/24.9/fftw/3.3.10/include/fftw -I/work/opt/local/aarch64/cores/nvidia/24.9/Linux_aarch64/24.9/math_libs/13.1/targets/sbsa-linux/include -I/work/opt/local/aarch64/cores/nvidia/24.9/Linux_aarch64/24.9/cuda/13.1/include
# FC = mpif90 -O2 -mtune=native -march=native -fopenmp -I${MKLROOT}/include
# -L/home/x10402/nvpl-linux-sbsa-24.7/lib -lnvpl_scalapack_lp64 -lnvpl_blacs_lp64_openmpi4 -lnvpl_lapack_lp64_gomp -lnvpl_blas_lp64_gomp -pgf90libs -lmpi_mpifh -lfftw3 -lpthread -lgfortran -lm -ldl -L/work/opt/local/aarch64/cores/nvidia/24.9/Linux_aarch64/24.9/cuda/13.1/lib64 -L/work/opt/local/aarch64/cores/nvidia/24.9/Linux_aarch64/24.9/math_libs/13.1/targets/sbsa-linux/lib -lcudart -lcusolver -lcublas -cuda

# CUDA may be installed separately from the SDK (26.9 bundles CUDA 13.3).
# Override these variables to build against an older SDK/toolkit.
NVHPC_ROOT          ?= /opt/nvidia/hpc_sdk/Linux_x86_64/26.9
NVHPC_CUDA_VERSION  ?= 13.4
NVHPC_CUDA_HOME     ?= $(if $(wildcard /usr/local/cuda-$(NVHPC_CUDA_VERSION)/include/cuda.h),/usr/local/cuda-$(NVHPC_CUDA_VERSION),$(NVHPC_ROOT)/cuda/$(NVHPC_CUDA_VERSION))
NVHPC_CUDA_TARGET   ?= $(NVHPC_CUDA_HOME)/targets/x86_64-linux
NVHPC_COMPILER_BIN  ?= $(NVHPC_ROOT)/compilers/bin
NVHPC_COMPILER_LIB  ?= $(NVHPC_ROOT)/compilers/lib
NVHPC_MPI_BIN       ?= $(NVHPC_ROOT)/comm_libs/mpi/bin
NVHPC_COMM_HOME     ?= $(or $(wildcard $(NVHPC_ROOT)/comm_libs/$(NVHPC_CUDA_VERSION)),$(lastword $(dir $(wildcard $(NVHPC_ROOT)/comm_libs/*/hpcx))))
export NVCOMPILER_COMM_LIBS_HOME := $(NVHPC_COMM_HOME)
# Resolve the wrapper's MPI library directory; its CUDA version need not
# match the external toolkit. In particular 26.9 uses comm_libs/13.3.
NVHPC_MPI_LIBDIR    ?= $(shell $(NVHPC_MPI_BIN)/mpicc --showme:libdirs 2>/dev/null | awk '{print $$1}')
# Prefer the toolkit's math libraries so headers and shared objects agree.
NVHPC_MATH_DIR      ?= $(if $(wildcard $(NVHPC_CUDA_TARGET)/include/cusolverDn.h),$(NVHPC_CUDA_TARGET),$(NVHPC_ROOT)/math_libs/$(NVHPC_CUDA_VERSION)/targets/x86_64-linux)
# nvc/nvfortran ignore CUDA_HOME. Export their documented override as well
# as explicit include/link paths; otherwise OpenACC silently uses CUDA 13.3.
export NVHPC_CUDA_HOME
export NVCOMPILER_CUDA_HOME := $(NVHPC_CUDA_HOME)
export NVCOMPILER_MATH_LIBS_HOME := $(NVHPC_MATH_DIR)

GPUSOLVER2_PREFIX ?= $(CURDIR)/third_party/gpusolver2-install

# Keep the caller's CMake selection when the compiler environment below
# replaces PATH. /usr/local/bin may contain an older CMake than the caller's.
OPENMX_CALLER_CMAKE := $(shell command -v cmake 2>/dev/null)
export PATH := $(NVHPC_COMPILER_BIN):$(NVHPC_MPI_BIN):$(NVHPC_CUDA_HOME)/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

unexport CPATH
unexport C_INCLUDE_PATH
unexport CPLUS_INCLUDE_PATH
unexport LIBRARY_PATH
unexport LD_LIBRARY_PATH
unexport CFLAGS
unexport CXXFLAGS
unexport CPPFLAGS
unexport FCFLAGS
unexport FFLAGS
unexport LDFLAGS
unexport LIBS
unexport CC
unexport CXX
unexport CPP
unexport GCC
unexport GXX
unexport AR
unexport AS
unexport LD
unexport NM
unexport RANLIB
unexport OBJDUMP
unexport STRIP
unexport READELF
unexport ADDR2LINE
unexport DLLTOOL
unexport BUILD
unexport HOST
unexport build_alias
unexport host_alias
unexport CUDA_HOME
unexport NVCC_PREPEND_FLAGS
unexport NVCC_APPEND_FLAGS

MPICC  ?= $(NVHPC_MPI_BIN)/mpicc
MPIF90 ?= $(NVHPC_MPI_BIN)/mpif90

FFTW3_REPO_URL ?= https://github.com/FFTW/fftw3.git
FFTW3_TAG      ?= fftw-3.3.11
FFTW3_VERSION  := $(patsubst fftw-%,%,$(FFTW3_TAG))
FFTW3_RELEASE_URL ?= https://fftw.org/fftw-$(FFTW3_VERSION).tar.gz
FFTW3_RELEASE_TARBALL ?= $(CURDIR)/third_party/fftw-$(FFTW3_VERSION).tar.gz
# SHA-256 of the bundled fftw-3.3.11.tar.gz (its MD5 matches the one fftw.org
# publishes); empty skips the check, as for another FFTW3_TAG.
FFTW3_RELEASE_SHA256 ?= $(if $(filter 3.3.11,$(FFTW3_VERSION)),5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1)
FFTW3_SRC_DIR  ?= third_party/fftw3
FFTW3_BUILD_DIR ?= build/fftw3
FFTW3_PREFIX   ?= $(CURDIR)/third_party/fftw3-install
FFTW3_CC       ?= $(NVHPC_COMPILER_BIN)/nvc
FFTW3_CXX      ?= $(NVHPC_COMPILER_BIN)/nvc++
FFTW3_FC       ?= $(NVHPC_COMPILER_BIN)/nvfortran
FFTW3_F77      ?= $(FFTW3_FC)
# Keep the portable build policy; site files can enable host optimization
# without importing unrelated user CFLAGS into this bundled dependency.
FFTW3_CFLAGS   ?=
# OpenMX links only libfftw3.a.  --disable-doc also spares the build makeinfo:
# the overlay below keeps the release dates of the tarball's files, so the
# info manual would look older than the checked-out .texi sources.
FFTW3_CONFIGURE_OPTIONS ?= --enable-static --disable-shared --disable-doc
FFTW3_CONFIGURE_TRIPLET ?= x86_64-pc-linux-gnu
FFTW3_BUILD_ENV = PATH="$(PATH)" CONFIG_SITE= build_alias= host_alias= BUILD= HOST= CPATH= C_INCLUDE_PATH= CPLUS_INCLUDE_PATH= LIBRARY_PATH= LD_LIBRARY_PATH= CFLAGS="$(FFTW3_CFLAGS)" CXXFLAGS= CPPFLAGS= FCFLAGS= FFLAGS= LDFLAGS= LIBS= CPP="$(FFTW3_CC) -E" AR=/usr/bin/ar AS=/usr/bin/as RANLIB=/usr/bin/ranlib NM=/usr/bin/nm LD=/usr/bin/ld OBJDUMP=/usr/bin/objdump STRIP=/usr/bin/strip READELF=/usr/bin/readelf ADDR2LINE=/usr/bin/addr2line DLLTOOL=
FFTW3_SRC_ABS  := $(abspath $(FFTW3_SRC_DIR))
FFTW3_BUILD_ABS := $(abspath $(FFTW3_BUILD_DIR))
FFTW3_CODELET_SENTINEL := $(FFTW3_SRC_ABS)/dft/scalar/codelets/n1_2.c
FFTW3_LIB      ?= $(FFTW3_PREFIX)/lib/libfftw3.a
FFTW3_INCLUDE  ?= -I$(FFTW3_PREFIX)/include
FFTW3_LIBS     ?= $(FFTW3_LIB)

COMMON_INCLUDES = $(FFTW3_INCLUDE) -I$(NVHPC_MATH_DIR)/include -I$(NVHPC_CUDA_TARGET)/include -I$(NVHPC_CUDA_HOME)/include
NVHPC_GPU_FLAGS ?= $(if $(GEMMUL8_BUILD_GPU_ARCH),-gpu=cc$(GEMMUL8_BUILD_GPU_ARCH))
# NVHPC 26.5 lowers explicit fmaf calls to an unsupported OpenACC FFMA
# operation in double-float evaluators. Keep their explicit library calls;
# the orbital evaluator's -gpu=nofma must also remain in effect.
NVHPC_EXPLICIT_FMA_FLAGS := $(shell "$(NVHPC_COMPILER_BIN)/nvc" --version 2>/dev/null | awk '$$1 == "nvc" && $$2 ~ /^26[.]5(-|$$)/ {print "-Mnobuiltin"; exit}')
COMMON_CFLAGS   = --diag_error=implicit_func_decl -Minfo=accel -acc $(NVHPC_GPU_FLAGS) -std=gnu11 -mtune=native -march=native -fopenmp $(COMMON_INCLUDES)

CC  = $(MPICC) $(COMMON_CFLAGS) -O3
CC2 = $(MPICC) $(COMMON_CFLAGS) -O1
CC3 = $(MPICC) $(COMMON_CFLAGS) -O2
CC_NOACC = $(filter-out -Minfo=accel -acc,$(CC2))
CC_NOACC_O3 = $(filter-out -Minfo=accel -acc,$(CC))
CC_DFTD3 = $(MPICC) -w -O0 -std=gnu11 -fopenmp $(COMMON_INCLUDES)
FC = $(MPIF90) -O2 -mtune=native -march=native -fopenmp $(FFTW3_INCLUDE) -I$(NVHPC_MATH_DIR)/include

NVPL_SCALAPACK_LIB ?= $(NVHPC_MPI_LIBDIR)/libscalapack_lp64.a
NVPL_LAPACK_LIB    ?= $(NVHPC_COMPILER_LIB)/liblapack_lp64.a
NVPL_BLAS_LIB      ?= $(NVHPC_COMPILER_LIB)/libblas_lp64.a
NVPL_LIBS          ?= -Wl,--start-group $(NVPL_SCALAPACK_LIB) $(NVPL_LAPACK_LIB) $(NVPL_BLAS_LIB) -Wl,--end-group
SDK_RPATH_TAGS     ?= -Wl,--disable-new-dtags
MPI_FORTRAN_LIBS   ?= -L$(NVHPC_MPI_LIBDIR) $(SDK_RPATH_TAGS) -Wl,-rpath,$(NVHPC_MPI_LIBDIR) -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi
CUDA_LIBS          ?= $(SDK_RPATH_TAGS) -L$(NVHPC_CUDA_TARGET)/lib -Wl,-rpath,$(NVHPC_CUDA_TARGET)/lib -L$(NVHPC_MATH_DIR)/lib -Wl,-rpath,$(NVHPC_MATH_DIR)/lib -L$(NVHPC_CUDA_TARGET)/lib/stubs -Wl,--no-as-needed -lnvJitLink -Wl,--as-needed -lcudart -lcusolver -lcublas -cuda

LIB= $(FFTW3_LIBS) $(GPUSOLVER2_LIBS) $(MAGMA_LIBS) $(NVPL_LIBS) $(MPI_FORTRAN_LIBS) -pgf90libs -mp -lpthread -lm -ldl $(CUDA_LIBS)

GEMMUL8_REPO_URL ?= https://github.com/RIKEN-RCCS/GEMMul8.git
GEMMUL8_REPO_DIR ?= third_party/GEMMul8
# gemmul8_openmx.cu instantiates GEMMul8 internals directly, so the
# checkout must stay pinned to a commit whose internal layout it matches.
# f8fdf0b is the v3.5.4 release.
GEMMUL8_COMMIT   ?= f8fdf0b2a5eafd4dbbfab6be8489d3356e0da06a
GEMMUL8_DIR      ?= $(GEMMUL8_REPO_DIR)
GEMMUL8_LIB      ?= $(GEMMUL8_DIR)/lib/libgemmul8.a
GEMMUL8_OPENMX_OBJ ?= $(GEMMUL8_DIR)/src/gemmul8_openmx.o
GEMMUL8_CUDA_PATH ?= $(NVHPC_CUDA_HOME)
GEMMUL8_GPU_ARCH ?= auto
ifeq ($(GEMMUL8_GPU_ARCH),auto)
GEMMUL8_DETECTED_GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | awk '/^[0-9]+[.][0-9]+$$/ {gsub(/[.]/, ""); print; exit}')
GEMMUL8_BUILD_GPU_ARCH := $(GEMMUL8_DETECTED_GPU_ARCH)
else
GEMMUL8_BUILD_GPU_ARCH := $(GEMMUL8_GPU_ARCH)
endif
NVCC ?= $(GEMMUL8_CUDA_PATH)/bin/nvcc
NVCC_HOST ?= $(NVHPC_COMPILER_BIN)/nvc++
NVCC_GEMMUL8_ENV = PATH="$(PATH)" CUDA_HOME="$(NVHPC_CUDA_HOME)" CPATH= C_INCLUDE_PATH= CPLUS_INCLUDE_PATH= LIBRARY_PATH= LD_LIBRARY_PATH= NVCC_PREPEND_FLAGS= NVCC_APPEND_FLAGS=
NVCC_GEMMUL8_FLAGS = -ccbin $(NVCC_HOST) -std=c++20 -O3 -diag-suppress 177 -DGPU_ARCH=$(GEMMUL8_BUILD_GPU_ARCH) -I$(GEMMUL8_DIR)/include -I$(GEMMUL8_DIR)/src -I$(GEMMUL8_CUDA_PATH)/include -I$(NVHPC_CUDA_TARGET)/include -I$(NVHPC_MATH_DIR)/include -gencode arch=compute_$(GEMMUL8_BUILD_GPU_ARCH),code=sm_$(GEMMUL8_BUILD_GPU_ARCH)

# MAGMA: the multi-GPU dense eigensolver of the single-node Cluster and Band
# solvers (the memory fallback of gpusolver when one GPU cannot hold a dense
# solve, and scf.eigen.lib gpusolver3).  It needs two or more GPUs, so
#   WITH_MAGMA=auto (default)  links it when nvidia-smi shows two or more
#                              GPUs on the build node, building it first
#                              when MAGMA_PREFIX has none; with fewer GPUs
#                              it links only an existing installation
#   WITH_MAGMA=1               builds and links it on any node (e.g. on a
#                              login node without GPUs, for multi-GPU
#                              compute nodes), and rebuilds it when its
#                              toolchain, build script or patch changes
#   WITH_MAGMA=0               builds without it
# Without MAGMA, a dense solve that does not fit on one GPU is recovered on
# the CPU.  The custom multi-GPU GEMM implementation is always built in.
WITH_MAGMA ?= auto
ifeq ($(filter auto 0 1,$(WITH_MAGMA)),)
$(error WITH_MAGMA must be auto, 1 or 0 (got "$(WITH_MAGMA)"))
endif
# GPUs on the build node (a command-line MAGMA_NODE_GPUS=N overrides it)
ifeq ($(WITH_MAGMA)$(origin MAGMA_NODE_GPUS),autoundefined)
MAGMA_NODE_GPUS := $(shell nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -c '^[0-9]')
endif
MAGMA_AUTO_BUILD := $(if $(filter auto,$(WITH_MAGMA)),$(if $(filter-out 0 1,$(MAGMA_NODE_GPUS)),1))
MAGMA_PREFIX ?= $(CURDIR)/third_party/magma-install
MAGMA_BUILD ?= $(CURDIR)/build/magma
# The pinned MAGMA 2.10.0 archive (build_magma.sh verifies its SHA-256) is
# bundled for the HIP backend; using it keeps the build off the network.
# It is found through the real location of this Makefile, because CMake
# builds read it through a link in the build tree.  When it is absent
# (MAGMA_ARCHIVE empty), build_magma.sh downloads the same archive.
MAGMA_ARCHIVE ?= $(wildcard $(dir $(realpath $(firstword $(MAKEFILE_LIST))))third_party/hip/magma/magma-2.10.0.tar.gz)
MAGMA_LIBRARY = $(MAGMA_PREFIX)/lib/libmagma.so
MAGMA_ENABLED = $(if $(filter 1,$(WITH_MAGMA)),1,$(if $(filter auto,$(WITH_MAGMA)),$(if $(MAGMA_AUTO_BUILD)$(filter magma-stack,$(MAKECMDGOALS))$(wildcard $(MAGMA_LIBRARY)),1,0),0))
MAGMA_CMAKE ?= $(or $(OPENMX_CALLER_CMAKE),cmake)
MAGMA_JOBS ?= 3
MAGMA_HOST_CC ?= $(GPUSOLVER2_HOST_CC)
MAGMA_HOST_CXX ?= $(NVCC_HOST)
MAGMA_HOST_LIBDIR ?=
MAGMA_LAPACK_LIBRARIES ?= $(NVHPC_COMPILER_LIB)/liblapack_lp64.so;$(NVHPC_COMPILER_LIB)/libblas_lp64.so
ifeq ($(MAGMA_ENABLED),1)
MAGMA_CPPFLAGS = -DOPENMX_HAVE_MAGMA -I$(MAGMA_PREFIX)/include
MAGMA_LIBS = -L$(MAGMA_PREFIX)/lib $(SDK_RPATH_TAGS) -Wl,-rpath,$(MAGMA_PREFIX)/lib -lmagma
MAGMA_DEPENDENCY = $(MAGMA_LIBRARY)
endif
# -lcusparse is not called directly but is a private dependency of
# libcusolver; naming it here (shielded from --as-needed) makes it a
# direct dependency of openmx, so the RPATH of the binary resolves it
# (a dependency loaded through libcusolver would follow
# LD_LIBRARY_PATH instead, which can mix CUDA library generations).
LIB += -lcublasLt -Wl,--no-as-needed -lcusparse -Wl,--as-needed -lcuda -lnvidia-ml -lstdc++ -latomic

# host compiler for the small build-time helper tools (force_rpath)
HOST_CC ?= gcc

# ============================================================================
# gpusolver2 (scf.eigen.lib=gpusolver2): distributed multi-GPU cluster
# diagonalization via ELPA 2026.02 with NVIDIA GPU kernels (eigensolver) and
# COSMA (pdgemm).  Both are built from the release archives bundled in
# third_party/dist by third_party/build_gpusolver2_stack.sh.
# ============================================================================
GPUSOLVER2_DIST      ?= $(CURDIR)/third_party/dist
GPUSOLVER2_BUILD     ?= $(CURDIR)/build/gpusolver2
GPUSOLVER2_PREFIX    ?= $(CURDIR)/third_party/gpusolver2-install
GPUSOLVER2_HOST_CC   ?= gcc
GPUSOLVER2_HOST_CXX  ?= g++
GPUSOLVER2_NVCC      ?= $(NVHPC_CUDA_HOME)/bin/nvcc
GPUSOLVER2_GPU_ARCH  ?= $(GEMMUL8_BUILD_GPU_ARCH)
GPUSOLVER2_MPI_BIN   ?= $(NVHPC_MPI_BIN)
GPUSOLVER2_MPI_LIBDIR ?= $(NVHPC_MPI_LIBDIR)
# HPC SDK's mpi/bin is a selector; ../include is not its active MPI tree.
GPUSOLVER2_MPI_INCLUDES ?= $(addprefix -I,$(shell $(GPUSOLVER2_MPI_BIN)/mpicc --showme:incdirs 2>/dev/null))
GPUSOLVER2_CUDA_HOME ?= $(NVHPC_CUDA_HOME)
GPUSOLVER2_MATH_DIR  ?= $(NVHPC_MATH_DIR)
GPUSOLVER2_COMPILER_LIB ?= $(NVHPC_COMPILER_LIB)
GPUSOLVER2_COMPILER_BIN ?= $(NVHPC_COMPILER_BIN)
GPUSOLVER2_SCALAPACK_SO ?= $(NVHPC_MPI_LIBDIR)/libscalapack_lp64.so
GPUSOLVER2_CMAKE     ?=
GPUSOLVER2_JOBS      ?= $(shell nproc 2>/dev/null || echo 8)
GPUSOLVER2_ELPA_VER  ?= 2026.02.002
GPUSOLVER2_ELPA_INCLUDE ?= $(GPUSOLVER2_PREFIX)/include/elpa-$(GPUSOLVER2_ELPA_VER)
GPUSOLVER2_STAMP     = $(GPUSOLVER2_BUILD)/stamp/02_cosma.done

# Keep vendor ScaLAPACK symbols intact. COSTA's unprefixed compatibility
# archive also replaces internal SVD transpose/redistribution calls and
# corrupts distributed singular vectors. Use only the explicit costa_ and
# cosma_ interfaces for accelerated operations.
GPUSOLVER2_LIBS = $(GPUSOLVER2_PREFIX)/lib/libelpa.a \
 -Wl,--start-group \
 $(GPUSOLVER2_PREFIX)/lib/libcosma_prefixed_pxgemm.a \
 $(GPUSOLVER2_PREFIX)/lib/libcosma.a \
 $(GPUSOLVER2_PREFIX)/lib/libcosta_prefixed_scalapack.a \
 $(GPUSOLVER2_PREFIX)/lib/libcosta.a \
 $(GPUSOLVER2_PREFIX)/lib/libTiled-MM.a \
 -Wl,--end-group

CFLAGS  = -g
CPPFLAGS =

OBJS    = openmx_grid_assembly.o openmx.o openmx_common.o Input_std.o Inputtools.o init.o LU_inverse.o ReLU_inverse.o \
          truncation.o readfile.o FT_PAO.o FT_NLP.o FT_ProExpn_VNA.o FT_VNA.o FT_ProductPAO.o \
          Hamiltonian_Cluster.o Hamiltonian_Cluster_Hs.o Hamiltonian_Cluster_NC_Hs2.o Hamiltonian_Band_NC_Hs2.o \
          Overlap_Cluster_NC_Ss2.o Overlap_Band_NC_Ss2.o Overlap_Cluster.o Overlap_Cluster_Ss.o Overlap_Cluster_LNO_Ss.o \
          Set_ContMat_Cluster_LNO.o Hamiltonian_Band.o Matrix_Band_LNO.o Overlap_Band.o Hamiltonian_Cluster_NC.o \
          Hamiltonian_Band_NC.o Hamiltonian_Cluster_SO.o Get_OneD_HS_Col.o \
          SetPara_DFT.o XC_Ceperly_Alder.o XC_CA_LSDA.o \
          XC_PW92C.o XC_PBE.o XC_EX.o DFT.o Mixing_DM.o Mixing_H.o Mixing_V.o Force.o Stress.o Poisson.o Poisson_ESM.o \
          Cluster_DFT_Col.o Cluster_DFT_NonCol.o Cluster_DFT_Dosout.o Cluster_DFT_ON2.o Cluster_DFT_Col_LNO.o \
          Band_DFT_Col.o Band_DFT_NonCol.o Band_DFT_NonCol_GB.o Band_DFT_kpath.o Band_DFT_kpath_LNO.o Band_DFT_MO.o \
          Cluster_DFT_Col_CWF.o Cluster_DFT_NonCol_CWF.o Band_DFT_Col_CWF.o Band_DFT_NonCol_CWF.o \
          Unfolding_Bands.o Band_DFT_Dosout.o Set_Density_Grid.o Set_Density_Grid_GPU.o Set_Orbitals_Grid.o Set_Aden_Grid.o \
          Gauss_Legendre.o zero_cfrac.o xyz2spherical.o AngularF.o RadialF.o Dr_RadialF.o PhiF.o \
          VNAF.o Dr_VNAF.o VH_AtomF.o Dr_VH_AtomF.o RF_BesselF.o QuickSort.o Nonlocal_RadialF.o \
          KumoF.o Dr_KumoF.o Mulliken_Charge.o Occupation_Number_LDA_U.o Eff_Hub_Pot.o Coulomb_Interaction.o \
          EulerAngle_Spin.o Smoothing_Func.o Orbital_Moment.o Pot_NeutralAtom.o Simple_Mixing_DM.o \
          DIIS_Mixing_DM.o ADIIS_Mixing_DM.o GR_Pulay_DM.o Kerker_Mixing_Rhok.o DIIS_Mixing_Rhok.o \
          Total_Energy.o Total_Energy_GPU.o Contract_Hamiltonian.o Contract_iHNL.o Cont_Matrix0.o Cont_Matrix1.o Cont_Matrix2.o \
          Cont_Matrix3.o Cont_Matrix4.o Opt_Contraction.o Initial_CntCoes.o Initial_CntCoes2.o \
          Set_XC_Grid.o \
          Get_Orbitals.o Get_dOrbitals.o Get_Cnt_Orbitals.o \
          Get_Cnt_dOrbitals.o Gaunt.o Find_CGrids.o MD_pac.o \
          RestartFileDFT.o Output_CompTime.o Merge_LogFile.o Make_FracCoord.o \
          Make_InputFile_with_FinalCoord.o Output_Energy_Decomposition.o \
          Divide_Conquer.o Divide_Conquer_LNO.o gemmul8_bridge.o eigen_refine_gpu.o grid_precision.o openmx_precision.o elpa_cosma_bridge.o Krylov.o \
          Divide_Conquer_Dosout.o EGAC_DFT.o LNO.o Calc_LNAO.o Calc_LNBO.o \
          Eigen_lapack.o gpusolver_Syevd.o gpusolver_Syevdx.o openmx_cusolver_compat.o set_cuda_default_device_from_local_rank.o set_openacc_device_from_local_rank.o Eigen_lapack2.o Eigen_lapack3.o EigenBand_lapack.o \
          Eigen_PReHH.o BroadCast_ReMatrix.o \
          Eigen_PHH.o BroadCast_ComplexMatrix.o \
          lapack_dstedc1.o lapack_dstedc2.o lapack_dstedc3.o\
          lapack_dstegr1.o lapack_dstegr2.o lapack_dstegr3.o \
          lapack_dstevx1.o lapack_dstevx2.o lapack_dstevx3.o \
          lapack_dstevx4.o lapack_dstevx5.o lapack_dsteqr1.o \
          Nonlocal_Basis.o Set_OLP_Kin.o Set_Nonlocal.o Set_ProExpn_VNA.o \
          Set_CoreHoleMatrix.o Set_XMCD_penalty.o Set_OLP_p.o Set_Hamiltonian.o set_hamiltonian_gpu.o Set_Vpot.o \
          Voronoi_Charge.o Voronoi_Orbital_Moment.o Fuzzy_Weight.o \
          dampingF.o deri_dampingF.o Spherical_Bessel.o \
          iterout.o iterout_md.o Allocate_Arrays.o Free_Arrays.o \
          CWF2W90.o Init_List_YOUSO.o outputfile1.o \
          malloc_multidimarray.o PrintMemory.o PrintMemory_Fix.o \
          dtime.o OutData.o OutData_Binary.o init_alloc_first.o File_CntCoes.o \
          SCF2File.o mimic_sse.o Make_Comm_Worlds.o \
          Set_Allocate_Atom2CPU.o Cutoff.o Generating_MP_Special_Kpt.o \
          Maketest.o Runtest.o Memory_Leak_test.o \
          Force_test.o Stress_test.o Show_DFT_DATA.o Generate_Wannier.o \
          TRAN_Allocate.o TRAN_DFT.o TRAN_DFT_Dosout.o TRAN_Apply_Bias2e.o \
          TRAN_Deallocate_Electrode_Grid.o TRAN_Deallocate_RestartFile.o \
          TRAN_RestartFile.o TRAN_Calc_CentGreen.o TRAN_Input_std.o \
          TRAN_Set_CentOverlap.o TRAN_Calc_CentGreenLesser.o \
          TRAN_Input_std_Atoms.o TRAN_Set_Electrode_Grid.o \
          TRAN_Calc_GridBound.o TRAN_Set_IntegPath.o TRAN_Output_HKS.o \
          TRAN_Set_MP.o TRAN_Calc_SelfEnergy.o TRAN_Output_Trans_HS.o \
          TRAN_Calc_Hopping_G.o TRAN_Calc_SurfGreen.o TRAN_Set_SurfOverlap.o \
          TRAN_Add_Density_Lead.o TRAN_Add_ADensity_Lead.o TRAN_Set_Value.o \
          TRAN_Poisson.o TRAN_adjust_Ngrid.o TRAN_Print.o TRAN_Print_Grid.o \
          Lapack_LU_inverse.o TRAN_Distribute_Node.o TRAN_Output_HKS_Write_Grid.o \
          TRAN_Credit.o TRAN_Check_Region_Lead.o TRAN_Check_Region.o TRAN_Check_Input.o \
          DFTDvdW_init.o DFTD3vdW_init.o neb.o neb_run.o neb_check.o \
          TRAN_Allocate_NC.o TRAN_DFT_NC.o TRAN_Set_CentOverlap_NC.o TRAN_Set_SurfOverlap_NC.o \
          TRAN_Calc_OneTransmission.o TRAN_Main_Analysis.o TRAN_Main_Analysis_NC.o \
          MTRAN_EigenChannel.o TRAN_Channel_Functions.o TRAN_Channel_Output.o \
          TRAN_Calc_CurrentDensity.o TRAN_CDen_Main.o \
          openmx_elpa_cpu.o \
          NBO_Cluster.o NBO_Krylov.o Population_Analysis_Wannier.o Population_Analysis_Wannier2.o \
          NabraMatrixElements.o Set_dOrbitals_Grid.o Calc_optical.o \
          Band_DFT_NonCol_Optical.o Cluster_DFT_Optical.o leakdetect.o \
          Band_DFT_Col_Optical_ScaLAPACK.o Cluster_DFT_Optical_ScaLAPACK.o \
          out_openmx_lpq.o \
          Allocate_DM_for_LPQ.o \
          Cluster_DFT_Col_DMmu.o Cluster_DFT_NonCol_DMmu.o Band_DFT_Col_DMmu.o Band_DFT_NonCol_DMmu.o \
          Electric_Polarization_BandCol.o Electric_Polarization_BandNonCol.o

OBJS += openmx_mgpu_runtime.o openmx_mgpu_gemm.o openmx_magma_mgpu.o
OBJS += openmx_gpu_workers.o openmx_gpu_error.o openmx_gpu_views.o openmx_globals.o
OBJS += openmx_gpu_transform.o
OBJS += openmx_gpu_dm.o openmx_gpu_dm_collective.o
OBJS += openmx_cluster_capacity.o openmx_host_memory.o elpa_cosma_memory.o openmx_hamiltonian_cache.o
OBJS += openmx_env.o
OBJS += openmx_host_simd.o
OBJS += openmx_dense_cpu.o
OBJS += openmx_gpu_collective.o
OBJS += openmx_gpu_vector.o
OBJS += openmx_gpu_vector_worker.o
OBJS += openmx_gpu_vector_packet.o
OBJS += openmx_gpu_task.o
OBJS += openmx_gpu_task_worker.o
OBJS += openmx_gpu_task_cpu.o
OBJS += openmx_gpu_task_packet.o
OBJS += openmx_gpu_bilinear.o openmx_gpu_bilinear_pack.o

include openmx_core.mk

# These interfaces also reach legacy objects through openmx_common.h.
# Rebuild consumers when a native API mapping or backend selection changes.
OPENMX_GPU_INTERFACE_HEADERS = openmx_gpu_backend.h openmx_gpu_runtime.h \
 openmx_gpu_blas.h openmx_gpu_solver.h openmx_gpu_complex.h \
 openmx_gpu_device.h openmx_gpu_error.h openmx_gpu_views.h
$(OBJS): $(OPENMX_GPU_INTERFACE_HEADERS)

# PROG    = openmx.exe
# PROG    = openmx

# CPU elpa1/elpa2 and GPU gpusolver2 share the bundled modern ELPA library.

#
# set program name
# destination directory
#

PROG    = openmx
DESTDIR = ../work
UTIL 	= DosMain cohp jx analysis_example esp polB calB Z2FH bandgnu13 bin2txt cube2xsf intensity_map md2axsf tp kSpin BandDispersion ADenBand FermiLoop GridCalc MulPOnly MulPCalc gcube2oned cif2omx diff_gcube add_gcube diff_geo

#
# FFTW3
#

.PHONY: fftw3 fftw3-clean gemmul8-clean clean

fftw3: $(FFTW3_LIB)

$(FFTW3_SRC_ABS):
	@echo "FFTW3 source directory was not found; cloning $(FFTW3_REPO_URL) at $(FFTW3_TAG)."
	git clone --branch $(FFTW3_TAG) --depth 1 $(FFTW3_REPO_URL) "$(FFTW3_SRC_ABS)"

# The FFTW3 Git sources lack the generated codelets (and configure), which
# come from the release tarball bundled in third_party/.  It is downloaded
# only when it is missing, into a temporary file that takes the tarball's
# name only after its SHA-256 has been verified; the bundled file is verified
# as well before it is used.
$(FFTW3_CODELET_SENTINEL): | $(FFTW3_SRC_ABS)
	@echo "FFTW3 git source lacks generated codelets; overlaying the $(FFTW3_VERSION) release tarball."
	@if [ ! -f "$(FFTW3_RELEASE_TARBALL)" ]; then \
	  echo "$(FFTW3_RELEASE_TARBALL) was not found; downloading $(FFTW3_RELEASE_URL)."; \
	  mkdir -p "$(dir $(FFTW3_RELEASE_TARBALL))" && \
	  curl -L --fail "$(FFTW3_RELEASE_URL)" -o "$(FFTW3_RELEASE_TARBALL).tmp" && \
	  { [ -z "$(FFTW3_RELEASE_SHA256)" ] || \
	    printf '%s  %s\n' "$(FFTW3_RELEASE_SHA256)" "$(FFTW3_RELEASE_TARBALL).tmp" | sha256sum -c --quiet -; } && \
	  mv -f "$(FFTW3_RELEASE_TARBALL).tmp" "$(FFTW3_RELEASE_TARBALL)" || \
	  { rm -f "$(FFTW3_RELEASE_TARBALL).tmp"; echo "Could not obtain a verified $(FFTW3_RELEASE_TARBALL)."; exit 1; }; \
	fi
	@if [ -n "$(FFTW3_RELEASE_SHA256)" ]; then \
	  printf '%s  %s\n' "$(FFTW3_RELEASE_SHA256)" "$(FFTW3_RELEASE_TARBALL)" | sha256sum -c --quiet - || \
	  { echo "SHA-256 mismatch of $(FFTW3_RELEASE_TARBALL); expected $(FFTW3_RELEASE_SHA256)."; exit 1; }; \
	fi
	tmpdir=$$(mktemp -d) && \
	  { tar -xzf "$(FFTW3_RELEASE_TARBALL)" -C "$$tmpdir" && \
	    rsync -a --ignore-existing "$$tmpdir/fftw-$(FFTW3_VERSION)/" "$(FFTW3_SRC_ABS)/"; \
	    status=$$?; rm -rf "$$tmpdir"; exit $$status; }

$(FFTW3_BUILD_DIR)/config.status: $(FFTW3_CODELET_SENTINEL) | $(FFTW3_SRC_ABS)
	@if [ -f "$(FFTW3_BUILD_ABS)/Makefile" ]; then $(FFTW3_BUILD_ENV) $(MAKE) -C "$(FFTW3_BUILD_ABS)" clean; fi
	@if [ -d "$(FFTW3_SRC_ABS)/.git" ]; then \
	  tag=$$(git -C "$(FFTW3_SRC_ABS)" describe --tags --exact-match 2>/dev/null || true); \
	  if [ "$$tag" != "$(FFTW3_TAG)" ]; then \
	    echo "FFTW3_SRC_DIR must be checked out at $(FFTW3_TAG); current tag is '$${tag:-not an exact tag}'."; \
	    exit 1; \
	  fi; \
	fi
	@if [ ! -x "$(FFTW3_SRC_ABS)/configure" ]; then \
	  cd "$(FFTW3_SRC_ABS)" && touch ChangeLog && rm -rf autom4te.cache && autoreconf --verbose --install --symlink --force && rm -f config.cache; \
	fi
	@if [ -f "$(FFTW3_SRC_ABS)/config.status" ]; then \
	  $(MAKE) -C "$(FFTW3_SRC_ABS)" distclean; \
	fi
	mkdir -p "$(FFTW3_BUILD_ABS)" "$(FFTW3_PREFIX)"
	cd "$(FFTW3_BUILD_ABS)" && $(FFTW3_BUILD_ENV) CC="$(FFTW3_CC)" CXX="$(FFTW3_CXX)" FC="$(FFTW3_FC)" F77="$(FFTW3_F77)" "$(FFTW3_SRC_ABS)/configure" --build="$(FFTW3_CONFIGURE_TRIPLET)" --host="$(FFTW3_CONFIGURE_TRIPLET)" --prefix="$(FFTW3_PREFIX)" $(FFTW3_CONFIGURE_OPTIONS)

$(FFTW3_LIB): $(FFTW3_BUILD_DIR)/config.status
	$(FFTW3_BUILD_ENV) $(MAKE) -C "$(FFTW3_BUILD_ABS)"
	$(FFTW3_BUILD_ENV) $(MAKE) -C "$(FFTW3_BUILD_ABS)" install

fftw3-clean:
	rm -rf "$(FFTW3_BUILD_ABS)" "$(FFTW3_PREFIX)"

gemmul8-clean:
	rm -f "$(GEMMUL8_OPENMX_OBJ)" "$(GEMMUL8_LIB)"

FFTW3_USERS = Poisson.o Stress.o Poisson_ESM.o TRAN_Poisson.o TRAN_CDen_Main.o TRAN_Set_Electrode_Grid.o Cluster_DFT_Col.o
$(FFTW3_USERS): | $(FFTW3_LIB)

# Rebuild when switching SDK, toolkit, architecture, or bundled GEMMul8.
# Updating the file only when its contents change preserves incremental builds.
OPENMX_BUILD_CONFIG := $(CURDIR)/build/openmx-toolchain.conf
.PHONY: openmx-toolchain-force
$(OPENMX_BUILD_CONFIG): openmx-toolchain-force
	@test -n "$(GEMMUL8_BUILD_GPU_ARCH)" || { echo "GPU architecture unavailable; set GEMMUL8_GPU_ARCH (e.g. 120 for RTX 5080, 90 for H100)."; exit 1; }
	@mkdir -p "$(dir $(OPENMX_BUILD_CONFIG))"
	@printf '%s\n' '$(NVHPC_ROOT)' '$(NVHPC_CUDA_HOME)' '$(NVHPC_MATH_DIR)' '$(NVHPC_MPI_BIN)' '$(NVHPC_MPI_LIBDIR)' '$(GEMMUL8_BUILD_GPU_ARCH)' '$(GEMMUL8_COMMIT)' '$(CC)' '$(FC)' '$(NVCC)' '$(NVCC_HOST)' '$(CC2)' '$(CC3)' '$(subst $(MAGMA_LIBS),,$(LIB))' '$(GPUSOLVER2_PREFIX)' '$(GPUSOLVER2_COMPILER_BIN)' '$(GPUSOLVER2_SCALAPACK_SO)' '$(GPUSOLVER2_ELPA_VER)' > "$@.tmp"
	@cmp -s "$@.tmp" "$@" && rm -f "$@.tmp" || mv "$@.tmp" "$@"

$(OBJS) $(GEMMUL8_OPENMX_OBJ) $(FFTW3_BUILD_DIR)/config.status: $(OPENMX_BUILD_CONFIG)

# A changed FFTW compiler/optimization policy must rebuild the static library,
# but should not recompile unrelated CUDA/OpenMX objects.
FFTW3_BUILD_CONFIG := $(FFTW3_BUILD_DIR)/make-toolchain.conf
.PHONY: fftw3-toolchain-force
$(FFTW3_BUILD_CONFIG): fftw3-toolchain-force
	@mkdir -p "$(dir $(FFTW3_BUILD_CONFIG))"
	@printf '%s\n' '$(FFTW3_CC)' '$(FFTW3_CXX)' '$(FFTW3_FC)' '$(FFTW3_F77)' '$(FFTW3_CFLAGS)' '$(FFTW3_CONFIGURE_OPTIONS)' '$(FFTW3_CONFIGURE_TRIPLET)' '$(FFTW3_PREFIX)' '$(FFTW3_TAG)' > "$@.tmp"
	@cmp -s "$@.tmp" "$@" && rm -f "$@.tmp" || mv "$@.tmp" "$@"

$(FFTW3_BUILD_DIR)/config.status: $(FFTW3_BUILD_CONFIG)

# Stack-only overrides must reach the dependency builder without rebuilding
# unrelated OpenMX objects or FFTW. Keep these fields in sync with its
# toolchain.conf signature; parallelism does not affect the build products.
GPUSOLVER2_BUILD_CONFIG := $(GPUSOLVER2_BUILD)/make-toolchain.conf
.PHONY: gpusolver2-toolchain-force
$(GPUSOLVER2_BUILD_CONFIG): gpusolver2-toolchain-force
	@test -n "$(GPUSOLVER2_GPU_ARCH)" || { echo "GPU architecture unavailable; set GEMMUL8_GPU_ARCH or GPUSOLVER2_GPU_ARCH."; exit 1; }
	@mkdir -p "$(dir $(GPUSOLVER2_BUILD_CONFIG))"
	@printf '%s\n' '$(GPUSOLVER2_COMPILER_LIB)' '$(GPUSOLVER2_MPI_BIN)' '$(GPUSOLVER2_MPI_LIBDIR)' '$(GPUSOLVER2_CUDA_HOME)' '$(GPUSOLVER2_MATH_DIR)' '$(GPUSOLVER2_NVCC)' '$(GPUSOLVER2_GPU_ARCH)' '$(GPUSOLVER2_ELPA_VER)' '$(GPUSOLVER2_HOST_CC)' '$(GPUSOLVER2_HOST_CXX)' '$(GPUSOLVER2_PREFIX)' '$(GPUSOLVER2_COMPILER_BIN)' '$(GPUSOLVER2_SCALAPACK_SO)' > "$@.tmp"
	@cmp -s "$@.tmp" "$@" && rm -f "$@.tmp" || mv "$@.tmp" "$@"

$(GPUSOLVER2_STAMP): $(GPUSOLVER2_BUILD_CONFIG)

#
# OpenMX
#

.DEFAULT_GOAL := openmx

# A failed post-link scan or relink must not leave an apparently current binary.
.DELETE_ON_ERROR:

# The device initialization raises the kernel stack limit to the largest
# frame of OpenMX's own kernels (cuobjdump -res-usage, STACK), which only
# the linked binary knows: scan it after the link for the kernels of the
# objects (not those of the linked libraries, ELPA's among them, whose
# launches happen where the device has room) and, when the figure changed,
# rebuild set_cuda_default_device_from_local_rank.o with it and link again
# (the second link does not change the kernels).  Commit the stack stamp
# only after that relink succeeds, so failed builds can be retried safely.
# All of OpenMX's kernels fit the driver's default 1 KiB frame at present,
# so the step is idle.
CUOBJDUMP ?= $(NVHPC_CUDA_HOME)/bin/cuobjdump

openmx:	$(FFTW3_LIB) $(GEMMUL8_LIB) $(OBJS) force_rpath
	$(CC) $(OBJS) $(STACK) $(GEMMUL8_LIB) $(LIB) -lm -o openmx
	@set -e; \
	trap 'rm -f gpu_kernel_names.tmp gpu_kernel_res.tmp gpu_kernel_err.tmp gpu_kernel_stack.max.tmp' EXIT HUP INT TERM; \
	: > gpu_kernel_names.tmp; \
	for o in $(OBJS); do \
	  if $(CUOBJDUMP) -res-usage $$o > gpu_kernel_res.tmp 2> gpu_kernel_err.tmp; then \
	    sed -n 's/^[[:space:]]*Function \(.*\):$$/\1/p' gpu_kernel_res.tmp >> gpu_kernel_names.tmp; \
	  elif ! grep -q 'does not contain device code' gpu_kernel_err.tmp; then \
	    cat gpu_kernel_err.tmp >&2; exit 1; \
	  fi; \
	done; \
	$(CUOBJDUMP) -res-usage openmx > gpu_kernel_res.tmp; \
	max=$$(awk 'FILENAME==ARGV[1] {want[$$0]=1; next} /^[[:space:]]*Function / {f=$$2; sub(/:$$/,"",f); keep=(f in want); next} keep {for (i=1;i<=NF;i++) if ($$i ~ /^STACK:/) {split($$i,s,":"); if (s[2]+0>m) m=s[2]+0}} END {print m+0}' gpu_kernel_names.tmp gpu_kernel_res.tmp); \
	if [ "$$max" != "$$(cat gpu_kernel_stack.max 2>/dev/null)" ]; then \
	  echo "GPU kernel stack frames: up to $$max bytes per thread; relinking with that limit in the device initialization"; \
	  $(CC) -DOPENMX_GPU_KERNEL_STACK_BYTES=$$max -c set_cuda_default_device_from_local_rank.c; \
	  $(CC) $(OBJS) $(STACK) $(GEMMUL8_LIB) $(LIB) -lm -o openmx; \
	  printf '%s\n' "$$max" > gpu_kernel_stack.max.tmp; \
	  mv gpu_kernel_stack.max.tmp gpu_kernel_stack.max; \
	fi
	./force_rpath openmx

# The nvc link driver forces --enable-new-dtags, so the recorded
# library search path is DT_RUNPATH, which LD_LIBRARY_PATH overrides;
# a stray CUDA installation there (e.g. /usr/local/cuda/lib64) mixes
# cuBLAS/cuSOLVER generations and corrupts the GPU solvers.  Flip the
# tag to DT_RPATH after linking so the build-time libraries win.
force_rpath: force_rpath.c
	$(HOST_CC) -O2 -o force_rpath force_rpath.c

#
#
# all
#
#

all: $(PROG) $(UTIL)
	cp $(PROG) $(UTIL) $(DESTDIR)/

openmx.o: openmx.c openmx_common.h tran_variables.h tran_prototypes.h 
	$(CC) -c openmx.c
openmx_common.o: openmx_common.c openmx_common.h
	$(CC) -c openmx_common.c
Input_std.o: Input_std.c openmx_common.h Inputtools.h tran_prototypes.h openmx_precision.h
	$(CC) -c Input_std.c
Inputtools.o: Inputtools.c
	$(CC) -c Inputtools.c

init.o: init.c openmx_common.h
	$(CC) -c init.c

LU_inverse.o: LU_inverse.c openmx_common.h
	$(CC) -c LU_inverse.c
ReLU_inverse.o: ReLU_inverse.c openmx_common.h
	$(CC) -c ReLU_inverse.c
truncation.o: truncation.c openmx_common.h tran_prototypes.h 
	$(CC2) -c truncation.c
Find_CGrids.o: Find_CGrids.c openmx_common.h
	$(CC) -c Find_CGrids.c
readfile.o: readfile.c openmx_common.h
	$(CC) -c readfile.c
#
#
#
Hamiltonian_Cluster.o: Hamiltonian_Cluster.c openmx_common.h
	$(CC) -c Hamiltonian_Cluster.c
Hamiltonian_Cluster_Hs.o: Hamiltonian_Cluster_Hs.c openmx_common.h
	$(CC) -c Hamiltonian_Cluster_Hs.c
Hamiltonian_Cluster_NC_Hs2.o: Hamiltonian_Cluster_NC_Hs2.c openmx_common.h
	$(CC) -c Hamiltonian_Cluster_NC_Hs2.c
Hamiltonian_Band_NC_Hs2.o: Hamiltonian_Band_NC_Hs2.c openmx_common.h
	$(CC) -c Hamiltonian_Band_NC_Hs2.c
Overlap_Cluster_NC_Ss2.o: Overlap_Cluster_NC_Ss2.c openmx_common.h
	$(CC) -c Overlap_Cluster_NC_Ss2.c
Overlap_Band_NC_Ss2.o: Overlap_Band_NC_Ss2.c openmx_common.h
	$(CC) -c Overlap_Band_NC_Ss2.c
Overlap_Cluster.o: Overlap_Cluster.c openmx_common.h
	$(CC) -c Overlap_Cluster.c  
Overlap_Cluster_Ss.o: Overlap_Cluster_Ss.c openmx_common.h
	$(CC) -c Overlap_Cluster_Ss.c  
Set_ContMat_Cluster_LNO.o: Set_ContMat_Cluster_LNO.c openmx_common.h
	$(CC) -c Set_ContMat_Cluster_LNO.c
Hamiltonian_Band.o: Hamiltonian_Band.c openmx_common.h
	$(CC) -c Hamiltonian_Band.c
Overlap_Band.o: Overlap_Band.c openmx_common.h
	$(CC) -c Overlap_Band.c
Matrix_Band_LNO.o: Matrix_Band_LNO.c openmx_common.h
	$(CC) -c Matrix_Band_LNO.c
Hamiltonian_Cluster_NC.o: Hamiltonian_Cluster_NC.c openmx_common.h
	$(CC) -c Hamiltonian_Cluster_NC.c
Hamiltonian_Cluster_SO.o: Hamiltonian_Cluster_SO.c openmx_common.h
	$(CC) -c Hamiltonian_Cluster_SO.c
Hamiltonian_Band_NC.o: Hamiltonian_Band_NC.c openmx_common.h
	$(CC) -c Hamiltonian_Band_NC.c
Get_OneD_HS_Col.o: Get_OneD_HS_Col.c openmx_common.h
	$(CC) -c Get_OneD_HS_Col.c
#
#
#  
SetPara_DFT.o: SetPara_DFT.c openmx_common.h
	$(CC) -c SetPara_DFT.c
XC_Ceperly_Alder.o: XC_Ceperly_Alder.c openmx_common.h
	$(CC) -c XC_Ceperly_Alder.c
XC_CA_LSDA.o: XC_CA_LSDA.c openmx_common.h
	$(CC) -c XC_CA_LSDA.c
XC_PW92C.o: XC_PW92C.c openmx_common.h
	$(CC) -c XC_PW92C.c
XC_PBE.o: XC_PBE.c openmx_common.h
	$(CC) -c XC_PBE.c
XC_EX.o: XC_EX.c openmx_common.h
	$(CC) -c XC_EX.c
#
# SCF
#

DFT.o: DFT.c openmx_common.h tran_prototypes.h grid_precision.h
	$(CC) -c DFT.c
Cluster_DFT_Col.o: Cluster_DFT_Col.c openmx_common.h eigen_refine_gpu.h openmx_precision.h
	$(CC) -c Cluster_DFT_Col.c
Cluster_DFT_NonCol.o: Cluster_DFT_NonCol.c openmx_common.h eigen_refine_gpu.h openmx_precision.h
	$(CC) -c Cluster_DFT_NonCol.c
Cluster_DFT_Dosout.o: Cluster_DFT_Dosout.c openmx_common.h
	$(CC) -c Cluster_DFT_Dosout.c
Cluster_DFT_ON2.o: Cluster_DFT_ON2.c openmx_common.h
	$(CC) -c Cluster_DFT_ON2.c
Cluster_DFT_Col_LNO.o: Cluster_DFT_Col_LNO.c openmx_common.h
	$(CC) -c Cluster_DFT_Col_LNO.c
Cluster_DFT_Col_CWF.o: Cluster_DFT_Col_CWF.c openmx_common.h
	$(CC) -c Cluster_DFT_Col_CWF.c
Cluster_DFT_NonCol_CWF.o: Cluster_DFT_NonCol_CWF.c openmx_common.h
	$(CC) -c Cluster_DFT_NonCol_CWF.c
Cluster_DFT_Col_DMmu.o: Cluster_DFT_Col_DMmu.c openmx_common.h
	$(CC) -c Cluster_DFT_Col_DMmu.c
Cluster_DFT_NonCol_DMmu.o: Cluster_DFT_NonCol_DMmu.c openmx_common.h
	$(CC) -c Cluster_DFT_NonCol_DMmu.c
Band_DFT_Col.o: Band_DFT_Col.c openmx_common.h eigen_refine_gpu.h openmx_precision.h
	$(CC) -c Band_DFT_Col.c
Band_DFT_Col_CWF.o: Band_DFT_Col_CWF.c openmx_common.h
	$(CC) -c Band_DFT_Col_CWF.c
Band_DFT_NonCol_CWF.o: Band_DFT_NonCol_CWF.c openmx_common.h
	$(CC) -c Band_DFT_NonCol_CWF.c
Band_DFT_Col_DMmu.o: Band_DFT_Col_DMmu.c openmx_common.h
	$(CC) -c Band_DFT_Col_DMmu.c
Band_DFT_NonCol_DMmu.o: Band_DFT_NonCol_DMmu.c openmx_common.h
	$(CC) -c Band_DFT_NonCol_DMmu.c
Calc_LNAO.o: Calc_LNAO.c openmx_common.h
	$(CC) -c Calc_LNAO.c
Calc_LNBO.o: Calc_LNBO.c openmx_common.h
	$(CC) -c Calc_LNBO.c
CWF2W90.o: CWF2W90.c openmx_common.h
	$(CC) -c CWF2W90.c
out_openmx_lpq.o: out_openmx_lpq.c openmx_common.h
	$(CC) -c out_openmx_lpq.c
Allocate_DM_for_LPQ.o: Allocate_DM_for_LPQ.c openmx_common.h
	$(CC) -c Allocate_DM_for_LPQ.c
Electric_Polarization_BandCol.o: Electric_Polarization_BandCol.c openmx_common.h
	$(CC) -c Electric_Polarization_BandCol.c
Electric_Polarization_BandNonCol.o: Electric_Polarization_BandNonCol.c openmx_common.h
	$(CC) -c Electric_Polarization_BandNonCol.c
Set_XMCD_penalty.o: Set_XMCD_penalty.c openmx_common.h
	$(CC) -c Set_XMCD_penalty.c
Overlap_Cluster_LNO_Ss.o: Overlap_Cluster_LNO_Ss.c openmx_common.h
	$(CC) -c Overlap_Cluster_LNO_Ss.c
leakdetect.o: leakdetect.c leakdetect.h
	$(CC) -c leakdetect.c
Band_DFT_NonCol.o: Band_DFT_NonCol.c openmx_common.h eigen_refine_gpu.h openmx_precision.h
	$(CC3) -c Band_DFT_NonCol.c
Band_DFT_NonCol_GB.o: Band_DFT_NonCol_GB.c openmx_common.h
	$(CC) -c Band_DFT_NonCol_GB.c
Band_DFT_kpath.o: Band_DFT_kpath.c openmx_common.h
	$(CC) -c Band_DFT_kpath.c
Band_DFT_kpath_LNO.o: Band_DFT_kpath_LNO.c openmx_common.h
	$(CC) -c Band_DFT_kpath_LNO.c
Band_DFT_MO.o: Band_DFT_MO.c openmx_common.h
	$(CC) -c Band_DFT_MO.c
Unfolding_Bands.o: Unfolding_Bands.c openmx_common.h
	$(CC) -c Unfolding_Bands.c
Band_DFT_Dosout.o: Band_DFT_Dosout.c openmx_common.h
	$(CC) -c Band_DFT_Dosout.c
Mixing_DM.o: Mixing_DM.c openmx_common.h
	$(CC) -c Mixing_DM.c
Mixing_H.o: Mixing_H.c openmx_common.h openmx_mixing_policy.h openmx_core_mixing.h
	$(CC) -c Mixing_H.c
Mixing_V.o: Mixing_V.c openmx_common.h openmx_mixing_policy.h openmx_core_mixing.h
	$(CC) -c Mixing_V.c
Force.o: Force.c openmx_common.h openmx_precision.h
	$(CC) -c Force.c
Stress.o: Stress.c openmx_common.h openmx_precision.h
	$(CC) -c Stress.c
Poisson.o: Poisson.c openmx_common.h
	$(CC) -c Poisson.c
Poisson_ESM.o: Poisson_ESM.c openmx_common.h
	$(CC) -c Poisson_ESM.c
Mulliken_Charge.o: Mulliken_Charge.c openmx_common.h
	$(CC) -c Mulliken_Charge.c
# NVHPC 26.3 + -acc miscompiles LDA+U Type2/cFLL occupation/potential paths.
Occupation_Number_LDA_U.o: Occupation_Number_LDA_U.c openmx_common.h
	$(CC_NOACC) -c Occupation_Number_LDA_U.c
Eff_Hub_Pot.o: Eff_Hub_Pot.c openmx_common.h
	$(CC_NOACC) -c Eff_Hub_Pot.c
Coulomb_Interaction.o: Coulomb_Interaction.c openmx_common.h
	$(CC) -c Coulomb_Interaction.c
EulerAngle_Spin.o: EulerAngle_Spin.c openmx_common.h
	$(CC) -c EulerAngle_Spin.c
Orbital_Moment.o: Orbital_Moment.c openmx_common.h
	$(CC) -c Orbital_Moment.c
Smoothing_Func.o: Smoothing_Func.c openmx_common.h
	$(CC) -c Smoothing_Func.c
Gauss_Legendre.o: Gauss_Legendre.c openmx_common.h
	$(CC) -c Gauss_Legendre.c
# NVHPC 26.3 + -acc makes DSYGV fail here, corrupting ON2 poles.
zero_cfrac.o: zero_cfrac.c openmx_common.h
	$(CC_NOACC) -c zero_cfrac.c
xyz2spherical.o: xyz2spherical.c openmx_common.h
	$(CC) -c xyz2spherical.c
AngularF.o: AngularF.c openmx_common.h
	$(CC) -c AngularF.c
RadialF.o: RadialF.c openmx_common.h
	$(CC) -c RadialF.c
Dr_RadialF.o: Dr_RadialF.c openmx_common.h
	$(CC) -c Dr_RadialF.c
PhiF.o: PhiF.c openmx_common.h
	$(CC) -c PhiF.c
VNAF.o: VNAF.c openmx_common.h
	$(CC) -c VNAF.c
Dr_VNAF.o: Dr_VNAF.c openmx_common.h
	$(CC) -c Dr_VNAF.c
VH_AtomF.o: VH_AtomF.c openmx_common.h
	$(CC) -c VH_AtomF.c
Dr_VH_AtomF.o: Dr_VH_AtomF.c openmx_common.h
	$(CC) -c Dr_VH_AtomF.c

RF_BesselF.o: RF_BesselF.c openmx_common.h
	$(CC) -c RF_BesselF.c
Nonlocal_RadialF.o: Nonlocal_RadialF.c openmx_common.h
	$(CC) -c Nonlocal_RadialF.c

# The double-float orbital evaluator (SOG_point_eval_df, the GeForce default
# of the on-the-fly tiles) is built from error-free transformations that
# reuse each rounded product.  nvc contracts device code into FMAs by
# default, which fuses those products into some of their uses and leaves
# the tiles at FP32 accuracy (-runtest: diff Utot up to 4e-7 Hartree on an
# RTX 4070 Ti).  The explicit fmaf() calls stay fused.
Set_Orbitals_Grid.o: Set_Orbitals_Grid.c openmx_common.h orbs_grid_gpu.h openmx_precision.h
Set_Orbitals_Grid.o Set_Density_Grid_GPU.o Set_Hamiltonian.o: Makefile
Set_Orbitals_Grid.o: openmx_host_simd.h openmx_core_orbitals_simd.h ../core/include/openmx_core_orbitals_simd.h openmx_core_host_simd.h ../core/include/openmx_core_host_simd.h
	$(CC) $(NVHPC_EXPLICIT_FMA_FLAGS) -gpu=nofma -c Set_Orbitals_Grid.c
Set_Density_Grid.o: Set_Density_Grid.c openmx_common.h
	$(CC) -c Set_Density_Grid.c
Set_Density_Grid_GPU.o: Set_Density_Grid_GPU.c openmx_common.h grid_precision.h
	$(CC) $(NVHPC_EXPLICIT_FMA_FLAGS) -c Set_Density_Grid_GPU.c
Set_Aden_Grid.o: Set_Aden_Grid.c openmx_common.h
	$(CC) -c Set_Aden_Grid.c

KumoF.o: KumoF.c openmx_common.h
	$(CC) -c KumoF.c
Dr_KumoF.o: Dr_KumoF.c openmx_common.h
	$(CC) -c Dr_KumoF.c
Pot_NeutralAtom.o: Pot_NeutralAtom.c openmx_common.h
	$(CC) -c Pot_NeutralAtom.c
Simple_Mixing_DM.o: Simple_Mixing_DM.c openmx_common.h openmx_mixing_policy.h openmx_core_mixing.h
	$(CC) -c Simple_Mixing_DM.c
DIIS_Mixing_DM.o: DIIS_Mixing_DM.c openmx_common.h
	$(CC) -c DIIS_Mixing_DM.c
ADIIS_Mixing_DM.o: ADIIS_Mixing_DM.c openmx_common.h
	$(CC) -c ADIIS_Mixing_DM.c
GR_Pulay_DM.o: GR_Pulay_DM.c openmx_common.h
	$(CC) -c GR_Pulay_DM.c
Kerker_Mixing_Rhok.o: Kerker_Mixing_Rhok.c openmx_common.h openmx_mixing_policy.h openmx_core_mixing.h
	$(CC) -c Kerker_Mixing_Rhok.c
DIIS_Mixing_Rhok.o: DIIS_Mixing_Rhok.c openmx_common.h openmx_mixing_policy.h openmx_core_mixing.h
	$(CC) -c DIIS_Mixing_Rhok.c
# Same LDA+U Type2/cFLL workaround as above.
Total_Energy.o: Total_Energy.c openmx_common.h openmx_energy_workspace.h openmx_solver_workspace.h openmx_core_solver_workspace.h ../core/include/openmx_core_solver_workspace.h
	$(CC_NOACC) -c Total_Energy.c
Total_Energy_GPU.o: Total_Energy_GPU.c openmx_common.h
	$(CC) -c Total_Energy_GPU.c
Contract_Hamiltonian.o: Contract_Hamiltonian.c openmx_common.h
	$(CC) -c Contract_Hamiltonian.c
Contract_iHNL.o: Contract_iHNL.c openmx_common.h
	$(CC) -c Contract_iHNL.c
Cont_Matrix0.o: Cont_Matrix0.c openmx_common.h
	$(CC) -c Cont_Matrix0.c
Cont_Matrix1.o: Cont_Matrix1.c openmx_common.h
	$(CC) -c Cont_Matrix1.c
Cont_Matrix2.o: Cont_Matrix2.c openmx_common.h
	$(CC) -c Cont_Matrix2.c
Cont_Matrix3.o: Cont_Matrix3.c openmx_common.h
	$(CC) -c Cont_Matrix3.c
Cont_Matrix4.o: Cont_Matrix4.c openmx_common.h
	$(CC) -c Cont_Matrix4.c
Opt_Contraction.o: Opt_Contraction.c openmx_common.h
	$(CC) -c Opt_Contraction.c
Initial_CntCoes.o: Initial_CntCoes.c openmx_common.h
	$(CC) -c Initial_CntCoes.c
Initial_CntCoes2.o: Initial_CntCoes2.c openmx_common.h
	$(CC) -c Initial_CntCoes2.c


Set_XC_Grid.o: Set_XC_Grid.c openmx_common.h
	$(CC) -c Set_XC_Grid.c

Get_Orbitals.o: Get_Orbitals.c openmx_common.h
	$(CC) -c Get_Orbitals.c
Get_dOrbitals.o: Get_dOrbitals.c openmx_common.h
	$(CC) -c Get_dOrbitals.c
Get_Cnt_Orbitals.o: Get_Cnt_Orbitals.c openmx_common.h
	$(CC) -c Get_Cnt_Orbitals.c
Get_Cnt_dOrbitals.o: Get_Cnt_dOrbitals.c openmx_common.h
	$(CC) -c Get_Cnt_dOrbitals.c
Gaunt.o: Gaunt.c openmx_common.h
	$(CC) -c Gaunt.c
# NVHPC 26.3 + -acc can segfault in restart I/O under overdecomposed MPI runs.
RestartFileDFT.o: RestartFileDFT.c openmx_common.h
	$(CC_NOACC) -c RestartFileDFT.c
Output_CompTime.o: Output_CompTime.c openmx_common.h
	$(CC) -c Output_CompTime.c
Output_Energy_Decomposition.o: Output_Energy_Decomposition.c openmx_common.h
	$(CC) -c Output_Energy_Decomposition.c
Merge_LogFile.o: Merge_LogFile.c openmx_common.h
	$(CC) -c Merge_LogFile.c
Make_FracCoord.o: Make_FracCoord.c openmx_common.h
	$(CC) -c Make_FracCoord.c
Make_InputFile_with_FinalCoord.o: Make_InputFile_with_FinalCoord.c openmx_common.h
	$(CC) -c Make_InputFile_with_FinalCoord.c
#
#
#
QuickSort.o: QuickSort.c openmx_common.h
	$(CC) -c QuickSort.c
Eigen_lapack.o: Eigen_lapack.c openmx_common.h lapack_prototypes.h
	$(CC) -c Eigen_lapack.c
openmx_cusolver_compat.o: openmx_cusolver_compat.c openmx_cusolver_compat.h openmx_stdout_level.h openmx_mgpu_runtime.h
	$(CC) -c openmx_cusolver_compat.c

$(OBJS): openmx_cusolver_compat.h

gpusolver_Syevd.o: gpusolver_Syevd.c openmx_common.h
	$(CC) -c gpusolver_Syevd.c
gpusolver_Syevdx.o: gpusolver_Syevdx.c openmx_common.h
	$(CC) -c gpusolver_Syevdx.c
set_cuda_default_device_from_local_rank.o: set_cuda_default_device_from_local_rank.c set_cuda_default_device_from_local_rank.h openmx_common.h
	$(CC) -DOPENMX_GPU_KERNEL_STACK_BYTES=$$(cat gpu_kernel_stack.max 2>/dev/null || echo 0) -c set_cuda_default_device_from_local_rank.c
set_openacc_device_from_local_rank.o: set_openacc_device_from_local_rank.c openmx_common.h
	$(CC) -c set_openacc_device_from_local_rank.c
Eigen_lapack2.o: Eigen_lapack2.c openmx_common.h lapack_prototypes.h
	$(CC) -c Eigen_lapack2.c
Eigen_lapack3.o: Eigen_lapack3.c openmx_common.h lapack_prototypes.h
	$(CC) -c Eigen_lapack3.c
EigenBand_lapack.o: EigenBand_lapack.c openmx_common.h lapack_prototypes.h
	$(CC) -c EigenBand_lapack.c
Eigen_PReHH.o: Eigen_PReHH.c openmx_common.h
	$(CC) -c Eigen_PReHH.c
Eigen_PHH.o: Eigen_PHH.c openmx_common.h
	$(CC) -c Eigen_PHH.c
BroadCast_ReMatrix.o: BroadCast_ReMatrix.c openmx_common.h
	$(CC) -c BroadCast_ReMatrix.c
BroadCast_ComplexMatrix.o: BroadCast_ComplexMatrix.c openmx_common.h
	$(CC) -c BroadCast_ComplexMatrix.c
lapack_dstedc1.o: lapack_dstedc1.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstedc1.c
lapack_dstedc2.o: lapack_dstedc2.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstedc2.c
lapack_dstedc3.o: lapack_dstedc3.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstedc3.c
lapack_dstegr1.o: lapack_dstegr1.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstegr1.c
lapack_dstegr2.o: lapack_dstegr2.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstegr2.c
lapack_dstegr3.o: lapack_dstegr3.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstegr3.c
lapack_dstevx1.o: lapack_dstevx1.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstevx1.c
lapack_dstevx2.o: lapack_dstevx2.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstevx2.c
lapack_dstevx3.o: lapack_dstevx3.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstevx3.c
lapack_dstevx4.o: lapack_dstevx4.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstevx4.c
lapack_dstevx5.o: lapack_dstevx5.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstevx5.c
lapack_dsteqr1.o: lapack_dsteqr1.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dsteqr1.c
Nonlocal_Basis.o: Nonlocal_Basis.c openmx_common.h
	$(CC) -c Nonlocal_Basis.c
Set_OLP_Kin.o: Set_OLP_Kin.c openmx_common.h openmx_gpu_task.h openmx_core_overlap.h openmx_core_pair_bessel.h openmx_complex.h openmx_core_orbitals.h
Set_OLP_Kin.o: openmx_host_simd.h openmx_gpu_task_layout.h ../core/include/openmx_core_host_simd.h openmx_core_vna_simd.h ../core/include/openmx_core_vna_simd.h
	$(CC) -c Set_OLP_Kin.c
Set_Nonlocal.o: Set_Nonlocal.c openmx_common.h openmx_core_pair_bessel.h openmx_nonlocal_cpu.h openmx_core_nonlocal.h openmx_core_orbitals.h
Set_Nonlocal.o: openmx_host_simd.h ../core/include/openmx_core_host_simd.h openmx_core_vna_simd.h ../core/include/openmx_core_vna_simd.h
	$(CC) -c Set_Nonlocal.c
Set_ProExpn_VNA.o: Set_ProExpn_VNA.c openmx_common.h openmx_precision.h openmx_core_vna.h openmx_core_orbitals.h openmx_vna_cpu.h openmx_vna_gpu_context.h openmx_core_hvna.h openmx_core_hvna_workspace.h openmx_hvna_cpu.h openmx_complex.h openmx_core_vna_preparation.h openmx_vna_preparation.h openmx_core_dsvna.h openmx_dsvna_cpu.h
	$(CC) -c Set_ProExpn_VNA.c
Set_ProExpn_VNA.o: openmx_mgpu_runtime.h
Force.o: openmx_mgpu_runtime.h
Set_CoreHoleMatrix.o: Set_CoreHoleMatrix.c openmx_common.h
	$(CC) -c Set_CoreHoleMatrix.c
Set_OLP_p.o: Set_OLP_p.c openmx_common.h 
	$(CC) -c Set_OLP_p.c
Set_Hamiltonian.o: Set_Hamiltonian.c openmx_common.h grid_precision.h orbs_grid_gpu.h
	$(CC) $(NVHPC_EXPLICIT_FMA_FLAGS) -c Set_Hamiltonian.c
set_hamiltonian_gpu.o: set_hamiltonian_gpu.cu openmx_stdout_level.h
	$(NVCC_GEMMUL8_ENV) $(NVCC) $(NVCC_GEMMUL8_FLAGS) -c set_hamiltonian_gpu.cu -o set_hamiltonian_gpu.o
Set_Vpot.o: Set_Vpot.c openmx_common.h
	$(CC) -c Set_Vpot.c
#
#
#
FT_PAO.o: FT_PAO.c openmx_common.h
	$(CC) -c FT_PAO.c
FT_NLP.o: FT_NLP.c openmx_common.h
	$(CC) -c FT_NLP.c
FT_ProExpn_VNA.o: FT_ProExpn_VNA.c openmx_common.h
	$(CC) -c FT_ProExpn_VNA.c
FT_VNA.o: FT_VNA.c openmx_common.h
	$(CC) -c FT_VNA.c
FT_ProductPAO.o: FT_ProductPAO.c openmx_common.h
	$(CC) -c FT_ProductPAO.c
#
#
#
Divide_Conquer.o: Divide_Conquer.c openmx_common.h
	$(CC3) -c Divide_Conquer.c
Divide_Conquer_LNO.o: Divide_Conquer_LNO.c openmx_common.h
	$(CC) -c Divide_Conquer_LNO.c
eigen_refine_gpu.o: eigen_refine_gpu.c eigen_refine_gpu.h openmx_stdout_level.h openmx_precision.h openmx_native_resources.h openmx_core_native_resources.h
	$(CC) -c eigen_refine_gpu.c
grid_precision.o: grid_precision.c grid_precision.h openmx_precision.h
	$(CC) -c grid_precision.c
openmx_precision.o: openmx_precision.c openmx_precision.h
	$(CC) -c openmx_precision.c

gemmul8_bridge.o: gemmul8_bridge.cu openmx_gpu_memory.hpp openmx_common.h openmx_stdout_level.h openmx_mgpu_runtime.h $(GEMMUL8_DIR)/include/gemmul8.hpp $(GEMMUL8_LIB)
	$(NVCC_GEMMUL8_ENV) $(NVCC) $(NVCC_GEMMUL8_FLAGS) -c gemmul8_bridge.cu -o gemmul8_bridge.o

Input_std.o Cluster_DFT_Col.o Cluster_DFT_NonCol.o Band_DFT_Col.o Band_DFT_NonCol.o: openmx_mgpu_runtime.h
Input_std.o set_cuda_default_device_from_local_rank.o set_openacc_device_from_local_rank.o \
Set_XC_Grid.o Total_Energy_GPU.o Set_Orbitals_Grid.o Set_Nonlocal.o Set_ProExpn_VNA.o \
Force.o truncation.o Set_OLP_Kin.o Mixing_H.o Set_Density_Grid_GPU.o Set_Hamiltonian.o \
set_hamiltonian_gpu.o: openmx_gpu_workers.h openmx_gpu_views.h
openmx_gpu_workers.o: openmx_gpu_workers.c openmx_gpu_workers.h openmx_mgpu_runtime.h set_cuda_default_device_from_local_rank.h
	$(CC) -c $< -o $@

openmx_globals.o: openmx_globals.c openmx_common.h
	$(CC_NOACC) -c openmx_globals.c -o $@
openmx_gpu_views.o: openmx_gpu_views.cpp openmx_gpu_views.h openmx_gpu_memory.hpp openmx_gpu_runtime.h openmx_env.h
	$(GPUSOLVER2_HOST_CXX) -O2 -std=c++17 -fopenmp -I$(NVHPC_CUDA_HOME)/include -c $< -o $@
Cluster_DFT_Col.o Cluster_DFT_NonCol.o Band_DFT_Col.o Band_DFT_NonCol.o: openmx_gpu_transform.h
Cluster_DFT_Col.o Cluster_DFT_NonCol.o Band_DFT_Col.o Band_DFT_NonCol.o: openmx_gpu_dm.h
openmx_gpu_transform.o: openmx_gpu_transform.cu openmx_gpu_transform.h openmx_gpu_workers.h openmx_gpu_views.h openmx_gpu_memory.hpp
	$(NVCC_GEMMUL8_ENV) $(NVCC) $(NVCC_GEMMUL8_FLAGS) -c $< -o $@
openmx_gpu_dm.o: openmx_gpu_dm.cu openmx_gpu_dm_entry.h openmx_gpu_dm.h openmx_gpu_workers.h openmx_gpu_views.h openmx_gpu_memory.hpp
	$(NVCC_GEMMUL8_ENV) $(NVCC) $(NVCC_GEMMUL8_FLAGS) -c $< -o $@
Input_std.o Band_DFT_Col.o: openmx_mgpu_band_plan.h
Cluster_DFT_NonCol.o Band_DFT_NonCol.o: openmx_mgpu_density.h
openmx_mgpu_runtime.o: openmx_mgpu_runtime.cu openmx_mgpu_runtime.h openmx_mgpu_gemm.h openmx_magma_mgpu.h openmx_stdout_level.h
	$(NVCC_GEMMUL8_ENV) $(NVCC) $(NVCC_GEMMUL8_FLAGS) -Xcompiler -pthread -c $< -o $@
openmx_mgpu_gemm.o: openmx_mgpu_gemm.cu openmx_mgpu_gemm.h
	$(NVCC_GEMMUL8_ENV) $(NVCC) $(NVCC_GEMMUL8_FLAGS) -Xcompiler -pthread,-fopenmp -c $< -o $@

# Unlike the global toolchain signature this also records a switch between
# the stub and MAGMA (WITH_MAGMA=auto decides per build node), and reports
# the decision whenever it changes.
MAGMA_ADAPTER_CONFIG := $(CURDIR)/build/magma-adapter.conf
MAGMA_DECISION = $(if $(filter 1,$(MAGMA_ENABLED)),MAGMA: linking $(MAGMA_LIBRARY)$(if $(filter auto,$(WITH_MAGMA)), ($(if $(MAGMA_AUTO_BUILD),$(MAGMA_NODE_GPUS) GPUs on this node,installed by an earlier build))),MAGMA: not linked ($(if $(filter auto,$(WITH_MAGMA)),$(MAGMA_NODE_GPUS) GPU(s) on this node; WITH_MAGMA=1 builds it for multi-GPU nodes,WITH_MAGMA=0)); dense solves that do not fit on one GPU recover on the CPU)
$(MAGMA_ADAPTER_CONFIG): openmx-toolchain-force
	@mkdir -p "$(dir $@)"
	@printf '%s\n' '$(MAGMA_ENABLED)' '$(MAGMA_PREFIX)' '$(MAGMA_CPPFLAGS)' > "$@.tmp"
	@cmp -s "$@.tmp" "$@" && rm -f "$@.tmp" || { mv "$@.tmp" "$@"; echo "$(MAGMA_DECISION)"; }
openmx_magma_mgpu.o: openmx_magma_mgpu.cpp openmx_magma_mgpu.h openmx_magma_refine.h openmx_mgpu_gemm.h third_party/magma_device_map.h $(MAGMA_ADAPTER_CONFIG) $(MAGMA_DEPENDENCY)
	$(NVCC_GEMMUL8_ENV) $(NVCC) $(NVCC_GEMMUL8_FLAGS) $(MAGMA_CPPFLAGS) -Xcompiler -pthread,-fopenmp -c $< -o $@
.PHONY: magma-stack
magma-stack: $(MAGMA_LIBRARY)
# WITH_MAGMA=1 and the explicit magma-stack target build MAGMA and rebuild
# it when its toolchain, build script or patch changes.  WITH_MAGMA=auto on
# a build node with two or more GPUs builds it only when it is missing: an
# existing installation (which may be external and read-only) is used as
# it is.  Otherwise auto only consumes an installed library.
ifneq ($(strip $(filter 1,$(WITH_MAGMA)) $(filter magma-stack,$(MAKECMDGOALS))),)
MAGMA_BUILD_CONFIG := $(MAGMA_BUILD)/make-toolchain.conf
$(MAGMA_BUILD_CONFIG): openmx-toolchain-force
	@mkdir -p "$(dir $@)"
	@printf '%s\n' '$(NVHPC_CUDA_HOME)' '$(NVHPC_MATH_DIR)' '$(GEMMUL8_BUILD_GPU_ARCH)' '$(MAGMA_HOST_CC)' '$(MAGMA_HOST_CXX)' '$(MAGMA_HOST_LIBDIR)' '$(MAGMA_PREFIX)' '$(MAGMA_LAPACK_LIBRARIES)' > "$@.tmp"
	@cmp -s "$@.tmp" "$@" && rm -f "$@.tmp" || mv "$@.tmp" "$@"
MAGMA_LIBRARY_PREREQS = third_party/build_magma.sh third_party/magma-device-map.patch third_party/magma_device_map.h $(MAGMA_BUILD_CONFIG)
endif
ifneq ($(strip $(filter 1,$(WITH_MAGMA)) $(MAGMA_AUTO_BUILD) $(filter magma-stack,$(MAKECMDGOALS))),)
$(MAGMA_LIBRARY): $(MAGMA_LIBRARY_PREREQS)
	MAGMA_BUILD="$(MAGMA_BUILD)" MAGMA_PREFIX="$(MAGMA_PREFIX)" \
	MAGMA_ARCHIVE="$(MAGMA_ARCHIVE)" \
	MAGMA_CUDA_HOME="$(NVHPC_CUDA_HOME)" MAGMA_MATH_DIR="$(NVHPC_MATH_DIR)" \
	MAGMA_GPU_ARCH="$(GEMMUL8_BUILD_GPU_ARCH)" MAGMA_CC="$(MAGMA_HOST_CC)" MAGMA_CXX="$(MAGMA_HOST_CXX)" \
	MAGMA_HOST_LIBDIR="$(MAGMA_HOST_LIBDIR)" \
	MAGMA_CMAKE="$(MAGMA_CMAKE)" MAGMA_JOBS="$(MAGMA_JOBS)" \
	MAGMA_LAPACK_LIBRARIES="$(MAGMA_LAPACK_LIBRARIES)" MAGMA_RUNTIME_DIRS="$(NVHPC_COMPILER_LIB)" \
	sh third_party/build_magma.sh
endif

.PHONY: gpusolver2-stack gpusolver2-clean
gpusolver2-stack: $(GPUSOLVER2_STAMP)

$(GPUSOLVER2_STAMP): third_party/build_gpusolver2_stack.sh
	GPUSOLVER2_DIST="$(GPUSOLVER2_DIST)" GPUSOLVER2_BUILD="$(GPUSOLVER2_BUILD)" \
	GPUSOLVER2_PREFIX="$(GPUSOLVER2_PREFIX)" \
	GPUSOLVER2_HOST_CC="$(GPUSOLVER2_HOST_CC)" GPUSOLVER2_HOST_CXX="$(GPUSOLVER2_HOST_CXX)" \
	GPUSOLVER2_NVCC="$(GPUSOLVER2_NVCC)" GPUSOLVER2_GPU_ARCH="$(GPUSOLVER2_GPU_ARCH)" \
	GPUSOLVER2_MPI_BIN="$(GPUSOLVER2_MPI_BIN)" GPUSOLVER2_MPI_LIBDIR="$(GPUSOLVER2_MPI_LIBDIR)" \
	GPUSOLVER2_CUDA_HOME="$(GPUSOLVER2_CUDA_HOME)" GPUSOLVER2_MATH_DIR="$(GPUSOLVER2_MATH_DIR)" \
	GPUSOLVER2_COMPILER_LIB="$(GPUSOLVER2_COMPILER_LIB)" GPUSOLVER2_COMPILER_BIN="$(GPUSOLVER2_COMPILER_BIN)" \
	GPUSOLVER2_SCALAPACK_SO="$(GPUSOLVER2_SCALAPACK_SO)" \
	GPUSOLVER2_ELPA_VER="$(GPUSOLVER2_ELPA_VER)" \
	GPUSOLVER2_CMAKE="$(GPUSOLVER2_CMAKE)" GPUSOLVER2_JOBS="$(GPUSOLVER2_JOBS)" \
	sh third_party/build_gpusolver2_stack.sh
	@touch "$@"

gpusolver2-clean:
	rm -rf "$(GPUSOLVER2_BUILD)" "$(GPUSOLVER2_PREFIX)"

openmx_elpa_cpu.o: openmx_elpa_cpu.c openmx_elpa_cpu.h openmx_common.h $(GPUSOLVER2_STAMP)
	$(CC_NOACC_O3) -I$(GPUSOLVER2_ELPA_INCLUDE) -c openmx_elpa_cpu.c

elpa_cosma_bridge.o: elpa_cosma_bridge.c elpa_cosma_bridge.h openmx_stdout_level.h openmx_host_memory.h $(GPUSOLVER2_STAMP)
	$(CC_NOACC_O3) -I$(GPUSOLVER2_ELPA_INCLUDE) -c elpa_cosma_bridge.c
# COSMA's strategy (host memory of its transforms) with the C++ compiler
# of the COSMA build; mpi.h only
elpa_cosma_memory.o: elpa_cosma_memory.cpp $(GPUSOLVER2_STAMP)
	$(GPUSOLVER2_HOST_CXX) -std=c++17 -O2 -g -I$(GPUSOLVER2_PREFIX)/include $(GPUSOLVER2_MPI_INCLUDES) -c elpa_cosma_memory.cpp -o elpa_cosma_memory.o
openmx_host_memory.o: openmx_host_memory.c openmx_host_memory.h
	$(CC_NOACC_O3) -c openmx_host_memory.c
# The environment settings of the run (every reader includes openmx_env.h).
openmx_env.o: openmx_env.c openmx_env.h openmx_env_registry.h
	$(CC_NOACC_O3) -c openmx_env.c
openmx_host_simd.o: openmx_host_simd.c openmx_host_simd.h openmx_env.h openmx_env_registry.h openmx_core_host_simd.h ../core/include/openmx_core_host_simd.h openmx_core_grid.h ../core/include/openmx_core_grid.h openmx_gpu_device.h set_cuda_default_device_from_local_rank.h
	$(CC_NOACC_O3) -c openmx_host_simd.c
$(OBJS): openmx_env.h openmx_env_registry.h
openmx_cluster_capacity.o: openmx_cluster_capacity.c openmx_cluster_capacity.h openmx_mgpu_runtime.h elpa_cosma_bridge.h openmx_stdout_level.h openmx_host_memory.h
	$(CC_NOACC_O3) -c openmx_cluster_capacity.c
openmx_dense_cpu.o: openmx_dense_cpu.c openmx_dense_cpu.h
	$(CC_NOACC_O3) -c openmx_dense_cpu.c
openmx_gpu_collective.o: openmx_gpu_collective.c openmx_gpu_collective.h openmx_gpu_workers.h openmx_stdout_level.h
	$(CC_NOACC_O3) -c openmx_gpu_collective.c
openmx_gpu_vector.o: openmx_gpu_vector.c openmx_gpu_vector.h openmx_gpu_vector_worker.h openmx_gpu_vector_packet.h openmx_gpu_collective.h openmx_gpu_workers.h
	$(CC) -c openmx_gpu_vector.c
openmx_gpu_vector_worker.o: openmx_gpu_vector_worker.cpp openmx_gpu_vector_worker.h openmx_gpu_vector.h openmx_gpu_workers.h openmx_gpu_memory.hpp openmx_host_buffer.hpp
	$(NVHPC_COMPILER_BIN)/nvc++ -O3 $(filter-out -std=gnu11 --diag_error=implicit_func_decl,$(COMMON_CFLAGS)) -std=c++17 $(shell $(MPICC) --showme:compile) -c $< -o $@
openmx_gpu_vector_packet.o: openmx_gpu_vector_packet.cpp openmx_gpu_vector_packet.h openmx_gpu_vector_worker.h openmx_gpu_vector.h openmx_host_buffer.hpp
	$(GPUSOLVER2_HOST_CXX) -O3 -std=c++17 $(shell $(MPICC) --showme:compile) -c $< -o $@

openmx_gpu_task.o: openmx_gpu_task.c openmx_gpu_task.h openmx_gpu_task_layout.h openmx_gpu_task_cpu.h openmx_gpu_task_packet.h openmx_gpu_collective.h openmx_gpu_workers.h
	$(CC) -c openmx_gpu_task.c
openmx_gpu_task_cpu.o: openmx_gpu_task_cpu.cpp openmx_gpu_task_cpu.h openmx_gpu_task.h openmx_gpu_task_layout.h openmx_host_buffer.hpp
	$(GPUSOLVER2_HOST_CXX) -O3 -std=c++17 $(shell $(MPICC) --showme:compile) -c $< -o $@
openmx_gpu_task_packet.o: openmx_gpu_task_packet.cpp openmx_gpu_task_packet.h openmx_gpu_task.h openmx_gpu_task_layout.h openmx_host_buffer.hpp
	$(GPUSOLVER2_HOST_CXX) -O3 -std=c++17 $(shell $(MPICC) --showme:compile) -c $< -o $@
openmx_gpu_task_worker.o: openmx_gpu_task_worker.cpp openmx_gpu_task.h openmx_gpu_task_layout.h openmx_gpu_memory.hpp openmx_host_buffer.hpp openmx_gpu_workers.h
	$(GPUSOLVER2_HOST_CXX) -O3 -std=c++17 $(shell $(MPICC) --showme:compile) -I$(NVHPC_CUDA_HOME)/include -c $< -o $@
openmx_gpu_bilinear.o: openmx_gpu_bilinear.c openmx_gpu_bilinear.h openmx_gpu_collective.h openmx_gpu_workers.h
	$(CC) -c openmx_gpu_bilinear.c
openmx_gpu_bilinear_pack.o: openmx_gpu_bilinear_pack.cpp openmx_gpu_bilinear.h
	$(GPUSOLVER2_HOST_CXX) -std=c++17 -O2 $(GPUSOLVER2_MPI_INCLUDES) -c openmx_gpu_bilinear_pack.cpp -o openmx_gpu_bilinear_pack.o
openmx_mgpu_runtime.o: openmx_dense_cpu.h
Cluster_DFT_Col.o Cluster_DFT_NonCol.o: openmx_cluster_capacity.h elpa_cosma_bridge.h
$(GEMMUL8_DIR)/Makefile:
	mkdir -p $(GEMMUL8_REPO_DIR)
	cd $(GEMMUL8_REPO_DIR) && \
	  { [ -d .git ] || git init -q .; } && \
	  { git remote get-url origin >/dev/null 2>&1 || git remote add origin $(GEMMUL8_REPO_URL); } && \
	  git fetch --depth 1 origin $(GEMMUL8_COMMIT) && \
	  git checkout -q $(GEMMUL8_COMMIT)
# The header comes from the checkout above; the empty recipe gives it a rule
# so a fresh clone of this repository builds with a single make invocation.
# Order-only: an existing header is always up to date regardless of mtimes.
$(GEMMUL8_DIR)/include/gemmul8.hpp: | $(GEMMUL8_DIR)/Makefile
	@:
# GEMMul8's implementation includes nested oz2 and precision headers.
openmx_rwildcard = $(foreach d,$(wildcard $1*),$(call openmx_rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))
GEMMUL8_SOURCES = $(call openmx_rwildcard,$(GEMMUL8_DIR)/include/,*.hpp) $(call openmx_rwildcard,$(GEMMUL8_DIR)/src/,*.hpp *.cu)
$(GEMMUL8_OPENMX_OBJ): gemmul8_openmx.cu $(GEMMUL8_DIR)/Makefile $(GEMMUL8_SOURCES)
	$(NVCC_GEMMUL8_ENV) $(NVCC) $(NVCC_GEMMUL8_FLAGS) -c gemmul8_openmx.cu -o $(GEMMUL8_OPENMX_OBJ)
$(GEMMUL8_LIB): $(GEMMUL8_OPENMX_OBJ)
	mkdir -p $(GEMMUL8_DIR)/lib
	ar rcs $(GEMMUL8_LIB) $(GEMMUL8_OPENMX_OBJ)
Divide_Conquer_Dosout.o: Divide_Conquer_Dosout.c openmx_common.h
	$(CC) -c Divide_Conquer_Dosout.c
Krylov.o: Krylov.c openmx_common.h
	$(CC_NOACC_O3) -c Krylov.c
EGAC_DFT.o: EGAC_DFT.c openmx_common.h
	$(CC) -c EGAC_DFT.c
LNO.o: LNO.c openmx_common.h
	$(CC) -c LNO.c
#
#
#
# NVHPC 26.3 + -acc can segfault in the MD control path after SCF.
MD_pac.o: MD_pac.c openmx_common.h lapack_prototypes.h
	$(CC_NOACC) -c MD_pac.c
#
#
#
iterout.o: iterout.c openmx_common.h
	$(CC) -c iterout.c
iterout_md.o: iterout_md.c openmx_common.h
	$(CC) -c iterout_md.c
Allocate_Arrays.o: Allocate_Arrays.c openmx_common.h
	$(CC) -c Allocate_Arrays.c
Free_Arrays.o: Free_Arrays.c openmx_common.h
	$(CC) -c Free_Arrays.c
Init_List_YOUSO.o: Init_List_YOUSO.c openmx_common.h
	$(CC) -c Init_List_YOUSO.c
outputfile1.o: outputfile1.c openmx_common.h
	$(CC) -c outputfile1.c
malloc_multidimarray.o: malloc_multidimarray.c
	$(CC) -c malloc_multidimarray.c 
PrintMemory.o: PrintMemory.c
	$(CC) -c PrintMemory.c 
PrintMemory_Fix.o: PrintMemory_Fix.c openmx_common.h
	$(CC) -c PrintMemory_Fix.c 
dtime.o: dtime.c
	$(CC) -c dtime.c 
OutData.o: OutData.c openmx_common.h
	$(CC2) -c OutData.c
OutData_Binary.o: OutData_Binary.c openmx_common.h
	$(CC) -c OutData_Binary.c
init_alloc_first.o: init_alloc_first.c openmx_common.h
	$(CC) -c init_alloc_first.c
File_CntCoes.o: File_CntCoes.c openmx_common.h
	$(CC) -c File_CntCoes.c
SCF2File.o: SCF2File.c openmx_common.h
	$(CC) -c SCF2File.c
Cutoff.o: Cutoff.c openmx_common.h
	$(CC) -c Cutoff.c
Voronoi_Charge.o: Voronoi_Charge.c openmx_common.h
	$(CC) -c Voronoi_Charge.c
Voronoi_Orbital_Moment.o: Voronoi_Orbital_Moment.c openmx_common.h
	$(CC) -c Voronoi_Orbital_Moment.c
Fuzzy_Weight.o: Fuzzy_Weight.c openmx_common.h
	$(CC) -c Fuzzy_Weight.c
dampingF.o: dampingF.c openmx_common.h
	$(CC) -c dampingF.c
deri_dampingF.o: deri_dampingF.c openmx_common.h
	$(CC) -c deri_dampingF.c
Spherical_Bessel.o: Spherical_Bessel.c openmx_common.h
	$(CC) -c Spherical_Bessel.c
Generating_MP_Special_Kpt.o: Generating_MP_Special_Kpt.c openmx_common.h
	$(CC) -c Generating_MP_Special_Kpt.c
Generate_Wannier.o: Generate_Wannier.c openmx_common.h
	$(CC_NOACC) -c Generate_Wannier.c
DFTDvdW_init.o: DFTDvdW_init.c openmx_common.h
	$(CC) -c DFTDvdW_init.c
DFTD3vdW_init.o: DFTD3vdW_init.c openmx_common.h
	$(CC_DFTD3) -c DFTD3vdW_init.c
neb.o:	neb.c openmx_common.h Inputtools.h lapack_prototypes.h
	$(CC) -c neb.c
neb_run.o: neb_run.c openmx_common.h
	$(CC) -c neb_run.c
neb_check.o: neb_check.c openmx_common.h Inputtools.h 
	$(CC) -c neb_check.c
NBO_Cluster.o: NBO_Cluster.c openmx_common.h Inputtools.h
	$(CC) -c NBO_Cluster.c
NBO_Krylov.o: NBO_Krylov.c openmx_common.h Inputtools.h
	$(CC) -c NBO_Krylov.c
Population_Analysis_Wannier.o: Population_Analysis_Wannier.c openmx_common.h Inputtools.h
	$(CC) -c Population_Analysis_Wannier.c
Population_Analysis_Wannier2.o: Population_Analysis_Wannier2.c openmx_common.h Inputtools.h
	$(CC) -c Population_Analysis_Wannier2.c

#
# codes for calculating optical conductivities and dielectric functions developed by YTL
#

NabraMatrixElements.o: NabraMatrixElements.c openmx_common.h
	$(CC) -c NabraMatrixElements.c
Set_dOrbitals_Grid.o: Set_dOrbitals_Grid.c openmx_common.h
	$(CC) -c Set_dOrbitals_Grid.c
Calc_optical.o: Calc_optical.c openmx_common.h
	$(CC) -c Calc_optical.c
Band_DFT_NonCol_Optical.o: Band_DFT_NonCol_Optical.c openmx_common.h
	$(CC) -c Band_DFT_NonCol_Optical.c
Cluster_DFT_Optical.o: Cluster_DFT_Optical.c openmx_common.h
	$(CC) -c Cluster_DFT_Optical.c
Band_DFT_Col_Optical_ScaLAPACK.o: Band_DFT_Col_Optical_ScaLAPACK.c openmx_common.h
	$(CC) -c Band_DFT_Col_Optical_ScaLAPACK.c
Cluster_DFT_Optical_ScaLAPACK.o: Cluster_DFT_Optical_ScaLAPACK.c openmx_common.h
	$(CC) -c Cluster_DFT_Optical_ScaLAPACK.c

#
#
#
mimic_sse.o: mimic_sse.c mimic_sse.h
	$(CC) -c mimic_sse.c
Make_Comm_Worlds.o: Make_Comm_Worlds.c
	$(CC) -c Make_Comm_Worlds.c
Set_Allocate_Atom2CPU.o: Set_Allocate_Atom2CPU.c openmx_common.h
	$(CC) -c Set_Allocate_Atom2CPU.c

#
#
# Maketest, Runtest, Memory_Leak_test, Force_test, Show_DFT_DATA
#
#

Maketest.o: Maketest.c openmx_common.h Inputtools.h
	$(CC) -c Maketest.c
Runtest.o: Runtest.c openmx_common.h Inputtools.h
	$(CC) -c Runtest.c
Memory_Leak_test.o: Memory_Leak_test.c openmx_common.h Inputtools.h
	$(CC) -c Memory_Leak_test.c
Force_test.o: Force_test.c openmx_common.h Inputtools.h
	$(CC) -c Force_test.c
Stress_test.o: Stress_test.c openmx_common.h Inputtools.h
	$(CC) -c Stress_test.c
Show_DFT_DATA.o: Show_DFT_DATA.c openmx_common.h Inputtools.h
	$(CC) -c Show_DFT_DATA.c

#
# install
#
#

install: $(PROG)
#	strip $(PROG)
	cp $(PROG) $(DESTDIR)/$(PROG)

#
#
# clean executable and object files 
#
#

clean: fftw3-clean gemmul8-clean
	rm -f $(PROG) $(OBJS) $(OBJS_jx) $(UTIL) force_rpath *.o *.mod *.dbg gpu_kernel_stack.max

#
#
# programs for generating DOS from files *.Dos.val and *.Dos.vec
#
#

DosMain: DosMain.o Inputtools.o malloc_multidimarray.o Tetrahedron_Blochl.o 
	$(CC) -o $@ DosMain.o Inputtools.o malloc_multidimarray.o Tetrahedron_Blochl.o -lm 
	cp DosMain $(DESTDIR)/DosMain

DosMain.o :DosMain.c openmx_common.h
	$(CC) -o $@ -c DosMain.c
Tetrahedron_Blochl.o : Tetrahedron_Blochl.c
	$(CC) -o $@ -c Tetrahedron_Blochl.c 

#
#  COHP: Crystal Orbital Hamilton Population
#

cohp: cohp.o Inputtools.o Tetrahedron_Blochl.o
	$(CC) -o $@ cohp.o Inputtools.o Tetrahedron_Blochl.o -lm
	cp cohp $(DESTDIR)/cohp

cohp.o :cohp.c openmx_common.h
	$(CC) -o $@ -c cohp.c

#
#
#  exchange interaction coupling constant J between two atoms
#
#

OBJ_jx = Inputtools.o read_scfout.o jx_quicksort.o jx_LNO.o jx_config.o \
	jx_tools.o jx_cluster.o jx_band_psum.o jx_band_indiv.o jx.o

jx_quicksort.o: jx_quicksort.c
	$(CC) -c jx_quicksort.c

jx_LNO.o: jx_LNO.c jx_tools.h jx_LNO.h read_scfout.h
	$(CC) -c jx_LNO.c

jx_config.o: jx_config.c jx_config.h
	$(CC) -c jx_config.c

jx_tools.o: jx_tools.c jx_tools.h jx_total_mem.h
	$(CC) -c jx_tools.c

jx_cluster.o: jx_cluster.c jx_tools.c jx.h jx_tools.h jx_total_mem.h
	$(CC) -c jx_cluster.c

jx_band_psum.o: jx_band_psum.c jx_tools.c jx.h jx_tools.h jx_total_mem.h
	$(CC) -c jx_band_psum.c

jx_band_indiv.o: jx_band_indiv.c jx_tools.c jx.h jx_tools.h jx_total_mem.h
	$(CC) -c jx_band_indiv.c

jx.o: jx.c jx_tools.c read_scfout.h jx_tools.h jx_total_mem.h jx.h
	$(CC) -c jx.c

jx: $(OBJ_jx)
	$(CC) $(OBJ_jx) $(LIB) -lm -o jx
	cp jx $(DESTDIR)/jx

#
#
#  transition probability between two states 
#
#

tp: tp.o read_scfout.o
	$(CC) tp.o read_scfout.o $(LIB) -lm -o tp
	cp tp $(DESTDIR)/tp

tp.o: tp.c read_scfout.h 
	$(CC) -c tp.c

#
#
# analysis_example
#
#

analysis_example: analysis_example.o read_scfout.o
	$(CC) analysis_example.o read_scfout.o $(LIB)  -lm -o analysis_example
	cp analysis_example $(DESTDIR)/analysis_example

analysis_example.o: analysis_example.c read_scfout.h 
	$(CC) -c analysis_example.c

read_scfout.o: read_scfout.c read_scfout.h 
	$(CC) -c read_scfout.c

#
#
# program for generating EPS from files *.out and *.vhart
#
#

OBJS_ESP  = esp.o Inputtools.o
esp:	$(OBJS_ESP)
	$(CC) $(OBJS_ESP) $(LIB) -lm -o $@
	cp esp $(DESTDIR)/esp
esp.o : esp.c Inputtools.h
	$(CC) -o $@ -c esp.c

#
#
# check_lead
#
#

check_lead: check_lead.o Inputtools.o
	$(CC) check_lead.o Inputtools.o -lm -o check_lead
	cp check_lead $(DESTDIR)/check_lead

check_lead.o: check_lead.c Inputtools.h 
	$(CC) -c check_lead.c

#
#
#  optical conductivity 
#
#

OpticalConductivityMain: OpticalConductivityMain.o \
              Inputtools.o  malloc_multidimarray.o
	$(CC) -o $@   OpticalConductivityMain.o  Inputtools.o  malloc_multidimarray.o -lm 
	cp OpticalConductivityMain $(DESTDIR)/OpticalConductivityMain

#
#
#  electric polarization using Berry's phase
#
#

OBJS_polB = polB.o read_scfout.o
polB:	$(OBJS_polB)
	$(CC) $(OBJS_polB) $(LIB) -lm -o polB
	cp polB $(DESTDIR)/polB

polB.o: polB.c read_scfout.h 
	$(CC) -c polB.c

#
#
# Code for calculating Berry Curvature and Chern Number
#
#

OBJS_calB = calB.o read_scfout.o
calB:	$(OBJS_calB)
	$(CC) $(OBJS_calB) $(LIB) -lm -o calB
	cp calB $(DESTDIR)/calB

calB.o: calB.c read_scfout.h 
	$(CC) -c calB.c

#
#
# Code for calculating Z2 invariant by Fukui-Hatsugai Method
#
#

OBJS_Z2FH = Z2FH.o read_scfout.o
Z2FH:	$(OBJS_Z2FH)
	$(CC) $(OBJS_Z2FH) $(LIB) -lm -o Z2FH
	cp Z2FH $(DESTDIR)/Z2FH

Z2FH.o: Z2FH.c read_scfout.h 
	$(CC) -c Z2FH.c

#
#
# example_mpi_spawn
#
#

example_mpi_spawn: example_mpi_spawn.o
	$(CC) example_mpi_spawn.o $(LIB) -lm -o example_mpi_spawn
	cp example_mpi_spawn $(DESTDIR)/example_mpi_spawn

example_mpi_spawn.o: example_mpi_spawn.c
	$(CC) -c example_mpi_spawn.c


#
#
# test_mpi
#
#

test_mpi: test_mpi.o
	$(CC) test_mpi.o $(LIB) -lm -o test_mpi
	cp test_mpi $(DESTDIR)/test_mpi

test_mpi.o: test_mpi.c
	$(CC) -c test_mpi.c

MAIN_TRAN_Display_Gridvalue: MAIN_TRAN_Display_Gridvalue.o TRAN_Read.o TRAN_Print.o
	$(CC) -o $@  MAIN_TRAN_Display_Gridvalue.o TRAN_Read.o TRAN_Print.o -lm $(LIB)  

TRAN_Main_Analysis.o: TRAN_Main_Analysis.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Main_Analysis.c
TRAN_Main_Analysis_NC.o: TRAN_Main_Analysis_NC.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Main_Analysis_NC.c

TRAN_Allocate.o: TRAN_Allocate.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Allocate.c
TRAN_Calc_GridBound.o: TRAN_Calc_GridBound.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Calc_GridBound.c
TRAN_DFT.o: TRAN_DFT.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_DFT.c
TRAN_DFT_Dosout.o: TRAN_DFT_Dosout.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_DFT_Dosout.c
TRAN_Input_std.o: TRAN_Input_std.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Input_std.c
TRAN_Input_std_Atoms.o: TRAN_Input_std_Atoms.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Input_std_Atoms.c
TRAN_Output_HKS.o: TRAN_Output_HKS.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Output_HKS.c
TRAN_Output_Trans_HS.o: TRAN_Output_Trans_HS.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Output_Trans_HS.c
TRAN_Add_Density_Lead.o: TRAN_Add_Density_Lead.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Add_Density_Lead.c
TRAN_Add_ADensity_Lead.o: TRAN_Add_ADensity_Lead.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Add_ADensity_Lead.c
TRAN_Poisson.o: TRAN_Poisson.c tran_variables.h tran_prototypes.h lapack_prototypes.h
	$(CC) -c TRAN_Poisson.c
TRAN_RestartFile.o: TRAN_RestartFile.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_RestartFile.c
TRAN_Set_CentOverlap.o: TRAN_Set_CentOverlap.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Set_CentOverlap.c
TRAN_Set_Electrode_Grid.o: tran_variables.h tran_prototypes.h openmx_common.h
	$(CC) -c TRAN_Set_Electrode_Grid.c
TRAN_Set_IntegPath.o: TRAN_Set_IntegPath.c tran_variables.h tran_prototypes.h lapack_prototypes.h  
	$(CC) -c TRAN_Set_IntegPath.c
TRAN_Set_SurfOverlap.o: TRAN_Set_SurfOverlap.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Set_SurfOverlap.c
TRAN_adjust_Ngrid.o: TRAN_adjust_Ngrid.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_adjust_Ngrid.c
Lapack_LU_inverse.o: Lapack_LU_inverse.c tran_prototypes.h lapack_prototypes.h
	$(CC) -c Lapack_LU_inverse.c
TRAN_Deallocate_Electrode_Grid.o: TRAN_Deallocate_Electrode_Grid.c tran_variables.h 
	$(CC) -c TRAN_Deallocate_Electrode_Grid.c
TRAN_Deallocate_RestartFile.o: TRAN_Deallocate_RestartFile.c tran_variables.h 
	$(CC) -c TRAN_Deallocate_RestartFile.c
TRAN_Apply_Bias2e.o: TRAN_Apply_Bias2e.c tran_prototypes.h 
	$(CC) -c TRAN_Apply_Bias2e.c
TRAN_Calc_CentGreen.o: TRAN_Calc_CentGreen.c tran_prototypes.h 
	$(CC) -c TRAN_Calc_CentGreen.c
TRAN_Calc_CentGreenLesser.o: TRAN_Calc_CentGreenLesser.c tran_prototypes.h lapack_prototypes.h
	$(CC) -c TRAN_Calc_CentGreenLesser.c
TRAN_Calc_OneTransmission.o: TRAN_Calc_OneTransmission.c tran_prototypes.h lapack_prototypes.h
	$(CC) -c TRAN_Calc_OneTransmission.c
TRAN_Calc_SelfEnergy.o: TRAN_Calc_SelfEnergy.c tran_prototypes.h lapack_prototypes.h
	$(CC) -c TRAN_Calc_SelfEnergy.c
TRAN_Calc_SurfGreen.o: TRAN_Calc_SurfGreen.c tran_prototypes.h lapack_prototypes.h
	$(CC) -c TRAN_Calc_SurfGreen.c
TRAN_Calc_Hopping_G.o: TRAN_Calc_Hopping_G.c tran_prototypes.h lapack_prototypes.h
	$(CC) -c TRAN_Calc_Hopping_G.c
TRAN_Credit.o: TRAN_Credit.c tran_prototypes.h 
	$(CC) -c TRAN_Credit.c
TRAN_Output_HKS_Write_Grid.o: TRAN_Output_HKS_Write_Grid.c tran_prototypes.h 
	$(CC) -c TRAN_Output_HKS_Write_Grid.c
TRAN_Print.o: TRAN_Print.c tran_prototypes.h 
	$(CC) -c TRAN_Print.c
TRAN_Print_Grid.o: TRAN_Print_Grid.c tran_prototypes.h 
	$(CC) -c TRAN_Print_Grid.c
TRAN_Read.o: TRAN_Read.c tran_prototypes.h 
	$(CC) -c TRAN_Read.c
TRAN_Set_Value.o: TRAN_Set_Value.c tran_prototypes.h 
	$(CC) -c TRAN_Set_Value.c
TRAN_Check_Region_Lead.o: TRAN_Check_Region_Lead.c tran_variables.h
	$(CC) -c TRAN_Check_Region_Lead.c
TRAN_Check_Region.o: TRAN_Check_Region.c tran_prototypes.h 
	$(CC) -c TRAN_Check_Region.c
TRAN_Check_Input.o: TRAN_Check_Input.c tran_prototypes.h 
	$(CC) -c TRAN_Check_Input.c
TRAN_Set_MP.o: TRAN_Set_MP.c
	$(CC) -c TRAN_Set_MP.c
TRAN_Distribute_Node.o: TRAN_Distribute_Node.c
	$(CC) -c TRAN_Distribute_Node.c
TRAN_Allocate_NC.o: TRAN_Allocate_NC.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Allocate_NC.c
TRAN_DFT_NC.o: TRAN_DFT_NC.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_DFT_NC.c
TRAN_Set_CentOverlap_NC.o: TRAN_Set_CentOverlap_NC.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Set_CentOverlap_NC.c
TRAN_Set_SurfOverlap_NC.o: TRAN_Set_SurfOverlap_NC.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Set_SurfOverlap_NC.c

# S MitsuakiKAWAMURA                                                                     
MTRAN_EigenChannel.o: MTRAN_EigenChannel.c tran_prototypes.h \
	TRAN_Calc_SurfGreen.o TRAN_Calc_SelfEnergy.o TRAN_Calc_CentGreen.o
	$(CC) -c MTRAN_EigenChannel.c
TRAN_Channel_Functions.o: TRAN_Channel_Functions.c lapack_prototypes.h
	$(CC) -c TRAN_Channel_Functions.c
TRAN_Channel_Output.o: TRAN_Channel_Output.c openmx_common.h
	$(CC) -c TRAN_Channel_Output.c
TRAN_Calc_CurrentDensity.o: TRAN_Calc_CurrentDensity.c tran_prototypes.h lapack_prototypes.h
	$(CC) -c TRAN_Calc_CurrentDensity.c
TRAN_CDen_Main.o: TRAN_CDen_Main.c openmx_common.h lapack_prototypes.h tran_prototypes.h tran_variables.h
	$(CC) -c TRAN_CDen_Main.c
# E MitsuakiKAWAMURA                       

#
#
# bandgnu13 and other utilities
#
#

bandgnu13: bandgnu13.c
	   gcc bandgnu13.c -lm -o bandgnu13
bin2txt: bin2txt.c
	   gcc bin2txt.c -lm -o bin2txt
cube2xsf: cube2xsf.c
	   gcc cube2xsf.c -lm -o cube2xsf
intensity_map: intensity_map.c
	   gcc intensity_map.c -lm -o intensity_map
md2axsf: md2axsf.c
	   gcc md2axsf.c -lm -o md2axsf
gcube2oned: gcube2oned.c
	   gcc gcube2oned.c -lm -o gcube2oned
cif2omx: cif2omx.c
	gcc -std=c99 -O2 -Wall -Wextra -pedantic -DCIF2OPENMX_MAIN cif2omx.c -lm -o cif2omx
diff_gcube: diff_gcube.c
	gcc diff_gcube.c -lm -o diff_gcube
add_gcube: add_gcube.c
	gcc add_gcube.c -lm -o add_gcube
diff_geo: diff_geo.c
	gcc diff_geo.c -lm -o diff_geo

#
#
# kSpin
#
#

kSpin: kSpin.o BD.o read_scfout.o Inputtools_kSpin.o Tools_BandCalc.o GetOrbital.o EigenValue_Problem.o Eigen_HH.o Tools_Search.o FL.o GC.o MO.o
	$(CC) $^ $(LIB) -lm -o $@

kSpin.o: kSpin.c BandDispersion.h read_scfout.h Inputtools.h Tools_BandCalc.h GetOrbital.h
	$(CC) -c $< -DSIGMAEK

Inputtools_kSpin.o: Inputtools_kSpin.c
	$(CC) -c $<

Tools_BandCalc.o: Tools_BandCalc.c lapack_prototypes.h read_scfout.h Tools_BandCalc.h
	$(CC) -c $<

GetOrbital.o: GetOrbital.c Inputtools.h read_scfout.h Tools_BandCalc.h
	$(CC) -c $<

EigenValue_Problem.o: EigenValue_Problem.c read_scfout.h Tools_BandCalc.h lapack_prototypes.h f77func.h Eigen_HH.h EigenValue_Problem.h
	$(CC) -c $<

Eigen_HH.o: Eigen_HH.c read_scfout.h Tools_BandCalc.h lapack_prototypes.h f77func.h Eigen_HH.h
	$(CC) -c $<

Tools_Search.o: Tools_Search.c Tools_BandCalc.h EigenValue_Problem.h
	$(CC) -c $<

BandDispersion: BandDispersion.o read_scfout.o Inputtools_kSpin.o Tools_BandCalc.o GetOrbital.o EigenValue_Problem.o Eigen_HH.o SigmaEK.o
	$(CC) $^ $(LIB) -lm -o $@

BandDispersion.o: BandDispersion.c GetOrbital.h Inputtools.h read_scfout.h Tools_BandCalc.h EigenValue_Problem.h Eigen_HH.h EigenValue_Problem.h
	$(CC) -c $< 

BD.o: BandDispersion.c GetOrbital.h Inputtools.h read_scfout.h Tools_BandCalc.h EigenValue_Problem.h Eigen_HH.h EigenValue_Problem.h
	$(CC) -c $< -DSIGMAEK -o $@

Circular_Search.o: Circular_Search.c GetOrbital.h Inputtools.h read_scfout.h Tools_BandCalc.h EigenValue_Problem.h Eigen_HH.h EigenValue_Problem.h
	$(CC) -c $<

#
#
# MulPCalc
#
#

MulPCalc: MulPCalc.o Tools_BandCalc.o Inputtools_kSpin.o
	$(CC) $^ $(LIB) -lm -o $@

MulPCalc.o: MulPCalc.c Tools_BandCalc.h Inputtools.h read_scfout.h
	$(CC) -c $<

#
#
# ADenBand
#
#

ADenBand: ADenBand.o Tools_BandCalc.o Inputtools_kSpin.o
	$(CC) $^ $(LIB) -lm -o $@

ADenBand.o: ADenBand.c Tools_BandCalc.h Inputtools.h read_scfout.h
	$(CC) -c $<

#
#
# FermiLoop
#
#

FermiLoop: FermiLoop.o read_scfout.o Inputtools_kSpin.o Tools_BandCalc.o GetOrbital.o EigenValue_Problem.o Eigen_HH.o SigmaEK.o
	$(CC) $^ $(LIB) -lm -o $@

FermiLoop.o: FermiLoop.c read_scfout.h Inputtools.h EigenValue_Problem.h Eigen_HH.h Tools_BandCalc.h GetOrbital.h
	$(CC) -c $< 

FL.o: FermiLoop.c read_scfout.h Inputtools.h EigenValue_Problem.h Eigen_HH.h Tools_BandCalc.h GetOrbital.h
	$(CC) -c $< -DSIGMAEK -o $@

#
#
# GridCalc
#
#

GridCalc: GridCalc.o read_scfout.o Inputtools_kSpin.o Tools_BandCalc.o GetOrbital.o EigenValue_Problem.o Eigen_HH.o SigmaEK.o
	$(CC) $^ $(LIB) -lm -o $@

GridCalc.o: GridCalc.c read_scfout.h Inputtools.h EigenValue_Problem.h Eigen_HH.h Tools_BandCalc.h GetOrbital.h
	$(CC) -c $< 

GC.o: GridCalc.c read_scfout.h Inputtools.h EigenValue_Problem.h Eigen_HH.h Tools_BandCalc.h GetOrbital.h
	$(CC) -c $< -DSIGMAEK -o $@

#
#
# MulPOnly
#
#

MulPOnly: MulPOnly.o read_scfout.o Inputtools_kSpin.o EigenValue_Problem.o Eigen_HH.o Tools_BandCalc.o GetOrbital.o SigmaEK.o
	$(CC) $^ $(LIB) -lm -o $@

MulPOnly.o: MulPOnly.c Tools_BandCalc.h Inputtools.h read_scfout.h EigenValue_Problem.h GetOrbital.h SigmaEK.o
	$(CC) -c $< 

MO.o: MulPOnly.c Tools_BandCalc.h Inputtools.h read_scfout.h EigenValue_Problem.h GetOrbital.h
	$(CC) -c $< -DSIGMAEK -o $@

SigmaEK.o: SigmaEK.c
	$(CC) -c $< -o $@

# Host-backed capacity adapters share the task descriptor ABI.
Total_Energy_GPU.o Force.o Set_Orbitals_Grid.o Set_Hamiltonian.o Set_Nonlocal.o Set_ProExpn_VNA.o: openmx_gpu_task.h
Set_Density_Grid_GPU.o: openmx_gpu_bilinear.h
Mixing_H.o: openmx_gpu_vector.h openmx_host_memory.h
Set_XC_Grid.o: openmx_gpu_collective.h
openmx_gpu_bilinear_pack.o: openmx_host_memory.h

truncation.o: openmx_gpu_task.h openmx_host_memory.h
truncation.o: openmx_host_simd.h openmx_core_olg.h openmx_core_host_simd.h ../core/include/openmx_core_olg.h ../core/include/openmx_core_host_simd.h

openmx_gpu_dm_collective.o: openmx_gpu_dm_collective.c openmx_gpu_dm_collective.h openmx_gpu_dm_entry.h openmx_gpu_task.h openmx_gpu_workers.h openmx_mgpu_runtime.h
	$(CC) -c openmx_gpu_dm_collective.c

Band_DFT_Col.o Band_DFT_NonCol.o Cluster_DFT_Col.o Cluster_DFT_NonCol.o: openmx_gpu_dm_collective.h openmx_gpu_workers.h

# Invocation-owned projector communication and shared VNA orchestration.
Force.o Stress.o Set_Nonlocal.o Set_CoreHoleMatrix.o Set_ProExpn_VNA.o: openmx_projector_exchange.h openmx_core_projector_exchange.h
Set_ProExpn_VNA.o: openmx_vna_assembly.h openmx_core_vna_invocation.h openmx_core_atom_pairs.h openmx_vna_tables.h openmx_vna_status.h
Set_ProExpn_VNA.o: openmx_host_simd.h openmx_core_vna_simd.h ../core/include/openmx_core_vna_simd.h ../core/include/openmx_core_host_simd.h openmx_setpro_host.h openmx_setpro_gaunt.h

# Shared flat orbital optimization and Hubbard force boundaries.
DFT.o Opt_Contraction.o: openmx_orbital_optimization.h openmx_core_orbital_optimization.h
Force.o: openmx_hubbard_force.h
Force.o: openmx_force_workspace.h openmx_core_force.h openmx_force_stages.h \
         openmx_solver_workspace.h openmx_core_solver_workspace.h \
         openmx_core_orbital_workspace.h openmx_core_local_basis.h
Force.o: openmx_force_projectors.h openmx_core_core_hole.h
Force.o: openmx_core_projector_force.h
Force.o: openmx_force_hnl_gpu.h openmx_native_resources.h openmx_core_native_resources.h
Force.o: openmx_force_gpu_resources.h openmx_force_archive_lease.h openmx_force_recovery.h openmx_core_force_recovery.h
Force.o: ../core/include/openmx_core_force_recovery.h
Force.o: openmx_force_hnl_batch.h openmx_core_force_hnl_batch.h ../core/include/openmx_core_force_hnl_batch.h
Force.o: openmx_host_simd.h openmx_core_force_simd.h ../core/include/openmx_core_force_simd.h ../core/include/openmx_core_projector_force.h
Force.o: openmx_core_force_vna_simd.h ../core/include/openmx_core_force_vna_simd.h
Force.o: openmx_core_orbitals_simd.h ../core/include/openmx_core_orbitals_simd.h

# Solver invocation RAII and flat spectral workspaces.
Cluster_DFT_Col.o Cluster_DFT_NonCol.o Band_DFT_Col.o Band_DFT_NonCol.o Divide_Conquer.o Divide_Conquer_LNO.o Krylov.o: openmx_solver_workspace.h openmx_core_solver_workspace.h

# Calculation-owned native GPU resources.
DFT.o Cluster_DFT_Col.o Cluster_DFT_NonCol.o Band_DFT_Col.o Band_DFT_NonCol.o Divide_Conquer.o Divide_Conquer_LNO.o Krylov.o: openmx_native_resources.h openmx_core_native_resources.h
Free_Arrays.o: openmx_core_native_resources.h

openmx_grid_assembly.o: openmx_grid_assembly.c openmx_grid_assembly.h openmx_core_grid.h openmx_common.h openmx_host_simd.h
	$(CC_NOACC_O3) -c openmx_grid_assembly.c
DFT.o Set_Hamiltonian.o Set_Density_Grid.o Free_Arrays.o: openmx_grid_assembly.h openmx_core_grid.h
Set_Hamiltonian.o: openmx_host_simd.h

DFT.o Cluster_DFT_Col.o: openmx_cluster_col_state.h

DFT.o Cluster_DFT_NonCol.o: openmx_cluster_noncol_state.h

Set_Density_Grid.o Set_Density_Grid_GPU.o: openmx_density_cache_hooks.h

DFT.o TRAN_DFT.o Band_DFT_Col.o: openmx_band_col_state.h
DFT.o TRAN_DFT_NC.o Band_DFT_NonCol.o: openmx_band_noncol_state.h

DFT.o Divide_Conquer.o: openmx_dc_state.h
DFT.o Divide_Conquer_LNO.o: openmx_dclno_state.h
DFT.o Krylov.o: openmx_krylov_state.h

Total_Energy.o: openmx_core_dispersion.h ../core/include/openmx_core_dispersion.h

Stress.o: openmx_stress_workspace.h openmx_solver_workspace.h openmx_core_solver_workspace.h ../core/include/openmx_core_solver_workspace.h

Total_Energy.o Stress.o: openmx_xc_workspace.h openmx_core_angular_grid.h ../core/include/openmx_core_angular_grid.h
Total_Energy.o Stress.o: openmx_neutral_atom.h openmx_host_simd.h openmx_core_energy_simd.h ../core/include/openmx_core_energy_simd.h openmx_core_host_simd.h ../core/include/openmx_core_host_simd.h
Total_Energy_GPU.o: openmx_host_simd.h openmx_core_energy_simd.h ../core/include/openmx_core_energy_simd.h openmx_core_host_simd.h ../core/include/openmx_core_host_simd.h
Band_DFT_Col.o: openmx_host_simd.h openmx_core_energy_simd.h ../core/include/openmx_core_energy_simd.h openmx_core_host_simd.h ../core/include/openmx_core_host_simd.h
Cluster_DFT_Col.o Band_DFT_NonCol.o: openmx_host_simd.h openmx_core_host_simd.h ../core/include/openmx_core_host_simd.h

Force.o Stress.o: openmx_mechanical_workspace.h openmx_core_force.h ../core/include/openmx_core_force.h

openmx_hamiltonian_cache.o: openmx_hamiltonian_cache.c openmx_hamiltonian_cache.h openmx_hamiltonian_pack.h openmx_core_native_resources.h
	$(CC_NOACC_O3) -c openmx_hamiltonian_cache.c

DFT.o: openmx_hamiltonian_pack.h
$(OBJS): openmx_hamiltonian_cache.h
