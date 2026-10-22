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

# Backend inputs belong to the source tree; every mutable library, object and
# installation prefix belongs to the projected build directory.  The unified
# build driver also supplies OPENMX_SOURCE_ROOT explicitly.
ifndef OPENMX_SOURCE_ROOT
OPENMX_SOURCE_ROOT := $(abspath $(dir $(realpath $(lastword $(MAKEFILE_LIST))))/../..)
endif
OPENMX_BUILD_ROOT ?= $(CURDIR)
OPENMX_HIP_DEPS ?= $(OPENMX_SOURCE_ROOT)/third_party/hip

ROCM      ?= /opt/rocm-7.2.0
ROCM_LLVM ?= $(ROCM)/llvm
ROCM_LIB  ?= $(ROCM)/lib
ROCM_LLVM_LIB ?= $(ROCM_LLVM)/lib

# ROCm clang searches LIBRARY_PATH before explicit -L arguments. Keep its
# OpenMP runtimes ahead of ambient Intel/AOCC runtimes with a different ABI.
export LIBRARY_PATH := $(ROCM_LLVM_LIB)$(if $(LIBRARY_PATH),:$(LIBRARY_PATH))

# offload-arch/rocminfo only answer on a node that owns a GPU, so keep a
# fallback for builds driven from a GPU-less login node.  Both tools may append
# target features (e.g. gfx942:sramecc+:xnack-); MAGMA's GPU_TARGET rejects
# those, so keep the bare gfx name.
GPU_ARCH_FALLBACK ?= gfx942
ARCH ?= $(shell { $(ROCM_LLVM)/bin/offload-arch 2>/dev/null || $(ROCM)/bin/rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+'; } | head -n 1 | cut -d: -f1)
ifeq ($(strip $(ARCH)),)
ARCH := $(GPU_ARCH_FALLBACK)
endif
OMP_OFFLOAD := -fopenmp --offload-arch=$(ARCH)

CLANG ?= $(ROCM_LLVM)/bin/clang
CPPLANG ?= $(ROCM_LLVM)/bin/clang++
HIPCC ?= $(ROCM)/bin/hipcc
FLANG ?= $(ROCM_LLVM)/bin/flang

# Set_Density_Grid contains explicit AVX-512/AVX2 runtime dispatch.  Compile
# its baseline code without -march=native so the dispatcher and scalar path
# themselves never acquire AVX-512 instructions on an AVX-512 build host.
CLANG_HOST_TRIPLE := $(shell $(CLANG) -dumpmachine 2>/dev/null)
ifeq ($(findstring x86_64,$(CLANG_HOST_TRIPLE)),x86_64)
SDG_CPU_BASELINE_FLAGS ?= -march=x86-64-v2 -mtune=native
else
SDG_CPU_BASELINE_FLAGS ?=
endif

# Set_Hamiltonian contains separately targeted AVX-512 and AVX2 kernels.
# Keep every dispatcher, fallback, and unrelated function in this translation
# unit on the same non-AVX baseline used by Set_Density_Grid.
SETH_CPU_BASELINE_FLAGS ?= $(SDG_CPU_BASELINE_FLAGS)

# OpenMPI built against the same ROCm as $(ROCM); override MPI_HOME elsewhere.
# (module load openmpi/5.0.10/rocm7.2.0_amdflang)
# The Fortran-90 "mpi" module is compiler-specific: the plain rocm7.2.0 build
# ships a gfortran mpi.mod that $(FLANG) cannot read, so ELPA's "use mpi" needs
# the amdflang-built variant of the very same OpenMPI.
MPI_HOME ?= /system/apps/rhel/openmpi/5.0.10/rocm/7.2.0-rocm_amdflang/ompi
MPICC  ?= $(MPI_HOME)/bin/mpicc
MPICXX ?= $(MPI_HOME)/bin/mpicxx
MPIF90 ?= $(MPI_HOME)/bin/mpif90

MPI_CFLAGS  := $(shell $(MPICC) --showme:compile 2>/dev/null)
MPI_FCFLAGS := $(shell $(MPIF90) --showme:compile 2>/dev/null)

COMMON_INCLUDES = -I. -I$(ROCM)/include -I$(AOCL_INC) -I$(GEMMUL8_DIR)/include -I$(GEMMUL8_DIR)/src -I$(MAGMA_DIR)/include
CFLAGS_COMMON = -fcommon -Werror=implicit-function-declaration -std=gnu11 -mtune=native -march=native -D__HIP_PLATFORM_AMD__ -DM_PI=3.14159265358979323846264338327950288 $(COMMON_INCLUDES)
CPPFLAGS_COMMON = -w -fcommon -std=c++20 -mtune=native -march=native -D__HIP_PLATFORM_AMD__ $(COMMON_INCLUDES)

CC  = $(CLANG) -O3 $(MPI_CFLAGS) $(CFLAGS_COMMON) $(OMP_OFFLOAD)
CC2 = $(CLANG) -O1 $(MPI_CFLAGS) $(CFLAGS_COMMON) $(OMP_OFFLOAD)
CC3 = $(CLANG) -O2 $(MPI_CFLAGS) $(CFLAGS_COMMON) $(OMP_OFFLOAD)
CC_NOACC = $(CLANG) -O1 $(MPI_CFLAGS) $(CFLAGS_COMMON) -fopenmp
CC_NOACC_O3 = $(CLANG) -O3 $(MPI_CFLAGS) $(CFLAGS_COMMON) -fopenmp
CC_DFTD3 = $(CLANG) -w -O0 $(MPI_CFLAGS) $(CFLAGS_COMMON) -fopenmp
CXX = $(CPPLANG) -O3 $(MPI_CFLAGS) $(CPPFLAGS_COMMON) $(OMP_OFFLOAD)
FC = $(FLANG) -O2 $(MPI_FCFLAGS) -mtune=native -march=native -fopenmp

MPI_LIBDIR ?= $(MPI_HOME)/lib
# AOCL's ScaLAPACK/libFLAME were built with AOCC's classic flang, so they pull
# in libflang/libflangrti/libpgmath from the AOCC install.  Keep $(AOCC_LIB)
# last: it also ships an older libomp.so that must not shadow ROCm's.
AOCC_LIB ?= /system/apps/rhel/aocc/5.0.0/lib
RPATHS = -Wl,--disable-new-dtags -Wl,-rpath,$(ROCM_LLVM_LIB) -Wl,-rpath,$(ROCM_LIB) -Wl,-rpath,$(MPI_LIBDIR) -Wl,-rpath,$(AOCL_LIB) -Wl,-rpath,$(AOCC_LIB)
ROCM_GPU_LIBS := -L$(ROCM_LIB) -lamd_smi -lhipsolver -lrocsolver -lhipblas -lhipblaslt -lrocblas -lhipsparse -lamdhip64
# ROCm ships libflang_rt.* under the clang resource directory, not directly in
# $(ROCM_LLVM_LIB); ask the compiler where that is instead of hardcoding the
# LLVM major version.
FLANG_RT_LIBDIR ?= $(shell $(FLANG) -print-resource-dir 2>/dev/null)/lib/$(shell $(FLANG) -print-target-triple 2>/dev/null)
FLANG_RTL_LIBS := -L$(FLANG_RT_LIBDIR) -L$(ROCM_LLVM_LIB) -lflang_rt.runtime -lflang_rt.quadmath -lquadmath
# AOCL (BLIS / libFLAME / ScaLAPACK / FFTW3); module load aocl/5.1.0/aocc5.0.0.
# AOCL_INC also supplies fftw3.h, so it is part of COMMON_INCLUDES above.
AOCL_HOME ?= /system/apps/rhel/aocl/5.1.0/aocc
AOCL_INC ?= $(AOCL_HOME)/include
AOCL_LIB ?= $(AOCL_HOME)/lib
SCALAPACK_LIBS := -L$(AOCL_LIB) -lscalapack
LAPACK_LIBS := -L$(AOCL_LIB) -lflame
BLAS_LIBS := -L$(AOCL_LIB) -lblis
AOCL_UTILS_LIBS := -L$(AOCL_LIB) -laoclutils
FFTW_LIBS := -L$(AOCL_LIB) -lfftw3 -lm
FFTW3_LIB ?=
MATH_GROUP := -Wl,--start-group $(SCALAPACK_LIBS) $(LAPACK_LIBS) $(BLAS_LIBS) $(AOCL_UTILS_LIBS) -Wl,--end-group
MPI_FORTRAN_LIBS := \
  $(MPI_LIBDIR)/libmpi_usempif08.so \
  $(MPI_LIBDIR)/libmpi_usempi_ignore_tkr.so \
  $(MPI_LIBDIR)/libmpi_mpifh.so \
  $(MPI_LIBDIR)/libmpi.so

LIB= $(RPATHS) $(FFTW_LIBS) $(GPUSOLVER2_LIBS) $(MATH_GROUP) $(MPI_FORTRAN_LIBS) $(ROCM_GPU_LIBS) $(FLANG_RTL_LIBS) -lstdc++ -lpthread -lm -ldl

GEMMUL8_REPO_URL ?= https://github.com/RIKEN-RCCS/GEMMul8.git
GEMMUL8_REPO_DIR ?= $(OPENMX_BUILD_ROOT)/third_party/GEMMul8
GEMMUL8_SOURCE_DIR ?= $(OPENMX_SOURCE_ROOT)/third_party/GEMMul8
# gemmul8_openmx.cu instantiates GEMMul8 internals directly, so the
# checkout must stay pinned to a commit whose internal layout it matches.
# 603b523 is the v3.5.2 release (grouped CRT reconstruction and the
# HIP host/device constexpr compilation fix).
GEMMUL8_COMMIT ?= 603b52363715796a0af5e4aa1ed8d386349b4251
GEMMUL8_DIR ?= $(GEMMUL8_REPO_DIR)
GEMMUL8_LIB ?= $(GEMMUL8_DIR)/lib/libgemmul8.a
GEMMUL8_OPENMX_OBJ ?= $(GEMMUL8_DIR)/src/gemmul8_openmx.o
# GEMMul8 keeps the handle-local configuration (memory-saving mode) in a
# registry defined in this host-only TU; the gemm path queries it via
# gemmul8::config::get_config, so it must be part of the library.
GEMMUL8_CONFIG_SRC ?= $(GEMMUL8_DIR)/src/worksize/cu_recipe/memory_saving.cu
GEMMUL8_CONFIG_OBJ ?= $(GEMMUL8_DIR)/src/gemmul8_config.o
GEMMUL8_DEPFILES = $(GEMMUL8_OPENMX_OBJ:.o=.d) $(GEMMUL8_CONFIG_OBJ:.o=.d)
GEMMUL8_GPU_ARCH ?= $(ARCH)
HIPCC_GEMMUL8_FLAGS = -std=c++20 -O3 -fPIC --offload-arch=$(GEMMUL8_GPU_ARCH) -I. -I$(GEMMUL8_DIR)/include -I$(GEMMUL8_DIR)/src -I$(ROCM)/include -D__HIP_PLATFORM_AMD__ -ffp-contract=off -Wno-unused-result -Wno-unused-command-line-argument -Wno-unused-value -DOCML_BASIC_ROUNDED_OPERATIONS -DGPU_ARCH=$(GEMMUL8_GPU_ARCH)
MAGMA_VERSION ?= 2.10.0
MAGMA_DIR ?= $(OPENMX_BUILD_ROOT)/third_party/magma
MAGMA_ARCHIVE_BASENAME ?= magma-$(MAGMA_VERSION).tar.gz
MAGMA_ARCHIVE ?= $(OPENMX_HIP_DEPS)/magma/$(MAGMA_ARCHIVE_BASENAME)
MAGMA_LIB ?= $(MAGMA_DIR)/lib/libmagma.a
MAGMA_GPU_TARGET ?= $(ARCH)
MAGMA_MAKE_INC ?= $(OPENMX_HIP_DEPS)/magma.openmx.make.inc
MAGMA_OZAKI_PATCH ?= $(OPENMX_HIP_DEPS)/magma-ozaki2.patch
MAGMA_DEVICE_MAP_SCRIPT ?= $(OPENMX_HIP_DEPS)/apply_magma_device_map.py
MAGMA_DEVICE_MAP_HEADER ?= $(OPENMX_HIP_DEPS)/magma_device_map.h
MAGMA_STAMP ?= $(MAGMA_DIR)/.openmx-magma-$(MAGMA_VERSION).stamp
MAGMA_DEVICE_MAP_STAMP = $(MAGMA_DIR)/.openmx-magma-device-map-v1.stamp
MAGMA_EXTRA_GFXS ?= 1200
HIPCC_MAGMA_FLAGS = $(HIPCC_GEMMUL8_FLAGS) -I$(MAGMA_DIR)/include -I$(MAGMA_DIR)/control

# ============================================================================
# gpusolver2 (scf.eigen.lib=gpusolver2): distributed multi-GPU cluster
# diagonalization via ELPA 2026.02 with AMD GPU kernels (eigensolver) and
# COSMA (pdgemm).  Both are built from the release archives bundled in
# third_party/dist by third_party/build_gpusolver2_stack.sh.
# ============================================================================
GPUSOLVER2_DIST      ?= $(OPENMX_SOURCE_ROOT)/third_party/dist
GPUSOLVER2_SCRIPT    ?= $(OPENMX_HIP_DEPS)/build_gpusolver2_stack.sh
GPUSOLVER2_BUILD     ?= $(OPENMX_BUILD_ROOT)/build/gpusolver2
GPUSOLVER2_PREFIX    ?= $(OPENMX_BUILD_ROOT)/third_party/gpusolver2-install
GPUSOLVER2_HOST_CC   ?= gcc
GPUSOLVER2_HOST_CXX  ?= g++
GPUSOLVER2_HIPCC     ?= $(HIPCC)
GPUSOLVER2_GPU_ARCH  ?= $(ARCH)
GPUSOLVER2_MPI_BIN   ?= $(MPI_HOME)/bin
GPUSOLVER2_MPI_LIBDIR ?= $(MPI_LIBDIR)
GPUSOLVER2_ROCM      ?= $(ROCM)
GPUSOLVER2_AOCL_LIB  ?= $(AOCL_LIB)
GPUSOLVER2_AOCL_INC  ?= $(AOCL_INC)
GPUSOLVER2_AOCC_LIB  ?= $(AOCC_LIB)
GPUSOLVER2_SCALAPACK_SO ?= $(AOCL_LIB)/libscalapack.so
GPUSOLVER2_CMAKE     ?=
GPUSOLVER2_JOBS      ?= $(shell nproc 2>/dev/null || echo 8)
GPUSOLVER2_ELPA_VER  ?= 2026.02.002
GPUSOLVER2_ELPA_INCLUDE ?= $(GPUSOLVER2_PREFIX)/include/elpa-$(GPUSOLVER2_ELPA_VER)
GPUSOLVER2_STAMP     = $(GPUSOLVER2_BUILD)/stamp/02_cosma.done

GPUSOLVER2_LIBS = $(GPUSOLVER2_PREFIX)/lib/libelpa.a \
 -Wl,--start-group \
 $(GPUSOLVER2_PREFIX)/lib/libcosma_prefixed_pxgemm.a \
 $(GPUSOLVER2_PREFIX)/lib/libcosma.a \
 $(GPUSOLVER2_PREFIX)/lib/libcosta_scalapack.a \
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
          Cluster_DFT_Col.o Cluster_DFT_Col_dm_hip.o Cluster_DFT_NonCol.o Cluster_DFT_NonCol_dm_hip.o Cluster_DFT_Dosout.o Cluster_DFT_ON2.o Cluster_DFT_Col_LNO.o \
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
          Divide_Conquer.o Divide_Conquer_LNO.o my_hipblasDgemm.o my_hipblasZgemm.o gemmul8_bridge.o eigen_refine_gpu.o openmx_precision.o orbs_otf_hip.o set_xc_grid_hip.o elpa_cosma_bridge.o magma_openmx.o Krylov.o \
          Divide_Conquer_Dosout.o EGAC_DFT.o LNO.o Calc_LNAO.o Calc_LNBO.o \
          Eigen_lapack.o gpusolver_Syevd.o gpusolver_Syevdx.o set_hip_default_device_from_local_rank.o set_openmp_device_from_local_rank.o Eigen_lapack2.o Eigen_lapack3.o EigenBand_lapack.o \
          Eigen_PReHH.o BroadCast_ReMatrix.o \
          Eigen_PHH.o BroadCast_ComplexMatrix.o \
          lapack_dstedc1.o lapack_dstedc2.o lapack_dstedc3.o\
          lapack_dstegr1.o lapack_dstegr2.o lapack_dstegr3.o \
          lapack_dstevx1.o lapack_dstevx2.o lapack_dstevx3.o \
          lapack_dstevx4.o lapack_dstevx5.o lapack_dsteqr1.o \
          Nonlocal_Basis.o Set_OLP_Kin.o Set_Nonlocal.o Set_ProExpn_VNA.o \
          Set_CoreHoleMatrix.o Set_XMCD_penalty.o Set_OLP_p.o Set_Hamiltonian.o set_hamiltonian_hip.o Set_Vpot.o \
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

# PROG    = openmx.exe
# PROG    = openmx

# CPU elpa1/elpa2 and GPU gpusolver2 share the bundled modern ELPA library.

#
# set program name
# destination directory
#

PROG    = openmx
DESTDIR ?= ../work
UTIL 	= DosMain cohp jx analysis_example esp polB calB Z2FH bandgnu13 bin2txt cube2xsf intensity_map md2axsf tp kSpin BandDispersion ADenBand FermiLoop GridCalc MulPOnly MulPCalc gcube2oned cif2omx diff_gcube add_gcube diff_geo

#
# FFTW3
#

.PHONY: fftw3 fftw3-clean gemmul8-clean magma-clean gpusolver2-stack gpusolver2-clean clean

fftw3: $(FFTW3_LIB)

$(FFTW3_SRC_ABS):
	@echo "FFTW3 source directory was not found: $(FFTW3_SRC_ABS)"
	@echo "Clone $(FFTW3_REPO_URL), check out $(FFTW3_TAG), or override FFTW3_SRC_DIR."
	@false

$(FFTW3_CODELET_SENTINEL): | $(FFTW3_SRC_ABS)
	@echo "FFTW3 git source lacks generated codelets; overlaying $(FFTW3_RELEASE_URL)."
	mkdir -p "$(dir $(FFTW3_RELEASE_TARBALL))"
	curl -L --fail "$(FFTW3_RELEASE_URL)" -o "$(FFTW3_RELEASE_TARBALL)"
	tmpdir=$$(mktemp -d); \
	  tar -xzf "$(FFTW3_RELEASE_TARBALL)" -C "$$tmpdir"; \
	  rsync -a --ignore-existing "$$tmpdir/fftw-$(FFTW3_VERSION)/" "$(FFTW3_SRC_ABS)/"; \
	  rm -rf "$$tmpdir"

$(FFTW3_BUILD_DIR)/config.status: $(FFTW3_CODELET_SENTINEL) | $(FFTW3_SRC_ABS)
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
	rm -f "$(GEMMUL8_OPENMX_OBJ)" "$(GEMMUL8_CONFIG_OBJ)" "$(GEMMUL8_LIB)" $(GEMMUL8_DEPFILES)

magma-clean:
	if [ -f "$(MAGMA_DIR)/Makefile" ]; then $(MAKE) -C "$(MAGMA_DIR)" clean; fi
	find "$(MAGMA_DIR)" -mindepth 1 -maxdepth 1 ! -name "$(MAGMA_ARCHIVE_BASENAME)" -exec rm -rf {} +
	rm -f magma_openmx.o

FFTW3_USERS = Poisson.o Stress.o Poisson_ESM.o TRAN_Poisson.o TRAN_CDen_Main.o TRAN_Set_Electrode_Grid.o Cluster_DFT_Col.o
$(FFTW3_USERS): | $(FFTW3_LIB)

#
# OpenMX
#

.DEFAULT_GOAL := openmx
.DELETE_ON_ERROR:

# Capacity recovery and bounded multi-device execution. C kernels use
# OpenMP target; the explicit device kernels and runtime use HIP.
MGPU_C_OBJECTS = openmx_cluster_capacity.o openmx_dense_cpu.o \
 openmx_gpu_bilinear.o openmx_gpu_collective.o openmx_gpu_dm_collective.o \
 openmx_gpu_task.o openmx_gpu_vector.o openmx_gpu_workers.o openmx_gpu_error.o openmx_host_memory.o \
 openmx_env.o
MGPU_HIP_OBJECTS = openmx_gpu_dm.o openmx_gpu_transform.o \
 openmx_mgpu_gemm.o openmx_mgpu_runtime.o
OBJS += $(MGPU_C_OBJECTS) $(MGPU_HIP_OBJECTS) openmx_gpu_views.o \
 openmx_gpu_bilinear_pack.o openmx_magma_mgpu.o elpa_cosma_memory.o openmx_globals.o

openmx_globals.o: openmx_globals.c openmx_common.h
	$(CC_NOACC) -fno-common -c openmx_globals.c -o $@

include openmx_core.mk
# These interfaces also reach legacy objects through openmx_common.h.
# Rebuild consumers when a native API mapping or backend selection changes.
OPENMX_GPU_INTERFACE_HEADERS = openmx_gpu_backend.h openmx_gpu_runtime.h \
 openmx_gpu_blas.h openmx_gpu_solver.h openmx_gpu_complex.h \
 openmx_gpu_device.h openmx_gpu_error.h
$(OBJS): $(OPENMX_GPU_INTERFACE_HEADERS)
MGPU_HEADERS = $(wildcard openmx_gpu_*.h openmx_mgpu_*.h) \
 openmx_magma_mgpu.h openmx_magma_lock.h openmx_target.h openmx_host_memory.h \
 openmx_cluster_capacity.h openmx_dense_cpu.h openmx_env.h

# Keep incremental builds valid when compilers, ROCm, GPU targets, MPI, or
# mathematical libraries change.  An unchanged configuration keeps its mtime.
OPENMX_BUILD_CONFIG := $(OPENMX_BUILD_ROOT)/build/openmx-toolchain.conf
.PHONY: openmx-toolchain-force
$(OPENMX_BUILD_CONFIG): openmx-toolchain-force
	@test -n "$(ARCH)" || { echo "Set ARCH to an AMD GPU target, for example gfx942 or gfx1200."; exit 1; }
	@mkdir -p "$(dir $@)"
	@printf '%s\n' '$(ROCM)' '$(ROCM_LLVM)' '$(ROCM_LIB)' '$(ROCM_LLVM_LIB)' '$(ARCH)' '$(CLANG)' '$(CPPLANG)' '$(HIPCC)' '$(FLANG)' '$(CC)' '$(CC2)' '$(CC3)' '$(CC_NOACC)' '$(CC_NOACC_O3)' '$(CC_DFTD3)' '$(CXX)' '$(FC)' '$(CFLAGS)' '$(CPPFLAGS)' '$(SDG_CPU_BASELINE_FLAGS)' '$(SETH_CPU_BASELINE_FLAGS)' '$(MPI_HOME)' '$(MPI_CFLAGS)' '$(MPI_FCFLAGS)' '$(MPI_LIBDIR)' '$(AOCL_INC)' '$(AOCL_LIB)' '$(AOCC_LIB)' '$(LIB)' '$(GEMMUL8_COMMIT)' '$(HIPCC_GEMMUL8_FLAGS)' '$(HIPCC_MAGMA_FLAGS)' '$(MAGMA_GPU_TARGET)' '$(MAGMA_EXTRA_GFXS)' > "$@.tmp"
	@cmp -s "$@.tmp" "$@" && rm -f "$@.tmp" || mv "$@.tmp" "$@"

$(OBJS) $(GEMMUL8_OPENMX_OBJ) $(GEMMUL8_CONFIG_OBJ): $(OPENMX_BUILD_CONFIG)
$(MAGMA_STAMP): $(OPENMX_BUILD_CONFIG)

# Stack-only overrides trigger its own builder, whose toolchain signature
# invalidates stale ELPA/COSMA products when its inputs change.
GPUSOLVER2_BUILD_CONFIG := $(GPUSOLVER2_BUILD)/make-toolchain.conf
.PHONY: gpusolver2-toolchain-force
$(GPUSOLVER2_BUILD_CONFIG): gpusolver2-toolchain-force
	@mkdir -p "$(dir $@)"
	@printf '%s\n' '$(GPUSOLVER2_DIST)' '$(GPUSOLVER2_PREFIX)' '$(GPUSOLVER2_HOST_CC)' '$(GPUSOLVER2_HOST_CXX)' '$(GPUSOLVER2_HIPCC)' '$(GPUSOLVER2_GPU_ARCH)' '$(GPUSOLVER2_MPI_BIN)' '$(GPUSOLVER2_MPI_LIBDIR)' '$(GPUSOLVER2_ROCM)' '$(GPUSOLVER2_AOCL_LIB)' '$(GPUSOLVER2_AOCL_INC)' '$(GPUSOLVER2_AOCC_LIB)' '$(GPUSOLVER2_SCALAPACK_SO)' '$(GPUSOLVER2_ELPA_VER)' '$(GPUSOLVER2_CMAKE)' > "$@.tmp"
	@cmp -s "$@.tmp" "$@" && rm -f "$@.tmp" || mv "$@.tmp" "$@"
$(GPUSOLVER2_STAMP): $(GPUSOLVER2_BUILD_CONFIG)

$(MGPU_C_OBJECTS): %.o: %.c $(MGPU_HEADERS)
	$(CC) -c $< -o $@
$(MGPU_HIP_OBJECTS): %.o: %.cu $(MGPU_HEADERS)
	$(HIPCC) $(HIPCC_GEMMUL8_FLAGS) $(MPI_CFLAGS) -c $< -o $@
openmx_gpu_views.o: openmx_gpu_views.c openmx_gpu_views.h hip_runtime_compat.h
	$(HIPCC) $(HIPCC_GEMMUL8_FLAGS) -fopenmp -x hip -c $< -o $@
openmx_gpu_bilinear_pack.o: openmx_gpu_bilinear_pack.cpp openmx_gpu_bilinear.h
	$(CPPLANG) -O3 -std=c++17 $(MPI_CFLAGS) -c $< -o $@
openmx_magma_mgpu.o: openmx_magma_mgpu.cpp openmx_magma_mgpu.h openmx_magma_lock.h $(MAGMA_LIB)
	$(HIPCC) $(HIPCC_MAGMA_FLAGS) -DOPENMX_HAVE_MAGMA=1 -c $< -o $@
elpa_cosma_memory.o: elpa_cosma_memory.cpp $(GPUSOLVER2_STAMP)
	$(GPUSOLVER2_HOST_CXX) -O2 -std=c++17 $(MPI_CFLAGS) -I$(GPUSOLVER2_PREFIX)/include -c $< -o $@
Input_std.o DFT.o Cluster_DFT_Col.o Cluster_DFT_NonCol.o Band_DFT_Col.o \
 Band_DFT_NonCol.o Set_Hamiltonian.o Set_Density_Grid_GPU.o Set_Orbitals_Grid.o \
 Set_Nonlocal.o Set_OLP_Kin.o Set_ProExpn_VNA.o Set_XC_Grid.o Force.o \
 Total_Energy_GPU.o Mixing_H.o truncation.o gemmul8_bridge.o magma_openmx.o \
 set_hip_default_device_from_local_rank.o set_openmp_device_from_local_rank.o \
 Cluster_DFT_Col_dm_hip.o Cluster_DFT_NonCol_dm_hip.o: $(MGPU_HEADERS)

openmx:	$(FFTW3_LIB) $(GEMMUL8_LIB) $(MAGMA_LIB) $(OBJS)
	$(CC) $(OBJS) $(STACK) $(MAGMA_LIB) $(GEMMUL8_LIB) $(LIB) -lm -o openmx
	
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
Input_std.o: Input_std.c openmx_common.h Inputtools.h tran_prototypes.h  openmx_precision.h
	$(CC) -c Input_std.c
Inputtools.o: Inputtools.c
	$(CC) -c Inputtools.c

init.o: init.c openmx_common.h
	$(CC) -c init.c

LU_inverse.o: LU_inverse.c openmx_common.h
	$(CC) -c LU_inverse.c
ReLU_inverse.o: ReLU_inverse.c openmx_common.h
	$(CC) -c ReLU_inverse.c
truncation.o: truncation.c openmx_common.h set_hip_default_device_from_local_rank.h tran_prototypes.h
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

DFT.o: DFT.c openmx_common.h openmx_vna_status.h tran_prototypes.h set_hip_default_device_from_local_rank.h eigen_refine_hooks.h
	$(CC) -c DFT.c
Cluster_DFT_Col.o: Cluster_DFT_Col.c openmx_common.h eigen_refine_gpu.h eigen_refine_hooks.h openmx_precision.h
	$(CC) -c Cluster_DFT_Col.c
Cluster_DFT_Col_dm_hip.o: Cluster_DFT_Col_dm_hip.cu
	$(HIPCC) $(HIPCC_GEMMUL8_FLAGS) -c Cluster_DFT_Col_dm_hip.cu -o Cluster_DFT_Col_dm_hip.o
magma_openmx.o: magma_openmx.cu openmx_common.h $(MAGMA_LIB) openmx_stdout_level.h openmx_gpu_arch_policy.h openmx_magma_lock.h
	$(HIPCC) $(HIPCC_MAGMA_FLAGS) -c magma_openmx.cu -o magma_openmx.o
Cluster_DFT_NonCol.o: Cluster_DFT_NonCol.c openmx_common.h eigen_refine_gpu.h eigen_refine_hooks.h openmx_precision.h
	$(CC) -c Cluster_DFT_NonCol.c
Cluster_DFT_NonCol_dm_hip.o: Cluster_DFT_NonCol_dm_hip.cu
	$(HIPCC) $(HIPCC_GEMMUL8_FLAGS) -c Cluster_DFT_NonCol_dm_hip.cu -o Cluster_DFT_NonCol_dm_hip.o
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
Band_DFT_Col.o: Band_DFT_Col.c openmx_common.h eigen_refine_gpu.h eigen_refine_hooks.h openmx_precision.h
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
Band_DFT_NonCol.o: Band_DFT_NonCol.c openmx_common.h eigen_refine_gpu.h eigen_refine_hooks.h openmx_precision.h
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
Mixing_H.o: Mixing_H.c openmx_common.h
	$(CC) -c Mixing_H.c
Mixing_V.o: Mixing_V.c openmx_common.h
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

Set_Orbitals_Grid.o: Set_Orbitals_Grid.c openmx_common.h set_hip_default_device_from_local_rank.h orbs_otf_hip.h openmx_precision.h
	$(CC) -c Set_Orbitals_Grid.c
Set_Density_Grid.o: Set_Density_Grid.c openmx_common.h
	$(CC_NOACC_O3) $(SDG_CPU_BASELINE_FLAGS) -c Set_Density_Grid.c
Set_Density_Grid_GPU.o: Set_Density_Grid_GPU.c openmx_common.h set_hip_default_device_from_local_rank.h
	$(CC) -c Set_Density_Grid_GPU.c
Set_Aden_Grid.o: Set_Aden_Grid.c openmx_common.h
	$(CC) -c Set_Aden_Grid.c

KumoF.o: KumoF.c openmx_common.h
	$(CC) -c KumoF.c
Dr_KumoF.o: Dr_KumoF.c openmx_common.h
	$(CC) -c Dr_KumoF.c
Pot_NeutralAtom.o: Pot_NeutralAtom.c openmx_common.h
	$(CC) -c Pot_NeutralAtom.c
Simple_Mixing_DM.o: Simple_Mixing_DM.c openmx_common.h
	$(CC) -c Simple_Mixing_DM.c
DIIS_Mixing_DM.o: DIIS_Mixing_DM.c openmx_common.h
	$(CC) -c DIIS_Mixing_DM.c
ADIIS_Mixing_DM.o: ADIIS_Mixing_DM.c openmx_common.h
	$(CC) -c ADIIS_Mixing_DM.c
GR_Pulay_DM.o: GR_Pulay_DM.c openmx_common.h
	$(CC) -c GR_Pulay_DM.c
Kerker_Mixing_Rhok.o: Kerker_Mixing_Rhok.c openmx_common.h
	$(CC) -c Kerker_Mixing_Rhok.c
DIIS_Mixing_Rhok.o: DIIS_Mixing_Rhok.c openmx_common.h
	$(CC) -c DIIS_Mixing_Rhok.c
# Same LDA+U Type2/cFLL workaround as above.
Total_Energy.o: Total_Energy.c openmx_common.h set_hip_default_device_from_local_rank.h
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


Set_XC_Grid.o: Set_XC_Grid.c openmx_common.h set_hip_default_device_from_local_rank.h
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
gpusolver_Syevd.o: gpusolver_Syevd.c openmx_common.h set_hip_default_device_from_local_rank.h
	$(CC) -c gpusolver_Syevd.c
gpusolver_Syevdx.o: gpusolver_Syevdx.c openmx_common.h openmx_stdout_level.h
	$(CC) -c gpusolver_Syevdx.c
set_hip_default_device_from_local_rank.o: set_hip_default_device_from_local_rank.c openmx_common.h set_hip_default_device_from_local_rank.h
	$(CC) -c set_hip_default_device_from_local_rank.c
set_openmp_device_from_local_rank.o: set_openmp_device_from_local_rank.c openmx_common.h set_openmp_device_from_local_rank.h set_hip_default_device_from_local_rank.h
	$(CC) -c set_openmp_device_from_local_rank.c
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
Set_OLP_Kin.o: Set_OLP_Kin.c openmx_common.h set_hip_default_device_from_local_rank.h openmx_core_overlap.h openmx_core_pair_bessel.h openmx_complex.h openmx_core_orbitals.h
	$(CC) -c Set_OLP_Kin.c
Set_Nonlocal.o: Set_Nonlocal.c openmx_common.h openmx_core_pair_bessel.h openmx_nonlocal_cpu.h openmx_core_nonlocal.h openmx_core_orbitals.h
	$(CC) -c Set_Nonlocal.c
Set_ProExpn_VNA.o: Set_ProExpn_VNA.c openmx_common.h openmx_vna_status.h openmx_precision.h openmx_core_vna.h openmx_core_orbitals.h openmx_vna_cpu.h openmx_vna_gpu_context.h openmx_core_hvna.h openmx_core_hvna_workspace.h openmx_hvna_cpu.h openmx_complex.h openmx_core_vna_preparation.h openmx_vna_preparation.h openmx_core_dsvna.h openmx_dsvna_cpu.h
	$(CC) -c Set_ProExpn_VNA.c
Set_CoreHoleMatrix.o: Set_CoreHoleMatrix.c openmx_common.h
	$(CC) -c Set_CoreHoleMatrix.c
Set_OLP_p.o: Set_OLP_p.c openmx_common.h 
	$(CC) -c Set_OLP_p.c
Set_Hamiltonian.o: Set_Hamiltonian.c openmx_common.h set_hip_default_device_from_local_rank.h
	$(CC) $(SETH_CPU_BASELINE_FLAGS) -c Set_Hamiltonian.c
set_hamiltonian_hip.o: set_hamiltonian_hip.cu openmx_stdout_level.h openmx_precision.h
	$(HIPCC) $(HIPCC_GEMMUL8_FLAGS) -c set_hamiltonian_hip.cu -o set_hamiltonian_hip.o
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
my_hipblasDgemm.o: my_hipblasDgemm.c openmx_common.h
	$(CC) -c my_hipblasDgemm.c
my_hipblasZgemm.o: my_hipblasZgemm.c openmx_common.h
	$(CC) -c my_hipblasZgemm.c
gemmul8_bridge.o: gemmul8_bridge.cu openmx_common.h $(GEMMUL8_DIR)/include/gemmul8.hpp $(GEMMUL8_LIB) openmx_stdout_level.h openmx_gpu_arch_policy.h
	$(HIPCC) $(HIPCC_GEMMUL8_FLAGS) -c gemmul8_bridge.cu -o gemmul8_bridge.o
openmx_precision.o: openmx_precision.c openmx_precision.h
	$(CC_NOACC) -c openmx_precision.c

eigen_refine_gpu.o: eigen_refine_gpu.cu eigen_refine_gpu.h hip_runtime_compat.h hipblas_compat.h openmx_stdout_level.h openmx_precision.h openmx_native_resources.h openmx_core_native_resources.h
	$(HIPCC) $(HIPCC_GEMMUL8_FLAGS) $(MPI_CFLAGS) -c eigen_refine_gpu.cu -o eigen_refine_gpu.o
# double-float arithmetic: no floating-point contraction (-ffp-contract=off in the flags)
orbs_otf_hip.o: orbs_otf_hip.cu orbs_otf_hip.h openmx_precision.h
	$(HIPCC) $(HIPCC_GEMMUL8_FLAGS) -c orbs_otf_hip.cu -o orbs_otf_hip.o

set_xc_grid_hip.o: set_xc_grid_hip.cu
	$(HIPCC) $(HIPCC_GEMMUL8_FLAGS) -c set_xc_grid_hip.cu -o set_xc_grid_hip.o

gpusolver2-stack: $(GPUSOLVER2_STAMP)

$(GPUSOLVER2_STAMP): $(GPUSOLVER2_SCRIPT)
	GPUSOLVER2_DIST="$(GPUSOLVER2_DIST)" GPUSOLVER2_BUILD="$(GPUSOLVER2_BUILD)" \
	GPUSOLVER2_PREFIX="$(GPUSOLVER2_PREFIX)" \
	GPUSOLVER2_HOST_CC="$(GPUSOLVER2_HOST_CC)" GPUSOLVER2_HOST_CXX="$(GPUSOLVER2_HOST_CXX)" \
	GPUSOLVER2_HIPCC="$(GPUSOLVER2_HIPCC)" GPUSOLVER2_GPU_ARCH="$(GPUSOLVER2_GPU_ARCH)" \
	GPUSOLVER2_MPI_BIN="$(GPUSOLVER2_MPI_BIN)" GPUSOLVER2_MPI_LIBDIR="$(GPUSOLVER2_MPI_LIBDIR)" \
	GPUSOLVER2_ROCM="$(GPUSOLVER2_ROCM)" \
	GPUSOLVER2_AOCL_LIB="$(GPUSOLVER2_AOCL_LIB)" GPUSOLVER2_AOCL_INC="$(GPUSOLVER2_AOCL_INC)" \
	GPUSOLVER2_AOCC_LIB="$(GPUSOLVER2_AOCC_LIB)" \
	GPUSOLVER2_SCALAPACK_SO="$(GPUSOLVER2_SCALAPACK_SO)" \
	GPUSOLVER2_ELPA_VER="$(GPUSOLVER2_ELPA_VER)" \
	GPUSOLVER2_CMAKE="$(GPUSOLVER2_CMAKE)" GPUSOLVER2_JOBS="$(GPUSOLVER2_JOBS)" \
	sh "$(GPUSOLVER2_SCRIPT)"

gpusolver2-clean:
	rm -rf "$(GPUSOLVER2_BUILD)" "$(GPUSOLVER2_PREFIX)"

openmx_elpa_cpu.o: openmx_elpa_cpu.c openmx_elpa_cpu.h openmx_common.h $(GPUSOLVER2_STAMP)
	$(CC_NOACC_O3) -I$(GPUSOLVER2_ELPA_INCLUDE) -c openmx_elpa_cpu.c

elpa_cosma_bridge.o: elpa_cosma_bridge.c elpa_cosma_bridge.h $(GPUSOLVER2_STAMP) openmx_stdout_level.h
	$(CC_NOACC_O3) -I$(GPUSOLVER2_ELPA_INCLUDE) -c elpa_cosma_bridge.c
# Reuse the pinned, common GEMMul8 source checkout without writing GPU objects
# into it.  A source archive can provide this tree without .git metadata.
$(GEMMUL8_DIR)/Makefile:
	mkdir -p "$(GEMMUL8_DIR)"
	@if [ -f "$(GEMMUL8_SOURCE_DIR)/Makefile" ]; then \
	  tar -C "$(GEMMUL8_SOURCE_DIR)" --exclude=.git --exclude=lib --exclude='*.o' --exclude='*.d' --exclude='*.a' -cf - . | tar -C "$(GEMMUL8_DIR)" -xf -; \
	else \
	  cd "$(GEMMUL8_DIR)" && \
	  { [ -d .git ] || git init -q .; } && \
	  { git remote get-url origin >/dev/null 2>&1 || git remote add origin "$(GEMMUL8_REPO_URL)"; } && \
	  git fetch --depth 1 origin "$(GEMMUL8_COMMIT)" && \
	  git checkout -q "$(GEMMUL8_COMMIT)"; \
	fi
$(GEMMUL8_DIR)/include/gemmul8.hpp $(GEMMUL8_CONFIG_SRC): | $(GEMMUL8_DIR)/Makefile
	@test -f "$@"
$(GEMMUL8_OPENMX_OBJ): gemmul8_openmx.cu $(GEMMUL8_DIR)/Makefile makefile
	$(HIPCC) $(HIPCC_GEMMUL8_FLAGS) -MMD -MP -MF $(@:.o=.d) -c gemmul8_openmx.cu -o $(GEMMUL8_OPENMX_OBJ)
$(GEMMUL8_CONFIG_OBJ): $(GEMMUL8_CONFIG_SRC) $(GEMMUL8_DIR)/Makefile makefile
	$(HIPCC) $(HIPCC_GEMMUL8_FLAGS) -MMD -MP -MF $(@:.o=.d) -c $(GEMMUL8_CONFIG_SRC) -o $(GEMMUL8_CONFIG_OBJ)
$(GEMMUL8_LIB): $(GEMMUL8_OPENMX_OBJ) $(GEMMUL8_CONFIG_OBJ)
	mkdir -p $(GEMMUL8_DIR)/lib
	ar rcs $(GEMMUL8_LIB) $(GEMMUL8_OPENMX_OBJ) $(GEMMUL8_CONFIG_OBJ)
$(MAGMA_ARCHIVE):
	@echo "Bundled MAGMA archive was not found: $(MAGMA_ARCHIVE)"
	@echo "Place magma-$(MAGMA_VERSION).tar.gz from https://icl.utk.edu/magma/ under $(MAGMA_DIR)/, or override MAGMA_ARCHIVE."
	@false

$(MAGMA_MAKE_INC):
	@echo "OpenMX MAGMA make.inc template was not found: $(MAGMA_MAKE_INC)"
	@false

$(MAGMA_STAMP): $(MAGMA_ARCHIVE) $(MAGMA_MAKE_INC) $(MAGMA_OZAKI_PATCH)
	mkdir -p "$(MAGMA_DIR)"
	find "$(MAGMA_DIR)" -mindepth 1 -maxdepth 1 ! -name "$(MAGMA_ARCHIVE_BASENAME)" -exec rm -rf {} +
	tar -xzf "$(MAGMA_ARCHIVE)" -C "$(MAGMA_DIR)" --strip-components=1
	ln -sf Makefile.gen.hip "$(MAGMA_DIR)/make.gen.hipMAGMA"
	sed -i 's/^\([[:space:]]*VALID_GFXS =.*\)/\1 $(MAGMA_EXTRA_GFXS)/' "$(MAGMA_DIR)/Makefile"
	install -m 0644 "$(MAGMA_MAKE_INC)" "$(MAGMA_DIR)/make.inc"
	patch -d "$(MAGMA_DIR)" -p1 < "$(MAGMA_OZAKI_PATCH)"
	sleep 1
	touch "$(MAGMA_DIR)/interface_hip/blas_s_v2.cpp" "$(MAGMA_DIR)/interface_hip/blas_c_v2.cpp" \
	      "$(MAGMA_DIR)/interface_hip/blas_d_v2.cpp" "$(MAGMA_DIR)/src/slaex3.cpp" \
	      "$(MAGMA_DIR)/src/ssyevdx_ozaki2.cpp" "$(MAGMA_DIR)/src/ssyevdx_ozaki2_gpu.cpp" \
	      "$(MAGMA_DIR)/src/dsyevdx_ozaki2.cpp" "$(MAGMA_DIR)/src/dsyevdx_ozaki2_gpu.cpp" \
	      "$(MAGMA_DIR)/src/cheevdx_ozaki2.cpp" "$(MAGMA_DIR)/src/cheevdx_ozaki2_gpu.cpp" \
	      "$(MAGMA_DIR)/testing/testing_ssyevdx_ozaki2.cpp" "$(MAGMA_DIR)/testing/testing_dsyevdx_ozaki2.cpp" \
	      "$(MAGMA_DIR)/testing/testing_cheevdx_ozaki2.cpp"
	rm -f magma_openmx.o
	touch "$@"

$(MAGMA_DIR)/Makefile $(MAGMA_DIR)/make.inc: $(MAGMA_STAMP)

$(MAGMA_DEVICE_MAP_STAMP): $(MAGMA_STAMP) $(MAGMA_DEVICE_MAP_SCRIPT) $(MAGMA_DEVICE_MAP_HEADER)
	python3 "$(MAGMA_DEVICE_MAP_SCRIPT)" "$(MAGMA_DIR)"
	touch "$@"

$(MAGMA_LIB): $(MAGMA_STAMP) $(MAGMA_DEVICE_MAP_STAMP)
	$(MAKE) -C "$(MAGMA_DIR)" BACKEND=hip ROCM_PATH="$(ROCM)" GPU_TARGET="$(MAGMA_GPU_TARGET)" AOCL_LIB="$(AOCL_LIB)" HIPCC="$(HIPCC)" CC="$(HIPCC)" CXX="$(HIPCC)" DEVCC="$(HIPCC)" ARCH=ar lib/libmagma.a
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

clean: fftw3-clean gemmul8-clean magma-clean
	rm -f $(PROG) $(OBJS) $(OBJS_jx) $(UTIL) *.o *.mod *.dbg

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

# The local instantiation TU includes upstream implementation headers.
# Track them so a submodule update also rebuilds the embedded kernels.
-include $(GEMMUL8_DEPFILES)

# Invocation-owned projector communication and shared VNA orchestration.
Force.o Stress.o Set_Nonlocal.o Set_CoreHoleMatrix.o Set_ProExpn_VNA.o: openmx_projector_exchange.h openmx_core_projector_exchange.h
Set_ProExpn_VNA.o: openmx_vna_assembly.h openmx_core_vna_invocation.h openmx_core_atom_pairs.h openmx_vna_tables.h openmx_vna_status.h

# Shared flat orbital optimization and Hubbard force boundaries.
DFT.o Opt_Contraction.o: openmx_orbital_optimization.h openmx_core_orbital_optimization.h
Force.o: openmx_hubbard_force.h
Force.o: openmx_force_workspace.h openmx_core_force.h openmx_force_stages.h \
         openmx_solver_workspace.h openmx_core_solver_workspace.h \
         openmx_core_orbital_workspace.h openmx_core_local_basis.h
Force.o: openmx_force_projectors.h openmx_core_core_hole.h
Force.o: openmx_core_projector_force.h
Force.o: openmx_force_hnl_gpu.h openmx_native_resources.h openmx_core_native_resources.h

# Solver invocation RAII and flat spectral workspaces.
Cluster_DFT_Col.o Cluster_DFT_NonCol.o Band_DFT_Col.o Band_DFT_NonCol.o Divide_Conquer.o Divide_Conquer_LNO.o Krylov.o: openmx_solver_workspace.h openmx_core_solver_workspace.h

# Calculation-owned native GPU resources.
DFT.o Cluster_DFT_Col.o Cluster_DFT_NonCol.o Band_DFT_Col.o Band_DFT_NonCol.o Divide_Conquer.o Divide_Conquer_LNO.o Krylov.o: openmx_native_resources.h openmx_core_native_resources.h
Free_Arrays.o: openmx_core_native_resources.h

openmx_grid_assembly.o: openmx_grid_assembly.c openmx_grid_assembly.h openmx_core_grid.h openmx_common.h
	$(CC_NOACC_O3) -c openmx_grid_assembly.c
DFT.o Set_Hamiltonian.o Set_Density_Grid.o Free_Arrays.o: openmx_grid_assembly.h openmx_core_grid.h
