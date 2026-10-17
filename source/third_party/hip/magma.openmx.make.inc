# MAGMA configuration for OpenMX AMDGPU builds.

BACKEND      = hip
ROCM_PATH   ?= /opt/rocm-7.2.0
GPU_TARGET  ?= gfx942

CC           = $(ROCM_PATH)/bin/hipcc
CXX          = $(ROCM_PATH)/bin/hipcc
FORT         =
HIPCC        = $(ROCM_PATH)/bin/hipcc
DEVCC        = $(HIPCC)

ARCH         ?= ar
ARCHFLAGS    ?= cr
RANLIB       ?= ranlib

FPIC         = -fPIC
FOPENMP      = -fopenmp

CFLAGS       = -O3 $(FPIC) $(FOPENMP) -DNDEBUG -DADD_ -Wall -std=c99
CXXFLAGS     = -O3 $(FPIC) $(FOPENMP) -DNDEBUG -DADD_ -Wall -std=c++14
FFLAGS       = -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument
F90FLAGS     = -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument -x f95-cpp-input
LDFLAGS      = $(FPIC) $(FOPENMP)

DEVCCFLAGS   = -O3 -DNDEBUG -DADD_ $(FPIC) $(FOPENMP) -std=c++14 -fno-gpu-rdc

AOCL_LIB    ?= /system/apps/rhel/aocl/5.1.0/aocc/lib

LIB          =
LIB         += -lflame -lblis -laoclutils
LIB         += -lhipblas -lhipsparse -lrocblas -lamdhip64

LIBDIR      += -L$(AOCL_LIB)
LIBDIR      += -L$(ROCM_PATH)/lib

INC         += -I$(ROCM_PATH)/include

# GEMMul8 (Ozaki Scheme II) for the _ozaki2 eigensolver adapter
INC += -I../GEMMul8/include
LIBDIR += -L../GEMMul8/lib
LIB += -lgemmul8 -lhipblaslt
