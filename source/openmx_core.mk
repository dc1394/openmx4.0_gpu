# Shared C++ core build for direct Makefile users. The top-level CMake driver
# supplies OPENMX_CORE_LIBRARY and manages this dependency itself.
OPENMX_CORE_SOURCE_DIR ?= $(abspath $(dir $(realpath openmx_core.mk))/../core)
OPENMX_CORE_BUILD_DIR ?= $(CURDIR)/build/cpp-core
OPENMX_CORE_CMAKE ?= $(or $(OPENMX_CALLER_CMAKE),cmake)
OPENMX_CORE_CXX ?= g++
OPENMX_CORE_BACKEND ?= SERIAL
OPENMX_CORE_CMAKE_OPTIONS ?=

ifeq ($(strip $(OPENMX_CORE_LIBRARY)),)
OPENMX_CORE_LIBRARY := $(OPENMX_CORE_BUILD_DIR)/libopenmx_core.so
# Always let CMake check configuration and dependencies, including changes in
# command-line backend/options. Its incremental build leaves the library's
# timestamp unchanged when no work is needed, avoiding unnecessary relinks.
.PHONY: openmx-core
$(OPENMX_CORE_LIBRARY): openmx-core
	@test -f "$@"
openmx-core:
	"$(OPENMX_CORE_CMAKE)" -S "$(OPENMX_CORE_SOURCE_DIR)" -B "$(OPENMX_CORE_BUILD_DIR)" \
	  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$(OPENMX_CORE_CXX)" \
	  -DOPENMX_CORE_BACKEND=$(OPENMX_CORE_BACKEND) -DBUILD_TESTING=OFF $(OPENMX_CORE_CMAKE_OPTIONS)
	"$(OPENMX_CORE_CMAKE)" --build "$(OPENMX_CORE_BUILD_DIR)" --parallel $(GPUSOLVER2_JOBS)
endif

# work/openmx uses work/lib; an explicit installation prefix uses bin/../lib.
LIB += $(OPENMX_CORE_LIBRARY) -Wl,-rpath,$(dir $(OPENMX_CORE_LIBRARY)) -Wl,-rpath,'$$ORIGIN/lib' -Wl,-rpath,'$$ORIGIN/../lib'
openmx: $(OPENMX_CORE_LIBRARY)
$(OBJS): openmx_core.h openmx_periodic.h
FT_PAO.o FT_NLP.o FT_VNA.o RadialF.o Nonlocal_RadialF.o VNAF.o PhiF.o RF_BesselF.o Set_Nonlocal.o: openmx_core_radial.h
SetPara_DFT.o: openmx_gpu_device.h
SetPara_DFT.o FT_ProExpn_VNA.o: openmx_core_neutral.h openmx_core_radial.h
Allocate_Arrays.o Free_Arrays.o: openmx_core_pao.h openmx_core_vps.h openmx_core_product.h
FT_ProductPAO.o: openmx_core_product.h openmx_core_radial.h
