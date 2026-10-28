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
# The post-processing tools that only need the keyword-file reader link the
# io package's archive directly (the application gets it through the core).
OPENMX_IO_LIBRARY ?= $(dir $(OPENMX_CORE_LIBRARY))modules/io/libopenmx_io.a
$(OBJS): openmx_core.h openmx_periodic.h openmx_complex.h
# Track the canonical public headers behind source/ compatibility includes.
$(OBJS) openmx_api.o: $(wildcard $(OPENMX_CORE_SOURCE_DIR)/include/*.h)
FT_PAO.o FT_NLP.o FT_VNA.o RadialF.o Nonlocal_RadialF.o VNAF.o PhiF.o RF_BesselF.o Set_Nonlocal.o: openmx_core_radial.h
SetPara_DFT.o: openmx_gpu_device.h
SetPara_DFT.o FT_ProExpn_VNA.o: openmx_core_neutral.h openmx_core_radial.h
Allocate_Arrays.o Free_Arrays.o: openmx_core_pao.h openmx_core_vps.h openmx_core_product.h
FT_ProductPAO.o: openmx_core_product.h openmx_core_radial.h

# Calculation-scoped ragged matrices are visible through openmx_common.h.
$(OBJS): openmx_core_vna_matrix.h
Set_ProExpn_VNA.o: openmx_projector_matrix.h
DFT.o truncation.o: openmx_vna_matrix.h
openmx_api.o: openmx_common.h openmx_core.h openmx_core_vna_matrix.h

$(OBJS) openmx_api.o: openmx_core_species.h openmx_core_atomic.h
$(OBJS) openmx_api.o: openmx_core_reciprocal.h
$(OBJS) openmx_api.o: openmx_core_reciprocal_archive.h
DFT.o FT_PAO.o FT_NLP.o FT_ProExpn_VNA.o FT_VNA.o FT_ProductPAO.o SetPara_DFT.o Generate_Wannier.o: openmx_reciprocal.h
DFT.o FT_PAO.o FT_NLP.o FT_ProExpn_VNA.o FT_VNA.o FT_ProductPAO.o SetPara_DFT.o: openmx_atomic.h
# Public native views use the same C complex type as the standalone module.
$(OBJS) openmx_api.o: openmx_core_real_harmonics.h ../modules/basis/include/openmx/basis/real_harmonics.h ../modules/basis/include/openmx/basis/complex.h

$(OBJS) openmx_api.o: openmx_core_orbital_workspace.h
Get_Orbitals.o Get_Cnt_Orbitals.o Get_dOrbitals.o Get_Cnt_dOrbitals.o Set_Orbitals_Grid.o Force.o Stress.o: openmx_orbitals.h
Get_Orbitals.o Get_Cnt_Orbitals.o Get_dOrbitals.o Get_Cnt_dOrbitals.o Set_Orbitals_Grid.o Force.o Stress.o: openmx_orbital_harmonics.h
# Shared angular kernels are inlined by both native backends.
openmx_common.o Get_Orbitals.o Get_Cnt_Orbitals.o Get_dOrbitals.o Get_Cnt_dOrbitals.o Set_Orbitals_Grid.o Force.o Stress.o: openmx_basis_harmonics.h ../modules/basis/include/openmx/basis/detail/harmonics.h
openmx.o openmx_api.o Runtest.o: openmx_atomic.h

# These shared native helpers contain inline numerical/ownership code outside
# core/include. Track them in both GPU builds as well as the public C ABI.
Total_Energy.o Stress.o: openmx_neutral_atom.h
truncation.o: openmx_core_olg_cache.h ../core/include/openmx_core_olg_cache.h
Cluster_DFT_Col_CWF.o Cluster_DFT_NonCol_CWF.o Band_DFT_Col_CWF.o Band_DFT_NonCol_CWF.o: openmx_cwf_workspace.h

# The angular-rule view crosses the Energy CPU/native backend boundary.
Total_Energy.o Total_Energy_GPU.o Stress.o: openmx_basis_lebedev.h ../modules/basis/include/openmx/basis/lebedev.h openmx_xc_workspace.h openmx_energy_workspace.h

$(OBJS) openmx_api.o: openmx_basis_gaunt.h ../modules/basis/include/openmx/basis/detail/gaunt.h ../modules/basis/include/openmx/basis/gaunt.h

$(OBJS) openmx_api.o: openmx_core_vna_simd_workspace.h ../core/include/openmx_core_vna_simd_workspace.h

$(OBJS) openmx_api.o: openmx_core_radial_views.h ../core/include/openmx_core_radial_views.h openmx_core_vna_host_tables.h ../core/include/openmx_core_vna_host_tables.h

$(OBJS) openmx_api.o: openmx_basis_orbital_radial.h ../modules/basis/include/openmx/basis/orbital_radial.h ../modules/basis/include/openmx/basis/detail/orbital_radial.h
Set_XC_Grid.o Force.o Stress.o Total_Energy.o: openmx_xc.h ../modules/xc/include/openmx/xc/xc.h
Inputtools.o Input_std.o: openmx_io.h ../modules/io/include/openmx/io/keyword_file.h ../modules/io/include/openmx/io/basis_spec.h
Set_XC_Grid.o: openmx_xc_kernels.h ../modules/xc/include/openmx/xc/kernels.h
