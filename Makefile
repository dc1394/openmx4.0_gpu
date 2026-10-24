# Unified entry point. The existing NVIDIA makefile remains at source/Makefile.
BACKEND ?= CUDA
BUILD_DIR ?= build/$(shell printf '%s' '$(BACKEND)' | tr '[:upper:]' '[:lower:]')
JOBS ?= 4
CMAKE ?= cmake
CMAKE_OPTIONS ?=
CLEAN_TARGET = $(if $(filter CPU cpu HOST host NONE none,$(BACKEND)),clean,native-clean)

.PHONY: all configure test clean install
all: configure
	$(CMAKE) --build "$(BUILD_DIR)"

configure:
	$(CMAKE) -S . -B "$(BUILD_DIR)" -DOPENMX_GPU_BACKEND="$(BACKEND)" -DOPENMX_BUILD_JOBS="$(JOBS)" $(CMAKE_OPTIONS)

test: configure
	ctest --test-dir "$(BUILD_DIR)" --output-on-failure

clean:
	$(CMAKE) --build "$(BUILD_DIR)" --target $(CLEAN_TARGET)

install: all
	$(CMAKE) --install "$(BUILD_DIR)"
