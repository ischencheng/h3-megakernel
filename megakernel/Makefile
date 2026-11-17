ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3
PYTHON_CONFIG ?= $(PYTHON)-config
NVCC ?= nvcc
GPU ?= H100
THUNDERKITTENS_ROOT ?= $(ROOT)third_party/ThunderKittens
CUTLASS_ROOT ?= $(ROOT)third_party/CUTLASS
BUILD_DIR := $(ROOT)build/$(GPU)
TARGET := $(BUILD_DIR)/mk_h3$(shell $(PYTHON_CONFIG) --extension-suffix)
CUTLASS_TARGET := $(BUILD_DIR)/cutlass_projection$(shell $(PYTHON_CONFIG) --extension-suffix)
HAZY_PROJECTION_TARGET := $(BUILD_DIR)/mk_h3_projection$(shell $(PYTHON_CONFIG) --extension-suffix)

ifeq ($(GPU),H100)
ARCH_FLAGS := -DKITTENS_HOPPER -arch=sm_90a
else ifeq ($(GPU),B200)
ARCH_FLAGS := -DKITTENS_HOPPER -DKITTENS_BLACKWELL -gencode=arch=compute_100a,code=sm_100a
else ifeq ($(GPU),RTXPRO6000)
# The SM120 smoke runtime uses Hopper's semaphore/TMA headers but never
# instantiates the SM100 tensor-memory path. H3 tensor ops need separate ports.
ARCH_FLAGS := -DKITTENS_HOPPER -DH3_SM120 -arch=sm_120a
else
$(error GPU=$(GPU) is not supported. Use H100, B200, or RTXPRO6000)
endif

.PHONY: all bootstrap smoke graph cutlass_projection hazy_projection clean
all: $(TARGET)
bootstrap:
	$(PYTHON) $(ROOT)scripts/bootstrap.py

$(TARGET): $(wildcard $(ROOT)src/*.cu $(ROOT)src/*.cuh $(ROOT)src/ops/*.cuh $(ROOT)include/*.cuh $(ROOT)include/controller/*.cuh)
	@test -f "$(THUNDERKITTENS_ROOT)/include/kittens.cuh" || (echo "Run make bootstrap first"; exit 1)
	@$(PYTHON) -c "import pybind11"
	@mkdir -p "$(BUILD_DIR)"
	$(NVCC) $(ROOT)src/h3.cu -o $@ -std=c++20 -O3 --use_fast_math \
		--expt-extended-lambda --expt-relaxed-constexpr -lineinfo \
		-shared -Xcompiler=-fPIC $(ARCH_FLAGS) \
		-I$(ROOT)include -I$(THUNDERKITTENS_ROOT)/include \
		$(shell $(PYTHON) -m pybind11 --includes) $(shell $(PYTHON_CONFIG) --ldflags) -lcuda
smoke: all
	cd $(BUILD_DIR) && PYTHONPATH="$(BUILD_DIR):$$PYTHONPATH" $(PYTHON) $(ROOT)tests/smoke.py
graph: all
	cd $(BUILD_DIR) && PYTHONPATH="$(BUILD_DIR):$$PYTHONPATH" $(PYTHON) $(ROOT)tests/graph.py
ifeq ($(GPU),RTXPRO6000)
$(HAZY_PROJECTION_TARGET): $(ROOT)src/h3_projection.cu $(wildcard $(ROOT)src/*.cuh $(ROOT)src/ops/*.cuh $(ROOT)include/*.cuh $(ROOT)include/controller/*.cuh)
	@test -f "$(THUNDERKITTENS_ROOT)/include/kittens.cuh" || (echo "Run make bootstrap first"; exit 1)
	@test -f "$(CUTLASS_ROOT)/include/cutlass/gemm/kernel/gemm_universal.hpp" || (echo "Run scripts/bootstrap_cutlass.py first"; exit 1)
	@$(PYTHON) -c "import pybind11"
	@mkdir -p "$(BUILD_DIR)"
	$(NVCC) $(ROOT)src/h3_projection.cu -o $@ -std=c++20 -O3 -arch=sm_120a \
		--expt-extended-lambda --expt-relaxed-constexpr -lineinfo \
		-shared -Xcompiler=-fPIC -diag-suppress=20013,20015,2908 \
		-DKITTENS_HOPPER -DH3_SM120 \
		-I$(ROOT)include -I$(THUNDERKITTENS_ROOT)/include \
		-I$(CUTLASS_ROOT)/include -I$(CUTLASS_ROOT)/tools/util/include \
		$(shell $(PYTHON) -m pybind11 --includes) $(shell $(PYTHON_CONFIG) --ldflags) -lcuda -lcudart
hazy_projection: $(HAZY_PROJECTION_TARGET)
$(CUTLASS_TARGET): $(ROOT)src/cutlass_projection.cu
	@test -f "$(CUTLASS_ROOT)/include/cutlass/gemm/kernel/gemm_universal.hpp" || (echo "Run scripts/bootstrap_cutlass.py first"; exit 1)
	@$(PYTHON) -c "import pybind11"
	@mkdir -p "$(BUILD_DIR)"
	$(NVCC) $< -o $@ -std=c++20 -O3 -arch=sm_120a -shared -Xcompiler=-fPIC \
		--expt-relaxed-constexpr --expt-extended-lambda -diag-suppress=20013,20015,2908 \
		-I$(CUTLASS_ROOT)/include -I$(CUTLASS_ROOT)/tools/util/include \
		$(shell $(PYTHON) -m pybind11 --includes) $(shell $(PYTHON_CONFIG) --ldflags) -lcudart
cutlass_projection: $(CUTLASS_TARGET)
else
cutlass_projection:
	@echo "cutlass_projection requires GPU=RTXPRO6000"; exit 1
hazy_projection:
	@echo "hazy_projection requires GPU=RTXPRO6000"; exit 1
endif
clean:
	rm -f $(TARGET) $(CUTLASS_TARGET) $(HAZY_PROJECTION_TARGET)
