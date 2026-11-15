ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3
PYTHON_CONFIG ?= $(PYTHON)-config
NVCC ?= nvcc
GPU ?= H100
THUNDERKITTENS_ROOT ?= $(ROOT)third_party/ThunderKittens
TARGET := $(ROOT)mk_h3$(shell $(PYTHON_CONFIG) --extension-suffix)

ifeq ($(GPU),H100)
ARCH_FLAGS := -DKITTENS_HOPPER -arch=sm_90a
else ifeq ($(GPU),B200)
ARCH_FLAGS := -DKITTENS_HOPPER -DKITTENS_BLACKWELL -arch=sm_100a
else
$(error GPU=$(GPU) is not supported by the upstream runtime; SM120 requires a port. Use H100 or B200 for the reference smoke test)
endif

.PHONY: all bootstrap smoke clean
all: $(TARGET)
bootstrap:
	$(PYTHON) $(ROOT)scripts/bootstrap.py

$(TARGET): $(wildcard $(ROOT)src/*.cu $(ROOT)src/*.cuh $(ROOT)src/ops/*.cuh $(ROOT)include/*.cuh $(ROOT)include/controller/*.cuh)
	@test -f "$(THUNDERKITTENS_ROOT)/include/kittens.cuh" || (echo "Run make bootstrap first"; exit 1)
	@$(PYTHON) -c "import pybind11"
	$(NVCC) $(ROOT)src/h3.cu -o $@ -std=c++20 -O3 --use_fast_math \
		--expt-extended-lambda --expt-relaxed-constexpr -lineinfo \
		-shared -Xcompiler=-fPIC $(ARCH_FLAGS) \
		-I$(ROOT)include -I$(THUNDERKITTENS_ROOT)/include \
		$(shell $(PYTHON) -m pybind11 --includes) $(shell $(PYTHON_CONFIG) --ldflags) -lcuda
smoke: all
	cd $(ROOT) && PYTHONPATH="$(ROOT):$$PYTHONPATH" $(PYTHON) tests/smoke.py
clean:
	rm -f $(TARGET)
