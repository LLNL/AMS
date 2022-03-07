blueos_3_ppc64le_ib_p9.cxx      = nvcc
blueos_3_ppc64le_ib_p9.cxxflags = -O3 -std=c++11 -x=cu --expt-extended-lambda -arch=sm_70 -ccbin g++
blueos_3_ppc64le_ib_p9.cfg      = MFEM_USE_CUDA=YES

use-if-def = $(if $1,$1,$2)

cxx      ?= $(call use-if-def,$($(SYS_TYPE).cxx),g++)
cxxflags ?= $(call use-if-def,$($(SYS_TYPE).cxxflags),-O3 -std=c++11)
cfg      ?= $(call use-if-def,$($(SYS_TYPE).cfg))

mfem.dir    = mfem
mfem.exists = $(mfem.dir)/makefile
mfem.prefix = $(realpath .)/build/$(SYS_TYPE)
mfem.lib    = $(mfem.prefix)/lib/libmfem.a
mfem.build  = $(mfem.prefix)/build

exe = mmp-$(SYS_TYPE)
src = src/main.cpp

.DEFAULT_GOAL := $(exe)

# the timestamp on the dir itself causes repeated builds
$(mfem.exists):
	git clone https://github.com/mfem/mfem.git $(mfem.dir)

$(mfem.lib): $(mfem.exists)
	$(MAKE) -C mfem config $(mfem.opts) MFEM_STATIC=YES MFEM_CXX=$(cxx) MFEM_CXXFLAGS="$(cxxflags)" BUILD_DIR=$(mfem.build) PREFIX=$(mfem.prefix) $(cfg)
	$(MAKE) -C $(mfem.build) install

$(exe): $(src) $(mfem.lib)
	$(cxx) $(cxxflags) $< -I$(mfem.prefix)/include -L$(mfem.prefix)/lib -lmfem -o $@

clean:
	$(MAKE) -C mfem distclean
	rm -f $(exe)
	rm -rf $(mfem.prefix)

format:
	clang-format -i $(src)

info:
	@echo "compiler = $(cxx)"
	@echo "   flags = $(cxxflags)"
	@echo "     cfg = $(cfg)"
	@echo "  prefix = $(mfem.prefix)"
