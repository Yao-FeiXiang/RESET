GRAPH_DATASET_NAME = bm #sc18,sc19,sc20,bm,gp,wt
IR_DATASET_NAME = ms #ce,fe,hp,lt,ms,nq

# Parallel compilation: use make -jN or set MAKEFLAGS
MAKEFLAGS += -j$(shell nproc)

CCACHE := $(shell command -v ccache 2> /dev/null)

# Line info: default off, enable with LINEINFO=1
LINEINFO ?= 0

CUDA_HOME ?= /home/cliu26/cuda-12.6              
# CUCO_ROOT := ../baseline/cuCollections

CUCO_ROOT := ./cuCollections
CUCO_BUILD := $(CUCO_ROOT)/build
CCCL_SRC  := $(CUCO_BUILD)/_deps/cccl-src
CCCL_LIBCUDACXX := $(CCCL_SRC)/libcudacxx/include
CCCL_THRUST     := $(CCCL_SRC)/thrust
CCCL_CUB        := $(CCCL_SRC)/cub

NVCC = $(if $(CCACHE),$(CCACHE) ,)$(CUDA_HOME)/bin/nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80 $(if $(filter 1,$(LINEINFO)),-lineinfo) -Xcompiler -fopenmp \
  -I$(CCCL_LIBCUDACXX) \
  -I$(CCCL_CUB) \
  -I$(CCCL_THRUST) \
  -I$(CUCO_ROOT)/include \
  -I$(CUDA_HOME)/targets/x86_64-linux/include \
  --expt-extended-lambda --expt-relaxed-constexpr

# Common objects
COMMON_DIR = common
COMMON_OBJS = $(COMMON_DIR)/utils.o $(COMMON_DIR)/hash_table.o $(COMMON_DIR)/graph_data.o \
              $(COMMON_DIR)/cuckoo_hash.o $(COMMON_DIR)/hopscotch_hash.o $(COMMON_DIR)/roaring_bitmap.o \
              $(COMMON_DIR)/cuco.o

# Experiments
EXPERIMENTS = ir sss tc

ir_SRCS  := ir/main.cu ir/ir.cu ir/ir_cuco.cu
sss_SRCS := sss/main.cu sss/sss.cu sss/sss_cuco.cu \
            sss/sss_cuckoo.cu sss/sss_hopscotch.cu sss/sss_roaring.cu
tc_SRCS  := tc/main.cu tc/tc.cu tc/tc_cuco.cu

ir_OBJS  := $(ir_SRCS:.cu=.o)
sss_OBJS := $(sss_SRCS:.cu=.o)
tc_OBJS  := $(tc_SRCS:.cu=.o)

all: $(EXPERIMENTS)

%.o: %.cu
	@echo "Compiling $<"
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

ir: $(COMMON_OBJS) $(ir_OBJS)
	@echo "Linking ir/ir"
	$(NVCC) $(NVCC_FLAGS) $^ -o ir/ir
	@echo "✓ ir build successful"

sss: $(COMMON_OBJS) $(sss_OBJS)
	@echo "Linking sss/sss"
	$(NVCC) $(NVCC_FLAGS) $^ -o sss/sss
	@echo "✓ sss build successful"

tc: $(COMMON_OBJS) $(tc_OBJS)
	@echo "Linking tc/tc"
	$(NVCC) $(NVCC_FLAGS) $^ -o tc/tc
	@echo "✓ tc build successful"

test-ir: ir
	./ir/ir ../ir_datasets/$(IR_DATASET_NAME) --method=original,cuco

test-sss: sss
	./sss/sss ../graph_datasets/$(GRAPH_DATASET_NAME) --method=original,cuco

test-tc: tc
	./tc/tc ../graph_datasets/$(GRAPH_DATASET_NAME) --method=original,cuco

test-sss-all:
	./sss/sss ../graph_datasets/$(GRAPH_DATASET_NAME) --method=all
clean:
	rm -f $(COMMON_DIR)/*.o ir/*.o ir/ir sss/*.o sss/sss tc/*.o tc/tc

cuckoo: sss
	./sss/sss ../graph_datasets/$(GRAPH_DATASET_NAME) --method=cuckoo
hopscotch: sss
	./sss/sss ../graph_datasets/$(GRAPH_DATASET_NAME) --method=hopscotch
roaring: sss
	./sss/sss ../graph_datasets/$(GRAPH_DATASET_NAME) --method=roaring

.PHONY: all clean $(EXPERIMENTS) test-ir test-sss test-tc
