GRAPH_DATASET_NAME = sc20 #sc18,sc19,sc20,bm,gp,wt
IR_DATASET_NAME = fever #cqadupstack,fever,hotpotqa,lt,msmarco,nq

CUDA_HOME ?= /home/cliu26/cuda-12.6 
# CUCO_ROOT := ../baseline/cuCollections
CUCO_ROOT := ./cuCollections
CUCO_BUILD := $(CUCO_ROOT)/build
CCCL_SRC  := $(CUCO_BUILD)/_deps/cccl-src
CCCL_LIBCUDACXX := $(CCCL_SRC)/libcudacxx/include
CCCL_THRUST     := $(CCCL_SRC)/thrust
CCCL_CUB        := $(CCCL_SRC)/cub

NVCC = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80 -lineinfo -Xcompiler -fopenmp \
  -I$(CUCO_ROOT)/include \
  -I$(CCCL_LIBCUDACXX) \
  -I$(CCCL_THRUST) \
  -I$(CCCL_CUB) \
  -I$(CUDA_HOME)/targets/x86_64-linux/include \
  --expt-extended-lambda --expt-relaxed-constexpr

# Common objects
COMMON_DIR = common
COMMON_OBJS = $(COMMON_DIR)/utils.o $(COMMON_DIR)/hash_table.o $(COMMON_DIR)/graph_data.o \
              $(COMMON_DIR)/cuckoo_hash.o $(COMMON_DIR)/hopscotch_hash.o $(COMMON_DIR)/roaring_bitmap.o

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
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

ir: $(COMMON_OBJS) $(ir_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o ir/ir

sss: $(COMMON_OBJS) $(sss_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o sss/sss

tc: $(COMMON_OBJS) $(tc_OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o tc/tc

test-ir: ir
	./ir/ir ../ir_datasets/$(IR_DATASET_NAME)

test-sss: sss
	./sss/sss ../graph_datasets/$(GRAPH_DATASET_NAME)

test-tc: tc
	./tc/tc ../graph_datasets/$(GRAPH_DATASET_NAME)

test-all:
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
