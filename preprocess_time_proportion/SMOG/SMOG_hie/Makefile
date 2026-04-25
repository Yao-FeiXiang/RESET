exe=subgraphmatch.bin
N=1
cucc= "$(shell which nvcc)"
# Thrust 头文件随 CUDA Toolkit 提供；mpicxx 编译 .cpp 时必须显式加入 include 路径。
CUDA_DIR := $(shell dirname $$(dirname $$(which nvcc)))
commflags=-I$(CUDA_DIR)/include -lcudart -L$(CUDA_DIR)/lib64 -O3 -W -Wall -Wno-unused-function -Wno-unused-parameter
# nvcc 对宿主 GCC 版本有上限；优先使用系统 g++-12，避免 Conda 里过新的 GCC 触发不支持错误。
NVCC_HOST_CXX ?= $(shell command -v g++-12 2>/dev/null || command -v g++ 2>/dev/null || echo g++)
# 与 CUDA 12 工具链一起使用时，不要用链接了 CUDA 11 libcudart 的 MPICH（会无法链接/运行）。
# 若已激活 Conda 且安装了 conda-forge 的 mpich，则优先用其 mpicxx；否则用 PATH 中的 mpicxx。
MPICXX ?=
ifeq ($(strip $(MPICXX)),)
  ifdef CONDA_PREFIX
    MPICXX := $(firstword $(wildcard $(CONDA_PREFIX)/bin/mpicxx))
  endif
endif
ifeq ($(strip $(MPICXX)),)
  MPICXX := $(shell command -v mpicxx 2>/dev/null)
endif
cc := "$(MPICXX)"
cuflags= --compiler-options -Wall -ccbin $(NVCC_HOST_CXX) --gpu-architecture=compute_80 -lineinfo --gpu-code=sm_80  -O3   # --resource-usage 
# cuflags= --compiler-options -v -Xcudafe  -\# --resource-usage 

.SILENT: cucc
.SILENT: cc
.SILENT: cuflags
.SILENT: %.o


objs	= $(patsubst %.cu,%.o,$(wildcard *.cu)) \
	$(patsubst %.cpp,%.o,$(wildcard *.cpp))

deps	= $(wildcard ./*.cuh) \
	$(wildcard ./*.hpp) \
	$(wildcard ./*.h) \


# %.o:%.cu $(deps)
# 	$(cucc) -c  $(cuflags)  $< -o $@ -g -G 

# %.o:%.cpp $(deps)
# 	$(cc) -c -g  $< -o $@ 

# $(exe):$(objs)
# 	$(cc) $(objs) $(commflags) -O3 -o $(exe)

%.o:%.cu $(deps)
	$(cucc) -c $(cuflags) $<  -o $@ 

%.o:%.cpp $(deps)
	$(cc) -c  $(commflags) $< -o $@ 

$(exe):$(objs)
	$(cc) $(objs) $(commflags) -o $(exe)

test:$(exe)
	./$(exe) cit-Patents_adj triangle 0.3 8 216 1024 1

clean:
	rm -rf *.o ${exe}
