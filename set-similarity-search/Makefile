# 编译器设置
NVCC = nvcc
CXXFLAGS = -O3 -Wall
NVCCFLAGS = --compiler-options -Wall --gpu-architecture=compute_80 --gpu-code=sm_80 -O3

# 目标可执行文件
TARGET = sss

# 源文件
SRCS = hashtable.cu sss.cu
OBJS = $(SRCS:.cu=.o)

# 头文件
DEPS = sss.cuh

# 默认目标
all: $(TARGET)

# 链接目标文件生成可执行文件
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# 编译CUDA源文件
%.o: %.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 清理生成的文件
clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean