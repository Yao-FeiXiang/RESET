NVCC = nvcc
TARGET = intersection

# 添加 GPU 架构等编译参数
cuflags = --compiler-options -Wall --gpu-architecture=compute_80 --gpu-code=sm_80 -O3

SRCS = intersection.cu
OBJS = $(SRCS:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) -o $@ $^

%.o: %.cu
	$(NVCC) -c $(cuflags) $< -o $@

clean:
	rm -f *.o *.exe *.lib *.exp $(TARGET)