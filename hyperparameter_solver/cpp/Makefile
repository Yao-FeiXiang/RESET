# 超参数求解器 C++ 接口 Makefile

CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2

# Python 配置 - 根据实际环境调整
# 如果需要直接嵌入 Python 解释器，取消注释以下行
# PYTHON_CFLAGS = $(shell python3-config --cflags)
# PYTHON_LDFLAGS = $(shell python3-config --ldflags) -lpython3.11

# 目标文件
TARGETS = example_usage

# 默认目标
all: $(TARGETS)

# 示例程序（通过命令行调用 Python，不需要链接 Python 库）
example_usage: example_usage.cpp hyper_solver.cpp hyper_solver.h
	$(CXX) $(CXXFLAGS) example_usage.cpp hyper_solver.cpp -o $@

# 清理
clean:
	rm -f $(TARGETS) *.o

# 运行示例
run: example_usage
	./example_usage

.PHONY: all clean run
