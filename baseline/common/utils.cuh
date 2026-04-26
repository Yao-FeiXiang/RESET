#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>

#include <fstream>
#include <string>
#include <vector>

// CUDA错误检查宏
#define CHECK_CUDA_ERROR()                                             \
  do {                                                                 \
    cudaError_t err = cudaGetLastError();                              \
    if (err != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

// L2缓存刷新工具类,用于性能测试前清空缓存
struct l2flush {
  l2flush() {
    cudaDeviceGetAttribute(&l2_size_, cudaDevAttrL2CacheSize, 0);
    data_ = nullptr;
    cudaMalloc(&data_, l2_size_);
  }
  ~l2flush() { cudaFree(data_); }
  void flush() {
    unsigned char c = 0;
    for (int i = 0; i < l2_size_; i += 128) {
      c += data_[i];
    }
    cudaMemcpy(data_, data_, l2_size_, cudaMemcpyDeviceToDevice);
  }
  unsigned char* data_;
  int l2_size_;
};

// 设备端哈希函数
__device__ __forceinline__ int d_hash_normal(int x, int length) {
  return x & (length - 1);
}

__device__ __forceinline__ int d_hash_hierarchical(const int x, int length,
                                                   int max_length) {
  int shift = __ffs(max_length) - __ffs(length);
  return (x & (max_length - 1)) >> shift;
}

// 主机端哈希函数
__host__ int h_hash_normal(int x, int length);
__host__ int h_hash_hierarchical(int x, int length, int max_length);

// 设备端计算哈希表桶长度(对齐到2的幂)
__device__ __forceinline__ int calculate_length(int degrees, float load_factor,
                                                int bucket_size) {
  int total_size = degrees / load_factor;
  int length = total_size / bucket_size;
  if (length == 0) return 8;
  length |= length >> 1;
  length |= length >> 2;
  length |= length >> 4;
  length |= length >> 8;
  length |= length >> 16;
  length++;
  if (length < 8) length = 8;
  return length;
}

// 从二进制文件读取数据到主机端vector<int>
void read_i32_vec(const std::string& path, std::vector<int>& vec);

// 从二进制文件读取数据到设备端数组
int* read_int_binary(const std::string& path);

// 检查当前GPU剩余内存是否至少需要min_required_mib MiB
// 通过调用nvidia-smi获取剩余显存信息
// 如果不足,打印错误信息并退出程序
void check_gpu_memory(int min_required_mib = 20000);

#endif  // UTILS_CUH
