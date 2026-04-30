#ifndef UTILS_CUH
#define UTILS_CUH

#include <assert.h>
#include <cuda_runtime.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define DEV 1
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

__device__ __forceinline__ void warmup_kernel_d() {}

// 深度预热内核：模拟哈希表构建规模的GPU操作
static __global__ void deep_warmup_kernel(int* temp_data, long long size) {
  long long idx = blockIdx.x * 256LL + threadIdx.x;
  if (idx < size) {
    // 模拟哈希表初始化和插入操作
    int val = temp_data[idx];
    val = val * 3 + 7;       // 一些计算
    val = (val << 5) ^ val;  // 模拟哈希计算
    temp_data[idx] = val;
  }
}

// L2 缓存刷新内核：通过读取大量数据驱逐现有缓存内容
static __global__ void flush_l2_cache_kernel(int* __restrict__ flush_buffer,
                                             long long size) {
  long long idx = blockIdx.x * 256LL + threadIdx.x;
  if (idx < size) {
    volatile int val = const_cast<const volatile int*>(flush_buffer)[idx];
    (void)val;
  }
}

inline void flush_l2_cache(long long flush_size = 100LL * 1024LL *
                                                  1024LL) {  // 100MB 刷新缓冲区
  int* d_flush = nullptr;
  cudaMalloc(&d_flush, flush_size * sizeof(int));
  cudaMemset(d_flush, 0xCC, flush_size * sizeof(int));  // 写入一些数据
  cudaDeviceSynchronize();

  // 执行缓存刷新内核
  long long num_blocks = (flush_size + 256LL - 1) / 256LL;
  flush_l2_cache_kernel<<<num_blocks, 256>>>(d_flush, flush_size);
  cudaDeviceSynchronize();

  cudaFree(d_flush);
  cudaDeviceSynchronize();
}

inline void warmup_gpu(long long warmup_size = 50LL * 1024LL *
                                               1024LL) {  // 默认50M 元素预热
  // 分配临时内存进行深度预热 - 模拟哈希表构建的GPU活动
  int* d_temp = nullptr;
  cudaMalloc(&d_temp, warmup_size * sizeof(int));

  // 执行多轮预热内核 - 模拟 cuco 的 initialize_slots_kernel +
  // node_insert_kernel
  long long num_blocks = (warmup_size + 256LL - 1) / 256LL;

  // 第一轮：模拟初始化
  deep_warmup_kernel<<<num_blocks, 256>>>(d_temp, warmup_size);
  cudaDeviceSynchronize();

  // 第二轮：模拟插入
  deep_warmup_kernel<<<num_blocks, 256>>>(d_temp, warmup_size);
  cudaDeviceSynchronize();

  // 第三轮：额外的内存访问预热
  deep_warmup_kernel<<<num_blocks, 256>>>(d_temp, warmup_size);
  cudaDeviceSynchronize();

  cudaFree(d_temp);
  cudaDeviceSynchronize();
}

// 设备端哈希函数
__device__ __forceinline__ int d_hash_normal(int x, int length) {
  return x & (length - 1);
}

__device__ __forceinline__ int d_hash_hierarchical(const int x, int length,
                                                   int max_length) {
  int shift = __ffs(max_length) - __ffs(length);
  return (x & (max_length - 1)) >> shift;
}

/**
 * @brief 标准哈希函数
 */
__device__ __forceinline__ int standard_hash(int key, int node_id,
                                             int capacity) {
  return (key ^ (key >> 8) ^ node_id) & (capacity - 1);
}

/**
 * @brief 标准哈希函数2
 */
__device__ __forceinline__ int standard_hash2(int key, int node_id,
                                              int capacity) {
  return (key ^ (key >> 16) ^ (node_id << 1) ^ 0xAAAAAAAA) & (capacity - 1);
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
void check_gpu_memory(int min_required_mib = 1000);

// 解析逗号分隔的方法列表
inline std::vector<std::string> parse_methods(const std::string& arg) {
  std::vector<std::string> methods;
  std::stringstream ss(arg);
  std::string method;
  while (std::getline(ss, method, ',')) {
    if (!method.empty()) {
      methods.push_back(method);
    }
  }
  return methods;
}

/**
 * @brief 编码(term, doc)对为64位键 (用于IR)
 * 将32位term和32位doc编码为一个64位整数存储
 */
__host__ __device__ __forceinline__ std::uint64_t encode_posting_key(int term,
                                                                     int doc) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(term)) << 32) |
         static_cast<std::uint32_t>(doc);
}

/**
 * @brief 编码(u, v)边对为64位键 (用于SSS/TC)
 * 确保u < v避免重复存储
 */
__host__ __device__ __forceinline__ std::uint64_t encode_edge_key(int u,
                                                                  int v) {
  if (u > v) {
    int tmp = u;
    u = v;
    v = tmp;
  }
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(u)) << 32) |
         static_cast<std::uint32_t>(v);
}

/**
 * @brief CUDA内核：从分层哈希表中提取有效元素，按哈希顺序重建CSR列
 *
 * 利用哈希表构建时的分层布局，直接提取每个节点的有效邻居（跳过-1空槽），
 * 按哈希桶顺序排列，实现"预排序"效果。每个Warp处理一个节点，通过
 * warp原语高效协作，无需真正排序即可获得局部性优化。
 */
__global__ void extract_hashtable_to_csr_kernel(
    int num_nodes, int* csr_offsets, long long* hashtable_offsets,
    int* hashtable_data, long long total_buckets, int slots_per_bucket,
    int* output_csr_cols, int* work_index);

/**
 * @brief 主机端包装：启动内核，从分层哈希表提取CSR列
 *
 * 利用哈希表构建时的已有布局，提取有效元素并按哈希顺序排列，
 * 这比通用thrust排序更快且无需数据传输。
 */
void launch_extract_hashtable_to_csr(int num_nodes, int* d_csr_offsets,
                                     long long* d_hashtable_offsets,
                                     int* d_hashtable_data,
                                     long long total_buckets,
                                     int slots_per_bucket,
                                     int* d_output_csr_cols);

#endif  // UTILS_CUH
