/**
 * @file cuco.cu
 * @brief CucoHash实现 - 手动线性探测
 */

#include <iostream>

#include "cuco.cuh"
#include "utils.cuh"

// 定义CUDA_CHECK宏
#define CUDA_CHECK(expr)                                               \
  do {                                                                 \
    cudaError_t err = (expr);                                          \
    if (err != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

/**
 * @brief 初始化slot为EMPTY_KEY的内核
 */
__global__ void initialize_slots_kernel(int* slots, long long total_slots,
                                        int empty_key) {
  long long idx = blockIdx.x * 256LL + threadIdx.x;
  if (idx < total_slots) {
    slots[idx] = empty_key;
  }
}

/**
 * @brief 每个节点插入其邻居到自己哈希表的内核
 * 手动线性探测实现
 */
__global__ void node_insert_kernel(long long* offsets, int* slots,
                                   const int* csr_offsets, const int* csr_cols,
                                   int num_nodes, int empty_key) {
  int node_id = blockIdx.x;
  int tid = threadIdx.x;

  if (node_id >= num_nodes) return;

  long long table_start = offsets[node_id];
  int capacity = static_cast<int>(offsets[node_id + 1] - table_start);

  if (capacity == 0) return;

  int num_neighbors = csr_offsets[node_id + 1] - csr_offsets[node_id];
  const int* neighbors = csr_cols + csr_offsets[node_id];

  int mask = capacity - 1;  // capacity是2的幂

  // 每个线程负责插入一部分邻居
  for (int i = tid; i < num_neighbors; i += 64) {
    int key = neighbors[i];
    std::uint32_t hash = murmur3_32(key);
    int probe_count = 0;

    // 线性探测插入
    while (probe_count < capacity) {
      int pos = (hash + probe_count) & mask;
      int expected = empty_key;

      // CAS确保线程安全
      int* addr = slots + table_start + pos;
      int old = atomicCAS(addr, expected, key);

      if (old == empty_key || old == key) {
        break;
      }
      probe_count++;
    }
  }
}

CucoHash::CucoHash(int num_nodes, const std::vector<int>& degrees,
                   float load_factor)
    : num_nodes_(num_nodes), load_factor_(load_factor), total_capacity_(0) {
  // 计算每个节点的capacity和offset
  h_offsets_.resize(num_nodes_ + 1);
  h_offsets_[0] = 0;

  for (int i = 0; i < num_nodes_; ++i) {
    // 容量需是2的幂
    std::size_t required = static_cast<std::size_t>(degrees[i] / load_factor_);
    std::size_t capacity = next_power_of_two(required);
    if (capacity < 1) capacity = 1;
    total_capacity_ += capacity;
    h_offsets_[i + 1] = total_capacity_;
  }

  // 分配设备内存
  CUDA_CHECK(cudaMalloc(&d_slots_, total_capacity_ * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_offsets_, (num_nodes_ + 1) * sizeof(long long)));

  // 拷贝offset数组到设备
  CUDA_CHECK(cudaMemcpy(d_offsets_, h_offsets_.data(),
                        (num_nodes_ + 1) * sizeof(long long),
                        cudaMemcpyHostToDevice));

  // 初始化所有slot为EMPTY_KEY
  long long num_blocks = (total_capacity_ + 256LL - 1) / 256LL;
  initialize_slots_kernel<<<num_blocks, 256>>>(d_slots_, total_capacity_,
                                               EMPTY_KEY);
  CUDA_CHECK(cudaDeviceSynchronize());
}

CucoHash::~CucoHash() {
  if (d_slots_) cudaFree(d_slots_);
  if (d_offsets_) cudaFree(d_offsets_);
}

void CucoHash::bulk_insert(const std::vector<int>& csr_offsets,
                           const std::vector<int>& csr_cols,
                           cudaStream_t stream) {
  // 拷贝CSR到设备
  int* d_csr_offsets;
  int* d_csr_cols;

  CUDA_CHECK(cudaMalloc(&d_csr_offsets, csr_offsets.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_csr_cols, csr_cols.size() * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_csr_offsets, csr_offsets.data(),
                        csr_offsets.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_csr_cols, csr_cols.data(),
                        csr_cols.size() * sizeof(int), cudaMemcpyHostToDevice));

  // 每个节点用一个block处理插入
  node_insert_kernel<<<num_nodes_, 64, 0, stream>>>(
      d_offsets_, d_slots_, d_csr_offsets, d_csr_cols, num_nodes_, EMPTY_KEY);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // 释放临时内存
  CUDA_CHECK(cudaFree(d_csr_offsets));
  CUDA_CHECK(cudaFree(d_csr_cols));
}
