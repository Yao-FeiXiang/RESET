/**
 * @file roaring_bitmap.cu
 * @brief 动态压缩Roaring位图实现 - 只分配实际使用的容器，避免内存爆炸
 *
 * 实现流程：
 * 1. 主机端扫描每个节点，收集该节点实际使用的所有high-bit容器
 * 2. 计算偏移，紧凑分配，只分配需要的空间
 * 3. GPU端并行插入，每个block处理一个节点
 * 4. 查询使用二分查找定位容器，O(log C) 查找，C是每个节点容器数
 */

#include <thrust/device_vector.h>

#include <algorithm>

#include "roaring_bitmap.cuh"
#include "utils.cuh"

/**
 * 内核：批量插入所有节点的邻接点到动态压缩Roaring
 * 每个block处理一个节点，每个线程处理一个邻接点
 */
__global__ void roaring_bulk_insert_kernel(int* node_offsets,
                                           int* container_index,
                                           unsigned long long* bitmap,
                                           const int* csr_offsets,
                                           const int* csr_cols) {
  const int node_id = blockIdx.x;
  const int start = csr_offsets[node_id];
  const int end = csr_offsets[node_id + 1];
  const int node_container_start = node_offsets[node_id];
  const int node_container_end = node_offsets[node_id + 1];
  const int num_containers_node = node_container_end - node_container_start;

  // 只有有数据的节点才处理（避免死锁：空block不会执行__syncthreads）
  if (num_containers_node > 0) {
    // 该节点的所有容器位图初始化为0
    for (int i = threadIdx.x;
         i < num_containers_node * RoaringBitmap::WORDS_PER_CONTAINER;
         i += blockDim.x) {
      // 计算全局偏移
      // node_container_start:
      // 本节点第一个container在container_index数组中的索引 i:
      // 从本节点第一个word开始的偏移
      // 使用size_t避免32位溢出！当容器总数>2M时，container_idx*1024会超过2^31
      size_t container_idx = (size_t)node_container_start +
                             (size_t)(i / RoaringBitmap::WORDS_PER_CONTAINER);
      int word_in_container = i % RoaringBitmap::WORDS_PER_CONTAINER;
      size_t word_offset =
          container_idx * (size_t)RoaringBitmap::WORDS_PER_CONTAINER +
          (size_t)word_in_container;
      bitmap[word_offset] = 0;
    }
    __syncthreads();

    // 并行插入每个邻接点
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
      const int key = csr_cols[i];
      const int container = key >> 16;
      const int bit_pos = key & 0xFFFF;
      const int word_idx = bit_pos >> RoaringBitmap::LOG_BITS_PER_WORD;
      const uint64_t bit_mask =
          1ULL << (bit_pos & RoaringBitmap::MASK_BITS_PER_WORD);

      // 在该节点的容器列表中二分查找找到容器索引
      int left = 0, right = num_containers_node - 1;
      int found_container_pos = -1;
      while (left <= right) {
        const int mid = (left + right) / 2;
        const int mid_container = container_index[node_container_start + mid];
        if (mid_container == container) {
          found_container_pos = mid;
          break;
        } else if (mid_container < container) {
          left = mid + 1;
        } else {
          right = mid - 1;
        }
      }

      // 容器一定存在（预分配）
      if (found_container_pos >= 0) {
        // 计算最终bitmap位置
        // 使用size_t避免32位溢出
        const size_t global_container_idx =
            (size_t)node_container_start + (size_t)found_container_pos;
        const size_t bitmap_offset =
            global_container_idx * (size_t)RoaringBitmap::WORDS_PER_CONTAINER;
        atomicOr(reinterpret_cast<unsigned long long*>(
                     &bitmap[bitmap_offset + word_idx]),
                 static_cast<unsigned long long>(bit_mask));
      }
    }
  }
}  // if (num_containers_node > 0)

/**
 * 构造函数：主机端收集每个节点实际使用的容器，计算所需内存分配
 */
__host__ RoaringBitmap::RoaringBitmap(int num_nodes,
                                      const std::vector<int>& csr_offsets,
                                      const std::vector<int>& csr_cols)
    : num_nodes_(num_nodes), csr_offsets_(csr_offsets), csr_cols_(csr_cols) {
  // 第一步：扫描每个节点，收集所有唯一container编号
  host_node_offsets_.resize(num_nodes + 1);
  host_node_offsets_[0] = 0;

  for (int node_id = 0; node_id < num_nodes; node_id++) {
    const int start = csr_offsets[node_id];
    const int end = csr_offsets[node_id + 1];

    // 收集该节点所有邻接点的container编号
    std::vector<int> containers;
    containers.reserve((end - start + 65535) / 65536);

    for (int i = start; i < end; i++) {
      const int key = csr_cols[i];
      const int container = key >> 16;
      containers.push_back(container);
    }

    // 去重排序
    std::sort(containers.begin(), containers.end());
    containers.erase(std::unique(containers.begin(), containers.end()),
                     containers.end());

    // 添加到全局索引
    host_container_index_.insert(host_container_index_.end(),
                                 containers.begin(), containers.end());
    host_node_offsets_[node_id + 1] = host_container_index_.size();
  }

  total_containers_ = host_container_index_.size();
  total_words_ = (size_t)total_containers_ * (size_t)WORDS_PER_CONTAINER;
  // 调试输出
  // printf(
  //     "[Roaring] 节点数: %d, 总容器数: %d, 总bitmap字数: %zu, 总内存: %.2f "
  //     "MB\n",
  //     num_nodes, total_containers_, total_words_,
  //     (double)(total_words_ * sizeof(unsigned long long) +
  //              total_containers_ * sizeof(int) +
  //              (num_nodes + 1) * sizeof(int)) /
  //         (1024.0 * 1024.0));

  // 分配设备内存
  cudaError_t err;
  err = cudaMalloc(&d_node_offsets_, (num_nodes + 1) * sizeof(int));
  if (err != cudaSuccess)
    printf("Error alloc d_node_offsets_: %s\n", cudaGetErrorString(err));
  err = cudaMalloc(&d_container_index_, total_containers_ * sizeof(int));
  if (err != cudaSuccess)
    printf("Error alloc d_container_index_: %s\n", cudaGetErrorString(err));
  err = cudaMalloc(&d_bitmap_, total_words_ * sizeof(unsigned long long));
  if (err != cudaSuccess)
    printf("Error alloc d_bitmap_: %s\n", cudaGetErrorString(err));

  // 复制索引数据到设备
  cudaMemcpy(d_node_offsets_, host_node_offsets_.data(),
             (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_container_index_, host_container_index_.data(),
             total_containers_ * sizeof(int), cudaMemcpyHostToDevice);

  CHECK_CUDA_ERROR();
}

/**
 * 析构函数：释放所有内存
 */
__host__ RoaringBitmap::~RoaringBitmap() {
  if (d_node_offsets_) cudaFree(d_node_offsets_);
  if (d_container_index_) cudaFree(d_container_index_);
  if (d_bitmap_) cudaFree(d_bitmap_);
}

/**
 * 批量插入：GPU并行插入所有邻接点
 */
__host__ int RoaringBitmap::bulk_insert() {
  thrust::device_vector<int> csr_offsets_d(csr_offsets_);
  thrust::device_vector<int> csr_cols_d(csr_cols_);

  // 复制csr数据到设备
  csr_offsets_d = csr_offsets_;
  csr_cols_d = csr_cols_;

  // 一个block处理一个节点，256线程并行插入
  const int grid_size = num_nodes_;
  const int block_size = 256;

  roaring_bulk_insert_kernel<<<grid_size, block_size>>>(
      d_node_offsets_, d_container_index_, d_bitmap_,
      thrust::raw_pointer_cast(csr_offsets_d.data()),
      thrust::raw_pointer_cast(csr_cols_d.data()));

  CHECK_CUDA_ERROR();
  return 0;

  // Roaring位图永远不会失败
  return 0;
}
