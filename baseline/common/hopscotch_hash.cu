/**
 * @file hopscotch_hash.cu
 * @brief 扁平化跳房子哈希实现
 *
 * 实现特点：
 * - 每个节点一个block,block内第一个线程串行插入
 * - 避免并发冲突,保证正确性
 * - 使用标准跳房子算法：移动空闲位置到邻域
 */

#include <thrust/device_vector.h>

#include "hopscotch_hash.cuh"
#include "utils.cuh"

/**
 * 设备端哈希函数：将key映射到bucket索引
 * ✅ 使用统一的standard_hash哈希函数，确保插入和查询完全一致
 */
__device__ __forceinline__ int compute_hash(int key, int node_id,
                                            int capacity) {
  return standard_hash(key, node_id, capacity);
}

/**
 * 内核：批量插入所有节点的邻接点
 * 每个block处理一个节点,第一个线程串行插入所有元素
 */
__global__ void hopscotch_bulk_insert_kernel(int* offset, int* table,
                                             int* bitmap,
                                             const int* csr_offsets,
                                             const int* csr_cols,
                                             int* d_failed_count) {
  __shared__ int block_failed;
  if (threadIdx.x == 0) {
    block_failed = 0;
  }
  __syncthreads();

  int node_id = blockIdx.x;
  int start = csr_offsets[node_id];
  int end = csr_offsets[node_id + 1];
  int node_start = offset[node_id];
  // ✅ 修正：总容量减去邻域大小H才是实际哈希表容量
  int total_with_h = offset[node_id + 1] - node_start;
  int capacity = total_with_h - HopscotchHash::H;

  // 只有第一个线程插入,避免竞态
  if (threadIdx.x != 0) return;

  // 初始化该节点的哈希表（包括邻域H空间）
  for (int i = 0; i < total_with_h; i++) {
    table[node_start + i] = HopscotchHash::EMPTY_KEY;
  }
  for (int i = 0; i < total_with_h; i++) {
    bitmap[node_start + i] = 0;
  }

  // 顺序插入每个邻接点
  for (int i = start; i < end; i++) {
    int key = csr_cols[i];
    int home = compute_hash(key, node_id, capacity);

    // 尝试在邻域H范围内找到空位
    bool inserted = false;
    for (int j = 0; j < HopscotchHash::H; j++) {
      int pos = home + j;
      if (pos < total_with_h &&
          table[node_start + pos] == HopscotchHash::EMPTY_KEY) {
        table[node_start + pos] = key;
        bitmap[node_start + home] |= (1U << j);
        inserted = true;
        break;
      }
    }

    if (inserted) continue;

    // 邻域已满,需要向后找空位并移动过来
    // 只能找home+H之后的空位,因为交换算法只能把空位向左移动
    // 如果空位在home左边,无法通过向左交换移动到home邻域范围
    int empty_pos = home + HopscotchHash::H;
    while (empty_pos < total_with_h &&
           table[node_start + empty_pos] != HopscotchHash::EMPTY_KEY) {
      empty_pos++;
    }

    // 找不到空位,插入失败
    if (empty_pos >= total_with_h) {
      // 计数在全局失败统计中
      if (threadIdx.x == 0) {
        atomicAdd(d_failed_count, 1);
      }
      continue;
    }

    // 不断向上移动空位直到进入home的邻域[home, home+H)
    while (empty_pos >= home + HopscotchHash::H) {
      int current = empty_pos;
      bool found = false;
      int found_j = -1;
      int found_current_home = -1;

      // current位置能被所有 home_bucket ∈ [current-H+1, current] 包含
      int start_home = max(0, current - HopscotchHash::H + 1);
      // 从后往前找,找到最后一个可交换元素,给空位更多移动空间
      for (int current_home = current; current_home >= start_home;
           current_home--) {
        uint32_t bm = bitmap[node_start + current_home];
        // 倒序遍历找最后一个可交换位置
        for (int j = HopscotchHash::H - 1; j >= 0; j--) {
          if ((bm >> j) & 1) {
            int swap_pos = current_home + j;
            // 只交换位置 < current 的,让空位不断往回走(靠近home)
            if (swap_pos < current) {
              // 找到可交换元素,记录下来
              found_j = j;
              found_current_home = current_home;
              found = true;
              break;
            }
          }
        }
        if (found) break;
      }

      if (found) {
        // 执行交换
        int current_home = found_current_home;
        int j = found_j;
        uint32_t bm = bitmap[node_start + current_home];
        int swap_pos = current_home + j;
        int swap_key = table[node_start + swap_pos];
        table[node_start + current] = swap_key;
        int new_j = current - current_home;
        bitmap[node_start + current_home] = (bm & ~(1U << j)) | (1U << new_j);
        empty_pos = swap_pos;
      }

      // 找不到任何可以交换的,无法继续移动,插入失败
      if (!found) break;
    }

    // 如果空位成功移动到了home的邻域范围内 (home <= empty_pos < home+H),插入key
    if (empty_pos >= home && empty_pos < home + HopscotchHash::H) {
      int j = empty_pos - home;
      table[node_start + empty_pos] = key;
      bitmap[node_start + home] |= (1U << j);
    } else {
      // 无法移入邻域,插入失败
      if (threadIdx.x == 0) {
        atomicAdd(d_failed_count, 1);
      }
    }
  }

  __syncthreads();
  // 汇总到全局
  if (threadIdx.x == 0 && block_failed > 0) {
    atomicAdd(d_failed_count, block_failed);
  }
}

/**
 * 构造函数：预分配所有内存
 */
__host__ HopscotchHash::HopscotchHash(int num_nodes,
                                      const std::vector<int>& degrees,
                                      float load_factor)
    : num_nodes_(num_nodes) {
  // 计算每个节点的偏移和总容量
  float actual_load_factor = min(load_factor, 0.1f);
  std::vector<int> offset_host(num_nodes + 1);
  offset_host[0] = 0;
  for (int i = 0; i < num_nodes; i++) {
    double required = static_cast<double>(degrees[i]) / actual_load_factor;
    int capacity = 1;
    while (capacity < required) {
      capacity <<= 1;
    }
    // ✅ 确保capacity至少为H，避免邻域越界
    if (capacity < HopscotchHash::H) {
      capacity = HopscotchHash::H;
    }
    offset_host[i + 1] = offset_host[i] + capacity + HopscotchHash::H;
  }
  total_capacity_ = offset_host[num_nodes];

  // 分配设备内存
  cudaMalloc(&d_offset_, (num_nodes + 1) * sizeof(int));
  cudaMalloc(&d_table_, total_capacity_ * sizeof(int));
  cudaMalloc(&d_bitmap_, total_capacity_ * sizeof(int));

  // 复制偏移到设备
  cudaMemcpy(d_offset_, offset_host.data(), (num_nodes + 1) * sizeof(int),
             cudaMemcpyHostToDevice);

  CHECK_CUDA_ERROR();
}

/**
 * 析构函数：释放所有内存
 */
__host__ HopscotchHash::~HopscotchHash() {
  if (d_offset_) cudaFree(d_offset_);
  if (d_table_) cudaFree(d_table_);
  if (d_bitmap_) cudaFree(d_bitmap_);
}

/**
 * 批量插入所有节点的邻接点
 */
__host__ int HopscotchHash::bulk_insert(const std::vector<int>& csr_offsets,
                                        const std::vector<int>& csr_cols) {
  thrust::device_vector<int> csr_offsets_d(csr_offsets);
  thrust::device_vector<int> csr_cols_d(csr_cols);
  thrust::device_vector<int> failed_count_d(1, 0);

  // 一个block处理一个节点
  int grid_size = num_nodes_;

  hopscotch_bulk_insert_kernel<<<grid_size, 1>>>(
      d_offset_, d_table_, d_bitmap_,
      thrust::raw_pointer_cast(csr_offsets_d.data()),
      thrust::raw_pointer_cast(csr_cols_d.data()),
      thrust::raw_pointer_cast(failed_count_d.data()));

  CHECK_CUDA_ERROR();

  // 读取失败计数
  std::vector<int> failed_count_host(1);
  thrust::copy(failed_count_d.begin(), failed_count_d.end(),
               failed_count_host.begin());

  return failed_count_host[0];
}
