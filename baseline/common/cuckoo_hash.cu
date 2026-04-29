/**
 * @file cuckoo_hash.cu
 * @brief 布谷鸟哈希实现 - 完整修复版本
 *
 * 修复内容：
 * 1. 保持布谷鸟不变性：每个key始终在自己的三个候选位置之一
 * 2. 改进kick策略：随机选择下一个位置,降低循环概率
 * 3. 添加stash机制：无法插入的元素存入stash,查询时也检查
 * 4. 保证查询正确性：只要插入成功,查询一定能找到
 */

#include <thrust/device_vector.h>

#include "cuckoo_hash.cuh"

/**
 * Device 辅助函数：计算key的两个标准哈希位置
 *  ✔ 优化：从3个复杂哈希减少到2个简单哈希,统一复杂度
 * - 原: 3 * (3 SHIFT + 2 IMUL + 2 XOR) + node_salt计算 ≈ 20条指令
 * - 新: 2 * (1 XOR + 1 AND) = 4条指令
 * - 哈希开销: -80%
 */
__device__ __forceinline__ void compute_hashes(int node_id, int key,
                                               long long start, int capacity,
                                               long long& pos1, long long& pos2,
                                               long long& pos3) {
  int h1 = standard_hash(key, node_id, capacity);
  int h2 = standard_hash2(key, node_id, capacity);

  pos1 = start + h1;
  pos2 = start + h2;
  pos3 = pos2;  // 第三位置与第二位置相同(双哈希方案)
}

/**
 * @brief 批量插入内核 - 并行版本
 * 每个块处理NODES_PER_BLOCK个节点,每个线程处理一个节点
 * 每个节点插入由单个线程串行完成,完全避免竞态条件
 * 大幅提高GPU利用率,减少插入时间
 *
 * 布谷鸟不变性：每次踢出后,所有key仍然保持在它们自己的三个候选位置之一
 */
constexpr int NODES_PER_BLOCK = 128;
__global__ void flat_cuckoo_insert_kernel(
    int* table, long long* offsets, long long* stash_starts, int* stash_data,
    const int* node_start_map, const int* csr_offsets, const int* csr_cols,
    int max_iterations, int stash_size, int* failed_count, int num_nodes) {
  int block_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  // 每个线程处理一个节点,一个block处理NODES_PER_BLOCK个节点
  // 只有前NODES_PER_BLOCK个线程工作,剩余线程空闲
  if (thread_idx >= NODES_PER_BLOCK) return;

  int node_id = block_idx * NODES_PER_BLOCK + thread_idx;
  if (node_id >= num_nodes) return;  // 超出总节点数退出

  int degree = csr_offsets[node_id + 1] - csr_offsets[node_id];
  long long start = offsets[node_id];
  int capacity = static_cast<int>(offsets[node_id + 1] - start);

  //  ✔ 修复：stash按node_id * STASH_SIZE寻址,与查询保持一致
  long long my_stash_offset =
      static_cast<long long>(node_id) * FlatCuckooHash::STASH_SIZE;
  int* my_stash = stash_data + my_stash_offset;
  int my_stash_count = 0;

  // 初始化stash计数
  stash_starts[node_id] = 0;

  // 串行插入所有元素,完全避免并发冲突
  for (int global_idx = 0; global_idx < degree; global_idx++) {
    int key = csr_cols[csr_offsets[node_id] + global_idx];

    // 计算这个key的三个候选位置
    long long pos1, pos2, pos3;
    compute_hashes(node_id, key, start, capacity, pos1, pos2, pos3);

    //  ✔ 修复：首先检查key是否已经存在（处理重复边）
    // 如果key已经在哈希表中,直接跳过,避免不必要的踢出
    if (table[pos1] == key || table[pos2] == key || table[pos3] == key) {
      continue;
    }

    // 依次尝试三个位置
    if (table[pos1] == FlatCuckooHash::EMPTY_KEY) {
      table[pos1] = key;
      continue;
    }
    if (table[pos2] == FlatCuckooHash::EMPTY_KEY) {
      table[pos2] = key;
      continue;
    }
    if (table[pos3] == FlatCuckooHash::EMPTY_KEY) {
      table[pos3] = key;
      continue;
    }

    // 三个位置都满,开始布谷鸟踢出
    int current_key = key;
    // 使用随机数选择起始位置,打破规律减少循环
    // 简单随机化：使用key本身作为随机源
    int r = key % 3;
    long long current_pos = (r == 0) ? pos1 : (r == 1) ? pos2 : pos3;

    // 布谷鸟踢出循环
    bool inserted = false;
    for (int i = 0; i < max_iterations; i++) {
      // 取出旧key,放入当前key
      int old_key = table[current_pos];
      table[current_pos] = current_key;

      // 插入到空位了,完成
      if (old_key == FlatCuckooHash::EMPTY_KEY) {
        inserted = true;
        break;
      }

      // 对old_key重新计算它的三个候选位置
      // 这保证了不变性：old_key只会被放到它自己的三个候选位置之一
      long long old_p1, old_p2, old_p3;
      compute_hashes(node_id, old_key, start, capacity, old_p1, old_p2, old_p3);

      // 收集所有不是当前位置的候选点
      long long candidates[2];
      int cnt = 0;
      if (old_p1 != current_pos) candidates[cnt++] = old_p1;
      if (old_p2 != current_pos) candidates[cnt++] = old_p2;
      if (old_p3 != current_pos) candidates[cnt++] = old_p3;

      // 随机选择下一个位置(简单随机化：用old_key值选)
      if (cnt == 1) {
        current_key = old_key;
        current_pos = candidates[0];
      } else if (cnt == 2) {
        // 随机选择一个
        current_key = old_key;
        current_pos = (old_key & 1) ? candidates[0] : candidates[1];
      } else {
        // cnt == 3,三个都可用,随机选一个
        current_key = old_key;
        current_pos = (old_key % 3 == 0)   ? old_p1
                      : (old_key % 3 == 1) ? old_p2
                                           : old_p3;
      }
    }

    if (!inserted) {
      // 踢出失败,尝试放入stash
      //  ✔ 修复：放入的是current_key而不是key！
      // 经过踢出循环后,我们要放的是最后被踢出来的current_key
      // 而不是最初的key(key已经被放到某个位置了)
      if (my_stash_count < stash_size) {
        my_stash[my_stash_count++] = current_key;
        inserted = true;
      } else {
        // stash也满了,真的失败了
        atomicAdd(failed_count, 1);
      }
    }
  }

  // 保存stash计数
  stash_starts[node_id] = my_stash_count;
}

FlatCuckooHash::FlatCuckooHash(int num_nodes, const std::vector<int>& degrees,
                               float load_factor)
    : num_nodes_(num_nodes),
      total_capacity_(0),
      total_stash_capacity_(0),
      failed_count_(0) {
  // 计算每个节点需要的容量,容量必须是2的幂次(方便位运算取模)
  h_offsets_.resize(num_nodes + 1);
  h_offsets_[0] = 0;

  for (int i = 0; i < num_nodes; ++i) {
    int degree = degrees[i];
    // 计算需要的容量：degree / load_factor,向上对齐到2的幂次
    int capacity = 1;
    int required_capacity = static_cast<int>(ceil(degree / load_factor));
    while (capacity < required_capacity) {
      capacity <<= 1;
    }
    h_offsets_[i + 1] = h_offsets_[i] + capacity;
  }

  total_capacity_ = h_offsets_[num_nodes];
  // 每个节点预留STASH_SIZE个位置
  total_stash_capacity_ = num_nodes * STASH_SIZE;
  // 计算总blocks: 每个block处理NODES_PER_BLOCK个节点
  // int total_blocks = (num_nodes + NODES_PER_BLOCK - 1) / NODES_PER_BLOCK;
  // printf(
  //     "[Cuckoo] 节点数: %d, 总容量: %lld, stash总容量: %lld, "
  //     "总blocks: %d, nodes/block: %d, 平均容量/度数: %.2f\n",
  //     num_nodes, total_capacity_, total_stash_capacity_, total_blocks,
  //     NODES_PER_BLOCK, static_cast<float>(total_capacity_) / degrees.size());

  // 分配设备内存
  cudaMalloc(&d_table_, total_capacity_ * sizeof(int));
  cudaMalloc(&d_offsets_, (num_nodes + 1) * sizeof(long long));
  cudaMalloc(&d_stash_starts_, (num_nodes + 1) * sizeof(long long));
  cudaMalloc(&d_stash_data_, total_stash_capacity_ * sizeof(int));
  cudaMalloc(&d_failed_, sizeof(int));

  // stash_starts[num_nodes] 存储总stash大小,方便边界检查
  h_stash_starts_.resize(num_nodes + 1);
  for (int i = 0; i <= num_nodes; ++i) {
    h_stash_starts_[i] = static_cast<long long>(i) * STASH_SIZE;
  }

  // 初始化所有位置为空键
  // EMPTY_KEY = std::numeric_limits<int>::max() = 0x7fffffff
  int* h_temp = new int[total_capacity_];
  for (long long i = 0; i < total_capacity_; i++) {
    h_temp[i] = FlatCuckooHash::EMPTY_KEY;
  }
  cudaMemcpy(d_table_, h_temp, total_capacity_ * sizeof(int),
             cudaMemcpyHostToDevice);
  delete[] h_temp;

  //  ✔ 修复：初始化stash数据为EMPTY_KEY,避免随机垃圾值导致false positive
  int* h_stash_temp = new int[total_stash_capacity_];
  for (long long i = 0; i < total_stash_capacity_; i++) {
    h_stash_temp[i] = FlatCuckooHash::EMPTY_KEY;
  }
  cudaMemcpy(d_stash_data_, h_stash_temp, total_stash_capacity_ * sizeof(int),
             cudaMemcpyHostToDevice);
  delete[] h_stash_temp;

  //  ✔ 修复：stash_starts初始化为0(表示每个node的stash计数为0),而不是偏移
  // 之前错误地初始化为 i * STASH_SIZE,导致内核中读取到错误值
  std::vector<long long> h_stash_count(num_nodes + 1, 0);
  cudaMemcpy(d_stash_starts_, h_stash_count.data(),
             (num_nodes + 1) * sizeof(long long), cudaMemcpyHostToDevice);

  // 拷贝偏移数组到设备
  cudaMemcpy(d_offsets_, h_offsets_.data(), (num_nodes + 1) * sizeof(long long),
             cudaMemcpyHostToDevice);

  CHECK_CUDA_ERROR();
}

FlatCuckooHash::~FlatCuckooHash() {
  if (d_table_) cudaFree(d_table_);
  if (d_offsets_) cudaFree(d_offsets_);
  if (d_stash_starts_) cudaFree(d_stash_starts_);
  if (d_stash_data_) cudaFree(d_stash_data_);
  if (d_failed_) cudaFree(d_failed_);
}

int FlatCuckooHash::bulk_insert(const std::vector<int>& csr_offsets,
                                const std::vector<int>& csr_cols) {
  // 拷贝CSR数据到设备端
  thrust::device_vector<int> d_csr_offsets_dev = csr_offsets;
  thrust::device_vector<int> d_csr_cols_dev = csr_cols;

  int* d_csr_offsets_ptr = thrust::raw_pointer_cast(d_csr_offsets_dev.data());
  int* d_csr_cols_ptr = thrust::raw_pointer_cast(d_csr_cols_dev.data());

  // 重置失败计数
  int zero = 0;
  cudaMemcpy(d_failed_, &zero, sizeof(int), cudaMemcpyHostToDevice);

  // 每个block处理NODES_PER_BLOCK个节点,每个线程处理一个节点
  const int block_size = 128;  // block_size必须 >= NODES_PER_BLOCK
  int total_blocks = (num_nodes_ + NODES_PER_BLOCK - 1) / NODES_PER_BLOCK;

  dim3 grid_size(total_blocks);
  dim3 block_size_dim(block_size);

  flat_cuckoo_insert_kernel<<<grid_size, block_size_dim>>>(
      d_table_, d_offsets_, d_stash_starts_, d_stash_data_, nullptr,
      d_csr_offsets_ptr, d_csr_cols_ptr, MAX_ITERATIONS, STASH_SIZE, d_failed_,
      num_nodes_);

  cudaDeviceSynchronize();

  // 读取失败计数
  cudaMemcpy(&failed_count_, d_failed_, sizeof(int), cudaMemcpyDeviceToHost);

  CHECK_CUDA_ERROR();

  return failed_count_;
}
