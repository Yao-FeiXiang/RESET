#include "utils.cuh"

int h_hash_normal(int x, int length) { return x % length; }

int h_hash_hierarchical(int x, int length, int max_length) {
  return x % max_length / (max_length / length);
}

void read_i32_vec(const std::string& path, std::vector<int>& vec) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + path);
  }
  size_t size;
  file.read((char*)&size, sizeof(size_t));
  vec.resize(size);
  file.read((char*)vec.data(), size * sizeof(int));
}

int* read_int_binary(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + path);
  }
  size_t size = file.tellg() / sizeof(int);
  int* h_ptr = new int[size];
  file.seekg(0, std::ios::beg);
  file.read((char*)h_ptr, size * sizeof(int));

  int* d_ptr;
  cudaMalloc(&d_ptr, size * sizeof(int));
  cudaMemcpy(d_ptr, h_ptr, size * sizeof(int), cudaMemcpyHostToDevice);
  delete[] h_ptr;
  return d_ptr;
}

void check_gpu_memory(int min_required_mib) {
  // 调用nvidia-smi获取内存信息
  FILE* pipe = popen(
      "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", "r");
  if (!pipe) {
    fprintf(stderr, "警告: 无法执行nvidia-smi检查GPU内存\n");
    return;
  }

  int current_device;
  cudaGetDevice(&current_device);

  char buffer[128];
  int free_memory_mib = 0;
  int device_idx = 0;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    if (device_idx == current_device) {
      sscanf(buffer, "%d", &free_memory_mib);
      break;
    }
    device_idx++;
  }
  pclose(pipe);

  if (free_memory_mib == 0) {
    fprintf(stderr, "警告: 无法获取GPU%d空闲内存信息\n", current_device);
    return;
  }

  if (free_memory_mib < min_required_mib) {
    fprintf(stderr, "错误: GPU%d空闲内存不足\n", current_device);
    fprintf(stderr, "需要至少: %d MiB,实际剩余: %d MiB\n", min_required_mib,
            free_memory_mib);
    exit(EXIT_FAILURE);
  }

  printf("GPU%d内存检查通过\n", current_device);
}

__global__ void extract_hashtable_to_csr_kernel(
    int num_nodes, int* csr_offsets, long long* hashtable_offsets,
    int* hashtable_data, long long total_buckets, int slots_per_bucket,
    int* output_csr_cols, int* work_index) {
  const int NODES_PER_WARP = 1;
  const int warp_idx = threadIdx.x / 32;
  const int lane_idx = threadIdx.x % 32;
  const int warps_per_block = blockDim.x / 32;

  // 动态分配工作：每个warp处理NODES_PER_WARP个节点
  int node_u = (blockIdx.x * warps_per_block + warp_idx) * NODES_PER_WARP;
  int node_end = node_u + NODES_PER_WARP;

  while (node_u < num_nodes) {
    const int csr_start = csr_offsets[node_u];
    const int degree = csr_offsets[node_u + 1] - csr_start;

    if (degree > 0) {
      // 获取当前节点哈希表的起始位置和长度
      const long long ht_start = hashtable_offsets[node_u];
      const int ht_length =
          static_cast<int>(hashtable_offsets[node_u + 1] - ht_start);
      int* node_hashtable = hashtable_data + static_cast<size_t>(ht_start);

      const int total_slots = ht_length * slots_per_bucket;
      int write_count = 0;
      const int iterations = (total_slots + 32 - 1) / 32;

      // Warp协作扫描哈希表，提取有效邻居
      for (int iter = 0; iter < iterations; iter++) {
        const int slot_idx = lane_idx + iter * warpSize;
        const bool is_active = (slot_idx < total_slots);
        int neighbor = -1;
        bool is_valid_entry = false;

        if (is_active) {
          // 分层哈希表布局: bucket + slot * total_buckets
          const int bucket = slot_idx / slots_per_bucket;
          const int slot = slot_idx % slots_per_bucket;
          neighbor = node_hashtable[bucket + slot * total_buckets];
          is_valid_entry = (neighbor != -1);  // -1表示空槽
        }

        // Warp投票：收集本迭代中有效元素的掩码
        const unsigned int valid_mask =
            __ballot_sync(0xffffffffu, is_valid_entry);
        const int valid_count = __popc(valid_mask);

        // 计算当前线程的写入位置（warp内前缀和）
        const int prefix_count = __popc(valid_mask & ((1u << lane_idx) - 1u));
        const int write_pos = csr_start + write_count + prefix_count;

        if (is_valid_entry && is_active) {
          output_csr_cols[write_pos] = neighbor;
        }
        write_count += valid_count;
      }
    }
    __syncwarp();

    // 获取下一批工作
    node_u++;
    if (node_u == node_end) {
      if (lane_idx == 0) {
        node_u = atomicAdd(work_index, NODES_PER_WARP);
      }
      node_u = __shfl_sync(0xffffffffu, node_u, 0);  // 广播给warp内所有线程
      node_end = node_u + NODES_PER_WARP;
    }
  }
}

void launch_extract_hashtable_to_csr(int num_nodes, int* d_csr_offsets,
                                     long long* d_hashtable_offsets,
                                     int* d_hashtable_data,
                                     long long total_buckets,
                                     int slots_per_bucket,
                                     int* d_output_csr_cols) {
  const int THREADS_PER_BLOCK = 128;
  const int warps_per_block = THREADS_PER_BLOCK / 32;
  const int blocks_needed = (num_nodes + warps_per_block - 1) / warps_per_block;

  // 动态工作分配索引
  int* d_work_index = nullptr;
  cudaMalloc(&d_work_index, sizeof(int));
  CHECK_CUDA_ERROR();

  // 初始偏移：跳过已有静态分配的工作
  const int init_work = blocks_needed * warps_per_block;
  cudaMemcpy(d_work_index, &init_work, sizeof(int), cudaMemcpyHostToDevice);
  CHECK_CUDA_ERROR();

  extract_hashtable_to_csr_kernel<<<blocks_needed, THREADS_PER_BLOCK>>>(
      num_nodes, d_csr_offsets, d_hashtable_offsets, d_hashtable_data,
      total_buckets, slots_per_bucket, d_output_csr_cols, d_work_index);
  CHECK_CUDA_ERROR();

  cudaDeviceSynchronize();
  cudaFree(d_work_index);
}
