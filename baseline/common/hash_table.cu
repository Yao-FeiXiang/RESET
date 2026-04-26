#include "hash_table.cuh"

__global__ void calculate_hash_length_kernel(int* hash_length, int num_nodes,
                                             int* offsets, int* max_length,
                                             float load_factor,
                                             int bucket_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < num_nodes; i += stride) {
    int start = offsets[i];
    int end = offsets[i + 1];
    int degrees = end - start;
    hash_length[i] = calculate_length(degrees, load_factor, bucket_size);
    atomicMax(max_length, hash_length[i]);
  }
}

__global__ void build_hierarchical_kernel(int* hash_table, int* hash_length,
                                          long long* hash_tables_offset,
                                          int element_count, int bucket_num,
                                          int max_length, int* node_ids,
                                          int* elements, int* conflict_count,
                                          int bucket_size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < element_count; i += stride) {
    int u = node_ids[i];
    int v = elements[i];
    int length = hash_length[u];
    int* hash_start = hash_table + hash_tables_offset[u];
    int bucket = d_hash_hierarchical(v, length, max_length);
    int k = 0;
    while (atomicCAS(&hash_start[bucket + k * bucket_num], -1, v) != -1) {
      k++;
      atomicAdd(conflict_count, 1);
      if (k == bucket_size) {
        k = 0;
        bucket = (bucket + 1) % length;
      }
    }
  }
}

__global__ void build_normal_kernel(int* hash_table, int* hash_length,
                                    long long* hash_tables_offset,
                                    int element_count, int bucket_num,
                                    int* node_ids, int* elements,
                                    int* conflict_count, int bucket_size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < element_count; i += stride) {
    int u = node_ids[i];
    int v = elements[i];
    int length = hash_length[u];
    int* hash_start = hash_table + hash_tables_offset[u];
    int bucket = d_hash_normal(v, length);
    int k = 0;
    while (atomicCAS(&hash_start[bucket + k * bucket_num], -1, v) != -1) {
      k++;
      atomicAdd(conflict_count, 1);
      if (k == bucket_size) {
        k = 0;
        bucket = (bucket + 1) % length;
      }
    }
  }
}

void HashTableBuilder::build_hash_tables(
    int num_nodes, const std::vector<int>& h_offsets,
    const std::vector<int>& h_elements, float load_factor, int bucket_size,
    const std::vector<int>& vertexs, const std::vector<int>& vertex_csr_cols) {
  // Allocate and copy offsets to device
  cudaMalloc(&d_hash_length_, sizeof(int) * num_nodes);
  int* d_offsets;
  cudaMalloc(&d_offsets, sizeof(int) * (num_nodes + 1));
  cudaMemcpy(d_offsets, h_offsets.data(), sizeof(int) * (num_nodes + 1),
             cudaMemcpyHostToDevice);
  int* d_max_length;
  cudaMalloc(&d_max_length, sizeof(int));

  // Calculate hash lengths
  calculate_hash_length_kernel<<<256, 1024>>>(d_hash_length_, num_nodes,
                                              d_offsets, d_max_length,
                                              load_factor, bucket_size);
  cudaDeviceSynchronize();
  cudaMemcpy(&max_length_, d_max_length, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_max_length);
  printf("Max length: %d\n", max_length_);
  printf("average degree: %f\n", (float)h_elements.size() / num_nodes);

  // Compute exclusive scan for table offsets
  cudaMalloc(&d_hash_tables_offset_, sizeof(long long) * (num_nodes + 1));
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                d_hash_length_, d_hash_tables_offset_,
                                num_nodes);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                d_hash_length_, d_hash_tables_offset_,
                                num_nodes);

  // Get total bucket count
  int last_hash_length;
  cudaMemcpy(&last_hash_length, d_hash_length_ + num_nodes - 1, sizeof(int),
             cudaMemcpyDeviceToHost);
  long long last_sum;
  cudaMemcpy(&last_sum, d_hash_tables_offset_ + num_nodes - 1,
             sizeof(long long), cudaMemcpyDeviceToHost);
  printf("last sum: %lld, last_hash_length: %d\n", last_sum, last_hash_length);
  bucket_num_ = last_sum + last_hash_length;
  printf("Total buckets: %lld\n", bucket_num_);
  cudaMemcpy(d_hash_tables_offset_ + num_nodes, &bucket_num_, sizeof(long long),
             cudaMemcpyHostToDevice);
  cudaFree(d_temp_storage);

  // Allocate hash table memory
  cudaMalloc(&d_hash_table_hierarchical_,
             sizeof(int) * bucket_num_ * bucket_size);
  cudaMalloc(&d_hash_table_normal_, sizeof(int) * bucket_num_ * bucket_size);
  cudaMemset(d_hash_table_hierarchical_, -1,
             sizeof(int) * bucket_num_ * bucket_size);
  cudaMemset(d_hash_table_normal_, -1, sizeof(int) * bucket_num_ * bucket_size);

  // 如果提供了顶点对数据,使用顶点对构建哈希表(与set-similarity-search一致)
  // 否则使用完整邻居列表构建
  std::vector<int> node_ids;
  std::vector<int> elements;

  if (!vertexs.empty() && !vertex_csr_cols.empty()) {
    // 使用顶点对数据构建哈希表
    node_ids = vertexs;
    elements = vertex_csr_cols;
    printf("使用顶点对数据构建哈希表, 数据量: %zu\n", node_ids.size());
  } else {
    // 使用完整邻居列表构建哈希表(原有行为)
    node_ids.reserve(h_elements.size());
    elements.reserve(h_elements.size());
    for (int i = 0; i < num_nodes; i++) {
      int start = h_offsets[i];
      int end = h_offsets[i + 1];
      for (int j = start; j < end; j++) {
        node_ids.push_back(i);
        elements.push_back(h_elements[j]);
      }
    }
    printf("使用完整邻居列表构建哈希表, 数据量: %zu\n", node_ids.size());
  }

  // Copy to device
  int *d_node_ids, *d_elements;
  cudaMalloc(&d_node_ids, sizeof(int) * node_ids.size());
  cudaMalloc(&d_elements, sizeof(int) * elements.size());
  cudaMemcpy(d_node_ids, node_ids.data(), sizeof(int) * node_ids.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_elements, elements.data(), sizeof(int) * elements.size(),
             cudaMemcpyHostToDevice);

  int* d_conflict_count;
  cudaMalloc(&d_conflict_count, sizeof(int));

  // Build normal hash table
  cudaMemset(d_conflict_count, 0, sizeof(int));
  printf("start building normal hash table...\n");
  build_normal_kernel<<<(node_ids.size() + 1024 - 1) / 1024, 1024>>>(
      d_hash_table_normal_, d_hash_length_, d_hash_tables_offset_,
      node_ids.size(), bucket_num_, d_node_ids, d_elements, d_conflict_count,
      bucket_size);
  cudaDeviceSynchronize();
  int conflict_count;
  cudaMemcpy(&conflict_count, d_conflict_count, sizeof(int),
             cudaMemcpyDeviceToHost);
  printf("Normal hash table built with %d conflicts.\n", conflict_count);

  // Build hierarchical hash table
  cudaMemset(d_conflict_count, 0, sizeof(int));
  printf("start building hierarchical hash table...\n");
  build_hierarchical_kernel<<<(node_ids.size() + 1024 - 1) / 1024, 1024>>>(
      d_hash_table_hierarchical_, d_hash_length_, d_hash_tables_offset_,
      node_ids.size(), bucket_num_, max_length_, d_node_ids, d_elements,
      d_conflict_count, bucket_size);
  cudaDeviceSynchronize();
  cudaMemcpy(&conflict_count, d_conflict_count, sizeof(int),
             cudaMemcpyDeviceToHost);
  printf("Hierarchical hash table built with %d conflicts.\n", conflict_count);

  cudaFree(d_conflict_count);
  cudaFree(d_node_ids);
  cudaFree(d_elements);
  cudaFree(d_offsets);
}
