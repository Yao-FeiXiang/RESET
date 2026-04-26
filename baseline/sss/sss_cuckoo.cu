#include <cuda_runtime.h>

#include <vector>

#include "../common/cuckoo_hash.cuh"
#include "../common/utils.cuh"
#include "sss_baselines.h"

/**
 * @file sss_cuckoo.cu
 * @brief SSS任务布谷鸟哈希基线实现 - 扁平化每节点版本
 *
 * 重构特点：
 * - 每个节点独立构建布谷鸟哈希表,保存其邻接点
 * - 所有哈希表扁平化连续存储在一个大数组中
 * - 通过offset数组定位每个节点的哈希表起始位置
 * - 添加30秒超时机制,避免无限循环构建
 */

/**
 * @brief SSS扁平化布谷鸟哈希查询内核
 * 每个warp处理一个查询对,动态负载均衡
 * 使用每个节点独立哈希表查询,支持stash
 */
__global__ void sss_cuckoo_kernel(
    int* d_table, long long* d_offsets, long long* d_stash_starts,
    int* d_stash_data, int num_pairs, int const* __restrict__ d_vertexs,
    int const* __restrict__ d_csr_cols_for_edges,
    int const* __restrict__ d_csr_cols, int const* __restrict__ d_csr_offsets,
    int* __restrict__ d_results, int* __restrict__ d_G_index, int CHUNK_SIZE,
    float threshold) {
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int num_warps = blockDim.x / 32;

  int pair_idx = (blockIdx.x * num_warps + warp_id) * CHUNK_SIZE;
  int pair_end = pair_idx + CHUNK_SIZE;

  while (pair_idx < num_pairs) {
    int u = d_vertexs[pair_idx];
    int v = d_csr_cols_for_edges[pair_idx];

    int u_size = d_csr_offsets[u + 1] - d_csr_offsets[u];
    int v_size = d_csr_offsets[v + 1] - d_csr_offsets[v];

    // 小度优先：总是让u是度数小的一方,减少遍历次数
    if (u_size > v_size) {
      int temp = u;
      u = v;
      v = temp;
      int temp_size = u_size;
      u_size = v_size;
      v_size = temp_size;
    }

    int u_neighbour_start = d_csr_offsets[u];
    int result_num = 0;
    int num_iters = (u_size + 32 - 1) / 32;

    for (int iter = 0; iter < num_iters; iter++) {
      int j = lane_id + iter * 32;
      bool active = (j < u_size);
      bool found = false;

      if (active) {
        int neighbor = d_csr_cols[u_neighbour_start + j];
        // 在v节点的哈希表中查找邻接点是否存在
        // 直接内联查询,检查三个候选位置 + stash
        long long start = d_offsets[v];
        int capacity = static_cast<int>(d_offsets[v + 1] - start);

        // 三个完全独立的哈希,和插入完全一致
        // 布谷鸟不变性：key一定在这三个位置之一或者在stash中
        uint64_t node_salt = (uint64_t)v * 1111111111ULL;
        uint64_t k1 = (uint64_t)neighbor ^ node_salt;
        uint64_t k2 = (uint64_t)neighbor + node_salt;
        uint64_t k3 = (uint64_t)neighbor * (node_salt | 0x12345678ULL);

        // 第一个哈希 - 和插入完全一致
        k1 ^= k1 >> 33;
        k1 *= 0xff51afd7ed558ccdULL;
        k1 ^= k1 >> 33;
        k1 *= 0xc4ceb9fe1a85ec53ULL;
        k1 ^= k1 >> 33;
        long long pos1 = start + ((long long)k1 & (capacity - 1));

        if (d_table[pos1] == neighbor) {
          found = true;
        } else {
          // 第二个哈希 - 和插入完全一致
          k2 ^= k2 >> 33;
          k2 *= 0xd6e8feb86b5680bfULL;
          k2 ^= k2 >> 33;
          k2 *= 0xcaaf0aaf9603b2e5ULL;
          k2 ^= k2 >> 33;
          long long pos2 = start + ((long long)k2 & (capacity - 1));

          if (d_table[pos2] == neighbor) {
            found = true;
          } else {
            // 第三个哈希 - 和插入完全一致
            k3 ^= k3 >> 33;
            k3 *= 0xaed549a354e3eb1bULL;
            k3 ^= k3 >> 33;
            k3 *= 0x8058d66927ac9adfULL;
            k3 ^= k3 >> 33;
            long long pos3 = start + ((long long)k3 & (capacity - 1));

            if (d_table[pos3] == neighbor) {
              found = true;
            } else {
              // 主表没找到,检查stash
              long long stash_start = d_stash_starts[v];
              int stash_count =
                  static_cast<int>(d_stash_starts[v + 1] - d_stash_starts[v]);
              for (int i = 0; i < stash_count; i++) {
                if (d_stash_data[stash_start + i] == neighbor) {
                  found = true;
                  break;
                }
              }
            }
          }
        }
      }

      unsigned int found_mask = __ballot_sync(0xffffffff, found);
      int step_found = __popc(found_mask);
      result_num += step_found;
    }

    result_num = __shfl_sync(0xffffffff, result_num, 0);
    if (lane_id == 0) {
      float jaccard = result_num / (float)(u_size + v_size - result_num);
      if (jaccard >= threshold) {
        d_results[pair_idx] = 1;
      }
    }

    __syncwarp();

    pair_idx++;
    if (pair_idx == pair_end) {
      if (lane_id == 0) {
        pair_idx = atomicAdd(d_G_index, CHUNK_SIZE);
      }
      pair_idx = __shfl_sync(0xffffffff, pair_idx, 0);
      pair_end = pair_idx + CHUNK_SIZE;
    }
  }
}

std::pair<int, float> run_sss_cuckoo(
    int num_pairs, int num_nodes, int const* d_vertexs,
    int const* d_csr_cols_for_edges, int const* d_csr_cols,
    int const* d_csr_offsets, std::vector<int> const& csr_offsets_host,
    std::vector<int> const& csr_cols_host, float threshold, int grid_size,
    int block_size, int CHUNK_SIZE, float load_factor, cudaStream_t stream) {
  // 计算每个节点的度数,构建扁平化布谷鸟哈希
  std::vector<int> degrees(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    degrees[i] = csr_offsets_host[i + 1] - csr_offsets_host[i];
  }

  // 记录开始时间,用于超时检测(30秒超时)
  clock_t start_time = clock();

  // 创建扁平化布谷鸟哈希表 - 每个节点独立构建
  FlatCuckooHash* d_cuckoo;
  cudaMallocManaged(&d_cuckoo, sizeof(FlatCuckooHash));
  new (d_cuckoo) FlatCuckooHash(num_nodes, degrees, load_factor);

  // 检查是否超时
  double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
  if (elapsed > 30.0) {
    printf("错误：构建布谷鸟哈希超时(超过30秒),强制退出\n");
    d_cuckoo->~FlatCuckooHash();
    cudaFree(d_cuckoo);
    return {-1, 0.0f};
  }

  // 批量插入所有节点的邻接点
  int failed = d_cuckoo->bulk_insert(csr_offsets_host, csr_cols_host);
  if (failed > 0) {
    printf("警告：%d 个键插入失败,可能影响结果正确性\n", failed);
  }

  // 再次检查超时
  elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
  if (elapsed > 30.0) {
    printf("错误：构建布谷鸟哈希超时(超过30秒),强制退出\n");
    d_cuckoo->~FlatCuckooHash();
    cudaFree(d_cuckoo);
    return {-1, 0.0f};
  }

  // 获取设备端指针
  int* d_table = d_cuckoo->get_device_table();
  long long* d_offsets = d_cuckoo->get_device_offsets();
  long long* d_stash_starts = d_cuckoo->get_device_stash_starts();
  int* d_stash_data = d_cuckoo->get_device_stash_data();

  // 分配结果数组 - 每个查询对一个标志
  int* d_results = nullptr;
  cudaMalloc(&d_results, static_cast<size_t>(num_pairs) * sizeof(int));
  cudaMemsetAsync(d_results, 0, static_cast<size_t>(num_pairs) * sizeof(int),
                  stream);

  // 动态负载均衡的全局索引
  int* d_G_index = nullptr;
  cudaMalloc(&d_G_index, sizeof(int));
  int h_G_index = grid_size * block_size / 32 * CHUNK_SIZE;
  cudaMemcpyAsync(d_G_index, &h_G_index, sizeof(int), cudaMemcpyHostToDevice,
                  stream);

  // 创建CUDA事件用于计时
  cudaEvent_t kernel_start, kernel_stop;
  cudaEventCreate(&kernel_start);
  cudaEventCreate(&kernel_stop);

  // 记录开始事件,然后启动内核
  cudaEventRecord(kernel_start, stream);

  // 启动内核,直接传递扁平化哈希表数据指针、偏移和stash
  sss_cuckoo_kernel<<<grid_size, block_size, 0, stream>>>(
      d_table, d_offsets, d_stash_starts, d_stash_data, num_pairs, d_vertexs,
      d_csr_cols_for_edges, d_csr_cols, d_csr_offsets, d_results, d_G_index,
      CHUNK_SIZE, threshold);

  // 记录停止事件
  cudaEventRecord(kernel_stop, stream);
  cudaEventSynchronize(kernel_stop);

  float kernel_time_ms = 0.0f;
  cudaEventElapsedTime(&kernel_time_ms, kernel_start, kernel_stop);

  // 读取结果并累加
  std::vector<int> h_results(num_pairs);
  cudaMemcpy(h_results.data(), d_results,
             static_cast<size_t>(num_pairs) * sizeof(int),
             cudaMemcpyDeviceToHost);

  int result_count = 0;
  for (int x : h_results) {
    result_count += x;
  }

  // 输出统计信息
  printf("总容量：%lld,插入失败：%d\n", d_cuckoo->get_total_capacity(),
         d_cuckoo->get_failed_count());

  // 清理资源
  cudaEventDestroy(kernel_start);
  cudaEventDestroy(kernel_stop);
  cudaFree(d_results);
  cudaFree(d_G_index);
  d_cuckoo->~FlatCuckooHash();
  cudaFree(d_cuckoo);

  return {result_count, kernel_time_ms};
}
