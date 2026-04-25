#include <cuda_runtime.h>

#include <vector>

#include "../common/hopscotch_hash.cuh"
#include "../common/utils.cuh"
#include "sss_baselines.h"

/**
 * @file sss_hopscotch.cu
 * @brief SSS任务跳房子哈希基线实现 - 扁平化每节点版本
 *
 * 重构特点：
 * - 每个节点独立构建跳房子哈希表,保存其邻接点
 * - 所有哈希表扁平化连续存储在一个大数组中
 * - 通过offset数组定位每个节点的哈希表起始位置
 */

/**
 * @brief SSS扁平化跳房子哈希查询内核
 * 每个warp处理一个查询对,动态负载均衡
 * 使用每个节点独立哈希表查询
 */
__global__ void sss_hopscotch_kernel(
    HopscotchHash* hash, int num_pairs, int const* __restrict__ d_vertexs,
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
        found = hash->contains(v, neighbor);
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

int run_sss_hopscotch(int num_pairs, int num_nodes, int const* d_vertexs,
                      int const* d_csr_cols_for_vertexs, int const* d_csr_cols,
                      int const* d_csr_offsets,
                      std::vector<int> const& csr_offsets_host,
                      std::vector<int> const& csr_cols_host, float threshold,
                      int grid_size, int block_size, int CHUNK_SIZE,
                      float load_factor, cudaStream_t stream) {
  // 预先计算每个节点度数,计算容量分配
  std::vector<int> degrees;
  for (int i = 0; i < num_nodes; i++) {
    degrees.push_back(csr_offsets_host[i + 1] - csr_offsets_host[i]);
  }

  // 创建跳房子哈希表 - 每个节点独立,扁平化存储
  HopscotchHash* d_hash;
  cudaMallocManaged(&d_hash, sizeof(HopscotchHash));
  new (d_hash) HopscotchHash(num_nodes, degrees, load_factor);

  // 批量插入所有节点的邻接点
  int failed = d_hash->bulk_insert(csr_offsets_host, csr_cols_host);
  printf("[跳房子哈希] 插入失败: %d 个key\n", failed);

  // 分配结果数组
  int* d_results;
  int* d_G_index;
  cudaMalloc(&d_results, num_pairs * sizeof(int));
  cudaMalloc(&d_G_index, sizeof(int));
  cudaMemset(d_G_index, 0, sizeof(int));
  cudaMemset(d_results, 0, num_pairs * sizeof(int));

  // 启动查询内核
  sss_hopscotch_kernel<<<grid_size, block_size>>>(
      d_hash, num_pairs, d_vertexs, d_csr_cols_for_vertexs, d_csr_cols,
      d_csr_offsets, d_results, d_G_index, CHUNK_SIZE, threshold);

  // 读取结果并累加
  std::vector<int> h_results(num_pairs);
  cudaMemcpy(h_results.data(), d_results,
             static_cast<size_t>(num_pairs) * sizeof(int),
             cudaMemcpyDeviceToHost);

  int result_count = 0;
  for (int x : h_results) {
    result_count += x;
  }

  size_t total_cap = d_hash->total_capacity();

  // 清理
  d_hash->~HopscotchHash();
  cudaFree(d_hash);
  cudaFree(d_results);
  cudaFree(d_G_index);

  printf("[跳房子哈希] 节点数: %d, 总容量: %zu\n", num_nodes, total_cap);

  return result_count;
}
