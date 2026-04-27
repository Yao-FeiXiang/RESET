/**
 * @file sss_cuco.cu
 * @brief SSS cuCollections基线实现 - 每节点版本
 *
 * 架构对齐：
 * - 每节点独立哈希表 + 连续存储
 * - 通过offset寻址
 * - 与cuckoo、hopscotch等方法保持一致的架构
 */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../common/cuco.cuh"
#include "../common/utils.cuh"
#include "sss_cuco.cuh"

/**
 * @brief SSS查询内核 - 使用cuco哈希表计算Jaccard相似度
 *
 * 架构变化：不再使用全局表 + 64位边编码
 * 改为直接查询节点v的哈希表中是否存在邻接点key
 *
 * 对齐：每个边一个结果位置,动态负载均衡
 */
__global__ void sss_cuco_kernel(
    int num_edges, int const* __restrict__ d_vertexs,
    int const* __restrict__ d_csr_cols_for_edges,
    int const* __restrict__ d_csr_cols, int const* __restrict__ d_csr_offsets,
    int* __restrict__ d_slots, long long* __restrict__ d_offsets,
    int* __restrict__ d_results, int* __restrict__ d_G_index, int CHUNK_SIZE,
    float threshold) {
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int num_warps = blockDim.x / 32;

  int edge = (blockIdx.x * num_warps + warp_id) * CHUNK_SIZE;
  int edge_end = edge + CHUNK_SIZE;

  while (edge < num_edges) {
    int u = d_vertexs[edge];
    int v = d_csr_cols_for_edges[edge];

    int u_size = d_csr_offsets[u + 1] - d_csr_offsets[u];
    int v_size = d_csr_offsets[v + 1] - d_csr_offsets[v];

    // 对齐：小度优先 swap
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
        int key = d_csr_cols[u_neighbour_start + j];
        // 直接查询v节点的表中是否存在key
        found = cuco_contains(v, key, d_slots, d_offsets, CucoHash::EMPTY_KEY);
      }

      unsigned int found_mask = __ballot_sync(0xffffffff, found);
      int step_found = __popc(found_mask);
      result_num += step_found;
    }

    result_num = __shfl_sync(0xffffffff, result_num, 0);
    if (lane_id == 0) {
      float jaccard = result_num / (float)(u_size + v_size - result_num);
      if (jaccard >= threshold) {
        d_results[edge] = 1;
      }
    }

    __syncwarp();

    edge++;
    if (edge == edge_end) {
      if (lane_id == 0) {
        edge = atomicAdd(d_G_index, CHUNK_SIZE);
      }
      edge = __shfl_sync(0xffffffff, edge, 0);
      edge_end = edge + CHUNK_SIZE;
    }
  }
}

/**
 * @brief 主机端启动SSS cuCollections查询的接口
 * 完全对齐baseline实现
 */
std::pair<int, float> run_sss_cuco(
    int num_edges, int num_nodes, int const* d_vertexs,
    int const* d_csr_cols_for_edges, int const* d_csr_cols,
    int const* d_csr_offsets, std::vector<int> const& csr_offsets_host,
    std::vector<int> const& csr_cols_host, float threshold, int grid_size,
    int block_size, int CHUNK_SIZE, float load_factor, cudaStream_t stream) {
  cudaEvent_t e1, e2, e3, e4;
  cudaEventCreate(&e1);
  cudaEventCreate(&e2);
  cudaEventCreate(&e3);
  cudaEventCreate(&e4);

  float t_build = 0.0f, t_alloc = 0.0f, t_kernel = 0.0f;

  // 阶段1: 构建cuco哈希表
  cudaEventRecord(e1, stream);

  // 计算每个节点的度数
  std::vector<int> degrees(num_nodes);
  for (int i = 0; i < num_nodes; i++) {
    degrees[i] = csr_offsets_host[i + 1] - csr_offsets_host[i];
  }

  // 创建哈希表并插入
  CucoHash cuco(num_nodes, degrees, load_factor);
  cuco.bulk_insert(csr_offsets_host, csr_cols_host, stream);

  cudaEventRecord(e2, stream);
  cudaEventSynchronize(e2);
  cudaEventElapsedTime(&t_build, e1, e2);

  // 阶段2: 分配结果数组
  int* d_results = nullptr;
  cudaMalloc(&d_results, static_cast<std::size_t>(num_edges) * sizeof(int));
  cudaMemsetAsync(d_results, 0,
                  static_cast<std::size_t>(num_edges) * sizeof(int), stream);

  int* d_G_index = nullptr;
  cudaMalloc(&d_G_index, sizeof(int));
  int h_G_index = grid_size * block_size / 32 * CHUNK_SIZE;
  cudaMemcpyAsync(d_G_index, &h_G_index, sizeof(int), cudaMemcpyHostToDevice,
                  stream);
  cudaEventRecord(e3, stream);
  cudaEventSynchronize(e3);
  cudaEventElapsedTime(&t_alloc, e2, e3);

  // 阶段3: 查询内核
  sss_cuco_kernel<<<grid_size, block_size, 0, stream>>>(
      num_edges, d_vertexs, d_csr_cols_for_edges, d_csr_cols, d_csr_offsets,
      cuco.get_device_table(), cuco.get_device_offsets(), d_results, d_G_index,
      CHUNK_SIZE, threshold);
  cudaEventRecord(e4, stream);
  cudaEventSynchronize(e4);
  cudaEventElapsedTime(&t_kernel, e3, e4);

  // 读取结果并累加
  std::vector<int> h_results(num_edges);
  cudaMemcpy(h_results.data(), d_results,
             static_cast<std::size_t>(num_edges) * sizeof(int),
             cudaMemcpyDeviceToHost);

  int result_count = 0;
  for (int x : h_results) {
    result_count += x;
  }

  cudaFree(d_results);
  cudaFree(d_G_index);
  cudaEventDestroy(e1);
  cudaEventDestroy(e2);
  cudaEventDestroy(e3);
  cudaEventDestroy(e4);

  cudaStreamSynchronize(stream);
  return {result_count, t_kernel};
}
