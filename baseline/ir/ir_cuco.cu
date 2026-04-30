/**
 * @file ir_cuco.cu
 * @brief IR cuCollections基线实现 - 每term版本
 *
 * 架构对齐：
 * - 每term独立哈希表 + 连续存储
 * - 通过offset寻址
 * - 与cuckoo、hopscotch等方法保持一致的架构
 */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>

#include "../common/cuco.cuh"
#include "../common/utils.cuh"
#include "ir_cuco.cuh"

/**
 * @brief IR查询内核 - 使用cuco哈希表进行交集查询
 *
 * 架构变化：不再使用全局表 + 64位(term, doc)编码
 * 改为直接查询term的哈希表中是否存在doc id
 *
 * 对齐：动态负载均衡,warp同步收集结果
 */
__global__ void ir_cuco_kernel(int const* inverted_index,
                               int const* inverted_index_offsets,
                               int const* query, int const* query_offsets,
                               int query_num, int* result,
                               long long const* result_offsets,
                               int* result_count, int* G_index, int CHUNK_SIZE,
                               int* slots, long long* offsets) {
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int num_warps = blockDim.x / 32;

  int vertex = (blockIdx.x * num_warps + warp_id) * CHUNK_SIZE;
  int vertex_end = vertex + CHUNK_SIZE;

  int query_start, query_end, query_len;
  int set_start_pos, set_len;
  int* result_start;

  while (vertex < query_num) {
    query_start = query_offsets[vertex];
    query_end = query_offsets[vertex + 1];
    query_len = query_end - query_start;

    int term0 = query[query_start];
    set_start_pos = inverted_index_offsets[term0];
    set_len = inverted_index_offsets[term0 + 1] - inverted_index_offsets[term0];

    result_start = result + result_offsets[vertex];

    for (int i = lane_id; i < set_len; i += 32) {
      result_start[i] = inverted_index[set_start_pos + i];
    }
    __syncwarp();
    int* set_ptr = result_start;
    int set_size = set_len;

    for (int i = 1; i < query_len; ++i) {
      int current_term = query[query_start + i];

      int result_num = 0;
      int num_iters = (set_size + 32 - 1) / 32;
      for (int iter = 0; iter < num_iters; ++iter) {
        int j = lane_id + iter * 32;
        bool active = (j < set_size);
        bool found = false;
        int key = -1;
        if (active) {
          key = set_ptr[j];
          // 直接查询current_term表中是否存在key
          found = cuco_contains(current_term, key, slots, offsets,
                                CucoHash::EMPTY_KEY);
        }
        unsigned int mask = __ballot_sync(0xffffffff, found);
        int step_found = __popc(mask);
        int write_pos = result_num + __popc(mask & ((1u << lane_id) - 1u));

        if (found && active) {
          result_start[write_pos] = key;
        }
        result_num += step_found;
      }
      set_ptr = result_start;
      set_size = __shfl_sync(0xffffffff, result_num, 0);
      __syncwarp();
      if (set_size == 0) break;
    }

    if (lane_id == 0) {
      result_count[vertex] = set_size;
    }
    __syncwarp();

    vertex++;
    if (vertex == vertex_end) {
      if (lane_id == 0) {
        vertex = atomicAdd(G_index, CHUNK_SIZE);
      }
      vertex = __shfl_sync(0xffffffff, vertex, 0);
      vertex_end = vertex + CHUNK_SIZE;
    }
  }
}

/**
 * @brief 主机端启动IR cuCollections查询的接口
 */
std::pair<int, float> run_ir_cuco(
    int inverted_index_num, int query_num, int const* d_inverted_index,
    int const* d_inverted_index_offsets, int const* d_query,
    int const* d_query_offsets, int* d_result,
    long long const* d_result_offsets, int* d_result_count, int* d_G_index,
    int CHUNK_SIZE, std::vector<int> const& inverted_index_offsets_host,
    std::vector<int> const& inverted_index_host, int grid_size, int block_size,
    float load_factor, cudaStream_t stream) {
  // 设备预热：执行一个空 kernel 来确保 CUDA 上下文完全激活
  warmup_gpu();

  // 构建cuco哈希表（使用原生cuCollections组件）
  std::vector<int> posting_counts(inverted_index_num);
  for (int i = 0; i < inverted_index_num; i++) {
    posting_counts[i] =
        inverted_index_offsets_host[i + 1] - inverted_index_offsets_host[i];
  }

  CucoHash cuco(inverted_index_num, posting_counts, load_factor);
  cuco.bulk_insert(inverted_index_offsets_host, inverted_index_host, stream);

  // 关键：哈希表构建后刷新L2缓存！
  // cuco构建过程中大量GPU操作会把哈希表数据留在L2缓存中
  // 这给cuco带来不公平的缓存优势，必须消除
  flush_l2_cache();

  // 使用cudaEvent_t进行GPU硬件级计时(最科学严谨)
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  ir_cuco_kernel<<<grid_size, block_size, 0, stream>>>(
      d_inverted_index, d_inverted_index_offsets, d_query, d_query_offsets,
      query_num, d_result, d_result_offsets, d_result_count, d_G_index,
      CHUNK_SIZE, cuco.get_device_table(), cuco.get_device_offsets());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float kernel_time_ms = 0.0f;
  cudaEventElapsedTime(&kernel_time_ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // 同步并读取结果计数
  cudaStreamSynchronize(stream);

  // 从设备读取结果计数并累加
  std::vector<int> h_result_count(query_num);
  cudaMemcpy(h_result_count.data(), d_result_count, sizeof(int) * query_num,
             cudaMemcpyDeviceToHost);
  int total_result = 0;
  for (int cnt : h_result_count) {
    total_result += cnt;
  }

  return {total_result, kernel_time_ms};
}
