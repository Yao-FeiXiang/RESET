/**
 * @file tc_cuco.cu
 * @brief TC cuCollections基线实现 - 每节点版本
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
#include "tc_cuco.cuh"

/**
 * @brief TC计数内核 - 使用cuco哈希表查找三角形
 *
 * 架构变化：不再使用全局表 + 64位边编码
 * 改为直接查询节点v的哈希表中是否存在邻接点w
 *
 * 对齐：小度数优先优化 + 动态负载均衡
 */
__global__ void tc_cuco_kernel(int num_edges, int const* __restrict__ d_vertexs,
                               int const* __restrict__ d_edge_cols,
                               int const* __restrict__ d_csr_row,
                               int const* __restrict__ d_csr_cols,
                               int* __restrict__ d_slots,
                               long long* __restrict__ d_offsets,
                               unsigned long long* d_total_count,
                               int* d_edge_index, int CHUNK_SIZE) {
  __shared__ unsigned long long block_count;
  if (threadIdx.x == 0) block_count = 0;
  __syncthreads();

  unsigned long long thread_count = 0;

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int num_warps = blockDim.x / 32;

  int edge = (blockIdx.x * num_warps + warp_id) * CHUNK_SIZE;
  int edge_end = edge + CHUNK_SIZE;

  while (edge < num_edges) {
    int u = d_vertexs[edge];
    int v = d_edge_cols[edge];

    int u_start = d_csr_row[u];
    int u_end = d_csr_row[u + 1];
    int v_start = d_csr_row[v];
    int v_end = d_csr_row[v + 1];

    int u_size = u_end - u_start;
    int v_size = v_end - v_start;

    // 小度数优先优化 - 减少迭代次数
    if (u_size > v_size) {
      int tmp = u;
      u = v;
      v = tmp;
      int tmp_s = u_size;
      u_size = v_size;
      v_size = tmp_s;
      u_start = d_csr_row[u];
      u_end = d_csr_row[u + 1];
    }

    int num_iters = (u_size + 32 - 1) / 32;
    for (int iter = 0; iter < num_iters; ++iter) {
      int j = lane_id + iter * 32;
      if (j < u_size) {
        int w = d_csr_cols[u_start + j];
        // 关键约束：只计数 w > v 的公共邻居，确保三角形(u, v, w)只被计数一次
        // 由于u < v（TC格式保证），w > v 意味着 u < v < w，三角形唯一由边(u,
        // v)计数
        if (w > v) {
          // 直接查询v节点的表中是否存在w
          if (cuco_contains(v, w, d_slots, d_offsets, CucoHash::EMPTY_KEY)) {
            thread_count++;
          }
        }
      }
    }

    __syncwarp();

    edge++;
    if (edge == edge_end) {
      if (lane_id == 0) {
        edge = atomicAdd(d_edge_index, CHUNK_SIZE);
      }
      edge = __shfl_sync(0xffffffff, edge, 0);
      edge_end = edge + CHUNK_SIZE;
    }
  }

  // 将线程计数累加到块计数（与基准实现对齐）
  atomicAdd(&block_count, thread_count);
  __syncthreads();

  // 将块计数累加到全局结果
  if (threadIdx.x == 0) {
    atomicAdd(d_total_count, block_count);
  }
}

/**
 * @brief 主机端启动TC cuCollections计数的接口
 * 完全对齐baseline实现
 */
std::pair<unsigned long long, float> run_tc_cuco(
    int num_nodes, int num_edges, int const* d_vertexs, int const* d_csr_row,
    int const* d_csr_cols_for_traversal, std::vector<int> const& csr_row_host,
    std::vector<int> const& csr_cols_host, int grid_size, int block_size,
    int CHUNK_SIZE, float load_factor, cudaStream_t stream) {
  // 统一GPU预热
  warmup_gpu();

  // 分配结果计数空间
  unsigned long long* d_triangle_count;
  cudaMalloc(&d_triangle_count, sizeof(unsigned long long));
  cudaMemset(d_triangle_count, 0, sizeof(unsigned long long));

  // 动态负载均衡索引
  int* d_edge_index = nullptr;
  cudaMalloc(&d_edge_index, sizeof(int));
  int h_edge_index = grid_size * block_size / 32 * CHUNK_SIZE;
  cudaMemcpyAsync(d_edge_index, &h_edge_index, sizeof(int),
                  cudaMemcpyHostToDevice, stream);

  // 构建cuco哈希表（使用原生cuCollections组件）
  std::vector<int> degrees(num_nodes);
  for (int i = 0; i < num_nodes; i++) {
    degrees[i] = csr_row_host[i + 1] - csr_row_host[i];
  }

  CucoHash cuco(num_nodes, degrees, load_factor);
  cuco.bulk_insert(csr_row_host, csr_cols_host, stream);

  // 关键：哈希表构建后刷新L2缓存！
  // cuco构建过程中大量GPU操作会把哈希表数据留在L2缓存中
  // 这给cuco带来不公平的缓存优势，必须消除
  flush_l2_cache();

  // 使用cudaEvent_t进行GPU硬件级计时(最科学严谨)
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  tc_cuco_kernel<<<grid_size, block_size, 0, stream>>>(
      num_edges, d_vertexs, d_csr_cols_for_traversal, d_csr_row,
      d_csr_cols_for_traversal, cuco.get_device_table(),
      cuco.get_device_offsets(), d_triangle_count, d_edge_index, CHUNK_SIZE);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float kernel_time_ms = 0.0f;
  cudaEventElapsedTime(&kernel_time_ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // 读取结果
  unsigned long long triangle_count;
  cudaMemcpy(&triangle_count, d_triangle_count, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  cudaFree(d_triangle_count);
  cudaFree(d_edge_index);

  cudaStreamSynchronize(stream);
  return {triangle_count, kernel_time_ms};
}
