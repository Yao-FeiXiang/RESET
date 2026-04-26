#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../common/cuco_baseline.cuh"
#include "sss_cuco.cuh"

/**
 * @brief 编码(u, v)边对为64位键
 * 确保u < v避免重复存储
 */
__host__ __device__ __forceinline__ std::uint64_t encode_edge_key(int u,
                                                                  int v) {
  if (u > v) {
    int tmp = u;
    u = v;
    v = tmp;
  }
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(u)) << 32) |
         static_cast<std::uint32_t>(v);
}

/**
 * @brief SSS cuCollections基线类
 *
 * 继承公共基类,实现SSS特有的数据编码和内核启动
 */
class SSSCuCollections : public CuCollectionsStaticSetBase<std::uint64_t> {
 public:
  using base_type = CuCollectionsStaticSetBase<std::uint64_t>;

  /**
   * @brief 构造函数
   * @param total_edges 总边数量
   */
  explicit SSSCuCollections(std::size_t total_edges) : base_type(total_edges) {}

  /**
   * @brief 从CSR格式构建边集合
   * @param csr_offsets CSR偏移数组
   * @param csr_cols CSR列数组
   * @param num_nodes 节点数量
   * @param stream CUDA流
   */
  void build(std::vector<int> const& csr_offsets,
             std::vector<int> const& csr_cols, int num_nodes,
             cudaStream_t stream = 0) {
    std::size_t total_edges = csr_cols.size();
    thrust::host_vector<key_type> h_keys(total_edges);

    // 编码所有边 - 存储每条边,不过滤
    for (int u = 0; u < num_nodes; ++u) {
      int start = csr_offsets[u];
      int end = csr_offsets[u + 1];
      for (int p = start; p < end; ++p) {
        int v = csr_cols[p];
        h_keys[p] = encode_edge_key(u, v);
      }
    }

    // 插入键到cuCollections集合
    insert_keys(h_keys, stream);
  }
};

/**
 * @brief SSS查询内核 - 使用cuCollections计算Jaccard相似度
 * 完全对齐baseline实现：每个边一个结果位置,动态负载均衡
 */
template <typename SetContainsRef>
__global__ void sss_cuco_kernel(
    int num_edges, int const* __restrict__ d_vertexs,
    int const* __restrict__ d_csr_cols_for_edges,
    int const* __restrict__ d_csr_cols, int const* __restrict__ d_csr_offsets,
    SetContainsRef set_ref, int* __restrict__ d_results,
    int* __restrict__ d_G_index, int CHUNK_SIZE, float threshold) {
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
        auto qk = encode_edge_key(v, key);
        found = set_ref.contains(qk);
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

  // 阶段1: 构建cuCollections集合
  cudaEventRecord(e1, stream);
  std::size_t total_edges = csr_cols_host.size();
  SSSCuCollections sss_cuco(total_edges);
  sss_cuco.build(csr_offsets_host, csr_cols_host, num_nodes, stream);
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
  auto contains_ref = sss_cuco.get_contains_ref();
  sss_cuco_kernel<<<grid_size, block_size, 0, stream>>>(
      num_edges, d_vertexs, d_csr_cols_for_edges, d_csr_cols, d_csr_offsets,
      contains_ref, d_results, d_G_index, CHUNK_SIZE, threshold);
  cudaEventRecord(e4, stream);
  cudaEventSynchronize(e4);
  cudaEventElapsedTime(&t_kernel, e3, e4);

  printf("[cuCO 细分] 构建: %.3f ms, 分配: %.3f ms, 内核: %.3f ms\n", t_build,
         t_alloc, t_kernel);

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
