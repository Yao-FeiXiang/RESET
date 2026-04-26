#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../common/cuco_baseline.cuh"
#include "tc_cuco.cuh"

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
 * @brief TC cuCollections基线类
 *
 * 继承公共基类,实现TC特有的数据编码和内核启动
 */
class TCCuCollections : public CuCollectionsStaticSetBase<std::uint64_t> {
 public:
  using base_type = CuCollectionsStaticSetBase<std::uint64_t>;

  /**
   * @brief 构造函数
   * @param total_edges 总边数量
   */
  explicit TCCuCollections(std::size_t total_edges)
      : base_type(total_edges, 2.0f) {}

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
 * @brief TC计数内核 - 使用cuCollections查找三角形
 * 完全对齐baseline实现：小度数优先优化 + 动态负载均衡
 */
template <typename SetContainsRef>
__global__ void tc_cuco_kernel(int num_edges, int const* __restrict__ d_vertexs,
                               int const* __restrict__ d_edge_cols,
                               int const* __restrict__ d_csr_row,
                               int const* __restrict__ d_csr_cols,
                               SetContainsRef set_ref,
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
        if (w > u) {
          auto qk = encode_edge_key(v, w);
          if (set_ref.contains(qk)) {
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

  unsigned int found_mask = __ballot_sync(0xffffffff, thread_count > 0);
  unsigned long long total = thread_count;
  unsigned int mask = found_mask;
  while (mask) {
    int bit = __ffs(mask) - 1;
    unsigned long long count = thread_count;
    total += __shfl_sync(0xffffffff, count, bit);
    mask ^= (1 << bit);
  }

  if (lane_id == 0 && total > 0) {
    atomicAdd(d_total_count, total);
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

  // 构建cuCollections集合
  std::size_t total_edges = csr_cols_host.size();
  TCCuCollections tc_cuco(total_edges);
  tc_cuco.build(csr_row_host, csr_cols_host, num_nodes, stream);

  // 启动内核
  auto contains_ref = tc_cuco.get_contains_ref();

  // 使用cudaEvent_t进行GPU硬件级计时(最科学严谨)
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  tc_cuco_kernel<<<grid_size, block_size, 0, stream>>>(
      num_edges, d_vertexs, d_csr_cols_for_traversal, d_csr_row,
      d_csr_cols_for_traversal, contains_ref, d_triangle_count, d_edge_index,
      CHUNK_SIZE);
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

  cudaStreamSynchronize(stream);
  return {triangle_count, kernel_time_ms};
}
