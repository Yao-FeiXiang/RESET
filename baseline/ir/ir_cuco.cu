#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>

#include "../common/cuco_baseline.cuh"
#include "ir_cuco.cuh"

/**
 * @brief 编码(term, doc)对为64位键
 *
 * 将32位term和32位doc编码为一个64位整数存储
 */
__host__ __device__ __forceinline__ std::uint64_t encode_posting_key(int term,
                                                                     int doc) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(term)) << 32) |
         static_cast<std::uint32_t>(doc);
}

/**
 * @brief IR cuCollections基线类
 *
 * 继承公共基类,实现IR特有的数据编码和内核启动
 */
class IRCuCollections : public CuCollectionsStaticSetBase<std::uint64_t> {
 public:
  using base_type = CuCollectionsStaticSetBase<std::uint64_t>;

  /**
   * @brief 构造函数
   * @param total_postings 总posting数量
   */
  explicit IRCuCollections(std::size_t total_postings)
      : base_type(total_postings) {}

  /**
   * @brief 从主机端倒排索引构建集合
   * @param offsets 倒排索引偏移数组
   * @param postings 倒排记录数组
   * @param inverted_index_num 倒排索引数量
   * @param stream CUDA流
   */
  void build(std::vector<int> const& offsets, std::vector<int> const& postings,
             int inverted_index_num, cudaStream_t stream = 0) {
    std::size_t total = postings.size();
    thrust::host_vector<key_type> h_keys(total);

    // 编码所有(term, doc)对
    for (int term = 0; term < inverted_index_num; ++term) {
      int start = offsets[term];
      int end = offsets[term + 1];
      for (int p = start; p < end; ++p) {
        h_keys[p] = encode_posting_key(term, postings[p]);
      }
    }

    // 插入键到cuCollections集合
    insert_keys(h_keys, stream);
  }
};

/**
 * @brief IR查询内核 - 使用cuCollections进行交集查询
 * 完全对齐baseline实现：动态负载均衡,warp同步收集结果
 */
template <typename SetContainsRef>
__global__ void ir_cuco_kernel(int const* inverted_index,
                               int const* inverted_index_offsets,
                               int const* query, int const* query_offsets,
                               int query_num, int* result,
                               long long const* result_offsets,
                               int* result_count, int* G_index, int CHUNK_SIZE,
                               SetContainsRef set_ref) {
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
          auto qk = encode_posting_key(current_term, key);
          found = set_ref.contains(qk);
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
int run_ir_cuco(int inverted_index_num, int query_num,
                int const* d_inverted_index,
                int const* d_inverted_index_offsets, int const* d_query,
                int const* d_query_offsets, int* d_result,
                long long const* d_result_offsets, int* d_result_count,
                int* d_G_index, int CHUNK_SIZE,
                std::vector<int> const& inverted_index_offsets_host,
                std::vector<int> const& inverted_index_host, int grid_size,
                int block_size, float load_factor, cudaStream_t stream) {
  // 构建cuCollections集合
  std::size_t total_postings = inverted_index_host.size();
  IRCuCollections ir_cuco(total_postings);
  ir_cuco.build(inverted_index_offsets_host, inverted_index_host,
                inverted_index_num, stream);

  // 启动内核
  auto contains_ref = ir_cuco.get_contains_ref();
  ir_cuco_kernel<<<grid_size, block_size, 0, stream>>>(
      d_inverted_index, d_inverted_index_offsets, d_query, d_query_offsets,
      query_num, d_result, d_result_offsets, d_result_count, d_G_index,
      CHUNK_SIZE, contains_ref);

  // 同步并返回结果
  cudaStreamSynchronize(stream);
  return 0;
}
