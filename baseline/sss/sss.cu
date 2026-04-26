#include <cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include "sss.cuh"

using namespace cooperative_groups;
#define warpSize 32

__device__ __forceinline__ bool search_in_hashtable(int key, int* hashtable,
                                                    int bucket_num, int bucket,
                                                    int hash_length,
                                                    int bucket_size) {
  // 在哈希表中查找指定键
  bool found = false;
  int index = 0;
  while (1) {
    if (hashtable[bucket + index * bucket_num] == key) {
      found = true;
      break;
    } else if (hashtable[bucket + index * bucket_num] == -1) {
      // 遇到空标记,停止查找
      break;
    }
    index++;
    if (index == bucket_size) {
      // 当前桶已满,探测下一个位置
      index = 0;
      bucket = (bucket + 1) & (hash_length - 1);
    }
  }
  return found;
}

__global__ void sss_kernel(int num_edges, int* vertexs, int* csr_cols,
                           int* csr_offsets, int* hash_length, int* hash_table,
                           long long* hash_table_offsets, int* results,
                           int* G_index, int CHUNK_SIZE, bool opt,
                           int max_length, int bucket_num, float threshold,
                           int bucket_size) {
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int num_warps = blockDim.x / warpSize;
  int edge = (blockIdx.x * num_warps + warp_id) * CHUNK_SIZE;
  int edge_end = edge + CHUNK_SIZE;
  while (edge < num_edges) {
    // 获取当前边对应的两个顶点u和v
    int u = vertexs[edge];
    int v = csr_cols[edge];
    int u_size = csr_offsets[u + 1] - csr_offsets[u];
    int v_size = csr_offsets[v + 1] - csr_offsets[v];
    // 总是让较小集合遍历,较大集合构建哈希表加速交集计算
    if (u_size > v_size) {
      int temp = u;
      u = v;
      v = temp;
      int temp_size = u_size;
      u_size = v_size;
      v_size = temp_size;
    }
    // 获取较大集合v的哈希表
    int* hash_table_start = &hash_table[hash_table_offsets[v]];
    int length = hash_length[v];
    int u_neightbour_start = csr_offsets[u];
    int result_num = 0;
    // 遍历较小集合u中的每个邻居,检查是否在v的哈希表中
    int num_iters = (u_size + warpSize - 1) / warpSize;
    for (int iter = 0; iter < num_iters; iter++) {
      int j = lane_id + iter * warpSize;
      bool active = (j < u_size);
      bool found = false;
      if (active) {
        int key = csr_cols[u_neightbour_start + j];
        // 根据优化选项选择分层哈希或普通哈希计算桶位置
        int bucket = (opt) ? d_hash_hierarchical(key, length, max_length)
                           : d_hash_normal(key, length);
        found = search_in_hashtable(key, hash_table_start, bucket_num, bucket,
                                    length, bucket_size);
      }
      // 使用warp投票收集找到的元素
      unsigned int found_mask = __ballot_sync(0xffffffff, found);
      int step_found = __popc(found_mask);
      result_num += step_found;
    }
    // 广播相交元素数量
    result_num = __shfl_sync(0xffffffff, result_num, 0);
    // 计算Jaccard相似度,判断是否超过阈值
    if (lane_id == 0) {
      float jaccard = result_num / (float)(u_size + v_size - result_num);
      if (jaccard >= threshold) {
        results[edge] = 1;
      }
    }
    __syncwarp();
    // 处理下一个边对,使用原子操作动态分配工作
    edge++;
    if (edge == edge_end) {
      if (lane_id == 0) {
        edge = atomicAdd(G_index, CHUNK_SIZE);
      }
      edge = __shfl_sync(0xffffffff, edge, 0);
      edge_end = edge + CHUNK_SIZE;
    }
  }
}

// 元组比较器,用于对CSR列按行号和桶位置排序
// 按行号升序,相同行内按元素模max_length升序
struct TupleComparator {
  int max_length;
  __host__ __device__ TupleComparator(int ml = 8) : max_length(ml) {}
  __host__ __device__ bool operator()(const thrust::tuple<int, int>& a,
                                      const thrust::tuple<int, int>& b) const {
    int ra = thrust::get<0>(a);
    int rb = thrust::get<0>(b);
    if (ra != rb) return ra < rb;
    int ca = thrust::get<1>(a);
    int cb = thrust::get<1>(b);
    return (ca & (max_length - 1)) < (cb & (max_length - 1));
  }
};

// GPU排序CSR列,按行和桶位置重排元素,提高局部性
void gpu_sort_csr_cols(std::vector<int>& csr_cols,
                       const std::vector<int>& csr_row, int num_nodes,
                       int max_length) {
  int total_edges = csr_cols.size();
  std::vector<int> rows_host(total_edges);
  for (int i = 0; i < num_nodes; ++i) {
    int start = csr_row[i];
    int end = csr_row[i + 1];
    for (int p = start; p < end; ++p) rows_host[p] = i;
  }

  thrust::device_vector<int> d_csr_cols = csr_cols;
  thrust::device_vector<int> d_rows = rows_host;

  auto first = thrust::make_zip_iterator(
      thrust::make_tuple(d_rows.begin(), d_csr_cols.begin()));
  auto last = thrust::make_zip_iterator(
      thrust::make_tuple(d_rows.end(), d_csr_cols.end()));
  thrust::sort(first, last, TupleComparator(max_length));
  thrust::copy(d_csr_cols.begin(), d_csr_cols.end(), csr_cols.begin());
}

SSSBaseline::~SSSBaseline() {
  if (d_vertexs_) cudaFree(d_vertexs_);
  if (d_csr_cols_sorted_) cudaFree(d_csr_cols_sorted_);
  if (d_results_) cudaFree(d_results_);
  if (d_G_index_) cudaFree(d_G_index_);
}

void SSSBaseline::load_vertex_pairs(const std::string& vertexs_path) {
  read_i32_vec(vertexs_path, vertexs_);
  num_edges_ = vertexs_.size();

  // 复制到设备端
  cudaMalloc(&d_vertexs_, num_edges_ * sizeof(int));
  cudaMemcpy(d_vertexs_, vertexs_.data(), num_edges_ * sizeof(int),
             cudaMemcpyHostToDevice);
}

void SSSBaseline::allocate_buffers() {
  // 分配结果缓冲区
  cudaMalloc(&d_results_, num_edges_ * sizeof(int));
  cudaMemset(d_results_, 0, num_edges_ * sizeof(int));
}

// 预排序CSR列(用于hierarchical哈希),在计时前完成
void SSSBaseline::pre_sort_csr_cols(CSRGraph& graph) {
  std::vector<int> sorted_cols = graph.get_host_elements();
  gpu_sort_csr_cols(sorted_cols, graph.get_host_offsets(),
                    graph.get_num_nodes(), get_max_length());
  cudaMalloc(&d_csr_cols_sorted_, sorted_cols.size() * sizeof(int));
  cudaMemcpy(d_csr_cols_sorted_, sorted_cols.data(),
             sorted_cols.size() * sizeof(int), cudaMemcpyHostToDevice);
}

std::pair<int, float> SSSBaseline::run_hierarchical(
    CSRGraph& graph, int CHUNK_SIZE, int grid_size, int block_size,
    int bucket_size, float threshold, bool sorted) {
  int h_G_index = grid_size * block_size / warpSize * CHUNK_SIZE;
  cudaMalloc(&d_G_index_, sizeof(int));
  cudaMemcpy(d_G_index_, &h_G_index, sizeof(int), cudaMemcpyHostToDevice);

  // 使用预排序的CSR列(排序已在计时外完成)
  int* d_csr_cols_ptr =
      sorted ? d_csr_cols_sorted_ : graph.get_device_elements();

  cudaMemset(d_results_, 0, num_edges_ * sizeof(int));

  // 使用cudaEvent_t进行GPU硬件级计时(最科学严谨)
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  sss_kernel<<<grid_size, block_size>>>(
      num_edges_, d_vertexs_, d_csr_cols_ptr, graph.get_device_offsets(),
      get_d_hash_length(), get_d_hash_hierarchical(),
      get_d_hash_tables_offset(), d_results_, d_G_index_, CHUNK_SIZE, true,
      get_max_length(), get_bucket_num(), threshold, bucket_size);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float kernel_time_ms = 0.0f;
  cudaEventElapsedTime(&kernel_time_ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return {get_result_count(), kernel_time_ms};
}

std::pair<int, float> SSSBaseline::run_normal(CSRGraph& graph, int CHUNK_SIZE,
                                              int grid_size, int block_size,
                                              int bucket_size, float threshold,
                                              bool sorted) {
  int h_G_index = grid_size * block_size / warpSize * CHUNK_SIZE;
  cudaMalloc(&d_G_index_, sizeof(int));
  cudaMemcpy(d_G_index_, &h_G_index, sizeof(int), cudaMemcpyHostToDevice);

  cudaMemset(d_results_, 0, num_edges_ * sizeof(int));

  // 使用cudaEvent_t进行GPU硬件级计时(最科学严谨)
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  sss_kernel<<<grid_size, block_size>>>(
      num_edges_, d_vertexs_, graph.get_device_elements(),
      graph.get_device_offsets(), get_d_hash_length(), get_d_hash_normal(),
      get_d_hash_tables_offset(), d_results_, d_G_index_, CHUNK_SIZE, false,
      get_max_length(), get_bucket_num(), threshold, bucket_size);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float kernel_time_ms = 0.0f;
  cudaEventElapsedTime(&kernel_time_ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return {get_result_count(), kernel_time_ms};
}

// 统计超过阈值的结果数量
int SSSBaseline::get_result_count() {
  std::vector<int> results(num_edges_);
  cudaMemcpy(results.data(), d_results_, num_edges_ * sizeof(int),
             cudaMemcpyDeviceToHost);

  int sum = 0;
  for (int i = 0; i < num_edges_; i++) {
    sum += results[i];
  }
  return sum;
}
