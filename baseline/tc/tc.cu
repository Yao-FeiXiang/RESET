#include <cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "tc.cuh"

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

__global__ void tc_kernel(int num_edges, int* vertexs, int* csr_cols,
                          int* csr_offsets, int* hash_length, int* hash_table,
                          long long* hash_table_offsets,
                          unsigned long long* results, int* G_index,
                          int CHUNK_SIZE, bool opt, int max_length,
                          int bucket_num, int bucket_size) {
  // 共享内存存储块内三角形计数
  __shared__ unsigned long long block_count;
  if (threadIdx.x == 0) block_count = 0;
  __syncthreads();

  unsigned long long thread_count = 0;
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int num_warps = blockDim.x / warpSize;
  int edge = (blockIdx.x * num_warps + warp_id) * CHUNK_SIZE;
  int edge_end = edge + CHUNK_SIZE;
  while (edge < num_edges) {
    // 获取当前边(u, v)
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
    // 遍历较小集合u中的每个邻居,检查是否在v的哈希表中
    // 相交的公共邻居就是第三个顶点,构成三角形(u, v, w)
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
        // 如果找到,计数加一(构成一个三角形)
        if (found) thread_count++;
      }
    }
    __syncwarp();
    // 处理下一条边,使用原子操作动态分配工作
    edge++;
    if (edge == edge_end) {
      if (lane_id == 0) {
        edge = atomicAdd(G_index, CHUNK_SIZE);
      }
      edge = __shfl_sync(0xffffffff, edge, 0);
      edge_end = edge + CHUNK_SIZE;
    }
  }
  // 将线程计数累加到块计数
  atomicAdd(&block_count, thread_count);
  __syncthreads();

  // 将块计数累加到全局结果
  if (threadIdx.x == 0) {
    atomicAdd(results, block_count);
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

TCBaseline::~TCBaseline() {
  if (d_vertexs_) cudaFree(d_vertexs_);
  if (d_csr_cols_sorted_) cudaFree(d_csr_cols_sorted_);
  if (d_total_count_) cudaFree(d_total_count_);
  if (d_G_index_) cudaFree(d_G_index_);
}

void TCBaseline::prepare_vertex_list(const CSRGraph& graph) {
  num_edges_ = graph.get_num_elements();

  // 从CSR创建顶点列表,每个元素存储边的源顶点
  std::vector<int> vertexs(num_edges_);
  for (int i = 0; i < graph.get_num_nodes(); i++) {
    int start = graph.get_host_offsets()[i];
    int end = graph.get_host_offsets()[i + 1];
    for (int j = start; j < end; j++) {
      vertexs[j] = i;
    }
  }

  // 复制到设备端
  cudaMalloc(&d_vertexs_, sizeof(int) * num_edges_);
  cudaMemcpy(d_vertexs_, vertexs.data(), sizeof(int) * num_edges_,
             cudaMemcpyHostToDevice);
}

void TCBaseline::allocate_buffers() {
  // 分配结果缓冲区
  cudaMalloc(&d_total_count_, sizeof(unsigned long long));
  cudaMemset(d_total_count_, 0, sizeof(unsigned long long));
}

unsigned long long TCBaseline::run_hierarchical(CSRGraph& graph, int CHUNK_SIZE,
                                                int grid_size, int block_size,
                                                int bucket_size, bool sorted) {
  int h_G_index = grid_size * block_size / 32 * CHUNK_SIZE;
  cudaMalloc(&d_G_index_, sizeof(int));
  cudaMemcpy(d_G_index_, &h_G_index, sizeof(int), cudaMemcpyHostToDevice);

  // 如果需要排序,准备排序后的CSR
  int* d_csr_cols_ptr;
  if (sorted) {
    std::vector<int> sorted_cols = graph.get_host_elements();
    double start_sort = clock();
    gpu_sort_csr_cols(sorted_cols, graph.get_host_offsets(),
                      graph.get_num_nodes(), get_max_length());
    double end_sort = clock();
    double sort_time = (end_sort - start_sort) / CLOCKS_PER_SEC;
    printf("排序时间: %.6f 秒\n", sort_time);

    cudaMalloc(&d_csr_cols_sorted_, sorted_cols.size() * sizeof(int));
    cudaMemcpy(d_csr_cols_sorted_, sorted_cols.data(),
               sorted_cols.size() * sizeof(int), cudaMemcpyHostToDevice);
    d_csr_cols_ptr = d_csr_cols_sorted_;
  } else {
    d_csr_cols_ptr = graph.get_device_elements();
  }

  cudaMemset(d_total_count_, 0, sizeof(unsigned long long));

  tc_kernel<<<grid_size, block_size>>>(
      num_edges_, d_vertexs_, d_csr_cols_ptr, graph.get_device_offsets(),
      get_d_hash_length(), get_d_hash_hierarchical(),
      get_d_hash_tables_offset(), d_total_count_, d_G_index_, CHUNK_SIZE, true,
      get_max_length(), get_bucket_num(), bucket_size);
  cudaDeviceSynchronize();

  return get_total_count();
}

unsigned long long TCBaseline::run_normal(CSRGraph& graph, int CHUNK_SIZE,
                                          int grid_size, int block_size,
                                          int bucket_size, bool sorted) {
  int h_G_index = grid_size * block_size / 32 * CHUNK_SIZE;
  cudaMalloc(&d_G_index_, sizeof(int));
  cudaMemcpy(d_G_index_, &h_G_index, sizeof(int), cudaMemcpyHostToDevice);

  cudaMemset(d_total_count_, 0, sizeof(unsigned long long));

  tc_kernel<<<grid_size, block_size>>>(
      num_edges_, d_vertexs_, graph.get_device_elements(),
      graph.get_device_offsets(), get_d_hash_length(), get_d_hash_normal(),
      get_d_hash_tables_offset(), d_total_count_, d_G_index_, CHUNK_SIZE, false,
      get_max_length(), get_bucket_num(), bucket_size);
  cudaDeviceSynchronize();

  return get_total_count();
}

// 获取总三角形数量
unsigned long long TCBaseline::get_total_count() {
  unsigned long long total_count;
  cudaMemcpy(&total_count, d_total_count_, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  return total_count;
}
