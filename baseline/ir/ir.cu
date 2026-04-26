#include <cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include "ir.cuh"

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

__global__ void ir_kernel(int* inverted_index, int* inverted_index_offsets,
                          int* query, int* query_offsets, int query_num,
                          int* result, long long* result_offsets,
                          int* result_count, int* G_index, int CHUNK_SIZE,
                          int max_length, bool opt, int* hashtable,
                          long long* hashtable_offset, int bucket_num,
                          int bucket_size) {
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int num_warps = blockDim.x / warpSize;
  int vertex = (blockIdx.x * num_warps + warp_id) * CHUNK_SIZE;
  int vertex_end = vertex + CHUNK_SIZE;
  int query_start, query_end;
  int query_len;
  int* set_start;
  int set_len;
  int* result_start;
  int* hashtable_start;
  int hash_length;

  while (vertex < query_num) {
    // 获取当前查询的范围
    query_start = query_offsets[vertex];
    query_end = query_offsets[vertex + 1];
    query_len = query_end - query_start;

    // 以第一个词项的倒排表作为初始结果集合
    int term0 = query[query_start];
    set_start = inverted_index + inverted_index_offsets[term0];
    set_len = inverted_index_offsets[term0 + 1] - inverted_index_offsets[term0];

    // 将初始集合复制到结果缓冲区
    result_start = result + result_offsets[vertex];
    for (int i = lane_id; i < set_len; i += warpSize) {
      result_start[i] = set_start[i];
    }
    __syncwarp();
    int result_num = set_len;
    int* set = result_start;
    int set_size = result_num;
    // 对查询中剩余的每个词项,依次求交集
    for (int i = 1; i < query_len; i++) {
      int term = query[query_start + i];
      hashtable_start = hashtable + hashtable_offset[term];
      hash_length = hashtable_offset[term + 1] - hashtable_offset[term];

      result_num = 0;
      int num_iters = (set_size + warpSize - 1) / warpSize;
      for (int iter = 0; iter < num_iters; iter++) {
        int j = lane_id + iter * warpSize;
        bool active = (j < set_size);
        bool found = false;
        int key = -1;
        if (active) {
          key = set[j];
          // 根据优化选项选择分层哈希或普通哈希计算桶位置
          int bucket = (opt) ? d_hash_hierarchical(key, hash_length, max_length)
                             : d_hash_normal(key, hash_length);
          found = search_in_hashtable(key, hashtable_start, bucket_num, bucket,
                                      hash_length, bucket_size);
        }

        // 使用warp投票收集找到的元素
        unsigned int found_mask = __ballot_sync(0xffffffff, found);
        int step_found = __popc(found_mask);
        // 计算每个找到元素在结果中的写入位置(前缀和)
        int write_pos = result_num + __popc(found_mask & ((1 << lane_id) - 1));
        if (found && active) {
          result_start[write_pos] = key;
        }
        result_num += step_found;
      }
      // 更新当前交集集合
      set = result_start;
      set_size = __shfl_sync(0xffffffff, result_num, 0);
    }
    // 保存当前查询的结果大小
    if (lane_id == 0) {
      result_count[vertex] = set_size;
    }
    __syncwarp();

    // 处理下一个查询,使用原子操作动态分配工作
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

// GPU排序CSR列,按行和桶位置重排元素
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

IRBaseline::~IRBaseline() {
  // d_inverted_index_ 和 d_inverted_index_offsets_
  // 由InvertedIndex管理,不需要在这里释放
  if (d_query_) cudaFree(d_query_);
  if (d_query_offsets_) cudaFree(d_query_offsets_);
  if (d_result_) cudaFree(d_result_);
  if (d_result_offsets_) cudaFree(d_result_offsets_);
  if (d_result_count_) cudaFree(d_result_count_);
  if (d_G_index_) cudaFree(d_G_index_);
}

void IRBaseline::load_queries(const std::string& query_path,
                              const std::string& query_offsets_path,
                              const std::string& query_num_path) {
  read_i32_vec(query_path, global_query_);
  read_i32_vec(query_offsets_path, query_offsets_);

  // 从文件读取查询数量
  std::ifstream in(query_num_path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file: " + query_num_path);
  }
  in.read(reinterpret_cast<char*>(&query_num_), sizeof(int));

  // 分配设备内存并复制数据
  cudaMalloc(&d_query_, global_query_.size() * sizeof(int));
  cudaMalloc(&d_query_offsets_, (query_num_ + 1) * sizeof(int));
  cudaMemcpy(d_query_, global_query_.data(), global_query_.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_query_offsets_, query_offsets_.data(),
             (query_num_ + 1) * sizeof(int), cudaMemcpyHostToDevice);
}

void IRBaseline::allocate_result_buffers(const InvertedIndex& index) {
  // 保存倒排索引设备指针
  d_inverted_index_ = index.get_device_elements();
  d_inverted_index_offsets_ = index.get_device_offsets();

  // 计算每个查询的结果偏移量,预留存储空间
  std::vector<long long> result_offsets(query_num_ + 1, 0);
  for (int i = 0; i < query_num_; i++) {
    int term = global_query_[query_offsets_[i]];
    result_offsets[i + 1] = result_offsets[i] +
                            index.get_host_offsets()[term + 1] -
                            index.get_host_offsets()[term];
  }
  printf("result offset end: %lld\n", result_offsets[query_num_]);

  cudaMalloc(&d_result_offsets_, (query_num_ + 1) * sizeof(long long));
  cudaMemcpy(d_result_offsets_, result_offsets.data(),
             (query_num_ + 1) * sizeof(long long), cudaMemcpyHostToDevice);
  cudaMalloc(&d_result_, result_offsets.back() * sizeof(int));
  cudaMemset(d_result_, -1, result_offsets.back() * sizeof(int));
  cudaMalloc(&d_result_count_, query_num_ * sizeof(int));
  cudaMemset(d_result_count_, 0, query_num_ * sizeof(int));
}

int IRBaseline::run_hierarchical(int CHUNK_SIZE, int grid_size, int block_size,
                                 int bucket_size, bool sorted) {
  int h_G_index = grid_size * block_size / warpSize * CHUNK_SIZE;
  cudaMalloc(&d_G_index_, sizeof(int));
  cudaMemcpy(d_G_index_, &h_G_index, sizeof(int), cudaMemcpyHostToDevice);

  cudaMemset(d_result_count_, 0, query_num_ * sizeof(int));

  ir_kernel<<<grid_size, block_size>>>(
      d_inverted_index_, d_inverted_index_offsets_, d_query_, d_query_offsets_,
      query_num_, d_result_, d_result_offsets_, d_result_count_, d_G_index_,
      CHUNK_SIZE, get_max_length(), true, get_d_hash_hierarchical(),
      get_d_hash_tables_offset(), get_bucket_num(), bucket_size);
  cudaDeviceSynchronize();

  std::vector<int> h_counts(query_num_);
  cudaMemcpy(h_counts.data(), d_result_count_, sizeof(int) * query_num_,
             cudaMemcpyDeviceToHost);

  // 累加所有查询的结果总数
  int sum = 0;
  for (int x : h_counts) sum += x;
  return sum;
}

int IRBaseline::run_normal(int CHUNK_SIZE, int grid_size, int block_size,
                           int bucket_size, bool sorted) {
  int h_G_index = grid_size * block_size / warpSize * CHUNK_SIZE;
  cudaMalloc(&d_G_index_, sizeof(int));
  cudaMemcpy(d_G_index_, &h_G_index, sizeof(int), cudaMemcpyHostToDevice);

  cudaMemset(d_result_count_, 0, query_num_ * sizeof(int));

  ir_kernel<<<grid_size, block_size>>>(
      d_inverted_index_, d_inverted_index_offsets_, d_query_, d_query_offsets_,
      query_num_, d_result_, d_result_offsets_, d_result_count_, d_G_index_,
      CHUNK_SIZE, get_max_length(), false, get_d_hash_normal(),
      get_d_hash_tables_offset(), get_bucket_num(), bucket_size);
  cudaDeviceSynchronize();

  std::vector<int> h_counts(query_num_);
  cudaMemcpy(h_counts.data(), d_result_count_, sizeof(int) * query_num_,
             cudaMemcpyDeviceToHost);

  // 累加所有查询的结果总数
  int sum = 0;
  for (int x : h_counts) sum += x;
  return sum;
}

std::vector<int> IRBaseline::get_results() {
  std::vector<int> result_counts(query_num_);
  cudaMemcpy(result_counts.data(), d_result_count_, sizeof(int) * query_num_,
             cudaMemcpyDeviceToHost);
  return result_counts;
}
