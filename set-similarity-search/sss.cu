#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "sss.cuh"
vector<int> vertexs;
vector<int> csr_cols;
vector<int> csr_offsets;
int num_nodes;
int num_edges;

vector<int> read_vector_binary(const string& filename) {
  ifstream in(filename, ios::binary);
  if (!in) {
    cerr << "Error opening file: " << filename << endl;
    exit(1);
  }

  size_t size;
  in.read(reinterpret_cast<char*>(&size), sizeof(size_t));

  vector<int> vec(size);
  in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(int));

  return vec;
}

int read_int_binary(const string& filename) {
  ifstream in(filename, ios::binary);
  if (!in) {
    cerr << "Error opening file: " << filename << endl;
    exit(1);
  }

  int value;
  in.read(reinterpret_cast<char*>(&value), sizeof(int));

  return value;
}

bool cmp_hierarchical(int a, int b) { return a % max_length < b % max_length; }

struct l2flush {
  __forceinline__ l2flush() {
    int dev_id{};
    (cudaGetDevice(&dev_id));
    (cudaDeviceGetAttribute(&m_l2_size, cudaDevAttrL2CacheSize, dev_id));
    if (m_l2_size > 0) {
      void* buffer = m_l2_buffer;
      (cudaMalloc(&buffer, m_l2_size));
      m_l2_buffer = reinterpret_cast<int*>(buffer);
    }
  }

  __forceinline__ ~l2flush() {
    if (m_l2_buffer) {
      (cudaFree(m_l2_buffer));
    }
  }

  __forceinline__ void flush(cudaStream_t stream) {
    if (m_l2_size > 0) {
      (cudaMemsetAsync(m_l2_buffer, 0, m_l2_size, stream));
    }
  }

 private:
  int m_l2_size{};
  int* m_l2_buffer{};
};

__inline__ __device__ bool search_in_hashtable(int key, int* hashtable,
                                               int bucket_num, int bucket,
                                               int hash_length,
                                               int bucket_size) {
  bool found = false;
  int index = 0;
  while (1) {
    if (hashtable[bucket + index * bucket_num] == key) {
      found = true;
      break;
    } else if (hashtable[bucket + index * bucket_num] == -1) {
      break;
    }
    index++;
    if (index == bucket_size) {
      index = 0;
      bucket = (bucket + 1) & (hash_length - 1);
    }
  }
  return found;
}

__global__ void set_similarity_search_kernel(
    int num_edges, int* vertexs, int* csr_cols, int* csr_offsets,
    int* hash_length, int* hash_table, long long* hash_table_offsets,
    int* results, int* G_index, int CHUNK_SIZE, bool opt, int max_length,
    int bucket_num, float threshold, int bucket_size) {
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int num_warps = blockDim.x / warpSize;
  int edge = (blockIdx.x * num_warps + warp_id) * CHUNK_SIZE;
  int edge_end = edge + CHUNK_SIZE;
  while (edge < num_edges) {
    int u = vertexs[edge];
    int v = csr_cols[edge];
    int u_size = csr_offsets[u + 1] - csr_offsets[u];
    int v_size = csr_offsets[v + 1] - csr_offsets[v];
    if (u_size > v_size) {
      int temp = u;
      u = v;
      v = temp;
      int temp_size = u_size;
      u_size = v_size;
      v_size = temp_size;
    }
    int* hash_table_start = &hash_table[hash_table_offsets[v]];
    int length = hash_length[v];
    int u_neightbour_start = csr_offsets[u];
    int result_num = 0;
    int num_iters = (u_size + warpSize - 1) / warpSize;
    for (int iter = 0; iter < num_iters; iter++) {
      int j = lane_id + iter * warpSize;
      bool active = (j < u_size);
      bool found = false;
      // int key = -1;
      if (active) {
        int key = csr_cols[u_neightbour_start + j];
        int bucket = (opt) ? d_hash_hierarchical(key, length, max_length)
                           : d_hash_normal(key, length);
        found = search_in_hashtable(key, hash_table_start, bucket_num, bucket,
                                    length, bucket_size);
      }
      unsigned int found_mask = __ballot_sync(0xffffffff, found);
      int step_found = __popc(found_mask);
      result_num += step_found;
    }
    result_num = __shfl_sync(0xffffffff, result_num, 0);
    if (lane_id == 0) {
      float jaccard = result_num / (float)(u_size + v_size - result_num);
      if (jaccard >= threshold) {
        results[edge] = 1;
      }
    }
    __syncwarp();
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
    return (ca % max_length) < (cb % max_length);
  }
};

void gpu_sort_csr_cols(vector<int>& csr_cols, const vector<int>& csr_row,
                       int num_nodes, int max_length) {
  int total_edges = csr_cols.size();
  vector<int> rows_host(total_edges);
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

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <input_folder>" << endl;
    return 1;
  }
  string input_folder = argv[1];
  float load_factor = 0.2;
  int bucket_size = 5;

  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.rfind("--alpha=", 0) == 0) {
      load_factor = std::stof(arg.substr(8));
    } else if (arg.rfind("--bucket=", 0) == 0) {
      bucket_size = std::stoi(arg.substr(9));
    } else {
      cerr << "Unknown argument: " << arg << endl;
      return 1;
    }
  }

  cudaDeviceSynchronize();

  int dev = 0;  // 选择物理 GPU 编号 0
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  printf("Found %d devices\n", deviceCount);
  if (dev >= deviceCount) {
    fprintf(stderr, "Device %d out of range\n", dev);
    return 1;
  }
  cudaError_t e = cudaSetDevice(dev);
  if (e != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", dev,
            cudaGetErrorString(e));
    return 1;
  }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  printf("Using device %d: %s\n", dev, prop.name);

  printf("load_factor:%.2f , bucket_size:%d\n", load_factor, bucket_size);

  string vertexs_path = input_folder + "/vertexs.bin";
  string csr_cols_path = input_folder + "/csr_cols.bin";
  string csr_offsets_path = input_folder + "/csr_offsets.bin";
  string num_edges_path = input_folder + "/num_edges.bin";
  string nums_nodes_path = input_folder + "/num_nodes.bin";

  vertexs = read_vector_binary(vertexs_path);
  csr_cols = read_vector_binary(csr_cols_path);
  csr_offsets = read_vector_binary(csr_offsets_path);
  num_edges = read_int_binary(num_edges_path);
  num_nodes = read_int_binary(nums_nodes_path);

  printf("num_nodes: %d, num_edges: %d\n", num_nodes, num_edges);

  int max_degree = 0;
  for (int i = 0; i < num_nodes; i++) {
    int degree = csr_offsets[i + 1] - csr_offsets[i];
    if (degree > max_degree) max_degree = degree;
  }
  printf("max_degree: %d\n", max_degree);

  make_hash_table(load_factor, bucket_size);

  const int grid_size = 512;
  const int block_size = 1024;
  const int CHUNK_SIZE = 4;
  const float threshold = 0.25;

  int* d_results;
  cudaMalloc(&d_results, num_edges * sizeof(int));
  cudaMemset(d_results, 0, num_edges * sizeof(int));

  int* d_G_index;
  cudaMalloc(&d_G_index, sizeof(int));
  int h_G_index = grid_size * block_size / warpSize * CHUNK_SIZE;
  cudaMemcpy(d_G_index, &h_G_index, sizeof(int), cudaMemcpyHostToDevice);

  int* d_csr_cols_hierarchical;
  cudaMalloc(&d_csr_cols_hierarchical, csr_cols.size() * sizeof(int));

  gpu_sort_csr_cols(csr_cols, csr_offsets, num_nodes, max_length);

  cudaMemcpy(d_csr_cols_hierarchical, csr_cols.data(),
             csr_cols.size() * sizeof(int), cudaMemcpyHostToDevice);

  printf("Begin Kernel Execution\n");
  l2flush().flush(0);
  double time_start = clock();
  set_similarity_search_kernel<<<grid_size, block_size>>>(
      num_edges, d_vertexs, d_csr_cols, d_csr_offsets, d_hash_length,
      d_hash_table_normal, d_hash_tables_offset, d_results, d_G_index,
      CHUNK_SIZE, false, max_length, bucket_num, threshold, bucket_size);
  cudaDeviceSynchronize();

  double time_end = clock();
  double cmp_time = (time_end - time_start) / CLOCKS_PER_SEC;
  cout << "[Target] Normal Kernel execution time: " << cmp_time << " seconds"
       << endl;
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    cerr << "Kernel failed: " << cudaGetErrorString(err) << endl;
    cerr << "Error code: " << err << endl;
  }
  vector<int> results(num_edges);
  cudaMemcpy(results.data(), d_results, num_edges * sizeof(int),
             cudaMemcpyDeviceToHost);
  int sum1 = 0;
  for (int i = 0; i < num_edges; i++) {
    sum1 += results[i];
  }

  cudaMemset(d_results, 0, num_edges * sizeof(int));
  cudaMemcpy(d_G_index, &h_G_index, sizeof(int), cudaMemcpyHostToDevice);
  l2flush().flush(0);
  time_start = clock();
  set_similarity_search_kernel<<<grid_size, block_size>>>(
      num_edges, d_vertexs, d_csr_cols_hierarchical, d_csr_offsets,
      d_hash_length, d_hash_table_hierarchical, d_hash_tables_offset, d_results,
      d_G_index, CHUNK_SIZE, true, max_length, bucket_num, threshold,
      bucket_size);
  cudaDeviceSynchronize();
  time_end = clock();
  cmp_time = (time_end - time_start) / CLOCKS_PER_SEC;
  cout << "[Target] Hierarchical Kernel execution time: " << cmp_time
       << " seconds" << endl;
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cerr << "Kernel failed: " << cudaGetErrorString(err) << endl;
    cerr << "Error code: " << err << endl;
  }
  cudaMemcpy(results.data(), d_results, num_edges * sizeof(int),
             cudaMemcpyDeviceToHost);
  int sum2 = 0;
  for (int i = 0; i < num_edges; i++) {
    sum2 += results[i];
  }
  if (sum1 != sum2) {
    cout << "Wrong Answer!" << endl;
    printf("sum1: %d, sum2: %d\n", sum1, sum2);
  } else {
    cout << "num of pairs whose similarity is greater than threshold: " << sum1
         << endl;
  }
}
