#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "TC.cuh"

vector<int> csr_row;
vector<int> csr_cols;
int num_edges;
int num_nodes;

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

vector<int> read_i32_vec(const string& filename) {
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

__global__ void dynamic_triangle_count(
    int num_edges, int* vertexs, int* csr_cols, int* csr_offsets,
    int* hash_length, int* hash_table, long long* hash_table_offsets,
    unsigned long long* results, int* G_index, int CHUNK_SIZE, bool opt,
    int max_length, int bucket_num, int bucket_size) {
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
        if (found) thread_count++;
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
  atomicAdd(&block_count, thread_count);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(results, block_count);
  }
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <input_folder>" << endl;
    return 1;
  }

  // size_t free_mem = 0, total_mem = 0;
  // cudaMemGetInfo(&free_mem, &total_mem);
  // printf("GPU mem free=%zu MB total=%zu MB\n", free_mem / 1024 / 1024,
  // total_mem / 1024 / 1024);
  int dev = 0;
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
  printf("load_factor:%.2f , bucket_size:%d\n", load_factor, bucket_size);

  string input_folder = argv[1];
  string csr_cols_path = input_folder + "/csr_cols_tri.bin";
  string csr_offsets_path = input_folder + "/csr_offsets_tri.bin";
  string num_edges_path = input_folder + "/num_edges_tri.bin";
  string nums_nodes_path = input_folder + "/num_nodes_tri.bin";

  csr_cols = read_i32_vec(csr_cols_path);
  csr_row = read_i32_vec(csr_offsets_path);
  num_edges = read_int_binary(num_edges_path);
  num_nodes = read_int_binary(nums_nodes_path);

  printf("Graph loaded: %d nodes, %d edges\n", num_nodes, num_edges);

  make_hash_table(load_factor, bucket_size);

  double start_sort = clock();
  gpu_sort_csr_cols(csr_cols, csr_row, num_nodes, max_length);
  double end_sort = clock();
  double sort_time = (end_sort - start_sort) / CLOCKS_PER_SEC;
  printf("sort time: %.6f sec\n", sort_time);

  printf("finish hierarchical sort\n");

  int* d_csr_cols_hierarchical;
  unsigned long long* d_total_count;

  cudaMalloc(&d_csr_cols_hierarchical, num_edges * sizeof(int));
  cudaMalloc(&d_total_count, sizeof(unsigned long long));

  cudaMemcpy(d_csr_cols_hierarchical, csr_cols.data(), num_edges * sizeof(int),
             cudaMemcpyHostToDevice);

  const int block_size = 512;
  const int grid_size = 1024;
  const int CHUNK_SIZE = 4;
  bool use_static = false;

  int* d_vertex_index;
  cudaMalloc(&d_vertex_index, sizeof(int));
  int h_vertex_index = grid_size * block_size / 32 * CHUNK_SIZE;

  vector<int> vertexs(num_edges);
  for (int i = 0; i < num_nodes; i++) {
    for (int j = csr_row[i]; j < csr_row[i + 1]; j++) {
      vertexs[j] = i;
    }
  }
  int* d_vertexs;
  cudaMalloc(&d_vertexs, sizeof(int) * num_edges);
  cudaMemcpy(d_vertexs, vertexs.data(), sizeof(int) * num_edges,
             cudaMemcpyHostToDevice);

  // printf("TC max_length: %d\n", max_length);

  // normal

  l2flush();
  cudaDeviceSynchronize();

  cudaMemset(d_total_count, 0, sizeof(unsigned long long));
  cudaMemcpy(d_vertex_index, &h_vertex_index, sizeof(int),
             cudaMemcpyHostToDevice);

  double time_start = clock();
  dynamic_triangle_count<<<grid_size, block_size>>>(
      num_edges, d_vertexs, d_csr_cols, d_csr_row, d_hash_length,
      d_hash_table_normal, d_hash_tables_offset, d_total_count, d_vertex_index,
      CHUNK_SIZE, use_static, max_length, bucket_num, bucket_size);
  cudaDeviceSynchronize();
  double time_end = clock();
  double cmp_time = (time_end - time_start) / CLOCKS_PER_SEC;
  unsigned long long total_count;
  cudaMemcpy(&total_count, d_total_count, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("Normal hash - Triangles: %llu, Time: %.6f sec\n", total_count,
         cmp_time);

  l2flush();
  cudaDeviceSynchronize();

  cudaMemset(d_total_count, 0, sizeof(unsigned long long));
  cudaMemcpy(d_vertex_index, &h_vertex_index, sizeof(int),
             cudaMemcpyHostToDevice);
  time_start = clock();
  dynamic_triangle_count<<<grid_size, block_size>>>(
      num_edges, d_vertexs, d_csr_cols_hierarchical, d_csr_row, d_hash_length,
      d_hash_table_hierarchical, d_hash_tables_offset, d_total_count,
      d_vertex_index, CHUNK_SIZE, true, max_length, bucket_num, bucket_size);

  cudaDeviceSynchronize();
  time_end = clock();
  cmp_time = (time_end - time_start) / CLOCKS_PER_SEC;

  cudaMemcpy(&total_count, d_total_count, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("Hierarchical hash - Triangles: %llu, Time: %.6f sec\n", total_count,
         cmp_time);

  cudaFree(d_csr_row);
  cudaFree(d_csr_cols_hierarchical);
  cudaFree(d_csr_cols);
  cudaFree(d_total_count);
  cudaFree(d_hash_length);
  cudaFree(d_hash_tables_offset);
  cudaFree(d_hash_table_hierarchical);
  cudaFree(d_hash_table_normal);
  cudaFree(d_vertex_index);
  free(hash_table_hierarchical);
  free(hash_table_normal);

  return 0;
}