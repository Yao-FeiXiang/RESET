#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "sss.cuh"

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
    int num_edges, int* vertexs, int* adjcents, int* csr_cols, int* csr_offsets,
    int* hash_length, int* hash_table, long long* hash_table_offsets,
    int* G_index, int CHUNK_SIZE, bool opt, int max_length, int bucket_num,
    int* results, int bucket_size) {
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int num_warps = blockDim.x / warpSize;

  int edge_start_idx = (blockIdx.x * num_warps + warp_id) * CHUNK_SIZE;

  while (edge_start_idx < num_edges) {
    int edge_end_idx = edge_start_idx + CHUNK_SIZE;
    if (edge_end_idx > num_edges) edge_end_idx = num_edges;

    int current_edge = edge_start_idx;
    int work_id = lane_id;

    while (current_edge < edge_end_idx) {
      int u = vertexs[current_edge];
      int v = adjcents[current_edge];
      int u_size = csr_offsets[u + 1] - csr_offsets[u];
      int v_size = csr_offsets[v + 1] - csr_offsets[v];

      int work_size = (u_size < v_size) ? u_size : v_size;

      while (current_edge < edge_end_idx && work_id >= work_size) {
        work_id -= work_size;
        current_edge++;
        if (current_edge < edge_end_idx) {
          u = vertexs[current_edge];
          v = adjcents[current_edge];
          u_size = csr_offsets[u + 1] - csr_offsets[u];
          v_size = csr_offsets[v + 1] - csr_offsets[v];
          work_size = (u_size < v_size) ? u_size : v_size;
        }
      }

      if (current_edge < edge_end_idx) {
        // int key_node = u;
        int table_node = v;
        int key_start_offset = csr_offsets[u];

        if (u_size > v_size) {
          //   key_node = v;
          table_node = u;
          key_start_offset = csr_offsets[v];
        }

        int key = csr_cols[key_start_offset + work_id];
        int* table_ptr = &hash_table[hash_table_offsets[table_node]];
        int table_len = hash_length[table_node];
        int bucket = (opt) ? d_hash_hierarchical(key, table_len, max_length)
                           : d_hash_normal(key, table_len);

        if (search_in_hashtable(key, table_ptr, bucket_num, bucket, table_len,
                                bucket_size)) {
          atomicAdd(&results[current_edge], 1);
        }
      }

      __syncwarp();

      int last_edge = __shfl_sync(0xffffffff, current_edge, 31);
      int last_work = __shfl_sync(0xffffffff, work_id, 31);

      current_edge = last_edge;
      work_id = last_work + lane_id + 1;
    }

    if (lane_id == 0) {
      edge_start_idx = atomicAdd(G_index, CHUNK_SIZE);
    }
    edge_start_idx = __shfl_sync(0xffffffff, edge_start_idx, 0);
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

// namespace {
// inline double bytes_to_gb(size_t bytes) {
//   return bytes / (1024.0 * 1024.0 * 1024.0);
// }
// struct MemPeakTracker {
//   size_t peak_used = 0;
//   size_t total_mem = 0;
//   void sample(const char* tag, int dev) {
//     size_t free_mem = 0;
//     size_t total = 0;
//     cudaMemGetInfo(&free_mem, &total);
//     size_t used = total - free_mem;
//     if (used > peak_used) {
//       peak_used = used;
//       total_mem = total;
//       printf("Device %d peak used memory: %zu bytes (%.2f GB) at %s\n", dev,
//              used, bytes_to_gb(used), tag);
//     }
//   }
//   void report(int dev) const {
//     printf("Device %d peak used memory: %zu bytes (%.2f GB), total: %.2f
//     GB\n",
//            dev, peak_used, bytes_to_gb(peak_used), bytes_to_gb(total_mem));
//   }
// };
// }  // namespace
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

vector<int> partial_sss(vector<int>& vertexs, vector<int>& adjcents,
                        vector<int>& csr_cols, vector<int>& csr_offsets,
                        int num_nodes, int num_edges, double& kernel_time,
                        float load_factor, int bucket_size, int dev,
                        bool hierarchical) {
  // MemPeakTracker mem_peak;
  // mem_peak.sample("start", dev);

  int* d_hash_table_hierarchical;
  int *d_hash_length, *d_csr_offsets;
  long long* d_hash_tables_offset;
  int max_length;
  long long bucket_num = 0;

  int* d_csr_cols;
  int* d_vertexs;
  int* d_adjcents;

  cudaMalloc(&d_vertexs, sizeof(int) * num_edges);
  // mem_peak.sample("cudaMalloc d_vertexs", dev);
  cudaMemcpy(d_vertexs, vertexs.data(), sizeof(int) * num_edges,
             cudaMemcpyHostToDevice);
  cudaMalloc(&d_adjcents, sizeof(int) * adjcents.size());
  // mem_peak.sample("cudaMalloc d_adjcents", dev);
  cudaMemcpy(d_adjcents, adjcents.data(), sizeof(int) * adjcents.size(),
             cudaMemcpyHostToDevice);

  make_hash_table(num_nodes, &d_hash_length, &d_csr_offsets,
                  &d_hash_table_hierarchical, &d_hash_tables_offset,
                  &d_csr_cols, csr_cols, csr_offsets, max_length, bucket_num,
                  load_factor, bucket_size, hierarchical);
  // mem_peak.sample("make_hash_table", dev);

  const int grid_size = 512;
  const int block_size = 1024;
  const int CHUNK_SIZE = 4;

  cudaMalloc(&d_vertexs, sizeof(int) * num_edges);
  // mem_peak.sample("cudaMalloc d_vertexs (second)", dev);
  cudaMemcpy(d_vertexs, vertexs.data(), sizeof(int) * num_edges,
             cudaMemcpyHostToDevice);
  cudaMalloc(&d_adjcents, sizeof(int) * adjcents.size());
  // mem_peak.sample("cudaMalloc d_adjcents (second)", dev);
  cudaMemcpy(d_adjcents, adjcents.data(), sizeof(int) * adjcents.size(),
             cudaMemcpyHostToDevice);

  int* d_results;
  cudaMalloc(&d_results, num_edges * sizeof(int));
  // mem_peak.sample("cudaMalloc d_results", dev);
  cudaMemset(d_results, 0, num_edges * sizeof(int));

  int* d_G_index;
  cudaMalloc(&d_G_index, sizeof(int));
  // mem_peak.sample("cudaMalloc d_G_index", dev);
  int h_G_index = grid_size * block_size / warpSize * CHUNK_SIZE;
  cudaMemcpy(d_G_index, &h_G_index, sizeof(int), cudaMemcpyHostToDevice);

  int* d_csr_cols_hierarchical;
  cudaMalloc(&d_csr_cols_hierarchical, csr_cols.size() * sizeof(int));
  // mem_peak.sample("cudaMalloc d_csr_cols_hierarchical", dev);
  if (hierarchical)
    gpu_sort_csr_cols(csr_cols, csr_offsets, num_nodes, max_length);
  // mem_peak.sample("gpu_sort_csr_cols", dev);

  cudaMemcpy(d_csr_cols_hierarchical, csr_cols.data(),
             csr_cols.size() * sizeof(int), cudaMemcpyHostToDevice);
  // mem_peak.sample("copy d_csr_cols_hierarchical", dev);

  l2flush();
  cudaDeviceSynchronize();
  double time_start = clock();
  set_similarity_search_kernel<<<grid_size, block_size>>>(
      num_edges, d_vertexs, d_adjcents, d_csr_cols_hierarchical, d_csr_offsets,
      d_hash_length, d_hash_table_hierarchical, d_hash_tables_offset, d_G_index,
      CHUNK_SIZE, hierarchical, max_length, bucket_num, d_results, bucket_size);
  cudaDeviceSynchronize();
  // mem_peak.sample("after kernel", dev);
  double time_end = clock();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  double cmp_time = (time_end - time_start) / CLOCKS_PER_SEC;
  // cout << "Hierarchical Kernel execution time: " << cmp_time << " seconds" <<
  // endl;
  kernel_time = cmp_time;

  vector<int> results(num_edges);
  cudaMemcpy(results.data(), d_results, num_edges * sizeof(int),
             cudaMemcpyDeviceToHost);

  cudaFree(d_hash_table_hierarchical);
  cudaFree(d_hash_length);
  cudaFree(d_csr_offsets);
  cudaFree(d_hash_tables_offset);
  cudaFree(d_csr_cols);
  cudaFree(d_vertexs);
  cudaFree(d_adjcents);
  cudaFree(d_results);
  cudaFree(d_G_index);
  cudaFree(d_csr_cols_hierarchical);
  cudaDeviceReset();
  // mem_peak.report(dev);

  return results;
}
