#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace cooperative_groups;
using namespace std;

#define warpSize 32
#define threshold 0.25

void make_hash_table(int num_nodes, int** d_hash_length, int** d_csr_offsets,
                     int** d_hash_table_hierarchical,
                     long long** d_hash_tables_offset, int** d_csr_cols,
                     vector<int>& csr_cols, vector<int>& csr_offsets,
                     int& max_length, long long& bucket_num, float load_factor,
                     int bucket_size, bool hierarchical);

__device__ __forceinline__ int d_hash_hierarchical(const int x, int length,
                                                   int max_length) {
  int shift = __ffs(max_length) - __ffs(length);
  return (x & (max_length - 1)) >> shift;
  // return x % max_length / (max_length / length);
}

__device__ __forceinline__ int d_hash_normal(const int x, int length) {
  return x & (length - 1);
  // return x % length;
}

vector<int> partial_sss(vector<int>& vertexs, vector<int>& adjcents,
                        vector<int>& csr_cols, vector<int>& csr_offsets,
                        int num_nodes, int num_edges, double& kernel_time,
                        float load_factor, int bucket_size, int dev = -1,
                        bool hierarchical = true);
