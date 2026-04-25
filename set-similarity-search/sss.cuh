#pragma once
#include <cooperative_groups.h>
#include <cuda_runtime.h>

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

// #define load_factor 0.25
// #define bucket_size 4

extern vector<int> vertexs;
extern vector<int> csr_cols;
extern vector<int> csr_offsets;
extern int num_nodes;
extern int num_edges;

extern int* d_hash_table_hierarchical;
extern int* d_hash_table_normal;
extern int *d_hash_length, *d_csr_offsets;
extern long long* d_hash_tables_offset;
extern int max_length;
extern long long bucket_num;

extern int* hash_table_normal;
extern int* hash_table_hierarchical;
extern vector<int> hash_length;
extern vector<int> hash_start;

extern int* d_csr_cols;
extern int* d_vertexs;

void make_hash_table(float load_factor, int bucket_size);

__device__ __forceinline__ int d_hash_hierarchical(const int x, int length, int max_length)
{

  int shift = __ffs(max_length) - __ffs(length);
  return (x & (max_length - 1)) >> shift;
  // return x % max_length / (max_length / length);
}

__device__ __forceinline__ int d_hash_normal(const int x, int length)
{
  return x & (length - 1);
  // return x % length;
}
