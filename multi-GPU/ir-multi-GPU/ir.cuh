#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <tuple>
#include <algorithm>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
using namespace cooperative_groups;
using namespace std;

#define warpSize 32


void make_hash_table(int **d_hash_table_hierarchical, int **d_hash_length, int *d_inverted_index_offsets, long long **d_hash_tables_offset, int &max_length, long long &bucket_num, int inverted_index_num, vector<int> &inverted_index_offsets, vector<int> &inverted_index, float load_factor, int bucket_size);
int partial_ir(vector<int> &inverted_index_offsets, vector<int> &inverted_index, vector<int> &global_query, vector<int> &query_offsets, int query_num, double &kernel_time, int inverted_index_num, float load_factor, int bucket_size);

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