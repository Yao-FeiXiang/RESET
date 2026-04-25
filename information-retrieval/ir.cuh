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


extern int *d_hash_table_hierarchical;
extern int *d_hash_table_normal;
extern int *d_hash_length, *d_inverted_index_offsets;
extern long long *d_hash_tables_offset;
extern int max_length;
extern long long bucket_num;

extern vector<int> inverted_index_offsets;
extern vector<int> inverted_index;
extern int inverted_index_num;
extern vector<int> global_query;
extern vector<int> query_offsets;
extern int query_num;

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