#ifndef HT_HEADER
#define HT_HEADER

#include <iostream>
#include <string>
#include <cub/cub.cuh>
#include <math.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "common_hie.cuh"

using namespace std;

__global__ void buildHashTableLensKernel(int *hash_tables_lens, int *degree_offset, int vertex_count, float load_factor, int bucket_size);

__global__ void buildHashTableKernel(long long *hash_tables_offset, int *hash_tables, int *adjcant, int *vertex, int edge_count, long long bucket_num, int bucket_size);

tuple<long long *, int *,  long long, int,double> buildHashTable(int *d_adjcant, int *d_vertex, int *d_degree_offset,int* d_dispersed_mapping);

__inline__ __device__ void swap(int &a, int &b)
{
    int t = a;
    a = b;
    b = t;
}

__inline__ __device__ bool search_in_hashtable(int x, int bucket_size, long long bucket_num, int hash_table_len, int *hash_table,  int max_length,int* dispersed_mapping)
{
    x= dispersed_mapping[x];
    int mapped_value =  d_hash_hierarchical(x, hash_table_len, max_length) ;
    int *cmp = hash_table + mapped_value;
    int index = 0;
    while (*cmp != -1)
    {
        if (*cmp == x)
        {
            return true;
        }
        cmp = cmp + bucket_num;
        index++;
        if (index == bucket_size)
        {
            mapped_value++;
            index = 0;
            if (mapped_value == hash_table_len)
                mapped_value = 0;
            cmp = &hash_table[mapped_value];
        }
    }
    return false;
}

extern long long *d_hash_tables_offset;
extern int *d_hash_tables_lens;

#endif
