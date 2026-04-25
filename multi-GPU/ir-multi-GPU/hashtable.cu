#include "ir.cuh"
#include <queue>
#include <time.h>

__device__ int
calculate_length(int degrees,float load_factor,int bucket_size)
{
    int total_size = degrees / load_factor;
    int length = total_size / bucket_size;
    if (length == 0)
        return 8;
    length |= length >> 1;
    length |= length >> 2;
    length |= length >> 4;
    length |= length >> 8;
    length |= length >> 16;
    length++;
    if (length < 8)
        length = 8;
    return length;
}



__global__ void calculate_hash_length(int *hash_length, int inverted_index_num, int *inverted_index_offsets, int *max_length, float load_factor, int bucket_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < inverted_index_num; i += stride)
    {
        int start = inverted_index_offsets[i];
        int end = inverted_index_offsets[i + 1];
        int degrees = end - start;
        hash_length[i] = calculate_length(degrees, load_factor, bucket_size);
        atomicMax(max_length, hash_length[i]);
    }
}

__global__ void build_hash_table_hierarchical(int *hash_table, int *hash_length, long long *hash_tables_offset, int inverted_index_size, int bucket_num, int max_length, int *word_id, int *doc_id, int *conflict_count,int bucket_size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < inverted_index_size; i += stride)
    {
        int u = word_id[i];
        int v = doc_id[i];
        int length = hash_length[u];
        int *hash_start = hash_table + hash_tables_offset[u];
        int bucket = d_hash_hierarchical(v, length, max_length);
        int k = 0;
        while (atomicCAS(&hash_start[bucket + k * bucket_num], -1, v)!=-1)
        {
            k++;
            atomicAdd(conflict_count, 1);
            if (k == bucket_size)
            {
                k = 0;
                bucket = (bucket + 1) % length;
            }
        }
    }
}

void make_hash_table(int **d_hash_table_hierarchical, int **d_hash_length, int*d_inverted_index_offsets, long long **d_hash_tables_offset, int& max_length,long long& bucket_num,int inverted_index_num,vector<int>& inverted_index_offsets,vector<int>& inverted_index, float load_factor, int bucket_size)
{
    max_length = 8;
    bucket_num = 0;
    cudaMalloc(d_hash_length, sizeof(int) * inverted_index_num);
    int *d_max_length;
    cudaMalloc(&d_max_length, sizeof(int));
    cudaMemset(d_max_length, 0, sizeof(int));
    calculate_hash_length<<<256, 1024>>>(*d_hash_length, inverted_index_num, d_inverted_index_offsets, d_max_length, load_factor, bucket_size);
    cudaDeviceSynchronize();
    cudaMemcpy(&max_length, d_max_length, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_max_length);
    //printf("Max length: %d\n", max_length);


    cudaMalloc(d_hash_tables_offset, sizeof(long long) * (inverted_index_num + 1));
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, *d_hash_length, *d_hash_tables_offset, inverted_index_num);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, *d_hash_length, *d_hash_tables_offset, inverted_index_num);
    int last_hash_length;
    cudaMemcpy(&last_hash_length, *d_hash_length + inverted_index_num - 1, sizeof(int), cudaMemcpyDeviceToHost);
    long long last_sum;
    cudaMemcpy(&last_sum, *d_hash_tables_offset + inverted_index_num - 1, sizeof(long long), cudaMemcpyDeviceToHost);
    bucket_num = last_sum + last_hash_length;
    //printf("Total buckets: %lld\n", bucket_num);
    cudaMemcpy(*d_hash_tables_offset + inverted_index_num, &bucket_num, sizeof(long long), cudaMemcpyHostToDevice);
    cudaFree(d_temp_storage);
    cudaMalloc(d_hash_table_hierarchical, sizeof(int) * bucket_num * bucket_size);
    cudaMemset(*d_hash_table_hierarchical, -1, sizeof(int) * bucket_num * bucket_size);

    vector<int> word_id;
    vector<int> doc_id;
    //clock_t start_time = clock();
    for (int i = 0; i < inverted_index_num;i++)
    {
        int start = inverted_index_offsets[i];
        int end = inverted_index_offsets[i + 1];
        for (int j = start; j < end; j++)
        {
            word_id.push_back(i);
            doc_id.push_back(inverted_index[j]);
        }
    }

    //clock_t end_time = clock();
    //printf("Time to prepare word_id and doc_id: %f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
    int* d_word_id, *d_doc_id;
    cudaMalloc(&d_word_id, sizeof(int) * word_id.size());
    cudaMalloc(&d_doc_id, sizeof(int) * doc_id.size());
    cudaMemcpy(d_word_id, word_id.data(), sizeof(int) * word_id.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_doc_id, doc_id.data(), sizeof(int) * doc_id.size(), cudaMemcpyHostToDevice);

    int* d_conflict_count;
    cudaMalloc(&d_conflict_count, sizeof(int));
    cudaMemset(d_conflict_count, 0, sizeof(int));
    //printf("start building hash tables...\n");

    cudaMemset(d_conflict_count, 0, sizeof(int));
    double start_time = clock();
    build_hash_table_hierarchical<<<1024, 1024>>>(*d_hash_table_hierarchical, *d_hash_length, *d_hash_tables_offset, inverted_index.size(), bucket_num, max_length, d_word_id, d_doc_id,d_conflict_count, bucket_size);
    cudaDeviceSynchronize();
    double end_time = clock();
    //printf("Hierarchical hash table built in %f seconds.\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
    int conflict_count = 0;
    cudaMemcpy(&conflict_count, d_conflict_count, sizeof(int), cudaMemcpyDeviceToHost);
    //printf("Hierarchical hash table built with %d conflicts.\n", conflict_count);

    // vector<int> hash_table(20);
    // cudaMemcpy(hash_table.data(), *d_hash_table_hierarchical, 20 * sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i=0;i<20;i++)
    // {
    //     printf("%d ", hash_table[i]);
    // }
    // printf("\n");
}