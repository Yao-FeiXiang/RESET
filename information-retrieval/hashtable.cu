#include "ir.cuh"
#include <queue>
#include <time.h>

int *d_hash_table_hierarchical;
int *d_hash_table_normal;
int *d_hash_length, *d_inverted_index_offsets;
long long *d_hash_tables_offset;
int max_length;
long long bucket_num = 0;

int *hash_table_normal;
int *hash_table_hierarchical;
vector<int> hash_length;
vector<int> hash_start;

__device__ int
calculate_length(int degrees, float load_factor, int bucket_size)
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
        hash_length[i] = calculate_length(degrees,load_factor,bucket_size);
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

__global__ void build_hash_table_normal(int *hash_table, int *hash_length, long long *hash_tables_offset, int inverted_index_size, int bucket_num, int *word_id, int *doc_id,int* conflict_count,int bucket_size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < inverted_index_size; i += stride)
    {
        int u = word_id[i];
        int v = doc_id[i];
        int length = hash_length[u];
        int *hash_start = hash_table + hash_tables_offset[u];
        int bucket = d_hash_normal(v, length);
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

// void make_hashtable_normal()
// {
//     hash_table_normal = new int[bucket_num * bucket_size];
//     memset(hash_table_normal, -1, sizeof(int) * bucket_num * bucket_size);
//     vector<int> hash_length(inverted_index_num);
//     vector<int> hash_start(inverted_index_num + 1);
//     cudaMemcpy(hash_length.data(), d_hash_length, sizeof(int) * inverted_index_num, cudaMemcpyDeviceToHost);
//     cudaMemcpy(hash_start.data(), d_hash_tables_offset, sizeof(int) * (inverted_index_num + 1), cudaMemcpyDeviceToHost);
//     int conflict_count = 0;
//     for (int i = 0; i < inverted_index_num; i++)
//     {
//         int *record = new int[hash_length[i]];
//         memset(record, 0, hash_length[i] * sizeof(int));
//         int *hashtable_start = hash_table_normal + hash_start[i];
//         for (int j = inverted_index_offsets[i]; j < inverted_index_offsets[i + 1]; j++)
//         {
//             int bucket = h_hash_normal(inverted_index[j], hash_length[i]);
//             while (record[bucket] >= bucket_size)
//             {
//                 bucket = (bucket + 1) % hash_length[i];
//                 conflict_count += bucket_size;
//             }
//             conflict_count += record[bucket];
//             int pos = hash_length[i] * record[bucket] + bucket;
//             hashtable_start[pos] = inverted_index[j];
//             record[bucket]++;
//         }
//         delete[] record;
//     }
//     printf("Normal hash table built with %d conflicts.\n", conflict_count);
// }

// void make_hashtable_hierarchical()
// {
//     hash_table_hierarchical = new int[bucket_num * bucket_size];
//     memset(hash_table_hierarchical, -1, sizeof(int) * bucket_num * bucket_size);
//     int conflict_count = 0;
//     for (int i = 0; i < inverted_index_num; i++)
//     {
//         int *record = new int[hash_length[i]];
//         memset(record, 0, hash_length[i] * sizeof(int));
//         int *hashtable_start = hash_table_hierarchical + hash_start[i];
//         for (int j = inverted_index_offsets[i]; j < inverted_index_offsets[i + 1]; j++)
//         {
//             int bucket = h_hash_hierarchical(inverted_index[j], hash_length[i],max_length);
//             while (record[bucket] >= bucket_size)
//             {
//                 bucket = (bucket + 1) % hash_length[i];
//                 conflict_count += bucket_size;
//             }
//             conflict_count += record[bucket];
//             int pos = hash_length[i] * record[bucket] + bucket;
//             hashtable_start[pos] = inverted_index[j];
//             record[bucket]++;
//         }
//         delete[] record;
//     }
//     printf("Hierarchical hash table built with %d conflicts.\n", conflict_count);
// }

void make_hash_table(float load_factor, int bucket_size)
{
    cudaMalloc(&d_hash_length, sizeof(int) * inverted_index_num);
    cudaMalloc(&d_inverted_index_offsets, sizeof(int) * (inverted_index_num + 1));
    cudaMemcpy(d_inverted_index_offsets, inverted_index_offsets.data(), sizeof(int) * (inverted_index_num + 1), cudaMemcpyHostToDevice);
    int *d_max_length;
    cudaMalloc(&d_max_length, sizeof(int));

    calculate_hash_length<<<256, 1024>>>(d_hash_length, inverted_index_num, d_inverted_index_offsets, d_max_length,load_factor,bucket_size);
    cudaDeviceSynchronize();
    cudaMemcpy(&max_length, d_max_length, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_max_length);
    printf("Max length: %d\n", max_length);
    printf("average degree: %f\n", (float)inverted_index.size() / inverted_index_num);

    // vector<int> hash_length(inverted_index_num);
    // cudaMemcpy(hash_length.data(), d_hash_length, sizeof(int) * inverted_index_num, cudaMemcpyDeviceToHost);
    // sort(hash_length.begin(), hash_length.end(), greater<int>());
    // for(int i=0;i<10;i++)
    //     printf("Top %d length: %d\n", i + 1, hash_length[i]);

    cudaMalloc(&d_hash_tables_offset, sizeof(long long) * (inverted_index_num + 1));
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_hash_length, d_hash_tables_offset, inverted_index_num);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_hash_length, d_hash_tables_offset, inverted_index_num);
    int last_hash_length;
    cudaMemcpy(&last_hash_length, d_hash_length + inverted_index_num - 1, sizeof(int), cudaMemcpyDeviceToHost);
    long long last_sum;
    cudaMemcpy(&last_sum, d_hash_tables_offset + inverted_index_num - 1, sizeof(long long), cudaMemcpyDeviceToHost);
    printf("last sum: %d, last_hash_length: %d\n", last_sum, last_hash_length);
    bucket_num = last_sum + last_hash_length;
    printf("Total buckets: %lld\n", bucket_num);
    cudaMemcpy(d_hash_tables_offset + inverted_index_num, &bucket_num, sizeof(long long), cudaMemcpyHostToDevice);
    cudaFree(d_temp_storage);
    cudaMalloc(&d_hash_table_hierarchical, sizeof(int) * bucket_num * bucket_size);
    cudaMalloc(&d_hash_table_normal, sizeof(int) * bucket_num * bucket_size);
    cudaMemset(d_hash_table_hierarchical, -1, sizeof(int) * bucket_num * bucket_size);
    cudaMemset(d_hash_table_normal, -1, sizeof(int) * bucket_num * bucket_size);

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
    printf("start building hash tables...\n");
    clock_t start_time = clock();
    build_hash_table_normal<<<1024, 1024>>>(d_hash_table_normal, d_hash_length, d_hash_tables_offset, inverted_index.size(), bucket_num, d_word_id, d_doc_id,d_conflict_count,bucket_size);
    cudaDeviceSynchronize();
    clock_t end_time = clock();
    printf("Normal hash table built in %f seconds.\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
    int conflict_count;
    cudaMemcpy(&conflict_count, d_conflict_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Normal hash table built with %d conflicts.\n", conflict_count);

    cudaMemset(d_conflict_count, 0, sizeof(int));
    start_time = clock();
    build_hash_table_hierarchical<<<1024, 1024>>>(d_hash_table_hierarchical, d_hash_length, d_hash_tables_offset, inverted_index.size(), bucket_num, max_length, d_word_id, d_doc_id,d_conflict_count,bucket_size);
    cudaDeviceSynchronize();
    end_time = clock();
    printf("Hierarchical hash table built in %f seconds.\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
    cudaMemcpy(&conflict_count, d_conflict_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Hierarchical hash table built with %d conflicts.\n", conflict_count);

    cudaFree(d_word_id);
    cudaFree(d_doc_id);
    cudaFree(d_conflict_count);

    // hash_length.resize(inverted_index_num);
    // hash_start.resize(inverted_index_num + 1);
    // cudaMemcpy(hash_length.data(), d_hash_length, sizeof(int) * inverted_index_num, cudaMemcpyDeviceToHost);
    // cudaMemcpy(hash_start.data(), d_hash_tables_offset, sizeof(int) * (inverted_index_num + 1), cudaMemcpyDeviceToHost);

    // clock_t start_time = clock();
    // make_hashtable_normal();
    // clock_t end_time = clock();
    // printf("Normal hash table built in %f seconds.\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

    // start_time = clock();
    // make_hashtable_hierarchical();
    // end_time = clock();
    // printf("Hierarchical hash table built in %f seconds.\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
}