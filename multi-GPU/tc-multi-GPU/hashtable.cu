#include "tc.cuh"
#include <queue>
#include <time.h>

__device__ int calculate_length(int degrees,  float load_factor, int bucket_size)
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

__global__ void calculate_hash_length(int *hash_length, int num_nodes, int *csr_offsets, int *max_length,  float load_factor, int bucket_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < num_nodes; i += stride)
    {
        int start = csr_offsets[i];
        int end = csr_offsets[i + 1];
        int degrees = end - start;
        hash_length[i] = calculate_length(degrees, load_factor, bucket_size);
        atomicMax(max_length, hash_length[i]);
    }
}

__global__ void build_hash_table_hierarchical(int *hash_table, int *hash_length, long long *hash_tables_offset, int num_edges, int bucket_num, int max_length, int *vertexs, int *csr_cols, int *conflict_count,int bucket_size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < num_edges; i += stride)
    {
        int u = vertexs[i];
        int v = csr_cols[i];
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

void make_hash_table(int num_nodes, int **d_hash_length, int **d_csr_offsets, int **d_hash_table_hierarchical,long long** d_hash_tables_offset,int**d_csr_cols,vector<int>& csr_cols,vector<int>& csr_offsets,int& max_length,long long &bucket_num, float load_factor,int bucket_size)
{
    //printf("-1\n");
    cudaMalloc(d_hash_length, sizeof(int) * num_nodes);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     printf("CUDA error after memory operations: %s\n", cudaGetErrorString(err));
    //     return;
    // }
    // printf("0\n");
    cudaMemset(*d_hash_length, 0, sizeof(int) * num_nodes);
    // err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     printf("CUDA error after memory operations: %s\n", cudaGetErrorString(err));
    //     return;
    // }
    // printf("1\n");
    cudaMalloc(d_csr_offsets, sizeof(int) * (num_nodes + 1));
    // err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     printf("CUDA error after memory operations: %s\n", cudaGetErrorString(err));
    //     return;
    // }
    // printf("2\n");
    cudaMemcpy(*d_csr_offsets, csr_offsets.data(), sizeof(int) * (num_nodes + 1), cudaMemcpyHostToDevice);
    // err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     printf("CUDA error after memory operations: %s\n", cudaGetErrorString(err));
    //     return;
    // }
    // printf("3\n");
    int *d_max_length;
    cudaMalloc(&d_max_length, sizeof(int));
    cudaMemset(d_max_length, 0, sizeof(int));
    calculate_hash_length<<<256, 1024>>>(*d_hash_length, num_nodes, *d_csr_offsets, d_max_length, load_factor, bucket_size);
    cudaDeviceSynchronize();

    // err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     printf("CUDA error after calculate_hash_length kernel: %s\n", cudaGetErrorString(err));
    //     return;
    // }
    cudaMemcpy(&max_length, d_max_length, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_max_length);
    //printf("Max length: %d\n", max_length);
    //printf("average degree: %f\n", (float)csr_cols.size() /num_nodes);

    // vector<int> h_hash_length(num_nodes);
    // cudaMemcpy(h_hash_length.data(), *d_hash_length, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);
    // //sort(h_hash_length.begin(), h_hash_length.end(),greater<int>());
    // for(int i=0;i<10&&i<num_nodes;i++)
    // {
    //     printf("Top %d length: %d\n", i, h_hash_length[i]);
    // }


    cudaMalloc(d_hash_tables_offset, sizeof(long long) * (num_nodes + 1));
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, *d_hash_length, *d_hash_tables_offset, num_nodes);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, *d_hash_length, *d_hash_tables_offset, num_nodes);
    int last_hash_length;
    cudaMemcpy(&last_hash_length, *d_hash_length + num_nodes - 1, sizeof(int), cudaMemcpyDeviceToHost);
    long long last_sum;
    cudaMemcpy(&last_sum, *d_hash_tables_offset + num_nodes - 1, sizeof(long long), cudaMemcpyDeviceToHost);
    //printf("last_sum: %lld, last_hash_length: %d\n", last_sum, last_hash_length);
    bucket_num = last_sum + last_hash_length;
    //printf("Total buckets: %lld\n", bucket_num);
    cudaMemcpy(*d_hash_tables_offset + num_nodes, &bucket_num, sizeof(long long), cudaMemcpyHostToDevice);
    cudaFree(d_temp_storage);
    cudaMalloc(d_hash_table_hierarchical, sizeof(int) * bucket_num * bucket_size);
    cudaMemset(*d_hash_table_hierarchical, -1, sizeof(int) * bucket_num * bucket_size);

    cudaMalloc(d_csr_cols, sizeof(int) * csr_cols.size());
    cudaMemcpy(*d_csr_cols, csr_cols.data(), sizeof(int) * csr_cols.size(), cudaMemcpyHostToDevice);

    vector<int>vertexs(csr_cols.size());
    for (int i = 0; i < csr_offsets.size() - 1; i++)
    {
        for (int j = csr_offsets[i]; j < csr_offsets[i + 1]; j++)
        {
            vertexs[j] = i;
        }
    }
    int *d_vertexs;
    cudaMalloc(&d_vertexs, sizeof(int) * vertexs.size());
    cudaMemcpy(d_vertexs, vertexs.data(), sizeof(int) * vertexs.size(), cudaMemcpyHostToDevice);

    int* d_conflict_count;
    cudaMalloc(&d_conflict_count, sizeof(int));

    cudaMemset(d_conflict_count, 0, sizeof(int));
    double start_time = clock();
    build_hash_table_hierarchical<<<1024, 1024>>>(*d_hash_table_hierarchical, *d_hash_length, *d_hash_tables_offset, csr_cols.size(), bucket_num, max_length, d_vertexs, *d_csr_cols, d_conflict_count, bucket_size);
    cudaDeviceSynchronize();
    double end_time = clock();
    //printf("Hierarchical hash table built in %f seconds.\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
    int conflict_count=0;
    cudaMemcpy(&conflict_count, d_conflict_count, sizeof(int), cudaMemcpyDeviceToHost);
    //printf("Hierarchical hash table built with %d conflicts.\n", conflict_count);

}