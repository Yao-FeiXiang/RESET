#include "hash_table.cuh"
__global__ void buildHashTableLensKernel(int *hash_tables_lens, int *degree_offset, int vertex_count, float load_factor, int bucket_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int size;
    int len;
    for (int i = tid; i < vertex_count; i += stride)
    {
        size = degree_offset[i + 1] - degree_offset[i];
        len = (int)(size / load_factor / bucket_size);
        if (size / load_factor / bucket_size != (float)len)
            len++;
        // Round up to next power of 2 using bit manipulation (avoids float precision issues)
        if (len > 0)
        {
            len--;
            len |= len >> 1;
            len |= len >> 2;
            len |= len >> 4;
            len |= len >> 8;
            len |= len >> 16;
            len++;
        }
        if (len < 8 && len != 0)
            len = 8;
        hash_tables_lens[i] = len;
    }
}

__global__ void buildHashTableKernel(long long *hash_tables_offset, int *hash_tables, int *adjcant, int *vertex_list, int edge_count, long long bucket_num, int bucket_size, int *conflict, int *dispersed_mapping)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int key;
    int vertex;
    long long hash_table_start;
    long long hash_table_end;
    int hash_table_length;
    int value;
    // int mapped_value;
    for (int i = tid; i < edge_count; i += stride)
    {
        key = adjcant[i];
        vertex = vertex_list[i];
        hash_table_start = hash_tables_offset[vertex];
        hash_table_end = hash_tables_offset[vertex + 1];
        hash_table_length = hash_table_end - hash_table_start;
        key = dispersed_mapping[key];
        value = key % hash_table_length;
        int mapped_value = map_value(value, hash_table_length);
        int index = 0;
        while (atomicCAS(&hash_tables[hash_table_start + mapped_value + bucket_num * index], -1, key) != -1)
        {
            index++;
            atomicAdd(conflict, 1);
            if (index == bucket_size)
            {
                index = 0;
                mapped_value++;
                if (mapped_value == hash_table_length)
                    mapped_value = 0;
            }
        }
    }
}

tuple<long long *, int *, long long> buildHashTable(int *d_adjcant, int *d_vertex, int *d_degree_offset, int *d_dispersed_mapping)
{
    // get bucket_num
    int *d_hash_tables_lens;
    HRR(cudaMalloc(&d_hash_tables_lens, vertex_count * sizeof(int)));
    buildHashTableLensKernel<<<216, 1024>>>(d_hash_tables_lens, d_degree_offset, vertex_count, load_factor, bucket_size);
    // exclusiveSum
    long long *d_hash_tables_offset;
    HRR(cudaMalloc(&d_hash_tables_offset, (vertex_count + 1) * sizeof(long long)));
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_hash_tables_lens, d_hash_tables_offset, vertex_count);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_hash_tables_lens, d_hash_tables_offset, vertex_count);

    // get sum
    int last_count;
    long long last_sum;
    long long bucket_num;
    HRR(cudaMemcpy(&last_count, d_hash_tables_lens + vertex_count - 1, sizeof(int), cudaMemcpyDeviceToHost));
    HRR(cudaMemcpy(&last_sum, d_hash_tables_offset + vertex_count - 1, sizeof(long long), cudaMemcpyDeviceToHost));
    bucket_num = last_sum + last_count;
    HRR(cudaMemcpy(d_hash_tables_offset + vertex_count, &bucket_num, sizeof(long long), cudaMemcpyHostToDevice));
    // build hash table in device
    int *d_hash_tables;
    cout << "bucket_num is : " << bucket_num << endl;
    cout << "hash table size is : " << bucket_size * bucket_num * sizeof(int) << endl;

    HRR(cudaMalloc(&d_hash_tables, bucket_size * bucket_num * sizeof(int)));
    HRR(cudaMemset(d_hash_tables, -1, bucket_size * bucket_num * sizeof(int)));

    int *d_conflict;
    HRR(cudaMalloc(&d_conflict, sizeof(int)));
    HRR(cudaMemset(d_conflict, 0, sizeof(int)));
    buildHashTableKernel<<<216, 1024>>>(d_hash_tables_offset, d_hash_tables, d_adjcant, d_vertex, edge_count, bucket_num, bucket_size, d_conflict, d_dispersed_mapping);
    printf("build hash table finished\n");
    int h_conflict;
    HRR(cudaMemcpy(&h_conflict, d_conflict, sizeof(int), cudaMemcpyDeviceToHost));
    cout << "conflict num is : " << h_conflict << endl;
    cudaFree(d_conflict);
    return make_tuple(d_hash_tables_offset, d_hash_tables, bucket_num);
}
