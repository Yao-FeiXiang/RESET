#include "hash_table_hie.cuh"
int *d_hash_tables_lens;
long long *d_hash_tables_offset;

__global__ void buildHashTableLensKernel(int *hash_tables_lens, int *degree_offset, int vertex_count, float load_factor, int bucket_size, int *max_length)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int size;
    int len;
    for (int i = tid; i < vertex_count; i += stride)
    {
        size = degree_offset[i + 1] - degree_offset[i];
        if (size < 0)
            printf("!!\n");
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
        atomicMax(max_length, len);
    }
}

__global__ void buildHashTableKernel(long long *hash_tables_offset, int *hash_tables, int *adjcant, int *vertex_list, int edge_count, long long bucket_num, int bucket_size, int max_length, unsigned long long *conflict, int *dispersed_mapping)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int key;
    int vertex;
    long long hash_table_start;
    long long hash_table_end;
    int hash_table_length;
    int mapped_value;
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
        mapped_value = d_hash_hierarchical(key, hash_table_length, max_length);
        int index = 0;
        while (atomicCAS(&hash_tables[hash_table_start + mapped_value + bucket_num * index], -1, key) != -1)
        {
            atomicAdd(conflict, 1);
            index++;
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

tuple<long long *, int *, long long, int,double> buildHashTable(int *d_adjcant, int *d_vertex, int *d_degree_offset, int *d_dispersed_mapping)
{
    // get bucket_num

    HRR(cudaMalloc(&d_hash_tables_lens, vertex_count * sizeof(int)));
    int *d_max_length;
    HRR(cudaMalloc(&d_max_length, sizeof(int)));
    int init_max_length = 8;
    HRR(cudaMemcpy(d_max_length, &init_max_length, sizeof(int), cudaMemcpyHostToDevice));

    buildHashTableLensKernel<<<216, 1024>>>(d_hash_tables_lens, d_degree_offset, vertex_count, load_factor, bucket_size, d_max_length);
    // exclusiveSum

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
    int *d_hash_tables_hierarchical;
    cout << "bucket_num is : " << bucket_num << endl;
    cout << "hash table size is : " << bucket_size * bucket_num * sizeof(int) << endl;

    HRR(cudaMalloc(&d_hash_tables_hierarchical, bucket_size * bucket_num * sizeof(int)));
    HRR(cudaMemset(d_hash_tables_hierarchical, -1, bucket_size * bucket_num * sizeof(int)));

    int max_length = 0;
    HRR(cudaMemcpy(&max_length, d_max_length, sizeof(int), cudaMemcpyDeviceToHost));

    unsigned long long *d_conflict;
    HRR(cudaMalloc(&d_conflict, sizeof(unsigned long long)));
    HRR(cudaMemset(d_conflict, 0, sizeof(unsigned long long)));
    unsigned long long conflict_init = 0;

    // cudaEvent 量的是 GPU 上该内核区间耗时；clock() 多为 CPU 时间，不适合量 kernel
    cudaEvent_t ev_b0, ev_b1;
    HRR(cudaEventCreate(&ev_b0));
    HRR(cudaEventCreate(&ev_b1));
    HRR(cudaEventRecord(ev_b0));
    buildHashTableKernel<<<216, 1024>>>(d_hash_tables_offset, d_hash_tables_hierarchical, d_adjcant, d_vertex, edge_count, bucket_num, bucket_size, max_length, d_conflict, d_dispersed_mapping);
    HRR(cudaEventRecord(ev_b1));
    HRR(cudaEventSynchronize(ev_b1));
    float build_ms = 0.f;
    HRR(cudaEventElapsedTime(&build_ms, ev_b0, ev_b1));
    HRR(cudaEventDestroy(ev_b0));
    HRR(cudaEventDestroy(ev_b1));
    double build_time = (double)build_ms / 1000.0;
    HRR(cudaMemcpy(&conflict_init, d_conflict, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    cout << "hierarchical hash table conflict num is : " << conflict_init << endl;

    return make_tuple(d_hash_tables_offset, d_hash_tables_hierarchical, bucket_num, max_length,build_time);
}
