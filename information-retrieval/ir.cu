#include "ir.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
vector<int> inverted_index_offsets;
vector<int> inverted_index;
int inverted_index_num = 0;
vector<int> global_query;
vector<int> query_offsets;
int query_num = 0;

bool cmp(int a, int b)
{
    return a % max_length < b % max_length;
}

vector<int> read_vector_binary(const string &filename)
{
    ifstream in(filename, ios::binary);
    if (!in)
    {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    size_t size;
    in.read(reinterpret_cast<char *>(&size), sizeof(size_t));

    vector<int> vec(size);
    in.read(reinterpret_cast<char *>(vec.data()), size * sizeof(int));

    return vec;
}

int read_int_binary(const string &filename)
{
    ifstream in(filename, ios::binary);
    if (!in)
    {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    int value;
    in.read(reinterpret_cast<char *>(&value), sizeof(int));

    return value;
}

struct l2flush
{
    __forceinline__ l2flush()
    {
        int dev_id{};
        (cudaGetDevice(&dev_id));
        (cudaDeviceGetAttribute(&m_l2_size, cudaDevAttrL2CacheSize, dev_id));
        if (m_l2_size > 0)
        {
            void *buffer = m_l2_buffer;
            (cudaMalloc(&buffer, m_l2_size));
            m_l2_buffer = reinterpret_cast<int *>(buffer);
        }
    }

    __forceinline__ ~l2flush()
    {
        if (m_l2_buffer)
        {
            (cudaFree(m_l2_buffer));
        }
    }

    __forceinline__ void flush(cudaStream_t stream)
    {
        if (m_l2_size > 0)
        {
            (cudaMemsetAsync(m_l2_buffer, 0, m_l2_size, stream));
        }
    }

private:
    int m_l2_size{};
    int *m_l2_buffer{};
};

__inline__ __device__ bool search_in_hashtable(int key, int *hashtable, int bucket_num, int bucket, int hash_length, int bucket_size)
{
    bool found = false;
    int index = 0;
    while (1)
    {
        if (hashtable[bucket + index * bucket_num] == key)
        {
            found = true;
            break;
        }
        else if (hashtable[bucket + index * bucket_num] == -1)
        {
            break;
        }
        index++;
        if (index == bucket_size)
        {
            index = 0;
            bucket = (bucket + 1) & (hash_length - 1);
        }
    }
    return found;
}

__global__ void ir_kernel(int *inverted_index, int *inverted_index_offsets, int *query, int *query_offsets, int query_num, int *result, long long *result_offsets, int *result_count, int *G_index, int CHUNK_SIZE, int max_length, bool opt, int *hashtable, long long *hashtable_offset, int bucket_num, int bucket_size)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / warpSize;
    int vertex = (blockIdx.x * num_warps + warp_id) * CHUNK_SIZE;
    int vertex_end = vertex + CHUNK_SIZE;
    int query_start, query_end;
    int query_len;
    int *set_start;
    int set_len;
    int *result_start;
    int *hashtable_start;
    int hash_length;

    while (vertex < query_num)
    {
        query_start = query_offsets[vertex];
        query_end = query_offsets[vertex + 1];
        query_len = query_end - query_start;

        int term0 = query[query_start];
        set_start = inverted_index + inverted_index_offsets[term0];
        set_len = inverted_index_offsets[term0 + 1] - inverted_index_offsets[term0];

        result_start = result + result_offsets[vertex];
        for (int i = lane_id; i < set_len; i += warpSize)
        {
            result_start[i] = set_start[i];
        }
        __syncwarp();
        int result_num = set_len;
        int *set = result_start;
        int set_size = result_num;
        for (int i = 1; i < query_len; i++)
        {
            int term = query[query_start + i];
            hashtable_start = hashtable + hashtable_offset[term];
            hash_length = hashtable_offset[term + 1] - hashtable_offset[term];

            result_num = 0;
            int num_iters = (set_size + warpSize - 1) / warpSize;
            for (int iter = 0; iter < num_iters; iter++)
            {
                int j = lane_id + iter * warpSize;
                bool active = (j < set_size);
                bool found = false;
                int key = -1;
                if (active)
                {
                    key = set[j];
                    int bucket = (opt) ? d_hash_hierarchical(key, hash_length, max_length) : d_hash_normal(key, hash_length);
                    found = search_in_hashtable(key, hashtable_start, bucket_num, bucket, hash_length, bucket_size);
                }

                unsigned int found_mask = __ballot_sync(0xffffffff, found);
                int step_found = __popc(found_mask);
                int write_pos = result_num + __popc(found_mask & ((1 << lane_id) - 1));
                if (found && active)
                {
                    result_start[write_pos] = key;
                }
                result_num += step_found;
            }
            set = result_start;
            set_size = __shfl_sync(0xffffffff, result_num, 0);
        }
        if (lane_id == 0)
        {
            result_count[vertex] = set_size;
        }
        __syncwarp();

        vertex++;
        if (vertex == vertex_end)
        {
            if (lane_id == 0)
            {
                vertex = atomicAdd(G_index, CHUNK_SIZE);
            }
            vertex = __shfl_sync(0xffffffff, vertex, 0);
            vertex_end = vertex + CHUNK_SIZE;
        }
    }
}

struct TupleComparator
{
    int max_length;
    __host__ __device__ TupleComparator(int ml = 8) : max_length(ml) {}
    __host__ __device__ bool operator()(const thrust::tuple<int, int> &a, const thrust::tuple<int, int> &b) const
    {
        int ra = thrust::get<0>(a);
        int rb = thrust::get<0>(b);
        if (ra != rb)
            return ra < rb;
        int ca = thrust::get<1>(a);
        int cb = thrust::get<1>(b);
        return (ca % max_length) < (cb % max_length);
    }
};

void gpu_sort_csr_cols(vector<int> &csr_cols, const vector<int> &csr_row, int num_nodes, int max_length)
{
    int total_edges = csr_cols.size();
    vector<int> rows_host(total_edges);
    for (int i = 0; i < num_nodes; ++i)
    {
        int start = csr_row[i];
        int end = csr_row[i + 1];
        for (int p = start; p < end; ++p)
            rows_host[p] = i;
    }

    thrust::device_vector<int> d_csr_cols = csr_cols;
    thrust::device_vector<int> d_rows = rows_host;

    auto first = thrust::make_zip_iterator(thrust::make_tuple(d_rows.begin(), d_csr_cols.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(d_rows.end(), d_csr_cols.end()));
    thrust::sort(first, last, TupleComparator(max_length));
    thrust::copy(d_csr_cols.begin(), d_csr_cols.end(), csr_cols.begin());
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <input_folder>" << endl;
        return 1;
    }

    int dev = 1; // 选择物理 GPU 编号 1
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d devices\n", deviceCount);
    if (dev >= deviceCount)
    {
        fprintf(stderr, "Device %d out of range\n", dev);
        return 1;
    }
    cudaError_t e = cudaSetDevice(dev);
    if (e != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", dev, cudaGetErrorString(e));
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Using device %d: %s\n", dev, prop.name);

    string input_folder = argv[1];
    float load_factor = 0.25;
    int bucket_size = 4;

    for (int i = 2; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.rfind("--alpha=", 0) == 0)
        {
            load_factor = std::stof(arg.substr(8));
        }
        else if (arg.rfind("--bucket=", 0) == 0)
        {
            bucket_size = std::stoi(arg.substr(9));
        }
        else
        {
            cerr << "Unknown argument: " << arg << endl;
            return 1;
        }
    }
    printf("load_factor:%.2f , bucket_size:%d\n", load_factor, bucket_size);

    string inverted_index_offsets_path = input_folder + "/inverted_index_offsets.bin";
    string inverted_index_path = input_folder + "/inverted_index.bin";
    string inverted_index_num_path = input_folder + "/inverted_index_num.bin";
    string query_path = input_folder + "/query.bin";
    string query_offsets_path = input_folder + "/query_offsets.bin";
    string query_num_path = input_folder + "/query_num.bin";

    inverted_index_offsets = read_vector_binary(inverted_index_offsets_path);
    inverted_index = read_vector_binary(inverted_index_path);
    inverted_index_num = read_int_binary(inverted_index_num_path);
    global_query = read_vector_binary(query_path);
    query_offsets = read_vector_binary(query_offsets_path);
    query_num = read_int_binary(query_num_path);

    cout << "Successfully loaded data:" << endl;
    cout << "inverted_index_offsets size: " << inverted_index_offsets.size() << endl;
    cout << "inverted_index size: " << inverted_index.size() << endl;
    cout << "inverted_index_num: " << inverted_index_num << endl;
    cout << "query size: " << global_query.size() << endl;
    cout << "query_offsets size: " << query_offsets.size() << endl;
    cout << "query_num: " << query_num << endl;

    int max_degree = 0;
    for(int i=0;i<inverted_index_num;i++)
    {
        max_degree = max(max_degree, inverted_index_offsets[i + 1] - inverted_index_offsets[i]);
    }
    printf("max degree: %d\n", max_degree);

    // int max_doc_id = 0;
    // for (int i = 0; i < inverted_index.size();i++)
    // {
    //     max_doc_id = max(max_doc_id, inverted_index[i]);
    // }
    // printf("max doc id: %d\n", max_doc_id);
    // vector<int> doc_length(inverted_index_num, 0);
    // for (int i = 0;i<inverted_index_num;i++)
    // {
    //     doc_length[i] = inverted_index_offsets[i + 1] - inverted_index_offsets[i];
    // }
    // sort(doc_length.begin(), doc_length.end(), greater<int>());
    // for(int i=0;i<100;i++)
    // {
    //     printf("Top %d length: %d\n", i + 1, doc_length[i]);
    // }

    make_hash_table(load_factor, bucket_size);
    printf("finish make hash table\n");
    int CHUNK_SIZE = 2;
    int grid_size = 512;
    int block_size = 1024;
    int *d_G_index;
    cudaMalloc(&d_G_index, sizeof(int));
    int h_G_index = grid_size * block_size / warpSize * CHUNK_SIZE;
    cudaMemcpy(d_G_index, &h_G_index, sizeof(int), cudaMemcpyHostToDevice);
    int *d_query, *d_query_offsets;
    cudaMalloc(&d_query, global_query.size() * sizeof(int));
    cudaMalloc(&d_query_offsets, (query_num + 1) * sizeof(int));
    cudaMemcpy(d_query, global_query.data(), global_query.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_offsets, query_offsets.data(), (query_num + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int *d_inverted_index;
    cudaMalloc(&d_inverted_index, inverted_index.size() * sizeof(int));
    cudaMemcpy(d_inverted_index, inverted_index.data(), inverted_index.size() * sizeof(int), cudaMemcpyHostToDevice);
    vector<long long> result_offsets(query_num + 1, 0);
    for (int i = 0; i < query_num; i++)
    {
        result_offsets[i + 1] = result_offsets[i] + inverted_index_offsets[global_query[query_offsets[i]] + 1] - inverted_index_offsets[global_query[query_offsets[i]]];
    }
    printf("result offset end: %lld\n", result_offsets[query_num]);
    long long *d_result_offsets;
    cudaMalloc(&d_result_offsets, (query_num + 1) * sizeof(long long));
    cudaMemcpy(d_result_offsets, result_offsets.data(), (query_num + 1) * sizeof(long long), cudaMemcpyHostToDevice);
    int *d_result;
    cudaMalloc(&d_result, result_offsets.back() * sizeof(int));
    cudaMemset(d_result, -1, result_offsets.back() * sizeof(int));
    int *d_result_count;
    cudaMalloc(&d_result_count, query_num * sizeof(int));
    cudaMemset(d_result_count, 0, query_num * sizeof(int));
    bool opt = false;

    // hierarchical sort
    // for (int i = 0; i < inverted_index_num; i++)
    // {
    //     sort(inverted_index.begin() + inverted_index_offsets[i], inverted_index.begin() + inverted_index_offsets[i + 1], cmp);
    // }
    gpu_sort_csr_cols(inverted_index, inverted_index_offsets, inverted_index_num, max_length);

    printf("finish hierarchical sort\n");
    int *d_inverted_index_hierarchical;
    cudaMalloc(&d_inverted_index_hierarchical, inverted_index.size() * sizeof(int));
    cudaMemcpy(d_inverted_index_hierarchical, inverted_index.data(), inverted_index.size() * sizeof(int), cudaMemcpyHostToDevice);

    // vector<int> hash_table(20);
    // cudaMemcpy(hash_table.data(), d_hash_table_hierarchical, 20 * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 20; i++)
    // {
    //     printf("%d ", hash_table[i]);
    // }
    // printf("\n");

    l2flush();
    double time_start = clock();
    ir_kernel<<<grid_size, block_size>>>(d_inverted_index, d_inverted_index_offsets, d_query, d_query_offsets, query_num, d_result, d_result_offsets, d_result_count, d_G_index, CHUNK_SIZE, max_length, opt, d_hash_table_normal, d_hash_tables_offset, bucket_num, bucket_size);
    cudaDeviceSynchronize();
    double time_end = clock();
    double cmp_time = (time_end - time_start) / CLOCKS_PER_SEC;
    cout << "Noramal Kernel execution time: " << cmp_time << " seconds" << endl;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cerr << "Kernel failed: " << cudaGetErrorString(err) << endl;
        cerr << "Error code: " << err << endl;
    }

    vector<int> result(result_offsets.back());
    cudaMemcpy(result.data(), d_result, result_offsets.back() * sizeof(int), cudaMemcpyDeviceToHost);
    vector<int> result_count(query_num);
    cudaMemcpy(result_count.data(), d_result_count, query_num * sizeof(int), cudaMemcpyDeviceToHost);
    int sum1 = 0;
    for (int i = 0; i < query_num; i++)
    {
        sum1 += result_count[i];
    }

    cudaMemset(d_result, -1, result_offsets.back() * sizeof(int));
    cudaMemset(d_result_count, 0, query_num * sizeof(int));
    cudaMemcpy(d_G_index, &h_G_index, sizeof(int), cudaMemcpyHostToDevice);

    l2flush();
    time_start = clock();
    ir_kernel<<<grid_size, block_size>>>(d_inverted_index_hierarchical, d_inverted_index_offsets, d_query, d_query_offsets, query_num, d_result, d_result_offsets, d_result_count, d_G_index, CHUNK_SIZE, max_length, true, d_hash_table_hierarchical, d_hash_tables_offset, bucket_num, bucket_size);
    cudaDeviceSynchronize();
    time_end = clock();
    cmp_time = (time_end - time_start) / CLOCKS_PER_SEC;
    cout << "Hierarchical Kernel execution time: " << cmp_time << " seconds" << endl;
    err = cudaGetLastError();
    result.clear();
    result.resize(result_offsets.back());
    cudaMemcpy(result.data(), d_result, result_offsets.back() * sizeof(int), cudaMemcpyDeviceToHost);
    result_count.clear();
    result_count.resize(query_num);
    cudaMemcpy(result_count.data(), d_result_count, query_num * sizeof(int), cudaMemcpyDeviceToHost);
    int sum2 = 0;
    for (int i = 0; i < query_num; i++)
    {
        sum2 += result_count[i];
    }

    if (sum1 != sum2)
    {
        printf("Result mismatch between normal and hierarchical!\n");
    }
    else
    {
        printf("Total result count: %d\n", sum1);
    }

    return 0;
}