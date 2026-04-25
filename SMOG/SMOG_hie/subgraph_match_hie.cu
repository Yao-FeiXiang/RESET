#include <sstream>
#include <string>
#include <math.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <cooperative_groups.h>
#include "hash_table_hie.cuh"
#include "subgraph_match_hie.cuh"

using namespace cooperative_groups;
// using namespace std;
//  h : height of subtree; h = pattern vertex number
__inline__ __device__ bool checkDuplicate(int *mapping, int &level, int item)
{
    for (int i = 0; i < level; i++)
        if (mapping[i] == item)
            return true;
    return false;
}
__inline__ __device__ bool checkRestriction(int *mapping, int &level, int item, int *restriction) // 返回true表示违背限制
{
#ifdef withDuplicate
    if (restriction[level] == -1)
    {
        for (int i = 0; i < level; i++)
            if (mapping[i] == item)
                return true;
    }
#endif
#ifdef withRestriction
    if (restriction[level] == -1)
        return false;
    if (item < mapping[restriction[level]])
        return false;
    return true;
#else
    return false;
#endif
}

__inline__ __device__ void loadNextVertex(int &start_index, int &this_chunk_index_end, int *G_INDEX, int &lid, int &chunk_size, int &process_num)
{
    start_index += process_num;
    if (start_index == this_chunk_index_end)
    {
        if (lid == 0)
        {
            start_index = atomicAdd(G_INDEX, chunk_size * process_num);
        }
        start_index = __shfl_sync(FULL_MASK, start_index, 0);
        this_chunk_index_end = start_index + chunk_size * process_num;
    }
}

__global__ void DFSKernelForClique(int *reuse, int process_id, int process_num, int chunk_size, int index_length, int bucket_size, long long bucket_num, int max_degree, int *subgraph_adj, int *subgraph_offset, int *restriction, int *degree_offset, int *adjcant, long long *hash_tables_offset, int *hash_tables, int *vertex, int *candidates_of_all_warp, unsigned long long *sum, int *G_INDEX, int max_length, int *d_dispersed_mapping)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // warpid
    int lid = threadIdx.x % 32;                             // landid
    int next_candidate_array[H];                            // 该层下一个候选
    int candidate_number_array[H];                          // 记录一下每一层保存的数据大小
    int mapping[H];                                         // 记录每次的中间结果
    int *my_candidates_for_all_mapping = candidates_of_all_warp + (long long)wid * H * max_degree;
    // int my_count = 0;
    // __shared__ int shared_count;
    unsigned long long my_count = 0;
    __shared__ unsigned long long shared_count;
    if (threadIdx.x == 0)
        shared_count = 0;
    int start_index = wid * chunk_size * process_num + process_id;
    int this_chunk_index_end = start_index + chunk_size * process_num;
    int level;
    // each warp process a subtree
    for (; start_index < index_length; loadNextVertex(start_index, this_chunk_index_end, G_INDEX, lid, chunk_size, process_num))
    {
#ifdef useVertexAsStart
        mapping[0] = start_index;
        level = 0;
#else
        mapping[0] = vertex[start_index];
        mapping[1] = adjcant[start_index];
        level = 1;
        if (checkRestriction(mapping, level, mapping[1], restriction))
            continue;
#endif
        for (;;)
        {
            level++;
            int &candidate_number = candidate_number_array[level];
            next_candidate_array[level] = -1;
            candidate_number = 0;
            int subgraph_adj_start;
            int min_degree_vertex;
            int min_degree;
            int subgraph_degree;
            int *neighbor_list_of_min_degree_vertex;
            if (reuse[level] == -1)
            {
                // find possible connection and maintain in S
                subgraph_adj_start = subgraph_offset[level - 1];
                subgraph_degree = subgraph_offset[level] - subgraph_adj_start;
                // get degree
                min_degree_vertex = mapping[subgraph_adj[subgraph_adj_start]];
                int cur_degree;
                min_degree = degree_offset[min_degree_vertex + 1] - degree_offset[min_degree_vertex];
                for (int i = 1; i < subgraph_degree; i++)
                {
                    cur_degree = degree_offset[mapping[subgraph_adj[subgraph_adj_start + i]] + 1] - degree_offset[mapping[subgraph_adj[subgraph_adj_start + i]]];
                    if (cur_degree < min_degree)
                    {
                        min_degree_vertex = mapping[subgraph_adj[subgraph_adj_start + i]];
                        min_degree = cur_degree;
                    }
                }
                neighbor_list_of_min_degree_vertex = adjcant + degree_offset[min_degree_vertex];
            }
            else
            {
                // find possible connection and maintain in S
                subgraph_adj_start = subgraph_offset[level - 1];
                subgraph_degree = subgraph_offset[level] - subgraph_adj_start;
                min_degree = candidate_number_array[reuse[level]];
                min_degree_vertex = -1;
                neighbor_list_of_min_degree_vertex = my_candidates_for_all_mapping + reuse[level] * max_degree;
            }
            // if (mapping[0] == 0 && mapping[1] == 1 && lid == 0)
            // printf("level : %d min_degree = %d \n", level, min_degree);
            // Start intersection, load neighbor of m1 into my_candidates
            int *my_wirttern_candidates = my_candidates_for_all_mapping + level * max_degree;
            int processing_vertex_in_map;

            // 要做交集，则不抄出来,记录一下这个neighbour list所在的位置
            if (subgraph_degree == 1 && reuse[level] == -1)
                for (int i = lid; i < min_degree; i += 32)
                {
                    my_wirttern_candidates[i] = neighbor_list_of_min_degree_vertex[i];
                }
            if (subgraph_degree == 1 && reuse[level] != -1)
            {
                if (level < H - 1)
                    for (int i = lid; i < candidate_number_array[reuse[level]]; i += 32)
                    {
                        my_wirttern_candidates[i] = neighbor_list_of_min_degree_vertex[i];
                    }
                else
                {
                    for (int i = lid; i < candidate_number_array[reuse[level]]; i += 32)
                    {
                        // printf("level : %d reuse[level] : %d", level, reuse[level]);
                        if (!checkRestriction(mapping, level, neighbor_list_of_min_degree_vertex[i], restriction))
                        {
                            my_count += 1;
                            // printf("my count : %d", my_count);
                        }
                    }
                }
            }
            // if (mapping[0] == 0 && mapping[1] == 1 && lid == 0)
            // printf("min_degree_vertex : %d min_degree : %d level : %d reuse[level] : %d candidate_number_array[level - 1] : %d\n", min_degree_vertex, min_degree, level, reuse[level], candidate_number_array[level - 1]);
            // intersect
            candidate_number = min_degree;
            int *my_read_candidates;
            if (reuse[level] == -1)
                my_read_candidates = neighbor_list_of_min_degree_vertex;
            else
                my_read_candidates = my_candidates_for_all_mapping + reuse[level] * max_degree;
            for (int j = 0, is_not_last = subgraph_degree - 2; j < subgraph_degree; j++)
            {
                processing_vertex_in_map = mapping[subgraph_adj[subgraph_adj_start + j]];
                if (processing_vertex_in_map == min_degree_vertex || subgraph_adj[subgraph_adj_start + j] == -1)
                    continue;
                int *cur_hashtable = hash_tables + hash_tables_offset[processing_vertex_in_map];
                int len = int(hash_tables_offset[processing_vertex_in_map + 1] - hash_tables_offset[processing_vertex_in_map]); // len记录当前hash_table的长度

                int candidate_number_previous = candidate_number;
                candidate_number = 0;
                if (level < H - 1 || is_not_last)
                {
                    for (int i = lid; i < candidate_number_previous; i += 32)
                    {
                        int item = my_read_candidates[i];
                        int is_exist = search_in_hashtable(item, bucket_size, bucket_num, len, cur_hashtable, max_length, d_dispersed_mapping);
                        // 使用 prefix sum 计算写入位置，保持原顺序
                        unsigned int active_mask = __activemask();
                        unsigned int mask = __ballot_sync(active_mask, is_exist);
                        int prefix = __popc(mask & ((1u << lid) - 1));
                        if (is_exist)
                        {
                            my_wirttern_candidates[candidate_number + prefix] = item;
                        }
                        candidate_number += __popc(mask);
                    }
                }
                else
                {
                    for (int i = lid; i < candidate_number_previous; i += 32)
                    {
                        int item = my_read_candidates[i];
                        int is_exist = search_in_hashtable(item, bucket_size, bucket_num, len, cur_hashtable, max_length, d_dispersed_mapping);
                        if (!checkRestriction(mapping, level, item, restriction))
                        {
                            my_count += is_exist;
                        }
                    }
                }
                candidate_number = __shfl_sync(FULL_MASK, candidate_number, 0);
                my_read_candidates = my_wirttern_candidates;
                is_not_last--;
            }
            if (level == H - 1)
            {
                // if (lid == 0)
                // {
                //     my_count += candidate_number;
                // }
                level--;
            }
            for (;; level--)
            {
                if (level == break_level)
                    break;
                next_candidate_array[level]++;
                while (checkRestriction(mapping, level, my_candidates_for_all_mapping[level * max_degree + next_candidate_array[level]], restriction) && next_candidate_array[level] < candidate_number_array[level])
                {
                    next_candidate_array[level]++;
                }
                if (next_candidate_array[level] < candidate_number_array[level])
                {
                    mapping[level] = my_candidates_for_all_mapping[level * max_degree + next_candidate_array[level]];
                    break;
                }
            }
            if (level == break_level)
                break;
        }
    }
    // my_count = __reduce_add_sync(FULL_MASK, my_count);
    // if (lid == 0)
    // {
    atomicAdd(&shared_count, my_count);
    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(sum, shared_count);
    // }
}

__global__ void DFSKernelForGeneral(int process_id, int process_num, int chunk_size, int index_length, int bucket_size, long long bucket_num, int max_degree, int *subgraph_adj, int *subgraph_offset, int *restriction, int *degree_offset, int *adjcant, long long *hash_tables_offset, int *hash_tables, int *vertex, int *candidates_of_all_warp, unsigned long long *sum, int *G_INDEX, int max_length, int *d_dispersed_mapping)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // warpid
    int lid = threadIdx.x % 32;                             // landid
    int next_candidate_array[H];                            // 记录一下每一层保存的数据大小
    // int candidate_number[H];
    int mapping[H]; // 记录每次的中间结果
    int *my_candidates_for_all_mapping = candidates_of_all_warp + (long long)wid * H * max_degree;
    int my_count = 0;
    __shared__ int shared_count;
    if (threadIdx.x == 0)
        shared_count = 0;
    int start_index = wid * chunk_size * process_num + process_id;
    int this_chunk_index_end = start_index + chunk_size * process_num;
    int level;
    // if (lid == 0 && wid == 0)
    // each warp process a subtree
    for (; start_index < index_length; loadNextVertex(start_index, this_chunk_index_end, G_INDEX, lid, chunk_size, process_num))
    {
#ifdef useVertexAsStart
        mapping[0] = start_index;
        level = 0;
#else
        mapping[0] = vertex[start_index];
        mapping[1] = adjcant[start_index];
        level = 1;
        if (checkRestriction(mapping, level, mapping[1], restriction))
            continue;
#endif
        for (;;)
        {
            level++;
            int candidate_number;
            candidate_number = 0;
            // find possible connection and maintain in S
            int subgraph_adj_start = subgraph_offset[level - 1];
            int subgraph_degree = subgraph_offset[level] - subgraph_adj_start;
            // get degree
            int min_degree;
            int min_degree_vertex = mapping[subgraph_adj[subgraph_adj_start]];
            int cur_degree;
            min_degree = degree_offset[min_degree_vertex + 1] - degree_offset[min_degree_vertex];
            for (int i = 1; i < subgraph_degree; i++)
            {
                cur_degree = degree_offset[mapping[subgraph_adj[subgraph_adj_start + i]] + 1] - degree_offset[mapping[subgraph_adj[subgraph_adj_start + i]]];
                if (cur_degree < min_degree)
                {
                    min_degree_vertex = mapping[subgraph_adj[subgraph_adj_start + i]];
                    min_degree = cur_degree;
                }
            }
            // Start intersection, load neighbor of m1 into my_candidates
            int *my_wirttern_candidates = my_candidates_for_all_mapping + level * max_degree;
            int processing_vertex_in_map;
            int *neighbor_list_of_min_degree_vertex = adjcant + degree_offset[min_degree_vertex];

            // 要做交集，则不抄出来,记录一下这个neighbour list所在的位置
            if (subgraph_degree == 1)
            {
                if (level == H - 1)
                {
                    for (int i = lid; i < min_degree; i += 32)
                    {
                        if (!checkRestriction(mapping, level, neighbor_list_of_min_degree_vertex[i], restriction))
                            my_count++;
                    }
                }
                else
                {
                    for (int i = lid; i < min_degree; i += 32)
                    {
                        my_wirttern_candidates[i] = neighbor_list_of_min_degree_vertex[i];
                    }
                }
            }
            // intersect
            candidate_number = min_degree;
            int *my_read_candidates = neighbor_list_of_min_degree_vertex;
            for (int j = 0, is_not_last = subgraph_degree - 2; j < subgraph_degree; j++)
            {
                processing_vertex_in_map = mapping[subgraph_adj[subgraph_adj_start + j]];
                if (processing_vertex_in_map == min_degree_vertex)
                    continue;
                int *cur_hashtable = hash_tables + hash_tables_offset[processing_vertex_in_map];
                int len = int(hash_tables_offset[processing_vertex_in_map + 1] - hash_tables_offset[processing_vertex_in_map]); // len记录当前hash_table的长度

                int candidate_number_previous = candidate_number;
                candidate_number = 0;
                if (level < H - 1 || is_not_last)
                {
                    for (int i = lid; i < candidate_number_previous; i += 32)
                    {
                        int item = my_read_candidates[i];
                        int is_exist = search_in_hashtable(item, bucket_size, bucket_num, len, cur_hashtable, max_length, d_dispersed_mapping);
                        // 使用 prefix sum 计算写入位置，保持原顺序
                        unsigned int active_mask = __activemask();
                        unsigned int mask = __ballot_sync(active_mask, is_exist);
                        int prefix = __popc(mask & ((1u << lid) - 1));
                        if (is_exist)
                        {
                            my_wirttern_candidates[candidate_number + prefix] = item;
                        }
                        candidate_number += __popc(mask);
                    }
                }
                else
                {
                    for (int i = lid; i < candidate_number_previous; i += 32)
                    {
                        int item = my_read_candidates[i];
                        int is_exist = search_in_hashtable(item, bucket_size, bucket_num, len, cur_hashtable, max_length, d_dispersed_mapping);
                        if (!checkRestriction(mapping, level, item, restriction))
                        {
                            my_count += is_exist;
                        }
                    }
                }
                candidate_number = __shfl_sync(FULL_MASK, candidate_number, 0);
                my_read_candidates = my_wirttern_candidates;
                is_not_last--;
            }
            next_candidate_array[level] = candidate_number;
            if (level == H - 1)
            {
                // if (lid == 0)
                // {
                //     my_count += candidate_number;
                // }
                level--;
            }
            for (;; level--)
            {
                if (level == break_level)
                    break;
                next_candidate_array[level]--;
                while (checkRestriction(mapping, level, my_candidates_for_all_mapping[level * max_degree + next_candidate_array[level]], restriction) && next_candidate_array[level] >= 0)
                {
                    next_candidate_array[level]--;
                }
                if (next_candidate_array[level] > -1)
                {
                    mapping[level] = my_candidates_for_all_mapping[level * max_degree + next_candidate_array[level]];
                    break;
                }
            }
            if (level == break_level)
                break;
        }
    }
    // my_count = __reduce_add_sync(FULL_MASK, my_count);
    // if (lid == 0)
    // {
    atomicAdd(&shared_count, my_count);
    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(sum, shared_count);
    // }
}

struct ModComparator
{
    int max_length;
    int *dispersed_mapping;
    __host__ __device__ ModComparator(int ml, int *dm) : max_length(ml), dispersed_mapping(dm) {}
    __host__ __device__ bool operator()(const thrust::tuple<int, int> &a, const thrust::tuple<int, int> &b) const
    {
        int ra = thrust::get<0>(a);
        int rb = thrust::get<0>(b);
        if (ra != rb)
            return ra < rb;
        int ca = thrust::get<1>(a);
        int cb = thrust::get<1>(b);
        return (dispersed_mapping[ca] % max_length) < (dispersed_mapping[cb] % max_length);
    }
};

void gpu_sort_adjcant_by_mod(int *d_adjcant, int *d_degree_offset, int node_num, int num_edges, int max_length, int *d_adjcant_hierarchical, int *d_dispersed_mapping)
{
    std::vector<int> h_offsets(node_num + 1);
    cudaMemcpy(h_offsets.data(), d_degree_offset, sizeof(int) * (node_num + 1), cudaMemcpyDeviceToHost);

    std::vector<int> h_rows(num_edges);
    for (int i = 0; i < node_num; ++i)
    {
        int start = h_offsets[i];
        int end = h_offsets[i + 1];
        for (int p = start; p < end; ++p)
            h_rows[p] = i;
    }

    thrust::device_vector<int> d_rows(num_edges);
    cudaMemcpy(thrust::raw_pointer_cast(d_rows.data()), h_rows.data(),
               sizeof(int) * num_edges, cudaMemcpyHostToDevice);

    thrust::device_vector<int> d_cols(num_edges);
    cudaMemcpy(thrust::raw_pointer_cast(d_cols.data()), d_adjcant,
               sizeof(int) * num_edges, cudaMemcpyDeviceToDevice);

    auto first = thrust::make_zip_iterator(thrust::make_tuple(d_rows.begin(), d_cols.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(d_rows.end(), d_cols.end()));

    thrust::sort(first, last, ModComparator(max_length, d_dispersed_mapping));
    cudaMemcpy(d_adjcant_hierarchical, thrust::raw_pointer_cast(d_cols.data()),
               sizeof(int) * num_edges, cudaMemcpyDeviceToDevice);
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

int gcd(int a, int b)
{
    while (b)
    {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

vector<int> createDispersedMapping(int n, vector<int> &reverse_mapping)
{
    vector<int> mapping(n);

    // Find a step size that is coprime with n
    // Start with a large prime-like number
    int step = 10007;
    while (gcd(step, n) != 1)
    {
        step++;
    }

    // Create permutation: perm[i] = (i * step) % n
    // This guarantees uniform distribution and complete scrambling
    for (int i = 0; i < n; i++)
    {
        mapping[i] = ((long long)i * step) % n;
        mapping[i] = ((long long)mapping[i] * step) % n; // double shuffle for better randomness
        reverse_mapping[mapping[i]] = i;
    }

    printf("Created coprime dispersed mapping: step=%d (gcd=%d), completely scrambled\n",
           step, gcd(step, n));
    return mapping;
}

struct arguments SubgraphMatching(int process_id, int process_num, struct arguments args, char *argv[])
{
    int deviceCount = 0;
    cudaError_t ec = cudaGetDeviceCount(&deviceCount);
    if (ec != cudaSuccess || deviceCount <= 0)
    {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(ec));
        exit(1);
    }
    // 原先硬编码 dev=1：在仅可见一块卡（NCU/CUDA_VISIBLE_DEVICES）或单卡节点上会 cudaSetDevice 失败，
    // Thrust 可能报 “not compiled for SM 80” 与 invalid device ordinal 等连锁错误。
    int dev = process_id % deviceCount;
    cudaError_t e = cudaSetDevice(dev);
    if (e != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice(%d) failed (%d GPUs visible): %s\n", dev, deviceCount,
                cudaGetErrorString(e));
        exit(1);
    }
    cudaDeviceProp prop{};
    e = cudaGetDeviceProperties(&prop, dev);
    if (e != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDeviceProperties(%d) failed: %s\n", dev, cudaGetErrorString(e));
        exit(1);
    }
    printf("Using device %d: %s\n", dev, prop.name);
    string Infilename = argv[1];
    string pattern = argv[2];
    load_factor = atof(argv[4]);
    bucket_size = atoi(argv[5]);
    block_size = atoi(argv[6]);
    block_number = atoi(argv[7]);
    chunk_size = atoi(argv[8]);

    int *d_adjcant, *d_vertex, *d_degree_offset;
    int max_degree;
    tie(d_adjcant, d_vertex, d_degree_offset, max_degree) = loadGraphWithName(Infilename, pattern);
    // printGpuInfo();
    printf("max degree is : %d\n", max_degree);

    int total_edges;
    cudaMemcpy(&total_edges, d_degree_offset + vertex_count, sizeof(int), cudaMemcpyDeviceToHost);
    int *d_adjcant_hierarchical;
    cudaMalloc(&d_adjcant_hierarchical, sizeof(int) * total_edges);

    int *d_hash_tables_hierarchical;
    long long *d_hash_tables_offset;
    long long bucket_num;
    int max_length;

    vector<int> reverse_mapping(vertex_count);
    vector<int> dispersed_mapping = createDispersedMapping(vertex_count, reverse_mapping);
    int *d_dispersed_mapping;
    cudaMalloc(&d_dispersed_mapping, sizeof(int) * vertex_count);
    cudaMemcpy(d_dispersed_mapping, dispersed_mapping.data(), sizeof(int) * vertex_count, cudaMemcpyHostToDevice);

    tie(d_hash_tables_offset, d_hash_tables_hierarchical, bucket_num, max_length) = buildHashTable(d_adjcant, d_vertex, d_degree_offset, d_dispersed_mapping);
    printf("finish make hash table\n");

    gpu_sort_adjcant_by_mod(d_adjcant, d_degree_offset, vertex_count, total_edges, max_length, d_adjcant_hierarchical, d_dispersed_mapping);

    // Debug: uncomment below to verify hash table consistency
    /*
    printf("max_length is : %d\n", max_length);
    vector<int> adjcant(edge_count);
    vector<int> adjcant_hierarchical(edge_count);
    vector<int> degree_offset(vertex_count + 1);
    cudaMemcpy(adjcant.data(), d_adjcant, sizeof(int) * edge_count, cudaMemcpyDeviceToHost);
    cudaMemcpy(adjcant_hierarchical.data(), d_adjcant_hierarchical, sizeof(int) * edge_count, cudaMemcpyDeviceToHost);
    cudaMemcpy(degree_offset.data(), d_degree_offset, sizeof(int) * (vertex_count + 1), cudaMemcpyDeviceToHost);

    vector<int> hashtable_hierarchical(bucket_size * bucket_num);
    cudaMemcpy(hashtable_hierarchical.data(), d_hash_tables_hierarchical, sizeof(int) * bucket_size * bucket_num, cudaMemcpyDeviceToHost);
    vector<int> hash_length(vertex_count);
    cudaMemcpy(hash_length.data(), d_hash_tables_lens, sizeof(int) * vertex_count, cudaMemcpyDeviceToHost);
    vector<long long> hash_offset(vertex_count + 1);
    cudaMemcpy(hash_offset.data(), d_hash_tables_offset, sizeof(long long) * (vertex_count + 1), cudaMemcpyDeviceToHost);

    // Verify hash table build/search consistency for first few vertices
    printf("\n=== Hash Table Verification ===\n");
    for (int v = 0; v < min(5, vertex_count); v++) {
        int deg = degree_offset[v + 1] - degree_offset[v];
        if (deg == 0) continue;

        printf("\nVertex %d (degree=%d, hash_len=%d):\n", v, deg, hash_length[v]);
        printf("  Original neighbors: ");
        for (int i = degree_offset[v]; i < min(degree_offset[v] + 10, degree_offset[v + 1]); i++) {
            printf("%d ", adjcant[i]);
        }
        printf("\n  Sorted neighbors: ");
        for (int i = degree_offset[v]; i < min(degree_offset[v] + 10, degree_offset[v + 1]); i++) {
            printf("%d ", adjcant_hierarchical[i]);
        }

        // Check if all neighbors can be found in hash table
        printf("\n  Hash verification:\n");
        int found_count = 0;
        int not_found_count = 0;
        for (int i = degree_offset[v]; i < degree_offset[v + 1]; i++) {
            int neighbor = adjcant_hierarchical[i];
            int hash_len = hash_length[v];
            long long hash_start = hash_offset[v];

            // Compute hash value
            int shift = __builtin_ffs(max_length) - __builtin_ffs(hash_len);
            int mapped_value = ((neighbor & (max_length - 1)) >> shift);

            // Search in hash table
            bool found = false;
            for (int slot = 0; slot < bucket_size && !found; slot++) {
                for (int bucket = 0; bucket < hash_len && !found; bucket++) {
                    int idx = hash_start + bucket + slot * bucket_num;
                    if (hashtable_hierarchical[idx] == neighbor) {
                        found = true;
                    }
                }
            }

            if (found) {
                found_count++;
            } else {
                not_found_count++;
                if (not_found_count <= 3) {  // Print first 3 missing neighbors
                    printf("    NOT FOUND: neighbor=%d, hash_val=%d\n", neighbor, mapped_value);
                }
            }
        }
        printf("  Found: %d/%d, Not found: %d\n", found_count, deg, not_found_count);
    }
    printf("=== End Verification ===\n\n");
    */

    // // printf("max_length is : %d\n", max_length);
    // // printf("Original adjcant:\n");
    // // for(int i=0;i<10;i++)
    // // {
    // //     for(int j=degree_offset[i];j<degree_offset[i+1];j++)
    // //     {
    // //         printf("%d ",adjcant[j]);
    // //     }
    // //     printf("\n");
    // // }
    // printf("Hierarchical adjcant:\n");
    // for (int i = 0; i < 5; i++)
    // {
    //     for (int j = degree_offset[i]; j < degree_offset[i + 1]; j++)
    //     {
    //         printf("%d ", adjcant_hierarchical[j]);
    //     }
    //     printf("\n");
    // }

    // printf("Hashtable_hierarchical\n");
    // for (int i = 0; i < 5; i++)
    // {
    //     int length = hash_length[i];
    //     printf("Node %d, length: %d\n", i, length);
    //     for (int j = 0; j < bucket_size; j++)
    //     {
    //         for (int k = 0; k < length; k++)
    //         {
    //             printf("%d ", hashtable_hierarchical[hash_offset[i] + k + j * bucket_num]);
    //         }
    //         printf("\n");
    //     }
    // }

    int *d_ir; // intermediate result;
    // refine the malloc
    HRR(cudaMalloc(&d_ir, (long long)216 * 32 * max_degree * H * sizeof(int)));

    // cout << "ir memory size is : " << 216 * 32 * max_degree * H * sizeof(int) / 1024 / 1024 << "MB" << endl;

    int *d_intersection_orders;
    HRR(cudaMalloc(&d_intersection_orders, intersection_size * sizeof(int)));
    HRR(cudaMemcpy(d_intersection_orders, intersection_orders, intersection_size * sizeof(int), cudaMemcpyHostToDevice));
    int *d_intersection_offset;
    HRR(cudaMalloc(&d_intersection_offset, intersection_size * sizeof(int)));
    HRR(cudaMemcpy(d_intersection_offset, intersection_offset, intersection_size * sizeof(int), cudaMemcpyHostToDevice));
    int *d_restriction;
    HRR(cudaMalloc(&d_restriction, restriction_size * sizeof(int)));
    HRR(cudaMemcpy(d_restriction, restriction, restriction_size * sizeof(int), cudaMemcpyHostToDevice));
    int *d_reuse;
    HRR(cudaMalloc(&d_reuse, H * sizeof(int)));
    HRR(cudaMemcpy(d_reuse, reuse, H * sizeof(int), cudaMemcpyHostToDevice));
    int *G_INDEX;
    HRR(cudaMalloc(&G_INDEX, sizeof(int)));

    unsigned long long *d_sum;
    HRR(cudaMalloc(&d_sum, sizeof(unsigned long long)));
    HRR(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
    // double start_time = wtime();

    double cmp_time;
    double time_start;
    double max_time = 0;
    double min_time = 1000;
    double ave_time = 0;

    long long sum1 = 0;
    long long sum2 = 0;
    int origin_process_id = process_id;
    l2flush();
    cudaDeviceSynchronize();
    time_start = clock();

    // hierarchical
    HRR(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
    for (; process_id < process_num; process_id += deviceCount)
    {
        int temp = block_size * block_number / 32 * chunk_size * process_num + process_id;
        HRR(cudaMemcpy(G_INDEX, &temp, sizeof(int), cudaMemcpyHostToDevice));
        printf("start kernel\n");
        int length;
#ifdef useVertexAsStart
        length = vertex_count;
#else
        length = edge_count;
#endif
        time_start = clock();
        if (pattern.compare("Q5") == 0 || pattern.compare("Q7") == 0 || pattern.compare("Q3") == 0 || pattern.compare("Q0") == 0 || pattern.compare("Q4") == 0 || pattern.compare("Q6") == 0 || pattern.compare("Q8") == 0)
        {
            time_start = clock();
            DFSKernelForClique<<<block_size, block_number>>>(d_reuse, process_id, process_num, chunk_size, length, bucket_size, bucket_num, max_degree, d_intersection_orders, d_intersection_offset, d_restriction, d_degree_offset, d_adjcant_hierarchical, d_hash_tables_offset, d_hash_tables_hierarchical, d_vertex, d_ir, d_sum, G_INDEX, max_length, d_dispersed_mapping);
        }
        else
        {
            time_start = clock();
            DFSKernelForGeneral<<<block_size, block_number>>>(process_id, process_num, chunk_size, length, bucket_size, bucket_num, max_degree, d_intersection_orders, d_intersection_offset, d_restriction, d_degree_offset, d_adjcant_hierarchical, d_hash_tables_offset, d_hash_tables_hierarchical, d_vertex, d_ir, d_sum, G_INDEX, max_length, d_dispersed_mapping);
        }
        HRR(cudaDeviceSynchronize());
        cmp_time = clock() - time_start;
        cmp_time = cmp_time / CLOCKS_PER_SEC;
        if (cmp_time > max_time)
            max_time = cmp_time;
        if (cmp_time < min_time)
            min_time = cmp_time;
        ave_time += cmp_time;
        printf("finish kernel\n");
        // cout << "this time" << cmp_time << ' ' << max_time << endl;
        HRR(cudaFree(d_ir));
        HRR(cudaMalloc(&d_ir, (long long)216 * 32 * max_degree * H * sizeof(int)));

        cudaMemcpy(&sum2, d_sum, sizeof(long long), cudaMemcpyDeviceToHost);
        std::cout << "Hierarchical time: " << cmp_time * 1000 << " ms" << std::endl;
    }

    HRR(cudaFree(d_hash_tables_offset));
    HRR(cudaFree(d_adjcant));
    HRR(cudaFree(d_vertex));
    HRR(cudaFree(d_degree_offset));
    HRR(cudaFree(d_ir));
    HRR(cudaFree(d_intersection_orders));
    HRR(cudaFree(d_intersection_offset));
    HRR(cudaFree(d_restriction));
    HRR(cudaFree(d_reuse));
    HRR(cudaFree(G_INDEX));

    // Note: sum1 is 0 because we only run the hierarchical version
    // To compare with original version, run both and set sum1 accordingly
    // if (sum1 != sum2)
    //     std::cout << "Error: Normal sum " << sum1 << " Hierarchical sum " << sum2 << std::endl;
    std::cout << pattern << " count is " << sum2 << std::endl;

    args.time = max_time;
    args.count = sum2;
    return args;
}
