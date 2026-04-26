#include <iostream>
#include <vector>
#include <ctime>
#include <tuple>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <climits>
#include <fstream>
using namespace std;

#define LOAD_FACTOR 0.2
#define BUCKET_SIZE 5

#define GRID_SIZE 216
#define BLOCK_SIZE 1024
#define WARP_SIZE 32
#define NUM_WARP (GRID_SIZE * BLOCK_SIZE / WARP_SIZE)

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

int *generate_random_array(int size, int range)
{
    std::random_device rd;                                     
    std::mt19937 generator(rd());                              
    std::uniform_int_distribution<int> distribution(0, range); 

    int *randomArray = new int[size];

    for (int i = 0; i < size; ++i)
    {
        randomArray[i] = distribution(generator); 
    }
    return randomArray; 
}

int calculate_length(int degrees, float load_factor, int bucket_size)
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

bool cmp(int a, int b)
{
    return a < b;
}

struct HashModCompare
{
    int hash_length;
    HashModCompare(int h) : hash_length(h) {}
    bool operator()(int a, int b) const
    {
        return (a % hash_length) < (b % hash_length);
    }
};

__device__ __forceinline__ int d_hash(const int x, int length)
{
    return x & (length - 1);
}

__device__ bool device_hash_search(int w, int *hashtable, int len, int bucket, int bucket_num)
{
    while (1)
    {
        for (int i = 0; i < BUCKET_SIZE; i++)
        {
            int idx = i * bucket_num + bucket;
            int value = hashtable[idx];
            if (value == w)
                return true;
            if (value == -1)
                return false;
        }
        bucket = (bucket + 1) & (len - 1);
    }
    return false;
}

__global__ void make_hashtable_kernel(int* data,int data_size,int*hashtable,int bucket_num,int hash_length)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_tid = threadIdx.x % WARP_SIZE;
    if (warp_id >= NUM_WARP) return;
    int *hashtable_start = hashtable + warp_id * hash_length;
    for (int i = warp_tid; i < data_size;i+=WARP_SIZE)
    {
        int key=data[i];
        int bucket = d_hash(key, hash_length);
        int k = 0;
        while (atomicCAS(&hashtable_start[bucket + k * bucket_num], -1, key) != -1)
        {
            k++;
            if (k == BUCKET_SIZE)
            {
                k = 0;
                bucket = (bucket + 1) & (hash_length - 1);
            }
        }
    }
}

__global__ void intersection_kernel(int*a,int a_size,int*hashtable,int hash_length,int*result_count,int bucket_num)
{
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_tid = threadIdx.x % WARP_SIZE;
    if (warp_id >= NUM_WARP)
        return;
    int *hashtable_start = hashtable + warp_id * hash_length;
    int *a_start = a + warp_id * a_size;
    int local_count = 0;
    for (int i = warp_tid; i < a_size; i += WARP_SIZE)
    {
        int key = a_start[i];
        int bucket = d_hash(key, hash_length);
        if (device_hash_search(key, hashtable_start, hash_length, bucket, bucket_num))
        {
            local_count++;
        }
    }

    if(warp_id==0)
    {
        atomicAdd(result_count, local_count);
    }

}

int main(int argc, char *argv[])
{
    if(argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <set A size> <set B size>" << std::endl;
        return 1;
    }

    int a_size = std::atoi(argv[1]);
    int b_size = std::atoi(argv[2]);


    int *a = generate_random_array(a_size, INT_MAX);
    int *b = generate_random_array(b_size, INT_MAX);

    if (a_size > b_size)
    {
        std::swap(a, b);
        std::swap(a_size, b_size);
    }
    vector<int> b_vec(b_size );
    memcpy(b_vec.data(), b, sizeof(int) * b_size);
    int max_length = (int)(b_size / LOAD_FACTOR / BUCKET_SIZE);
    int log = log2f(max_length);
    max_length = powf(2, log) == max_length ? max_length : powf(2, log + 1);
    // printf("Hashtable length: %d\n", max_length);

    // for (int i = 0;i<b_size;i++)
    // {
    //     printf("%d ", b_vec[i]);
    // }
    // printf("\n");

    sort(a, a + a_size, cmp);
    // printf("finish sort array a with cmp\n");
    vector<int> array_normal(a_size * NUM_WARP);
    for (int i = 0; i < NUM_WARP; i++)
    {
        memcpy(&array_normal[i * a_size], a, a_size * sizeof(int));
    }

    sort(a,a+a_size,HashModCompare(max_length));
    vector<int> array_opt(a_size * NUM_WARP);
    for (int i = 0; i < NUM_WARP; i++)
    {
        memcpy(&array_opt[i * a_size], a, a_size * sizeof(int));
    }
    // printf("finish sort array a with HashModCompare\n");


    int bucket_num = max_length * NUM_WARP;
    int *d_hashtable;
    cudaMalloc(&d_hashtable, sizeof(int) * bucket_num * BUCKET_SIZE);
    cudaMemset(d_hashtable, -1, sizeof(int) * bucket_num * BUCKET_SIZE);
    // int *d_a;
    // cudaMalloc(&d_a, sizeof(int) * a_size * NUM_WARP);
    // cudaMemcpy(d_a, array.data(), sizeof(int) * a_size * NUM_WARP, cudaMemcpyHostToDevice);
    int *d_b;
    cudaMalloc(&d_b, sizeof(int) * b_size );
    cudaMemcpy(d_b, b_vec.data(), sizeof(int) * b_size, cudaMemcpyHostToDevice);

    // for (int i = 0; i < 16;i++)
    // {
    //     printf("%d ", b_vec[i]);
    // }
    // printf("\n");

    make_hashtable_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_b, b_size, d_hashtable, bucket_num, max_length);


    int *d_array_normal;
    int* d_array_opt;
    cudaMalloc(&d_array_normal, sizeof(int) * a_size * NUM_WARP);
    cudaMalloc(&d_array_opt, sizeof(int) * a_size * NUM_WARP);
    cudaMemcpy(d_array_normal, array_normal.data(), sizeof(int) * a_size * NUM_WARP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_array_opt, array_opt.data(), sizeof(int) * a_size * NUM_WARP, cudaMemcpyHostToDevice);
    int *d_result_count;
    cudaMalloc(&d_result_count, sizeof(int));
    cudaMemset(d_result_count, 0, sizeof(int));

    l2flush();
    cudaDeviceSynchronize();
    double clock_start = clock();
    intersection_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_array_normal, a_size, d_hashtable, max_length, d_result_count, bucket_num);
    cudaDeviceSynchronize();
    double clock_end = clock();
    double time_normal = (clock_end - clock_start) / CLOCKS_PER_SEC;
    int h_result_count_normal;
    cudaMemcpy(&h_result_count_normal, d_result_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Normal Intersection Time: %f ms\n", time_normal*1000);

    cudaMemset(d_result_count, 0, sizeof(int));
    l2flush();
    cudaDeviceSynchronize();
    clock_start = clock();
    intersection_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_array_opt, a_size, d_hashtable, max_length , d_result_count, bucket_num);
    cudaDeviceSynchronize();
    clock_end = clock();
    double time_opt = (clock_end - clock_start) / CLOCKS_PER_SEC;
    int h_result_count_opt;
    cudaMemcpy(&h_result_count_opt, d_result_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Optimized Intersection Time: %f ms\n", time_opt*1000);

    if(h_result_count_normal != h_result_count_opt)
    {
        std::cerr << "Mismatch in intersection results!" << std::endl;
    }
    else
    {
        printf("Intersection Result Count: %d\n", h_result_count_normal);
    }

    return 0;

}

