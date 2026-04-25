#include "tc.cuh"

vector<int> vertexs;
vector<int> csr_cols;
vector<int> csr_offsets;
int num_nodes;
int num_edges;

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

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <input_folder>" << endl;
        return 1;
    }

    int dev = 0; // 选择物理 GPU 编号 1
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

    float load_factor = 0.25;
    int bucket_size = 4;
    int total_device = 1;

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
        else if (arg.rfind("--total_device=", 0) == 0)
        {
            total_device = std::stoi(arg.substr(15));
        }
        else
        {
            cerr << "Unknown argument: " << arg << endl;
            return 1;
        }
    }
    printf("load_factor:%.2f , bucket_size:%d\n", load_factor, bucket_size);

    string input_folder = argv[1];
    string vertexs_path = input_folder + "/vertexs.bin";
    string csr_cols_path = input_folder + "/csr_cols.bin";
    string csr_offsets_path = input_folder + "/csr_offsets.bin";
    string num_edges_path = input_folder + "/num_edges.bin";
    string nums_nodes_path = input_folder + "/num_nodes.bin";

    vertexs = read_vector_binary(vertexs_path);
    csr_cols = read_vector_binary(csr_cols_path);
    csr_offsets = read_vector_binary(csr_offsets_path);
    num_edges = read_int_binary(num_edges_path);
    num_nodes = read_int_binary(nums_nodes_path);

    vector<int> adjcents = csr_cols;

    unsigned long long total_count = 0;
    double max_kernel_time = 0.0;
    for (int dev = 0; dev < total_device;dev++)
    {
        vector<int> local_csr_cols;
        vector<int> local_csr_rows(num_nodes+1,0);
        for (int i = 0; i < num_nodes;i++)
        {
            int temp_count = 0;
            for(int j = csr_offsets[i]; j < csr_offsets[i+1];j++)
            {
                if(csr_cols[j] % total_device == dev)
                {
                    local_csr_cols.push_back(csr_cols[j]);
                    temp_count++;
                }
            }
            local_csr_rows[i+1] = local_csr_rows[i] + temp_count;
        }
        //printf("Device %d: local_csr_cols size: %lu\n", dev, local_csr_cols.size());
        double kernel_time = 0.0;
        unsigned long long count = partial_tc(vertexs, adjcents, local_csr_cols, local_csr_rows, num_nodes, num_edges, kernel_time, load_factor, bucket_size);
        total_count += count;
        if (kernel_time > max_kernel_time)
            max_kernel_time = kernel_time;
    }
    printf("Total triangle count: %llu\n", total_count / 6);
    printf("Max kernel time: %f s\n", max_kernel_time);
    return 0;
}