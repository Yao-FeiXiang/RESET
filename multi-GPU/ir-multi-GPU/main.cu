#include "ir.cuh"

vector<int> inverted_index_offsets;
vector<int> inverted_index;
int inverted_index_num = 0;
vector<int> global_query;
vector<int> query_offsets;
int query_num = 0;

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

    // cout << "Successfully loaded data:" << endl;
    // cout << "inverted_index_offsets size: " << inverted_index_offsets.size() << endl;
    // cout << "inverted_index size: " << inverted_index.size() << endl;
    // cout << "inverted_index_num: " << inverted_index_num << endl;
    // cout << "query size: " << global_query.size() << endl;
    // cout << "query_offsets size: " << query_offsets.size() << endl;
    // cout << "query_num: " << query_num << endl;


    int total_count = 0;
    double max_kernel_time = 0.0;
    for (int dev = 0; dev < total_device; dev++)
    {
        // split inverted index
        vector<int> local_inverted_index_offsets(inverted_index_offsets.size(), 0);
        vector<int> local_inverted_index;
        for (int i = 0; i < inverted_index_offsets.size()-1; i++)
        {
            int temp_count = 0;
            for (int j = inverted_index_offsets[i]; j < inverted_index_offsets[i + 1]; j++)
            {
                if (inverted_index[j] % total_device == dev)
                {
                    local_inverted_index.push_back(inverted_index[j]);
                    temp_count++;
                }
            }
            local_inverted_index_offsets[i + 1] = local_inverted_index_offsets[i] + temp_count;
        }
        //printf("Device %d: local_inverted_index size: %lu\n", dev, local_inverted_index.size());
        double kernel_time = 0;
        // for (int i = 0; i < 50; i++)
        // {
        //     printf("%d ", local_inverted_index_offsets[i]);
        // }
        // printf("\n");
        // for (int i = 0; i < 50; i++)
        // {
        //     printf("%d ", local_inverted_index[i]);
        // }
        // printf("\n");
        int sum = partial_ir(local_inverted_index_offsets, local_inverted_index, global_query, query_offsets, query_num, kernel_time, local_inverted_index_offsets.size() - 1, load_factor, bucket_size);
        max_kernel_time = max(max_kernel_time, kernel_time);
        total_count += sum;
    }
    cout << "Total matched document count: " << total_count << endl;
    printf("Max kernel time: %f s\n", max_kernel_time);
    return 0;
}