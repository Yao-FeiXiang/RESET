#include "sss.cuh"

vector<int> vertexs;
vector<int> csr_cols;
vector<int> csr_offsets;
int num_nodes;
int num_edges;

vector<int> read_vec_32(const string& filename) {
  ifstream in(filename, ios::binary);
  if (!in) {
    cerr << "Error opening file: " << filename << endl;
    exit(1);
  }

  size_t size;
  in.read(reinterpret_cast<char*>(&size), sizeof(size_t));

  vector<int> vec(size);
  in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(int32_t));

  return vec;
}

vector<int> read_vec_64(const string& filename) {
  ifstream in(filename, ios::binary);
  if (!in) {
    cerr << "Error opening file: " << filename << endl;
    exit(1);
  }

  size_t size;
  in.read(reinterpret_cast<char*>(&size), sizeof(size_t));

  vector<int64_t> tmp(size);
  in.read(reinterpret_cast<char*>(tmp.data()), size * sizeof(int64_t));

  vector<int> vec(size);
  for (size_t i = 0; i < size; ++i) {
    vec[i] = static_cast<int>(tmp[i]);
  }

  return vec;
}

int read_num_64(const string& filename) {
  ifstream in(filename, ios::binary);
  if (!in) {
    cerr << "Error opening file: " << filename << endl;
    exit(1);
  }

  int64_t value;
  in.read(reinterpret_cast<char*>(&value), sizeof(int64_t));

  return static_cast<int>(value);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <input_folder>" << endl;
    return 1;
  }

  int dev = 1;  // 选择物理 GPU 编号 1
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  printf("Found %d devices\n", deviceCount);
  if (dev >= deviceCount) {
    fprintf(stderr, "Device %d out of range\n", dev);
    return 1;
  }
  cudaError_t e = cudaSetDevice(dev);
  if (e != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", dev,
            cudaGetErrorString(e));
    return 1;
  }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  printf("Using device %d: %s\n", dev, prop.name);

  float load_factor = 0.2;
  int bucket_size = 5;
  int total_device = 4;
  bool opt = true;
  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.rfind("--alpha=", 0) == 0) {
      load_factor = std::stof(arg.substr(8));
    } else if (arg.rfind("--bucket=", 0) == 0) {
      bucket_size = std::stoi(arg.substr(9));
    } else if (arg.rfind("--total_device=", 0) == 0) {
      total_device = std::stoi(arg.substr(15));
    } else if (arg == "--normal") {
      opt = false;
    } else {
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

  vertexs = read_vec_32(vertexs_path);
  csr_cols = read_vec_32(csr_cols_path);
  csr_offsets = read_vec_64(csr_offsets_path);
  num_edges = read_num_64(num_edges_path);
  num_nodes = read_num_64(nums_nodes_path);
  vector<int> adjcents = csr_cols;
  printf("finish reading input files\n");

  unsigned long long total_count = 0;
  vector<int> total_result(num_edges, 0);
  double max_kernel_time = 0.0;
  double start_time = clock();
  for (int dev = 0; dev < total_device; dev++) {
    vector<int> local_csr_cols;
    vector<int> local_csr_rows(num_nodes + 1, 0);
    for (int i = 0; i < num_nodes; i++) {
      int temp_count = 0;
      for (int j = csr_offsets[i]; j < csr_offsets[i + 1]; j++) {
        if (csr_cols[j] % total_device == dev) {
          local_csr_cols.push_back(csr_cols[j]);
          temp_count++;
        }
      }
      local_csr_rows[i + 1] = local_csr_rows[i] + temp_count;
    }

    double kernel_time = 0.0;
    vector<int> result = partial_sss(
        vertexs, adjcents, local_csr_cols, local_csr_rows, num_nodes, num_edges,
        kernel_time, load_factor, bucket_size, dev, opt);
    for (int i = 0; i < num_edges; i++) total_result[i] += result[i];
    if (kernel_time > max_kernel_time) max_kernel_time = kernel_time;
  }
  for (int i = 0; i < num_edges; i++) {
    int u = vertexs[i];
    int v = csr_cols[i];
    int u_size = csr_offsets[u + 1] - csr_offsets[u];
    int v_size = csr_offsets[v + 1] - csr_offsets[v];
    if (total_result[i] / (float)(u_size + v_size - total_result[i]) >=
        threshold)
      total_count++;
  }
  printf("Total count: %llu\n", total_count);
  printf("Max kernel time: %f s\n", max_kernel_time);
  printf("Total process time: %f s\n", (clock() - start_time) / CLOCKS_PER_SEC);
  return 0;
}