#include "utils.cuh"

int h_hash_normal(int x, int length) { return x % length; }

int h_hash_hierarchical(int x, int length, int max_length) {
  return x % max_length / (max_length / length);
}

void read_i32_vec(const std::string& path, std::vector<int>& vec) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + path);
  }
  size_t size;
  file.read((char*)&size, sizeof(size_t));
  vec.resize(size);
  file.read((char*)vec.data(), size * sizeof(int));
}

int* read_int_binary(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + path);
  }
  size_t size = file.tellg() / sizeof(int);
  int* h_ptr = new int[size];
  file.seekg(0, std::ios::beg);
  file.read((char*)h_ptr, size * sizeof(int));

  int* d_ptr;
  cudaMalloc(&d_ptr, size * sizeof(int));
  cudaMemcpy(d_ptr, h_ptr, size * sizeof(int), cudaMemcpyHostToDevice);
  delete[] h_ptr;
  return d_ptr;
}

void check_gpu_memory(int min_required_mib) {
  // 调用nvidia-smi获取内存信息
  FILE* pipe = popen(
      "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", "r");
  if (!pipe) {
    fprintf(stderr, "警告: 无法执行nvidia-smi检查GPU内存\n");
    return;
  }

  int current_device;
  cudaGetDevice(&current_device);

  char buffer[128];
  int free_memory_mib = 0;
  int device_idx = 0;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    if (device_idx == current_device) {
      sscanf(buffer, "%d", &free_memory_mib);
      break;
    }
    device_idx++;
  }
  pclose(pipe);

  if (free_memory_mib == 0) {
    fprintf(stderr, "警告: 无法获取GPU%d空闲内存信息\n", current_device);
    return;
  }

  if (free_memory_mib < min_required_mib) {
    fprintf(stderr, "错误: GPU%d空闲内存不足\n", current_device);
    fprintf(stderr, "需要至少: %d MiB,实际剩余: %d MiB\n", min_required_mib,
            free_memory_mib);
    exit(EXIT_FAILURE);
  }

  printf("GPU%d内存检查通过\n", current_device);
}
