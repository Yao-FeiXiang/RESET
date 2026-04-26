#include <cuda_runtime.h>

#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../common/graph_data.cuh"
#include "../common/utils.cuh"
#include "tc.cuh"
#include "tc_cuco.cuh"

// 默认使用的GPU设备编号
#define DEV 0

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "用法: " << argv[0] << " <输入文件夹>" << std::endl;
    return 1;
  }

  int dev = DEV;
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  printf("找到 %d 个设备\n", deviceCount);
  if (dev >= deviceCount) {
    fprintf(stderr, "设备 %d 超出范围\n", dev);
    return 1;
  }

  cudaError_t e = cudaSetDevice(dev);
  if (e != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice(%d) 失败: %s\n", dev, cudaGetErrorString(e));
    return 1;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  printf("使用设备 %d: %s\n", dev, prop.name);

  float load_factor = 0.2;
  int bucket_size = 5;

  bool run_original = true;
  bool run_cuco = true;

  // ==================== 参数解析 ====================
  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.rfind("--alpha=", 0) == 0) {
      load_factor = std::stof(arg.substr(8));
    } else if (arg.rfind("--bucket=", 0) == 0) {
      bucket_size = std::stoi(arg.substr(9));
    } else if (arg == "--method=original") {
      run_original = true;
      run_cuco = false;
    } else if (arg == "--method=cuco") {
      run_original = false;
      run_cuco = true;
    } else {
      std::cerr << "未知参数: " << arg << std::endl;
      std::cerr << "可用方法: --method=[original|cuco]" << std::endl;
      return 1;
    }
  }

  printf("负载因子: %.2f , 桶大小: %d , 运行方法: %s%s\n", load_factor,
         bucket_size, run_original ? "original " : "", run_cuco ? "cuco" : "");

  // ==================== 数据加载 ====================
  std::string input_folder = argv[1];
  std::string csr_cols_path = input_folder + "/csr_cols_tri.bin";
  std::string csr_offsets_path = input_folder + "/csr_offsets_tri.bin";
  std::string num_nodes_path = input_folder + "/num_nodes.bin";

  std::ifstream in_num_nodes(num_nodes_path, std::ios::binary);
  int num_nodes;
  in_num_nodes.read(reinterpret_cast<char*>(&num_nodes), sizeof(int));

  CSRGraph graph(csr_offsets_path, csr_cols_path);

  std::cout << "图加载完成: " << num_nodes << " 个顶点, "
            << graph.get_num_elements() << " 条边" << std::endl;

  int max_degree = 0;
  for (int i = 0; i < graph.get_num_nodes(); i++) {
    int degree = graph.get_host_offsets()[i + 1] - graph.get_host_offsets()[i];
    if (degree > max_degree) max_degree = degree;
  }
  printf("最大度数: %d\n", max_degree);

  // ==================== 构建哈希 ====================
  TCBaseline baseline;
  baseline.build_hash_tables(graph.get_num_nodes(), graph.get_host_offsets(),
                             graph.get_host_elements(), load_factor,
                             bucket_size);

  baseline.prepare_vertex_list(graph);
  baseline.allocate_buffers();

  // ==================== Kernel配置 ====================
  const int block_size = 1024;
  const int grid_size = 1024;
  const int CHUNK_SIZE = 4;

  check_gpu_memory();

  l2flush flush;

  // ==================== ORIGINAL ====================
  if (run_original) {
    // 普通哈希
    flush.flush();
    cudaDeviceSynchronize();
    double time_start = clock();

    unsigned long long result_normal = baseline.run_normal(
        graph, CHUNK_SIZE, grid_size, block_size, bucket_size, false);

    double cmp_time = (clock() - time_start) / CLOCKS_PER_SEC;

    printf("[Native] 内核执行时间: %.6f 秒\n", cmp_time);
    printf("三角形数量: %llu\n", result_normal);

    // 分层哈希
    flush.flush();
    cudaDeviceSynchronize();
    time_start = clock();

    unsigned long long result_hierarchical = baseline.run_hierarchical(
        graph, CHUNK_SIZE, grid_size, block_size, bucket_size, true);

    cmp_time = (clock() - time_start) / CLOCKS_PER_SEC;

    printf("[RESET] 内核执行时间: %.6f 秒\n", cmp_time);
    printf("三角形数量: %llu\n", result_hierarchical);
  }

  // ==================== CUCO ====================
  if (run_cuco) {
    flush.flush();
    cudaDeviceSynchronize();
    double time_start = clock();

    unsigned long long result_cuco = run_tc_cuco(
        num_nodes, graph.get_num_elements(), baseline.get_d_vertex_list(),
        graph.get_device_offsets(), graph.get_device_elements(),
        graph.get_host_offsets(), graph.get_host_elements(), grid_size,
        block_size, CHUNK_SIZE, load_factor, 0);

    double cmp_time = (clock() - time_start) / CLOCKS_PER_SEC;

    printf("[cuCollections] 内核执行时间: %.6f 秒\n", cmp_time);
    printf("三角形数量: %llu\n", result_cuco);
  }

  return 0;
}