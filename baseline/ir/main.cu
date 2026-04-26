#include <cuda_runtime.h>

#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "../common/graph_data.cuh"
#include "../common/utils.cuh"
#include "ir.cuh"
#include "ir_cuco.cuh"

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

  std::string input_folder = argv[1];
  float load_factor = 0.2;
  int bucket_size = 5;

  bool run_original = true;
  bool run_cuco = false;

  // ==================== 解析参数 ====================
  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.rfind("--alpha=", 0) == 0) {
      load_factor = std::stof(arg.substr(8));
    } else if (arg.rfind("--bucket=", 0) == 0) {
      bucket_size = std::stoi(arg.substr(9));
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

  // ==================== 加载倒排索引 ====================
  std::string inverted_index_offsets_path =
      input_folder + "/inverted_index_offsets.bin";
  std::string inverted_index_path = input_folder + "/inverted_index.bin";

  InvertedIndex index(inverted_index_offsets_path, inverted_index_path);

  std::cout << "成功加载数据:" << std::endl;
  std::cout << "倒排索引偏移量大小: " << index.get_host_offsets().size()
            << std::endl;
  std::cout << "倒排索引大小: " << index.get_host_elements().size()
            << std::endl;
  std::cout << "词项数量: " << index.get_num_nodes() << std::endl;

  // 最大度数
  int max_degree = 0;
  for (int i = 0; i < index.get_num_nodes(); i++) {
    max_degree = std::max(max_degree, (int)(index.get_host_offsets()[i + 1] -
                                            index.get_host_offsets()[i]));
  }
  printf("最大度数: %d\n", max_degree);
  // ==================== 构建哈希 ====================
  IRBaseline baseline;
  baseline.build_hash_tables(index.get_num_nodes(), index.get_host_offsets(),
                             index.get_host_elements(), load_factor,
                             bucket_size);

  // 加载查询
  std::string query_path = input_folder + "/query.bin";
  std::string query_offsets_path = input_folder + "/query_offsets.bin";
  std::string query_num_path = input_folder + "/query_num.bin";

  baseline.load_queries(query_path, query_offsets_path, query_num_path);

  baseline.allocate_result_buffers(index);

  // ==================== Kernel配置 ====================
  int CHUNK_SIZE = 2;
  int grid_size = 512;
  int block_size = 1024;

  check_gpu_memory();

  l2flush flush;

  // ==================== ORIGINAL ====================
  if (run_original) {
    // 分层哈希
    flush.flush();
    double t0 = clock();
    int result_hierarchical = baseline.run_hierarchical(
        CHUNK_SIZE, grid_size, block_size, bucket_size, false);
    double secs = (clock() - t0) / CLOCKS_PER_SEC;

    printf("[RESET] 内核执行时间: %.6f 秒\n", secs);
    printf("[RESET] 信息相似度结果数: %d\n", result_hierarchical);

    // 普通哈希
    flush.flush();
    t0 = clock();
    int result_normal = baseline.run_normal(CHUNK_SIZE, grid_size, block_size,
                                            bucket_size, false);
    secs = (clock() - t0) / CLOCKS_PER_SEC;

    printf("[Native] 内核执行时间: %.6f 秒\n", secs);
    printf("[Native] 信息相似度结果数: %d\n", result_normal);
  }

  // ==================== CUCO ====================
  if (run_cuco) {
    flush.flush();
    double t0 = clock();

    run_ir_cuco(index.get_num_nodes(), baseline.get_query_num(),
                index.get_device_elements(), index.get_device_offsets(),
                baseline.get_d_query(), baseline.get_d_query_offsets(),
                baseline.get_d_result(), baseline.get_d_result_offsets(),
                baseline.get_d_result_count(), baseline.get_d_G_index(),
                CHUNK_SIZE, index.get_host_offsets(), index.get_host_elements(),
                grid_size, block_size, load_factor, 0);

    double secs = (clock() - t0) / CLOCKS_PER_SEC;
    printf("[cuCollections] 内核执行时间: %.6f 秒\n", secs);
  }

  return 0;
}