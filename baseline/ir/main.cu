#include <cuda_runtime.h>

#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "../common/graph_data.cuh"
#include "../common/utils.cuh"
#include "ir.cuh"
#include "ir_cuco.cuh"

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

  // 显式激活CUDA上下文，确保后续CUDA操作（包括cuCollections）在正确设备上
  cudaFree(0);
  cudaDeviceSynchronize();

  std::string input_folder = argv[1];
  float load_factor = 0.2;
  int bucket_size = 5;

  bool run_original = true;
  bool run_cuco = true;

  // ==================== 解析参数 ====================
  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.rfind("--alpha=", 0) == 0) {
      load_factor = std::stof(arg.substr(8));
    } else if (arg.rfind("--bucket=", 0) == 0) {
      bucket_size = std::stoi(arg.substr(9));
    } else if (arg.rfind("--method=", 0) == 0) {
      // 重置所有方法为 false
      run_original = false;
      run_cuco = false;

      // 解析逗号分隔的方法列表
      std::vector<std::string> methods = parse_methods(arg.substr(9));
      if (methods.empty()) {
        run_original = true;
      }

      for (const auto& method : methods) {
        if (method == "original") {
          run_original = true;
        } else if (method == "cuco") {
          run_cuco = true;
        } else {
          std::cerr << "未知方法: " << method << std::endl;
          std::cerr << "可用方法: original, cuco" << std::endl;
          return 1;
        }
      }
    } else {
      std::cerr << "未知参数: " << arg << std::endl;
      std::cerr << "用法: " << argv[0]
                << " <输入文件夹> [--alpha=负载因子] [--bucket=桶大小] "
                   "[--method=original,cuco]"
                << std::endl;
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

  // 估算所需内存并检查
  int num_nodes = index.get_num_nodes();
  size_t num_elements = index.get_host_elements().size();

  // 估算哈希表大小 (每个哈希表: bucket_num * bucket_size * 4 bytes)
  float avg_degree = (float)num_elements / num_nodes;
  size_t estimated_buckets =
      (size_t)(num_nodes * avg_degree / load_factor / bucket_size);
  size_t hash_table_bytes =
      estimated_buckets * bucket_size * 4 * 2;  // 两个哈希表

  printf("估算哈希表内存: %.2f GB\n",
         hash_table_bytes / (1024.0 * 1024.0 * 1024.0));

  check_gpu_memory();

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

  // 确保CUDA设备上下文已激活（在任何CUDA对象创建之前）
  cudaDeviceSynchronize();

  l2flush flush;

  // ==================== ORIGINAL ====================
  if (run_original) {
    // 普通哈希 (不需要排序)
    flush.flush();
    auto [result_normal, kernel_time_normal] = baseline.run_normal(
        CHUNK_SIZE, grid_size, block_size, bucket_size, false);

    printf("[Native] 内核执行时间: %.6f 秒\n", kernel_time_normal / 1000.0);
    printf("[Native] 信息相似度结果数: %d\n", result_normal);

    baseline.pre_sort_inverted_index(index, bucket_size);

    flush.flush();
    auto [result_hierarchical, kernel_time_hierarchical] =
        baseline.run_hierarchical(CHUNK_SIZE, grid_size, block_size,
                                  bucket_size, true);

    printf("[RESET] 内核执行时间: %.6f 秒\n",
           kernel_time_hierarchical / 1000.0);
    printf("[RESET] 信息相似度结果数: %d\n", result_hierarchical);
  }

  // ==================== CUCO ====================
  if (run_cuco) {
    baseline.free_hash_tables();
    flush.flush();

    // 重置结果缓冲区和G_index
    int h_G_index = grid_size * block_size / 32 * CHUNK_SIZE;
    cudaMemcpy(baseline.get_d_G_index(), &h_G_index, sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemset(baseline.get_d_result_count(), 0,
               baseline.get_query_num() * sizeof(int));

    cudaDeviceSynchronize();

    auto [result_cuco, kernel_time_cuco] = run_ir_cuco(
        index.get_num_nodes(), baseline.get_query_num(),
        index.get_device_elements(), index.get_device_offsets(),
        baseline.get_d_query(), baseline.get_d_query_offsets(),
        baseline.get_d_result(), baseline.get_d_result_offsets(),
        baseline.get_d_result_count(), baseline.get_d_G_index(), CHUNK_SIZE,
        index.get_host_offsets(), index.get_host_elements(), grid_size,
        block_size, load_factor, 0);

    printf("[cuCollections] 内核执行时间: %.6f 秒\n",
           kernel_time_cuco / 1000.0);
    printf("[cuCollections] 信息相似度结果数: %d\n", result_cuco);
  }

  return 0;
}