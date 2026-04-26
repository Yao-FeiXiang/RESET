#include <cuda_runtime.h>

#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "../common/cuckoo_hash.cuh"
#include "../common/graph_data.cuh"
#include "../common/hopscotch_hash.cuh"
#include "../common/roaring_bitmap.cuh"
#include "../common/utils.cuh"
#include "sss.cuh"
#include "sss_baselines.h"
#include "sss_cuco.cuh"

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "用法: " << argv[0] << " <输入文件夹>" << std::endl;
    return 1;
  }

  std::string input_folder = argv[1];
  float load_factor = 0.2;  // 负载因子
  int bucket_size = 5;      // 每个位置可存储的桶数量

  // 运行控制标志
  bool run_original = true;
  bool run_cucollections = false;
  bool run_cuckoo = false;
  bool run_hopscotch = false;
  bool run_roaring = false;

  // 解析命令行参数
  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.rfind("--alpha=", 0) == 0) {
      load_factor = std::stof(arg.substr(8));
    } else if (arg.rfind("--bucket=", 0) == 0) {
      bucket_size = std::stoi(arg.substr(9));
    } else if (arg == "--method=cuco") {
      run_cucollections = true;
    } else if (arg == "--method=cuckoo") {
      run_cuckoo = true;
    } else if (arg == "--method=hopscotch") {
      run_hopscotch = true;
    } else if (arg == "--method=roaring") {
      run_roaring = true;
    } else if (arg == "--method=all") {
      run_original = true;
      run_cucollections = true;
      run_cuckoo = true;
      run_hopscotch = true;
      run_roaring = true;
    } else {
      std::cerr << "未知参数: " << arg << std::endl;
      std::cerr << "可用方法: --method=[all|cuco|cuckoo|hopscotch|roaring]"
                << std::endl;
      return 1;
    }
  }

  int dev = 0;
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

  printf("负载因子: %.2f , 桶大小: %d , 运行方法:", load_factor, bucket_size);
  if (run_original) printf(" original");
  if (run_cucollections) printf(" cuco");
  if (run_cuckoo) printf(" cuckoo");
  if (run_hopscotch) printf(" hopscotch");
  if (run_roaring) printf(" roaring");
  printf("\n");

  // ==================== 加载CSR图 ====================
  std::string csr_cols_path = input_folder + "/csr_cols.bin";
  std::string csr_offsets_path = input_folder + "/csr_offsets.bin";
  std::string vertexs_path = input_folder + "/vertexs.bin";

  CSRGraph graph(csr_offsets_path, csr_cols_path);

  std::cout << "成功加载数据:" << std::endl;
  std::cout << "顶点数量: " << graph.get_num_nodes() << std::endl;
  std::cout << "边数量: " << graph.get_num_elements() << std::endl;

  // 计算最大度数
  int max_degree = 0;
  for (int i = 0; i < graph.get_num_nodes(); i++) {
    int degree = graph.get_host_offsets()[i + 1] - graph.get_host_offsets()[i];
    if (degree > max_degree) max_degree = degree;
  }
  printf("最大度数: %d\n", max_degree);

  // ==================== 构建哈希表 ====================
  // 使用完整邻居列表构建哈希表(基线方法的标准做法)
  // 注意: 与set-similarity-search的差异在于排序和计时范围,已对齐
  SSSBaseline baseline;
  baseline.build_hash_tables(graph.get_num_nodes(), graph.get_host_offsets(),
                             graph.get_host_elements(), load_factor,
                             bucket_size);

  // 加载顶点对到设备
  baseline.load_vertex_pairs(vertexs_path);
  baseline.allocate_buffers();

  // 内核配置参数
  const int grid_size = 512;
  const int block_size = 1024;
  const int CHUNK_SIZE = 4;
  const float threshold = 0.25;

  check_gpu_memory();

  // 预排序CSR列(用于hierarchical哈希),在计时前完成(与set-similarity-search一致)
  if (run_original) {
    baseline.pre_sort_csr_cols(graph);
  }

  // 清空L2缓存(保证公平)
  l2flush flush;

  // ==================== original ====================
  if (run_original) {
    // 普通哈希
    std::cout << std::endl;

    flush.flush();
    auto [result_normal, kernel_time_normal] =
        baseline.run_normal(graph, CHUNK_SIZE, grid_size, block_size,
                            bucket_size, threshold, false);

    std::cout << "[Native] 内核执行时间: " << kernel_time_normal / 1000.0
              << " 秒" << std::endl;
    printf("[Native] 集合相似度结果数: %d\n", result_normal);

    // 分层哈希
    std::cout << std::endl;

    flush.flush();
    auto [result_hierarchical, kernel_time_hierarchical] =
        baseline.run_hierarchical(graph, CHUNK_SIZE, grid_size, block_size,
                                  bucket_size, threshold, true);

    std::cout << "[RESET] 内核执行时间: " << kernel_time_hierarchical / 1000.0
              << " 秒" << std::endl;
    printf("[RESET] 集合相似度结果数: %d\n", result_hierarchical);

    // 一致性检查
    if (result_normal != result_hierarchical) {
      printf("警告: 不同哈希类型的结果不一致!\n");
      printf("  普通哈希: %d, 分层哈希: %d\n", result_normal,
             result_hierarchical);
    }
  }

  // ==================== cuco ====================
  if (run_cucollections) {
    std::cout << std::endl;
    flush.flush();

    auto [result_cuco, kernel_time_cuco] = run_sss_cuco(
        baseline.get_num_pairs(), graph.get_num_nodes(),
        baseline.get_d_vertexs(), baseline.get_d_csr_cols_for_vertexs(),
        graph.get_device_elements(), graph.get_device_offsets(),
        graph.get_host_offsets(), graph.get_host_elements(), threshold,
        grid_size, block_size, CHUNK_SIZE, load_factor, 0);

    std::cout << "[cuCollections] 内核执行时间: " << kernel_time_cuco / 1000.0
              << " 秒" << std::endl;
    printf("[cuCollections] 集合相似度结果数: %d\n", result_cuco);
  }

  // ==================== 布谷鸟哈希 ====================
  if (run_cuckoo) {
    std::cout << std::endl;

    flush.flush();
    auto [result_cuckoo, kernel_time_cuckoo] = run_sss_cuckoo(
        baseline.get_num_pairs(), graph.get_num_nodes(),
        baseline.get_d_vertexs(), baseline.get_d_csr_cols_for_vertexs(),
        graph.get_device_elements(), graph.get_device_offsets(),
        graph.get_host_offsets(), graph.get_host_elements(), threshold,
        grid_size, block_size, CHUNK_SIZE, load_factor, 0);

    std::cout << "[Cuckoo] 内核执行时间: " << kernel_time_cuckoo / 1000.0
              << " 秒" << std::endl;
    printf("[Cuckoo] 集合相似度结果数: %d\n", result_cuckoo);
  }

  // ==================== 跳房子哈希 ====================
  if (run_hopscotch) {
    std::cout << std::endl;

    flush.flush();
    auto [result_hopscotch, kernel_time_hopscotch] = run_sss_hopscotch(
        baseline.get_num_pairs(), graph.get_num_nodes(),
        baseline.get_d_vertexs(), baseline.get_d_csr_cols_for_vertexs(),
        graph.get_device_elements(), graph.get_device_offsets(),
        graph.get_host_offsets(), graph.get_host_elements(), threshold,
        grid_size, block_size, CHUNK_SIZE, load_factor, 0);

    std::cout << "[Hopscotch] 内核执行时间: " << kernel_time_hopscotch / 1000.0
              << " 秒" << std::endl;
    printf("[Hopscotch] 集合相似度结果数: %d\n", result_hopscotch);
  }

  // ==================== 咆哮位图 ====================
  if (run_roaring) {
    std::cout << std::endl;

    flush.flush();
    auto [result_roaring, kernel_time_roaring] = run_sss_roaring(
        baseline.get_num_pairs(), graph.get_num_nodes(),
        baseline.get_d_vertexs(), baseline.get_d_csr_cols_for_vertexs(),
        graph.get_device_elements(), graph.get_device_offsets(),
        graph.get_host_offsets(), graph.get_host_elements(), threshold,
        grid_size, block_size, CHUNK_SIZE, load_factor, 0);

    std::cout << "[Roaring] 内核执行时间: " << kernel_time_roaring / 1000.0
              << " 秒" << std::endl;
    printf("[Roaring] 集合相似度结果数: %d\n", result_roaring);
  }

  return 0;
}