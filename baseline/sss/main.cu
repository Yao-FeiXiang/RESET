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
  float load_factor = 0.2;  // 负载因子 - Hopscotch受邻域约束，暂使用0.2
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
    } else if (arg.rfind("--method=", 0) == 0) {
      // 重置所有方法为 false
      run_original = false;
      run_cucollections = false;
      run_cuckoo = false;
      run_hopscotch = false;
      run_roaring = false;

      // 解析逗号分隔的方法列表
      std::vector<std::string> methods = parse_methods(arg.substr(9));
      if (methods.empty()) {
        run_original = true;  // 默认运行original方法
      }
      for (const auto& method : methods) {
        if (method == "original") {
          run_original = true;
        } else if (method == "cuco") {
          run_cucollections = true;
        } else if (method == "cuckoo") {
          run_cuckoo = true;
        } else if (method == "hopscotch") {
          run_hopscotch = true;
        } else if (method == "roaring") {
          run_roaring = true;
        } else if (method == "all") {
          run_original = true;
          run_cucollections = true;
          run_cuckoo = true;
          run_hopscotch = true;
          run_roaring = true;
        } else {
          std::cerr << "未知方法: " << method << std::endl;
          std::cerr
              << "可用方法: original, cuco, cuckoo, hopscotch, roaring, all"
              << std::endl;
          return 1;
        }
      }
    } else {
      std::cerr << "未知参数: " << arg << std::endl;
      std::cerr << "用法: " << argv[0]
                << " <输入文件夹> [--alpha=负载因子] [--bucket=桶大小] "
                   "[--method=original,cuco,cuckoo,hopscotch,roaring]"
                << std::endl;
      return 1;
    }
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

  // ==================== 初始化数据和哈希表 ====================
  SSSBaseline baseline;

  // 1. 加载顶点对（所有方法都需要）
  baseline.load_vertex_pairs(vertexs_path);
  baseline.allocate_buffers();

  // 2. 构建Native哈希表
  baseline.build_hash_tables(graph.get_num_nodes(), graph.get_host_offsets(),
                             graph.get_host_elements(), load_factor,
                             bucket_size);

  // 3. 重排CSR列（初始化d_csr_cols_sorted_供查询使用）
  baseline.reorder_csr_by_hash_layout(graph, bucket_size);

  //  ✔ 不运行original方法时，立即释放Native哈希表内存
  if (!run_original) {
    baseline.free_hash_tables();  // 释放~15GB显存
  }

  // 内核配置参数
  const int grid_size = 512;
  const int block_size = 1024;
  const int CHUNK_SIZE = 4;
  const float threshold = 0.25;

  // 结果变量 - 用于跨方法一致性检查
  int result_normal = -1;
  int result_hierarchical = -1;
  int result_cuco = -1;
  int result_cuckoo = -1;
  int result_hopscotch = -1;
  int result_roaring = -1;

  double time_normal = 0;
  double time_hierarchical = 0;
  double time_cuco = 0;
  double time_cuckoo = 0;
  double time_hopscotch = 0;
  double time_roaring = 0;

  check_gpu_memory();

  // 预排序CSR列(用于hierarchical哈希),在计时前完成(与set-similarity-search一致)
  if (run_original) {
    baseline.pre_sort_csr_cols(graph, bucket_size);
  }

  // 清空L2缓存(保证公平)
  l2flush flush;

  // ==================== original ====================
  if (run_original) {
    // 普通哈希
    std::cout << std::endl;

    flush.flush();
    auto [result_normal_val, kernel_time_normal_val] =
        baseline.run_normal(graph, CHUNK_SIZE, grid_size, block_size,
                            bucket_size, threshold, false);
    result_normal = result_normal_val;
    time_normal = kernel_time_normal_val;

    std::cout << "[Native] 内核执行时间: " << time_normal / 1000.0 << " 秒"
              << std::endl;
    printf("[Native] 集合相似度结果数: %d\n", result_normal);

    // 分层哈希
    std::cout << std::endl;

    flush.flush();
    auto [result_hierarchical_val, kernel_time_hierarchical_val] =
        baseline.run_hierarchical(graph, CHUNK_SIZE, grid_size, block_size,
                                  bucket_size, threshold, true);
    result_hierarchical = result_hierarchical_val;
    time_hierarchical = kernel_time_hierarchical_val;

    std::cout << "[RESET] 内核执行时间: " << time_hierarchical / 1000.0 << " 秒"
              << std::endl;
    printf("[RESET] 集合相似度结果数: %d\n", result_hierarchical);

    // 一致性检查
    if (result_normal != result_hierarchical) {
      printf("警告: 不同哈希类型的结果不一致!\n");
      printf("  普通哈希: %d, 分层哈希: %d\n", result_normal,
             result_hierarchical);
    }

    if (run_cucollections || run_cuckoo || run_hopscotch || run_roaring) {
      baseline.free_hash_tables();
      check_gpu_memory();
    }
  }

  // ==================== cuco ====================
  if (run_cucollections) {
    std::cout << std::endl;
    flush.flush();

    auto [result_cuco_val, kernel_time_cuco_val] = run_sss_cuco(
        baseline.get_num_pairs(), graph.get_num_nodes(),
        baseline.get_d_vertexs(), baseline.get_d_csr_cols_for_vertexs(),
        graph.get_device_elements(), graph.get_device_offsets(),
        graph.get_host_offsets(), graph.get_host_elements(), threshold,
        grid_size, block_size, CHUNK_SIZE, load_factor, 0);
    result_cuco = result_cuco_val;
    time_cuco = kernel_time_cuco_val;

    std::cout << "[cuCollections] 内核执行时间: " << time_cuco / 1000.0 << " 秒"
              << std::endl;
    printf("[cuCollections] 集合相似度结果数: %d\n", result_cuco);

    if (run_original && result_cuco != result_normal) {
      printf("\n   警告: cuCollections与original结果不一致!\n");
      printf("  original(Native): %d, cuCollections: %d, 差异: %d\n",
             result_normal, result_cuco, std::abs(result_cuco - result_normal));
    }
  }

  // ==================== 布谷鸟哈希 ====================
  if (run_cuckoo) {
    std::cout << std::endl;

    flush.flush();
    auto [result_cuckoo_val, kernel_time_cuckoo_val] = run_sss_cuckoo(
        baseline.get_num_pairs(), graph.get_num_nodes(),
        baseline.get_d_vertexs(), baseline.get_d_csr_cols_for_vertexs(),
        graph.get_device_elements(), graph.get_device_offsets(),
        graph.get_host_offsets(), graph.get_host_elements(), threshold,
        grid_size, block_size, CHUNK_SIZE, load_factor, 0);
    result_cuckoo = result_cuckoo_val;
    time_cuckoo = kernel_time_cuckoo_val;

    std::cout << "[Cuckoo] 内核执行时间: " << time_cuckoo / 1000.0 << " 秒"
              << std::endl;
    printf("[Cuckoo] 集合相似度结果数: %d\n", result_cuckoo);

    // 与original结果一致性检查
    if (run_original && result_cuckoo != result_normal) {
      printf("\n   警告: Cuckoo与original结果不一致!\n");
      printf("  original(Native): %d, Cuckoo: %d, 差异: %d\n", result_normal,
             result_cuckoo, std::abs(result_cuckoo - result_normal));
    }
  }

  // ==================== 跳房子哈希 ====================
  if (run_hopscotch) {
    std::cout << std::endl;

    flush.flush();
    auto [result_hopscotch_val, kernel_time_hopscotch_val] = run_sss_hopscotch(
        baseline.get_num_pairs(), graph.get_num_nodes(),
        baseline.get_d_vertexs(), baseline.get_d_csr_cols_for_vertexs(),
        graph.get_device_elements(), graph.get_device_offsets(),
        graph.get_host_offsets(), graph.get_host_elements(), threshold,
        grid_size, block_size, CHUNK_SIZE, load_factor, 0);
    result_hopscotch = result_hopscotch_val;
    time_hopscotch = kernel_time_hopscotch_val;

    std::cout << "[Hopscotch] 内核执行时间: " << time_hopscotch / 1000.0
              << " 秒" << std::endl;
    printf("[Hopscotch] 集合相似度结果数: %d\n", result_hopscotch);

    // 与original结果一致性检查
    if (run_original && result_hopscotch != result_normal) {
      printf("\n   警告: Hopscotch与original结果不一致!\n");
      printf("  original(Native): %d, Hopscotch: %d, 差异: %d\n", result_normal,
             result_hopscotch, std::abs(result_hopscotch - result_normal));
    }
  }

  // ==================== 咆哮位图 ====================
  if (run_roaring) {
    std::cout << std::endl;

    flush.flush();
    auto [result_roaring_val, kernel_time_roaring_val] = run_sss_roaring(
        baseline.get_num_pairs(), graph.get_num_nodes(),
        baseline.get_d_vertexs(), baseline.get_d_csr_cols_for_vertexs(),
        graph.get_device_elements(), graph.get_device_offsets(),
        graph.get_host_offsets(), graph.get_host_elements(), threshold,
        grid_size, block_size, CHUNK_SIZE, load_factor, 0);
    result_roaring = result_roaring_val;
    time_roaring = kernel_time_roaring_val;

    std::cout << "[Roaring] 内核执行时间: " << time_roaring / 1000.0 << " 秒"
              << std::endl;
    printf("[Roaring] 集合相似度结果数: %d\n", result_roaring);

    // 与original结果一致性检查
    if (run_original && result_roaring != result_normal) {
      printf("\n   警告: Roaring与original结果不一致!\n");
      printf("  original(Native): %d, Roaring: %d, 差异: %d\n", result_normal,
             result_roaring, std::abs(result_roaring - result_normal));
    }
  }

  // 最终一致性验证总结
  if (run_original) {
    bool all_consistent = true;
    printf("\n一致性验证总结:  ");
    if (run_cucollections && result_cuco != result_normal)
      all_consistent = false;
    if (run_cuckoo && result_cuckoo != result_normal) all_consistent = false;
    if (run_hopscotch && result_hopscotch != result_normal)
      all_consistent = false;
    if (run_roaring && result_roaring != result_normal) all_consistent = false;

    if (all_consistent) {
      printf("   ✔ 所有已运行方法与original(Native)结果一致\n");
    } else {
      printf("  ❌ 部分方法与original(Native)结果存在差异,请检查\n");
    }
  }

  return 0;
}