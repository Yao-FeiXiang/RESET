/**
 * 超参数求解器 C++ 使用示例
 */

#include <iostream>

#include "hyper_solver.h"

int main() {
  using namespace hyper_solver;

  std::cout << "=== 超参数求解器 C++ 接口示例 ===" << std::endl;

  // 示例1: 基本用法
  std::cout << "\n1. 基本用法 - 求解指定数组大小的超参数" << std::endl;
  {
    long long arr_size = 1000000;  // 1百万条目

    // 直接调用
    HyperParams result = solve_hyperparameters(arr_size);

    std::cout << "  数组大小: " << arr_size << std::endl;
    std::cout << "  推荐 alpha: " << result.alpha << std::endl;
    std::cout << "  推荐 b: " << result.b << std::endl;
    std::cout << "  预测 cost: " << result.cost << std::endl;
  }

  // 示例2: 使用自定义硬件配置
  std::cout << "\n2. 自定义硬件配置" << std::endl;
  {
    HardwareConfig custom_hw;
    custom_hw.W = 32;
    custom_hw.S_trans = 128;
    custom_hw.S_slot = 4;

    long long arr_size = 5000000;  // 5百万条目
    HyperParams result = solve_hyperparameters(arr_size, custom_hw, true);

    std::cout << "  数组大小: " << arr_size << std::endl;
    std::cout << "  推荐 alpha: " << result.alpha << std::endl;
    std::cout << "  推荐 b: " << result.b << std::endl;
  }

  // 示例3: 使用类接口
  std::cout << "\n3. 使用类接口 (可以多次调用)" << std::endl;
  {
    HyperSolver solver;

    // 求解不同规模
    for (long long size : {100000, 500000, 1000000}) {
      HyperParams result = solver.solve(size);
      std::cout << "  规模 " << size << ": alpha=" << result.alpha
                << ", b=" << result.b << std::endl;
    }
  }

  // 示例4: 从数据集路径求解
  std::cout << "\n4. 从数据集路径求解" << std::endl;
  {
    std::string dataset_path = "../../graph_datasets/bm/bio-mouse-gene.edges";
    try {
      HyperParams result = solve_hyperparameters_from_dataset(dataset_path);
      std::cout << "  数据集: " << dataset_path << std::endl;
      std::cout << "  推荐 alpha: " << result.alpha << ", b: " << result.b
                << std::endl;
    } catch (const std::exception& e) {
      std::cout << "  跳过: 数据集不存在" << std::endl;
    }
  }

  // 示例5: 缓存管理
  std::cout << "\n5. 缓存管理" << std::endl;
  {
    std::cout << "  预计算常见配置... (首次运行可能需要几分钟)" << std::endl;
    // precompute_recommended_cache();  // 取消注释执行预计算
    std::cout << "  完成 (示例中跳过实际预计算)" << std::endl;
  }

  std::cout << "\n=== 示例完成 ===" << std::endl;
  return 0;
}
