/**
 * 超参数求解器 C++ 接口头文件
 * 使用Python C-API调用Python超参数求解器
 *
 * 依赖: Python 3.x, pybind11 或 ctypes
 *
 * 使用方法:
 *   1. 初始化: HyperSolver solver;
 *   2. 求解: auto result = solver.solve(arr_size);
 *   3. 获取: result.alpha, result.b
 */

#ifndef HYPER_SOLVER_H
#define HYPER_SOLVER_H

#include <stdexcept>
#include <string>
#include <tuple>

namespace hyper_solver {

struct HyperParams {
  double alpha;
  int b;
  double cost;

  HyperParams() : alpha(0.35), b(4), cost(0.0) {}
  HyperParams(double a, int bucket, double c) : alpha(a), b(bucket), cost(c) {}
};

struct HardwareConfig {
  int W = 32;         // warp大小
  int S_trans = 128;  // 每次事务字节数
  int S_slot = 4;     // 每个槽字节数
};

/**
 * 超参数求解器 C++ 包装类
 * 提供两种调用方式:
 *   1. 使用嵌入式Python解释器 (推荐)
 *   2. 使用系统命令调用Python脚本
 */
class HyperSolver {
 public:
  HyperSolver();
  explicit HyperSolver(const HardwareConfig& config);
  ~HyperSolver();

  /**
   * 求解超参数
   * @param arr_size 哈希表条目总数
   * @param use_cache 是否使用缓存（推荐使用）
   * @return 推荐的超参数
   */
  HyperParams solve(long long arr_size, bool use_cache = true);

  /**
   * 从数据集路径求解超参数
   * @param dataset_path 数据集路径
   * @param use_cache 是否使用缓存
   * @return 推荐的超参数
   */
  HyperParams solve_from_dataset(const std::string& dataset_path,
                                 bool use_cache = true);

  /**
   * 获取缓存中的超参数（不运行求解器）
   * @return 如果缓存存在返回有效值，否则返回(0, 0, inf)
   */
  HyperParams get_cached(long long arr_size);

  /**
   * 预计算常见配置的缓存
   */
  void precompute_cache();

  /**
   * 清空缓存
   */
  void clear_cache();

  /**
   * 设置硬件配置
   */
  void set_hardware_config(const HardwareConfig& config);

  /**
   * 获取默认硬件配置（基于当前GPU检测）
   */
  static HardwareConfig detect_hardware();

 private:
  HardwareConfig hw_config_;
  bool python_initialized_;

  // 通过命令行调用Python脚本
  HyperParams call_python_via_cli(long long arr_size, bool use_cache);
  HyperParams call_python_dataset_via_cli(const std::string& dataset_path,
                                          bool use_cache);

  // 执行Python命令并解析结果
  std::string exec_python_command(const std::string& cmd);
  HyperParams parse_result(const std::string& output);
};

/**
 * 简易版本：直接通过命令行调用，不需要链接Python库
 * 使用方法:
 *   auto result = solve_hyperparameters(arr_size);
 */
HyperParams solve_hyperparameters(
    long long arr_size, const HardwareConfig& config = HardwareConfig(),
    bool use_cache = true);

HyperParams solve_hyperparameters_from_dataset(
    const std::string& dataset_path,
    const HardwareConfig& config = HardwareConfig(), bool use_cache = true);

/**
 * 缓存管理函数
 */
void precompute_recommended_cache();
void clear_recommended_cache();

}  // namespace hyper_solver

#endif  // HYPER_SOLVER_H
