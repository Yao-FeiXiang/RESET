/**
 * 超参数求解器 C++ 实现
 * 通过命令行调用Python C-API
 */

#include "hyper_solver.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>

namespace hyper_solver {

HyperSolver::HyperSolver() : python_initialized_(false) {}

HyperSolver::HyperSolver(const HardwareConfig& config)
    : hw_config_(config), python_initialized_(false) {}

HyperSolver::~HyperSolver() {}

void HyperSolver::set_hardware_config(const HardwareConfig& config) {
  hw_config_ = config;
}

HardwareConfig HyperSolver::detect_hardware() {
  HardwareConfig config;
  // 默认配置
  config.W = 32;
  config.S_trans = 128;
  config.S_slot = 4;

  // 可以在这里添加GPU检测逻辑
  // 使用nvidia-smi检测GPU型号

  return config;
}

std::string HyperSolver::exec_python_command(const std::string& cmd) {
  std::array<char, 128> buffer;
  std::string result;

  // 获取Python脚本所在目录
  const char* script_dir = std::getenv("HYPER_SOLVER_SCRIPT_DIR");
  std::string cd_cmd;
  if (script_dir) {
    cd_cmd = "cd " + std::string(script_dir) + " && ";
  } else {
    // 默认: 假设在src目录下运行
    cd_cmd = "cd $(dirname \"$0\") && ";
  }

  std::string full_cmd = cd_cmd + "python3 -c \"" + cmd + "\"";

  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(full_cmd.c_str(), "r"),
                                                pclose);

  if (!pipe) {
    throw std::runtime_error("Failed to run Python command");
  }

  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }

  return result;
}

HyperParams HyperSolver::parse_result(const std::string& output) {
  double alpha = 0.0;
  int b = 0;
  double cost = 0.0;

  // 解析输出格式: "alpha,b,cost"
  std::istringstream iss(output);
  std::string token;
  int idx = 0;

  while (std::getline(iss, token, ',') && idx < 3) {
    try {
      if (idx == 0)
        alpha = std::stod(token);
      else if (idx == 1)
        b = std::stoi(token);
      else if (idx == 2)
        cost = std::stod(token);
    } catch (...) {
      // 解析失败，使用默认值
    }
    idx++;
  }

  return HyperParams(alpha, b, cost);
}

HyperParams HyperSolver::call_python_via_cli(long long arr_size,
                                             bool use_cache) {
  std::ostringstream oss;
  oss << "from src.c_api import solve_hyperparameters; "
      << "res = solve_hyperparameters(" << arr_size << ", " << hw_config_.W
      << ", " << hw_config_.S_trans << ", " << hw_config_.S_slot << ", "
      << (use_cache ? "True" : "False") << ", "
      << "b'sss'"
      << "); "
      << "print(f\"{res[0]},{res[1]},{res[2]}\")";

  std::string output = exec_python_command(oss.str());
  return parse_result(output);
}

HyperParams HyperSolver::call_python_dataset_via_cli(
    const std::string& dataset_path, bool use_cache) {
  std::ostringstream oss;
  oss << "from src.c_api import solve_for_dataset; "
      << "res = solve_for_dataset("
      << "b'" << dataset_path << "', " << hw_config_.W << ", "
      << hw_config_.S_trans << ", " << hw_config_.S_slot << ", "
      << (use_cache ? "True" : "False") << ", "
      << "b'sss'"
      << "); "
      << "print(f\"{res[0]},{res[1]},{res[2]}\")";

  std::string output = exec_python_command(oss.str());
  return parse_result(output);
}

HyperParams HyperSolver::solve(long long arr_size, bool use_cache) {
  return call_python_via_cli(arr_size, use_cache);
}

HyperParams HyperSolver::solve_from_dataset(const std::string& dataset_path,
                                            bool use_cache) {
  return call_python_dataset_via_cli(dataset_path, use_cache);
}

HyperParams HyperSolver::get_cached(long long arr_size) {
  std::ostringstream oss;
  oss << "from src.c_api import get_cached_hyperparameters; "
      << "res = get_cached_hyperparameters(" << arr_size << ", " << hw_config_.W
      << ", " << hw_config_.S_trans << ", " << hw_config_.S_slot << ", "
      << "b'sss'"
      << "); "
      << "print(f\"{res[0]},{res[1]},{res[2]}\")";

  std::string output = exec_python_command(oss.str());
  return parse_result(output);
}

void HyperSolver::precompute_cache() {
  std::string cmd =
      "from src.c_api import precompute_cache; precompute_cache()";
  exec_python_command(cmd);
}

void HyperSolver::clear_cache() {
  std::string cmd = "from src.c_api import clear_cache; clear_cache()";
  exec_python_command(cmd);
}

// ============================================================
//                    简易函数版本
// ============================================================

HyperParams solve_hyperparameters(long long arr_size,
                                  const HardwareConfig& config,
                                  bool use_cache) {
  HyperSolver solver(config);
  return solver.solve(arr_size, use_cache);
}

HyperParams solve_hyperparameters_from_dataset(const std::string& dataset_path,
                                               const HardwareConfig& config,
                                               bool use_cache) {
  HyperSolver solver(config);
  return solver.solve_from_dataset(dataset_path, use_cache);
}

void precompute_recommended_cache() {
  HyperSolver solver;
  solver.precompute_cache();
}

void clear_recommended_cache() {
  HyperSolver solver;
  solver.clear_cache();
}

}  // namespace hyper_solver
