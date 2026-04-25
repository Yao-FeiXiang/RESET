"""

本脚本运行超参数优化完整管线：
1. 仅分析求解：运行分析成本模型预测（快速）+ 生成热力图
2. 完整测试：同时运行预测 + NCU内核性能测量进行验证

"""

import json
import os
import time
from typing import Optional, Dict

from kernel_runner import KernelRunner
from solver import HyperparameterSolver
from result import ResultManager
from figure import ResultVisualizer
from util import merge_res


# ============================================================================
#                          配置
# ============================================================================

# 应用模式: "sss" (集合相似度搜索), "ir" (信息检索), "tc" (三角形计数)
MODE = "sss"
# 图数据集路径
GRAPH_DIR = "../../graph_datasets"
# 是否运行完整NCU测量 (True) 或仅运行分析预测 (False)
EXECUTE_ALL = True


# ============================================================================
#                          Pipeline Functions
# ============================================================================


def validate(dataset_name: str, simplify_level: Optional[int] = None) -> None:
    """
    完整验证：同时运行求解器和NCU内核性能分析。

    参数:
        dataset_name: 要处理的数据集名称
        simplify_level: 如果提供，简化搜索空间用于快速测试
            - 1: alpha [0.2-0.8], b <= 10
            - 2: alpha [0.35-0.4], b <= 4
    """
    with open("config.json") as f:
        config: Dict = json.load(f)

    if simplify_level is not None:
        config = simplify_config(config, simplify_level)

    start_time = time.time()
    dataset_path = os.path.join(config["datasets"][MODE], dataset_name)

    # 运行分析求解器
    solver = HyperparameterSolver(config, MODE, dataset_path)
    result_manager = ResultManager(dataset_name)

    # 运行内核性能分析
    kernel_runner = KernelRunner(config, dataset_path, MODE)
    res_solver = solver.solve()
    res_runner = kernel_runner.run_all()

    # 合并并保存结果
    merged = merge_res(res_solver, res_runner)
    result_manager.save_res(merged)

    elapsed = time.time() - start_time
    print(f"[INFO] 完成数据集 '{dataset_name}'，用时 {elapsed:.2f}秒")


def simplify_config(config: Dict, level: int = 1) -> Dict:
    """
    简化搜索空间，用于快速测试。
    """
    if level <= 1:
        config["alpha_min"] = 0.2
        config["alpha_max"] = 0.8
        config["b_max"] = 10
    else:
        config["alpha_min"] = 0.35
        config["alpha_max"] = 0.4
        config["b_max"] = 4
    return config


def test_kernel_runner(dataset_name: str) -> None:
    """
    快速测试：仅在简化搜索空间上运行内核测试器。
    """
    with open("config.json") as f:
        config = json.load(f)

    config = simplify_config(config, simplify_level=2)
    dataset_path = os.path.join(config["datasets"][MODE], dataset_name)
    kernel_runner = KernelRunner(config, dataset_path, MODE)
    result_manager = ResultManager(dataset_name, filename="test_results.csv")
    result_manager.save_res(kernel_runner.run_all())


def run_analytical_solver(
    dataset_name: str, suffix: str = "", arr_size: Optional[int] = None
) -> None:
    """
    仅运行分析超参数求解器并生成可视化。

    参数:
        dataset_name: 数据集名称
        suffix: 输出文件的可选后缀
        arr_size: 预计算的数组大小（如果为None，从数据集计算）
    """
    with open("config.json") as f:
        config = json.load(f)

    dataset_path = os.path.join(config["datasets"][MODE], dataset_name)
    solver = HyperparameterSolver(config, MODE, dataset_path, arr_size=arr_size)

    output_filename = f"{dataset_name}{suffix}.csv"
    result_manager = ResultManager(dataset_name, filename=output_filename)

    results = solver.solve()
    result_manager.save_res(results)

    # 生成所有图表
    visualizer = ResultVisualizer(result_manager)
    visualizer.plot_all()
    print(f"[INFO] 结果已保存到 {output_filename}，图表已生成")


# ============================================================================
#                          主入口
# ============================================================================

if __name__ == "__main__":
    dataset_list = os.listdir(GRAPH_DIR)
    for dataset_name in dataset_list:
        if EXECUTE_ALL:
            validate(dataset_name)
        else:
            run_analytical_solver(dataset_name)
        print("\n")
