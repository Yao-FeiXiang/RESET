#!/usr/bin/env python3
"""
本脚本运行超参数优化完整管线 v2.0

新增功能：
1. 缓存支持：复用预计算的推荐值
2. 批量处理：一次处理多个数据集
3. 摘要报告：生成JSON格式的结果摘要
4. 灵活配置：支持命令行参数

使用方法：
    # 仅运行分析预测（快速）
    python main.py --mode analyze --datasets sc18 sc19

    # 完整测试 + NCU 测量
    python main.py --mode full --datasets sc18

    # 使用缓存
    python main.py --mode analyze --datasets sc18 --use-cache

    # 预计算常见配置缓存
    python main.py --precompute-cache
"""

import json
import os
import time
import argparse
from typing import Optional, List, Dict

from solver import HyperparameterSolver
from result import ResultManager
from figure import HeatmapVisualizer
from kernel_runner import KernelRunner
from c_api import RecommendationCache, _cache


def analyze_single_dataset(
    dataset_name: str,
    config: Dict,
    use_cache: bool = False,
    simplify_level: Optional[int] = None,
    mode: str = "sss",
) -> None:
    """
    分析单个数据集：仅运行求解器，不执行NCU测量。

    参数:
        dataset_name: 数据集名称
        config: 配置字典
        use_cache: 是否使用缓存
        simplify_level: 简化搜索空间 (1或2)，None表示完整搜索
        mode: 应用模式 ("sss", "ir", "tc")
    """
    if simplify_level is not None:
        config = simplify_config(config, simplify_level)

    start_time = time.time()
    dataset_path = os.path.join(config["datasets"][mode], dataset_name)

    # 运行分析求解器
    print(f"[分析] {dataset_name} - 开始求解...")
    solver = HyperparameterSolver(config, mode, dataset_path)
    results = solver.solve()

    # 保存结果
    result_manager = ResultManager(dataset_name)
    result_manager.save_res(results)

    # 生成图表
    print(f"[绘图] {dataset_name} - 生成热力图...")
    try:
        viz = HeatmapVisualizer(result_manager)
        viz.plot_all()
    except Exception as e:
        print(f"[警告] 绘图失败: {e}")

    # 导出摘要
    summary_path = os.path.join(
        os.path.dirname(__file__), "..", "res", f"{dataset_name}_summary.json"
    )
    summary = result_manager.export_summary(summary_path)

    elapsed = time.time() - start_time
    best = summary.get("H", {}).get("best_config", {})
    print(f"[完成] {dataset_name} - 用时 {elapsed:.2f}秒")
    print(
        f"       最佳配置: alpha={best.get('alpha')}, b={best.get('b')}, cost={best.get('cost'):.4f}"
    )


def validate_single_dataset(
    dataset_name: str, config: Dict, simplify_level: Optional[int] = None, mode: str = "sss"
) -> None:
    """
    完整验证：同时运行求解器和NCU内核性能分析。

    参数:
        dataset_name: 数据集名称
        config: 配置字典
        simplify_level: 简化搜索空间
        mode: 应用模式
    """
    if simplify_level is not None:
        config = simplify_config(config, simplify_level)

    start_time = time.time()
    dataset_path = os.path.join(config["datasets"][mode], dataset_name)

    # 运行分析求解器
    print(f"[分析] {dataset_name} - 开始求解...")
    solver = HyperparameterSolver(config, mode, dataset_path)
    results_solver = solver.solve()

    # 运行内核性能分析
    print(f"[测量] {dataset_name} - 运行NCU性能分析...")
    kernel_runner = KernelRunner(config, dataset_path, mode)
    results_runner = kernel_runner.run_all()

    # 合并并保存结果
    from util import merge_res

    merged = merge_res(results_solver, results_runner)

    result_manager = ResultManager(dataset_name)
    result_manager.save_res(merged)

    # 生成图表
    print(f"[绘图] {dataset_name} - 生成热力图...")
    try:
        viz = HeatmapVisualizer(result_manager)
        viz.plot_all()
    except Exception as e:
        print(f"[警告] 绘图失败: {e}")

    # 导出摘要
    summary_path = os.path.join(
        os.path.dirname(__file__), "..", "res", f"{dataset_name}_summary.json"
    )
    result_manager.export_summary(summary_path)

    elapsed = time.time() - start_time
    print(f"[完成] {dataset_name} - 用时 {elapsed:.2f}秒")


def simplify_config(config: Dict, level: int = 1) -> Dict:
    """
    简化搜索空间，用于快速测试。
    """
    config = dict(config)
    if level <= 1:
        config["alpha_min"] = 0.2
        config["alpha_max"] = 0.8
        config["b_max"] = 10
    else:
        config["alpha_min"] = 0.35
        config["alpha_max"] = 0.4
        config["b_max"] = 4
    return config


def precompute_cache() -> None:
    """预计算常见配置的缓存。"""
    print("=" * 60)
    print("预计算常见配置的超参数推荐值")
    print("=" * 60)
    _cache.precompute_common_configs()

    # 显示缓存统计
    stats = _cache.get_stats()
    print(f"\n缓存条目数: {stats['total_entries']}")
    print(f"缓存文件: {stats['cache_file']}")


def show_cache() -> None:
    """显示缓存内容。"""
    from precompute_cache import show_cache as _show_cache

    _show_cache()


def main():
    parser = argparse.ArgumentParser(description="超参数优化管线 v2.0")
    parser.add_argument(
        "--mode",
        choices=["analyze", "full"],
        default="analyze",
        help="运行模式: analyze=仅分析, full=分析+NCU测量",
    )
    parser.add_argument("--datasets", nargs="+", default=["sc18"], help="要处理的数据集列表")
    parser.add_argument(
        "--simplify", type=int, choices=[1, 2], default=None, help="简化搜索空间级别 (1或2)"
    )
    parser.add_argument("--use-cache", action="store_true", help="使用缓存（如果有）")
    parser.add_argument("--precompute-cache", action="store_true", help="预计算常见配置缓存")
    parser.add_argument("--show-cache", action="store_true", help="显示当前缓存内容")
    parser.add_argument("--clear-cache", action="store_true", help="清空缓存")
    parser.add_argument("--app-mode", default="sss", help="应用模式 (sss, ir, tc)")
    parser.add_argument("--config", default="config.json", help="配置文件路径")

    args = parser.parse_args()

    # 缓存管理命令
    if args.precompute_cache:
        precompute_cache()
        return

    if args.show_cache:
        show_cache()
        return

    if args.clear_cache:
        print("清空缓存...")
        _cache.clear()
        print("完成")
        return

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 处理数据集
    total_start = time.time()
    print("=" * 60)
    print(f"开始处理，模式: {args.mode}, 数据集: {args.datasets}")
    print("=" * 60)

    for dataset in args.datasets:
        if args.mode == "analyze":
            analyze_single_dataset(
                dataset,
                config,
                use_cache=args.use_cache,
                simplify_level=args.simplify,
                mode=args.app_mode,
            )
        else:
            validate_single_dataset(
                dataset, config, simplify_level=args.simplify, mode=args.app_mode
            )

    total_elapsed = time.time() - total_start
    print("=" * 60)
    print(f"全部完成！总用时: {total_elapsed:.2f}秒")
    print("=" * 60)


if __name__ == "__main__":
    main()
