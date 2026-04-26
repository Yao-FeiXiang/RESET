#!/usr/bin/env python3
"""
SSS 实验自动化运行脚本
功能:
  - 批量在所有图数据集上运行SSS任务的各种哈希表方法
  - 对每个方法进行端到端计时
  - 使用Nsight Compute (NCU)采集GPU硬件性能指标
  - 支持超时控制和失败自动重试
  - 支持只重跑之前失败的条目，完善已有数据
  - 结果输出为CSV文件,便于后续分析

"""

import json
import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# 导入本地工具函数
from utils import (
    expand_path,
    run_command_with_retry,
    parse_stdout_timing,
    parse_ncu_output,
    ensure_executable,
    save_results_to_csv,
    print_experiment_summary,
    print_dataset_progress,
    print_final_summary,
)


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    加载JSON配置文件
    参数:
        config_path: 配置文件路径(相对于脚本目录)
    返回:
        配置字典
    """
    script_dir = Path(__file__).parent.absolute()
    config_file = script_dir / config_path

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


def get_datasets(dataset_root: Path) -> List[Path]:
    """
    获取所有待处理的数据集目录
    参数:
        dataset_root: 数据集根目录
    返回:
        数据集目录列表(已排序)
    """
    datasets: List[Path] = []

    # resolve()将路径归一化，解析掉所有../..，避免C++程序路径解析错误
    dataset_root = dataset_root.resolve()

    if not dataset_root.exists():
        print(f"⚠ 数据集根目录不存在: {dataset_root}")
        return datasets

    for d in sorted(dataset_root.iterdir()):
        if d.is_dir():
            # 检查是否有数据文件(.bin或.edges或.txt)
            has_data = any(d.glob("*.bin")) or any(d.glob("*.edges")) or any(d.glob("*.txt"))
            if has_data:
                # 同样归一化路径
                datasets.append(d.resolve())

    return datasets


def run_sss_timing(
    executable_path: Path,
    dataset_path: Path,
    alpha: float,
    bucket: int,
    method: Dict[str, Any],
    timeout: int,
    max_retries: int,
    log_dir: Path,
) -> Dict[str, Any]:
    """
    运行单个方法并获取执行时间
    参数:
        executable_path: SSS可执行文件路径
        dataset_path: 数据集路径
        alpha: 负载因子
        bucket: 桶大小
        method: 方法配置字典
        timeout: 超时时间
        max_retries: 最大重试次数
        log_dir: 日志目录
    返回:
        包含计时结果的字典
    """
    method_name = method["method_name"]
    output_tag = method["output_tag"]
    is_original = method["is_original"]

    # 构建命令行参数
    cmd = [str(executable_path), str(dataset_path), f"--alpha={alpha}", f"--bucket={bucket}"]

    # 如果不是original方法，需要指定--method参数
    if not is_original:
        method_arg_map = {
            "cuCollections": "cuco",
            "Cuckoo": "cuckoo",
            "Hopscotch": "hopscotch",
            "Roaring": "roaring",
        }
        if method_name in method_arg_map:
            cmd.append(f"--method={method_arg_map[method_name]}")

    # 失败日志路径
    log_path = log_dir / f"timing_{dataset_path.name}_{method_name}.log"

    # 执行命令
    success, stdout, stderr = run_command_with_retry(
        cmd, cwd=executable_path.parent, timeout=timeout, max_retries=max_retries, log_path=log_path
    )

    if not success:
        return {"method_name": method_name, "kernel_time_ms": None, "timing_success": False}

    # 解析计时结果
    timing_ms = parse_stdout_timing(stdout, output_tag)

    # 计时成功需要同时满足：命令执行成功 AND 成功解析到kernel时间
    timing_success = timing_ms is not None

    if not timing_success:
        print(f"  ⚠  命令执行成功但未能解析kernel时间，标记为失败")

    return {
        "method_name": method_name,
        "kernel_time_ms": timing_ms,
        "timing_success": timing_success,
    }


def run_ncu_collection(
    executable_path: Path,
    dataset_path: Path,
    alpha: float,
    bucket: int,
    method: Dict[str, Any],
    ncu_config: Dict[str, Any],
    timeout: int,
    max_retries: int,
    log_dir: Path,
) -> Dict[str, Any]:
    """
    运行NCU性能采集，提取硬件指标
    参数:
        executable_path: SSS可执行文件路径
        dataset_path: 数据集路径
        alpha: 负载因子
        bucket: 桶大小
        method: 方法配置字典
        ncu_config: NCU配置字典
        timeout: 超时时间
        max_retries: 最大重试次数
        log_dir: 日志目录
    返回:
        包含NCU采集结果的字典
    """
    method_name = method["method_name"]
    kernel_name = method["kernel_name"]
    is_original = method["is_original"]

    # 展开NCU临时目录
    ncu_tmp = expand_path(ncu_config["tmp_dir"])
    ncu_tmp.mkdir(parents=True, exist_ok=True)

    # 构建NCU命令
    metrics_list = list(ncu_config["metrics_map"].keys())
    cmd = [
        "ncu",
        "--metrics",
        ",".join(metrics_list),
        f"--launch-skip={ncu_config['launch_skip']}",
        f"--launch-count={ncu_config['launch_count']}",
        f"--kernel-name={kernel_name}",
        str(executable_path),
        str(dataset_path),
        f"--alpha={alpha}",
        f"--bucket={bucket}",
    ]

    # 添加方法参数
    if not is_original:
        method_arg_map = {
            "cuCollections": "cuco",
            "Cuckoo": "cuckoo",
            "Hopscotch": "hopscotch",
            "Roaring": "roaring",
        }
        if method_name in method_arg_map:
            cmd.append(f"--method={method_arg_map[method_name]}")

    # 失败日志路径
    log_path = log_dir / f"ncu_{dataset_path.name}_{method_name}.log"

    # 设置环境变量
    env = os.environ.copy()
    env["TMPDIR"] = str(ncu_tmp)
    env["TMP"] = str(ncu_tmp)
    env["TEMP"] = str(ncu_tmp)

    # 执行NCU命令（自定义，支持环境变量）
    for attempt in range(max_retries):
        try:

            result = subprocess.run(
                cmd,
                cwd=executable_path.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
                env=env,
            )
            combined = (result.stdout or "") + "\n" + (result.stderr or "")

            # 检查输出中是否有GPU内存相关错误，如果有需要等待后重试
            gpu_out_of_memory = (
                "out of memory" in combined.lower()
                or "cuda out of memory" in combined.lower()
                or "out of memory" in combined.lower()
                or "gpu memory" in combined.lower()
            )

            if result.returncode == 0:
                metrics = parse_ncu_output(combined, ncu_config["metrics_map"])
                return {"ncu_status": "OK", "metrics": metrics, "retry_count": attempt}

            print(f"  ✗ NCU失败,退出码: {result.returncode}")
            if "InterprocessLockFailed" in combined or "nsight-compute-lock" in combined:
                print("  ℹ NCU锁冲突,重试中...")
            if gpu_out_of_memory:
                print(f"  📢 GPU内存不足,等待释放后重试")

            # 保存失败日志
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"CMD: {' '.join(cmd)}\n\n")
                f.write(f"Attempt {attempt+1}\n")
                f.write(f"Exit code: {result.returncode}\n\n")
                f.write("OUTPUT:\n" + combined + "\n")

            if attempt < max_retries - 1:
                # GPU内存错误等待更长时间让显存释放
                wait_sec = 60 * (attempt + 1) if gpu_out_of_memory else 10 * (attempt + 1)
                print(f"  ⏳ {wait_sec}s后重试...")
                time.sleep(wait_sec)

        except subprocess.TimeoutExpired as e:
            print(f"  ⏰ NCU超时 ({timeout} 秒)")
            combined = (
                (e.stdout.decode() if e.stdout else "")
                + "\n"
                + (e.stderr.decode() if e.stderr else "")
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"CMD: {' '.join(cmd)}\n\n")
                f.write(f"TIMEOUT after {timeout} seconds\n")
                f.write(f"Attempt {attempt+1}\n\n")
                f.write("OUTPUT (partial):\n" + combined + "\n")

            if attempt < max_retries - 1:
                wait_sec = 15 * (attempt + 1)
                print(f"  ⏳ 超时,{wait_sec}s后重试...")
                time.sleep(wait_sec)

    # 所有尝试失败
    print(f"  ✗ NCU全部失败")
    return {"ncu_status": "FAILED_ALL", "metrics": {}, "retry_count": max_retries}


def load_existing_results(csv_path: Path) -> List[Dict[str, Any]]:
    """
    读取已有的CSV结果文件
    参数:
        csv_path: CSV文件路径
    返回:
        已有的结果列表，如果文件不存在返回空列表
    """
    if not csv_path.exists():
        return []

    import csv

    results: List[Dict[str, Any]] = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 将字符串转换回适当的类型
            processed: Dict[str, Any] = {}
            for key, value in row.items():
                if value == '' or value == 'None':
                    processed[key] = None
                elif key in ['timing_success']:
                    processed[key] = value.lower() == 'true'
                elif key in [
                    'kernel_time_ms',
                    'retry_count',
                    'GPU_duration_ms',
                    'L1GlobalLoadReq',
                    'L1GlobalLoadSectors',
                ]:
                    try:
                        processed[key] = float(value) if '.' in value else int(value)
                    except (ValueError, TypeError):
                        processed[key] = None
                else:
                    processed[key] = value
            results.append(processed)

    return results


def is_result_failed(result: Dict[str, Any]) -> bool:
    """
    判断一个结果是否需要重跑：只要timing失败、kernel时间为空 或者 NCU失败就需要重跑
    参数:
        result: 结果字典
    返回:
        是否需要重跑
    """
    # 如果计时不成功，需要重跑
    if not result.get('timing_success', False):
        return True
    # 如果kernel时间为空，需要重跑
    if result.get('kernel_time_ms') is None:
        return True
    # 如果NCU状态不是OK，需要重跑
    if result.get('ncu_status', '') != 'OK':
        return True
    return False


def find_failed_entries(existing_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    从已有结果中找出所有需要重跑的条目
    参数:
        existing_results: 已有结果列表
    返回:
        需要重跑的条目列表
    """
    failed = [r for r in existing_results if is_result_failed(r)]
    print(f"🔍 在已有 {len(existing_results)} 条结果中找到 {len(failed)} 条需要重跑")
    return failed


def find_method_by_name(
    methods: List[Dict[str, Any]], method_name: str
) -> Optional[Dict[str, Any]]:
    """
    根据方法名查找方法配置
    参数:
        methods: 所有方法配置列表
        method_name: 要查找的方法名
    返回:
        方法配置，如果没找到返回None
    """
    for m in methods:
        if m['method_name'] == method_name:
            return m
    return None


def find_dataset_path_by_name(dataset_path_list: List[Path], dataset_name: str) -> Optional[Path]:
    """
    根据数据集名查找数据集路径
    参数:
        dataset_path_list: 所有数据集路径列表
        dataset_name: 要查找的数据集名
    返回:
        数据集路径，如果没找到返回None
    """
    for dp in dataset_path_list:
        if dp.name == dataset_name:
            return dp
    return None


def main():
    """主函数：完整实验流程，支持全量运行或只重跑失败"""
    import subprocess

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='SSS 实验自动化，支持全量运行或只重跑之前失败的条目'
    )
    parser.add_argument(
        '--rerun-failed',
        action='store_true',
        help='只重跑已有CSV结果中失败的条目（计时失败或NCU失败）',
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config()
    app_cfg = config["app"]
    exp_cfg = config["experiment"]
    ncu_cfg = config["ncu"]
    methods = config["methods"]
    robust_cfg = config["robustness"]
    log_cfg = config["logging"]

    # 构建路径
    script_dir = Path(__file__).parent.absolute()
    executable_path = (script_dir.parent / app_cfg["executable"].lstrip('./')).resolve()
    dataset_root = expand_path(str(script_dir / app_cfg["dataset_root"])).resolve()
    output_dir = (script_dir / app_cfg["output_dir"].lstrip('./')).resolve()
    log_dir = (script_dir / log_cfg["log_dir"].lstrip('./')).resolve()
    csv_path = output_dir / f"{app_cfg['name']}_experiment_results.csv"

    # 获取所有数据集
    all_datasets = get_datasets(dataset_root)
    if not all_datasets:
        print("❌ 没有找到任何数据集，退出")
        return

    # 判断运行模式
    if args.rerun_failed:
        print("\n🔄 模式：只重跑失败条目")
        print(f"📄 读取已有结果: {csv_path}")

        if not csv_path.exists():
            print(f"❌ 已有结果文件不存在: {csv_path}，无法重跑，请先运行全量实验")
            return

        # 读取已有结果，找出失败条目
        all_results = load_existing_results(csv_path)
        failed_entries = find_failed_entries(all_results)

        if not failed_entries:
            print("✅ 没有失败条目，所有结果都已成功，无需重跑")
            return
    else:
        print("\n🔄 模式：全量运行所有实验")
        all_results = []
        failed_entries = []

    # 打印实验摘要（根据模式不同）
    if not args.rerun_failed:
        print_experiment_summary(config, all_datasets)

    # 确保可执行文件存在
    if not ensure_executable(executable_path, script_dir.parent):
        print("❌ 无法获取可执行文件，退出")
        return

    # 根据模式选择运行方式
    if args.rerun_failed and failed_entries:
        # 只重跑失败条目
        print(f"\n🚀 开始重跑 {len(failed_entries)} 个失败条目...\n")

        for idx, failed_entry in enumerate(failed_entries, 1):
            dataset_name = failed_entry['dataset']
            method_name = failed_entry['method_name']

            print(
                f"\n📝 [{idx}/{len(failed_entries)}] 重跑: dataset={dataset_name}, method={method_name}"
            )

            # 找到对应的数据集路径和方法配置
            dataset_path = find_dataset_path_by_name(all_datasets, dataset_name)
            method_config = find_method_by_name(methods, method_name)

            if dataset_path is None or method_config is None:
                print(f"  ⚠  找不到数据集或方法配置，跳过")
                continue

            timing_result = run_sss_timing(
                executable_path,
                dataset_path,
                exp_cfg["alpha"],
                exp_cfg["bucket"],
                method_config,
                robust_cfg["command_timeout_seconds"],
                robust_cfg["max_retries"],
                log_dir,
            )

            ncu_result = run_ncu_collection(
                executable_path,
                dataset_path,
                exp_cfg["alpha"],
                exp_cfg["bucket"],
                method_config,
                ncu_cfg,
                robust_cfg["command_timeout_seconds"] * 2,  # NCU通常需要更长时间
                robust_cfg["max_retries"],
                log_dir,
            )

            result_row = {
                "app": app_cfg["name"],
                "dataset": dataset_name,
                "method_name": method_name,
                "kernel_time_ms": timing_result["kernel_time_ms"],
                "timing_success": timing_result["timing_success"],
                "ncu_status": ncu_result["ncu_status"],
                "retry_count": ncu_result["retry_count"],
            }

            for metric_name, metric_value in ncu_result["metrics"].items():
                result_row[metric_name] = metric_value

            for output_metric in ncu_cfg["metrics_map"].values():
                if output_metric not in result_row:
                    result_row[output_metric] = None

            for i, existing in enumerate(all_results):
                if existing['dataset'] == dataset_name and existing['method_name'] == method_name:
                    all_results[i] = result_row
                    break

            # 打印当前方法结果
            kt = (
                f"{timing_result['kernel_time_ms']:.3f}"
                if timing_result['kernel_time_ms']
                else "N/A"
            )
            print(f"  ✔ 完成重跑: 计时={kt}ms, NCU={ncu_result['ncu_status']}")

    else:
        for idx, dataset_path in enumerate(all_datasets, 1):
            dataset_name = dataset_path.name
            print_dataset_progress(dataset_name, idx, len(all_datasets))

            for method in methods:
                timing_result = run_sss_timing(
                    executable_path,
                    dataset_path,
                    exp_cfg["alpha"],
                    exp_cfg["bucket"],
                    method,
                    robust_cfg["command_timeout_seconds"],
                    robust_cfg["max_retries"],
                    log_dir,
                )

                ncu_result = run_ncu_collection(
                    executable_path,
                    dataset_path,
                    exp_cfg["alpha"],
                    exp_cfg["bucket"],
                    method,
                    ncu_cfg,
                    robust_cfg["command_timeout_seconds"] * 2,  # NCU通常需要更长时间
                    robust_cfg["max_retries"],
                    log_dir,
                )

                result_row = {
                    "app": app_cfg["name"],
                    "dataset": dataset_name,
                    "method_name": method["method_name"],
                    "kernel_time_ms": timing_result["kernel_time_ms"],
                    "timing_success": timing_result["timing_success"],
                    "ncu_status": ncu_result["ncu_status"],
                    "retry_count": ncu_result["retry_count"],
                }

                for metric_name, metric_value in ncu_result["metrics"].items():
                    result_row[metric_name] = metric_value

                for output_metric in ncu_cfg["metrics_map"].values():
                    if output_metric not in result_row:
                        result_row[output_metric] = None

                all_results.append(result_row)

                # 打印当前方法结果
                kt = (
                    f"{timing_result['kernel_time_ms']:.3f}"
                    if timing_result['kernel_time_ms']
                    else "N/A"
                )
                print(f"  ✔ 完成: 计时={kt}ms, NCU={ncu_result['ncu_status']}")

    # 保存结果到CSV
    save_results_to_csv(all_results, csv_path)

    # 打印最终统计
    print_final_summary(all_results)


if __name__ == "__main__":
    main()
