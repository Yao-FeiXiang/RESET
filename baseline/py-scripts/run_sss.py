#!/usr/bin/env python3
"""
SSS实验自动化脚本
功能:
- 全量运行所有数据集+所有方法的性能测试
- 只重跑已有CSV中失败的条目(--rerun)
- 老CSV格式转换为新CSV格式(--convert)
- NCU硬件指标自动采集
- 可配置CSV列顺序

使用方法:
  全量运行:      python run_sss.py
  重跑失败:      python run_sss.py --rerun
  格式转换:      python run_sss.py --convert old.csv new.csv

配置文件(config.json):
  csv_columns:    自定义CSV列顺序的数组

输出列说明:
  kernel_time_ms: 内核执行时间(ms, 2位小数)
  gpu_cycles_M: GPU执行周期数(百万周期)
  L1GlobalLoadReq: L1全局加载请求数(整数)
  L1GlobalLoadSectors: L1全局加载扇区数(整数)
  sector_per_request: 平均每个请求的扇区数
  _e6后缀: 数值 / 1,000,000(2位小数)
  _e0后缀: 原始比例(2位小数)
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
    calculate_sector_per_request,
    calculate_sector_per_request_row,
    format_results,
    format_results_row,
    convert_csv_format,
    load_existing_results,
)


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """加载JSON配置文件"""
    script_dir = Path(__file__).parent.absolute()
    with open(script_dir / config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_datasets(dataset_root: Path) -> List[Path]:
    """获取所有数据集目录(已排序)"""
    datasets: List[Path] = []
    dataset_root = dataset_root.resolve()

    if not dataset_root.exists():
        print(f"⚠ 数据集目录不存在: {dataset_root}")
        return datasets

    for d in sorted(dataset_root.iterdir()):
        if d.is_dir() and (any(d.glob("*.bin")) or any(d.glob("*.edges")) or any(d.glob("*.txt"))):
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

    timing_success = timing_ms is not None
    if not timing_success:
        print(f"  ⚠ 命令执行成功但未解析到kernel时间")

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

    ncu_tmp = expand_path(ncu_config["tmp_dir"])
    ncu_tmp.mkdir(parents=True, exist_ok=True)

    method_launch_skip = method.get("launch_skip", ncu_config.get("launch_skip", 0))
    method_launch_count = method.get("launch_count", ncu_config.get("launch_count", 1))

    # 构建NCU命令
    metrics_list = list(ncu_config["metrics_map"].keys())
    cmd = [
        "ncu",
        "--metrics",
        ",".join(metrics_list),
        f"--launch-skip={method_launch_skip}",
        f"--launch-count={method_launch_count}",
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

    # 执行NCU命令(自定义，支持环境变量)
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

            print(f"  ✗ NCU失败(退出码:{result.returncode})")
            if "InterprocessLockFailed" in combined or "nsight-compute-lock" in combined:
                print("  ℹ NCU锁冲突,等待...")
            if gpu_out_of_memory:
                print(f"  📢 GPU内存不足,等待释放...")

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
            print(f"  ⏰ NCU超时({timeout}s)")
            if attempt < max_retries - 1:
                wait_sec = 15 * (attempt + 1)
                print(f"  ⏳ {wait_sec}s后重试...")
                time.sleep(wait_sec)

    print(f"  ✗ NCU全部失败")
    return {"ncu_status": "FAILED_ALL", "metrics": {}, "retry_count": max_retries}


def is_result_failed(result: Dict[str, Any]) -> bool:
    """判断是否需要重跑:计时失败或NCU失败"""
    if not result.get('timing_success', False):
        return True
    if result.get('kernel_time_ms') is None:
        return True
    if result.get('ncu_status', '') != 'OK':
        return True
    return False


def find_failed_entries(existing_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """找出所有需要重跑的条目"""
    failed = [r for r in existing_results if is_result_failed(r)]
    print(f"🔍 {len(existing_results)}条结果中找到{len(failed)}条需重跑")
    return failed


def find_method_by_name(
    methods: List[Dict[str, Any]], method_name: str
) -> Optional[Dict[str, Any]]:
    """根据方法名查找配置"""
    for m in methods:
        if m['method_name'] == method_name:
            return m
    return None


def find_dataset_path_by_name(dataset_path_list: List[Path], dataset_name: str) -> Optional[Path]:
    """根据数据集名查找路径"""
    for dp in dataset_path_list:
        if dp.name == dataset_name:
            return dp
    return None


def main():
    """主函数:支持全量运行、只重跑失败、CSV格式转换"""

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='SSS 实验自动化，支持全量运行、重跑失败条目、CSV格式转换'
    )
    parser.add_argument(
        '--rerun', action='store_true', help='只重跑已有CSV结果中失败的条目(计时失败或NCU失败)'
    )
    parser.add_argument(
        '--convert',
        nargs=2,
        metavar=('INPUT_CSV', 'OUTPUT_CSV'),
        help='将老CSV格式转换为新格式: --convert 输入文件 输出文件',
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config()
    app_cfg = config["app"]
    exp_cfg = config["experiment"]
    ncu_cfg = config["ncu"]
    methods = config["methods"]
    robust_cfg = config["robustness"]
    custom_columns = config.get("csv_columns", None)

    # 如果是CSV格式转换，直接处理并退出
    if args.convert:
        input_csv = Path(args.convert[0])
        output_csv = Path(args.convert[1])
        print(f"🔄 转换CSV格式: {input_csv} -> {output_csv}")
        convert_csv_format(input_csv, output_csv, custom_columns)
        return

    # 构建路径
    script_dir = Path(__file__).parent.absolute()
    executable_path = (script_dir.parent / app_cfg["executable"].lstrip('./')).resolve()
    dataset_root = expand_path(str(script_dir / app_cfg["dataset_root"])).resolve()
    output_dir = (script_dir / app_cfg["output_dir"].lstrip('./')).resolve()
    log_dir = (script_dir / app_cfg.get("log_dir", "./logs").lstrip('./')).resolve()
    csv_path = output_dir / app_cfg.get("csv_filename", f"{app_cfg['name']}_experiment_results.csv")

    # 获取所有数据集
    all_datasets = get_datasets(dataset_root)
    if not all_datasets:
        print("❌ 没有找到任何数据集，退出")
        return

    # 判断运行模式
    if args.rerun:
        print("\n🔄 模式: 只重跑失败条目")
        print(f"📄 读取: {csv_path}")

        if not csv_path.exists():
            print(f"❌ 结果文件不存在, 请先运行全量实验")
            return

        # 读取已有结果，找出失败条目
        all_results = load_existing_results(csv_path)
        # 对已有数据计算 sector/request 列
        all_results = calculate_sector_per_request(all_results)
        failed_entries = find_failed_entries(all_results)

        if not failed_entries:
            print("✅ 没有失败条目，所有结果都已成功，无需重跑")
            return
    else:
        print("\n🔄 模式: 全量运行")
        all_results = []
        failed_entries = []

    # 打印实验摘要(根据模式不同)
    if not args.rerun_failed:
        print_experiment_summary(config, all_datasets)

    # 确保可执行文件存在
    if not ensure_executable(executable_path, script_dir.parent):
        print("❌ 无法获取可执行文件，退出")
        return

    # 根据模式选择运行方式
    if args.rerun and failed_entries:
        print(f"\n🚀 开始重跑 {len(failed_entries)} 个失败条目...\n")

        for idx, failed_entry in enumerate(failed_entries, 1):
            dataset_name = failed_entry['dataset']
            method_name = failed_entry['method_name']

            print(f"\n📝 [{idx}/{len(failed_entries)}] {dataset_name} / {method_name}")

            dataset_path = find_dataset_path_by_name(all_datasets, dataset_name)
            method_config = find_method_by_name(methods, method_name)

            if dataset_path is None or method_config is None:
                print(f"  ⚠ 找不到配置, 跳过")
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
                robust_cfg["command_timeout_seconds"] * 5,  # NCU采集有额外开销,需要更长时间
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
                    # 对重跑后的单条结果计算 sector/request
                    calculate_sector_per_request_row(result_row)
                    all_results[i] = result_row
                    break

            kt = (
                f"{timing_result['kernel_time_ms']:.1f}"
                if timing_result['kernel_time_ms']
                else "N/A"
            )
            print(f"  ✔ 重跑完成: 计时={kt}ms, NCU={ncu_result['ncu_status']}")

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
                    robust_cfg["command_timeout_seconds"] * 5,  # NCU采集有额外开销,需要更长时间
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

                # 计算 sector/request 列
                calculate_sector_per_request_row(result_row)
                all_results.append(result_row)

                kt = (
                    f"{timing_result['kernel_time_ms']:.1f}"
                    if timing_result['kernel_time_ms']
                    else "N/A"
                )
                spr = (
                    f"{result_row['sector_per_request']:.2f}"
                    if result_row.get('sector_per_request')
                    else "N/A"
                )
                print(f"  ✔ 完成: 计时={kt}ms, NCU={ncu_result['ncu_status']}, sector/req={spr}")

    # 保存结果到CSV前先格式化所有数据
    all_results = format_results(all_results)
    save_results_to_csv(all_results, csv_path, custom_columns)

    # 打印最终统计
    print_final_summary(all_results)


if __name__ == "__main__":
    main()
