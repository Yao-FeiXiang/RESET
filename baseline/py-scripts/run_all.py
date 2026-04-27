#!/usr/bin/env python3
"""
全应用自动化测试脚本
功能:
- 运行三个应用(SSS, IR, TC)的所有数据集和所有方法
- 输出类似LaTeX表格格式的结果
- 输出完整CSV数据,包含NCU硬件指标
- 支持自定义参数

==================== 简单使用方法 ====================

1. 运行所有应用的所有测试(含NCU):
   python run_all.py

2. 只运行计时,不运行NCU(快速测试):
   python run_all.py --no-ncu

3. 只运行单个应用(如SSS):
   python run_all.py --apps sss --no-ncu

4. 只运行指定数据集和方法:
   python run_all.py --apps sss --datasets bm --methods Native cuCollections --no-ncu

5. 运行多个应用:
   python run_all.py --apps sss tc --no-ncu

6. 自定义参数:
   python run_all.py --alpha 0.3 --bucket 8 --timeout 600 --retries 2

7. 指定输出文件路径:
   python run_all.py --latex-output output/my_table.tex --csv-output output/my_data.csv

常用组合(快速验证):
   python run_all.py --apps sss --datasets bm --methods Native cuCollections --no-ncu

====================================================
"""

import os
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys

# 导入本地工具函数
from utils import (
    expand_path,
    run_command_with_retry,
    parse_stdout_timing,
    parse_stdout_timing_from_output,
    parse_ncu_output,
    calculate_sector_per_request,
    format_results,
    save_results_to_csv,
)


def load_config():
    """加载配置"""
    # 应用配置
    apps = {
        "sss": {
            "name": "Set Similarity Search",
            "executable": "../sss/sss",
            "dataset_root": "../../graph_datasets/",
            "datasets": ["bm", "gp", "sc18", "sc19", "sc20", "wt"],
            "methods": [
                {
                    "name": "Native",
                    "arg": "--method=original",
                    "tag": "Native",
                    "kernel_name": "sss_kernel",
                    "launch_skip": 0,
                    "launch_count": 1,
                },
                {
                    "name": "RESET",
                    "arg": "--method=original",
                    "tag": "RESET",
                    "kernel_name": "sss_kernel",
                    "launch_skip": 1,
                    "launch_count": 1,
                },
                {
                    "name": "cuCollections",
                    "arg": "--method=cuco",
                    "tag": "cuCollections",
                    "kernel_name": "sss_cuco_kernel",
                    "launch_skip": 0,
                    "launch_count": 1,
                },
            ],
        },
        "ir": {
            "name": "Information Retrieval",
            "executable": "../ir/ir",
            "dataset_root": "../../ir_datasets/",
            "datasets": ["ce", "fe", "hp", "lt", "ms", "nq"],
            "methods": [
                {
                    "name": "Native",
                    "arg": "--method=original",
                    "tag": "Native",
                    "kernel_name": "ir_kernel",
                    "launch_skip": 0,
                    "launch_count": 1,
                },
                {
                    "name": "RESET",
                    "arg": "--method=original",
                    "tag": "RESET",
                    "kernel_name": "ir_kernel",
                    "launch_skip": 1,
                    "launch_count": 1,
                },
                {
                    "name": "cuCollections",
                    "arg": "--method=cuco",
                    "tag": "cuCollections",
                    "kernel_name": "ir_cuco_kernel",
                    "launch_skip": 0,
                    "launch_count": 1,
                },
            ],
        },
        "tc": {
            "name": "Triangle Counting",
            "executable": "../tc/tc",
            "dataset_root": "../../graph_datasets/",
            "datasets": ["bm", "gp", "sc18", "sc19", "sc20", "wt"],
            "methods": [
                {
                    "name": "Native",
                    "arg": "--method=original",
                    "tag": "Native",
                    "kernel_name": "tc_kernel",
                    "launch_skip": 0,
                    "launch_count": 1,
                },
                {
                    "name": "RESET",
                    "arg": "--method=original",
                    "tag": "RESET",
                    "kernel_name": "tc_kernel",
                    "launch_skip": 1,
                    "launch_count": 1,
                },
                {
                    "name": "cuCollections",
                    "arg": "--method=cuco",
                    "tag": "cuCollections",
                    "kernel_name": "tc_cuco_kernel",
                    "launch_skip": 0,
                    "launch_count": 1,
                },
            ],
        },
    }

    # NCU配置
    ncu_config = {
        "tmp_dir": "/tmp/ncu_tmp",
        "timeout": 300,
        "max_retries": 3,
        "metrics_map": {
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "L1GlobalLoadSectors",
            "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": "L1GlobalLoadReq",
        },
    }

    return apps, ncu_config


def run_timing_with_output(
    executable: Path,
    dataset_path: Path,
    method: Dict,
    app_name: str,
    dataset_name: str,
    log_dir: Path,
    timeout: int,
    max_retries: int,
) -> Tuple[Dict[str, Any], str]:
    """运行计时并返回 stdout（用于缓存）
    返回: (结果字典, stdout字符串)
    """
    method_name = method["name"]
    method_tag = method["tag"]
    method_arg = method.get("arg", "")

    cmd = [str(executable), str(dataset_path)]
    if method_arg:
        cmd.append(method_arg)

    log_path = log_dir / f"timing_{app_name}_{dataset_name}_{method_name}.log"

    success, stdout, stderr = run_command_with_retry(
        cmd, cwd=executable.parent, timeout=timeout, max_retries=max_retries, log_path=log_path
    )

    if not success:
        return (
            {
                "app": app_name,
                "dataset": dataset_name,
                "application_name": app_name,
                "method_name": method_name,
                "kernel_time_ms": None,
                "timing_success": False,
                "ncu_status": "SKIPPED",
                "retry_count": 0,
            },
            stdout,
        )

    timing_ms = parse_stdout_timing(stdout, method_tag)
    timing_success = timing_ms is not None

    return (
        {
            "app": app_name,
            "dataset": dataset_name,
            "application_name": app_name,
            "method_name": method_name,
            "kernel_time_ms": timing_ms,
            "timing_success": timing_success,
            "ncu_status": "SKIPPED",
            "retry_count": 0,
        },
        stdout,
    )


def run_timing_only(
    executable: Path,
    dataset_path: Path,
    method: Dict,
    app_name: str,
    dataset_name: str,
    log_dir: Path,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    """只运行计时,不运行NCU（简化版本，不返回stdout）"""
    result, _ = run_timing_with_output(
        executable, dataset_path, method, app_name, dataset_name, log_dir, timeout, max_retries
    )
    return result


def run_ncu_collection(
    executable: Path,
    dataset_path: Path,
    method: Dict,
    ncu_config: Dict,
    timeout: int,
    max_retries: int,
    log_dir: Path,
) -> Dict[str, Any]:
    """运行NCU性能采集"""
    method_name = method["name"]
    method_arg = method.get("arg", "")
    kernel_name = method.get("kernel_name", "")

    ncu_tmp = expand_path(ncu_config["tmp_dir"])
    ncu_tmp.mkdir(parents=True, exist_ok=True)

    method_launch_skip = method.get("launch_skip", ncu_config.get("launch_skip", 5))
    method_launch_count = method.get("launch_count", ncu_config.get("launch_count", 1))

    metrics_list = list(ncu_config["metrics_map"].keys())
    cmd = [
        "ncu",
        "--metrics",
        ",".join(metrics_list),
        f"--launch-skip={method_launch_skip}",
        f"--launch-count={method_launch_count}",
        str(executable),
        str(dataset_path),
    ]

    if method_arg:
        cmd.append(method_arg)

    env = os.environ.copy()
    env["TMPDIR"] = str(ncu_tmp)

    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                cmd,
                cwd=executable.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
                env=env,
            )
            combined = (result.stdout or "") + "\n" + (result.stderr or "")

            gpu_out_of_memory = (
                "out of memory" in combined.lower() or "cuda out of memory" in combined.lower()
            )

            if result.returncode == 0:
                metrics = parse_ncu_output(combined, ncu_config["metrics_map"])
                return {"ncu_status": "OK", "metrics": metrics, "retry_count": attempt}

            print(f"  ✗ NCU失败(退出码:{result.returncode})")
            if gpu_out_of_memory:
                print(f"  📢 GPU内存不足,等待释放...")

            if attempt < max_retries - 1:
                wait_sec = 30 * (attempt + 1) if gpu_out_of_memory else 5 * (attempt + 1)
                print(f"  ⏳ {wait_sec}s后重试...")
                import time

                time.sleep(wait_sec)

        except subprocess.TimeoutExpired:
            print(f"  ⏰ NCU超时")
            if attempt < max_retries - 1:
                import time

                time.sleep(10 * (attempt + 1))

        except Exception as e:
            print(f"  💥 NCU异常: {e}")
            if attempt < max_retries - 1:
                import time

                time.sleep(5 * (attempt + 1))

    return {"ncu_status": "FAILED", "metrics": {}, "retry_count": max_retries}


def run_app_test(
    app_config: Dict,
    app_name: str,
    alpha: float,
    bucket: int,
    ncu_config: Dict,
    enable_ncu: bool,
    log_dir: Path,
    timeout: int,
    max_retries: int,
) -> Tuple[Dict, List[Dict]]:
    """运行单个应用的所有测试,同时返回:
    - 用于LaTeX的字典结果
    - 用于CSV的列表结果

    优化: Native+RESET 使用同一个 --method=original 运行，避免重复执行两次
    """
    script_dir = Path(__file__).parent.absolute()
    executable = script_dir / app_config["executable"]
    dataset_root = script_dir / app_config["dataset_root"]

    latex_results = {}
    csv_results = []

    print(f"\n{'='*60}")
    print(f"  {app_config['name']}")
    print(f"{'='*60}")

    # 缓存: --method=original 运行一次后，缓存计时结果供 Native 和 RESET 共享
    timing_cache: Dict[Tuple[str, str], Dict[str, Any]] = (
        {}
    )  # (dataset, "original") -> timing_result dict

    for dataset in app_config["datasets"]:
        dataset_path = dataset_root / dataset
        if not dataset_path.exists():
            print(f"  ⚠ 数据集不存在: {dataset_path}")
            continue

        print(f"\n  数据集: {dataset}")
        latex_results[dataset] = {}

        # 数据集内存需求检查（避免GPU内存不足导致运行失败）
        MEMORY_THRESHOLD_GB = 18.0
        dataset_memory_requirements = {
            "ms": 25.0,  # ms 数据集需要约 25GB
            "fe": 12.0,  # fe 数据集需要约 12GB
            "gp": 8.0,
            "bm": 6.0,
        }

        required_gb = dataset_memory_requirements.get(dataset, 4.0)
        if required_gb > MEMORY_THRESHOLD_GB:
            print(f"  ⚠ 数据集 {dataset} 预计需要 {required_gb}GB GPU内存")
            print(f"  ⚠ 当前阈值为 {MEMORY_THRESHOLD_GB}GB，跳过此数据集")
            continue

        for method in app_config["methods"]:
            method_name = method["name"]
            method_arg = method.get("arg", "")
            print(f"\n  ── {method_name} ──")

            # 1. 运行计时 - 使用缓存机制避免重复运行
            print(f"  [1/2] 运行计时...")
            cache_key = (dataset, method_arg)

            if cache_key in timing_cache:
                # 使用缓存的计时结果，但解析当前 method 的特定 tag
                cached_result = timing_cache[cache_key]
                timing_result = cached_result.copy()
                # 重新解析当前 method 的计时（从缓存的 stdout）
                timing_ms = parse_stdout_timing_from_output(cached_result["_stdout"], method["tag"])
                timing_result["method_name"] = method_name
                timing_result["kernel_time_ms"] = timing_ms
                timing_result["timing_success"] = timing_ms is not None
                print(f"    ✓ 使用缓存计时结果")
            else:
                # 首次运行，保存 stdout 到缓存
                timing_result, stdout = run_timing_with_output(
                    executable,
                    dataset_path,
                    method,
                    app_name,
                    dataset,
                    log_dir,
                    timeout,
                    max_retries,
                )
                # 缓存原始 stdout 供后续同 arg 的方法使用
                if method_arg == "--method=original":
                    timing_result["_stdout"] = stdout
                    timing_cache[cache_key] = timing_result

            # 2. 运行NCU(如果启用)
            ncu_result = {}
            if enable_ncu and timing_result["timing_success"]:
                print(f"  [2/2] 运行NCU...")
                ncu_result = run_ncu_collection(
                    executable,
                    dataset_path,
                    method,
                    ncu_config,
                    ncu_config["timeout"],
                    max_retries,
                    log_dir,
                )
            else:
                ncu_result = {"ncu_status": "SKIPPED", "metrics": {}, "retry_count": 0}
                if enable_ncu and not timing_result["timing_success"]:
                    print(f"  [2/2] NCU跳过(计时失败)")

            # 合并结果 - 删除内部缓存字段
            row = timing_result.copy()
            row.pop("_stdout", None)
            row.update(ncu_result.get("metrics", {}))
            row["ncu_status"] = ncu_result.get("ncu_status", "SKIPPED")
            row["retry_count"] = ncu_result.get("retry_count", 0)

            csv_results.append(row)

            # 更新LaTeX结果
            if row["kernel_time_ms"] is not None:
                latex_results[dataset][method_name] = row["kernel_time_ms"]
                print(f"    ✓ {method_name}: {row['kernel_time_ms']:.2f} ms")
            else:
                latex_results[dataset][method_name] = "N/A"
                print(f"    ✗ {method_name}: 失败")

    return latex_results, csv_results


def format_latex_value(val: Any, column_name: str) -> str:
    """
    根据列名智能格式化 LaTeX 表格数值
    - 时间列 (ms, cycles, 等): 保留2位小数
    - 大数值列 (Requests, Sectors, 等): 使用千位分隔符
    - 比率列 (sector_per_request, 等): 保留2位小数
    """
    if val is None or val == "N/A" or (isinstance(val, float) and val != val):
        return "N/A"

    # 数值格式化规则
    if isinstance(val, (int, float)):
        # 时间类列 - 2位小数
        if any(t in column_name.lower() for t in ["time", "ms", "cycle", "latency", "duration"]):
            return f"{val:.2f}"
        # 比率类列 - 2位小数
        elif any(r in column_name.lower() for r in ["per_", "ratio", "rate"]):
            return f"{val:.2f}"
        # 大数值计数类 - 千位分隔符整数
        elif any(c in column_name.lower() for c in ["req", "sector", "count", "bytes"]):
            if isinstance(val, float):
                val = int(val)
            return f"{val:,}".replace(",", "\\,")  # LaTeX 千位分隔符
        # 默认 - 2位小数
        else:
            return f"{val:.2f}"

    return str(val)


def generate_latex_table(latex_results: Dict, apps_config: Dict) -> str:
    """
    生成 LaTeX 表格输出
    根据列名自动应用相应的数值格式规则
    """
    lines = []

    for app_name, app_results in latex_results.items():
        app_config = apps_config[app_name]
        methods = [m["name"] for m in app_config["methods"]]

        lines.append("")
        lines.append(f"% {app_config['name']} - Kernel Time (ms)")
        lines.append("\\begin{table}[htbp]")
        lines.append("  \\centering")
        lines.append(f"  \\caption{{{app_config['name']} 内核执行时间比较 (ms)}}")
        lines.append("  \\begin{tabular}{@{}l" + "r" * len(methods) + "@{}}")
        lines.append("    \\toprule")

        # 表头
        header = "    Dataset"
        for m in methods:
            header += f" & {m}"
        header += " \\\\"
        lines.append(header)
        lines.append("    \\midrule")

        # 数据行 - 时间列使用 .2f 格式
        for dataset in app_config["datasets"]:
            if dataset in app_results:
                row = f"    {dataset}"
                for m in methods:
                    val = app_results[dataset].get(m)
                    row += f" & {format_latex_value(val, 'kernel_time')}"
                row += " \\\\"
                lines.append(row)

        lines.append("    \\bottomrule")
        lines.append("  \\end{tabular}")
        lines.append(f"  \\label{{tab:{app_name}_kernel_time}}")
        lines.append("\\end{table}")
        lines.append("")

    return "\n".join(lines)


def print_console_summary(latex_results: Dict, apps_config: Dict):
    """在控制台打印汇总表格"""
    for app_name, app_results in latex_results.items():
        app_config = apps_config[app_name]
        methods = [m["name"] for m in app_config["methods"]]

        print(f"\n{'='*80}")
        print(f"  {app_config['name']} - Kernel Time (ms)")
        print(f"{'='*80}")

        # 表头
        header = f"{'Dataset':<12}"
        for m in methods:
            header += f"{m:>15}"
        print(header)
        print("-" * 80)

        # 数据行
        for dataset in app_config["datasets"]:
            if dataset in app_results:
                row = f"{dataset:<12}"
                for m in methods:
                    val = app_results[dataset].get(m, "N/A")
                    if isinstance(val, float):
                        row += f"{val:>15.2f}"
                    else:
                        row += f"{val:>15}"
                print(row)

    print()


def main():
    parser = argparse.ArgumentParser(description="全应用性能测试自动化脚本")
    parser.add_argument(
        "--apps",
        nargs="+",
        default=["sss", "ir", "tc"],
        choices=["sss", "ir", "tc"],
        help="要运行的应用列表",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None, help="要运行的数据集列表(不指定则运行全部)"
    )
    parser.add_argument(
        "--methods", nargs="+", default=None, help="要运行的方法列表(不指定则运行全部)"
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="负载因子")
    parser.add_argument("--bucket", type=int, default=5, help="桶大小")
    parser.add_argument("--timeout", type=int, default=300, help="每个运行的超时时间(秒)")
    parser.add_argument("--retries", type=int, default=3, help="失败重试次数")
    parser.add_argument("--no-ncu", action="store_true", help="不运行NCU,只运行计时")
    parser.add_argument(
        "--latex-output", type=str, default="output/results_table.tex", help="输出LaTeX文件名"
    )
    parser.add_argument("--csv-output", type=str, default="output/res.csv", help="输出CSV文件名")
    parser.add_argument("--log-dir", type=str, default="logs", help="日志目录")
    args = parser.parse_args()

    apps_config, ncu_config = load_config()
    enable_ncu = not args.no_ncu

    # 只运行指定的应用、数据集、方法
    apps_to_run = {}
    for k, v in apps_config.items():
        if k in args.apps:
            app_config = v.copy()
            # 过滤数据集
            if args.datasets:
                app_config["datasets"] = [d for d in app_config["datasets"] if d in args.datasets]
            # 过滤方法
            if args.methods:
                app_config["methods"] = [
                    m for m in app_config["methods"] if m["name"] in args.methods
                ]
            apps_to_run[k] = app_config

    script_dir = Path(__file__).parent.absolute()
    log_dir = script_dir / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # 运行所有测试
    all_latex_results = {}
    all_csv_results = []

    for app_name, app_config in apps_to_run.items():
        latex_results, csv_results = run_app_test(
            app_config,
            app_name,
            args.alpha,
            args.bucket,
            ncu_config,
            enable_ncu,
            log_dir,
            args.timeout,
            args.retries,
        )
        all_latex_results[app_name] = latex_results
        all_csv_results.extend(csv_results)

    # 计算 sector_per_request 并格式化
    all_csv_results = calculate_sector_per_request(all_csv_results)
    all_csv_results = format_results(all_csv_results)

    # 控制台输出汇总
    print_console_summary(all_latex_results, apps_config)

    # 生成LaTeX表格
    latex_content = generate_latex_table(all_latex_results, apps_config)

    # 保存LaTeX文件
    latex_output_path = script_dir / args.latex_output
    latex_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latex_output_path, "w", encoding="utf-8") as f:
        f.write(latex_content)

    # 保存CSV文件
    csv_output_path = script_dir / args.csv_output
    # CSV 列顺序：原始数值优先，不使用科学计数法列
    custom_columns = [
        "app",
        "dataset",
        "application_name",
        "method_name",
        "kernel_time_ms",
        "L1GlobalLoadReq",
        "L1GlobalLoadSectors",
        "sector_per_request",
        "gpu_cycles_M",
        "timing_success",
        "ncu_status",
        "retry_count",
    ]
    save_results_to_csv(all_csv_results, csv_output_path, custom_columns)

    print(f"\n✅ 测试完成!")
    print(f"  LaTeX表格: {latex_output_path}")
    print(f"  CSV数据: {csv_output_path}")


if __name__ == "__main__":
    main()
