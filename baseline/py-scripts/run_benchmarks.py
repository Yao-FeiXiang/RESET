#!/usr/bin/env python3
"""
全应用自动化测试脚本
功能:
- 通过 make test-sss/test-ir/test-tc 运行三个应用的基准测试
- 方法限定为: Native, RESET, cuCollections
- 输出完整CSV数据,包含NCU硬件指标
- 根据CSV生成与example.tex格式完全一致的LaTeX表格
- 支持单独将已有CSV转换为LaTeX表格

==================== 简单使用方法 ====================

1. 运行所有应用的所有测试(含NCU):
   python run_benchmarks.py

2. 只运行计时,不运行NCU(快速测试):
   python run_benchmarks.py --no-ncu

3. 只运行单个应用(如SSS):
   python run_benchmarks.py --apps sss --no-ncu

4. 只运行指定数据集:
   python run_benchmarks.py --apps sss --datasets bm --no-ncu

5. 运行多个应用:
   python run_benchmarks.py --apps sss tc --no-ncu

6. 自定义参数:
   python run_benchmarks.py --timeout 600 --retries 2

7. 指定输出文件路径:
   python run_benchmarks.py --latex-output output/my_table.tex --csv-output output/my_data.csv

8. 从已有CSV生成LaTeX:
   python run_benchmarks.py --csv-to-tex output/benchmark_res.csv
   python run_benchmarks.py --csv-to-tex output/benchmark_res.csv --latex-output output/my_table.tex

常用组合(快速验证):
   python run_benchmarks.py --apps sss --datasets bm --no-ncu

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
    load_existing_results,
)


def load_config():
    """加载配置"""
    # 应用配置 - 通过 make test-xxx 运行，仅包含 Native, RESET, cuCollections
    apps = {
        "ir": {
            "name": "Information Retrieval",
            "make_target": "test-ir",
            "dataset_var": "IR_DATASET_NAME",
            "dataset_root": "../ir_datasets/",
            "datasets": ["ce", "fe", "hp", "lt", "ms", "nq"],
            "methods": [
                {
                    "name": "Native",
                    "tag": "Native",
                    "kernel_name": "ir_kernel",
                    "launch_skip": 0,
                    "launch_count": 1,
                },
                {
                    "name": "RESET",
                    "tag": "RESET",
                    "kernel_name": "ir_kernel",
                    "launch_skip": 1,
                    "launch_count": 1,
                },
                {
                    "name": "cuCollections",
                    "tag": "cuCollections",
                    "kernel_name": "ir_cuco_kernel",
                    "launch_skip": 0,
                    "launch_count": 1,
                },
            ],
        },
        "tc": {
            "name": "Triangle Counting",
            "make_target": "test-tc",
            "dataset_var": "GRAPH_DATASET_NAME",
            "dataset_root": "../graph_datasets/",
            "datasets": ["bm", "gp", "sc18", "sc19", "sc20", "wt"],
            "methods": [
                {
                    "name": "Native",
                    "tag": "Native",
                    "kernel_name": "tc_kernel",
                    "launch_skip": 0,
                    "launch_count": 1,
                },
                {
                    "name": "RESET",
                    "tag": "RESET",
                    "kernel_name": "tc_kernel",
                    "launch_skip": 1,
                    "launch_count": 1,
                },
                {
                    "name": "cuCollections",
                    "tag": "cuCollections",
                    "kernel_name": "tc_cuco_kernel",
                    "launch_skip": 0,
                    "launch_count": 1,
                },
            ],
        },
        "sss": {
            "name": "Set Similarity Search",
            "make_target": "test-sss",
            "dataset_var": "GRAPH_DATASET_NAME",
            "dataset_root": "../graph_datasets/",
            "datasets": ["bm", "gp", "sc18", "sc19", "sc20", "wt"],
            "methods": [
                {
                    "name": "Native",
                    "tag": "Native",
                    "kernel_name": "sss_kernel",
                    "launch_skip": 0,
                    "launch_count": 1,
                },
                {
                    "name": "RESET",
                    "tag": "RESET",
                    "kernel_name": "sss_kernel",
                    "launch_skip": 1,
                    "launch_count": 1,
                },
                {
                    "name": "cuCollections",
                    "tag": "cuCollections",
                    "kernel_name": "sss_cuco_kernel",
                    "launch_skip": 0,
                    "launch_count": 1,
                },
            ],
        },
    }

    # NCU配置
    ncu_config = {
        "tmp_dir": "/tmp/ncu_tmp",
        "timeout": 600,  # 增加到600秒，NCU运行较慢
        "max_retries": 3,
        "metrics_map": {
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "L1GlobalLoadSectors",
            "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": "L1GlobalLoadReq",
        },
    }

    return apps, ncu_config


def run_make_test(
    app_config: Dict, dataset_name: str, log_dir: Path, timeout: int, max_retries: int
) -> Tuple[bool, str]:
    """运行 make test-xxx 命令，返回 stdout
    一次运行同时输出 Native, RESET, cuCollections 三个方法的结果
    """
    script_dir = Path(__file__).parent.absolute()
    baseline_dir = script_dir.parent  # py-scripts 的父目录是 baseline

    make_target = app_config["make_target"]
    dataset_var = app_config["dataset_var"]

    cmd = ["make", make_target, f"{dataset_var}={dataset_name}"]
    log_path = log_dir / f"timing_{app_config['name']}_{dataset_name}.log"

    success, stdout, stderr = run_command_with_retry(
        cmd, cwd=baseline_dir, timeout=timeout, max_retries=max_retries, log_path=log_path
    )

    return success, stdout


def run_timing_from_cached_output(
    stdout: str, method: Dict, app_name: str, dataset_name: str
) -> Dict[str, Any]:
    """从缓存的 stdout 中解析单个方法的计时结果"""
    method_name = method["name"]
    method_tag = method["tag"]

    timing_ms = parse_stdout_timing(stdout, method_tag)
    timing_success = timing_ms is not None

    return {
        "app": app_name,
        "dataset": dataset_name,
        "application_name": app_name,
        "method_name": method_name,
        "kernel_time_ms": timing_ms,
        "timing_success": timing_success,
        "ncu_status": "SKIPPED",
        "retry_count": 0,
    }


def run_ncu_collection_with_make(
    app_config: Dict,
    dataset_name: str,
    method: Dict,
    ncu_config: Dict,
    timeout: int,
    max_retries: int,
    log_dir: Path,
) -> Dict[str, Any]:
    """运行NCU性能采集（通过make调用可执行文件）"""
    script_dir = Path(__file__).parent.absolute()
    baseline_dir = script_dir.parent

    method_name = method["name"]
    method_tag = method["tag"]
    kernel_name = method.get("kernel_name", "")

    ncu_tmp = expand_path(ncu_config["tmp_dir"])
    ncu_tmp.mkdir(parents=True, exist_ok=True)

    method_launch_skip = method.get("launch_skip", ncu_config.get("launch_skip", 0))
    method_launch_count = method.get("launch_count", ncu_config.get("launch_count", 1))

    # 确定可执行文件和数据集路径
    workspace_dir = baseline_dir.parent  # baseline的父目录是code/
    if app_config["make_target"] == "test-sss":
        exe_path = baseline_dir / "sss" / "sss"
        dataset_path = workspace_dir / "graph_datasets" / dataset_name
        method_arg = "--method=original,cuco"
    elif app_config["make_target"] == "test-ir":
        exe_path = baseline_dir / "ir" / "ir"
        dataset_path = workspace_dir / "ir_datasets" / dataset_name
        method_arg = "--method=original,cuco"
    elif app_config["make_target"] == "test-tc":
        exe_path = baseline_dir / "tc" / "tc"
        dataset_path = workspace_dir / "graph_datasets" / dataset_name
        method_arg = "--method=original,cuco"
    else:
        return {"ncu_status": "FAILED", "metrics": {}, "retry_count": 0}

    metrics_list = list(ncu_config["metrics_map"].keys())
    cmd = [
        "ncu",
        "-k",
        f"regex:{kernel_name}",  # 只匹配指定kernel，避免采集到错误的kernel
        "--metrics",
        ",".join(metrics_list),
        f"--launch-skip={method_launch_skip}",
        f"--launch-count={method_launch_count}",
        str(exe_path),
        str(dataset_path),
        method_arg,
    ]

    env = os.environ.copy()
    env["TMPDIR"] = str(ncu_tmp)

    # 保存NCU输出日志
    ncu_log_path = log_dir / f"ncu_{method_name}_{dataset_name}.log"

    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                cmd,
                cwd=baseline_dir,
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
                # 验证是否真的采集到了指标
                if metrics:
                    return {"ncu_status": "OK", "metrics": metrics, "retry_count": attempt}
                else:
                    print(f"  ⚠ NCU成功但未采集到指标，可能kernel名称不匹配")
                    # 保存日志用于调试
                    with open(ncu_log_path, "w", encoding="utf-8") as f:
                        f.write(combined)

            print(f"  ✗ NCU失败(退出码:{result.returncode})")
            # 保存失败日志
            with open(ncu_log_path, "w", encoding="utf-8") as f:
                f.write(f"CMD: {' '.join(cmd)}\n\n")
                f.write(f"Exit code: {result.returncode}\n\n")
                f.write(combined)

            if gpu_out_of_memory:
                print(f"  📢 GPU内存不足,等待释放...")

            if attempt < max_retries - 1:
                wait_sec = 30 * (attempt + 1) if gpu_out_of_memory else 5 * (attempt + 1)
                print(f"  ⏳ {wait_sec}s后重试...")
                import time

                time.sleep(wait_sec)

        except subprocess.TimeoutExpired as e:
            print(f"  ⏰ NCU超时")
            # 保存部分输出
            partial_out = (
                (e.stdout.decode() if e.stdout else "")
                + "\n"
                + (e.stderr.decode() if e.stderr else "")
            )
            with open(ncu_log_path, "w", encoding="utf-8") as f:
                f.write(f"CMD: {' '.join(cmd)}\n\n")
                f.write("TIMEOUT\n\n")
                f.write(partial_out)

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

    使用 make test-xxx 命令，一次运行同时输出 Native, RESET, cuCollections 三个方法
    """
    script_dir = Path(__file__).parent.absolute()
    baseline_dir = script_dir.parent

    latex_results = {}
    csv_results = []

    print(f"\n{'='*60}")
    print(f"  {app_config['name']}")
    print(f"{'='*60}")

    for dataset in app_config["datasets"]:
        dataset_root = baseline_dir / app_config["dataset_root"]
        dataset_path = dataset_root / dataset
        if not dataset_path.exists():
            print(f"  ⚠ 数据集不存在: {dataset_path}")
            continue

        print(f"\n  数据集: {dataset}")
        latex_results[dataset] = {}

        # 数据集内存需求检查（仅过滤极端内存需求的数据集）
        # 所有SSS图数据集都在16GB显存范围内正常运行
        SKIP_DATASETS = []  # SSS数据集均可用，IR自行管理
        if app_name == "sss" and dataset in SKIP_DATASETS:
            print(f"  ⚠ 数据集 {dataset} 仅用于IR应用，跳过")
            continue

        # 1. 运行 make test-xxx 一次获取所有方法计时结果
        print(f"  [1/2] 运行 {app_config['make_target']} ...")
        success, stdout = run_make_test(app_config, dataset, log_dir, timeout, max_retries)

        if not success:
            print(f"  ✗ 运行失败，跳过此数据集")
            continue

        # 2. 从 stdout 解析每个方法的计时结果
        for method in app_config["methods"]:
            method_name = method["name"]
            print(f"  ── 解析 {method_name} ...")

            timing_result = run_timing_from_cached_output(stdout, method, app_name, dataset)

            # 3. 运行NCU(如果启用)
            ncu_result = {}
            if enable_ncu and timing_result["timing_success"]:
                print(f"  [2/2] 运行NCU ({method_name})...")
                ncu_result = run_ncu_collection_with_make(
                    app_config,
                    dataset,
                    method,
                    ncu_config,
                    ncu_config["timeout"],
                    max_retries,
                    log_dir,
                )
            else:
                ncu_result = {"ncu_status": "SKIPPED", "metrics": {}, "retry_count": 0}

            # 合并结果
            row = timing_result.copy()
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


def load_results_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """从CSV文件加载结果数据"""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"  ✗ CSV文件不存在: {csv_path}")
        return []
    return load_existing_results(csv_file)


def get_apps_config():
    """获取应用配置，用于确定数据集顺序"""
    return {
        "sss": {
            "name": "Set Similarity Search",
            "datasets": ["bm", "gp", "sc18", "sc19", "sc20", "wt"],
            "sector_scale": 1e8,  # SSS和TC的Load Sectors使用 ×10^8
            "sector_unit_label": "($\\times 10^{8}$)",
        },
        "ir": {
            "name": "Information Retrieval",
            "datasets": ["ce", "fe", "hp", "lt", "ms", "nq"],
            "sector_scale": 1e6,  # IR的Load Sectors使用 ×10^6
            "sector_unit_label": "($\\times 10^{6}$)",
        },
        "tc": {
            "name": "Triangle Counting",
            "datasets": ["bm", "gp", "sc18", "sc19", "sc20", "wt"],
            "sector_scale": 1e8,
            "sector_unit_label": "($\\times 10^{8}$)",
        },
    }


def generate_subtable_for_app(
    results: List[Dict[str, Any]], app_name: str, app_config: Dict
) -> List[str]:
    """为单个应用生成subtable代码，格式与example.tex完全一致"""
    methods = ["RESET", "Native", "cuCollections"]

    # 按数据集组织数据
    data_by_dataset = {}
    for row in results:
        if row.get("app") == app_name:
            dataset = row.get("dataset")
            method = row.get("method_name")
            if dataset not in data_by_dataset:
                data_by_dataset[dataset] = {}
            data_by_dataset[dataset][method] = {
                "time": row.get("kernel_time_ms"),
                "sectors": row.get("L1GlobalLoadSectors"),
            }

    lines = []
    lines.append("    \\begin{subtable}[t]{0.32\\textwidth}")
    lines.append("        \\centering")
    lines.append("        \\small")
    lines.append(
        f"        \\caption{{{app_config['name']}}}\\label{{tab:{app_name}_baseline_comparison}}"
    )
    lines.append("        \\setlength{\\tabcolsep}{3pt}\\scalebox{0.88}{")
    lines.append("            \\begin{tabular}{@{}c r rc rc@{}}")
    lines.append("                \\toprule")
    lines.append(
        "                                 & \\multicolumn{1}{c}{\\textbf{RESET}} & \\multicolumn{2}{c}{\\textbf{Native}} & \\multicolumn{2}{c}{\\textbf{cuCollections}}                                                           \\\\"
    )
    lines.append(
        "                \\textbf{Dataset} & \\multicolumn{1}{c}{value}          & \\multicolumn{1}{c}{value}           & \\multicolumn{1}{c}{speedup}                & \\multicolumn{1}{c}{value} & \\multicolumn{1}{c}{speedup} \\\\"
    )
    lines.append("                \\midrule")
    lines.append(
        "                \\multicolumn{6}{@{}c@{}}{\\textbf{Kernel Time (ms)}}                                                                                                                                                \\\\"
    )

    # Kernel Time 数据行
    for dataset in app_config["datasets"]:
        if dataset in data_by_dataset:
            row_data = data_by_dataset[dataset]
            reset_time = row_data.get("RESET", {}).get("time")
            native_time = row_data.get("Native", {}).get("time")
            cuco_time = row_data.get("cuCollections", {}).get("time")

            # 计算 speedup
            if reset_time and reset_time > 0:
                native_speedup = f"{native_time/reset_time:.2f}$\\times$" if native_time else "N/A"
                cuco_speedup = f"{cuco_time/reset_time:.2f}$\\times$" if cuco_time else "N/A"
            else:
                native_speedup = "N/A"
                cuco_speedup = "N/A"

            # 格式化数值
            reset_str = f"{reset_time:.2f}" if reset_time else "N/A"
            native_str = f"{native_time:.2f}" if native_time else "N/A"
            cuco_str = f"{cuco_time:.2f}" if cuco_time else "N/A"

            lines.append(
                f"                {dataset:<16} & {reset_str:<35} & {native_str:<35} & {native_speedup:<40} & {cuco_str:<25} & {cuco_speedup} \\\\"
            )

    lines.append("                \\midrule")
    lines.append(
        f"                \\multicolumn{{6}}{{@{{}}c@{{}}}}{{\\textbf{{Load Sectors}} {app_config['sector_unit_label']}}}                                                                                                                                  \\\\"
    )

    # Load Sectors 数据行
    sector_scale = app_config["sector_scale"]
    for dataset in app_config["datasets"]:
        if dataset in data_by_dataset:
            row_data = data_by_dataset[dataset]
            reset_sectors = row_data.get("RESET", {}).get("sectors")
            native_sectors = row_data.get("Native", {}).get("sectors")
            cuco_sectors = row_data.get("cuCollections", {}).get("sectors")

            # 缩放并格式化
            reset_scaled = (
                reset_sectors / sector_scale if reset_sectors and reset_sectors > 0 else None
            )
            native_scaled = (
                native_sectors / sector_scale if native_sectors and native_sectors > 0 else None
            )
            cuco_scaled = cuco_sectors / sector_scale if cuco_sectors and cuco_sectors > 0 else None

            # 计算 speedup
            if reset_scaled and reset_scaled > 0:
                native_speedup = (
                    f"{native_scaled/reset_scaled:.2f}$\\times$" if native_scaled else "N/A"
                )
                cuco_speedup = f"{cuco_scaled/reset_scaled:.2f}$\\times$" if cuco_scaled else "N/A"
            else:
                native_speedup = "N/A"
                cuco_speedup = "N/A"

            # 格式化数值
            reset_str = f"{reset_scaled:.2f}" if reset_scaled else "N/A"
            native_str = f"{native_scaled:.2f}" if native_scaled else "N/A"
            cuco_str = f"{cuco_scaled:.2f}" if cuco_scaled else "N/A"

            lines.append(
                f"                {dataset:<16} & {reset_str:<35} & {native_str:<35} & {native_speedup:<40} & {cuco_str:<25} & {cuco_speedup} \\\\"
            )

    lines.append("                \\bottomrule")
    lines.append("            \\end{tabular}}")
    lines.append("    \\end{subtable}")

    return lines


def generate_latex_from_csv(
    results: List[Dict[str, Any]], apps_to_include: List[str] = None
) -> str:
    """根据CSV结果生成完整的LaTeX表格，格式与example.tex完全一致"""
    if not results:
        return "% 无数据"

    apps_config = get_apps_config()

    # 确定要包含的应用
    if apps_to_include is None:
        apps_to_include = ["sss", "ir", "tc"]
    else:
        apps_to_include = [a for a in ["sss", "ir", "tc"] if a in apps_to_include]

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("    \\vspace{-2mm}")
    lines.append("    \\caption{Baseline comparison across applications.}")
    lines.append("    \\vspace{-2mm}")
    lines.append("    \\centering")
    lines.append("    \\small")
    lines.append("    \\setlength{\\tabcolsep}{3pt}")
    lines.append("    \\renewcommand{\\arraystretch}{0.95}")
    lines.append("")

    # 生成每个应用的subtable
    for i, app_name in enumerate(apps_to_include):
        subtable_lines = generate_subtable_for_app(results, app_name, apps_config[app_name])
        lines.extend(subtable_lines)

        # 在subtable之间添加\hfill
        if i < len(apps_to_include) - 1:
            lines.append("    \\hfill")

    lines.append("")
    lines.append("    \\label{tab:baseline_comparison}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def print_console_summary(latex_results: Dict, apps_config: Dict):
    if not latex_results:
        return "% 无数据"

    apps_config = get_apps_config()

    # 确定要包含的应用
    if apps_to_include is None:
        apps_to_include = ["sss", "ir", "tc"]
    else:
        apps_to_include = [a for a in ["sss", "ir", "tc"] if a in apps_to_include]

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("    \\vspace{-2mm}")
    lines.append("    \\caption{Baseline comparison across applications.}")
    lines.append("    \\vspace{-2mm}")
    lines.append("    \\centering")
    lines.append("    \\small")
    lines.append("    \\setlength{\\tabcolsep}{3pt}")
    lines.append("    \\renewcommand{\\arraystretch}{0.95}")
    lines.append("")

    # 生成每个应用的subtable
    for i, app_name in enumerate(apps_to_include):
        subtable_lines = generate_subtable_for_app(latex_results, app_name, apps_config[app_name])
        lines.extend(subtable_lines)

        # 在subtable之间添加\hfill
        if i < len(apps_to_include) - 1:
            lines.append("    \\hfill")

    lines.append("")
    lines.append("    \\label{tab:baseline_comparison}")
    lines.append("\\end{table*}")

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
        "--latex-output", type=str, default="output/benchmark_table.tex", help="输出LaTeX文件名"
    )
    parser.add_argument(
        "--csv-output", type=str, default="output/benchmark_res.csv", help="输出CSV文件名"
    )
    parser.add_argument("--log-dir", type=str, default="logs", help="日志目录")
    parser.add_argument(
        "--csv-to-tex", type=str, default=None, help="从已有的CSV文件生成LaTeX表格(指定CSV路径)"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.absolute()

    # 如果指定了 --csv-to-tex 参数，直接从CSV生成LaTeX
    if args.csv_to_tex:
        print(f"  从CSV生成LaTeX: {args.csv_to_tex}")
        results = load_results_from_csv(args.csv_to_tex)

        if not results:
            print(f"  ✗ 无法从CSV加载数据")
            return

        # 确定包含哪些应用
        apps_available = set(r.get("app") for r in results if r.get("app"))
        apps_to_include = [a for a in ["sss", "ir", "tc"] if a in apps_available]
        print(f"  检测到应用: {', '.join(apps_to_include)}")

        # 生成LaTeX
        latex_content = generate_latex_from_csv(results, apps_to_include)

        # 保存LaTeX文件
        latex_output_path = script_dir / args.latex_output
        latex_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(latex_output_path, "w", encoding="utf-8") as f:
            f.write(latex_content)

        print(f"  ✓ LaTeX表格已生成: {latex_output_path}")
        return

    # 正常运行测试流程
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

    # 从生成的CSV生成LaTeX表格（格式与example.tex完全一致）
    latex_content = generate_latex_from_csv(all_csv_results, args.apps)

    # 保存LaTeX文件
    latex_output_path = script_dir / args.latex_output
    latex_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latex_output_path, "w", encoding="utf-8") as f:
        f.write(latex_content)

    print(f"\n  测试完成!")
    print(f"  LaTeX表格: {latex_output_path}")
    print(f"  CSV数据: {csv_output_path}")


if __name__ == "__main__":
    main()
