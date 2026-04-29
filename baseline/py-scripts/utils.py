"""
实验工具函数模块
提供带超时的命令执行、失败重试、输出解析等通用功能
全部注释使用中文
"""

import os
import re
import time
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import csv


def expand_path(path: str) -> Path:
    """展开路径中的 ~ 和环境变量"""
    expanded = os.path.expanduser(os.path.expandvars(path))
    return Path(expanded)


def run_command_with_retry(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: int = 3600,
    max_retries: int = 3,
    log_path: Optional[Path] = None,
) -> Tuple[bool, str, str]:
    """
    带超时和重试的命令执行
    参数:
        cmd: 命令列表
        cwd: 工作目录
        timeout: 超时时间(秒)
        max_retries: 最大重试次数
        log_path: 失败日志保存路径
    返回:
        (是否成功, stdout, stderr)
    """
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
            combined = (result.stdout or "") + "\n" + (result.stderr or "")

            # 检查输出中是否有GPU内存相关错误,如果有需要等待后重试
            gpu_out_of_memory = (
                "out of memory" in combined.lower()
                or "cuda out of memory" in combined.lower()
                or "out of memory" in combined.lower()
                or "gpu memory" in combined.lower()
            )

            # 如果命令成功退出,直接返回
            if result.returncode == 0:
                return True, result.stdout, result.stderr

            # 失败了,打印信息并准备重试
            print(f"  ✗ 执行失败,退出码: {result.returncode}")
            if gpu_out_of_memory:
                print(f"  📢 检测到GPU内存不足,将等待更长时间后重试")
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"CMD: {' '.join(cmd)}\n\n")
                    f.write(f"Attempt {attempt+1}\n")
                    f.write(f"Exit code: {result.returncode}\n\n")
                    f.write("OUTPUT:\n" + combined + "\n")

            # 如果不是最后一次尝试,等待一会儿重试
            if attempt < max_retries - 1:
                # GPU内存错误等待更长时间让显存释放
                wait_sec = 30 * (attempt + 1) if gpu_out_of_memory else 5 * (attempt + 1)
                print(f"  ⏳ {wait_sec}s后重试...")
                time.sleep(wait_sec)

        except subprocess.TimeoutExpired as e:
            # 超时处理
            print(f"  ⏰ 尝试超时 ({timeout} 秒)")
            combined = (
                (e.stdout.decode() if e.stdout else "")
                + "\n"
                + (e.stderr.decode() if e.stderr else "")
            )
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"CMD: {' '.join(cmd)}\n\n")
                    f.write(f"TIMEOUT after {timeout} seconds\n")
                    f.write(f"Attempt {attempt+1}\n\n")
                    f.write("OUTPUT (partial):\n" + combined + "\n")

            if attempt < max_retries - 1:
                wait_sec = 10 * (attempt + 1)
                print(f"  ⏳ 超时, {wait_sec}s后重试...")
                time.sleep(wait_sec)

        except Exception as e:
            # 其他异常处理
            print(f"  💥 意外异常: {str(e)}")
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"CMD: {' '.join(cmd)}\n\n")
                    f.write(f"Exception: {str(e)}\n")

            if attempt < max_retries - 1:
                wait_sec = 5 * (attempt + 1)
                print(f"  ⏳ 等待 {wait_sec} 秒后重试...")
                time.sleep(wait_sec)

    # 所有尝试都失败了
    print(f"  ✗ 全部失败")
    return False, "", ""


def parse_stdout_timing(stdout: str, output_tag: str) -> Optional[float]:
    """
    从标准输出解析特定方法的内核执行时间
    参数:
        stdout: 可执行文件的标准输出
        output_tag: 方法输出标签,如"Native"
    返回:
        解析得到的执行时间(毫秒),解析失败返回None
    """
    time_seconds = parse_stdout_timing_from_output(stdout, output_tag)
    if time_seconds is not None:
        return time_seconds
    print(f"  ⚠  未能找到 {output_tag} 的执行时间")
    return None


def parse_stdout_timing_from_output(stdout: str, output_tag: str) -> Optional[float]:
    """
    从标准输出解析特定方法的内核执行时间（内部辅助函数，不打印日志）
    参数:
        stdout: 可执行文件的标准输出
        output_tag: 方法输出标签,如"Native"
    返回:
        解析得到的执行时间(毫秒),解析失败返回None
    """
    lines = stdout.splitlines()
    last_match_time = None

    # 使用更健壮的正则表达式，匹配包含标签和时间的多种格式
    pattern = re.compile(rf'\[{re.escape(output_tag)}\].*内核执行时间.*?([0-9.eE+-]+)')

    for line in lines:
        match = pattern.search(line)
        if match:
            try:
                last_match_time = float(match.group(1))
            except ValueError:
                continue

    if last_match_time is not None:
        # C++程序输出单位是秒,转换为毫秒
        time_ms = last_match_time * 1000
        return time_ms
    return None


def parse_ncu_output(ncu_output: str, metrics_map: Dict[str, str]) -> Dict[str, float]:
    """
    解析NCU输出,提取指定指标
    参数:
        ncu_output: NCU的标准输出合并
        metrics_map: NCU指标名 -> 输出列名的映射
    返回:
        解析得到的指标字典 {输出列名: 值}
    """
    results: Dict[str, float] = {}
    pattern = re.compile(r'^(\S+)\s+(?:(\S+)\s+)?([-+]?[\d,]+(?:\.\d+)?(?:[eE][-+]?\d+)?)$')

    for line in ncu_output.splitlines():
        line = line.strip()
        m = pattern.match(line)
        if not m:
            continue

        metric, unit, value_str = m.groups()
        if metric not in metrics_map:
            continue

        # 解析数值
        try:
            value = float(value_str.replace(",", ""))
        except ValueError:
            continue

        # gpu__time_duration.sum 的单位是周期(cycles)，不是时间！
        # 直接保存为百万周期(M cycles)，用于GPU行为分析
        if metric == "gpu__time_duration.sum":
            value = value / 1_000_000.0  # 转换为百万周期

        results[metrics_map[metric]] = value

    return results


def ensure_executable(executable_path: Path, make_dir: Path) -> bool:
    """
    检查可执行文件是否存在,不存在尝试make编译
    参数:
        executable_path: 目标可执行文件路径
        make_dir: Makefile所在目录
    返回:
        是否成功得到可执行文件
    """
    if executable_path.exists() and os.access(executable_path, os.X_OK):
        print(f"✓ 可执行文件已存在: {executable_path}")
        return True

    print(f"⚠ 可执行文件不存在: {executable_path},尝试运行 make 编译...")

    success, _, stderr = run_command_with_retry(
        ["make", "-C", str(make_dir)], cwd=make_dir.parent, timeout=600, max_retries=1
    )

    if success and executable_path.exists() and os.access(executable_path, os.X_OK):
        print(f"✓ 编译成功,可执行文件: {executable_path}")
        return True

    print(f"✗ 编译失败,请检查错误: {stderr}")
    return False


def calculate_sector_per_request_row(result_row: Dict[str, Any]) -> None:
    """
    计算单行结果的 sector_per_request
    参数:
        result_row: 单行结果字典
    """
    req = result_row.get("L1GlobalLoadReq")
    sec = result_row.get("L1GlobalLoadSectors")
    if req is not None and sec is not None and req > 0:
        result_row["sector_per_request"] = sec / req


def calculate_sector_per_request(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    计算所有结果的 sector_per_request
    参数:
        results: 结果列表
    返回:
        添加了sector_per_request列的结果列表
    """
    for row in results:
        calculate_sector_per_request_row(row)
    return results


def format_results_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    对单行结果进行数据格式化
    - 时间列(kernel_time_ms): 保留2位小数, gpu_cycles_M: 保留2位小数
    - L1指标(L1GlobalLoadReq, L1GlobalLoadSectors): 转为整数（原始数值，非科学计数）
    - sector_per_request: 保留2位小数

    注意: 不生成 _e6 科学计数法列，CSV中所有数值使用原始格式

    参数:
        row: 单行实验结果
    返回:
        格式化后的结果行
    """
    # 时间类列 - 保留2位小数
    for col in ["kernel_time_ms", "gpu_cycles_M"]:
        val = row.get(col)
        if val is not None and isinstance(val, (int, float)):
            row[col] = round(val, 2)

    # L1GlobalLoadReq - 原始值转为整数（不进行科学计数转换）
    val = row.get("L1GlobalLoadReq")
    if val is not None and isinstance(val, (int, float)):
        row["L1GlobalLoadReq"] = int(val)

    # L1GlobalLoadSectors - 原始值转为整数（不进行科学计数转换）
    val = row.get("L1GlobalLoadSectors")
    if val is not None and isinstance(val, (int, float)):
        row["L1GlobalLoadSectors"] = int(val)

    # sector_per_request - 保留2位小数
    val = row.get("sector_per_request")
    if val is not None and isinstance(val, (int, float)):
        row["sector_per_request"] = round(val, 2)
    elif val is not None and isinstance(val, str):
        try:
            row["sector_per_request"] = round(float(val), 2)
        except ValueError:
            row["sector_per_request"] = None

    # 删除科学计数法列(如果存在)
    for col in ["sector_per_request_e0", "L1GlobalLoadReq_e6", "L1GlobalLoadSectors_e6"]:
        if col in row:
            del row[col]

    return row


def format_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    格式化所有结果行
    参数:
        results: 结果列表
    返回:
        格式化后的结果列表
    """
    return [format_results_row(row) for row in results]


def load_existing_results(csv_path: Path) -> List[Dict[str, Any]]:
    """读取已有CSV结果文件"""
    if not csv_path.exists():
        return []

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
                    'gpu_cycles_M',
                    'L1GlobalLoadReq',
                    'L1GlobalLoadSectors',
                    'sector_per_request',
                ]:
                    try:
                        processed[key] = float(value) if '.' in value else int(value)
                    except (ValueError, TypeError):
                        processed[key] = None
                else:
                    processed[key] = value
            results.append(processed)

    return results


def save_results_to_csv(
    results: List[Dict[str, Any]], csv_path: Path, custom_columns: Optional[List[str]] = None
) -> None:
    """
    将实验结果保存为CSV文件
    参数:
        results: 结果列表,每个元素是一行数据字典
        csv_path: 输出CSV文件路径
        custom_columns: 自定义列顺序（可选），如果为None则按默认顺序
    """
    if not results:
        print("⚠ 没有结果可保存")
        return

    # 确保输出目录存在
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 默认列顺序
    default_columns = [
        "app",
        "dataset",
        "method_name",
        "kernel_time_ms",
        "gpu_cycles_M",
        "L1GlobalLoadReq",
        "L1GlobalLoadSectors",
        "sector_per_request",
        "L1GlobalLoadReq_e6",
        "L1GlobalLoadSectors_e6",
        "timing_success",
        "ncu_status",
        "retry_count",
    ]

    # 使用自定义列或默认列
    columns_to_use = custom_columns if custom_columns else default_columns

    # 获取所有列名，按自定义顺序排列，不存在的列放在后面
    all_keys = set(results[0].keys())
    fieldnames = []
    for col in columns_to_use:
        if col in all_keys:
            fieldnames.append(col)
            all_keys.remove(col)
    # 添加剩余列(如果有)
    fieldnames.extend(sorted(all_keys))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✓ 结果已保存到: {csv_path},共 {len(results)} 行")


def convert_csv_format(
    input_csv_path: Path, output_csv_path: Path, custom_columns: Optional[List[str]] = None
) -> None:
    """
    读取老的CSV数据，转换为新格式并输出
    - 自动计算缺失的列（sector_per_request, _e6列）
    - 应用新的格式化规则
    - 按自定义列顺序输出
    参数:
        input_csv_path: 输入CSV文件路径
        output_csv_path: 输出CSV文件路径
        custom_columns: 自定义列顺序（可选）
    """
    if not input_csv_path.exists():
        print(f"❌ 输入文件不存在: {input_csv_path}")
        return

    # 读取老的CSV数据
    results = load_existing_results(input_csv_path)
    print(f"📖 读取到 {len(results)} 条数据")

    # 计算缺失的列
    for row in results:
        # 计算sector_per_request如果不存在
        if "sector_per_request" not in row or row.get("sector_per_request") is None:
            calculate_sector_per_request_row(row)

    # 应用新的格式化
    results = format_results(results)

    # 保存为新格式
    save_results_to_csv(results, output_csv_path, custom_columns)
    print(f" ✔ 转换完成: {input_csv_path.name} -> {output_csv_path.name}")


def print_experiment_summary(config: Dict[str, Any], datasets: List[Path]) -> None:
    """
    打印实验配置摘要,方便用户确认
    参数:
        config: 配置字典
        datasets: 待处理数据集列表
    """
    app_name = config["app"]["name"]
    methods = config["methods"]
    timeout = config["robustness"]["command_timeout_seconds"]
    max_retries = config["robustness"]["max_retries"]

    print("\n" + "=" * 60)
    print(f"🧪 SSS 实验自动化 - 配置摘要")
    print("=" * 60)
    print(f"  应用: {app_name}")
    print(f"  待测试方法 ({len(methods)} 种): " + ", ".join(m["method_name"] for m in methods))
    print(f"  待处理数据集 ({len(datasets)} 个): " + ", ".join(d.name for d in datasets))
    print(
        f"  超参数: alpha={config['experiment']['alpha']}, bucket={config['experiment']['bucket']}"
    )
    print(f"  健壮性设置: 超时={timeout}s, 最大重试={max_retries}")
    print(f"  输出目录: {config['app']['output_dir']}")
    print(f"  日志目录: {config['app']['log_dir']}")
    print("=" * 60 + "\n")


def print_dataset_progress(dataset_name: str, current: int, total: int) -> None:
    """
    打印当前数据集处理进度
    参数:
        dataset_name: 数据集名称
        current: 当前索引(从1开始)
        total: 总数量
    """
    print("\n" + "-" * 60)
    print(f"📂 [{current}/{total}] 处理数据集: {dataset_name}")
    print("-" * 60)


def print_final_summary(results: List[Dict[str, Any]]) -> None:
    """
    打印实验完成后的最终统计摘要
    参数:
        results: 所有实验结果列表
    """
    total = len(results)
    success_ncu = sum(1 for r in results if r.get("ncu_status") == "OK")

    print("\n" + "=" * 60)
    print(f" ✔ 实验全部完成")
    print("=" * 60)
    print(f"  总运行次数: {total}")
    print(f"  NCU采集成功: {success_ncu}")
    print(f"  成功率: {success_ncu/total*100:.1f}%")
    print("=" * 60 + "\n")
