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
            print(f"  [{attempt+1}/{max_retries}] 执行命令: {' '.join(cmd)}")
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
                print(f"  ✓ 尝试成功,退出码: {result.returncode}")
                return True, result.stdout, result.stderr

            # 失败了,打印信息并准备重试
            print(f"  ✗ 尝试失败,退出码: {result.returncode}")
            if gpu_out_of_memory:
                print(f"  📢 检测到GPU内存不足,将等待更长时间后重试")
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"CMD: {' '.join(cmd)}\n\n")
                    f.write(f"Attempt {attempt+1}/{max_retries}\n")
                    f.write(f"Exit code: {result.returncode}\n\n")
                    f.write("OUTPUT:\n" + combined + "\n")

            # 如果不是最后一次尝试,等待一会儿重试
            if attempt < max_retries - 1:
                # GPU内存错误等待更长时间让显存释放
                if 'gpu_out_of_memory' in locals() and gpu_out_of_memory:
                    wait_sec = 30 * (attempt + 1)
                else:
                    wait_sec = 5 * (attempt + 1)
                print(f"  ⏳ 等待 {wait_sec} 秒后重试...")
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
                    f.write(f"Attempt {attempt+1}/{max_retries}\n\n")
                    f.write("OUTPUT (partial):\n" + combined + "\n")

            if attempt < max_retries - 1:
                wait_sec = 10 * (attempt + 1)
                print(f"  ⏳ 等待 {wait_sec} 秒后重试...")
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
    print(f"  ✗ 所有 {max_retries} 次尝试都失败了")
    return False, "", ""


def parse_stdout_timing(stdout: str, output_tag: str) -> Optional[float]:
    """
    从标准输出解析特定方法的内核执行时间
    参数:
        stdout: 可执行文件的标准输出
        output_tag: 方法输出标签,如"普通哈希"
    返回:
        解析得到的执行时间(毫秒),解析失败返回None
    """
    # 逐行查找匹配
    lines = stdout.splitlines()
    last_match_time = None

    for line in lines:
        # 只看包含标签的行,并且包含"内核执行时间"
        if f"[{output_tag}]" in line and "内核执行时间" in line:
            # 使用正则提取冒号后面的浮点数
            match = re.search(r':\s*([0-9.eE+-]+)', line)
            if match:
                # 不要break,找到最后一个匹配为止
                # 因为程序会输出前面所有方法的计时,当前方法在最后
                last_match_time = float(match.group(1))

    if last_match_time is not None:
        # C++程序输出单位是秒,转换为毫秒
        time_ms = last_match_time * 1000
        print(f"  ⏱  解析得到执行时间: {time_ms:.3f} ms")
        return time_ms

    # Debug: 如果没找到,输出匹配过程帮助诊断
    print(f"  🐛 调试: 完整输出内容预览(最后20行):")
    for line in lines[-20:]:
        print(f"    | {line}")

    print(f"  ✗ 未能找到 {output_tag} 的执行时间")
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

        # GPU时间需要转换单位到毫秒
        if metric == "gpu__time_duration.sum":
            if unit in ("s", None, ""):
                value = value * 1000.0
            elif unit == "ms":
                pass  # 已经是毫秒
            elif unit == "us":
                value = value / 1000.0

        results[metrics_map[metric]] = value

    if results:
        print(
            f"  📊 NCU解析结果: { {k: f'{v:.3f}' if isinstance(v, float) else v for k, v in results.items()} }"
        )

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


def save_results_to_csv(results: List[Dict[str, Any]], csv_path: Path) -> None:
    """
    将实验结果保存为CSV文件
    参数:
        results: 结果列表,每个元素是一行数据字典
        csv_path: 输出CSV文件路径
    """
    if not results:
        print("⚠ 没有结果可保存")
        return

    # 确保输出目录存在
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 获取所有列名
    fieldnames = list(results[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✓ 结果已保存到: {csv_path},共 {len(results)} 行")


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
    print(f"  日志目录: {config['logging']['log_dir']}")
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
    print(f"✅ 实验全部完成")
    print("=" * 60)
    print(f"  总运行次数: {total}")
    print(f"  NCU采集成功: {success_ncu}")
    print(f"  成功率: {success_ncu/total*100:.1f}%")
    print("=" * 60 + "\n")
