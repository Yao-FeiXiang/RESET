"""
NVIDIA NCU 内核性能分析器运行器
为不同超参数配置运行GPU内核性能测量。
使用NVIDIA计算分析器 (ncu) 获取硬件性能指标。
"""

from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
import subprocess
import re
import os


class KernelRunner:
    """
    使用NVIDIA NCU对不同超参数组合运行GPU内核性能分析。

    属性:
        config: 来自config.json的配置字典
        app_cfg: 应用特定配置
        kernel_dir: 包含内核可执行文件的目录
        executable: 编译后内核可执行文件路径
        dataset_path: 输入数据集路径
        log_dir: NCU日志目录
        metrics: 要收集的NCU性能指标列表
    """

    # ncu二进制文件默认路径
    NCU_PATH = "/usr/local/cuda/bin/ncu"

    def __init__(self, config: dict, dataset_path: str, mode: str = "sss"):
        self.config = config
        self.app_cfg = config["applications"][mode]

        # Resolve all paths absolutely
        self.kernel_dir = Path(self.app_cfg["kernel_dir"]).resolve()
        self.executable = self.kernel_dir / self.app_cfg["executable"]
        self.dataset_path = Path(dataset_path).resolve()

        # Create log directory
        self.log_dir = (Path("..") / "logs").resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # NCU metrics to collect
        # Default: measure global load sectors
        self.metrics: List[str] = [
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
            # Uncomment below for more detailed metrics:
            # "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
            # "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
        ]

    def build(self) -> None:
        """在内核目录使用make编译内核可执行文件。"""
        subprocess.run(
            ["make"],
            cwd=self.kernel_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def run_all(self) -> List[Dict]:
        """
        对网格搜索中所有超参数组合运行性能分析。

        返回:
            结果字典列表，每个(alpha, b, table_type)包含对应指标
        """
        # 如果可执行文件不存在则重新编译
        if not self.executable.exists():
            self.build()

        # 从配置提取搜索网格
        alpha_min = self.config["alpha_min"]
        alpha_max = self.config["alpha_max"]
        alpha_step = self.config["alpha_step"]
        b_min = self.config["b_min"]
        b_max = self.config["b_max"]

        steps = int(round((alpha_max - alpha_min) / alpha_step))
        total_combinations = (steps + 1) * (b_max - b_min + 1)

        results: List[Dict] = []

        with tqdm(total=total_combinations, desc="内核性能分析 (NCU)") as pbar:
            for i in range(steps + 1):
                alpha = round(alpha_min + i * alpha_step, 2)
                for b in range(b_min, b_max + 1):
                    try:
                        point_results = self._run_single_point(alpha, b)
                        results.extend(point_results)
                    except Exception as e:
                        print(f"[错误] alpha={alpha}, b={b}: {e}")
                    pbar.update(1)

        # 按alpha, b, table_type排序结果

    def _run_single_point(self, alpha: float, b: int) -> List[Dict]:
        """
        对单个超参数点运行性能分析。
        同时测试普通(N)和分层(H)两种表布局。

        参数:
            alpha: 负载因子
            b: 桶大小

        返回:
            包含两个结果字典的列表（每个表类型一个）
        """
        base_cmd = [
            str(self.executable),
            str(self.dataset_path),
            f"--alpha={alpha}",
            f"--bucket={b}",
        ]

        results: List[Dict] = []
        # 测试两种表布局: N = 普通, H = 分层
        for table_type in ("N", "H"):
            # 在分层模式跳过第一次内核启动用于预热
            launch_skip = 0 if table_type == "N" else 1
            tag = f"alpha_{alpha}_b_{b}_{table_type}"

            # 运行NCU并获取性能指标
            metrics = self._run_ncu(base_cmd, launch_skip, tag)

            # 从应用程序标准输出提取内核执行时间
            app_stdout = metrics.pop("__app_stdout", "")
            kernel_time = self._parse_kernel_time(app_stdout, table_type)

            result = {
                "alpha": alpha,
                "b": b,
                "table_type": table_type,
                "kernel_time": kernel_time,
                **metrics,
            }
            results.append(result)

        return results

    def _run_ncu(self, base_cmd: List[str], launch_skip: int, tag: str) -> Dict:
        """
        Execute NVIDIA NCU command to collect hardware metrics.

        Args:
            base_cmd: Base command to run the kernel
            launch_skip: Number of kernel launches to skip (for warmup)
            tag: Unique tag for this run for logging

        Returns:
            Parsed metrics dictionary
        """
        metrics_arg = ",".join(self.metrics)
        ncu_cmd = [
            self.NCU_PATH,
            f"--launch-skip={launch_skip}",
            "--launch-count=1",
            f"--metrics={metrics_arg}",
            "--replay-mode",
            "kernel",
            "--target-processes",
            "all",
            "--kernel-name",
            self.app_cfg["kernel_name"],
        ]
        ncu_cmd.extend(base_cmd)

        # Set custom temp directory for NCU
        env = os.environ.copy()
        ncu_tmp = self.log_dir / "ncu_tmp"
        ncu_tmp.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(ncu_tmp)
        env["TMP"] = str(ncu_tmp)
        env["TEMP"] = str(ncu_tmp)

        result = subprocess.run(
            ncu_cmd,
            cwd=self.kernel_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        # 如果NCU崩溃，记录失败详细信息
        if result.returncode != 0:
            log_path = self.log_dir / f"ncu_failed_{tag}.log"
            error_log = (
                "命令:\n"
                + " ".join(ncu_cmd)
                + "\n\n标准输出:\n"
                + (result.stdout or "")
                + "\n\n标准错误:\n"
                + (result.stderr or "")
                + "\n"
            )
            log_path.write_text(error_log, encoding="utf-8")
            raise RuntimeError(f"NCU性能分析失败，详见: {log_path}")

        combined_output = (result.stdout or "") + "\n" + (result.stderr or "")
        metrics = self._parse_ncu_output(combined_output)
        metrics["__app_stdout"] = combined_output
        return metrics

    @staticmethod
    def _parse_kernel_time(stdout: str, table_type: str) -> float:
        """
        从应用程序输出提取内核执行时间。

        参数:
            stdout: 应用程序标准输出
            table_type: "N" 普通, "H" 分层

        返回:
            解析得到的内核时间（秒），未找到返回 -1.0
        """
        key = "Normal" if table_type == "N" else "Hierarchical"

        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("[Target]"):
                continue
            if key not in line:
                continue
            match = re.search(r"([-+]?\d+(\.\d+)?([eE][-+]?\d+)?)", line)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return -1.0
        return -1.0

    def _parse_ncu_output(self, stdout: str) -> Dict:
        """
        解析NCU输出提取请求的性能指标。

        参数:
            stdout: NCU输出文本

        返回:
            字典，键是性能指标名称 → 值是性能指标数值
        """
        data = {metric: 0.0 for metric in self.metrics}
        # 正则表达式匹配行末的浮点数
        num_pattern = re.compile(r"([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$")

        for metric in self.metrics:
            for line in stdout.splitlines():
                if metric not in line:
                    continue
                match = num_pattern.search(line)
                if not match:
                    continue
                try:
                    val = match.group(1).replace(",", "")
                    data[metric] = float(val)
                except ValueError:
                    data[metric] = 0.0
                break
        return data


def test():
    """测试函数：运行单个超参数点进行测试。"""
    import json

    with open("config.json") as f:
        config = json.load(f)
    r = KernelRunner(config=config, dataset_path="../../graph_datasets/sc20", mode="sss")
    r.build()
    # 注意：原代码调用的是run_one，但实际公开接口是run_all
    # 这里保持原样，仅作为测试入口
    res = r._run_single_point(alpha=0.25, b=5)
    print(res)


if __name__ == "__main__":
    test()
