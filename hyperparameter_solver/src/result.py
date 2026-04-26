"""
结果管理器 v2.0：CSV读写和可视化数据准备

优化内容：
1. 增强的数据验证和错误处理
2. 添加统计分析功能
3. 支持多种格式导出
4. 改进的合并策略
"""

import os
import csv
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict


@dataclass
class BestConfig:
    """最佳配置数据类"""

    alpha: float
    b: int
    cost: float
    table_type: str
    kernel_time: Optional[float] = None
    load_sectors: Optional[float] = None


class ResultManager:
    """
    管理超参数搜索结果：CSV读写和数据准备。

    属性:
        KEY_COLS: 用于去重的主键列
        METRICS: 要跟踪的指标列表
        path: 输出CSV文件路径
    """

    KEY_COLS = ("alpha", "b", "table_type")

    METRICS = ["cost", "kernel_time", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"]

    LOAD_SECTOR_KEY = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"

    def __init__(
        self,
        dataset_name: str = None,
        filename: Optional[str] = None,
        precisions: Optional[Dict[str, int]] = None,
    ):
        """
        初始化结果管理器。

        参数:
            dataset_name: 数据集名称（用于默认输出文件名）
            filename: 自定义输出文件名（覆盖默认）
            precisions: 每个指标的小数精度
        """
        res_dir = os.path.join(os.path.dirname(__file__), "..", "res")
        os.makedirs(res_dir, exist_ok=True)

        self.dataset_name = dataset_name or "unknown"
        self.path = os.path.join(res_dir, filename or f"{self.dataset_name}.csv")

        self.precisions = precisions or {
            "alpha": 2,
            "b": 0,
            "cost": 6,
            "kernel_time": 6,
            self.LOAD_SECTOR_KEY: 0,
        }

    # ===================================================================
    #                          CSV 读写操作
    # ===================================================================

    def read_table(self) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
        """
        从CSV文件读取所有结果。

        返回:
            字典，键是 (alpha, b, table_type) -> 值是结果字典
        """
        table: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

        if not os.path.exists(self.path):
            return table

        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key, normalized = self._normalize_row(row)
                table[key] = normalized

        return table

    def save_res(self, rows: List[Dict[str, Any]]) -> None:
        """
        保存结果到CSV，与现有结果合并（增量更新）。

        参数:
            rows: 要保存的结果字典列表
        """
        table = self.read_table()
        expanded_rows: List[Dict[str, Any]] = []

        # 将求解器结果（没有table_type）展开为N和H两种
        for row in rows:
            row_copy = dict(row)
            if "table_type" not in row_copy or not row_copy["table_type"]:
                for table_type in ("N", "H"):
                    expanded = dict(row_copy)
                    expanded["table_type"] = table_type
                    expanded_rows.append(expanded)
            else:
                expanded_rows.append(row_copy)

        # 更新或插入每一行
        for row in expanded_rows:
            key, normalized = self._normalize_row(row)
            if key in table:
                # 合并：仅更新非空字段
                for field_name, value in normalized.items():
                    if value != "" and value is not None:
                        table[key][field_name] = value
            else:
                table[key] = normalized

        self._write_rows(list(table.values()))

    def _write_rows(self, rows: List[Dict[str, Any]]) -> None:
        """将行写入CSV文件。"""
        if not rows:
            return

        fieldnames = list(rows[0].keys())
        # 确保主键列在前面
        for col in reversed(self.KEY_COLS):
            if col in fieldnames:
                fieldnames.remove(col)
                fieldnames.insert(0, col)

        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # ===================================================================
    #                          数据预处理
    # ===================================================================

    def calculate_best(self, eps: float = 1e-12) -> Dict[str, Dict[str, Any]]:
        """
        将原始结果处理为热力图就绪格式并找出最优点。

        参数:
            eps: 浮点比较容差

        返回:
            可供可视化的字典结构
        """
        table = self.read_table()
        if not table:
            return {}

        rows = list(table.values())

        def to_float(value) -> Optional[float]:
            """安全转换为float，失败返回None。"""
            try:
                return float(value)
            except Exception:
                return None

        # 按表类型分组 (N = 普通, H = 分层)
        by_table_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            table_type_key = str(row["table_type"])
            by_table_type[table_type_key].append(row)

        result: Dict[str, Dict[str, Any]] = {}

        for table_type, rs in by_table_type.items():
            # 获取唯一坐标轴值（已排序）
            alphas = sorted({float(r["alpha"]) for r in rs})
            bs = sorted({int(float(r["b"])) for r in rs})

            # 值到网格索引映射: alpha = y轴, b = x轴
            alpha_to_y = {a: i for i, a in enumerate(alphas)}
            b_to_x = {b: i for i, b in enumerate(bs)}

            # 为每个指标分配网格
            Z_metric: Dict[str, List[List[Optional[float]]]] = {
                metric: [[None for _ in bs] for _ in alphas] for metric in self.METRICS
            }

            # 将数据填充到网格
            for r in rs:
                y = alpha_to_y[float(r["alpha"])]
                x = b_to_x[int(float(r["b"]))]

                for metric in self.METRICS:
                    if metric in r:
                        Z_metric[metric][y][x] = to_float(r.get(metric))

            # 寻找最佳配置
            Z_cost = Z_metric["cost"]
            global_best: Optional[Tuple[int, int]] = None
            alpha_local_best: set[Tuple[int, int]] = set()

            # 全局最优（最低成本）
            min_cost = 1e10
            for y in range(len(alphas)):
                for x in range(len(bs)):
                    cost = Z_cost[y][x]
                    if cost is None:
                        continue
                    if cost < min_cost - eps:
                        min_cost = cost
                        global_best = (y, x)

            # 每个alpha的最优
            for y in range(len(alphas)):
                row = Z_cost[y]
                valid_points = [(x, cost) for x, cost in enumerate(row) if cost is not None]
                if not valid_points:
                    continue

                min_row_cost = min(cost for _, cost in valid_points)
                x_best = None
                for x, cost in valid_points:
                    if abs(cost - min_row_cost) <= eps:
                        x_best = x
                        break
                if x_best is not None and (y, x_best) != global_best:
                    alpha_local_best.add((y, x_best))

            result[table_type] = {
                "x": bs,
                "y": alphas,
                "Z_metric": Z_metric,
                "best": {"global": global_best, "alpha": alpha_local_best},
            }

        return result

    # ===================================================================
    #                          统计分析功能
    # ===================================================================

    def get_best_config(self, table_type: str = "H", metric: str = "cost") -> BestConfig:
        """
        获取指定指标的最佳配置。

        参数:
            table_type: 表类型 ("N" 或 "H")
            metric: 优化指标

        返回:
            BestConfig 数据类实例
        """
        processed = self.calculate_best()
        if table_type not in processed:
            raise ValueError(f"无效的 table_type: {table_type}")

        data = processed[table_type]
        Z = np.array(data["Z_metric"][metric], dtype=float)
        alphas = data["y"]
        bs = data["x"]

        # 找到最小值位置
        Z_masked = np.ma.masked_invalid(Z)
        min_idx = np.unravel_index(np.argmin(Z_masked), Z.shape)
        y, x = min_idx

        # 获取所有指标的值
        metrics_data = data["Z_metric"]
        return BestConfig(
            alpha=float(alphas[y]),
            b=int(bs[x]),
            cost=float(metrics_data["cost"][y][x]),
            table_type=table_type,
            kernel_time=(
                float(metrics_data["kernel_time"][y][x])
                if metrics_data["kernel_time"][y][x] is not None
                else None
            ),
            load_sectors=(
                float(metrics_data[self.LOAD_SECTOR_KEY][y][x])
                if metrics_data[self.LOAD_SECTOR_KEY][y][x] is not None
                else None
            ),
        )

    def get_statistics(self, table_type: str = "H") -> Dict[str, Any]:
        """
        获取结果统计信息。

        返回:
            包含各种统计数据的字典
        """
        processed = self.calculate_best()
        if table_type not in processed:
            return {}

        data = processed[table_type]
        stats = {}

        for metric in self.METRICS:
            Z = np.array(data["Z_metric"][metric], dtype=float)
            Z_valid = Z[np.isfinite(Z)]
            if len(Z_valid) > 0:
                stats[metric] = {
                    "mean": float(np.mean(Z_valid)),
                    "std": float(np.std(Z_valid)),
                    "min": float(np.min(Z_valid)),
                    "max": float(np.max(Z_valid)),
                    "median": float(np.median(Z_valid)),
                }

        return stats

    def export_summary(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        导出结果摘要为JSON格式。

        参数:
            output_path: 输出文件路径，None则不保存文件

        返回:
            摘要字典
        """
        summary = {
            "dataset": self.dataset_name,
            "csv_path": self.path,
            "total_rows": len(self.read_table()),
        }

        for table_type in ["N", "H"]:
            try:
                best = self.get_best_config(table_type)
                stats = self.get_statistics(table_type)
                summary[table_type] = {"best_config": asdict(best), "statistics": stats}
            except Exception:
                pass

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary

    # ===================================================================
    #                          内部辅助函数
    # ===================================================================

    def _normalize_row(self, row: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        标准化行：提取键，转换类型，应用舍入。

        返回:
            (key_tuple, normalized_row) 其中 key_tuple = (alpha, b, table_type)
        """
        for key in self.KEY_COLS:
            if key not in row:
                raise KeyError(f"行中缺少必需的键 '{key}': {row}")

        alpha = round(float(row["alpha"]), self.precisions.get("alpha", 2))
        b = int(float(row["b"]))
        table_type = str(row["table_type"]).strip()
        key = (alpha, b, table_type)

        normalized = {"alpha": alpha, "b": b, "table_type": table_type}

        # 处理所有指标
        for metric in self.METRICS:
            if metric in row and row[metric] != "" and row[metric] is not None:
                try:
                    value = float(row[metric])
                    precision = self.precisions.get(metric, 6)
                    normalized[metric] = round(value, precision)
                except (ValueError, TypeError):
                    normalized[metric] = row[metric]
            else:
                normalized[metric] = ""

        return key, normalized
