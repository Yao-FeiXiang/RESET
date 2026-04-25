"""
结果管理器：CSV读写和可视化数据准备
处理结果的读写，并为热力图绘制准备数据。
"""

import os
import csv
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


class ResultManager:
    """
    管理超参数搜索结果：CSV读写和数据准备。

    属性:
        KEY_COLS: 用于去重的主键列
        METRICS: 要跟踪的指标列表
        path: 输出CSV文件路径
        precisions: 保存时每个指标的小数精度
    """

    # 用于识别唯一配置的主键列
    KEY_COLS = ("alpha", "b", "table_type")

    # 可用指标
    METRICS = [
        "cost",  # 来自模型的分析成本
        "kernel_time",  # 测量得到的内核执行时间
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",  # NCU: 全局加载扇区数
    ]

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
        # 确保输出目录存在
        res_dir = os.path.join(os.path.dirname(__file__), "..", "res")
        os.makedirs(res_dir, exist_ok=True)

        self.path = os.path.join(res_dir, filename or f"{dataset_name}.csv")

        # CSV输出的小数精度
        self.precisions = precisions or {
            "alpha": 2,
            "b": 0,
            "cost": 6,
            "kernel_time": 6,
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": 0,
        }

    # ------------------------------------------------------------------
    #                          CSV 输入 / 输出
    # ------------------------------------------------------------------

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
                    if value != "":
                        table[key][field_name] = value
            else:
                table[key] = normalized

        self._write_rows(list(table.values()))

    # ------------------------------------------------------------------
    #             为可视化准备数据：热力图 + 最佳点
    # ------------------------------------------------------------------

    def calculate_best(self, eps: float = 1e-12) -> Dict[str, Dict[str, Any]]:
        """
        将原始结果处理为热力图就绪格式并找出最优点。

        成本模型预测期望成本，我们找出:
        1. 所有(alpha, b)中的全局最优
        2. 每个alpha的最优（每个alpha对应的最佳b）

        参数:
            eps: 浮点比较容差

        返回:
            可供可视化的字典结构:
            {
              "H": {
                "x": [b values],
                "y": [alpha values],
                "Z_metric": {metric: 2D grid},
                "best": {"global": (y,x), "alpha": {(y,x), ...}}
              },
              "N": ...
            }
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

            # ==================================================================
            # 基于分析成本找到最佳配置
            # ==================================================================

            Z_cost = Z_metric["cost"]
            global_best: Optional[Tuple[int, int]] = None
            alpha_local_best: set[Tuple[int, int]] = set()

            # 寻找全局最优（最低成本）
            min_cost = 1e10
            for y in range(len(alphas)):
                for x in range(len(bs)):
                    cost = Z_cost[y][x]
                    if cost is None:
                        continue
                    if cost < min_cost - eps:
                        min_cost = cost
                        global_best = (y, x)

            # 寻找每个alpha的最优（每个alpha对应的最佳b）
            for y in range(len(alphas)):
                row = Z_cost[y]
                valid_points = [(x, cost) for x, cost in enumerate(row) if cost is not None]
                if not valid_points:
                    continue
                # 在本行寻找最小值
                min_row_cost = min(cost for _, cost in valid_points)
                # Find first point matching minimum
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

    # ------------------------------------------------------------------
    #                          Internal Helpers
    # ------------------------------------------------------------------

    def _normalize_row(self, row: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Normalize row: extract key, convert types, apply rounding.

        Returns:
            (key_tuple, normalized_row) where key_tuple = (alpha, b, table_type)
        """
        for key in self.KEY_COLS:
            if key not in row:
                raise KeyError(f"Missing required key '{key}' in row: {row}")

        alpha = float(row["alpha"])
        b = int(float(row["b"]))
        table_type = str(row["table_type"])

        normalized: Dict[str, Any] = {"alpha": alpha, "b": b, "table_type": table_type}

        # Normalize other fields
        for key, value in row.items():
            if key in normalized:
                continue
            if value is None or value == "":
                normalized[key] = ""
                continue

            processed_value = value
            if isinstance(processed_value, str):
                s = processed_value.strip()
                if s == "":
                    normalized[key] = ""
                    continue
                try:
                    processed_value = float(s)
                except ValueError:
                    normalized[key] = processed_value
                    continue

            # Apply precision rounding
            precision = self.precisions.get(key)
            if precision is not None and isinstance(processed_value, (int, float)):
                if precision <= 0:
                    processed_value = int(round(processed_value))
                else:
                    processed_value = round(float(processed_value), precision)

            normalized[key] = processed_value

        return (alpha, b, table_type), normalized

    def _write_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        cleaned_rows: List[Dict[str, Any]] = []
        for r in rows:
            rr = dict(r)
            rr.pop("isBest", None)
            cleaned_rows.append(rr)

        cols = set()
        for r in cleaned_rows:
            cols.update(r.keys())
        fieldnames = list(self.KEY_COLS) + sorted(cols - set(self.KEY_COLS))

        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in cleaned_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
