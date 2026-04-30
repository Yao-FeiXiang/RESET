"""
C-API风格接口:便于外部C/C++程序调用超参数求解器
使用ctypes进行跨语言调用

使用方法：
1. 在C++中包含 hyper_solver.h
2. 调用 solve_hyperparameters(arr_size, mode) 获取推荐的 (alpha, b)
"""

import json
import os
import pickle
import hashlib
from typing import Dict, Tuple, Optional

from solver import HyperparameterSolver

# ============================================================================
#                          缓存配置和路径
# ============================================================================

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "recommended_values.json")


def get_cache_config() -> Dict:
    """
    从config.json加载缓存配置（各参数的常见值列表）

    返回:
        {
            "W_values": [...],
            "S_trans_values": [...],
            "S_slot_values": [...],
            "arr_size_values": [...],
            "mode_values": [...]
        }
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config.get("cache_config", {})
    except Exception:
        # 默认配置
        return {
            "W_values": [32],
            "S_trans_values": [32, 64, 128],
            "S_slot_values": [4, 8, 16],
            "arr_size_values": [
                10_000,
                50_000,
                100_000,
                500_000,
                1_000_000,
                2_000_000,
                5_000_000,
                10_000_000,
            ],
            "mode_values": ["sss", "ir", "tc"],
        }


# 常见GPU硬件配置（用于向后兼容和快速选择）
COMMON_CONFIGS = {
    # ============== 数据中心/Ampere/Hopper架构 ==============
    "A100": {"W": 32, "S_trans": 128, "S_slot": 4},
    "H100": {"W": 32, "S_trans": 128, "S_slot": 4},
    "A10": {"W": 32, "S_trans": 128, "S_slot": 4},
    # ============== Volta/Turing 架构 ==============
    "V100": {"W": 32, "S_trans": 128, "S_slot": 4},
    "T4": {"W": 32, "S_trans": 128, "S_slot": 4},
    # ============== 消费级 ==============
    "4090": {"W": 32, "S_trans": 128, "S_slot": 4},
    "3090": {"W": 32, "S_trans": 128, "S_slot": 4},
}

# 常见数据集规模配置（用于向后兼容）
COMMON_DATA_SIZES = {
    "small": [10_000, 50_000, 100_000],
    "medium": [500_000, 1_000_000, 2_000_000],
    "large": [5_000_000, 10_000_000, 20_000_000],
}


# ============================================================================
#                          缓存管理
# ============================================================================


class RecommendationCache:
    """推荐值缓存管理器"""

    def __init__(self, cache_file: str = None):
        self.cache_file = cache_file or CACHE_FILE
        self.cache: Dict[str, Tuple[float, int, float]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """从磁盘加载缓存（JSON格式，可直接查看）"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 支持两种格式：直接字典或带_metadata的结构
                if isinstance(data, dict) and "entries" in data:
                    raw_cache = data["entries"]
                else:
                    raw_cache = data
                # 确保值是列表格式
                self.cache = {}
                for key, value in raw_cache.items():
                    if key.startswith("_"):  # 跳过元数据键
                        continue
                    if isinstance(value, dict):
                        # 兼容旧格式
                        if "alpha" in value:
                            self.cache[key] = [
                                float(value["alpha"]),
                                int(value["b"]),
                                float(value["cost"]),
                            ]
                    elif isinstance(value, (list, tuple)) and len(value) >= 3:
                        self.cache[key] = [float(value[0]), int(value[1]), float(value[2])]
            except Exception:
                self.cache = {}

    def _save_cache(self) -> None:
        """保存缓存到磁盘（JSON格式，可直接查看）"""
        import datetime

        output = {
            "_metadata": {
                "description": "Hyperparameter Recommendation Cache",
                "note": "可直接查看和编辑",
                "total_entries": len(self.cache),
                "last_updated": datetime.datetime.now().isoformat(),
                "gpu_models_supported": len(COMMON_CONFIGS),
                "data_sizes": sum(len(v) for v in COMMON_DATA_SIZES.values()),
            },
            "entries": self.cache,
        }
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _get_key(arr_size: int, W: int, S_trans: int, S_slot: int, mode: str = "sss") -> str:
        """生成缓存键 - 使用可读格式而非MD5"""
        return f"{mode}_size{arr_size}_W{W}_T{S_trans}_S{S_slot}"

    def get(self, arr_size: int, W: int, S_trans: int, S_slot: int, mode: str = "sss"):
        """
        从缓存获取推荐值

        返回: (alpha, b, cost) 或 None
        """
        key = self._get_key(arr_size, W, S_trans, S_slot, mode)
        if key in self.cache:
            return self.cache[key]
        return None

    def put(
        self,
        arr_size: int,
        W: int,
        S_trans: int,
        S_slot: int,
        alpha: float,
        b: int,
        cost: float,
        mode: str = "sss",
    ) -> None:
        """存入缓存 - JSON使用列表存储元数据"""
        key = self._get_key(arr_size, W, S_trans, S_slot, mode)
        self.cache[key] = [alpha, b, cost]
        self._save_cache()

    def precompute_common_configs(
        self,
        W_values: list = None,
        S_trans_values: list = None,
        S_slot_values: list = None,
        arr_size_values: list = None,
        mode_values: list = None,
    ) -> None:
        """
        预计算所有参数组合的推荐值（笛卡尔积）

        为每个参数提供常见值列表，然后计算所有组合。
        例如：
            W_values = [32]
            S_trans_values = [32, 64, 128]
            S_slot_values = [4, 8, 16]
            arr_size_values = [10000, 100000, 1000000]
            mode_values = ["sss", "ir", "tc"]

        这样将计算 1×3×3×3×3 = 81 种组合。

        参数:
            W_values: warp大小的常见值列表
            S_trans_values: 每次事务字节数的常见值列表
            S_slot_values: 每个槽字节数的常见值列表
            arr_size_values: 数组大小的常见值列表
            mode_values: 应用模式的常见值列表
        """
        from itertools import product

        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, "r") as f:
            base_config = json.load(f)

        # 使用默认配置或用户提供的配置
        cache_config = get_cache_config()
        W_values = W_values or cache_config.get("W_values", [32])
        S_trans_values = S_trans_values or cache_config.get("S_trans_values", [32, 64, 128])
        S_slot_values = S_slot_values or cache_config.get("S_slot_values", [4, 8, 16])
        arr_size_values = arr_size_values or cache_config.get(
            "arr_size_values", [10_000, 100_000, 1_000_000]
        )
        mode_values = mode_values or cache_config.get("mode_values", ["sss"])

        # 生成所有参数组合（笛卡尔积）
        all_combinations = list(
            product(W_values, S_trans_values, S_slot_values, arr_size_values, mode_values)
        )
        total_combinations = len(all_combinations)

        print(f"[预计算] 共 {total_combinations} 种参数组合")
        print(f"  - W_values: {W_values}")
        print(f"  - S_trans_values: {S_trans_values}")
        print(f"  - S_slot_values: {S_slot_values}")
        print(f"  - arr_size_values: {arr_size_values}")
        print(f"  - mode_values: {mode_values}")
        print()

        skipped = 0
        computed = 0

        for idx, (W, S_trans, S_slot, arr_size, mode) in enumerate(all_combinations, 1):
            # 检查是否已在缓存中
            cached = self.get(arr_size, W, S_trans, S_slot, mode)
            if cached is not None:
                skipped += 1
                print(
                    f"[{idx}/{total_combinations}] 跳过 W={W}, S_trans={S_trans}, S_slot={S_slot}, size={arr_size}, mode={mode} (已缓存)"
                )
                continue

            print(
                f"[{idx}/{total_combinations}] 计算 W={W}, S_trans={S_trans}, S_slot={S_slot}, size={arr_size}, mode={mode}..."
            )

            # 构建配置
            config = dict(base_config)
            config["W"] = W
            config["S_trans"] = S_trans
            config["S_slot"] = S_slot

            # 运行求解器
            solver = HyperparameterSolver(config, mode, arr_size=arr_size)
            results = solver.solve()

            # 找出最佳值
            best = min(results, key=lambda r: r["cost"])

            # 存入缓存
            self.put(
                arr_size,
                W,
                S_trans,
                S_slot,
                float(best["alpha"]),
                int(best["b"]),
                float(best["cost"]),
                mode,
            )

            computed += 1
            print(f"  -> alpha={best['alpha']}, b={best['b']}, cost={best['cost']:.4f}")

        print()
        print(f"[完成] 预计算完成！新增 {computed} 条，跳过 {skipped} 条（已缓存）")

    def clear(self) -> None:
        """清空缓存"""
        self.cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)

    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {"total_entries": len(self.cache), "cache_file": self.cache_file}


# ============================================================================
#                          C-API 函数
# ============================================================================

_cache = RecommendationCache()


def solve_hyperparameters(
    arr_size: int,
    W: int = 32,
    S_trans: int = 128,
    S_slot: int = 4,
    use_cache: bool = True,
    mode: bytes = b"sss",
) -> Tuple[float, int, float]:
    """
    C-API: 求解超参数

    参数:
        arr_size: 哈希表条目总数
        W: warp大小 (默认: 32)
        S_trans: 每次事务字节数 (默认: 128)
        S_slot: 每个槽字节数 (默认: 4)
        use_cache: 是否使用缓存 (默认: True)
        mode: 应用模式 (默认: b"sss")

    返回:
        (alpha, b, cost)
    """
    mode_str = mode.decode() if isinstance(mode, bytes) else str(mode)

    # 尝试从缓存获取
    if use_cache:
        cached = _cache.get(arr_size, W, S_trans, S_slot, mode_str)
        if cached is not None:
            return cached

    # 未命中缓存，运行求解器
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    config["W"] = W
    config["S_trans"] = S_trans
    config["S_slot"] = S_slot

    solver = HyperparameterSolver(config, mode_str, arr_size=arr_size)
    results = solver.solve()
    best = min(results, key=lambda r: r["cost"])

    alpha = float(best["alpha"])
    b = int(best["b"])
    cost = float(best["cost"])

    # 存入缓存
    if use_cache:
        _cache.put(arr_size, W, S_trans, S_slot, alpha, b, cost, mode_str)

    return (alpha, b, cost)


def get_cached_hyperparameters(
    arr_size: int, W: int = 32, S_trans: int = 128, S_slot: int = 4, mode: bytes = b"sss"
) -> Tuple[float, int, float]:
    """
    C-API: 仅从缓存获取超参数，不运行求解器

    如果缓存未命中，返回 (0.0, 0, float('inf'))
    """
    mode_str = mode.decode() if isinstance(mode, bytes) else str(mode)
    cached = _cache.get(arr_size, W, S_trans, S_slot, mode_str)
    if cached is not None:
        return cached
    return (0.0, 0, float('inf'))


def precompute_cache() -> None:
    """C-API: 为常见配置预计算缓存"""
    _cache.precompute_common_configs()


def clear_cache() -> None:
    """C-API: 清空缓存"""
    _cache.clear()


def get_cache_stats() -> Dict:
    """C-API: 获取缓存统计"""
    return _cache.get_stats()


def solve_for_dataset(
    dataset_path_bytes: bytes,
    W: int = 32,
    S_trans: int = 128,
    S_slot: int = 4,
    use_cache: bool = True,
    mode: bytes = b"sss",
) -> Tuple[float, int, float]:
    """
    C-API: 从数据集路径求解超参数

    参数:
        dataset_path_bytes: 数据集路径（字节字符串）
        W: warp大小
        S_trans: 每次事务字节数
        S_slot: 每个槽字节数
        use_cache: 是否使用缓存
        mode: 应用模式

    返回:
        (alpha, b, cost)
    """
    from .util import compute_arr_size

    dataset_path = dataset_path_bytes.decode()
    mode_str = mode.decode() if isinstance(mode, bytes) else str(mode)

    # 计算数据集大小
    arr_size = compute_arr_size(dataset_path, mode_str)

    return solve_hyperparameters(arr_size, W, S_trans, S_slot, use_cache, mode)
