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

# 常见硬件配置（预计算缓存）- 100+个GPU型号
COMMON_CONFIGS = {
    # ============== 数据中心/Ampere/Hopper架构 ==============
    "A100": {"W": 32, "S_trans": 128, "S_slot": 4},
    "A100-40GB": {"W": 32, "S_trans": 128, "S_slot": 4},
    "A100-80GB": {"W": 32, "S_trans": 128, "S_slot": 4},
    "A100-SXM4": {"W": 32, "S_trans": 128, "S_slot": 4},
    "A100-PCIe": {"W": 32, "S_trans": 128, "S_slot": 4},
    "H100": {"W": 32, "S_trans": 128, "S_slot": 4},
    "H100-SXM5": {"W": 32, "S_trans": 128, "S_slot": 4},
    "H100-NVL": {"W": 32, "S_trans": 128, "S_slot": 4},
    "H100-PCIe": {"W": 32, "S_trans": 128, "S_slot": 4},
    "H200": {"W": 32, "S_trans": 128, "S_slot": 4},
    "A10": {"W": 32, "S_trans": 128, "S_slot": 4},
    "A10G": {"W": 32, "S_trans": 128, "S_slot": 4},
    "A30": {"W": 32, "S_trans": 128, "S_slot": 4},
    "A2": {"W": 32, "S_trans": 128, "S_slot": 4},
    "A16": {"W": 32, "S_trans": 128, "S_slot": 4},
    "A40": {"W": 32, "S_trans": 128, "S_slot": 4},
    # ============== Volta/Turing 架构 ==============
    "V100": {"W": 32, "S_trans": 128, "S_slot": 4},
    "V100-16GB": {"W": 32, "S_trans": 128, "S_slot": 4},
    "V100-32GB": {"W": 32, "S_trans": 128, "S_slot": 4},
    "V100-SXM2": {"W": 32, "S_trans": 128, "S_slot": 4},
    "V100-PCIe": {"W": 32, "S_trans": 128, "S_slot": 4},
    "T4": {"W": 32, "S_trans": 128, "S_slot": 4},
    "T4G": {"W": 32, "S_trans": 128, "S_slot": 4},
    # ============== 消费级/专业级 - RTX 40系列 ==============
    "4090": {"W": 32, "S_trans": 128, "S_slot": 4},
    "RTX4090": {"W": 32, "S_trans": 128, "S_slot": 4},
    "4080": {"W": 32, "S_trans": 128, "S_slot": 4},
    "RTX4080": {"W": 32, "S_trans": 128, "S_slot": 4},
    "4080Super": {"W": 32, "S_trans": 128, "S_slot": 4},
    "4070": {"W": 32, "S_trans": 128, "S_slot": 4},
    "4070Ti": {"W": 32, "S_trans": 128, "S_slot": 4},
    "4070Super": {"W": 32, "S_trans": 128, "S_slot": 4},
    "4070TiSuper": {"W": 32, "S_trans": 128, "S_slot": 4},
    "4060": {"W": 32, "S_trans": 128, "S_slot": 4},
    "4060Ti": {"W": 32, "S_trans": 128, "S_slot": 4},
    "4060Ti-8GB": {"W": 32, "S_trans": 128, "S_slot": 4},
    "4060Ti-16GB": {"W": 32, "S_trans": 128, "S_slot": 4},
    "4050": {"W": 32, "S_trans": 128, "S_slot": 4},
    # ============== 消费级/专业级 - RTX 30系列 ==============
    "3090": {"W": 32, "S_trans": 128, "S_slot": 4},
    "RTX3090": {"W": 32, "S_trans": 128, "S_slot": 4},
    "3090Ti": {"W": 32, "S_trans": 128, "S_slot": 4},
    "3080": {"W": 32, "S_trans": 128, "S_slot": 4},
    "3080Ti": {"W": 32, "S_trans": 128, "S_slot": 4},
    "3070": {"W": 32, "S_trans": 128, "S_slot": 4},
    "3070Ti": {"W": 32, "S_trans": 128, "S_slot": 4},
    "3060": {"W": 32, "S_trans": 128, "S_slot": 4},
    "3060Ti": {"W": 32, "S_trans": 128, "S_slot": 4},
    "3050": {"W": 32, "S_trans": 128, "S_slot": 4},
    # ============== 消费级/专业级 - RTX 20系列 ==============
    "2080Ti": {"W": 32, "S_trans": 128, "S_slot": 4},
    "2080": {"W": 32, "S_trans": 128, "S_slot": 4},
    "2080Super": {"W": 32, "S_trans": 128, "S_slot": 4},
    "2070": {"W": 32, "S_trans": 128, "S_slot": 4},
    "2070Super": {"W": 32, "S_trans": 128, "S_slot": 4},
    "2060": {"W": 32, "S_trans": 128, "S_slot": 4},
    "2060Super": {"W": 32, "S_trans": 128, "S_slot": 4},
    # ============== 消费级/专业级 - GTX 10系列 ==============
    "1080Ti": {"W": 32, "S_trans": 128, "S_slot": 4},
    "1080": {"W": 32, "S_trans": 128, "S_slot": 4},
    "1070Ti": {"W": 32, "S_trans": 128, "S_slot": 4},
    "1070": {"W": 32, "S_trans": 128, "S_slot": 4},
    "1060": {"W": 32, "S_trans": 128, "S_slot": 4},
    "1060-6GB": {"W": 32, "S_trans": 128, "S_slot": 4},
    "1060-3GB": {"W": 32, "S_trans": 128, "S_slot": 4},
    "1050": {"W": 32, "S_trans": 128, "S_slot": 4},
    "1050Ti": {"W": 32, "S_trans": 128, "S_slot": 4},
    # ============== 专业级工作站 ==============
    "RTX6000Ada": {"W": 32, "S_trans": 128, "S_slot": 4},
    "RTX5000Ada": {"W": 32, "S_trans": 128, "S_slot": 4},
    "RTX4000Ada": {"W": 32, "S_trans": 128, "S_slot": 4},
    "RTX4000SFFAda": {"W": 32, "S_trans": 128, "S_slot": 4},
    "RTX6000": {"W": 32, "S_trans": 128, "S_slot": 4},
    "RTX5000": {"W": 32, "S_trans": 128, "S_slot": 4},
    "RTX4000": {"W": 32, "S_trans": 128, "S_slot": 4},
    "RTX3000": {"W": 32, "S_trans": 128, "S_slot": 4},
    "A6000": {"W": 32, "S_trans": 128, "S_slot": 4},
    "A5000": {"W": 32, "S_trans": 128, "S_slot": 4},
    "A4000": {"W": 32, "S_trans": 128, "S_slot": 4},
    "A2000": {"W": 32, "S_trans": 128, "S_slot": 4},
    # ============== 移动版/笔记本 ==============
    "RTX4090Laptop": {"W": 32, "S_trans": 128, "S_slot": 4},
    "RTX4080Laptop": {"W": 32, "S_trans": 128, "S_slot": 4},
    "RTX4070Laptop": {"W": 32, "S_trans": 128, "S_slot": 4},
    "RTX4060Laptop": {"W": 32, "S_trans": 128, "S_slot": 4},
    "RTX4050Laptop": {"W": 32, "S_trans": 128, "S_slot": 4},
    "RTX3080Laptop": {"W": 32, "S_trans": 128, "S_slot": 4},
    "RTX3070Laptop": {"W": 32, "S_trans": 128, "S_slot": 4},
    "RTX3060Laptop": {"W": 32, "S_trans": 128, "S_slot": 4},
    "RTX3050Laptop": {"W": 32, "S_trans": 128, "S_slot": 4},
    # ============== 嵌入式/Jetson ==============
    "JetsonOrin": {"W": 32, "S_trans": 128, "S_slot": 4},
    "JetsonOrinNX": {"W": 32, "S_trans": 128, "S_slot": 4},
    "JetsonOrinNano": {"W": 32, "S_trans": 128, "S_slot": 4},
    "JetsonXavier": {"W": 32, "S_trans": 128, "S_slot": 4},
    "JetsonXavierNX": {"W": 32, "S_trans": 128, "S_slot": 4},
    "JetsonNano": {"W": 32, "S_trans": 128, "S_slot": 4},
    "JetsonTX2": {"W": 32, "S_trans": 128, "S_slot": 4},
    "JetsonTX1": {"W": 32, "S_trans": 128, "S_slot": 4},
    # ============== 云计算常用别名 ==============
    "AWS-G4dn": {"W": 32, "S_trans": 128, "S_slot": 4},
    "AWS-G5": {"W": 32, "S_trans": 128, "S_slot": 4},
    "AWS-G6": {"W": 32, "S_trans": 128, "S_slot": 4},
    "AWS-P3": {"W": 32, "S_trans": 128, "S_slot": 4},
    "AWS-P4d": {"W": 32, "S_trans": 128, "S_slot": 4},
    "AWS-P4de": {"W": 32, "S_trans": 128, "S_slot": 4},
    "AWS-P5": {"W": 32, "S_trans": 128, "S_slot": 4},
    "GCP-A2": {"W": 32, "S_trans": 128, "S_slot": 4},
    "GCP-A3": {"W": 32, "S_trans": 128, "S_slot": 4},
    "GCP-T4": {"W": 32, "S_trans": 128, "S_slot": 4},
    "GCP-V100": {"W": 32, "S_trans": 128, "S_slot": 4},
    "Azure-NC": {"W": 32, "S_trans": 128, "S_slot": 4},
    "Azure-NC24": {"W": 32, "S_trans": 128, "S_slot": 4},
    "Azure-ND": {"W": 32, "S_trans": 128, "S_slot": 4},
    "Azure-ND96": {"W": 32, "S_trans": 128, "S_slot": 4},
    "Azure-NV": {"W": 32, "S_trans": 128, "S_slot": 4},
}

# 常见数据集规模配置
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

    def precompute_common_configs(self) -> None:
        """为常见硬件配置和数据规模预计算推荐值"""
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, "r") as f:
            base_config = json.load(f)

        for gpu_name, hw_params in COMMON_CONFIGS.items():
            for size_category, sizes in COMMON_DATA_SIZES.items():
                for arr_size in sizes:
                    # 检查是否已在缓存中
                    cached = self.get(
                        arr_size, hw_params["W"], hw_params["S_trans"], hw_params["S_slot"], "sss"
                    )
                    if cached is not None:
                        print(f"[缓存] 跳过 GPU={gpu_name}, size={arr_size} (已存在)")
                        continue

                    print(f"[预计算] GPU={gpu_name}, size={arr_size}...")

                    # 构建配置
                    config = dict(base_config)
                    config.update(hw_params)

                    # 运行求解器
                    solver = HyperparameterSolver(config, "sss", arr_size=arr_size)
                    results = solver.solve()

                    # 找出最佳值
                    best = min(results, key=lambda r: r["cost"])

                    # 存入缓存
                    self.put(
                        arr_size,
                        hw_params["W"],
                        hw_params["S_trans"],
                        hw_params["S_slot"],
                        float(best["alpha"]),
                        int(best["b"]),
                        float(best["cost"]),
                        "sss",
                    )

                    print(f"  -> alpha={best['alpha']}, b={best['b']}, cost={best['cost']:.4f}")

        print("[完成] 所有常见配置预计算完成！")

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
