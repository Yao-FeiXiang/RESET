"""
局部感知哈希表的分析超参数求解器
预测不同 (alpha, b) 组合的期望成本（期望缓存缺失次数）。

核心优化:
- 多进程并行网格搜索
- 对重复计算使用LRU缓存
- NumPy向量化加速
- 对溢出情况提前剪枝

成本模型基于马尔可夫链溢出分布分析。
"""

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from util import compute_arr_size
from tqdm import tqdm
import numpy as np

# Whether to use penalty on alpha deviating from center
USE_PENALTY = False


class HyperparameterSolver:
    """
    分析超参数求解器，预测网格搜索的期望成本。

    使用概率模型估计不同负载因子(alpha)和桶大小(b)的期望缓存缺失次数。

    属性:
        config: 来自config.json的配置
        arr_size: 哈希表中总条目数
        alpha_min/max/step: 负载因子α的搜索网格
        b_min/max: 桶大小b的搜索范围
        k_max: 最大溢出链长度
        convergence: 迭代求解收敛容差
        W, S_trans, S_slot: 成本计算的硬件参数
            W: warp大小
            S_trans: 每次事务字节数
            S_slot: 每个槽字节数
    """

    def __init__(self, config: dict, mode: str = "sss", dataset_path: str = None, arr_size=None):
        # Load search grid parameters
        self.alpha_min = config["alpha_min"]
        self.alpha_max = config["alpha_max"]
        self.alpha_step = config["alpha_step"]
        self.b_min = config["b_min"]
        self.b_max = config["b_max"]
        self.k_max = config["k_max"]
        self.convergence = config["convergence"]

        # 硬件成本模型参数
        self.W = config["W"]
        self.S_trans = config["S_trans"]
        self.S_slot = config["S_slot"]

        # 从数据集获取数组大小（条目数）
        if arr_size is not None:
            self.arr_size = arr_size
        else:
            self.arr_size = compute_arr_size(dataset_path, mode)
            print(f"[INFO] 计算得到数组大小: {self.arr_size}")

    def get_arr_size(self) -> int:
        """返回哈希表中总条目数。"""
        return self.arr_size

    def solve(self):
        """
        计算整个超参数网格的成本。

        返回:
            结果字典列表: [{"alpha": float, "b": int, "cost": float}, ...]
        """
        # 生成所有要测试的超参数组合
        tasks = []
        steps = int(round((self.alpha_max - self.alpha_min) / self.alpha_step))
        for i in range(steps + 1):
            alpha = round(self.alpha_min + i * self.alpha_step, 2)
            for b in range(self.b_min, self.b_max + 1):
                tasks.append((b, alpha))

        total_points = len(tasks)
        results = []

        # 多进程并行计算（CPU密集型，绕过GIL）
        with ProcessPoolExecutor() as executor:
            # future到参数组合映射
            future_to_params = {
                executor.submit(self._estimate_cost, b, alpha): (alpha, b) for (b, alpha) in tasks
            }

            with tqdm(total=total_points, desc="求解器进度") as pbar:
                for future in as_completed(future_to_params):
                    alpha, b = future_to_params[future]
                    try:
                        cost = future.result()
                        results.append({"alpha": alpha, "b": b, "cost": cost})
                    except Exception as e:
                        print(f"[ERROR] alpha={alpha}, b={b} failed: {e}")
                        # Use a very large cost for failed computations
                        cost = 1e10
                        results.append({"alpha": alpha, "b": b, "cost": cost})
                    pbar.update(1)

        print(f"[INFO] hyperparams solved, total points = {len(results)}")
        return results

    # ------------------------------------------------------------------
    # Cost model
    # ------------------------------------------------------------------

    def penalty(
        self,
        alpha: float,
        arr_size: int,
        alpha_center: float = 0.2,
        width: float = 0.12,
        scale: float = 1.0,
    ) -> float:
        """
        Quadratic penalty on alpha deviating from optimal center.
        Unused by default (USE_PENALTY = False).
        """
        if alpha <= 0:
            return 1e10
        x = math.log(alpha / alpha_center)
        return scale * (x / math.log(1 + width / alpha_center)) ** 2

    def _estimate_cost(self, b: int, alpha: float, probability_model: str = "binomial") -> float:
        """
        Estimate expected cost (cache miss steps) for given (b, alpha).

        The probability model gives the distribution of the number of
        elements mapped to the same bucket.

        Args:
            b: Bucket size (number of slots per bucket)
            alpha: Load factor = n_entries / n_buckets
            probability_model: "binomial" or "poisson"

        Returns:
            Expected cost (lower = better)
        """
        if probability_model == "binomial":
            # Binomial distribution: exact for independent assignments
            n = self.arr_size
            p = alpha * b / n
            u = list(_compute_binomial_probs(n=int(n), p=p, max_j=self.k_max))
        elif probability_model == "poisson":
            # Poisson approximation: good when n is large
            lam = alpha * b
            u = list(_compute_poisson_probs(lam=lam, max_j=self.k_max))
        else:
            raise ValueError(f"Unknown probability model: {probability_model}")

        # Solve overflow distribution using iterative method
        pi = list(
            _solve_overflow_distribution(
                b=b, u_tuple=tuple(u), k_max=self.k_max, convergence=self.convergence
            )
        )
        # Compute total occupancy distribution by convolution
        psi = _compute_psi_m(pi, u, b, self.k_max)
        # Compute statistics: probability bucket is full, expected occupied slots
        P_full, E_S = _compute_bucket_stats(psi, b)

        # Early pruning: if completely overflowed, return large cost
        if P_full >= 1.0 or E_S <= 0:
            return 1e10

        # Early pruning: if almost full, cost will be very large
        if P_full > 0.95:
            return 1e9

        # Expected number of cache misses per lookup
        expected_cache_misses = _compute_E_L_MISS(psi, b, P_full)
        # Cost per cache miss step based on hardware parameters
        step_cost = _compute_T_step(self.W, self.S_slot, self.S_trans, E_S)

        # Convert to native Python float to avoid serialization issues
        total_cost = expected_cache_misses * step_cost
        if USE_PENALTY:
            total_cost += self.penalty(alpha, self.arr_size)

        return float(total_cost)


# ==================================================================
#                 概率辅助函数
# ==================================================================
# 这些函数被缓存，因为网格搜索中经常出现相同的参数组合。
# LRU缓存避免重复计算概率分布。


@lru_cache(maxsize=None)
def _compute_poisson_probs(lam: float, max_j: int) -> tuple[float, ...]:
    """
    计算直到max_j的泊松概率分布。
    P(j个元素 | lambda = alpha*b)。

    当n（总元素数）很大时，泊松是一个很好的近似。
    结果通过LRU缓存缓存。
    """
    u = [0.0] * (max_j + 1)
    u[0] = math.exp(-lam)
    for j in range(1, max_j + 1):
        u[j] = u[j - 1] * lam / j
    return tuple(u)


@lru_cache(maxsize=None)
def _compute_binomial_probs(n: int, p: float, max_j: int) -> tuple[float, ...]:
    """
    计算直到max_j的精确二项概率分布。
    P(j个元素 | n次试验，概率p = alpha*b / n)。

    这是随机独立分配的精确模型。
    结果通过LRU缓存缓存。
    """
    n = int(n)
    p = float(p)
    if n <= 0 or p <= 0.0:
        return tuple([1.0] + [0.0] * max_j)
    if p >= 1.0:
        u = [0.0] * (max_j + 1)
        if n <= max_j:
            u[n] = 1.0
        return tuple(u)

    q = 1.0 - p
    u = [0.0] * (max_j + 1)
    u[0] = q**n
    for j in range(1, min(n, max_j) + 1):
        u[j] = u[j - 1] * (n - j + 1) / j * (p / q)

    # Normalize to ensure sum is 1 (due to truncation at max_j)
    up_to = min(n, max_j)
    s = sum(u[: up_to + 1])
    if max_j < n:
        u[max_j] += max(0.0, 1.0 - s)
    else:
        u[up_to] += max(0.0, 1.0 - s)

    return tuple(u)


@lru_cache(maxsize=None)
def _solve_overflow_distribution(
    b: int, u_tuple: tuple[float, ...], k_max: int, convergence: float
) -> tuple[float, ...]:
    """
    求解溢出数量的平稳分布。

    使用不动点迭代配合NumPy矩阵乘法替换原始嵌套Python循环，获得巨大加速。

    参数:
        b: 桶大小（每个桶的槽数）
        u_tuple: 每个桶输入元素的概率分布
        k_max: 要考虑的最大溢出数
        convergence: 迭代收敛阈值

    返回:
        pi[k] = 一个桶中有k次溢出的概率
    """
    u_np = np.array(u_tuple, dtype=np.float64)
    pi_old = np.zeros(k_max + 1, dtype=np.float64)
    pi_old[0] = 1.0
    max_iter = 10000

    # 预计算前缀和用于高效范围查询
    prefix_u = np.concatenate([[0.0], np.cumsum(u_np)])

    # 预计算转换矩阵T
    # pi_new[k] = sum_j pi_old[j] * u[k + b - j]
    # 所有计算在一次矩阵乘法中完成
    T = np.zeros((k_max + 1, k_max + 1), dtype=np.float64)

    # 第一行特殊情况：范围求和查询
    for j in range(min(b, k_max) + 1):
        T[0, j] = prefix_u[b - j + 1]

    # 使用卷积结构填充其他行
    for k in range(1, k_max + 1):
        max_j = min(k + b, k_max)
        for j in range(max_j + 1):
            idx = k + b - j
            if idx < len(u_np):
                T[k, j] = u_np[idx]

    # 不动点迭代直到收敛
    for _ in range(max_iter):
        # 矩阵乘法一步计算所有pi_new
        # 这替换了原始的双层嵌套Python循环
        pi_new = T @ pi_old

        # Normalize to ensure it's a probability distribution
        norm = np.sum(pi_new)
        if norm > 0:
            pi_new = pi_new / norm

        # Check convergence: maximum difference across all entries
        max_diff = np.max(np.abs(pi_new - pi_old))

        if max_diff < convergence:
            break
        pi_old = pi_new

    # Return as tuple for LRU cache compatibility
    return tuple(pi_old.tolist())


def _compute_psi_m(pi: list[float], u: list[float], b: int, k_max: int) -> list[float]:
    """
    Compute total occupancy distribution by convolution.

    psi[m] = sum_{k} pi[k] * u[m - k] where:
    - pi[k] = probability of k overflows
    - u[m-k] = probability of m-k incoming elements
    - result is probability of total m elements in bucket including overflows

    Uses NumPy convolution for speed.
    """
    pi_np = np.array(pi)
    u_np = np.array(u)
    # We only need the first k_max+1 entries
    psi_full = np.convolve(pi_np, u_np)
    psi = psi_full[: k_max + 1].tolist()
    return psi


def _compute_bucket_stats(psi: list[float], b: int) -> tuple[float, float]:
    """
    Compute key statistics from occupancy distribution.

    Args:
        psi: Probability distribution of occupancy
        b: Bucket capacity (number of slots)

    Returns:
        (P_full, E_S):
        - P_full: Probability bucket is completely full
        - E_S: Expected number of occupied slots
    """
    psi_np = np.array(psi)
    indices = np.arange(len(psi_np))
    P_full = np.sum(psi_np[b:])
    E_S = np.sum(indices * psi_np)
    return P_full, E_S


def _compute_E_L_MISS(psi: list[float], b: int, P_full: float) -> float:
    """
    Compute expected number of cache misses (probes) per lookup.

    Uses the formula: (sum_{m=0}^{b-1} (m+1) psi[m] + b P_full) / (1 - P_full)

    If bucket is full, search continues in overflow chain.
    """
    if P_full >= 1.0:
        return 1e10

    psi_np = np.array(psi[:b])
    m = np.arange(b)
    numerator = P_full * b + np.sum((m + 1) * psi_np)

    return numerator / (1.0 - P_full)


def _compute_T_step(W: int, S_slot: float, S_trans: float, E_S: float) -> float:
    """
    Compute step cost (in memory cycles) based on hardware parameters.

    Args:
        W: Warp size (e.g., 32). Concurrent requests in a warp.
        S_slot: Slot size in bytes.
        S_trans: DRAM transfer size in bytes per cycle.
        E_S: Expected number of slots to check.

    Returns:
        Number of cycles for one probe step.
    """
    if E_S <= 0:
        return 1e10
    return math.ceil((W * S_slot) / (E_S * S_trans)) + 1
