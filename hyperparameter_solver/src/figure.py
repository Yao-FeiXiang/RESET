"""
结果可视化器：为超参数搜索结果生成热力图。
生成发布质量的图表，展示成本、内核时间和硬件性能指标。
"""

import os
import re
import math
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import seaborn as sns

from util import auto_boundaries
from result import ResultManager

# ===========================================================================
#                               全局绘图样式
# ===========================================================================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.use14corefonts": False,
        "text.usetex": False,
        # 字体大小
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.linewidth": 1.2,
        "savefig.transparent": False,
        "image.composite_image": False,
    }
)


class ResultVisualizer:
    """
    从超参数搜索结果生成发布质量的热力图。

    属性:
        manager: 包含结果的ResultManager
        table_type: 要可视化的表布局 ("N"=普通, "H"=分层)
        dataset: 数据集名称（用于文件名）
        out_dir: 图表输出目录
        best_data: 从ResultManager处理后的最佳数据
    """

    # 坐标轴和指标的可读名称（保留英文，方便发表）
    DISPLAY_NAMES = {
        "alpha": "Load Factor (α)",
        "b": "Bucket Size (b)",
        "cost": "Computed Cost",
        "kernel_time": "Kernel Time (s)",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "Global Load Sectors",
    }

    # 每个指标的配置：格式、颜色映射、颜色边界
    METRIC_CONFIG: Dict[str, Dict[str, Any]] = {
        "cost": {
            "format": ".2f",
            "colormap": "viridis_r",
            "boundaries": {"mode": "log-quantile", "n_bins": 11, "clip_quantiles": (0.05, 0.85)},
            "scientific_notation": False,
        },
        "kernel_time": {
            "format": ".3f",
            "colormap": "viridis_r",
            "boundaries": {"mode": "log-quantile", "n_bins": 10, "clip_quantiles": (0.06, 0.85)},
            "scientific_notation": False,
        },
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": {
            "format": ".3f",
            "colormap": "viridis_r",
            "boundaries": {"mode": "log-quantile", "n_bins": 10, "clip_quantiles": (0.03, 0.95)},
            "scientific_notation": True,
        },
    }

    # NCU全局加载扇区指标键
    GLOBAL_LOAD_KEY = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"

    def __init__(
        self, manager: ResultManager, default_table_type: str = "H", save_dir: Optional[str] = None
    ):
        """
        初始化可视化器。

        参数:
            manager: 包含加载结果的ResultManager
            default_table_type: 默认要绘制的表布局 ("N" 或 "H")
            save_dir: 自定义输出目录 (默认: ../figure/)
        """
        self.manager = manager
        self.table_type = default_table_type

        # 从路径提取数据集名称
        self.dataset_name = (
            getattr(manager, "dataset_name", None)
            or os.path.splitext(os.path.basename(manager.path))[0]
        )

        # 创建输出目录
        self.out_dir = save_dir or os.path.join(os.path.dirname(__file__), "..", "figure")
        os.makedirs(self.out_dir, exist_ok=True)

        # 从管理器获取处理后的数据
        self.processed_data = self.manager.calculate_best()
        if default_table_type not in self.processed_data:
            raise KeyError(
                f"table_type='{default_table_type}' 未找到，可用={list(self.processed_data.keys())}"
            )

    # ===========================================================================
    #                               辅助工具
    # ===========================================================================

    @staticmethod
    def _slugify(filename: str) -> str:
        """将字符串转换为安全的文件名。"""
        s = filename.replace(os.sep, "_")
        s = re.sub(r"[^A-Za-z0-9]+", "_", s)
        return re.sub(r"_+", "_", s).strip("_").lower()

    def _build_colormap_and_norm(
        self, Z: np.ndarray, cmap_name: str, boundary_config: Dict[str, Any]
    ) -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        """构建自定义颜色映射和边界归一化。"""
        df = pd.DataFrame(Z)
        boundaries = auto_boundaries(
            df,
            n_bins=boundary_config.get("n_bins", 10),
            clip_q=boundary_config.get("clip_quantiles", (0.02, 0.98)),
            mode=boundary_config.get("mode", "quantile"),
            min_step=boundary_config.get("min_step", 1e-12),
            keep_clip=boundary_config.get("keep_clip", True),
        )
        cmap = mcolors.ListedColormap(sns.color_palette(cmap_name, len(boundaries) - 1))
        norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
        return cmap, norm

    @staticmethod
    def _auto_figure_size(n_x: int, n_y: int) -> Tuple[float, float]:
        """根据网格尺寸自动选择图表大小。"""
        cell_width = 0.65
        cell_height = 0.35
        width = max(7.5, min(16.0, 2.6 + n_x * cell_width))
        height = max(4.2, min(9.5, 1.6 + n_y * cell_height))
        return width, height

    @staticmethod
    def _auto_annotation_font_size(n_cells: int) -> int:
        """根据单元格数量自动选择注释字体大小。"""
        if n_cells <= 120:
            return 16
        if n_cells <= 240:
            return 14
        return 12

    # ===========================================================================
    #                            核心绘图函数
    # ===========================================================================

    def _plot_single_metric(
        self, metric: str, mark_alpha_best: bool, mark_global_best: bool, name=None
    ):
        cfg = self.METRIC_CONFIG[metric]
        data = self.processed_data[self.table_type]

        Z_raw = np.array(data["Z_metric"][metric], dtype=float)
        alphas = list(data["y"])
        bs = list(data["x"])
        best_points = data["best"]

        # ---------- scientific scaling ----------
        s = s.replace(os.sep, "_")
        s = re.sub(r"[^A-Za-z0-9]+", "_", s)
        return re.sub(r"_+", "_", s).strip("_").lower()

    def _build_norm_and_cmap(
        self, Z: np.ndarray, cmap_name: str, boundary_cfg: Dict[str, Any]
    ) -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
        df = pd.DataFrame(Z)
        boundaries = auto_boundaries(
            df,
            n_bins=boundary_cfg.get("n_bins", 10),
            clip_q=boundary_cfg.get("clip_q", (0.02, 0.98)),
            mode=boundary_cfg.get("mode", "quantile"),
            min_step=boundary_cfg.get("min_step", 1e-12),
            keep_clip=boundary_cfg.get("keep_clip", True),
        )
        cmap = mcolors.ListedColormap(sns.color_palette(cmap_name, len(boundaries) - 1))
        norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
        return cmap, norm

    @staticmethod
    def _auto_figsize(n_x: int, n_y: int) -> Tuple[float, float]:
        cell_w = 0.65
        cell_h = 0.35
        w = max(7.5, min(16.0, 2.6 + n_x * cell_w))
        h = max(4.2, min(9.5, 1.6 + n_y * cell_h))
        return w, h

    @staticmethod
    def _auto_annot_size(n_cells: int) -> int:
        if n_cells <= 120:
            return 16
        if n_cells <= 240:
            return 14
        return 12

    # ========================================================
    # ===================== Plot Core ========================
    # ========================================================

    def _plot_metric(self, metric: str, mark_alpha_best: bool, mark_global_best: bool, name=None):
        cfg = self.METRIC_CFG[metric]
        data = self.best_data[self.table_type]

        Z_raw = np.array(data["Z_metric"][metric], dtype=float)
        alphas = list(data["y"])
        bs = list(data["x"])
        best = data["best"]

        # ---------- scientific scaling ----------
        Z = Z_raw
        scale_exp = None
        if cfg.get("sci", False):
            vmax = np.nanmax(Z_raw)
            if np.isfinite(vmax) and vmax > 0:
                scale_exp = int(math.floor(math.log10(vmax)))
                Z = Z_raw / (10**scale_exp)

        n_y, n_x = Z.shape
        figsize = self._auto_figsize(n_x=n_x, n_y=n_y)
        annot_size = self._auto_annot_size(n_cells=n_x * n_y)

        fig, ax = plt.subplots(figsize=figsize)

        cmap, norm = self._build_norm_and_cmap(
            Z, cmap_name=cfg["cmap"], boundary_cfg=cfg["boundaries"]
        )

        hm = sns.heatmap(
            Z,
            ax=ax,
            cmap=cmap,
            norm=norm,
            annot=True,
            fmt=cfg["fmt"],
            cbar_kws={"shrink": 0.85, "pad": 0.02},
            annot_kws={"size": annot_size},
            linewidths=0.40,
            linecolor="black",
            square=False,
        )
        if ax.collections:
            ax.collections[0].set_antialiased(False)

        # ax.set_aspect(0.75)
        # ---------- ticks ----------
        ax.set_xticks(np.arange(n_x) + 0.5)
        ax.set_yticks(np.arange(n_y) + 0.5)
        ax.set_xticklabels(bs, rotation=0)
        ax.set_yticklabels(alphas, rotation=0)

        ax.set_xlabel(self.DISPLAY["b"])
        ax.set_ylabel(self.DISPLAY["alpha"])

        # ---------- colorbar format ----------
        cbar = ax.collections[0].colorbar
        cbar.formatter = mticker.FormatStrFormatter("%.2f")
        cbar.update_ticks()

        # ---------- best markers ----------

        # ---------- best markers (ULTRA HIGH CONTRAST) ----------

        if mark_global_best and best.get("global") is not None:
            y, x = best["global"]
            ax.add_patch(
                patches.Rectangle((x, y), 1, 1, fill=False, edgecolor="red", linewidth=3.2)
            )

        if mark_alpha_best:
            for y, x in best.get("alpha", []):
                a = alphas[y]
                if 0.1 <= float(a) <= 1.0:
                    ax.add_patch(
                        patches.Rectangle(
                            (x, y),
                            1,
                            1,
                            fill=False,
                            edgecolor="#e9ad6c",
                            linewidth=2.0,
                            linestyle="-",
                            joinstyle="round",
                            capstyle="round",
                            zorder=10,
                        )
                    )

        # ---------- title ----------
        title = f"{self.dataset} {self.DISPLAY[metric]}"
        if scale_exp is not None:
            title += rf"  ($\times 10^{{{scale_exp}}}$)"
        ax.set_title(title, pad=12)

        self._save(fig, metric, name)
        plt.close(fig)

    # ========================================================
    # ===================== Public API =======================
    # ========================================================

    def plot_cost(self):
        self._plot_metric("cost", mark_alpha_best=True, mark_global_best=True)

    def plot_kernel_time(self):
        self._plot_metric("kernel_time", mark_alpha_best=True, mark_global_best=True)

    def plot_load_sector(self):
        self._plot_metric(
            self.LOAD_SECTOR_KEY, mark_alpha_best=True, mark_global_best=True, name="load_sector"
        )

    def plot_all(self):
        self.plot_cost()
        self.plot_kernel_time()
        self.plot_load_sector()

    # ========================================================
    # ======================= Save ===========================
    # ========================================================

    def _save(self, fig: plt.Figure, metric: str, name=None):
        if name is None:
            name = f"{self._slug(self.dataset)}_{self._slug(metric)}_heatmap.pdf"
        else:
            name = f"{self._slug(self.dataset)}_{name}_heatmap.pdf"
        out_path = os.path.join(self.out_dir, name)

        fig.tight_layout()

        fig.savefig(
            out_path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.02,
            transparent=False,
            metadata={"Creator": "matplotlib"},
        )


# ============================================================
# ========================== Main ============================
# ============================================================


def generate_all():
    datasets = ["bm", "gp", "sc18", "sc19", "sc20", "wt"]
    for name in datasets:
        generate(name)


def generate(dataset_name: str):
    m = ResultManager(dataset_name=dataset_name)
    ResultVisualizer(m).plot_all()


if __name__ == "__main__":
    generate("sc18")
    # generate_all()
