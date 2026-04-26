"""
结果可视化器 v2.0：为超参数搜索结果生成热力图
生成发布质量的图表，展示成本、内核时间和硬件性能指标。

优化内容：
1. 移除重复代码，统一命名规范
2. 改进配置结构，增强可扩展性
3. 添加批量绘图支持
4. 优化色阶和标注样式
5. 增加自定义图表选项
"""

import os
import re
import math
from typing import Optional, Dict, Any, Tuple, List

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
#                          全局绘图样式配置
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


# ===========================================================================
#                          图表配置类
# ===========================================================================


class PlotConfig:
    """绘图配置：统一管理所有图表相关配置"""

    # 坐标轴和指标的可读名称
    DISPLAY_NAMES = {
        "alpha": "Load Factor (α)",
        "b": "Bucket Size (b)",
        "cost": "Computed Cost",
        "kernel_time": "Kernel Time (s)",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "Global Load Sectors",
    }

    # 每个指标的详细配置
    METRIC_CONFIG: Dict[str, Dict[str, Any]] = {
        "cost": {
            "format": ".2f",
            "colormap": "viridis_r",
            "boundaries": {"mode": "log-quantile", "n_bins": 11, "clip_quantiles": (0.05, 0.85)},
            "scientific_notation": False,
            "short_name": "cost",
        },
        "kernel_time": {
            "format": ".3f",
            "colormap": "viridis_r",
            "boundaries": {"mode": "log-quantile", "n_bins": 10, "clip_quantiles": (0.06, 0.85)},
            "scientific_notation": False,
            "short_name": "time",
        },
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": {
            "format": ".3f",
            "colormap": "viridis_r",
            "boundaries": {"mode": "log-quantile", "n_bins": 10, "clip_quantiles": (0.03, 0.95)},
            "scientific_notation": True,
            "short_name": "load",
        },
    }

    # 最佳点标注样式
    MARKER_STYLES = {
        "global": {"edgecolor": "#dc143c", "linewidth": 3.5, "linestyle": "-"},  # 深红色
        "alpha": {"edgecolor": "#ff8c00", "linewidth": 2.2, "linestyle": "-"},  # 深橙色
    }

    # 图表大小计算参数
    SIZE_PARAMS = {
        "cell_width": 0.65,
        "cell_height": 0.35,
        "min_width": 7.5,
        "max_width": 16.0,
        "min_height": 4.2,
        "max_height": 9.5,
        "base_width": 2.6,
        "base_height": 1.6,
    }

    # 标注字体大小阈值
    FONT_SIZE_THRESHOLDS = [(120, 16), (240, 14), (float("inf"), 12)]


# ===========================================================================
#                          主可视化器类
# ===========================================================================


class HeatmapVisualizer:
    """
    从超参数搜索结果生成发布质量的热力图。

    使用方法:
        manager = ResultManager(dataset_name="sc18")
        viz = HeatmapVisualizer(manager)
        viz.plot_all()  # 绘制所有指标
        # 或单独绘制
        viz.plot_cost()
        viz.plot_kernel_time()
        viz.plot_load_sectors()
    """

    GLOBAL_LOAD_KEY = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"

    def __init__(
        self,
        manager: ResultManager,
        table_type: str = "H",
        save_dir: Optional[str] = None,
        config: PlotConfig = None,
    ):
        """
        初始化可视化器。

        参数:
            manager: 包含加载结果的ResultManager
            table_type: 要绘制的表布局 ("N"=普通, "H"=分层)
            save_dir: 自定义输出目录 (默认: ../figure/)
            config: 自定义绘图配置
        """
        self.manager = manager
        self.table_type = table_type
        self.config = config or PlotConfig()

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
        if table_type not in self.processed_data:
            available = list(self.processed_data.keys())
            raise KeyError(f"table_type='{table_type}' 未找到，可用选项: {available}")

        self.data = self.processed_data[table_type]

    # ===================================================================
    #                          工具函数
    # ===================================================================

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

    def _calculate_figure_size(self, n_x: int, n_y: int) -> Tuple[float, float]:
        """根据网格尺寸自动计算图表大小。"""
        p = self.config.SIZE_PARAMS
        width = max(p["min_width"], min(p["max_width"], p["base_width"] + n_x * p["cell_width"]))
        height = max(
            p["min_height"], min(p["max_height"], p["base_height"] + n_y * p["cell_height"])
        )
        return width, height

    def _calculate_annotation_size(self, n_cells: int) -> int:
        """根据单元格数量自动计算标注字体大小。"""
        for threshold, size in self.config.FONT_SIZE_THRESHOLDS:
            if n_cells <= threshold:
                return size
        return 12

    # ===================================================================
    #                          核心绘图逻辑
    # ===================================================================

    def _plot_heatmap(
        self,
        metric: str,
        mark_alpha_best: bool = True,
        mark_global_best: bool = True,
        custom_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        绘制单个指标的热力图。

        参数:
            metric: 要绘制的指标名称
            mark_alpha_best: 是否标记每个alpha的最佳b
            mark_global_best: 是否标记全局最佳点
            custom_name: 自定义输出文件名
            **kwargs: 额外的绘图参数
        """
        cfg = self.config.METRIC_CONFIG[metric]
        Z_raw = np.array(self.data["Z_metric"][metric], dtype=float)
        alphas = list(self.data["y"])
        bs = list(self.data["x"])
        best_points = self.data["best"]

        # 科学计数法缩放
        Z = Z_raw
        scale_exp = None
        if cfg.get("scientific_notation", False):
            vmax = np.nanmax(Z_raw)
            if np.isfinite(vmax) and vmax > 0:
                scale_exp = int(math.floor(math.log10(vmax)))
                Z = Z_raw / (10**scale_exp)

        n_y, n_x = Z.shape
        figsize = self._calculate_figure_size(n_x, n_y)
        annot_size = self._calculate_annotation_size(n_x * n_y)

        fig, ax = plt.subplots(figsize=figsize)

        # 构建色图和归一化
        cmap, norm = self._build_colormap_and_norm(
            Z, cmap_name=cfg["colormap"], boundary_config=cfg["boundaries"]
        )

        # 绘制热力图
        heatmap = sns.heatmap(
            Z,
            ax=ax,
            cmap=cmap,
            norm=norm,
            annot=True,
            fmt=cfg["format"],
            cbar_kws={"shrink": 0.85, "pad": 0.02},
            annot_kws={"size": annot_size},
            linewidths=0.40,
            linecolor="black",
            square=False,
            **kwargs,
        )

        # 抗锯齿设置
        if ax.collections:
            ax.collections[0].set_antialiased(False)

        # 坐标轴设置
        ax.set_xticks(np.arange(n_x) + 0.5)
        ax.set_yticks(np.arange(n_y) + 0.5)
        ax.set_xticklabels(bs, rotation=0)
        ax.set_yticklabels(alphas, rotation=0)

        ax.set_xlabel(self.config.DISPLAY_NAMES["b"])
        ax.set_ylabel(self.config.DISPLAY_NAMES["alpha"])

        # 色标格式设置
        cbar = ax.collections[0].colorbar
        cbar.formatter = mticker.FormatStrFormatter("%.2f")
        cbar.update_ticks()

        # 标记最佳点
        if mark_global_best and best_points.get("global") is not None:
            y, x = best_points["global"]
            style = self.config.MARKER_STYLES["global"]
            ax.add_patch(patches.Rectangle((x, y), 1, 1, fill=False, **style))

        if mark_alpha_best:
            style = self.config.MARKER_STYLES["alpha"]
            for y, x in best_points.get("alpha", []):
                a = alphas[y]
                if 0.1 <= float(a) <= 1.0:  # 仅在有效范围内标记
                    ax.add_patch(patches.Rectangle((x, y), 1, 1, fill=False, **style))

        # 标题设置
        title = f"{self.dataset_name} {self.config.DISPLAY_NAMES[metric]}"
        if scale_exp is not None:
            title += rf"  ($\times 10^{{{scale_exp}}}$)"
        ax.set_title(title, pad=12)

        # 保存图表
        self._save_figure(fig, metric, custom_name)
        plt.close(fig)

    # ===================================================================
    #                          保存和导出
    # ===================================================================

    def _save_figure(self, fig: plt.Figure, metric: str, custom_name: Optional[str] = None) -> None:
        """保存图表到文件。"""
        if custom_name is None:
            short_name = self.config.METRIC_CONFIG[metric].get("short_name", metric)
            filename = f"{self._slugify(self.dataset_name)}_{short_name}_heatmap.pdf"
        else:
            filename = (
                f"{self._slugify(self.dataset_name)}_{self._slugify(custom_name)}_heatmap.pdf"
            )

        out_path = os.path.join(self.out_dir, filename)

        fig.tight_layout()
        fig.savefig(
            out_path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.02,
            transparent=False,
            metadata={"Creator": "HeatmapVisualizer"},
        )

    # ===================================================================
    #                          公开API方法
    # ===================================================================

    def plot_cost(self, **kwargs) -> None:
        """绘制预测成本热力图。"""
        self._plot_heatmap("cost", **kwargs)

    def plot_kernel_time(self, **kwargs) -> None:
        """绘制内核执行时间热力图。"""
        self._plot_heatmap("kernel_time", **kwargs)

    def plot_load_sectors(self, **kwargs) -> None:
        """绘制全局加载扇区热力图。"""
        self._plot_heatmap(self.GLOBAL_LOAD_KEY, custom_name="load_sector", **kwargs)

    def plot_all(self, metrics: Optional[List[str]] = None) -> None:
        """
        绘制所有或指定指标的热力图。

        参数:
            metrics: 要绘制的指标列表，None表示全部
        """
        if metrics is None:
            self.plot_cost()
            self.plot_kernel_time()
            self.plot_load_sectors()
        else:
            for metric in metrics:
                if metric == "load_sectors":
                    self.plot_load_sectors()
                elif metric in self.config.METRIC_CONFIG:
                    self._plot_heatmap(metric)

    def compare_metrics(self, metrics: List[str], save_name: str = "comparison") -> None:
        """
        在同一个图中并排比较多个指标（用于论文插图）。

        参数:
            metrics: 要比较的指标列表
            save_name: 输出文件名
        """
        n_metrics = len(metrics)
        if n_metrics == 0:
            return

        # 创建子图网格
        fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]

        for idx, (ax, metric) in enumerate(zip(axes, metrics)):
            cfg = self.config.METRIC_CONFIG[metric]
            Z_raw = np.array(self.data["Z_metric"][metric], dtype=float)
            alphas = list(self.data["y"])
            bs = list(self.data["x"])

            # 科学计数法缩放
            Z = Z_raw
            scale_exp = None
            if cfg.get("scientific_notation", False):
                vmax = np.nanmax(Z_raw)
                if np.isfinite(vmax) and vmax > 0:
                    scale_exp = int(math.floor(math.log10(vmax)))
                    Z = Z_raw / (10**scale_exp)

            cmap, norm = self._build_colormap_and_norm(
                Z, cmap_name=cfg["colormap"], boundary_config=cfg["boundaries"]
            )

            annot_size = self._calculate_annotation_size(Z.shape[0] * Z.shape[1])

            sns.heatmap(
                Z,
                ax=ax,
                cmap=cmap,
                norm=norm,
                annot=True,
                fmt=cfg["format"],
                cbar_kws={"shrink": 0.8, "pad": 0.02},
                annot_kws={"size": annot_size - 2},
                linewidths=0.35,
                linecolor="black",
                square=False,
            )

            # 坐标轴设置
            ax.set_xticklabels(bs, rotation=0)
            ax.set_yticklabels(alphas, rotation=0)

            # 仅在第一个子图显示y轴标签
            ax.set_xlabel(self.config.DISPLAY_NAMES["b"], fontsize=16)
            if idx == 0:
                ax.set_ylabel(self.config.DISPLAY_NAMES["alpha"], fontsize=16)
            else:
                ax.set_ylabel("")

            # 标题
            title = self.config.DISPLAY_NAMES[metric]
            if scale_exp is not None:
                title += rf" ($\times 10^{{{scale_exp}}}$)"
            ax.set_title(title, fontsize=18, pad=10)

        fig.suptitle(f"{self.dataset_name} - 指标对比", fontsize=20, y=1.02)
        fig.tight_layout()

        out_path = os.path.join(self.out_dir, f"{self._slugify(self.dataset_name)}_{save_name}.pdf")
        fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02, transparent=False)
        plt.close(fig)


# ===========================================================================
#                          便捷函数
# ===========================================================================


def generate_heatmaps(
    dataset_name: str,
    res_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    table_type: str = "H",
) -> None:
    """
    为指定数据集生成所有热力图。

    参数:
        dataset_name: 数据集名称
        res_dir: 结果CSV文件目录
        save_dir: 图表输出目录
        table_type: 表类型 ("N" 或 "H")
    """
    manager = ResultManager(dataset_name=dataset_name)
    if res_dir:
        manager.path = os.path.join(res_dir, f"{dataset_name}.csv")

    viz = HeatmapVisualizer(manager, table_type=table_type, save_dir=save_dir)
    viz.plot_all()
    print(f"[完成] {dataset_name} 所有图表已生成")


def generate_all_datasets(datasets: List[str] = None, table_type: str = "H") -> None:
    """
    为所有数据集批量生成热力图。

    参数:
        datasets: 数据集名称列表，None使用默认列表
        table_type: 表类型
    """
    if datasets is None:
        datasets = ["bm", "gp", "sc18", "sc19", "sc20", "wt"]

    for name in datasets:
        try:
            generate_heatmaps(name, table_type=table_type)
        except Exception as e:
            print(f"[警告] 处理 {name} 时出错: {e}")


if __name__ == "__main__":
    # 示例：生成单个数据集的图表
    generate_heatmaps("sc18")

    # 批量生成所有数据集
    # generate_all_datasets()
