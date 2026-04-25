import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

# -----------------------------------------------------------
# 全局字体与绘图设置 (所有字体调大)
# -----------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 28,  # 增大标题
        "axes.labelsize": 26,  # 增大轴标签
        "xtick.labelsize": 26, # 增大刻度标签
        "ytick.labelsize": 26,
        "legend.fontsize": 28,
        "axes.linewidth": 1.5,
    }
)

def set_pow2_ticklabels(ax):
    """
    将坐标轴刻度转换为 2^k 格式
    """
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()
    
    try:
        # 尝试从标签文本解析数值 (防备 seaborn 自动格式化)
        xtick_vals = [int(float(label.get_text())) for label in xticklabels]
        ytick_vals = [int(float(label.get_text())) for label in yticklabels]
    except Exception:
        # 降级方案：直接使用位置值
        xtick_vals = [int(t) for t in xticks if t > 0]
        ytick_vals = [int(t) for t in yticks if t > 0]
    
    # 过滤无效值
    xtick_vals = [v for v in xtick_vals if v > 0]
    ytick_vals = [v for v in ytick_vals if v > 0]
    
    # 设置 2的幂次 格式
    ax.set_xticklabels([f"$2^{{{int(np.log2(v))}}}$" if v > 0 and (v & (v-1)) == 0 else str(v) for v in xtick_vals])
    ax.set_yticklabels([f"$2^{{{int(np.log2(v))}}}$" if v > 0 and (v & (v-1)) == 0 else str(v) for v in ytick_vals])

def plot_heatmap(df, value_col, fmt, cmap, norm, ax, title):
    """
    绘制单个 heatmap，不包含 colorbar
    """
    # 构造矩阵
    pivot = df.pivot(index="a_size", columns="b_size", values=value_col)
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot.sort_index(axis=1, ascending=True)
    pivot = pivot.iloc[::-1]

    # 绘图
    hm = sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        norm=norm,
        cbar=False,           # 关闭独立 colorbar
        annot_kws={"size": 30}, # 稍微增大数字注释
        ax=ax
    )
    
    # 手动调整标签字体
    ax.tick_params(axis="x", labelsize=28, rotation=0)
    ax.tick_params(axis="y", labelsize=28, rotation=0)
    ax.set_xlabel("Set Size", fontsize=32)
    ax.set_ylabel("Set Size", fontsize=32)
    ax.set_title(title, fontsize=36, pad=15)
    
    set_pow2_ticklabels(ax)

if __name__ == "__main__":
    input_file = "new_result.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        exit(1)
        
    df = pd.read_csv(input_file, sep="\t")
    
    # 1. 计算全局 Colorbar 的范围 (使得两张图颜色刻度一致)
    max_val_time = df["TimeSpeedup"].max()
    max_val_mem = df["MemSpeedup"].max()
    global_max = max(max_val_time, max_val_mem)
    
    # 2. 生成更细密的分档 (n_bins 增加到 25)
    # 向上取整作为上限，保证最大值被覆盖
    upper_bound = np.ceil(global_max)
    if upper_bound < global_max: 
        upper_bound += 1.0
        
    n_bins = 30  # 增加分档
    boundaries = np.linspace(1.0, upper_bound, n_bins)
    
    # 3. 创建共享的 Colormap 和 Normalization
    cmap_name = "viridis_r"
    cmap = mcolors.ListedColormap(sns.color_palette(cmap_name, len(boundaries) - 1))
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)

    # 4. 创建画布 (增大尺寸以适应大字体)
    fig, axes = plt.subplots(1, 2, figsize=(24, 10)) 
    
    # 5. 分别绘制 Heatmap
    plot_heatmap(df, "TimeSpeedup", ".2f", cmap, norm, axes[0], "Time Speedup")
    plot_heatmap(df, "MemSpeedup", ".2f", cmap, norm, axes[1], "Memory Transaction Speedup")

    # 6. 调整布局为 Colorbar 腾出空间
    # left, bottom, right, top
    plt.subplots_adjust(bottom=0.15, top=0.9, left=0.08, right=0.88, wspace=0.20)
    
    # 7. 添加公用 Colorbar
    # 位置: [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.75])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    
    # Colorbar 样式设置
    cbar.set_label("Optimization Effect (Speedup)", fontsize=26)
    cbar.ax.tick_params(labelsize=26)
    # 使用 %.1f 避免刻度过于拥挤
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # 8. 保存
    output_path = "../../figure/set_size_speedup_heatmaps.pdf"
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    print(f"Saving to {output_path}")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
