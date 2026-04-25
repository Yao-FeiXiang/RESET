from __future__ import annotations

import os
import math
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置论文风格字体 (Times New Roman)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
})


# ============================================================
# =========================== 配置 ============================
# ============================================================

INPUT_FILE = "res.txt"
OUTPUT_DIR = "../../../figure"

CONFIG = {
    # -------- 输出 --------
    "SAVE_DPI": 300,
    "SAVE_FMT": "pdf",
    # -------- 画布 --------
    "FIGSIZE": (8, 4.8),  # 稍微增大画布以容纳大字体
    "TITLE_SIZE": 24,
    "LABEL_SIZE": 22,
    "TICK_SIZE": 18,
    "GRID_ALPHA": 0.35,
    "BAR_EDGE_COLOR": "white",
    "BAR_EDGE_LW": 0.8,
    "X_ROT": 0,  # Q0-Q5 比较短，不需要旋转
    "SECPERREQ_PAD_TOP": 1.20,
    # -------- 柱/图例顺序 --------
    "KERNEL_ORDER": ["hierarchical", "normal"],
    "LEGEND_NAME": {"hierarchical": "RESET", "normal": "Native"},
    "COLOR": {
        "hierarchical": "#1F77B4",  # 深蓝
        "normal": "#FF7F0E",  # 橙
    },
    # -------- 文案 --------
    "TITLE": {"sec_per_req": "SMOG - Coalescing Comparison"},
    "YLABEL": {"sec_per_req": "Sectors / Request"},
    "XLABEL": "Pattern",
    "LEGEND": {
        "enable": True,
        "loc": "upper right",
        # "bbox_to_anchor": (1.0, 1.0),
        "ncol": 1,
        "frameon": True,
        "fontsize": 16,
    },
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# ============================ util ============================
# ============================================================


def read_data(path: str) -> pd.DataFrame:
    """
    读取 res.txt (宽表格式)，转换为 long format 以适配绘图逻辑。
    res.txt columns:
      Dataset, Pattern, Normal Time(ms), Hier Time(ms), Speedup, 
      Normal Req, Hier Req, Normal Sec, Hier Sec, 
      Sec/Req(N), Sec/Req(H), SecOpt
    """
    # 假设它是 tab 分隔，参考 figure_v3.py
    # 如果是空格分隔且列名带空格（如 Normal Time(ms)），需要小心
    # 这里默认使用 \t，因为 res.txt 看起来是 TSV
    df = pd.read_csv(path, sep=r"\t", comment="#", engine="python")
    
    # 我们需要构建一个类似于 figure_v3 的 long table
    # columns: kernel_type, pattern, sec_per_req
    
    records = []
    for _, row in df.iterrows():
        dataset = str(row["Dataset"])
        pattern = str(row["Pattern"])
        
        # Native (Normal)
        records.append({
            "dataset": dataset,
            "pattern": pattern,
            "kernel_type": "normal",
            "sec_per_req": row["Sec/Req(N)"]
        })
        
        # RESET (Hierarchical)
        records.append({
            "dataset": dataset,
            "pattern": pattern,
            "kernel_type": "hierarchical",
            "sec_per_req": row["Sec/Req(H)"]
        })
        
    res_df = pd.DataFrame(records)
    return res_df


# ============================================================
# ============================ 绘图 ============================
# ============================================================


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", alpha=CONFIG["GRID_ALPHA"])
    ax.tick_params(axis="y", labelsize=CONFIG["TICK_SIZE"])
    ax.tick_params(axis="x", labelsize=CONFIG["TICK_SIZE"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def draw_bar_sec_per_req(df: pd.DataFrame, out_path: str) -> None:
    order = CONFIG["KERNEL_ORDER"]
    # 按照 Q0, Q1, Q2... 排序
    patterns = sorted(df["pattern"].unique().tolist())

    # pivot 方便取值: index=pattern, columns=kernel_type
    wide = df.pivot(index="pattern", columns="kernel_type", values="sec_per_req")
    
    # 准备数据矩阵 (N_patterns x N_kernels)
    mat = np.zeros((len(patterns), len(order)), dtype=float)
    for i, pat in enumerate(patterns):
        for j, kt in enumerate(order):
            if kt in wide.columns:
                mat[i, j] = wide.loc[pat, kt]
            else:
                mat[i, j] = 0.0

    x = np.arange(len(patterns))
    # 两个柱子，稍微宽一点
    total_width = 0.7 
    bar_width = total_width / len(order)

    fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE"])
    
    # 绘制
    for j, kt in enumerate(order):
        # 计算偏移量，让它们居中
        # j=0 -> offset = -0.5 * bar_width
        # j=1 -> offset = +0.5 * bar_width
        # general formula: x + (j - (len-1)/2) * bar_width
        offset = (j - (len(order) - 1) / 2) * bar_width
        
        ax.bar(
            x + offset,
            mat[:, j],
            width=bar_width,
            label=CONFIG["LEGEND_NAME"][kt],
            color=CONFIG["COLOR"][kt],
            edgecolor=CONFIG["BAR_EDGE_COLOR"],
            linewidth=CONFIG["BAR_EDGE_LW"],
            zorder=3
        )

    ax.set_title(
        CONFIG["TITLE"]["sec_per_req"],
        fontsize=CONFIG["TITLE_SIZE"],
        pad=12,
        weight="bold",
    )
    ax.set_ylabel(CONFIG["YLABEL"]["sec_per_req"], fontsize=CONFIG["LABEL_SIZE"])
    ax.set_xlabel(CONFIG["XLABEL"], fontsize=CONFIG["LABEL_SIZE"])

    ax.set_xticks(x)
    ax.set_xticklabels(patterns, rotation=CONFIG["X_ROT"])

    y_max = float(mat.max())
    ax.set_ylim(0.0, y_max * float(CONFIG["SECPERREQ_PAD_TOP"]))

    leg_cfg = CONFIG["LEGEND"]
    if leg_cfg.get("enable", True):
        kwargs = dict(
            loc=leg_cfg.get("loc", "upper right"),
            frameon=leg_cfg.get("frameon", True),
            fontsize=leg_cfg.get("fontsize", CONFIG["TICK_SIZE"]),
            ncol=leg_cfg.get("ncol", 1),
        )
        bba = leg_cfg.get("bbox_to_anchor", None)
        if bba is not None:
            kwargs["bbox_to_anchor"] = tuple(bba)
        ax.legend(**kwargs)

    style_axes(ax)
    fig.tight_layout(pad=0.6)
    print(f"Saving figure to {out_path}")
    fig.savefig(out_path, dpi=CONFIG["SAVE_DPI"], bbox_inches="tight")
    plt.close(fig)


# ============================================================
# ============================ 主流程 ============================
# ============================================================


def main() -> None:
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = read_data(INPUT_FILE)
    
    # 可能会有多个 Dataset，这里假设我们只关心其中一个，或者把它们都画出来
    # 如果 res.txt 混杂了 sc18, sc19，可以考虑 split
    datasets = df["dataset"].unique()
    
    for ds in datasets:
        # 筛选特定 dataset
        # 如果你想把所有 dataset 的 Qx 都画在一张图，就不用 filter
        # 但通常 SMOG 不同 dataset 的 scale 差异可能不大？或者分别画
        sub_df = df[df["dataset"] == ds].copy()
        
        filename = f"SMOG_{ds}_sec_per_req.{CONFIG['SAVE_FMT']}"
        out_path = os.path.join(OUTPUT_DIR, filename)
        
        draw_bar_sec_per_req(sub_df, out_path)
        print(f"[OK] dataset={ds}, plot -> {out_path}")

if __name__ == "__main__":
    main()
