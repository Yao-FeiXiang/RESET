from __future__ import annotations

import os
import math
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# ============================================================
# =========================== 配置 ============================
# ============================================================

INPUT_FILE = "./output/res.csv"  # 兼容旧文件名
OUTPUT_DIR = "./output/figure"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = {
    # -------- 输出 --------
    "SAVE_DPI": 300,
    "SAVE_FMT": "pdf",
    # -------- 画布 --------
    "FIGSIZE_SEC": (7, 4.2),
    "FIGSIZE_SPEEDUP": (7, 4.2),
    "FIGSIZE_INTRO": (3.15, 3.15),
    "TITLE_SIZE": 16,
    "LABEL_SIZE": 22,
    "TICK_SIZE": 24,
    "GRID_ALPHA": 0.35,
    "BAR_EDGE_COLOR": "white",
    "BAR_EDGE_LW": 0.8,
    "X_ROT": 0,
    "SECPERREQ_PAD_TOP": 1.10,
    # -------- 柱/图例顺序 --------
    "METHOD_ORDER": ["RESET", "Native", "cuCollections"],
    "LEGEND_NAME": {"RESET": "RESET", "Native": "Native", "cuCollections": "cuCollections"},
    "COLOR": {"RESET": "#ffbd79", "Native": "#8dcec8", "cuCollections": "#f97e6e"},
    "COLOR_TIME": "#6badd6",
    "COLOR_MEM": "#c6e9bf",
    # -------- 文案 --------
    "YLABEL_SEC": "Sectors / Request",
    "YLABEL_SPEEDUP": "Speedup (×)",
    "XLABEL": "",
    # -------- 图例样式--------
    "LEGEND": {
        "enable": True,
        "loc": "upper right",
        "bbox_to_anchor": (1.2, 1.18),
        "ncol": 3,
        "frameon": True,
        "fontsize": 14,
    },
    # -------- legend 单独输出 --------
    "LEGEND_ONLY": {
        "enable": True,
        "filename": "legend_sec_per_req",
        "figsize": (6.8, 1.0),
        "ncol": 3,
    },
    # -------- hatch --------
    "HATCH_TIME": "///",
    "HATCH_MEM": "\\\\\\",
}

# ============================================================
# ============================ util ============================
# ============================================================


def read_data(path: str) -> pd.DataFrame:
    """读取 res.csv 数据"""
    df = pd.read_csv(path)
    df["method_name"] = df["method_name"].astype(str).str.strip()
    df["dataset"] = df["dataset"].astype(str)
    df["app"] = df["app"].astype(str)
    return df


def ensure_three_methods(df: pd.DataFrame) -> pd.DataFrame:
    """确保每个数据集有三种方法"""
    need = set(CONFIG["METHOD_ORDER"])
    bad = []
    for (app, ds), g in df.groupby(["app", "dataset"]):
        have = set(g["method_name"].tolist())
        if have != need:
            bad.append((f"{app}/{ds}", sorted(have)))
    if bad:
        msg = "\n".join([f"- {key}, have={have}" for key, have in bad])
        raise ValueError("以下数据集缺少三种 method：\n" + msg)
    return df


def _wide_by_method(df_app: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """pivot 成 dataset x method_name 的宽表，并按 METHOD_ORDER 排序列。"""
    wide = (
        df_app.pivot_table(
            index="dataset", columns="method_name", values=value_col, aggfunc="first"
        )
        .reindex(columns=CONFIG["METHOD_ORDER"])
        .sort_index()
    )
    return wide


def _geomean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(x))))


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", alpha=CONFIG["GRID_ALPHA"])
    ax.tick_params(axis="y", labelsize=CONFIG["TICK_SIZE"])
    ax.tick_params(axis="x", labelsize=CONFIG["TICK_SIZE"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(True)


def save_legend_only(handles, labels, out_path: str) -> None:
    leg_cfg = CONFIG.get("LEGEND", {})
    only_cfg = CONFIG.get("LEGEND_ONLY", {})

    fig = plt.figure(figsize=tuple(only_cfg.get("figsize", (2.8, 1.0))))
    fig.patch.set_alpha(0.0)

    ncol = int(only_cfg.get("ncol", leg_cfg.get("ncol", 1)))

    legend = fig.legend(
        handles,
        labels,
        loc="center",
        frameon=bool(leg_cfg.get("frameon", True)),
        fontsize=leg_cfg.get("fontsize", 12),
        ncol=ncol,
    )

    if bool(leg_cfg.get("frameon", True)):
        legend.get_frame().set_linewidth(0.8)

    fig.savefig(out_path, dpi=CONFIG["SAVE_DPI"], bbox_inches="tight", transparent=True)
    plt.close(fig)


# ============================================================
# ======================= 绘图函数 ============================
# ============================================================


def draw_bar_sec_per_req(df_app: pd.DataFrame, out_path: str, return_legend_items: bool = False):
    """绘制 Sectors / Request 柱状图"""
    order = CONFIG["METHOD_ORDER"]
    datasets = sorted(df_app["dataset"].unique().tolist())

    # ---------------- 数据矩阵 ----------------
    mat = np.zeros((len(datasets), len(order)), dtype=float)
    for i, ds in enumerate(datasets):
        g = df_app[df_app["dataset"] == ds].set_index("method_name")
        for j, kt in enumerate(order):
            mat[i, j] = float(g.loc[kt, "sector_per_request"])

    x = np.arange(len(datasets))
    width = 0.26

    plt.rcParams["hatch.linewidth"] = 1.0

    # ---------------- 绘图 ----------------
    fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE_SEC"])

    for j, method in enumerate(order):
        ax.bar(
            x + (j - 1) * width,
            mat[:, j],
            width=width,
            label=CONFIG["LEGEND_NAME"][method],
            color=CONFIG["COLOR"][method],
            edgecolor=CONFIG["BAR_EDGE_COLOR"],
            linewidth=CONFIG["BAR_EDGE_LW"],
            hatch="///",
        )

    # ---------------- 坐标轴 ----------------
    ax.set_ylabel(CONFIG["YLABEL_SEC"], fontsize=CONFIG["LABEL_SIZE"])
    ax.set_xlabel(CONFIG["XLABEL"], fontsize=CONFIG["LABEL_SIZE"])

    ax.set_xticks(x)
    rot = float(CONFIG["X_ROT"])
    ha = "center" if abs(rot) < 1e-9 else "right"
    ax.set_xticklabels(datasets, rotation=rot, ha=ha)
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ax.tick_params(axis="x", pad=4)

    # ---------------- Y 轴 ----------------
    y_max = float(mat.max())
    ax.set_ylim(0.0, y_max * float(CONFIG["SECPERREQ_PAD_TOP"]))

    # ---------------- 图例 ----------------
    handles, labels = ax.get_legend_handles_labels()

    style_axes(ax)
    fig.tight_layout(pad=0.6)

    fig.savefig(out_path, dpi=CONFIG["SAVE_DPI"], bbox_inches="tight")
    plt.close(fig)

    if return_legend_items:
        return handles, labels
    return None


def draw_intro_average_speedup(df: pd.DataFrame, out_path: str):
    """绘制介绍图：平均加速比（Time vs Memory）"""
    apps = sorted(df["app"].unique().tolist())

    time_vals = []
    mem_vals = []

    for app in apps:
        df_app = df[df["app"] == app]
        wide_t = _wide_by_method(df_app, "kernel_time_ms")
        wide_m = _wide_by_method(df_app, "L1GlobalLoadSectors")

        sp_t = _geomean(wide_t["Native"] / wide_t["RESET"])
        sp_m = _geomean(wide_m["Native"] / wide_m["RESET"])

        time_vals.append(sp_t)
        mem_vals.append(sp_m)

    max_ref = max(max(time_vals), max(mem_vals))
    ymax = max(2.0, (int(max_ref * 10) + 2) / 10)

    fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE_INTRO"], dpi=300)

    x = list(range(len(apps)))
    width = 0.30

    bars_time = ax.bar(
        [i - width / 2 for i in x],
        time_vals,
        width=width,
        label="Time",
        color=CONFIG["COLOR_TIME"],
        edgecolor="white",
        linewidth=0.4,
        hatch=CONFIG["HATCH_TIME"],
    )

    bars_mem = ax.bar(
        [i + width / 2 for i in x],
        mem_vals,
        width=width,
        label="Memory",
        color=CONFIG["COLOR_MEM"],
        edgecolor="white",
        linewidth=0.4,
        hatch=CONFIG["HATCH_MEM"],
    )

    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in apps], fontsize=12)
    ax.set_ylabel("Speedup (×)", fontsize=12)
    ax.set_ylim(1.0, ymax)
    ax.tick_params(axis="y", labelsize=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    ax.legend(
        loc="upper right",
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="0.9",
        framealpha=1.0,
        handlelength=1.2,
        handleheight=0.8,
        borderaxespad=0.3,
    )

    def add_labels(bars):
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 0.02,
                f"{h:.2f}×",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    add_labels(bars_time)
    add_labels(bars_mem)

    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# ============================ 主流程 ============================
# ============================================================


def main() -> None:
    print(f"Reading data from: {INPUT_FILE}")
    df = read_data(INPUT_FILE)
    df = ensure_three_methods(df)
    print(f"Loaded {len(df)} records")

    apps = sorted(df["app"].unique().tolist())
    print(f"Found apps: {apps}")

    # ---------- 1. Sectors / Request 图 (按应用) ----------
    legend_saved = False
    legend_only_cfg = CONFIG.get("LEGEND_ONLY", {})
    want_legend_only = bool(legend_only_cfg.get("enable", False)) and bool(
        CONFIG.get("LEGEND", {}).get("enable", True)
    )

    for app in apps:
        df_app = df[df["app"] == app]
        fig_path = os.path.join(OUTPUT_DIR, f"{app}_sec_per_req.{CONFIG['SAVE_FMT']}")

        if want_legend_only and (not legend_saved):
            ret = draw_bar_sec_per_req(df_app=df_app, out_path=fig_path, return_legend_items=True)
            handles, labels = ret

            legend_path = os.path.join(
                OUTPUT_DIR, f"{legend_only_cfg.get('filename', 'legend')}.{CONFIG['SAVE_FMT']}"
            )
            save_legend_only(handles, labels, legend_path)
            print(f"  legend: {legend_path}")
            legend_saved = True
        else:
            draw_bar_sec_per_req(df_app=df_app, out_path=fig_path, return_legend_items=False)

        print(f"[OK] app={app}, sec_per_req plot: {fig_path}")

    # ---------- 2. 介绍图 (平均加速比) ----------
    intro_path = os.path.join(OUTPUT_DIR, f"average_speedup_intro.{CONFIG['SAVE_FMT']}")
    draw_intro_average_speedup(df, intro_path)
    print(f"[OK] intro speedup figure: {intro_path}")

    # ---------- 3. 打印加速比数据 ----------
    apps = sorted(df["app"].unique().tolist())
    print("\nSpeedup summary (Native vs RESET):")
    for app in apps:
        df_app = df[df["app"] == app]
        wide_t = _wide_by_method(df_app, "kernel_time_ms")
        wide_m = _wide_by_method(df_app, "L1GlobalLoadSectors")
        sp_t = _geomean(wide_t["Native"] / wide_t["RESET"])
        sp_m = _geomean(wide_m["Native"] / wide_m["RESET"])
        print(f"  {app.upper()}: Time={sp_t:.2f}x, Memory={sp_m:.2f}x")

    print("\nAll figures generated successfully!")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
