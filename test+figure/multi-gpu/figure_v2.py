import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 13,
        "legend.fontsize": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

INPUT_FILE = "./ncu_out/res.csv"
OUTPUT_DIR = Path("./figure")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_SIZE = (4.5, 3.2)
TIME_COLOR = "#ffbd79"
MEM_COLOR = "#8dcec8"
KEEP_NGPU = [4, 8, 16, 32, 64]

df = pd.read_csv(INPUT_FILE)
df = df[df["status"].astype(str).str.upper().eq("OK")].copy()

for col in ["NGPU", "normal", "GPU_duration_ms", "L1GlobalLoadSectors"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["NGPU", "normal", "GPU_duration_ms", "L1GlobalLoadSectors"])
df = df[df["NGPU"].isin(KEEP_NGPU)]

time_pivot = df.pivot_table(
    index="NGPU", columns="normal", values="GPU_duration_ms", aggfunc="mean"
)
mem_pivot = df.pivot_table(
    index="NGPU", columns="normal", values="L1GlobalLoadSectors", aggfunc="mean"
)

common = sorted(set(time_pivot.index) & set(mem_pivot.index))
time_speedup = (time_pivot.loc[common][1] / time_pivot.loc[common][0]).values
mem_speedup = (mem_pivot.loc[common][1] / mem_pivot.loc[common][0]).values

fig, ax = plt.subplots(figsize=FIG_SIZE)
idx = np.arange(len(common))
bar_w = 0.34

# 柱子
ax.bar(idx - bar_w / 2, mem_speedup, width=bar_w, color=MEM_COLOR, edgecolor="black", linewidth=0.8)
ax.bar(
    idx + bar_w / 2, time_speedup, width=bar_w, color=TIME_COLOR, edgecolor="black", linewidth=0.8
)

# 白色斜线
ax.bar(idx - bar_w / 2, mem_speedup, width=bar_w, color="none", edgecolor="white", hatch="///")
ax.bar(idx + bar_w / 2, time_speedup, width=bar_w, color="none", edgecolor="white", hatch="///")


# x tick
def pow2(v):
    return rf"$2^{{{int(math.log2(v))}}}$"


ax.set_xticks(idx)
ax.set_xticklabels([pow2(v) for v in common])


def fmt(v, _):
    return f"{v:.1f}×" if v > 0 else ""


ax.yaxis.set_major_formatter(FuncFormatter(fmt))
ax.yaxis.tick_left()
ax.yaxis.set_label_position("left")
ax.spines["left"].set_visible(True)

ax.set_xlabel("Number of Devices")
ax.set_ylabel("Speedup Over Native")
ax.set_ylim(0, max(max(time_speedup), max(mem_speedup)) * 1.1)

legend_mem = mpatches.Patch(facecolor=MEM_COLOR, edgecolor="white", hatch="///", label="Memory")
legend_time = mpatches.Patch(facecolor=TIME_COLOR, edgecolor="white", hatch="///", label="Time")
ax.legend(handles=[legend_mem, legend_time], loc="upper left", frameon=False)

# -------- 固定边距（与左图一致）--------
fig.subplots_adjust(left=0.14, right=0.98, bottom=0.18, top=0.96)

out = OUTPUT_DIR / "large_scale_multi_gpu.pdf"
fig.savefig(out, dpi=300)
plt.close(fig)

print("Saved:", out)
