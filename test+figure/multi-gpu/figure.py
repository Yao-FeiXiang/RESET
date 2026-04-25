import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

OUTPUT_DIR = "./figure"
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = {"TC": "tc_res.txt", "SSS": "sss_res.txt", "IR": "ir_res.txt"}
markers = {"TC": "o", "SSS": "s", "IR": "^"}
colors = {"TC": "#ffbd79", "SSS": "#8dcec8", "IR": "#f97e6e"}

legend_names = {
    "TC": "Triangle-counting: gp",
    "SSS": "Set Similarity Search: gp",
    "IR": "Information Retrieval: ms",
}

FIGSIZE = (4.5, 3.2)
fig, ax = plt.subplots(figsize=FIGSIZE)

devices = None

for label, filename in files.items():
    df = pd.read_csv(filename, sep="\t")
    df = df.sort_values("total_device")
    x = df["total_device"].values
    y = df["speedup"].values

    if devices is None:
        devices = x

    ax.plot(
        x,
        y,
        color=colors[label],
        marker=markers[label],
        markersize=4.5,
        markeredgewidth=0.8,
        markerfacecolor=colors[label],
        label=legend_names[label],
    )

# -------- 轴设置 --------
x_log2 = [int(np.log2(v)) for v in devices]
ax.set_xscale("log", base=2)
ax.set_xticks(devices)
ax.set_xticklabels([f"$2^{{{i}}}$" for i in x_log2])

ax.set_yscale("log", base=2)
yticks = [2**i for i in range(6)]
ax.set_yticks(yticks)
ax.set_yticklabels([f"$2^{{{i}}}$" if i != 5 else "" for i in range(6)])
ax.set_ylim(0.7, 2**5)

ax.set_xlabel("Number of Devices")
ax.set_ylabel("Speedup over Single-GPU")

ax.legend(loc="upper left", frameon=False)

# -------- 固定边距（核心）--------
fig.subplots_adjust(left=0.14, right=0.98, bottom=0.18, top=0.96)

save_path = os.path.join(OUTPUT_DIR, "multi_gpu_speedup.pdf")
fig.savefig(save_path, dpi=300)
plt.close(fig)

print("Saved:", save_path)
