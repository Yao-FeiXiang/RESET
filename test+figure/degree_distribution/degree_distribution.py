import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import PercentFormatter

def read_graph_degrees(folder):
    num_nodes = np.fromfile(os.path.join(folder, "num_nodes.bin"), dtype=np.int32)[0]
    offsets = np.fromfile(os.path.join(folder, "csr_offsets.bin"), dtype=np.int32)
    degrees = offsets[1:] - offsets[:-1]
    return degrees

def read_ir_degrees(folder):
    num_terms = np.fromfile(os.path.join(folder, "inverted_index_num.bin"), dtype=np.int32)[0]
    offsets = np.fromfile(os.path.join(folder, "inverted_index_offsets.bin"), dtype=np.int32)
    degrees = offsets[1:] - offsets[:-1]
    return degrees

graph_datasets = [
    ("Wiki-Talk", "../../graph_datasets/wiki-Talk"),
    ("Gplus", "../../graph_datasets/gplus"),
    ("Graph500-Scale20", "../../graph_datasets/sc20"),
]

ir_datasets = [
    ("Fever", "../../ir_datasets/fever"),
    ("Msmarco", "../../ir_datasets/msmarco"),
    # ("hotpotqa", "../../ir_datasets/hotpotqa"),
    # ("cqadupstack-english", "../../ir_datasets/cqadupstack-english"),
    ("Lotte","../../ir_datasets/lotte"),
]

fig, axes = plt.subplots(2, 3, figsize=(10.5, 7))

bins = np.logspace(0, 5, num=15)  # 10^0 ~ 10^5, 50 bins
major_ticks = [1, 10, 100, 1000, 10000, 100000]

class MyPercentFormatter(PercentFormatter):
    def __call__(self, x, pos=None):
        if abs(x - int(x)) < 1e-6:
            return f"{int(x)}%"
        else:
            return super().__call__(x, pos)

for idx, (name, folder) in enumerate(graph_datasets):
    degrees = read_graph_degrees(folder)
    degrees = degrees[degrees > 0]
    hist, bin_edges = np.histogram(degrees, bins=bins)
    percentage = hist / hist.sum() * 100
    ax = axes[0, idx]
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    ax.plot(bin_centers, percentage, linewidth=1.5)  # 曲线粗度减小
    ax.set_xscale('log')
    ax.set_xlim(1, 1e5)
    ax.set_ylim(0, None)
    ax.set_xlabel("Set Size", fontsize=16)
    ax.set_ylabel("Percentage", fontsize=16)
    ax.set_title(name, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)  # 坐标数字加大
    ax.set_xticks(major_ticks)
    ax.get_xaxis().set_minor_locator(plt.NullLocator())
    ax.yaxis.set_major_formatter(MyPercentFormatter())

for idx, (name, folder) in enumerate(ir_datasets):
    degrees = read_ir_degrees(folder)
    degrees = degrees[degrees > 0]
    hist, bin_edges = np.histogram(degrees, bins=bins)
    percentage = hist / hist.sum() * 100
    ax = axes[1, idx]
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    ax.plot(bin_centers, percentage, linewidth=1.5)  # 曲线粗度减小
    ax.set_xscale('log')
    ax.set_xlim(1, 1e5)
    ax.set_ylim(0, None)
    ax.set_xlabel("Set Size", fontsize=16)
    ax.set_ylabel("Percentage", fontsize=16)
    ax.set_title(name, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)  # 坐标数字加大
    ax.set_xticks(major_ticks)
    ax.get_xaxis().set_minor_locator(plt.NullLocator())
    ax.yaxis.set_major_formatter(MyPercentFormatter())

plt.tight_layout()
plt.savefig("../../../figure/degree_distribution.pdf")
plt.close()