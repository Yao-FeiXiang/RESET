import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.lines as mlines
import math

file_path = "../../graph_datasets/tc_figure.txt"

rows = []

try:
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line or line.startswith('Dataset') or line.startswith('---'):
            continue
            
        parts = line.split()
        if len(parts) >= 5:
            dataset = parts[0]
            try:
                # Time Speedup
                time_speedup = float(parts[1])
                
                # Transaction Calculation
                sectors_str = parts[3]
                sectors_base = float(sectors_str.split('/')[0])
                sectors_opt = float(sectors_str.split('/')[1])
                # 计算 Transaction Speedup = Base / Opt
                trans_speedup = sectors_base / sectors_opt if sectors_opt > 0 else 0
                
                # Sec/Req Data
                seq_req_str = parts[4]
                seq_req_base = float(seq_req_str.split('/')[0])
                seq_req_opt = float(seq_req_str.split('/')[1])

                # 保留两位小数并向上取整
                time_speedup_ceil = math.ceil(time_speedup * 100) / 100
                trans_speedup_ceil = math.ceil(trans_speedup * 100) / 100

                rows.append({
                    'Dataset': dataset, 
                    'Time Speedup': time_speedup_ceil,
                    'Trans Speedup': trans_speedup_ceil,
                    'Sec/Req (Base)': seq_req_base,
                    'Sec/Req (Opt)': seq_req_opt
                })
            except ValueError:
                continue
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit(1)

if not rows:
    print("No data found.")
    exit(1)

df = pd.DataFrame(rows)

sns.set_theme(style="white", context="notebook")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12

COLOR_BASE = "#3fab1a" # Time Speedup / Base
COLOR_OPT = "#0D6093"  # Trans Speedup / Opt

df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), gridspec_kw={'wspace': 0.3})

# === Plot 1: Speedup Comparisons (Lollipop) ===
ax = axes[0]
y = np.arange(len(df_shuffled))
offset = 0.15 

text_y_pos = len(df_shuffled) - 0.5

ax.vlines(x=1.0, ymin=-0.5, ymax=text_y_pos, color='gray', linestyle='--', linewidth=1.5, zorder=0)
ax.text(1.0, text_y_pos, 'w/o RESET', 
        color='gray', fontsize=11, ha='center', va='bottom', rotation=0, fontweight='bold', zorder=1)

ax.hlines(y - offset, 1.0, df_shuffled['Time Speedup'], color=COLOR_BASE, alpha=0.8, linewidth=3)
ax.scatter(df_shuffled['Time Speedup'], y - offset, color=COLOR_BASE, s=80, label='Time Speedup', zorder=3)

ax.hlines(y + offset, 1.0, df_shuffled['Trans Speedup'], color=COLOR_OPT, alpha=0.8, linewidth=3)
ax.scatter(df_shuffled['Trans Speedup'], y + offset, color=COLOR_OPT, s=80, marker='D', label='Trans. Speedup', zorder=3)

# 标注数值（两位小数，向上取整）
for i, (idx, row) in enumerate(df_shuffled.iterrows()):
    ax.text(row['Time Speedup'] + 0.1, i - offset, f"{row['Time Speedup']:.2f}x", 
            va='center', ha='left', fontsize=13, color=COLOR_BASE, fontweight='bold')
    ax.text(row['Trans Speedup'] + 0.1, i + offset, f"{row['Trans Speedup']:.2f}x", 
            va='center', ha='left', fontsize=13, color=COLOR_OPT, fontweight='bold')

ax.set_yticks(y)
ax.set_yticklabels(df_shuffled['Dataset'], fontsize=13)
ax.set_xlabel('Speedup', fontsize=14, color='#34495e')
ax.set_title('TC - Speedup', fontsize=16, pad=15, color='#2c3e50', weight='bold')

max_speedup = max(df_shuffled['Time Speedup'].max(), df_shuffled['Trans Speedup'].max())
ax.set_xlim(0.8, max_speedup * 1.1) 
ax.set_ylim(-0.5, len(df_shuffled))

ax.tick_params(axis='x', labelsize=12) 
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.07), frameon=True, fontsize=10, borderpad=0.4, handletextpad=0.2)
sns.despine(left=False, bottom=True, ax=ax)

# === Plot 2: Sec/Req Comparison (Bar Plot) ===
ax = axes[1]
df_melted2 = df_shuffled.melt(id_vars=['Dataset'], 
                              value_vars=['Sec/Req (Base)', 'Sec/Req (Opt)'], 
                              var_name='Configuration', value_name='Value')
df_melted2['Configuration'] = df_melted2['Configuration'].apply(
    lambda x: 'w/o RESET' if 'Base' in x else 'w/ RESET'
)
sns.barplot(x='Dataset', y='Value', hue='Configuration', 
            data=df_melted2, palette=[COLOR_BASE, COLOR_OPT], edgecolor="white", ax=ax)

ax.set_title('TC - Coalescing Comparison', fontsize=16, pad=15, color='#2c3e50', weight='bold')
ax.set_ylabel('Sectors / Request', fontsize=14, color='#34495e')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlabel('')
ax.set_xticklabels(df_shuffled['Dataset'], rotation=30, ha='right', fontsize=12)

ax.legend(title='', frameon=True, loc='upper right', bbox_to_anchor=(1.1, 1.07), fontsize=10)
sns.despine(left=True, ax=ax)
ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout(pad=0.5)
plt.savefig('../../../figure/tc_summary_all.pdf', dpi=300, bbox_inches='tight')
print("Generated tc_summary_all.pdf")
plt.close()