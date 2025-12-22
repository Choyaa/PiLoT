import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import rcParams

# === 1. 字体设置 (已修改) ===
# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman'] 
# 移除全局坐标轴标签加粗的设置
# rcParams['axes.labelweight'] = 'bold' # 此行已被注释或删除

# === 数据表格 (未修改) ===
data = {
    "Method": ["Render2ORB", "FPVLoc", "Render2Loc", "Render2RAFT", "PixLoc"],
    "Sunny (1m/3m/5m Recall %)": ["6.75 / 27.48 / 52.50", "92.82 / 99.99 / 100.00", "69.90 / 96.86 / 99.52", "24.08 / 54.16 / 69.77", "46.64 / 89.63 / 96.29"],
    "Sunset (1m/3m/5m Recall %)": ["9.42 / 32.98 / 57.01", "92.43 / 99.90 / 99.93", "69.47 / 97.00 / 99.03", "21.06 / 49.90 / 65.29", "46.53 / 92.18 / 97.17"],
    "Night (1m/3m/5m Recall %)": ["7.70 / 31.59 / 50.84", "83.10 / 99.94 / 100.00", "55.89 / 91.97 / 96.21", "15.94 / 44.21 / 60.82", "28.37 / 71.02 / 87.75"],
    "Cloudy (1m/3m/5m Recall %)": ["3.72 / 23.98 / 55.58", "88.48 / 100.00 / 100.00", "59.22 / 96.13 / 99.04", "14.31 / 41.28 / 57.69", "44.69 / 95.37 / 99.91"],
    "Rainy (1m/3m/5m Recall %)": ["11.12 / 32.75 / 64.19", "94.28 / 99.98 / 100.00", "77.49 / 96.57 / 98.59", "25.61 / 56.04 / 70.13", "40.21 / 73.33 / 82.99"],
    "Foggy (1m/3m/5m Recall %)": ["8.98 / 30.56 / 56.10", "92.42 / 99.98 / 100.00", "66.66 / 95.74 / 98.27", "19.54 / 47.80 / 62.64", "40.84 / 85.30 / 93.34"]
}
df = pd.DataFrame(data)

# === 其余设置 (未修改) ===
methods_name = {
    "FPVLoc": "PiLoT", "Render2Loc": "Render2Loc", "PixLoc": "PixLoc",
    "Render2RAFT": "Render2RAFT", "Render2ORB": "Render2ORB",
}
methods_color = {
    "PiLoT":    "#01665e",
    "Render2Loc":  "#35978f",
    "PixLoc":      "#80cdc1",
    "Render2RAFT": "#c7eae5",
    "Render2ORB":  "#f6fff9"
}
weather_labels = ['Sunny',  'Sunset', 'Night', 'Foggy','Rainy', 'Cloudy']
angles = np.linspace(0, 2 * np.pi, len(weather_labels), endpoint=False).tolist()
angles += angles[:1]
output_dir = "/mnt/sda/MapScape/query/estimation/result_images/outputs" # 修改为当前目录下的outputs文件夹，方便测试
os.makedirs(output_dir, exist_ok=True)
recall_indices = { "Recall@1m": 0, "Recall@3m": 1, "Recall@5m": 2 }

# === 2. 绘图循环 (已修改字体大小和样式) ===
for recall_name, recall_idx in recall_indices.items():
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True)) # 稍微增大了画布
    ax.set_ylim(0, 105)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    method_lines = []
    for i, row in df.iterrows():
        method_key = row["Method"]
        method_label = methods_name.get(method_key, method_key)
        color = methods_color[method_label]

        values = []
        for col in df.columns[1:]:
            parts = [float(x.strip()) for x in row[col].split('/')]
            values.append(parts[recall_idx])
        values += values[:1]
        mean_recall = np.mean(values[:-1])
        method_lines.append((mean_recall, method_label, values, color))
        
    method_lines.sort()

    for z, (mean_recall, method_label, values, color) in enumerate(method_lines[::-1]):
        ax.fill(angles, values, color=color, alpha=0.6, linewidth=0, zorder=z * 2)
        ax.plot(angles, values, label=method_label, color=color, linewidth=2.5, zorder=z * 2 + 1) # 线条加粗一点更好看

    ax.grid(True, linestyle="--", linewidth=1, color="#AAAAAA", zorder=1000)
    ax.set_xticks([])

    # 自定义天气标签 (字体放大, 去除加粗)
    for angle, label in zip(angles[:-1], weather_labels):
        ax.text(angle, 120, label, ha='center', va='center',
                fontsize=16, zorder=2000) # fontweight='bold' 已移除

    ax.set_yticklabels([])

    # 手动绘制极径标签 (字体放大, 去除加粗)
    for r in [20, 40, 60, 80, 100]:
        ax.text(np.pi / 2, r, f"{r}%", ha='center', va='bottom',
                fontsize=12, zorder=2000) # fontweight='bold' 已移除
    
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 105], color="#AAAAAA", linewidth=1, linestyle='--', zorder=1000)

    # 图标题 (字体放大, 去除加粗)
    # ax.set_title(recall_name, fontsize=22, pad=25) # fontweight='bold' 已移除

    # 图例 (字体放大)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=15, frameon=True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"radar_{recall_name}_times_final.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

print(f"✅ 所有雷达图已保存至：{out_path}")

