import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from scipy.stats import gaussian_kde
import h5py

# 设置全局字体大小
plt.rcParams.update({
    'font.size': 12,  # 全局字体大小
    'axes.titlesize': 14,  # 标题字体大小
    'axes.labelsize': 12,  # 坐标轴标签字体大小
    'xtick.labelsize': 11,  # X轴刻度标签字体大小
    'ytick.labelsize': 11,  # Y轴刻度标签字体大小
    'legend.fontsize': 12  # 图例字体大小改为12（放大）
})

# 创建1列2行的子图，设置DPI为500
fig, axes = plt.subplots(2, 1, figsize=(8, 10), dpi=500)  # 稍微增加图形尺寸

# ============================
# 第一个数据集：Luding_dataset
# ============================
label_name = []
file_path = r"E:\data\label\label.h5"
with h5py.File(file_path, 'r') as f:
    label_name = list(f.keys())

widths1 = []
heights1 = []

for i, file in enumerate(label_name):
    with h5py.File(file_path, 'r') as hf:
        label_image = np.asarray(hf[file][:])

    labeled_array, num_features = ndimage.label(label_image)

    for i in range(1, num_features + 1):
        slip_y, slip_x = np.where(labeled_array == i)
        if len(slip_y) > 0 and len(slip_x) > 0:
            min_x, max_x = np.min(slip_x), np.max(slip_x)
            min_y, max_y = np.min(slip_y), np.max(slip_y)
            slip_height = max_y - min_y + 1
            slip_width = max_x - min_x + 1
            widths1.append(slip_width)
            heights1.append(slip_height)

widths1 = np.asarray(widths1)
heights1 = np.asarray(heights1)

# 绘制第一个子图
ax1 = axes[0]
if len(widths1) > 0 and len(heights1) > 0:
    xy1 = np.vstack([widths1, heights1])
    kde1 = gaussian_kde(xy1)
    density1 = kde1(xy1)
    idx1 = density1.argsort()
    x1, y1, z1 = widths1[idx1], heights1[idx1], density1[idx1]
    scatter1 = ax1.scatter(x1, y1, c=z1, s=15, cmap='Reds', edgecolor='red', alpha=0.1)

# 在(20,20)处绘制L形线（第一个子图）
ax1.plot([0, 20], [20, 20], color='blue', linestyle='--', linewidth=1.5, alpha=0.8, label='(20,20)')
ax1.plot([20, 20], [0, 20], color='blue', linestyle='--', linewidth=1.5, alpha=0.8)

# 在(50,50)处绘制L形线（第一个子图）
ax1.plot([0, 50], [50, 50], color='green', linestyle='--', linewidth=1.5, alpha=0.8, label='(50,50)')
ax1.plot([50, 50], [0, 50], color='green', linestyle='--', linewidth=1.5, alpha=0.8)

# 设置第一个子图的标题和标签（使用更大的字体）
ax1.set_xlabel('Width (pixels)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Height (pixels)', fontsize=14, fontweight='bold')
ax1.set_title('Luding Dataset', fontsize=16, fontweight='bold', pad=15)
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 128)

# 调整刻度标签大小
ax1.tick_params(axis='both', which='major', labelsize=12)

# 调整网格线
ax1.grid(True, alpha=0.3, linewidth=0.5)

# 添加图例（第一个子图） - 字体已经通过全局设置放大
ax1.legend(loc='upper right')

# ============================
# 第二个数据集：landslide4Scence
# ============================
folder_path = r"E:\landslide4scen\train\mask"
all_files = os.listdir(folder_path)
mask_files = [file for file in all_files if file.endswith(".h5")]

widths2 = []
heights2 = []

for mask in mask_files:
    full_path = os.path.join(folder_path, mask)
    with h5py.File(full_path, 'r') as hf:
        label_image = np.asarray(hf['mask'][:])

    labeled_array, num_features = ndimage.label(label_image)

    for i in range(1, num_features + 1):
        slip_y, slip_x = np.where(labeled_array == i)
        if len(slip_y) > 0 and len(slip_x) > 0:
            min_x, max_x = np.min(slip_x), np.max(slip_x)
            min_y, max_y = np.min(slip_y), np.max(slip_y)
            slip_height = max_y - min_y + 1
            slip_width = max_x - min_x + 1
            widths2.append(slip_width)
            heights2.append(slip_height)

widths2 = np.asarray(widths2)
heights2 = np.asarray(heights2)

# 绘制第二个子图
ax2 = axes[1]
if len(widths2) > 0 and len(heights2) > 0:
    xy2 = np.vstack([widths2, heights2])
    kde2 = gaussian_kde(xy2)
    density2 = kde2(xy2)
    idx2 = density2.argsort()
    x2, y2, z2 = widths2[idx2], heights2[idx2], density2[idx2]
    scatter2 = ax2.scatter(x2, y2, c=z2, s=15, cmap='Reds', edgecolor='red', alpha=0.1)

# 在(20,20)处绘制L形线（第二个子图）
ax2.plot([0, 20], [20, 20], color='blue', linestyle='--', linewidth=1.5, alpha=0.8, label='(20,20)')
ax2.plot([20, 20], [0, 20], color='blue', linestyle='--', linewidth=1.5, alpha=0.8)

# 在(50,50)处绘制L形线（第二个子图）
ax2.plot([0, 50], [50, 50], color='green', linestyle='--', linewidth=1.5, alpha=0.8, label='(50,50)')
ax2.plot([50, 50], [0, 50], color='green', linestyle='--', linewidth=1.5, alpha=0.8)

# 设置第二个子图的标题和标签（使用更大的字体）
ax2.set_xlabel('Width (pixels)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Height (pixels)', fontsize=14, fontweight='bold')
ax2.set_title('Landslide4Scene Dataset', fontsize=16, fontweight='bold', pad=15)
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 128)

# 调整刻度标签大小
ax2.tick_params(axis='both', which='major', labelsize=12)

# 调整网格线
ax2.grid(True, alpha=0.3, linewidth=0.5)

# 添加图例（第二个子图） - 字体已经通过全局设置放大
ax2.legend(loc='upper right')

# 调整布局，增加子图之间的间距
plt.tight_layout(pad=3.0)

# 保存高分辨率图像（可选）
# plt.savefig('landslide_size_distribution.png', dpi=500, bbox_inches='tight')

plt.show()