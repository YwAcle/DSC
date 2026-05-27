# import matplotlib.pyplot as plt

# # Set global font to Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'

# # 数据准备：Neighbor Detect 参数值及其对应的 mAP 和 Rank-1 平均值
# # 平均值 = (RGB→IR + IR→RGB) / 2
# params = [0.75, 0.8, 0.85, 0.9, 0.95]
# # mAP 平均值：从每个文件的选定 epoch 计算
# map_avg = [
#     (41.45 + 38.94) / 2,  # 0.75, Epoch 35
#     (41.45 + 38.94) / 2,  # 0.8, Epoch 35
#     (41.37 + 37.63) / 2,  # 0.85, Epoch 45
#     (40.09 + 40.14) / 2,  # 0.9, Epoch 33
#     (39.71 + 36.99) / 2,  # 0.95, Epoch 37
# ]
# # Rank-1 平均值：从每个文件的选定 epoch 计算
# rank1_avg = [
#     (42.77 + 40.04) / 2,  # 0.75, Epoch 35
#     (42.77 + 40.04) / 2,  # 0.8, Epoch 35
#     (41.41 + 36.40) / 2,  # 0.85, Epoch 45
#     (41.80 + 41.38) / 2,  # 0.9, Epoch 33
#     (41.80 + 38.89) / 2,  # 0.95, Epoch 37
# ]

# # 创建折线图
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # 绘制 mAP 折线（左侧纵坐标，浅蓝色）
# ax1.plot(params, map_avg, color='lightgreen', marker='o', label='mAP Average')
# ax1.set_xlabel('(a) Neighbor Detect Parameter of DSJL', fontsize=24)  # 横坐标标签字体设为24
# ax1.set_ylabel('mAP Average (%)', color='green', fontsize=24)  # 纵坐标标签字体设为24
# ax1.tick_params(axis='y', labelcolor='green', labelsize=20)  # 左侧纵坐标刻度字体设为20
# ax1.tick_params(axis='x', labelsize=20)  # 横坐标刻度字体设为20
# # 设置横坐标只显示 0.75, 0.8, 0.85, 0.9, 0.95
# ax1.set_xticks(params)
# # 设置mAP纵坐标范围为[35, 45]
# ax1.set_ylim(35, 45)

# # 创建右侧纵坐标并绘制 Rank-1 折线（橙色）
# ax2 = ax1.twinx()
# ax2.plot(params, rank1_avg, color='orange', marker='s', label='Rank-1 Average')
# ax2.set_ylabel('Rank-1 Average (%)', color='orange', fontsize=24)  # 右侧纵坐标标签字体设为24
# ax2.tick_params(axis='y', labelcolor='orange', labelsize=20)  # 右侧纵坐标刻度字体设为20
# # 设置Rank-1纵坐标范围为[35, 45]
# ax2.set_ylim(35, 45)

# # 添加图例
# fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=20)

# # 移除背景网格
# ax1.grid(False)
# ax2.grid(False)

# # 调整布局并显示
# plt.tight_layout()
# plt.savefig('hm_ht_ablation_visual_robust_largefont.png')
# plt.show()

# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# 如需 Times New Roman：
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']

# ---- 数据 ----
alpha = np.array([0.75, 0.8, 0.85, 0.9, 0.95])
map_avg = np.array([
    (41.45 + 38.94) / 2,
    (41.45 + 38.94) / 2,
    (41.37 + 37.63) / 2,
    (40.09 + 40.14) / 2,
    (39.71 + 36.99) / 2
])
rank1_avg = np.array([
    (42.77 + 40.04) / 2,
    (42.77 + 40.04) / 2,
    (41.41 + 36.40) / 2,
    (41.80 + 41.38) / 2,
    (41.80 + 38.89) / 2
])

# ---- 配色（与你原图保持一致）----
COLOR_MAP   = '#F57C00'  # 橙色：mAP
COLOR_RANK1 = '#448AFF'  # 蓝色：Rank-1

# ---- 尺寸与字号：按最终插入论文的大小来（建议单子图宽 3.5 inch 左右）----
FIG_W, FIG_H = 3.5, 2.6
FONT_LABEL = 9
FONT_TICK  = 8
FONT_LEG   = 8

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

# 左轴：mAP（橙色）
l1 = ax.plot(alpha, map_avg,
             color=COLOR_MAP,
             marker='o',
             linewidth=2.0,
             markersize=4.5,
             label='mAP')
ax.set_ylabel('mAP', fontsize=FONT_LABEL)
ax.tick_params(axis='y', labelsize=FONT_TICK)

# 右轴：Rank-1（蓝色）
axr = ax.twinx()
l2 = axr.plot(alpha, rank1_avg,
              color=COLOR_RANK1,
              marker='v',
              linewidth=2.0,
              markersize=4.5,
              label='Rank-1')
axr.set_ylabel('Rank-1', fontsize=FONT_LABEL)
axr.tick_params(axis='y', labelsize=FONT_TICK)

# x轴
ax.set_xlabel(r'$\alpha$', fontsize=FONT_LABEL)
ax.set_xticks(alpha)
ax.tick_params(axis='x', labelsize=FONT_TICK)

# y轴范围（无断轴，留少量边距）
ax.set_ylim(map_avg.min() - 0.5, map_avg.max() + 0.5)
axr.set_ylim(rank1_avg.min() - 0.8, rank1_avg.max() + 0.8)

# 网格（保持你原来的虚线风格）
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

# 合并 legend（右上角，与你原图习惯一致）
lines = l1 + l2
labels = [ln.get_label() for ln in lines]
ax.legend(lines, labels,
          loc='upper left',
          ncol=2,                 # 一行两列
          fontsize=FONT_LEG,
          frameon=True,
          columnspacing=1.2,      # 两列之间的距离
          handletextpad=0.4,      # 线段到文字的距离
          borderaxespad=0.3)      # legend 与坐标轴的间距

# 边框
for sp in ax.spines.values():
    sp.set_linewidth(1.0)
for sp in axr.spines.values():
    sp.set_linewidth(1.0)

plt.tight_layout()
plt.savefig('fig4_a_alpha.png', dpi=600, bbox_inches='tight')
plt.show()

