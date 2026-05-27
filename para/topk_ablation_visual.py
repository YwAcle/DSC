# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# 如需 Times New Roman：
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']

# ---------------- Data ----------------
topk = np.array([10, 15, 20, 25, 30])

map_avg = np.array([
    (38.09 + 36.66) / 2,
    (38.09 + 36.66) / 2,
    (40.09 + 40.14) / 2,
    (39.21 + 37.18) / 2,
    (37.69 + 39.01) / 2,
])

rank1_avg = np.array([
    (42.38 + 37.16) / 2,
    (42.38 + 37.16) / 2,
    (41.80 + 41.38) / 2,
    (40.43 + 38.12) / 2,
    (37.70 + 38.89) / 2,
])

# ---------------- Style (keep consistent with your original) ----------------
COLOR_MAP   = '#F57C00'  # orange
COLOR_RANK1 = '#448AFF'  # blue

# 建议按“最终插入论文的物理尺寸”来设定 figsize（这里给一个单子图常用尺寸）
FIG_W, FIG_H = 3.5, 2.6
FONT_LABEL = 9
FONT_TICK  = 8
FONT_LEG   = 8

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

# Left y-axis: mAP
l1 = ax.plot(topk, map_avg,
             color=COLOR_MAP, marker='o',
             linewidth=2.0, markersize=4.5,
             label='mAP')
ax.set_ylabel('mAP', fontsize=FONT_LABEL)
ax.tick_params(axis='y', labelsize=FONT_TICK)

# Right y-axis: Rank-1
axr = ax.twinx()
l2 = axr.plot(topk, rank1_avg,
              color=COLOR_RANK1, marker='v',
              linewidth=2.0, markersize=4.5,
              label='Rank-1')
axr.set_ylabel('Rank-1', fontsize=FONT_LABEL)
axr.tick_params(axis='y', labelsize=FONT_TICK)

# X-axis
ax.set_xticks(topk)
ax.tick_params(axis='x', labelsize=FONT_TICK)
ax.set_xlabel('top-k', fontsize=FONT_LABEL)

# 如果你想像之前一样只显示 (b)（不显示 top-k），把上面两行改成：
# ax.set_xlabel('(b)', fontsize=FONT_LABEL)

# Y limits (no broken axis) — 给一点边距即可
ax.set_ylim(map_avg.min() - 0.8, map_avg.max() + 0.8)
axr.set_ylim(rank1_avg.min() - 1.0, rank1_avg.max() + 1.0)

# Grid (light)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

# Legend in ONE line and avoid overlap: put it above the axes
lines = l1 + l2
labels = [ln.get_label() for ln in lines]
ax.legend(lines, labels,
          loc='lower center',
          bbox_to_anchor=(0.5, 1.02),  # 图外顶部，基本不会压线
          ncol=2,
          fontsize=FONT_LEG,
          frameon=True,
          columnspacing=1.0,
          handletextpad=0.4)

# Optional: add (b) subfigure label under plot (更像论文排版)
# ax.set_title('(b)', y=-0.45, fontsize=FONT_LABEL)

# Borders
for sp in ax.spines.values():
    sp.set_linewidth(1.0)
for sp in axr.spines.values():
    sp.set_linewidth(1.0)

plt.tight_layout()
plt.savefig('fig4_b_topk.pdf', bbox_inches='tight')
plt.savefig('fig4_b_topk.png', dpi=600, bbox_inches='tight')
plt.show()
