# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False

# ---- Data ----
alpha = np.array([0.75, 0.8, 0.85, 0.9, 0.95])
map_avg = np.array([
    (41.45 + 38.94) / 2,
    (41.45 + 38.94) / 2,
    (41.37 + 37.63) / 2,
    (40.09 + 40.14) / 2,
    (39.71 + 36.99) / 2,
])
rank1_avg = np.array([
    (42.77 + 40.04) / 2,
    (42.77 + 40.04) / 2,
    (41.41 + 36.40) / 2,
    (41.80 + 41.38) / 2,
    (41.80 + 38.89) / 2,
])

x = np.arange(1, len(alpha) + 1)
group_labels = [f"{v:g}" for v in alpha]

fig, (ax2, ax1) = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(6, 4),
    gridspec_kw={'height_ratios': [1, 1]},
    constrained_layout=True,
)

# Bottom (mAP)
ax1.plot(x, map_avg, marker='o', markersize=8, color="#F97306", label="mAP", linewidth=2)
for (xi, yi) in zip(x, map_avg):
    ax1.text(xi, yi + 0.1, f"{yi:.2f}", ha='center', va='bottom', fontsize=12)
ax1.set_ylim(map_avg.min() - 0.8, map_avg.max() + 0.8)
ax1.yaxis.grid(True, linestyle='--')

# Top (Rank-1)
ax2.plot(x, rank1_avg, marker='v', markersize=8, color="#448EE4", label="Rank-1", linewidth=2)
for (xi, yi) in zip(x, rank1_avg):
    ax2.text(xi, yi + 0.15, f"{yi:.2f}", ha='center', va='bottom', fontsize=12)
ax2.set_ylim(rank1_avg.min() - 0.8, rank1_avg.max() + 0.8)
ax2.yaxis.grid(True, linestyle='--')

plt.xticks(x, group_labels, fontsize=14)
ax1.set_xlabel(r"(a) Neighbor Detect Parameter of DSJL", fontsize=14)

plt.setp(ax2.get_xticklabels(), visible=False)

# Broken-axis marks
d = .015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (-d, +d), **kwargs)
ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)

ax1.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.tick_params(bottom=False)
ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)

fig.legend(loc='center', bbox_to_anchor=(0.75, 0.91), ncol=2, fontsize=12)

plt.savefig('./fig4_a_alpha_refactor.png', format='png', dpi=600)
plt.show()
