import re
import pandas as pd

# 读取日志文件
with open("/data1/wangzj/experience/contrast/Neighbor-guided-USL-VI-ReID-main/logs/vcm_s2/log.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 匹配测试结果
pattern = re.compile(
    r"Test Epoch\[(\d+)\]:.*?"
    r"FC\('v2t'\)\s+Rank-1:\s+([\d.]+)%\s+Rank-5:\s+([\d.]+)%\s+Rank-10:\s+([\d.]+)%\s+Rank-20:\s+([\d.]+)%\s+mAP:\s+([\d.]+)%.*?"
    r"FC\('t2v'\)\s+Rank-1:\s+([\d.]+)%\s+Rank-5:\s+([\d.]+)%\s+Rank-10:\s+([\d.]+)%\s+Rank-20:\s+([\d.]+)%\s+mAP:\s+([\d.]+)%",
    re.DOTALL
)

# 提取数据
records = pattern.findall(text)

# 构建 DataFrame
data = []
for rec in records:
    epoch = int(rec[0])
    v2t_metrics = list(map(float, rec[1:7]))
    t2v_metrics = list(map(float, rec[7:]))
    data.append([epoch] + v2t_metrics + t2v_metrics)

columns = [
    "Epoch",
    "v2t_Rank1", "v2t_Rank5", "v2t_Rank10", "v2t_Rank20", "v2t_mAP",
    "t2v_Rank1", "t2v_Rank5", "t2v_Rank10", "t2v_Rank20", "t2v_mAP"
]

df = pd.DataFrame(data, columns=columns)

# 保存所有数据
# df.to_csv("t2v_v2t_full_metrics.csv", index=False)

# 找到 Rank-1 最好的 epoch
best_v2t = df.loc[df["v2t_Rank1"].idxmax()]
best_t2v = df.loc[df["t2v_Rank1"].idxmax()]

# 打印最佳表现的完整指标
print("====== 最佳 v2t Rank-1 表现 ======")
print(f"Epoch: {best_v2t['Epoch']}")
print(f"Rank-1: {best_v2t['v2t_Rank1']}%")
print(f"Rank-5: {best_v2t['v2t_Rank5']}%")
print(f"Rank-10: {best_v2t['v2t_Rank10']}%")
print(f"mAP: {best_v2t['v2t_mAP']}%")

print("\n====== 最佳 t2v Rank-1 表现 ======")
print(f"Epoch: {best_t2v['Epoch']}")
print(f"Rank-1: {best_t2v['t2v_Rank1']}%")
print(f"Rank-5: {best_t2v['t2v_Rank5']}%")
print(f"Rank-10: {best_t2v['t2v_Rank10']}%")
print(f"mAP: {best_t2v['t2v_mAP']}%")
