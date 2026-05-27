# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import shap

# ==================== 数据预处理 ====================
# 原始数据
data = {
    "鱼体大小": [0.007, 0.007, 0.007, 0.007, 0.007, 8.7, 8.7, 8.7, 8.7,
             10.06, 10.06, 10.06, 10.06, 10.06, 10.06, 10.06, 10.06,
             14.46, 14.46, 14.46, 14.46, 14.46, 14.46, 21.27, 21.27,
             21.27, 109, 109, 109],
    "温度范围": ["25.0~29.5", "25.0~29.5", "25.0~29.5", "25.0~29.5", "25.0~29.5",
             "25.6", "25.6", "25.6", "25.6",
             "28", "28", "28", "28", "28", "28", "28", "28",
             "24", "24", "24", "24", "24", "24",
             "25~32", "25~32", "25~32",
             "27~29", "27~29", "27~29"],
    "蛋白质添加量": [45, 48, 51, 54, 57, 42, 45, 48, 51, 31.71, 34.64, 37.57,
               40.51, 43.44, 46.37, 49.3, 52.23, 34, 38, 42, 46, 50, 54,
               46.64, 43.7, 44.64, 46, 50, 54],
    "蛋白质沉积率": [32.5, 32, 35.5, 22.2, 14.3, 37, 43.5, 37.8, 39.6,
               35.2, 40.9, 44.7, 44, 48.4, 49.7, 51.6, 44.5, 23.57,
               26.74, 33.4, 28.43, 32.12, 22.07, 43.43, 44.64, 45.1,
               34.36, 33.45, 33.66]
}

df = pd.DataFrame(data)


# 温度处理函数
def process_temp(temp_str):
    if "~" in temp_str:
        t1, t2 = map(float, temp_str.split("~"))
        return (t1 + t2) / 2
    return float(temp_str)


# 数据清洗
df["温度"] = df["温度范围"].apply(process_temp)
df_clean = df[
    (df["蛋白质沉积率"] > 20) &
    (df["鱼体大小"] > 0.1)
    ].copy()

# 特征标准化
scaler = StandardScaler()
features = ["鱼体大小", "蛋白质添加量", "温度"]
df_clean[["W_norm", "P_norm", "T_norm"]] = scaler.fit_transform(df_clean[features])


# ==================== 优化生物机理模型 ====================
class EnhancedBioModel:
    def __init__(self):
        self.params = None
        self.scaler = None
        self.feature_names = ["鱼体大小", "蛋白质添加量", "温度"]

    def _model_func(self, X, α, k, a, b, c, Ea):
        """优化后的模型方程"""
        W_norm, P, T = X
        # 逆标准化获得实际鱼体大小
        # 标准化时使用的是 ["鱼体大小", "蛋白质添加量", "温度"] 顺序
        # 检查W_norm是否为数组
        if isinstance(W_norm, (list, np.ndarray)):
            # 如果是批量数据，处理第一个元素
            W_norm_scalar = W_norm[0]
        else:
            # 如果已经是标量，则直接使用
            W_norm_scalar = W_norm
        
        normalized_vector = np.zeros(3)
        normalized_vector[0] = W_norm_scalar  # 只设置第一个特征值
        W = self.scaler.inverse_transform([normalized_vector])[0][0]
        R = 8.314  # 气体常数 (J/mol·K)

        Km = a * W + b * T + c
        f_W = np.exp(-k * W)
        g_T = np.exp(-Ea / (R * (T + 273.15)))
        return α * f_W * (P / (Km + P)) * g_T

    def fit(self, X, y):
        """两阶段优化拟合"""
        # 参数边界 (α, k, a, b, c, Ea)
        bounds = [
            (30, 80),  # α
            (0.03, 0.08),  # k 
            (0.3, 0.7),  # a
            (0.2, 0.5),  # b
            (25, 45),  # c
            (40000, 60000)  # Ea (J/mol)
        ]

        # 第一阶段：全局优化
        def loss_func(params):
            try:
                pred = [self._model_func((w, p, t), *params) for w, p, t in zip(*X)]
                return np.sum((np.array(pred) - y) ** 2)
            except:
                return np.inf

        global_result = differential_evolution(
            loss_func,
            bounds,
            strategy='best1bin',
            maxiter=100,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7
        )

        # 第二阶段：局部优化
        self.params, _ = curve_fit(
            self._model_func, X, y,
            p0=global_result.x,
            bounds=list(zip(*bounds)),
            maxfev=5000
        )
        return self

    def predict(self, X):
        """批量预测"""
        return np.array([self._model_func(x, *self.params) for x in zip(*X)])

    def evaluate(self, X, y):
        """模型评估"""
        pred = self.predict(X)
        return {
            "R2": r2_score(y, pred),
            "RMSE": np.sqrt(mean_squared_error(y, pred)),
            "预测值": pred
        }

    def validate_parameters(self):
        """参数生物学验证"""
        α, k, a, b, c, Ea = self.params
        print("\n=== 参数生物学验证 ===")
        # α验证
        valid_α = "合理" if 30 <= α <= 80 else "警告：超出常见范围"
        print(f"α(基础代谢): {α:.1f} {valid_α}")

        # k验证
        valid_k = "合理" if 0.03 <= k <= 0.08 else "警告：可能过度衰减"
        print(f"k(体型衰减): {k:.4f}/g {valid_k}")

        # Km验证
        W_test = [10, 50, 100]
        T_test = 28
        print("\nKm验证(半饱和浓度):")
        for W in W_test:
            Km = a * W + b * T_test + c
            status = "正常" if 30 <= Km <= 60 else "异常"
            print(f"  {W}g鱼体@{T_test}℃: {Km:.1f}% ({status})")

        # Ea验证
        Ea_kj = Ea / 1000
        valid_Ea = "合理" if 40 <= Ea_kj <= 60 else "警告：活化能异常"
        print(f"\nEa(活化能): {Ea_kj:.1f} kJ/mol {valid_Ea}")

    def plot_diagnostics(self, y_true, pred):
        """可视化诊断"""
        plt.figure(figsize=(15, 5))

        # 预测 vs 实际
        plt.subplot(131)
        plt.scatter(y_true, pred, alpha=0.6)
        plt.plot([20, 55], [20, 55], 'r--')
        plt.xlabel("实际值")
        plt.ylabel("预测值")
        plt.title(f"预测性能 (R²={r2_score(y_true, pred):.2f})")

        # 残差分析
        plt.subplot(132)
        residuals = y_true - pred
        plt.scatter(pred, residuals)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel("预测值")
        plt.ylabel("残差")
        plt.title("残差分布")

        # 温度效应
        plt.subplot(133)
        T_range = np.linspace(24, 32, 100)
        effect = [self.predict([[df_clean['W_norm'].mean()], [40], [t]])[0] for t in T_range]
        plt.plot(T_range, effect)
        plt.xlabel("温度 (℃)")
        plt.ylabel("预测沉积率")
        plt.title("温度效应曲线(P=40%)")

        plt.tight_layout()
        plt.show()


# ==================== 模型训练与验证 ====================
# 准备数据
X_train = [
    df_clean["W_norm"].values,
    df_clean["蛋白质添加量"].values,
    df_clean["温度"].values
]
y_train = df_clean["蛋白质沉积率"].values

# 初始化模型
model = EnhancedBioModel()
model.scaler = scaler

# 训练模型
model.fit(X_train, y_train)

# 评估模型
results = model.evaluate(X_train, y_train)
print(f"\n=== 模型性能 ===")
print(f"R²: {results['R2']:.3f}")
print(f"RMSE: {results['RMSE']:.2f}%")

# 参数验证
model.validate_parameters()

# 可视化诊断
model.plot_diagnostics(y_train, results["预测值"])

# ==================== 可解释性分析 ====================
# SHAP分析
# 准备SHAP数据
X_shap = df_clean[["W_norm", "P_norm", "T_norm"]].values

# 创建一个包装函数，直接处理numpy数组
def model_wrapper(X):
    # X是一个形状为(样本数, 特征数)的numpy数组
    return model.predict([X[:, 0], X[:, 1], X[:, 2]])

# 创建explainer
explainer = shap.Explainer(model_wrapper, X_shap)
shap_values = explainer(X_shap)
shap.plots.beeswarm(shap_values)


# ==================== 预测接口 ====================
def predict(W, P, T):
    """预测新样本（输入原始值）"""
    scaled = scaler.transform([[W, P, T]])
    return model._model_func(
        (scaled[0][0], P, T),
        *model.params
    )


# 示例预测
print("\n=== 示例预测 ===")
print(f"10g鱼体@28℃添加45% → {predict(10, 45, 28):.1f}%")
print(f"100g鱼体@30℃添加50% → {predict(100, 50, 30):.1f}%")