import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import LinearRegression

class STLBaselinePredictor:
    """
    基于 STL 分解的云资源基线预测器 (Robust Baseline Predictor)
    """
    def __init__(self, period, seasonal_smoother=13, robust=True):
        """
        初始化参数
        :param period: 周期长度 (例如: 24小时数据，period=24)
        :param seasonal_smoother: STL参数，季节平滑窗口，通常为奇数且 >= 7
        :param robust: 是否开启鲁棒模式 (关键参数，用于忽略突发)
        """
        self.period = period
        self.seasonal_smoother = seasonal_smoother
        self.robust = robust
        self.model_res = None
        self.history_trend = None
        self.history_seasonal = None

    def fit(self, time_series):
        """
        对历史序列进行 STL 分解
        :param time_series: 历史资源需求序列 (pd.Series)
        """
        # 1. STL 分解
        stl = STL(time_series, 
                  period=self.period, 
                  seasonal=self.seasonal_smoother,
                  robust=self.robust) # 开启鲁棒模式，抵抗突发噪音
        self.model_res = stl.fit()
        
        # 提取分量
        self.history_trend = self.model_res.trend
        self.history_seasonal = self.model_res.seasonal
        
        return self.model_res

    def predict(self, horizon):
        """
        预测未来一个周期的基线
        :param horizon: 预测步长 (例如未来 24 小时)
        :return: 预测的基线序列 (pd.Series)
        """
        if self.model_res is None:
            raise ValueError("Model must be fitted before prediction.")

        # --- A. 趋势预测 (Trend Forecasting) ---
        # 策略：取历史 Trend 的最后一段进行线性回归外推，保证趋势的连续性
        # 这里取最近的 2 个周期长度的数据来拟合趋势方向
        lookback = min(len(self.history_trend), self.period * 2)
        
        y_train = self.history_trend.values[-lookback:]
        x_train = np.arange(lookback).reshape(-1, 1)
        
        reg = LinearRegression()
        reg.fit(x_train, y_train)
        
        # 生成未来的时间索引
        x_future = np.arange(lookback, lookback + horizon).reshape(-1, 1)
        trend_future = reg.predict(x_future)
        
        # --- B. 季节预测 (Seasonal Forecasting) ---
        # 策略：直接复用历史中最近的一个完整周期的形态 (Naive Seasonal)
        # 取出最后 period 长度的季节分量
        last_season = self.history_seasonal.values[-self.period:]
        
        # 如果预测长度 horizon > period，需要循环拼接 (Tiling)
        seasonal_future = np.resize(last_season, horizon)
        
        # --- C. 合成基线 (Synthesis) ---
        # 核心策略：Baseline = Trend + Seasonal (忽略 Residual/Burst)
        baseline_forecast = trend_future + seasonal_future
        
        # 构造带有时间索引的 Series 返回 (假设输入是等间隔的)
        last_date = self.history_trend.index[-1]
        freq = pd.infer_freq(self.history_trend.index)
        if freq is None: freq = 'H' # 默认为小时
            
        future_dates = pd.date_range(start=last_date, periods=horizon+1, freq=freq)[1:]
        
        return pd.Series(baseline_forecast, index=future_dates)

# ==========================================
# 模拟实验：验证对突发流量的忽略能力
# ==========================================

# 1. 生成模拟数据 (Simulate Data)
# 设定：14天数据，每小时一个点，周期为24
np.random.seed(42)
days = 14
period = 24
n_points = days * period
time_index = pd.date_range(start='2024-01-01', periods=n_points, freq='H')

# 基础成分
trend = np.linspace(50, 80, n_points) # 缓慢增长的趋势
season = 10 * np.sin(2 * np.pi * np.arange(n_points) / period) # 周期性波动

# 突发噪音 (Bursts) - 这就是你要过滤的
bursts = np.zeros(n_points)
# 随机添加几个巨大的突发峰值
burst_indices = np.random.choice(n_points, 10, replace=False)
bursts[burst_indices] = np.random.randint(30, 60, 10) # 突发量很大

# 正常的白噪声
noise = np.random.normal(0, 2, n_points)

# 合成总数据
data = trend + season + bursts + noise
series = pd.Series(data, index=time_index)

# 2. 运行 STL 基线预测
# 假设我们要预测未来 24 小时
forecast_horizon = 24

predictor = STLBaselinePredictor(period=period, robust=True)
stl_result = predictor.fit(series)
future_baseline = predictor.predict(forecast_horizon)

# 3. 可视化结果
plt.figure(figsize=(15, 8))

# 绘制历史数据
plt.plot(series.index[-100:], series.values[-100:], label='History (Real Data)', color='lightgray', linewidth=2)
# 标记突发点
burst_mask = bursts[-100:] > 0
if np.any(burst_mask):
    plt.scatter(series.index[-100:][burst_mask], series.values[-100:][burst_mask], color='red', label='Bursts (Ignored)', zorder=5)

# 绘制 STL 提取的历史 Trend+Seasonal (验证拟合效果)
history_baseline = predictor.history_trend + predictor.history_seasonal
plt.plot(history_baseline.index[-100:], history_baseline.values[-100:], label='STL Fitted Baseline', color='blue', linestyle='--')

# 绘制未来预测
plt.plot(future_baseline.index, future_baseline.values, label='Future Baseline Prediction', color='green', linewidth=3)

plt.title(f"STL Robust Baseline Prediction (Ignoring Bursts)\nForecast Horizon: {forecast_horizon} Steps", fontsize=14)
plt.xlabel("Time")
plt.ylabel("Resource Demand")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 打印分解组件查看
# stl_result.plot()
# plt.show()