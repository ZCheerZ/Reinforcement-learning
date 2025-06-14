import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略特定警告,只显示ERROR和WARNING


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.callbacks import EarlyStopping

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 2. 全局设置字体大小
plt.rcParams['font.size'] = 12  # 全局默认字体大小

def load_single_column_time_series(file_path):

    # header=None表示没有标题行，values获取numpy数组
    df = pd.read_csv(file_path, header=None)
    #df = df[:288]  # 截取完整的天数（288的整数倍）
    # reshape(-1,1)确保变为二维数组，-1表示自动计算行数
    return df.values.reshape(-1, 1)


def preprocess_data(data, time_steps=10, train_ratio=0.8):

    # 1. 数据标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    # fit_transform同时计算并应用变换，输入必须是二维
    scaled_data = scaler.fit_transform(data)

    # 2. 创建监督学习数据集
    def create_dataset(dataset, look_back=1):

        X, y = [], []
        # 遍历创建每个窗口，i从0到len(dataset)-look_back-1
        for i in range(len(dataset) - look_back):
            # 取当前点开始的look_back个点作为特征
            X.append(dataset[i:(i + look_back), 0])
            # 取下一个点作为标签
            y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(y)

    # 调用创建数据集
    X, y = create_dataset(scaled_data, time_steps)

    # 3. 划分训练测试集
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 4. 调整LSTM输入形状 [samples, time_steps, features]
    # LSTM要求输入为3D张量：(样本数, 时间步长, 特征数)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test, scaler



def build_lstm_model(shape):

    model = Sequential([

        # 显式添加Input层作为第一层
        InputLayer(input_shape=shape),  # 替代原来的input_shape参数

        # 第一层LSTM
        LSTM(128, activation='relu', return_sequences=True),  #第一层128个神经元
        # 第二层LSTM
        LSTM(64, activation='relu'),   #第二层64个神经元
        # 输出层，1个神经元对应预测的单个值
        Dense(1)
    ])
    # 编译模型，回归问题使用均方误差
    model.compile(optimizer='adam', loss='mse')
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):

    # 早停回调，监控验证集损失，10轮无改善则停止
    early_stopping = EarlyStopping(
        monitor='val_loss',  # 监控指标
        patience=10,  # 容忍轮数
        restore_best_weights=True  # 恢复最佳权重
    )

    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1  # 显示进度条
    )
    return model, history


def make_predictions(model, data, scaler, time_steps):
    """
    （增强版）返回预测结果和对应的真实值（不含nan）
    """
    scaled_data = scaler.transform(data)

    X_pred = []
    for i in range(len(scaled_data) - time_steps):
        X_pred.append(scaled_data[i:(i + time_steps), 0])
    X_pred = np.array(X_pred).reshape(-1, time_steps, 1)

    scaled_predictions = model.predict(X_pred)
    predictions = scaler.inverse_transform(scaled_predictions)

    full_predictions = np.empty_like(data)
    full_predictions[:] = np.nan
    full_predictions[time_steps:, 0] = predictions.flatten()

    # 返回预测值和对应的真实值（跳过前time_steps个点）
    return full_predictions, data[time_steps:]

def calculate_metrics(y_true, y_pred):

    # 过滤掉nan值（初始无法预测的部分）
    mask = ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    metrics = {
        '平均绝对误差': mean_absolute_error(y_true, y_pred),    #平均绝对误差，越小越好
        #'均方误差': mean_squared_error(y_true, y_pred),     #均方误差，越小越好
        '均方根误差': np.sqrt(mean_squared_error(y_true, y_pred)),    #均方根误差，越小越好
        #'方差': r2_score(y_true, y_pred),     #方差，越小越好
        '平均绝对百分比误差': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)) * 100)  # 避免除以0，平均绝对百分比误差，越小越好
    }
    return metrics

def save_predictions(original_data, predictions, metrics, file_path):

    result_df = pd.DataFrame({
        'Original': original_data.flatten(),
        'Predicted': predictions.flatten()
    })

    # 添加评估指标到文件末尾
    metrics_str = "\n\nEvaluation Metrics:\n"
    for name, value in metrics.items():
        metrics_str += f"{name}: {value:.4f}" + ("%\n" if name == "平均绝对百分比误差" else "\n")

    # 保存到CSV
    result_df.to_csv(file_path, index=False)

    # 追加写入评估指标
    with open(file_path, 'a') as f:
        f.write(metrics_str)
    print(f"预测结果和评估指标已保存到: {file_path}")



def plot_results(original_data, predictions, metrics, time_steps):

    plt.figure(figsize=(12, 6))

    # 主图：原始数据和预测结果
    plt.plot(original_data, label='原始数据')
    plt.plot(predictions, label='预测数据', alpha=0.7)
    plt.axvline(x=len(original_data) * 0.8, color='r', linestyle='--', label='训练集/测试集分割线')

    # 在右上角显示评估指标
    metric_text = "\n".join([f"{name}: {value:.4f}{'%' if name == '平均绝对百分比误差' else ''}"
                             for name, value in metrics.items()])
    plt.text(0.95, 0.95, metric_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             fontsize=15,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round'))

    plt.legend(loc='upper left',fontsize=15)
    plt.xlabel("时间/5分钟",fontsize=15)
    plt.ylabel('CPU资源使用率(%)',fontsize=15)
    plt.title(f'')
    plt.tight_layout()
    plt.show()

def main():
    """主函数：执行完整预测流程"""
    # 参数配置
    FILE_PATH = r"Prediction/cpu_avg.txt"  # 输入数据路径
    SAVE_PATH = r"Prediction/LSTM_pre_cal.csv"  # 结果保存路径
    TIME_STEPS = 10    # 时间窗口大小
    TRAIN_RATIO = 0.8    # 训练集比例
    EPOCHS = 100   # 最大训练轮数
    BATCH_SIZE = 32   # 批量大小

    # 1. 加载数据
    print("步骤1: 加载数据...")
    time_series = load_single_column_time_series(FILE_PATH)
    print(f"数据加载完成，共{len(time_series)}个数据点")

    # 2. 数据预处理
    print("\n步骤2: 数据预处理...")
    X_train, y_train, X_test, y_test, scaler = preprocess_data(
        time_series, TIME_STEPS, TRAIN_RATIO)
    print(f"训练集形状: {X_train.shape} (样本, 时间步长, 特征)")
    print(f"测试集形状: {X_test.shape}")

    # 3. 构建模型
    print("\n步骤3: 构建LSTM模型...")
    model = build_lstm_model((TIME_STEPS, 1))
    model.summary()

    # 4. 训练模型
    print("\n步骤4: 训练模型...")
    model, history = train_model(model, X_train, y_train,
                                 X_test, y_test, EPOCHS, BATCH_SIZE)

    # 5. 绘制训练历史
    print("\n训练历史可视化:")
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.xlabel("训练轮次",fontsize=15)
    plt.ylabel('模型损失',fontsize=15)
    plt.legend(fontsize=15)
    plt.title('')
    plt.show()

    # 6. 进行预测和评估
    print("\n步骤5: 进行预测和评估...")
    predictions, y_true = make_predictions(model, time_series, scaler, TIME_STEPS)

    # 计算评估指标
    metrics = calculate_metrics(y_true, predictions[~np.isnan(predictions)])
    print("\n预测结果评估指标:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}" + ("%" if name == "MAPE" else ""))

    # 7. 保存结果
    save_predictions(time_series, predictions, metrics, SAVE_PATH)

    # 8. 可视化结果
    print("\n预测结果可视化:")
    plot_results(time_series, predictions, metrics, TIME_STEPS)

    return predictions, metrics


if __name__ == "__main__":
    predictions, metrics = main()
    print("\n前20个有效预测值:")
    valid_preds = predictions[~np.isnan(predictions)].flatten()[:20]
    print(valid_preds)