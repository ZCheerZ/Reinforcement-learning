import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.fftpack import fft , ifft 
from scipy.signal import correlate  
from statsmodels.tsa.ar_model import AutoReg   
  
def extract_dominant_period(U):  
    max_corr = 0  
    L = len(U)  
    for shift in range(1, L // 2):  
        corr = correlate(U, np.roll(U, shift))  
        if corr.max() > max_corr:  
            max_corr = corr.max()  
            L_dominant = shift 
    print("L==",L_dominant)
    return L_dominant  

def fft_prediction(U, L):  
    # FFT变换  
    fft_result = fft(U, L)  
    # 选择低频分量  
    low_freq_indices = np.arange(L // 2 + 1)  
    low_freq_components = fft_result[low_freq_indices]  
    # 逆FFT得到预测的利用时间序列  此处L可换成序列长，之前的显性重复周期  
    predicted_p = ifft(low_freq_components, L).real  
    return predicted_p   

def fft_prediction_all(U, L):  
    for i in range(0,U.__len__(),L):
        if(i==0): 
            qie = U[:288]
            predicted_p_all = fft_prediction(qie,L) 
        else:
            qie = U[i:i+288]
            predicted_p_all = [a + b for a, b in zip(fft_prediction(qie,L), predicted_p_all)] 
    predicted_p_all = [x / (U.__len__()//288) for x in predicted_p_all]
    return predicted_p_all 

def ar2_model(U):  
    # 使用statsmodels库的AutoReg类来估计AR(2)模型  
    model = AutoReg(U, lags=2)  
    # 拟合模型  
    model_fit = model.fit()  
    # 获取模型系数  
    phi_1, phi_2 = model_fit.params[1], model_fit.params[2]  
    # 使用模型系数计算残差  
    residuals = []  
    for t in range(2, len(U)):  
        predicted = phi_1 * U[t-1] + phi_2 * U[t-2]  
        residual = U[t] - predicted  
        residuals.append(residual)  
    # 转换残差列表为numpy数组  
    residuals = np.array(residuals)  
    # 填充前两个残差为NaN，因为AR(2)模型至少需要两个历史值  
    residuals = np.insert(residuals, [0, 1], [0, 0])#[np.nan, np.nan])  
    return residuals   
  
def ar2_residual(U, L):  
    # 实现了一个AR(2)模型来计算残差  
    residuals = ar2_model(U)  
    return residuals[:L]  
  
def final_prediction(U):  
    # 提取显性重复模式  
    L = extract_dominant_period(U)
    if(L==1): L = 288  
    # FFT预测  
    predicted_p = fft_prediction_all(U, L)  
    # AR(2)残差  
    residuals = ar2_residual(U, 288)  
    # 计算预测需求  
    predicted_demand = predicted_p #+ residuals  
    return predicted_demand  
  
# 这只是一个示例框架，具体实现细节可能需要根据实际情况调整
# 示例使用  
# U = [1,3,1,1,2,2,1,2,3]  # 假设U是历史使用序列 
path = "D:\\桌面\\code\\RL-code\\ZBJ_final\\first\\data\\Original\\0"
s = pd.read_csv(path+"\\output_cpu_avg.txt", sep=' ', header=None)[0]  
# 将Series转换为numpy数组  
U = np.array(s)   
U = U[:(U.__len__()//288-1)*288]
print(U.__len__())
final_demand_prediction = final_prediction(U)  
print("----------------")
print(final_demand_prediction) #可以把这个东西求个平均拿出来

with open(path+"\\1_FFT.txt", 'w') as file:  
    # 遍历数组中的每个元素，并将它们写入文件，每个元素后面跟一个换行符  
    for element in final_demand_prediction:  
        file.write(str(element) + '\n')
# 使用matplotlib绘制线图  
plt.plot(final_demand_prediction)  
  
# 设置标题和轴标签（可选）  
plt.title('CPU')  
plt.xlabel('index')  
plt.ylabel('resource')  
  
# 显示图形  
plt.show()