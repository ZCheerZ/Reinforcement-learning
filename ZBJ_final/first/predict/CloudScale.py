import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt

path = "D:\\桌面\\code\\RL-code\\ZBJ_final\\first\\data\\Original\\0"
# 从csv文件中读取数据  
s = pd.read_csv(path+"\\3938719206.csv")
s1 = s.iloc[:, 0]
# print(s1)
s2 = pd.read_csv(path+"\\output_cpu.txt", sep=' ', header=None)[0]  
# print(s2)
# 将Series转换为numpy数组  
idx = np.array(s1)  
cpu = np.array(s2)  
cpu_avg = []
interval = 300000000 #5分钟，数据是以毫秒为单位
last_time = idx[0]
avg = 0.0
len = 0
#里面会存在很多不是五分钟的数据，可能导致数据长度不统一！
for i in range(idx.__len__()):
    if(i==0) : 
        avg += cpu[i]
        len += 1
    elif(idx[i]==idx[i-1]) :
        avg += cpu[i]
        len += 1    
    else : 
        if(idx[i]==last_time+interval):  #解决上述问题
            avg /= len
            cpu_avg.append(avg*500)
            last_time += interval
            avg = cpu[i]
            len = 1
        else :
            avg += cpu[i]
            len += 1  

avg /= len
cpu_avg.append(avg*500)

cloudscale_pre = []
print(cpu_avg.__len__()) 
len_pre  =  cpu_avg.__len__() #288-1

for i in range(0,cpu_avg.__len__(),288):
    if(i==0): 
        qie = cpu_avg[:288]
        cloudscale_pre = qie
    elif(i+288*2>=cpu_avg.__len__()):
        # qie = cpu_avg[i:i+288] 
        # cloudscale_pre = [a + b for a, b in zip(qie, cloudscale_pre)] 
        break
    else:
        qie = cpu_avg[i:i+288]
        cloudscale_pre = [a + b for a, b in zip(qie, cloudscale_pre)] 
    print(qie.__len__()) 
    # with open(path+"\\i.txt", 'w') as file:  
    # # 遍历数组中的每个元素，并将它们写入文件，每个元素后面跟一个换行符  
    #     for element in qie:  
    #         file.write(str(element) + '\n')
    # continue

cloudscale_pre = [x / (len_pre//288-1) for x in cloudscale_pre]
print(cloudscale_pre)
with open(path+"\\1_Cloud.txt", 'w') as file:  
    # 遍历数组中的每个元素，并将它们写入文件，每个元素后面跟一个换行符  
        for element in cloudscale_pre:  
            file.write(str(element) + '\n')

# with open(path+"\\output_cpu_pre0.txt", 'w') as file:  
#     # 遍历数组中的每个元素，并将它们写入文件，每个元素后面跟一个换行符  
#     for element in cpu_avg:  
#         file.write(str(element) + '\n')
# 使用matplotlib绘制线图  
plt.plot(cloudscale_pre)  
  
# 设置标题和轴标签（可选）  
plt.title('CPU')  
plt.xlabel('index')  
plt.ylabel('resource')  
  
# 显示图形  
plt.show()