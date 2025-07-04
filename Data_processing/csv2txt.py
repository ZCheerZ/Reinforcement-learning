import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt

path = "Data_processing\\1"
# 从csv文件中读取数据  
job_id = 494787089
s = pd.read_csv(path+"\\"+str(job_id)+"_info.csv")
startTime = s.iloc[:, 0]
# print(startTime)

# 将Series转换为numpy数组  
idx = np.array(startTime)  
taskNum = []
interval = 300000000 #5分钟，数据是以毫秒为单位
last_time = idx[0]
len = 0
#里面会存在很多不是五分钟的数据，可能导致数据长度不统一！
for i in range(idx.__len__()):
    if(i==0) : 
        len += 1
    elif(idx[i]==idx[i-1]) :
        len += 1    
    else : 
        if(idx[i]==last_time+interval):  #解决上述问题
            #if(idx[i]>=idx[0]+86400000000): break  #这是24小时的数据
            taskNum.append(len)
            last_time += interval
            len = 1
        else :
            len += 1  

taskNum.append(len)


print(taskNum.__len__()) 

with open(path+"\\"+str(job_id)+".txt", 'w') as file:  
    # 遍历数组中的每个元素，并将它们写入文件，每个元素后面跟一个换行符  
    for element in taskNum:  
        file.write(str("{:.1f}".format(element)) + '\n')
# 使用matplotlib绘制线图  
plt.plot(taskNum)  
  
# 设置标题和轴标签（可选）  
plt.title(str(job_id)+' taskNum')  
plt.xlabel('index')  
plt.ylabel('number')  
  
# 显示图形  
plt.show()