import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
config = {
            "font.family": 'serif',
            "font.size": 15,
            "mathtext.fontset": 'stix',
            "font.serif": ['Microsoft YaHei'],
            'axes.unicode_minus':False
         }
plt.rcParams.update(config)

# 24avg 分割成24h也就是288个五分钟的数据，可以存成多个txt文件  把五分钟内的数据取平均值 并且这个平均值乘以500扩大比例

path = "D:\\桌面\\code\\RL-code\\ZBJ_final\\first\\data\\Original\\0\\"
# 从csv文件中读取数据  
s = pd.read_csv(path+"\\3938719206.csv")
s1 = s.iloc[:, 0]
s2 = pd.read_csv(path+"\\output_cpu.txt", sep=' ', header=None)[0] 

def get_data_avg(s1,s2):
    # 将Series转换为numpy数组  
    idx = np.array(s1)  
    cpu = np.array(s2)  
    cpu_avg = []
    interval = 300000000 #5分钟，数据是以微秒为单位
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
                cpu_avg.append(avg*500) #扩大比例
                last_time += interval
                avg = cpu[i]
                len = 1
            else :
                avg += cpu[i]
                len += 1  
    avg /= len
    cpu_avg.append(avg*500)
    print(cpu_avg.__len__()) 
    return cpu_avg

def write_data(filename,array):
    with open(path+"\\"+filename, 'w') as file:  
        # 遍历数组中的每个元素，并将它们写入文件，每个元素后面跟一个换行符  
        for element in array:  
            file.write(str(element) + '\n')

def cut_data_24h_write_data(cpu_avg):
    for i in range(0,cpu_avg.__len__(),288):
        if(i==0): 
            qie = cpu_avg[:288]
        elif(i+288>=cpu_avg.__len__()):
            qie = cpu_avg[i:]
            break
        else:
            qie = cpu_avg[i:i+288]
        print(qie.__len__()) 
        filename = "workerload"+str(i//288)+"(24).txt"
        write_data(filename,qie)
        # 使用matplotlib绘制线图  
        plt.figure(figsize=(8,5.5))
        plt.plot(qie,color='black')  
        plt.axhline(y=20, color='red', linestyle='-') 
        # 设置标题和轴标签（可选）  
        # plt.title("(d) VM2资源序列",y=-0.145)  #+str(i//288)
        plt.xlabel('单位时间(5min)')  
        plt.ylabel('CPU 资源利用率(%)')  
        # 显示图形  
        plt.show()
        continue

cpu_avg = get_data_avg(s1,s2)
write_data("output_cpu_avg.txt",cpu_avg)
# cut_data_24h_write_data(cpu_avg)

