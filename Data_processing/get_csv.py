import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt

def show_cpu(array):
    # 使用matplotlib绘制线图  
    plt.plot(array)  
    # 设置标题和轴标签（可选）  
    plt.title('CPU')  
    plt.xlabel('Time(5min)')  
    plt.ylabel('CPU Usage(%)')  
    # 显示图形  
    plt.show()


# 读取CSV文件  
path = "Data_processing"
job_ids = [515042969,6251983768]
type_id = "1\\"
data_set_N = 1
for i in range(0,data_set_N):
    df = pd.read_csv(path+"\\original_data\\part-"+str(i).zfill(5)+"-of-00500.csv\\part-"+str(i).zfill(5)+"-of-00500.csv",header=None)  
    for job_id in job_ids:
        # 如果CSV文件没有列名，或者不确定列名，可以使用列的整数位置来筛选  
        # 使用列的整数位置（从0开始）来筛选数据，这里取第3列（索引为2,取的是cpu资源）  
        filtered_data  = df[df.iloc[:, 2] == job_id]  
        # 打印筛选后的数据  
        # 读取现有的new.csv文件，如果它存在的话，并且跳过表头  
        try:  
            existing_data = pd.read_csv(path+"\\"+type_id+str(job_id)+"_info.csv", header=None)  # 假设new.csv也没有列名 
            # 删除缺失值 指定 他的 轴
            existing_data.dropna(axis=1,how='all')
        except FileNotFoundError:  
            existing_data = pd.DataFrame()  # 如果文件不存在，则创建一个空的DataFrame  
        # 将筛选后的数据追加到existing_data中  
        combined_data = pd.concat([existing_data,filtered_data], ignore_index=True)  
        # 将合并后的数据保存到info.csv文件中，不包含行索引，且只在第一次保存时包含表头          
        combined_data.to_csv(path+"\\"+type_id+str(job_id)+"_info.csv", index=False, header=(len(existing_data) == 0))
    print(i,"  finsh!")

# # 假设我们有一个DataFrame df，并且我们想要保存其中的'column_name'列  
# df = pd.read_csv(path+"\\"+type_id+str(job_id)+"_info.csv") 
# # print(df)
# # 将这一列数据保存到txt文件中，分隔符为空格  
# df["9"].to_csv(path+"\\"+type_id+"output_cpu.txt", sep=' ', index=False, header=False)

# # 从txt文件中读取数据  
# s = pd.read_csv(path+"\\"+type_id+"output_cpu.txt", sep=' ', header=None)[0]  
# # 将Series转换为numpy数组  
# array = np.array(s)   
# # print(array) 
# show_cpu(array)
