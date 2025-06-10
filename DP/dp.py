import os
import numpy as np

# 读取指定文件夹下的所有txt文件（每个文件都是一段周期所有时刻下的该应用类型任务的任务数），返回一个二维列表  但可能是没有进行降序排列的！
def read_data_from_txt(folder_path, num_types):
    data = []
    for i in range(num_types-1, -1, -1):  # 从类型3到类型0
        file_path = os.path.join(folder_path, f"type{i}.txt")
        with open(file_path, "r") as f:
            # 读取所有行，去除空行和换行符，并转为 float
            arr = [float(line.strip()) for line in f if line.strip()]
            # arr = [(line.strip()) for line in f if line.strip()]
            data.append(arr)
    return data

def main():
    folder = "DP/data"  # 假设数据文件夹名为data
    num_types = 3       # 文件夹内有多少种应用类型的需求数据
    data = read_data_from_txt(folder, num_types)
    print(data)
    # 自拟简化数据：3种类型，每种3个需求值（降序排列）
    # data = [
    #     [6.0, 5.0, 2.0, 1.0],   # 类型0：需求值降序，如5（最大）、3、1（最小）
    #     [4.0, 2.0, 1.0],   # 类型1
    #     [3.0, 1.0, 1.0]    # 类型2
    # ]
    # num_types = len(data)         # 3种类型
    max_len_per_type = max(len(arr) for arr in data)  # 由于数组长度不统一要对齐数组 取所有应用类型中的最大长度（k=0~3）
    total_max_unsatisfied = num_types * max_len_per_type  # 3*3=9  取最大长度* 类型数就是最大不满足数

    # 扩展数据到二维数组a，行数为num_types，列数为total_max_unsatisfied+1 与data内容一致，但保证了数组长度平齐
    # 不是data前缀和的原因是因为预测得到的就是当前时刻的任务“处理”数，已经包含了任务处理时长的因素，如果是任务到达数那就需要累加了
    a = [[float('0')] * (total_max_unsatisfied + 1) for _ in range(num_types)]
    for i in range(num_types):
        for j in range(max_len_per_type + 1):
            if j < len(data[i]):
                a[i][j] = data[i][j]

    # 初始化DP数组：f[i][j]表示前i+1种类型，共不满足j个需求的最小总需求值 默认值inf 无穷大
    dp = [[float('inf')] * (total_max_unsatisfied + 1) for _ in range(num_types)]
    # 初始化路径数组：path[i][j]记录前i+1种类型，不满足j个需求时的上一个选择的不满足需求数是多少
    path = [[-1] * (total_max_unsatisfied + 1) for _ in range(num_types)]

    # 初始化第一种类型（i=0）
    for k in range(total_max_unsatisfied + 1):
        if k < len(data[0]):
            dp[0][k] = a[0][k] # 仅第一种类型，不满足k个需求的总需求值
            path[0][k] = 0  # 记录选择k个不满足

        else:
            dp[0][k] = float('0')
    
    # 动态规划
    for i in range(1, num_types):  # 处理第i种类型（从1到2，0已处理）
        for j in range(total_max_unsatisfied + 1):
            # 枚举当前类型不满足k个需求（k最多为当前类型最大不满足数，且不超过j）
            for k in range(0, j+1):
                prev_j = j - k  # 前i-1种类型需要不满足prev_j个需求
                # 状态转移：当前总需求值 = 前i-1种的需求值 + 当前类型不满足k个的需求值
                if dp[i][j] > dp[i-1][prev_j] + a[i][k]:
                    dp[i][j] = dp[i-1][prev_j] + a[i][k]
                    path[i][j] = prev_j  # 记录当前选择k个不满足

    
    # 打印结果（以表格形式展示前3种类型的DP数组）
    print("动态规划结果f[i][j]（前i+1种类型，违规数为j个需求的最小总需求值）:")
    for i in range(num_types):
        print(f"类型{i+1}（前{i+1}种类型）:")
        for j in range(total_max_unsatisfied + 1):
            print(f"j={j}: {dp[i][j]}", end=" | ")
        print("\n" + "-" * 50)

    t = 2  # 假设一开始不满足3个需求
    now_sla = t
    for i in range(num_types-1,-1,-1):
        print(f"应用类型{i}允许违规数: {now_sla-path[i][t]}, 执行任务数为{a[i][now_sla-path[i][t]]}", end="\n")
        now_sla = path[i][t]
        t = path[i][t]
        
        
if __name__ == "__main__":
    main()

#  这个dp过程发现一个问题，在可能允许较多违规的情况下，它可能会把某种应用类型的任务的高需求值通通滤过去，
#  使得总需求值过低，导致可能某个应用类型的任务都不想去满足，使得其虚拟机很少，是不是应该设置个阈值

