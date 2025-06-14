import heapq
from collections import defaultdict, deque

def calculate_task_counts1(task_sequence, task_durations):
    """
    计算周期内每个时刻正在处理的任务数。
    
    参数:
        task_sequence: 任务序列，每个元素为任务类型（整数或字符串）
        task_durations: 字典，key为任务类型，value为执行时间（整数，单位：时隙）
        
    返回:
        counts: 列表，每个时刻的任务数（长度等于任务序列长度）
    """
    T = len(task_sequence)  # 周期总时长
    # 初始化差分数组（长度为T+1，多出一位用于处理结束时刻）
    diff = [0] * (T + 1)
    
    # 遍历每个时刻的任务
    for t, task_type in enumerate(task_sequence):
        d = task_durations[task_type]  # 获取当前任务的执行时间
        # 在开始时刻t标记+1
        diff[t] += 1
        # 在结束时刻t + d标记-1（若在周期内）
        end_time = t + d
        if end_time < T + 1:  # 确保不越界
            diff[end_time] -= 1
    
    # 通过差分数组计算每个时刻的任务数
    counts = []
    current_tasks = 0
    for i in range(T):
        current_tasks += diff[i]
        counts.append(current_tasks)
    
    return counts

def calculate_task_counts2(task_sequence, task_durations):
    """
    使用任务队列计算周期内每个时刻正在处理的任务数
    
    参数:
        task_sequence: 任务序列，每个元素为任务类型
        task_durations: 字典，key为任务类型，value为执行时间
        
    返回:
        counts: 列表，每个时刻的任务数
    """
    T = len(task_sequence)  # 周期总时长
    # 为每种任务类型创建一个队列（存储结束时间）
    task_queues = defaultdict(deque)
    # 优先队列（最小堆）用于管理所有任务的结束时间
    end_time_heap = []
    
    counts = [0] * T  # 初始化结果列表
    
    for t, task_type in enumerate(task_sequence):
        # 步骤1：移除在当前时刻结束的所有任务
        while end_time_heap and end_time_heap[0] == t:
            # 弹出最早结束的任务
            heapq.heappop(end_time_heap)
        
        # 步骤2：处理新到达的任务
        duration = task_durations[task_type]
        if duration > 0:  # 只处理执行时间大于0的任务
            end_time = t + duration
            # 将任务加入对应类型的队列
            task_queues[task_type].append(end_time)
            # 将结束时间加入优先队列
            heapq.heappush(end_time_heap, end_time)
        
        # 步骤3：记录当前时刻正在处理的任务总数
        counts[t] = len(end_time_heap)
    
    return counts


from collections import deque, defaultdict

def calculate_task_counts_by_type(task_sequence, task_durations):
    """
    计算每个时刻每种任务类型正在处理的任务数
    
    参数:
        task_sequence: 任务序列，每个元素为任务类型
        task_durations: 字典，key为任务类型，value为执行时间
        
    返回:
        counts: 二维数组，维度为[任务类型数 × 序列长度]
        type_order: 任务类型顺序，与counts的行索引对应
    """
    T = len(task_sequence)  # 周期总时长
    
    # 获取所有任务类型（按排序顺序）
    all_types = sorted(task_durations.keys())
    type_index = {type_name: idx for idx, type_name in enumerate(all_types)}
    
    # 为每种任务类型创建队列（存储结束时间）
    queues = {type_name: deque() for type_name in all_types}
    
    # 初始化结果数组（任务类型数 × 序列长度）
    counts = [[0] * T for _ in range(len(all_types))]
    
    # 遍历每个时刻
    for t, task_type in enumerate(task_sequence):
        # 步骤1：处理所有类型的任务结束（在当前时刻t结束的任务）
        for type_name in all_types:
            queue = queues[type_name]
            # 移除所有在当前时刻结束的任务
            while queue and queue[0] == t:
                queue.popleft()
        
        # 步骤2：处理新到达的任务
        duration = task_durations[task_type]
        if duration > 0:  # 只处理执行时间大于0的任务
            end_time = t + duration
            queues[task_type].append(end_time)
        
        # 步骤3：记录当前时刻所有类型的任务数
        for type_name in all_types:
            type_idx = type_index[type_name]
            counts[type_idx][t] = len(queues[type_name])
    
    return counts, all_types


# 示例用法
if __name__ == "__main__":
    task_sequence = [1, 2, 1, 3, 2, 1]
    task_durations = {1: 2, 2: 3, 3: 1}
    
    counts = calculate_task_counts1(task_sequence, task_durations)
    print(counts)  # 输出每个时刻的任务数

    counts = calculate_task_counts2(task_sequence, task_durations)
    print(counts)  # 输出每个时刻的任务数

    all_counts,all_types = calculate_task_counts_by_type(task_sequence, task_durations)
    print(all_counts)  # 输出每种任务类型在每个时刻的任务数
    print(all_types)  # 输出任务类型顺序    