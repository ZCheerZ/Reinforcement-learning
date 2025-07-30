import csv


def process_task_arrivals(file_path, target_task_type, start_timestamp, slot_duration=5000000, num_slots=500):
    """
    处理任务到达数据，统计固定周期内每个时隙的任务到达数

    参数:
        file_path: CSV文件路径
        target_task_type: 目标任务类型ID
        start_timestamp: 周期起始时间戳(微秒)
        slot_duration: 每个时隙的时长(微秒)，默认500
        num_slots: 时隙数量，默认1000

    返回:
        包含每个时隙任务到达数的列表(长度为num_slots)
    """
    # 初始化时隙计数器（1000个时隙，全部设为0）
    slot_counts = [0] * num_slots

    # 计算周期结束时间
    end_timestamp = start_timestamp + (num_slots * slot_duration)

    # 读取CSV文件并处理数据
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                # 确保行有足够列
                if len(row) < 3:
                    continue

                # 解析时间戳和任务类型
                timestamp = int(row[0].strip())
                task_type = int(row[2].strip())

                # 只处理目标类型且在周期内的时间戳
                if task_type == target_task_type and start_timestamp <= timestamp < end_timestamp:
                    # 计算时隙索引
                    time_offset = timestamp - start_timestamp
                    slot_idx = time_offset // slot_duration

                    # 确保索引在有效范围内
                    if 0 <= slot_idx < num_slots:
                        slot_counts[slot_idx] += 1

            except (ValueError, IndexError) as e:
                # 跳过格式错误或缺失数据的行
                continue

    return slot_counts


# 使用示例
if __name__ == "__main__":
    # ===== 用户需要配置的参数 =====
    csv_file = "E:\\zbj-download\\Desktop\\code\\RL_code\\Data_processing\\original_data\\part-00000-of-00500.csv\\part-00000-of-00500.csv"  # CSV文件路径
    task_type = 515042969  # 目标任务类型ID 515042969 6218406404   6251537638
    period_start = 600000000  # 周期起始时间戳(微秒)
    slot_duration=5000000
    num_slots=1000
    # ===========================

    # 可以问老师是否只提取任务序列，不关注是否任务序列都是同一时间开始，只保证时隙一致即可，这样就能提取多个任务的序列了

    # 处理数据并获取结果
    arrival_counts = process_task_arrivals(
        csv_file,
        task_type,
        start_timestamp=period_start,
        slot_duration = slot_duration,
        num_slots = num_slots
    )

    # 输出结果
    print(f"任务到达序列({num_slots}个时隙):")
    print(arrival_counts)
    # 写文件
    with open("Prediction/task.txt", 'w') as file:
        # 遍历数组中的每个元素，并将它们写入文件，每个元素后面跟一个换行符
        for element in arrival_counts:
            file.write(str("{:.1f}".format(element)) + '\n')

    # 可选：统计总任务数和空时隙数
    total_tasks = sum(arrival_counts)
    empty_slots = arrival_counts.count(0)
    print(f"\n统计摘要:")
    print(f"总任务数: {total_tasks}")
    print(f"空时隙数: {empty_slots} ({empty_slots / num_slots * 100}%)")
    print(f"最忙时隙: {max(arrival_counts)} 个任务")