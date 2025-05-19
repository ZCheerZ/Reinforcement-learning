import numpy as np
import matplotlib.pyplot as plt  # 用于绘图
import pickle
from model_definition import CloudEnv, QLearningAgent
from collections import defaultdict


# 环境参数
NUM_TASK_TYPES = 3
NUM_VMS_PER_TYPE = 3  # 每种任务类型有3台虚拟机

def load_agent_from_file(q_table_path="Q-leaning/q_table.pkl"):
    agent = QLearningAgent()
    with open(q_table_path, "rb") as f:
        loaded_q_table = pickle.load(f)
    # agent.q_table = loaded_q_table
    agent.q_table = defaultdict(lambda: np.zeros(NUM_VMS_PER_TYPE), loaded_q_table)
    print("Q表已从文件加载")
    return agent

def evaluate_load_balance():

    agent = load_agent_from_file()

    test_states = [
        (0, 1, 2, 1, 1, 1, 1, 2, 1, 1),
        (0, 1, 1, 1, 2, 1, 1, 1,2, 1),
        (1, 1, 2, 2, 1, 1, 1, 2, 1, 1)
    ]
    env = CloudEnv()
    for idx, test_state in enumerate(test_states):
        print(f"\n测试状态{idx+1}: {test_state}")
        q_values = agent.q_table[test_state]
        print(f"Q值分布: {q_values}")
        best_action = np.argmax(q_values)
        print(f"执行应用类型{test_state[0]}虚拟机中最大Q值对应动作: 选择{best_action}号虚拟机")
        vm_load = list(test_state[1:1 + NUM_TASK_TYPES * NUM_VMS_PER_TYPE])
        vm_load[test_state[0] * NUM_VMS_PER_TYPE + best_action] += 1
        print(f"选择完对应动作后的虚拟机负载: {vm_load}")
        # 分别输出每个任务类型下的虚拟机负载
        for t in range(NUM_TASK_TYPES):
            start = t * NUM_VMS_PER_TYPE
            end = start + NUM_VMS_PER_TYPE
            print(f"应用类型{t}的虚拟机负载: {vm_load[start:end]}")

        # 计算实体机负载
        entity_loads = []
        for e in range(NUM_VMS_PER_TYPE):
            load = sum(
                vm_load[i]
                for i in range(NUM_TASK_TYPES * NUM_VMS_PER_TYPE)
                if env.vm_to_entity[i] == e
            )
            entity_loads.append(load)
        print(f"实体机负载: {entity_loads}")

evaluate_load_balance()