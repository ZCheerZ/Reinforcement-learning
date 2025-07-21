import numpy as np
import matplotlib.pyplot as plt  # 用于绘图
import torch
import random
from model_definition import CloudEnv, DQNAgent
from model_definition import NUM_TASK_TYPES, NUM_VMS_PER_TYPE, VMS_PER_TYPE, NUM_PM



def load_agent_from_file(policy_net_path="DQN/model/policy_net(233).pth"):
    state_dim = 1 + sum(NUM_VMS_PER_TYPE)  # 状态维度
    action_dim = max(NUM_VMS_PER_TYPE)  # 动作维度
    agent = DQNAgent(state_dim, action_dim)
    agent.policy_net.load_state_dict(torch.load(policy_net_path,weights_only=True))
    # agent.policy_net.eval()  # 设置为评估模式（可选）
    print("模型已从 DQN/policy_net.pth 加载")
    return agent

def generate_random_state():
    """
    基于NUM_VMS_PER_TYPE随机生成一个状态
    返回格式: (task_type, vm_level_0, vm_level_1, ..., vm_level_n)
    """
    # 随机生成任务类型
    task_type = random.randint(0, NUM_TASK_TYPES - 1)
    # 随机生成每台虚拟机的等级
    state = [task_type]
    for i in range(len(VMS_PER_TYPE)):
        # 随机生成虚拟机负载等级（1-50级，对应0-100%负载）
        vm_level = random.randint(1, 20)
        state.append(vm_level)
    
    return tuple(state)

def evaluate_load_balance():

    agent = load_agent_from_file()
    test_states = [
        generate_random_state(),
        generate_random_state(),
        generate_random_state(),
    ]
    env = CloudEnv()
    for idx, test_state in enumerate(test_states):
        print(f"\n测试状态{idx+1}: {test_state}")
        test_state = np.array(test_state, dtype=np.int32)
        q_values = agent.policy_net(torch.FloatTensor(test_state).unsqueeze(0)).detach().numpy()[0]
        print(f"Q值分布: {q_values}")
        best_action = agent.choose_action_multi(test_state, 0)
        print(f"执行应用类型{test_state[0]}虚拟机中最大Q值对应动作: 选择{best_action}号虚拟机")
        vm_load = list(test_state[1:1 + env.prefix_NUM_VMS_PER_TYPE[-1]])
        vm_load[env.prefix_NUM_VMS_PER_TYPE[test_state[0]] + best_action] += 1
        print(f"选择完对应动作后的虚拟机负载: {vm_load}")
        # 分别输出每个任务类型下的虚拟机负载
        for t in range(NUM_TASK_TYPES):
            start = env.prefix_NUM_VMS_PER_TYPE[t]
            end = env.prefix_NUM_VMS_PER_TYPE[t+1]
            print(f"应用类型{t}的虚拟机负载: {vm_load[start:end]}")

        # 计算实体机负载
        entity_loads = []
        for e in range(NUM_PM):
            load = sum(
                vm_load[i]
                for i in range(sum(NUM_VMS_PER_TYPE))
                if env.vm_to_pm[i] == e
            )
            entity_loads.append(load)
        print(f"实体机负载: {entity_loads}")

evaluate_load_balance()