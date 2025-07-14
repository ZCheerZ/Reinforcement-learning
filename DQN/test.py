import numpy as np
import matplotlib.pyplot as plt  # 用于绘图
import torch
from model_definition import CloudEnv, DQNAgent



# 环境参数
NUM_TASK_TYPES = 3  # 应用类型数量
NUM_VMS_PER_TYPE = [2,4,3]  # 每种应用类型有3台虚拟机
NUM_PM = 3  # 实体机数量



def load_agent_from_file(policy_net_path="DQN/policy_net(243).pth"):
    state_dim = 1 + sum(NUM_VMS_PER_TYPE)  # 状态维度
    action_dim = max(NUM_VMS_PER_TYPE)  # 动作维度
    agent = DQNAgent(state_dim, action_dim)
    agent.policy_net.load_state_dict(torch.load(policy_net_path,weights_only=True))
    # agent.policy_net.eval()  # 设置为评估模式（可选）
    print("模型已从 DQN/policy_net.pth 加载")
    return agent

def evaluate_load_balance():

    agent = load_agent_from_file()

    test_states = [
        (0, 1, 3, 1, 1, 1, 1, 2, 1, 1),
        (0, 2, 1, 2, 1, 2, 2, 2, 2, 1),
        (1, 1, 2, 2, 1, 1, 1, 2, 1, 1)
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