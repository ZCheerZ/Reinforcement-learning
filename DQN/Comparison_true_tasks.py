import numpy as np
import random
import torch
from collections import defaultdict
import matplotlib.pyplot as plt  # 用于绘图
from model_definition import CloudEnv, DQNAgent, NUM_TASK_TYPES, NUM_VMS_PER_TYPE, NUM_PM


def load_agent_from_file(policy_net_path="DQN/model/policy_net(2423).pth"):
    state_dim = 1 + sum(NUM_VMS_PER_TYPE)  # 状态维度
    action_dim = max(NUM_VMS_PER_TYPE)  # 动作维度
    agent = DQNAgent(state_dim, action_dim)
    agent.policy_net.load_state_dict(torch.load(policy_net_path,weights_only=True))
    # agent.policy_net.eval()  # 设置为评估模式（可选）
    print("模型已从",policy_net_path,"加载")
    return agent

def get_task_sequence(episodes=10, max_tasks=5):
    """
    生成一批任务序列（每一周期的任务类型序列）
    :param max_tasks: 一个时隙内有多少个任务数
    :param episodes: 周期的最大步数 也就是一个周期有多少个时隙
    :return: all_task_types
    """
    all_task_types = []
    for _ in range(episodes):
        task_types = [random.randint(0, NUM_TASK_TYPES-1) for _ in range(max_tasks)]
        all_task_types.append(task_types)
    return all_task_types

def get_task_sequence_from_file(file_path, episodes=10, max_tasks=5):
    """
    从文件中读取任务序列（每一周期的任务类型序列）
    :param file_path: 文件路径
    :param episodes: 周期的最大步数 也就是一个周期有多少个时隙
    :param max_tasks: 一个时隙内有多少个任务数
    :return: all_task_types
    """
    all_task_types = []
    with open(file_path, 'r') as f:
        for line in f:
            task_types = list(map(int, line.strip().split()))
            if len(task_types) > max_tasks:
                task_types = task_types[:max_tasks]
            all_task_types.append(task_types)
            if len(all_task_types) >= episodes:
                break
    return all_task_types


def evaluate_with_true_tasks(agent, episodes=100):
    """
    用同一批任务序列分别评估Q-learning、随机分配和轮询分配的负载均衡效果
    :param agent: 已训练好的 QLearningAgent
    :param episodes: 评估轮数
    :return: (vm_var_q, entity_var_q, vm_var_random, entity_var_random, vm_var_rr, entity_var_rr)
    """
    # 1. 生成所有任务序列（每一轮的任务类型序列）
    all_task_types = get_task_sequence(episodes=episodes, max_tasks=5)
    print("生成的第一轮任务序列：", all_task_types)
    # 2. DQN评估
    env_q = CloudEnv()
    vm_var_q, entity_var_q,vm__utilization_q,pm__utilization_q =  [[] for _ in range(NUM_TASK_TYPES)], [], [[] for _ in range(sum(NUM_VMS_PER_TYPE))], [[] for _ in range(NUM_PM)]
    env_q.reset()
    for ep in range(episodes):
        vm_load, entity_loads, overload_flag, overload_vms,vm_utilization, pm_utilization  = env_q.step_batch(all_task_types[ep], agent, choose_function="DQN")
        # 记录每种类型虚拟机的负载方差
        for task_type in range(NUM_TASK_TYPES):
            start = env_q.prefix_NUM_VMS_PER_TYPE[task_type]
            end = env_q.prefix_NUM_VMS_PER_TYPE[task_type+1]
            vm_var_q[task_type].append(np.var(vm_load[start:end]))
        # print("env_q.vm_load:", env_q.vm_load)
        # 记录实体机方差
        entity_var_q.append(np.var(entity_loads))
        if(overload_flag): 
            print(f"DQN Episode {ep} has overload VMs: {overload_vms}")
        for i in range(len(vm_utilization)):
            vm__utilization_q[i].append(vm_utilization[i])
        for i in range(len(pm_utilization)):
            pm__utilization_q[i].append(pm_utilization[i])

    # 3. 随机分配评估（用同样的任务序列）
    env_r = CloudEnv()
    vm_var_random, entity_var_random,vm__utilization_random,pm__utilization_random =  [[] for _ in range(NUM_TASK_TYPES)], [], [[] for _ in range(sum(NUM_VMS_PER_TYPE))], [[] for _ in range(NUM_PM)]
    env_r.reset()
    for ep in range(episodes):
        vm_load, entity_loads, overload_flag, overload_vms,vm_utilization, pm_utilization  = env_r.step_batch(all_task_types[ep], agent= None, choose_function="Random")
        # 记录每种类型虚拟机的负载方差
        for task_type in range(NUM_TASK_TYPES):
            start = env_r.prefix_NUM_VMS_PER_TYPE[task_type]
            end = env_r.prefix_NUM_VMS_PER_TYPE[task_type+1]
            vm_var_random[task_type].append(np.var(vm_load[start:end]))
        # print("env_r.vm_load:", env_r.vm_load)
        # 记录实体机方差   
        entity_var_random.append(np.var(entity_loads))     
        if(overload_flag): 
            print(f"Random Episode {ep} has overload VMs: {overload_vms}")
        for i in range(len(vm_utilization)):
            vm__utilization_random[i].append(vm_utilization[i])
        for i in range(len(pm_utilization)):
            pm__utilization_random[i].append(pm_utilization[i])

    # 4. 轮询分配评估（用同样的任务序列）
    env_rr = CloudEnv()
    vm_var_rr, entity_var_rr ,vm__utilization_rr,pm__utilization_rr =  [[] for _ in range(NUM_TASK_TYPES)], [],[[] for _ in range(sum(NUM_VMS_PER_TYPE))], [[] for _ in range(NUM_PM)]
    env_rr.reset()
    for ep in range(episodes):
        vm_load, entity_loads, overload_flag, overload_vms,vm_avg_utilization, pm_avg_utilization  = env_rr.step_batch(all_task_types[ep], agent = None, choose_function="RR")
        # 记录每种类型虚拟机的负载方差
        for task_type in range(NUM_TASK_TYPES):
            start = env_rr.prefix_NUM_VMS_PER_TYPE[task_type]
            end = env_rr.prefix_NUM_VMS_PER_TYPE[task_type+1]
            vm_var_rr[task_type].append(np.var(vm_load[start:end]))
        # print("env_rr.vm_load:", env_rr.vm_load)
        # 记录实体机方差
        entity_var_rr.append(np.var(entity_loads))
        if(overload_flag): 
            print(f"RR Episode {ep} has overload VMs: {overload_vms}")
        for i in range(len(vm_avg_utilization)):
            vm__utilization_rr[i].append(vm_avg_utilization[i])
        for i in range(len(pm_avg_utilization)):
            pm__utilization_rr[i].append(pm_avg_utilization[i])

    return vm_var_q, entity_var_q,vm__utilization_q,pm__utilization_q,\
            vm_var_random, entity_var_random,vm__utilization_random,pm__utilization_random,\
            vm_var_rr, entity_var_rr,vm__utilization_rr,pm__utilization_rr

def evaluate():

    agent = load_agent_from_file()

    # 用真实任务序列评估
    vm_var_q, entity_var_q,vm__utilization_q,pm__utilization_q,\
            vm_var_random, entity_var_random,vm__utilization_random,pm__utilization_random,\
            vm_var_rr, entity_var_rr,vm__utilization_rr,pm__utilization_rr = evaluate_with_true_tasks(agent, episodes=10)

    # vm_var_q, entity_var_q, vm_var_rr, entity_var_rr= evaluate_with_true_tasks(agent, episodes=100)


    # 计算每种类型虚拟机方差的均值
    avg_vm_var_q = [np.mean(vm_var_q[i]) for i in range(NUM_TASK_TYPES)]
    avg_vm_var_random = [np.mean(vm_var_random[i]) for i in range(NUM_TASK_TYPES)]
    avg_vm_var_rr = [np.mean(vm_var_rr[i]) for i in range(NUM_TASK_TYPES)]
    avg_entity_var_q = np.mean(entity_var_q)
    avg_entity_var_random = np.mean(entity_var_random)
    avg_entity_var_rr = np.mean(entity_var_rr)
    avg_vm_utilization_q = np.mean([np.mean(vm__utilization_q[i]) for i in range(len(vm__utilization_q))])
    avg_vm_utilization_random = np.mean([np.mean(vm__utilization_random[i]) for i in range(len(vm__utilization_random))])
    avg_vm_utilization_rr = np.mean([np.mean(vm__utilization_rr[i]) for i in range(len(vm__utilization_rr))])
    avg_pm_utilization_q = np.mean([np.mean(pm__utilization_q[i]) for i in range(len(pm__utilization_q))])
    avg_pm_utilization_random = np.mean([np.mean(pm__utilization_random[i]) for i in range(len(pm__utilization_random))])
    avg_pm_utilization_rr = np.mean([np.mean(pm__utilization_rr[i]) for i in range(len(pm__utilization_rr))])   
    # print("--------------------------------------")
    print("各算法每种应用类型虚拟机负载方差均值：")
    for i in range(NUM_TASK_TYPES):
        print(f"Type {i} - DQN: {avg_vm_var_q[i]:.4f}, Random: {avg_vm_var_random[i]:.4f}, RR: {avg_vm_var_rr[i]:.4f}")
    print("--------------------------------------")
    print("各算法实体机负载方差均值：")
    print(f"DQN:       {avg_entity_var_q:.4f}")
    print(f"Random:    {avg_entity_var_random:.4f}")
    print(f"RR:        {avg_entity_var_rr:.4f}")
    print("--------------------------------------")
    print("多目标加权后各算法对比值：")
    print(f"DQN:       {0.4*np.mean(vm_var_q) + 0.5*avg_entity_var_q:.4f}")
    print(f"Random:    {0.4*np.mean(avg_vm_var_random) + 0.5*avg_entity_var_random:.4f}")
    print(f"RR:        {0.4*np.mean(avg_vm_var_rr) + 0.5*avg_entity_var_rr:.4f}")
    print("--------------------------------------")
    print("各算法每种应用类型虚拟机平均利用率：")
    for i in range(sum(NUM_VMS_PER_TYPE)):
        print(f"VM {i} - DQN: {np.mean(vm__utilization_q[i]):.4f}, Random: {np.mean(vm__utilization_random[i]):.4f}, RR: {np.mean(vm__utilization_rr[i]):.4f}")
    print("--------------------------------------")
    print("各算法每种应用类型实体机平均利用率：")
    for i in range(NUM_PM):
        print(f"PM {i} - DQN: {np.mean(pm__utilization_q[i]):.4f}, Random: {np.mean(pm__utilization_random[i]):.4f}, RR: {np.mean(pm__utilization_rr[i]):.4f}")


    # 绘图
    plt.figure(figsize=(15,8))
    # 虚拟机负载方差（每种类型单独画线）
    for i in range(NUM_TASK_TYPES):
        plt.subplot(3,2,i+1)
        plt.plot(vm_var_q[i], label=f"DQN Type {i}")
        plt.plot(vm_var_random[i], '--', label=f"Random Type {i}")
        plt.plot(vm_var_rr[i], ':', label=f"RR Type {i}")
        plt.title("VM load variance (per type)")
        plt.xlabel("Episode")
        plt.ylabel("Variance")
        plt.legend()
        plt.grid()

    # 实体机负载方差
    plt.subplot(3,2,5)
    plt.plot(entity_var_q, label="DQN")
    plt.plot(entity_var_random, label="Random")
    plt.plot(entity_var_rr, label="RR")
    plt.title("PM load variance")
    plt.xlabel("Episode")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid()


    plt.tight_layout()
    plt.show()

evaluate()
