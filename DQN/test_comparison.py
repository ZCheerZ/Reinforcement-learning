import numpy as np
import random
import torch
import matplotlib.pyplot as plt  # 用于绘图
from nbformat.sign import algorithms

import model_definition
from model_definition import CloudEnv, DQNAgent, NUM_TASK_TYPES, NUM_VMS_PER_TYPE, NUM_PM
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
# plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号



def load_agent_from_file(policy_net_path="DQN/model/policy_net(xxxxxxx).pth"):
    state_dim = 1 + sum(model_definition.NUM_VMS_PER_TYPE)  # 状态维度
    action_dim = max(model_definition.NUM_VMS_PER_TYPE)  # 动作维度
    agent = DQNAgent(state_dim, action_dim)
    agent.policy_net.load_state_dict(torch.load(policy_net_path, weights_only=True))
    # agent.policy_net.eval()  # 设置为评估模式（可选）
    print("模型已从", policy_net_path, "加载")
    return agent

def get_task_sequence(episodes, max_tasks):
    """
    生成一批任务序列（每一周期的任务类型序列）
    :param max_tasks: 一个时隙内有多少个任务数
    :param episodes: 周期的最大步数 也就是一个周期有多少个时隙
    :return: all_task_types
    """
    all_task_types = []
    total_nums = 0
    with open("DQN/task_sequence.txt", "w") as f:
        for _ in range(episodes):
            task_types = [random.randint(0, model_definition.NUM_TASK_TYPES - 1) for _ in
                          range(random.randint(1, max_tasks))]
            total_nums += len(task_types)
            f.write(" ".join(map(str, task_types)) + "\n")
            all_task_types.append(task_types)
    # 将all_task_types 写到txt文件中
    return all_task_types, total_nums

def get_task_sequence_from_file(file_path):
    """
    从文件中读取任务序列（每一周期的任务类型序列）
    :param file_path: 文件路径
    :param episodes: 周期的最大步数 也就是一个周期有多少个时隙
    :param max_tasks: 一个时隙内有多少个任务数
    :return: all_task_types
    """
    all_task_types = []
    total_nums = 0
    T = 0
    with open(file_path, 'r') as f:
        for line in f:
            task_types = list(map(int, line.strip().split()))
            total_nums += len(task_types)
            T += 1
            all_task_types.append(task_types)
    # print("从文件中读取的任务序列：", all_task_types)
    return all_task_types, total_nums, T

# 生成符合正态分布的任务数量（处理负值问题）
def generate_normal_tasks(mean, std, size):
    tasks = np.random.normal(loc=mean, scale=std, size=size)
    # 将负数转换为0（任务数量不能为负）
    tasks[tasks < 0] = 0
    # 四舍五入到最接近的整数
    tasks = np.round(tasks).astype(int)
    return tasks

def get_task_sequence_by_type(episodes, means_tasks, dist_type):
    """
    生成任务序列，支持泊松分布、均匀分布、正态分布
    :param episodes: 周期数（时隙数）
    :param max_tasks: 均匀分布时每个时隙任务总数
    :param dist_type: 'uniform'/'poisson'/'normal' /'random'
    :param lam: 泊松分布参数
    :param mu: 正态分布均值
    :param sigma: 正态分布标准差
    :return: all_task_types, total_nums
    """
    sigma = int(0.25 * means_tasks)  # 正态分布的标准差
    burst_scale = 1.5
    burst_prob = 0.00  # 突发概率
    # np.random.seed(42)  # 设置随机种子确保结果可复现
    all_task_types = []
    match dist_type:
        case 'uniform':
            tasks_nums = [means_tasks for _ in range(episodes)]
        case 'poisson':
            tasks_nums = np.random.poisson(lam=means_tasks, size=episodes)
        case 'normal':
            tasks_nums = generate_normal_tasks(means_tasks, sigma, episodes)
        case 'random':
            tasks_nums = [random.randint(1, means_tasks) for _ in range(episodes)]
    total_nums = sum(tasks_nums)
    print(tasks_nums)

    for task_num in tasks_nums:
        # 均匀分配任务类型
        task_num = int(task_num)
        # 模拟突发值
        if random.random() < burst_prob:
            task_num = int(task_num * burst_scale)
        if task_num == 0:
            task_types = []
        else:
            base_num = task_num // model_definition.NUM_TASK_TYPES
            remainder = task_num % model_definition.NUM_TASK_TYPES
            task_counts = [base_num] * model_definition.NUM_TASK_TYPES
            for i in range(remainder):
                task_counts[i] += 1
            task_types = []
            for t, count in enumerate(task_counts):
                task_types += [t] * count
            random.shuffle(task_types)  # 可选，打乱时隙内任务顺序
        all_task_types.append(task_types)
    return all_task_types, total_nums

def evaluate_performance(all_task_types, choose_function, T=100, agent=None):
    """
    用同一批任务序列分别评估Q-learning、随机分配和轮询分配的负载均衡效果
    :param all_task_types: 任务类型序列
    :param choose_function: 选择函数（DQN、Random、RR）
    :param T: 评估轮数
    :param agent: 已训练好的 QLearningAgent
    :return: (vm_var, pm_var)
    """
    env = CloudEnv()
    env.reset()
    total_overload_nums = 0  # 记录超载的虚拟机数量
    print(f"rr_pointer: {env.rr_pointer}")
    vm_vars, pm_vars = [[] for _ in range(model_definition.NUM_TASK_TYPES)], []
    vm_utilizations, pm_utilizations = [[] for _ in range(sum(model_definition.NUM_VMS_PER_TYPE))], [[] for _ in range(
        model_definition.NUM_PM)]
    violation_rate_per_step = []
    for t in range(T):
        overload_flag, overload_nums = env.step_batch(all_task_types[t], agent, choose_function=choose_function)
        # 记录每种类型虚拟机的负载方差
        pm_loads, pm_utilization, pm_var = env.get_pm_info()  # 更新实体机负载信息
        vm_loads, vm_utilization, vm_var = env.get_vm_info()  # 更新虚拟机负载信息
        for task_type in range(model_definition.NUM_TASK_TYPES):
            vm_vars[task_type].append(vm_var[task_type])
        pm_vars.append(pm_var)
        for pm_id in range(model_definition.NUM_PM):
            pm_utilizations[pm_id].append(pm_utilization[pm_id])
        for i in range(len(vm_utilization)):
            vm_utilizations[i].append(vm_utilization[i])
        total_overload_nums += overload_nums
        if (pm_var > 8000):
            print(f"Episode {t} has overload PMs: {pm_loads}")
        task_num = len(all_task_types[t])
        rate = overload_nums / task_num if task_num > 0 else 0
        violation_rate_per_step.append(rate)
    print("违规数：", total_overload_nums)
    return vm_vars, pm_vars, vm_utilizations, pm_utilizations, total_overload_nums, violation_rate_per_step

def plot_show():
    """
    比较dp+DQN、padding+轮询分配的负载均衡效果
    :return: None
    """
    T = 500
    algorithms = "Static Provisioning + MinMin Scheduling"
    # 重置环境参数 7 2, 2, 4, 4, 2, 3, 3
    # E2ESLA
    # model_definition.env_params_reset(num_pm=3,  num_task_types = 4, num_vms_per_type=[2, 2, 3, 4])
    # Static Provisioning + MinMin Scheduling
    model_definition.env_params_reset(num_pm=5, num_task_types=4, num_vms_per_type=[4, 3, 4, 6])
    # RPRP + Round Robin Scheduling
    # model_definition.env_params_reset(num_pm=4,num_task_types=4, num_vms_per_type=[3, 2, 2, 4])
    # 1. 生成所有任务序列（每一轮的任务类型序列）
    # 2234 means_tasks = 55 244 means_tasks = 70  2234243 means_tasks = 70(good)
    all_task_types, total_nums = get_task_sequence_by_type(episodes=T, means_tasks=50, dist_type='normal')
    # all_task_types, total_nums, T = get_task_sequence_from_file("DQN/task_sequence.txt")
    # print("生成的第一轮任务序列：", all_task_types)
    num_vms_str = ''.join(str(x) for x in model_definition.NUM_VMS_PER_TYPE)
    print("NUM_VMS_PER_TYPE字符串形式:", num_vms_str)
    # 模型地址
    file_path = "DQN/model/policy_net(" + num_vms_str + ")best.pth"
    # agent = load_agent_from_file(file_path) # None
    agent = None;
    vm_var, entity_var, vm_utilization, pm_utilization, overload_nums ,violation_rate_per_step = evaluate_performance(
        all_task_types, choose_function="MinMin", T=T, agent= agent)
    # 计算每种类型虚拟机方差的均值
    avg_vm_var_rr = [np.mean(vm_var[i]) for i in range(model_definition.NUM_TASK_TYPES)]
    avg_entity_var_rr = np.mean(entity_var)
    # print("--------------------------------------")
    print("当前算法每种应用类型虚拟机负载方差均值：")
    for i in range(model_definition.NUM_TASK_TYPES):
        print(f"Type {i} - {algorithms}: {avg_vm_var_rr[i]:.4f}")
    print("--------------------------------------")
    print("当前算法实体机负载方差均值：")
    print(f"{algorithms}:        {avg_entity_var_rr:.4f}")
    print("--------------------------------------")
    print("多目标加权后各算法对比值：")
    print(f"{algorithms}:        {0.3 * np.mean(avg_vm_var_rr) + 0.7 * avg_entity_var_rr:.4f}")
    print("--------------------------------------")
    print("当前算法每种应用类型虚拟机平均利用率：")
    for i in range(sum(model_definition.NUM_VMS_PER_TYPE)):
        print(f"VM {i} - {algorithms}: {np.mean(vm_utilization[i]):.4f}")
    print("--------------------------------------")
    print("当前算法每种应用类型实体机平均利用率：")
    for i in range(len(pm_utilization)):
        print(f"PM {i} - {algorithms}: {np.mean(pm_utilization[i]):.4f}")
    print("--------------------------------------")
    print("当前算法违规率：")
    print(f"{algorithms}:        {overload_nums / total_nums:.4f}")

    # 绘图  怎么画出好看的图像 画一个实体机资源利用率对比的图像
    plt.figure(figsize=(15, 8))
    # # 虚拟机负载方差（每种类型单独画线）
    # for i in range(model_definition.NUM_TASK_TYPES):
    #     plt.subplot(4,2,i+1)
    #     plt.plot(vm_var[i], '--', label=f"{algorithms} Type {i}")
    #     plt.title("对应每个任务类型虚拟机负载方差图")
    #     plt.xlabel("步长")
    #     plt.ylabel("方差")
    #     plt.legend()
    #     plt.grid()
    # # 实体机负载方差
    # plt.subplot(4,2,8)
    plt.subplot(3, 1, 1)
    plt.plot(entity_var, label=f"{algorithms}") #预测驱动的动态资源配置算法+DQN调度算法
    plt.title(f"load variance of physical machines of {algorithms} algorithm")
    plt.xlabel("episode")
    plt.ylabel("load variance of physical machines")

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.subplot(3, 1, 2)
    for pm_id, pm_util in enumerate(pm_utilization):
        plt.plot(pm_util, label=f"PM{pm_id+1} ,average resource utilization rate:{np.mean(pm_util)*100:.2f}%")
    plt.title(f"resource utilization rate of {algorithms} algorithm")
    plt.xlabel("episode")
    plt.ylabel("resource utilization rate")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.subplot(3, 1, 3)
    plt.plot(violation_rate_per_step, color='red', label="the violation rate per episode")
    plt.title(f"violation rate of {algorithms} algorithm")
    plt.xlabel("episode")
    plt.ylabel("violation rate")
    plt.gcf().text(0.8, 0.23, f"total violation: {overload_nums / total_nums * 100:.4f}%", fontsize=14, color='red')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()



plot_show()
