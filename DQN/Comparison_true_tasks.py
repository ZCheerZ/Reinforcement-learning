import numpy as np
import random
import torch
from collections import defaultdict
import matplotlib.pyplot as plt  # 用于绘图
import model_definition
from model_definition import CloudEnv, DQNAgent, NUM_TASK_TYPES, NUM_VMS_PER_TYPE, NUM_PM
# todo : 这里的NUM_TASK_TYPES, NUM_VMS_PER_TYPE, NUM_PM 都是从model_definition.py里导入的 应该统一变为model_definition.xxx over
# todo : 把evaluate_with_true_tasks换了 冗余代码太多  还有模型定义以及这里的获取信息代码也需要更换 over
# todo : 把这个对比算法的画图逻辑顺好，还有就是强化学习过程的选择逻辑在信息更新之后，需要修改一下，把整个对比实验流程捋一下 没有预测的情况下
# todo : 把task序列的转换函数写一下，不知道真实序列是否需要转，那就暂时不转吧 真实序列可能就是1，2，3，1，1
# todo : 训练的时候就先调整任务队列  再进行选择  对比的时候可以把cpu改成核，而不是cpu的百分比  这样虚拟机方差就很小了，同时我的离散状态也很少 哦好像不会少
#          明天先把负载换成1步长  然后调整顺序                           但是这样虚拟机确实方差就很小，应该能体现出DQN的优势，


def load_agent_from_file(policy_net_path="DQN/model/policy_net(2234).pth"):
    state_dim = 1 + sum(model_definition.NUM_VMS_PER_TYPE)  # 状态维度
    action_dim = max(model_definition.NUM_VMS_PER_TYPE)  # 动作维度
    agent = DQNAgent(state_dim, action_dim)
    agent.policy_net.load_state_dict(torch.load(policy_net_path,weights_only=True))
    # agent.policy_net.eval()  # 设置为评估模式（可选）
    print("模型已从",policy_net_path,"加载")
    return agent

def get_task_sequence(episodes=10, max_tasks=20):
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
            task_types = [random.randint(0, model_definition.NUM_TASK_TYPES-1) for _ in range(random.randint(1, max_tasks))]
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

def get_task_sequence_by_type(episodes=10, means_tasks=20, dist_type='random'):
    """
    生成任务序列，支持泊松分布、均匀分布、正态分布
    :param episodes: 周期数（时隙数）
    :param max_tasks: 均匀分布时每个时隙任务总数
    :param dist_type: 'uniform'/'poisson'/'normal'
    :param lam: 泊松分布参数
    :param mu: 正态分布均值
    :param sigma: 正态分布标准差
    :return: all_task_types, total_nums
    """
    sigma = int(0.15 * means_tasks)  # 正态分布的标准差
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
        if task_num == 0:
            task_types = []
        else:
            base_num = task_num // NUM_TASK_TYPES
            remainder = task_num % NUM_TASK_TYPES
            task_counts = [base_num] * NUM_TASK_TYPES
            for i in range(remainder):
                task_counts[i] += 1
            task_types = []
            for t, count in enumerate(task_counts):
                task_types += [t] * count
            random.shuffle(task_types)  # 可选，打乱时隙内任务顺序
        all_task_types.append(task_types)
    return all_task_types, total_nums

def evaluate_performance(all_task_types,choose_function, T=100,agent= None):
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
    vm_vars, pm_vars = [[] for _ in range(model_definition.NUM_TASK_TYPES)], []
    vm_utilizations,pm_utilizations = [[] for _ in range(sum(model_definition.NUM_VMS_PER_TYPE))], [[] for _ in range(model_definition.NUM_PM)]
    for t in range(T):
        overload_flag,overload_nums = env.step_batch(all_task_types[t], agent, choose_function=choose_function)
        # 记录每种类型虚拟机的负载方差
        pm_loads,pm_utilization,pm_var = env.get_pm_info()  # 更新实体机负载信息
        vm_loads,vm_utilization,vm_var = env.get_vm_info()  # 更新虚拟机负载信息
        for task_type in range(model_definition.NUM_TASK_TYPES):
            vm_vars[task_type].append(vm_var[task_type])
        pm_vars.append(pm_var)
        for pm_id in range(model_definition.NUM_PM):
            pm_utilizations[pm_id].append(pm_utilization[pm_id])
        for i in range(len(vm_utilization)):
            vm_utilizations[i].append(vm_utilization[i])
        total_overload_nums += overload_nums
        if(pm_var>8000):
            print(f"Episode {t} has overload PMs: {pm_loads}")
    print("违规数：",total_overload_nums)
    return vm_vars, pm_vars,vm_utilizations, pm_utilizations,total_overload_nums



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
    vm_var_q, entity_var_q,vm__utilization_q,pm__utilization_q =  [[] for _ in range(model_definition.NUM_TASK_TYPES)], [], [[] for _ in range(sum(model_definition.NUM_VMS_PER_TYPE))], [[] for _ in range(model_definition.NUM_PM)]
    env_q.reset()
    for ep in range(episodes):
        vm_load, entity_loads, overload_flag, overload_vms,vm_utilization, pm_utilization  = env_q.step_batch(all_task_types[ep], agent, choose_function="DQN")
        # 记录每种类型虚拟机的负载方差
        for task_type in range(model_definition.NUM_TASK_TYPES):
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
    vm_var_random, entity_var_random,vm__utilization_random,pm__utilization_random =  [[] for _ in range(model_definition.NUM_TASK_TYPES)], [], [[] for _ in range(sum(model_definition.NUM_VMS_PER_TYPE))], [[] for _ in range(model_definition.NUM_PM)]
    env_r.reset()
    for ep in range(episodes):
        vm_load, entity_loads, overload_flag, overload_vms,vm_utilization, pm_utilization  = env_r.step_batch(all_task_types[ep], agent= None, choose_function="Random")
        # 记录每种类型虚拟机的负载方差
        for task_type in range(model_definition.NUM_TASK_TYPES):
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
    vm_var_rr, entity_var_rr ,vm__utilization_rr,pm__utilization_rr =  [[] for _ in range(model_definition.NUM_TASK_TYPES)], [],[[] for _ in range(sum(model_definition.NUM_VMS_PER_TYPE))], [[] for _ in range(model_definition.NUM_PM)]
    env_rr.reset()
    for ep in range(episodes):
        vm_load, entity_loads, overload_flag, overload_vms,vm_avg_utilization, pm_avg_utilization  = env_rr.step_batch(all_task_types[ep], agent = None, choose_function="RR")
        # 记录每种类型虚拟机的负载方差
        for task_type in range(model_definition.NUM_TASK_TYPES):
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
    # vm_var_q, entity_var_q,vm__utilization_q,pm__utilization_q,\
    #         vm_var_random, entity_var_random,vm__utilization_random,pm__utilization_random,\
    #         vm_var_rr, entity_var_rr,vm__utilization_rr,pm__utilization_rr = evaluate_with_true_tasks(agent, episodes=1000)\
    T = 500
    all_task_types = get_task_sequence(episodes=T, max_tasks=50)
    vm_var_q, entity_var_q,vm__utilization_q,pm__utilization_q = evaluate_performance(all_task_types, choose_function="DQN", T=T, agent=agent)
    vm_var_random, entity_var_random,vm__utilization_random,pm__utilization_random = evaluate_performance(all_task_types, choose_function="Random", T=T, agent=None)
    vm_var_rr, entity_var_rr,vm__utilization_rr,pm__utilization_rr = evaluate_performance(all_task_types, choose_function="RR", T=T, agent=None)



    # 计算每种类型虚拟机方差的均值
    avg_vm_var_q = [np.mean(vm_var_q[i]) for i in range(model_definition.NUM_TASK_TYPES)]
    avg_vm_var_random = [np.mean(vm_var_random[i]) for i in range(model_definition.NUM_TASK_TYPES)]
    avg_vm_var_rr = [np.mean(vm_var_rr[i]) for i in range(model_definition.NUM_TASK_TYPES)]
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
    for i in range(model_definition.NUM_TASK_TYPES):
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
    for i in range(sum(model_definition.NUM_VMS_PER_TYPE)):
        print(f"VM {i} - DQN: {np.mean(vm__utilization_q[i]):.4f}, Random: {np.mean(vm__utilization_random[i]):.4f}, RR: {np.mean(vm__utilization_rr[i]):.4f}")
    print("--------------------------------------")
    print("各算法每种应用类型实体机平均利用率：")
    for i in range(model_definition.NUM_PM):
        print(f"PM {i} - DQN: {np.mean(pm__utilization_q[i]):.4f}, Random: {np.mean(pm__utilization_random[i]):.4f}, RR: {np.mean(pm__utilization_rr[i]):.4f}")


    # 绘图
    plt.figure(figsize=(15,8))
    # 虚拟机负载方差（每种类型单独画线）
    for i in range(model_definition.NUM_TASK_TYPES):
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


def comparison_():
    """
    比较dp+DQN、padding+轮询分配的负载均衡效果
    :return: None
    """
    T = 1000
    # 1. 生成所有任务序列（每一轮的任务类型序列）
    # all_task_types,total_nums = get_task_sequence(episodes=T, max_tasks=140)
    # 2234 means_tasks = 55 244 means_tasks = 70
    all_task_types,total_nums = get_task_sequence_by_type(episodes=T, means_tasks=10, dist_type='poisson')

    # all_task_types, total_nums, T = get_task_sequence_from_file("DQN/task_sequence.txt")
    # print("生成的第一轮任务序列：", all_task_types)
    # 2. DQN评估
    num_vms_str = ''.join(str(x) for x in NUM_VMS_PER_TYPE)
    print("NUM_VMS_PER_TYPE字符串形式:", num_vms_str)
    # 保存模型
    file_path = "DQN/model/policy_net(" + num_vms_str + ").pth"
    agent = load_agent_from_file(file_path)
    vm_var_q, entity_var_q,vm__utilization_q,pm__utilization_q,overload_nums_q = evaluate_performance(all_task_types, choose_function="DQN", T=T, agent=agent)
    # model_definition.env_params_reset(num_pm=5,num_vms_per_type=[5,4,4])  # 重置环境参数
    vm_var_rr, entity_var_rr,vm__utilization_rr,pm__utilization_rr,overload_nums_rr = evaluate_performance(all_task_types, choose_function="RR", T=T, agent=None)
    # vm_var_random, entity_var_random,vm__utilization_random,pm__utilization_random,overload_nums_random = evaluate_performance(all_task_types, choose_function="Random", T=T, agent=None)
    # 计算每种类型虚拟机方差的均值
    avg_vm_var_q = [np.mean(vm_var_q[i]) for i in range(model_definition.NUM_TASK_TYPES)]
    avg_vm_var_rr = [np.mean(vm_var_rr[i]) for i in range(model_definition.NUM_TASK_TYPES)]
    avg_entity_var_q = np.mean(entity_var_q)
    avg_entity_var_rr = np.mean(entity_var_rr)
    # print("--------------------------------------")
    print("各算法每种应用类型虚拟机负载方差均值：")
    for i in range(model_definition.NUM_TASK_TYPES):
        print(f"Type {i} - DQN: {avg_vm_var_q[i]:.4f}, RR: {avg_vm_var_rr[i]:.4f}")
    print("--------------------------------------")
    print("各算法实体机负载方差均值：")
    print(f"DQN:       {avg_entity_var_q:.4f}")
    print(f"RR:        {avg_entity_var_rr:.4f}")
    print("--------------------------------------")
    print("多目标加权后各算法对比值：")
    print(f"DQN:       {0.3*np.mean(vm_var_q) + 0.7*avg_entity_var_q:.4f}")
    print(f"RR:        {0.3*np.mean(avg_vm_var_rr) + 0.7*avg_entity_var_rr:.4f}")
    # print("--------------------------------------")
    # print("各算法每种应用类型虚拟机平均利用率：")
    # for i in range(sum(model_definition.NUM_VMS_PER_TYPE)):
    #     print(f"VM {i} - DQN: {np.mean(vm__utilization_q[i]):.4f}, RR: {np.mean(vm__utilization_rr[i]):.4f}")
    # print("--------------------------------------")
    print("各算法每种应用类型实体机平均利用率：")
    for i in range(len(pm__utilization_q)): 
        print(f"PM {i} - DQN: {np.mean(pm__utilization_q[i]):.4f}")
    for i in range(len(pm__utilization_rr)): 
        print(f"PM {i} - RR: {np.mean(pm__utilization_rr[i]):.4f}")
    print("--------------------------------------")
    print("各算法违规率：")
    print(f"DQN:       {overload_nums_q/total_nums:.4f}")
    print(f"RR:        {overload_nums_rr/total_nums:.4f}")

    # 绘图  怎么画出好看的图像 画一个实体机资源利用率对比的图像
    plt.figure(figsize=(15,8))
    # 虚拟机负载方差（每种类型单独画线）
    # for i in range(model_definition.NUM_TASK_TYPES):
    #     plt.subplot(3,2,i+1)
    #     plt.plot(vm_var_q[i], label=f"DQN Type {i}")
    #     plt.plot(vm_var_rr[i], '--', label=f"RR Type {i}")
    #     plt.title("VM load variance (per type)")
    #     plt.xlabel("Episode")
    #     plt.ylabel("Variance")
    #     plt.legend()
    #     plt.grid()
    # 实体机负载方差
    # plt.subplot(3,2,5)
    plt.plot(entity_var_q, label="DP+DQN")
    plt.plot(entity_var_rr, label="Padding+RR")
    plt.title("PM load variance")
    plt.xlabel("Episode")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()



# model_definition.NUM_PM = 9  #根据这个来改  到时候model_definition.py里对应有个一改而改的操作函数  类似工具函数
# model_definition.env_params_reset(num_pm=6, num_task_types=3,num_vms_per_type=[2,2,2])  # 重置环境参数
comparison_()
# all_task_types, total_nums, T = get_task_sequence_from_file("DQN/task_sequence.txt")

# get_task_sequence_by_type(episodes=30, means_tasks=20,dist_type='poisson')
# get_task_sequence_by_type(episodes=30, means_tasks=20,dist_type='normal')
# get_task_sequence_by_type(episodes=30, means_tasks=20,dist_type='uniform')

# 示例用法
# np.random.seed(42)  # 设置随机种子确保结果可复现
# mean_tasks = 8     # 平均任务数量（与泊松分布示例一致）
# std_dev = 6        # 标准差（控制任务数量的波动程度）
# time_points = 10   # 时间点数量

# # 生成任务数量序列
# tasks = generate_normal_tasks(mean_tasks, std_dev, time_points)
# print(tasks)
#
# # 设置参数
# np.random.seed(42)  # 设置随机种子确保结果可复现
# lambda_ = 50  # 泊松分布的λ参数（平均任务数量）
# time_points = 10  # 时间点数量
#
# # 生成符合泊松分布的任务数量
# tasks = np.random.poisson(lam=lambda_, size=time_points)
# print(tasks)

