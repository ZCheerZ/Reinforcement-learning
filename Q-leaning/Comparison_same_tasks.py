import numpy as np
import random
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt  # 用于绘图
from model_definition import CloudEnv, QLearningAgent


# 超参数
MAX_STEPS = 50

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

def evaluate_with_same_tasks(agent, episodes=100):
    """
    用同一批任务序列分别评估Q-learning、随机分配和轮询分配的负载均衡效果
    :param agent: 已训练好的 QLearningAgent
    :param episodes: 评估轮数
    :return: (vm_var_q, entity_var_q, vm_var_random, entity_var_random, vm_var_rr, entity_var_rr)
    """
    # 1. 生成所有任务序列（每一轮的任务类型序列）
    all_task_types = []
    for _ in range(episodes):
        task_types = [random.randint(0, NUM_TASK_TYPES-1) for _ in range(MAX_STEPS)]
        all_task_types.append(task_types)

    # 2. Q-learning评估
    env_q = CloudEnv()
    vm_var_q, entity_var_q = [], []
    for ep in range(episodes):
        env_q.reset()
        state = env_q.get_state(all_task_types[ep][0])
        for t in range(MAX_STEPS):
            task_type = all_task_types[ep][t]
            available_actions = list(range(NUM_VMS_PER_TYPE))
            action = agent.choose_action(state, available_actions)
            env_q.step(task_type, task_type * NUM_VMS_PER_TYPE + action)
            # 下一个任务
            next_task_type = all_task_types[ep][t] if t+1 >= MAX_STEPS else all_task_types[ep][t+1]
            state = env_q.get_state(next_task_type)
        # 记录方差
        vm_var_q.append(np.var(env_q.vm_load))
        entity_loads = []
        for e in range(NUM_VMS_PER_TYPE):
            load = sum(
                env_q.vm_load[i]
                for i in range(NUM_TASK_TYPES * NUM_VMS_PER_TYPE)
                if env_q.vm_to_entity[i] == e
            )
            entity_loads.append(load)
        entity_var_q.append(np.var(entity_loads))

    # 3. 随机分配评估（用同样的任务序列）
    env_r = CloudEnv()
    vm_var_random, entity_var_random = [], []
    for ep in range(episodes):
        env_r.reset()
        state = env_r.get_state(all_task_types[ep][0])
        for t in range(MAX_STEPS):
            task_type = all_task_types[ep][t]
            available_actions = list(range(NUM_VMS_PER_TYPE))
            action = random.choice(available_actions)
            env_r.step(task_type, task_type * NUM_VMS_PER_TYPE + action)
            # 下一个任务
            next_task_type = all_task_types[ep][t] if t+1 >= MAX_STEPS else all_task_types[ep][t+1]
            state = env_r.get_state(next_task_type)
        # 记录方差
        vm_var_random.append(np.var(env_r.vm_load))
        
        entity_loads = []
        for e in range(NUM_VMS_PER_TYPE):
            load = sum(
                env_r.vm_load[i]
                for i in range(NUM_TASK_TYPES * NUM_VMS_PER_TYPE)
                if env_r.vm_to_entity[i] == e
            )
            entity_loads.append(load)
        entity_var_random.append(np.var(entity_loads))

    # 4. 轮询分配评估（用同样的任务序列）
    env_rr = CloudEnv()
    vm_var_rr, entity_var_rr = [], []
    rr_pointer = [0 for _ in range(NUM_TASK_TYPES)]  # 每种任务类型一个指针
    for ep in range(episodes):
        env_rr.reset()
        state = env_rr.get_state(all_task_types[ep][0])
        for t in range(MAX_STEPS):
            task_type = all_task_types[ep][t]
            available_actions = list(range(NUM_VMS_PER_TYPE))
            # 轮询分配：每种任务类型独立轮询
            action = rr_pointer[task_type]
            rr_pointer[task_type] = (rr_pointer[task_type] + 1) % NUM_VMS_PER_TYPE
            env_rr.step(task_type, task_type * NUM_VMS_PER_TYPE + action)
            # 下一个任务
            next_task_type = all_task_types[ep][t] if t+1 >= MAX_STEPS else all_task_types[ep][t+1]
            state = env_rr.get_state(next_task_type)
        # 记录方差
        vm_var_rr.append(np.var(env_rr.vm_load))
        entity_loads = []
        for e in range(NUM_VMS_PER_TYPE):
            load = sum(
                env_rr.vm_load[i]
                for i in range(NUM_TASK_TYPES * NUM_VMS_PER_TYPE)
                if env_rr.vm_to_entity[i] == e
            )
            entity_loads.append(load)
        entity_var_rr.append(np.var(entity_loads))

    return vm_var_q, entity_var_q, vm_var_random, entity_var_random, vm_var_rr, entity_var_rr

def train_Q_learning_and_evaluate():

    agent = load_agent_from_file()

    # 用同一任务序列评估
    vm_var_q, entity_var_q, vm_var_random, entity_var_random, vm_var_rr, entity_var_rr = evaluate_with_same_tasks(agent, episodes=500)

    # 计算平均方差
    avg_vm_var_q = np.mean(vm_var_q)
    avg_vm_var_random = np.mean(vm_var_random)
    avg_vm_var_rr = np.mean(vm_var_rr)
    avg_entity_var_q = np.mean(entity_var_q)
    avg_entity_var_random = np.mean(entity_var_random)
    avg_entity_var_rr = np.mean(entity_var_rr)

    print("各算法虚拟机负载方差均值：")
    print(f"Q-learning: {avg_vm_var_q:.4f}")
    print(f"Random:    {avg_vm_var_random:.4f}")
    print(f"RR:        {avg_vm_var_rr:.4f}")

    print("各算法实体机负载方差均值：")
    print(f"Q-learning: {avg_entity_var_q:.4f}")
    print(f"Random:    {avg_entity_var_random:.4f}")
    print(f"RR:        {avg_entity_var_rr:.4f}")

    # 绘图
    plt.figure(figsize=(12,5))
    plt.subplot(2,1,1)
    plt.plot(vm_var_q, label="Q-learning")
    plt.plot(vm_var_random, label="Random")
    plt.plot(vm_var_rr, label="RR")
    plt.title("VM load variance")
    plt.xlabel("Episode")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(entity_var_q, label="Q-learning")
    plt.plot(entity_var_random, label="Random")
    plt.plot(entity_var_rr, label="RR")
    plt.title("PM load variance")  
    plt.xlabel("Episode")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

train_Q_learning_and_evaluate()


# 当前我想检验一下该强化学习算法的奖励函数是否能维持虚拟机和实体机的负载均衡，
# 你可以在后面10000轮的每一轮将其存入变量并在最后为我最后画图展示嘛，
# 甚至还可以设置一个简单的对比算法来展示出强化学习得到的虚拟机和实体机的负载均衡！
# 请基于以上两个点为我写出相应代码并给出注释！