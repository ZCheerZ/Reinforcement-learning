import numpy as np
import pickle
import matplotlib.pyplot as plt  # 用于绘图
from model_definition import CloudEnv, QLearningAgent

# 超参数
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPISODES = 10
MAX_STEPS = 500

# 环境参数
NUM_TASK_TYPES = 3  # 应用类型数量
NUM_VMS_PER_TYPE = [3,3,3]  # 每种应用类型有3台虚拟机
VMS_PER_TYPE = [0,0,0,1,1,1,2,2,2]  # 每台虚拟机到应用类型的映射
NUM_PM = 3  # 实体机数量


def prefix_sum(arr):
    result = [0]
    total = 0
    for num in arr:
        total += num
        result.append(total)
    return result

def train_Q_learning():
    env = CloudEnv()
    agent = QLearningAgent()
    
    # 获取所有可能的状态（假设get_all_states方法已实现，否则需自定义）
    all_states = env.get_all_states()  # 你需要在CloudEnv中实现get_all_states方法，返回所有状态元组

    # 随机抽取
    tracked_state = [(0, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                    (0, 1, 1, 1, 2, 1, 1, 1, 1, 1),]
    tracked_action = [0,2]
    q_value_history = [[],[]]

    for episode in range(EPISODES):
        episode_reward = 0

        # 遍历所有任务类型和所有状态
        for task_type in range(NUM_TASK_TYPES):
            for state in all_states:
                available_actions = list(range(NUM_VMS_PER_TYPE[task_type]))
                # print(f"当前state {state}正在训练...")
                for action in available_actions:
                    # 执行动作并获取奖励
                    env.set_state(state)
                    prefix_NUM_VMS_PER_TYPE = prefix_sum(NUM_VMS_PER_TYPE)
                    reward, done = env.step(task_type, prefix_NUM_VMS_PER_TYPE[task_type] + action)
                    next_task_type = (task_type + 1) % NUM_TASK_TYPES  # 这里简单用下一个类型  到时候可以使用随机类型
                    next_state = env.get_state(next_task_type)

                    # Q学习更新
                    current_q = agent.q_table[state][action]
                    max_next_q = np.max(agent.q_table[next_state])
                    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q)
                    agent.q_table[state][action] = new_q
                    episode_reward += new_q

                    # 跟踪Q值
                    for i in range(len(tracked_state)):
                        if state == tracked_state[i] and action == tracked_action[i]:
                            q_value_history[i].append(agent.q_table[state][action])
                    
        print(f"Episode {episode}, Avg Reward: {episode_reward/(NUM_TASK_TYPES*len(all_states)*sum(NUM_VMS_PER_TYPE)/len(NUM_VMS_PER_TYPE)):.2f}")

    # 绘制 Q 值变化曲线
    plt.figure(figsize=(12,5))
    for i in range(len(tracked_state)):
        plt.subplot(1,2,i+1)
        plt.plot(q_value_history[i])
        plt.title(f"Q-value Convergence for State {tracked_state[i]} and Action {tracked_action[i]}")
        plt.xlabel("Training Steps")
        plt.ylabel("Q-value")
        plt.grid()
    
    plt.tight_layout()
    plt.show()

    # 保存Q表
    q_table_to_save = dict(agent.q_table)
    with open("Q-leaning/Experimental_environment/q_table.pkl", "wb") as f:
        pickle.dump(q_table_to_save, f)
    print("Q表已保存到 q_table.pkl")

train_Q_learning()


