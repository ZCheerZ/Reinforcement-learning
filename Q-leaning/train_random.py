import numpy as np
import random
import pickle
import matplotlib.pyplot as plt  # 用于绘图
from model_definition import CloudEnv, QLearningAgent

# 超参数
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPISODES = 50
MAX_STEPS = 500

# 环境参数
NUM_TASK_TYPES = 3
NUM_VMS_PER_TYPE = 3  # 每种任务类型有3台虚拟机

def train_Q_learning():
    # 初始化环境与智能体
    env = CloudEnv()
    agent = QLearningAgent()
    tracked_state = [(0, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                    (0, 1, 1, 1, 2, 1, 1, 1, 1, 1),]  # 固定跟踪的状态
    tracked_action = [0,2]  # 固定跟踪的动作
    q_value_history = [[],[]]  # 用于存储 Q 值变化

    # 训练循环
    for episode in range(EPISODES):
        env.reset()
        episode_reward = 0
        # 生成随机任务类型
        task_type = random.randint(0, NUM_TASK_TYPES-1)
            
        # 获取当前状态（不包含任务需求，因需求已预定义）
        state = env.get_state(task_type)
        
        for _ in range(MAX_STEPS):
            
            # 可选动作：该任务类型对应的3台虚拟机
            available_actions = list(range(NUM_VMS_PER_TYPE))
            
            # 选择动作
            action = agent.choose_action(state, available_actions)
            
            # 执行动作并获取奖励
            reward, done = env.step(task_type, task_type * NUM_VMS_PER_TYPE + action)
            # episode_reward += reward
            
            # 获取新状态（下一个任务的类型）
            next_task_type = random.randint(0, NUM_TASK_TYPES-1)
            next_state = env.get_state(next_task_type)
            
            # 更新Q表
            current_q = agent.q_table[state][action]
            max_next_q = np.max(agent.q_table[next_state])
            
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q)
            agent.q_table[state][action] = new_q
            episode_reward += new_q

            # 如果当前状态是被跟踪的状态，记录其 Q 值
            for i in range(len(tracked_state)):
                if state == tracked_state[i] and action == tracked_action[i]:
                    q_value_history[i].append(agent.q_table[state][action])

            # if state == tracked_state and action == tracked_action:
            #     q_value_history.append(agent.q_table[state][action])
            
            # 更新状态
            state = next_state
            
        if episode % 100 == 0:
            print(f"Episode {episode}, Avg Reward: {episode_reward/MAX_STEPS:.2f}")

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
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table_to_save, f)
    print("Q表已保存到 q_table.pkl")


