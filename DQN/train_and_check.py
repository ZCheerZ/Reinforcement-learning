import numpy as np
import random
import matplotlib.pyplot as plt  # 用于绘图
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from model_definition import CloudEnv, DQNAgent, NUM_TASK_TYPES, NUM_VMS_PER_TYPE, VMS_PER_TYPE, NUM_PM,TARGET_UPDATE_FREQ

# 超参数
EPSILON = 0.2
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
EPISODES = 1500
MAX_STEPS = 1024*3


def train_test():
    # 初始化环境与智能体
    env = CloudEnv()
    state_dim = 1 + sum(NUM_VMS_PER_TYPE)  # 状态维度
    action_dim = max(NUM_VMS_PER_TYPE)  # 动作维度
    agent = DQNAgent(state_dim, action_dim)
    # 记录每个回合的总奖励
    rewards_history = []

    # 训练循环
    for episode in range(EPISODES):
        env.reset()
        state = env.get_state(random.randint(0, NUM_TASK_TYPES - 1))
        state = np.array(state, dtype=np.float32)
        episode_reward = 0

        for step in range(MAX_STEPS):
            # 选择动作
            action = agent.choose_action_multi(state)

            # 执行动作
            task_type = state[0]  # 当前任务类型
            # 原来是这里出错了！！！
            # vm_id = int(task_type * NUM_VMS_PER_TYPE[int(task_type)] + action)
            vm_id = env.prefix_NUM_VMS_PER_TYPE[int(task_type)] + action  # 虚拟机ID
            reward, done = env.step(int(task_type), vm_id)
            next_state = env.get_state(random.randint(0, NUM_TASK_TYPES - 1))
            next_state = np.array(next_state, dtype=np.float32)

            # 存储经验
            agent.store_experience(state, action, reward, next_state, done)

            # 更新状态
            state = next_state
            episode_reward += reward

            # 训练网络
            agent.train()
            if done:
                break

        # 减少 epsilon
        agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)

        # 更新目标网络
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # if episode % 500 == 0:
        print(f"Episode {episode}, Avg Reward: {episode_reward:.2f}")

        rewards_history.append(episode_reward)
        

    # 测试示例
    test_task_type = 0
    test_state = env.get_state(test_task_type)
    # test_state = (0, 2, 1, 2, 1, 2, 2, 2, 2, 1,2)
    test_state = np.array(test_state, dtype=np.float32)
    print(f"测试状态: {test_state}")
    print(f"测试动作值分布: {agent.policy_net(torch.FloatTensor(test_state).unsqueeze(0))}")

    # 绘制奖励曲线
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.show()

    # 保存模型
    file_path = "DQN/model/policy_net(244).pth"
    torch.save(agent.policy_net.state_dict(), file_path)
    print("模型已保存到", file_path)


train_test()