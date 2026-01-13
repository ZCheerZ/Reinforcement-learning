import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from model_definition import CloudEnv, DQNAgent,NUM_TASK_TYPES, NUM_VMS_PER_TYPE, VMS_PER_TYPE, NUM_PM

EPSILON_START = 0.2
EPSILON_END = 0.01
EPISODES = 3000
MAX_STEPS = 512

def linear_epsilon(episode, total_episodes, e0=EPSILON_START, e1=EPSILON_END, warmup_ratio=0.3):
    ratio = episode / total_episodes
    if ratio < warmup_ratio:
        return e0
    return max(e1, e0 - (ratio - warmup_ratio) / (1 - warmup_ratio) * (e0 - e1))

def train_dueling_ddqn_per():
    env = CloudEnv()
    action_dim = max(NUM_VMS_PER_TYPE)
    state_dim = env.get_state_vec(0).shape[0]

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, use_double_dqn=True)

    rewards_history = []

    for episode in range(EPISODES):
        env.reset()
        agent.epsilon = linear_epsilon(episode, EPISODES)

        ep_reward = 0.0
        done = False
        task_type = env.sample_next_task_type()

        for step in range(MAX_STEPS):
            state = env.get_state_vec(task_type)
            action_mask = env.get_action_mask(task_type)
            action = agent.choose_action(state, action_mask)

            reward, done = env.step_with_action(task_type, action)
            ep_reward += reward

            next_task_type = env.sample_next_task_type()
            next_state = env.get_state_vec(next_task_type)

            agent.store_experience(state, action, reward, next_state, done)

            agent.train()

            task_type = next_task_type
            if done:
                break

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {ep_reward}")

        rewards_history.append(ep_reward)

        if episode % 100 == 0:
            agent.update_target_network()

    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.savefig('Dueling_DQN/training_progress.png')
    # plt.show()

    # 保存模型
    num_vms_str = ''.join(str(x) for x in NUM_VMS_PER_TYPE)
    torch.save(agent.policy_net.state_dict(), f"Dueling_DQN/model/policy_net({num_vms_str}).pth")
    print("模型已保存。")

if __name__ == "__main__":
    train_dueling_ddqn_per()
