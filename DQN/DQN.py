import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# 超参数
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.95
EPSILON = 0.2
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
EPISODES = 10000
MAX_STEPS = 4000
BATCH_SIZE = 2000
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 100
a = 0.4
b = 0.5

# 环境参数
NUM_TASK_TYPES = 3
NUM_VMS_PER_TYPE = 3
TASK_CONFIG = {
    0: {"demand": 20, "duration": 2},
    1: {"demand": 30, "duration": 3},
    2: {"demand": 40, "duration": 4},
}
VM_CAPACITY = 100
ENTITY_CAPACITY = 200

class CloudEnv:
    def __init__(self):
        # 虚拟机负载（百分比），格式：{虚拟机ID: 当前总负载}
        self.vm_load = np.zeros(NUM_TASK_TYPES * NUM_VMS_PER_TYPE, dtype=float)
        
        # 虚拟机到实体机的映射（假设虚拟机i部署在实体机i//3）
        self.vm_to_entity = [i // NUM_VMS_PER_TYPE for i in range(NUM_TASK_TYPES * NUM_VMS_PER_TYPE)]
        
        # 任务队列：记录每个虚拟机中正在执行的任务（剩余步长, 负载）
        self.task_queues = [deque() for _ in range(NUM_TASK_TYPES * NUM_VMS_PER_TYPE)]
        
    def _get_vm_level(self, load):
        """将负载百分比转换为离散等级（1/2/3）"""
        if load < 10:
            return 1
        elif 10 <= load < 20:
            return 2
        elif 20 <= load < 30:
            return 3
        elif 30 <= load < 40:
            return 4
        elif 40 <= load < 50:
            return 5
        elif 50 <= load < 60:
            return 6
        elif 60 <= load < 70:
            return 7
        elif 70 <= load < 80:
            return 8
        elif 80 <= load < 90:
            return 9
        else:
            return 10
        
    def get_state(self, task_type):
        """构建状态：任务类型 + 所有虚拟机负载等级"""
        # 所有虚拟机的负载等级（按任务类型分组）
        vm_levels = [self._get_vm_level(self.vm_load[i]) 
                    for i in range(NUM_TASK_TYPES * NUM_VMS_PER_TYPE)]
        return (task_type,) + tuple(vm_levels)
    
    def _update_entity_load(self, vm_id, delta):
        """更新实体机负载"""
        entity_id = self.vm_to_entity[vm_id]
        # 计算该实体机下所有虚拟机的总负载
        total_load = sum(
            self.vm_load[i] 
            for i in range(NUM_TASK_TYPES * NUM_VMS_PER_TYPE)
            if self.vm_to_entity[i] == entity_id
        )
        # 检查是否过载（仅用于奖励计算，不实际限制负载）
        return total_load + delta
    
    def step(self, task_type, vm_id):
        """执行动作：分配任务到虚拟机，并处理任务队列"""
        # --------------------
        # 1. 处理任务队列（减少剩余步长）
        # --------------------
        released_load = 0
        for vm in range(NUM_TASK_TYPES * NUM_VMS_PER_TYPE):
            new_queue = deque()
            while self.task_queues[vm]:
                remain_steps, load = self.task_queues[vm].popleft()
                remain_steps -= 1
                if remain_steps > 0:
                    new_queue.append((remain_steps, load))
                else:  # 任务完成，释放负载
                    self.vm_load[vm] -= load
                    released_load += load
            self.task_queues[vm] = new_queue
        
        # --------------------
        # 2. 添加新任务到队列
        # --------------------
        task_demand = TASK_CONFIG[task_type]["demand"]
        task_duration = TASK_CONFIG[task_type]["duration"]
        
        # 检查虚拟机容量是否足够
        if self.vm_load[vm_id] + task_demand > VM_CAPACITY:
            reward = -10  # 直接拒绝任务的惩罚
            return reward, False
        
        # 更新虚拟机负载
        self.vm_load[vm_id] += task_demand
        self.task_queues[vm_id].append((task_duration, task_demand))
        # for i in range(NUM_TASK_TYPES * NUM_VMS_PER_TYPE):
        #     print(f"虚拟机{i}当前负载：{self.vm_load[i]}%")
        # --------------------
        # 3. 计算奖励
        # --------------------
        # 同类型虚拟机的负载方差
        same_type_vms = [self.vm_load[i] for i in range(NUM_TASK_TYPES * NUM_VMS_PER_TYPE)
                       if i // NUM_VMS_PER_TYPE == task_type]
        vm_var = np.var(same_type_vms)
        
        # 实体机负载方差
        entity_loads = []
        for e in range(NUM_VMS_PER_TYPE):  # 实体机数量
            load = sum(
                self.vm_load[i] 
                for i in range(NUM_TASK_TYPES * NUM_VMS_PER_TYPE)
                if self.vm_to_entity[i] == e
            )
            entity_loads.append(load)
        entity_var = np.var(entity_loads)
        
        # 过载惩罚（任一实体机超载）
        overload_penalty = 10 if any(l > ENTITY_CAPACITY for l in entity_loads) else 0
        
        reward = -a * vm_var - b * entity_var - overload_penalty
        return reward, False  # 无终止条件

    def reset(self):
        self.vm_load.fill(0)
        for q in self.task_queues:
            q.clear()

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
# 能具体跟我说说这个DQNAgent它干了什么嘛，可以结合代码进行解释，我不太理解它怎么进行学习的，并且怎么去表示状态q值
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON
        self.memory = deque(maxlen=MEMORY_SIZE)

        # 主网络和目标网络 主网络根据输入的状态输出Q值  目标网络是记忆中最好的那一次怎么做的网络 提供标签值
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # states = torch.FloatTensor(states)
        # actions = torch.LongTensor(actions).unsqueeze(1)
        # rewards = torch.FloatTensor(rewards).unsqueeze(1)
        # next_states = torch.FloatTensor(next_states)
        # dones = torch.FloatTensor(dones).unsqueeze(1)

        # 将列表转换为 numpy.ndarray，然后再转换为 Tensor
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)


        # 计算当前 Q 值
        q_values = self.policy_net(states).gather(1, actions)

        # 计算目标 Q 值
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * DISCOUNT_FACTOR * max_next_q_values

        # 更新主网络
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train_test():
    # 初始化环境与智能体
    env = CloudEnv()
    state_dim = 1 + NUM_TASK_TYPES * NUM_VMS_PER_TYPE  # 状态维度
    action_dim = NUM_VMS_PER_TYPE  # 动作维度
    agent = DQNAgent(state_dim, action_dim)

    # 训练循环
    for episode in range(EPISODES):
        env.reset()
        state = env.get_state(random.randint(0, NUM_TASK_TYPES - 1))
        state = np.array(state, dtype=np.float32)
        episode_reward = 0

        for step in range(MAX_STEPS):
            # 选择动作
            action = agent.choose_action(state)

            # 执行动作
            task_type = state[0]  # 当前任务类型
            vm_id = int(task_type * NUM_VMS_PER_TYPE + action)
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

        if episode % 500 == 0:
            print(f"Episode {episode}, Avg Reward: {episode_reward / MAX_STEPS:.2f}")

    # 测试示例
    test_task_type = 0
    test_state = env.get_state(test_task_type)
    test_state = np.array(test_state, dtype=np.float32)
    print(f"测试状态: {test_state}")
    print(f"测试动作值分布: {agent.policy_net(torch.FloatTensor(test_state).unsqueeze(0))}")

def train_convergence():
    print("训练收敛")
    # 初始化环境与智能体
    env = CloudEnv()
    state_dim = 1 + NUM_TASK_TYPES * NUM_VMS_PER_TYPE  # 状态维度
    action_dim = NUM_VMS_PER_TYPE  # 动作维度
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
            action = agent.choose_action(state)

            # 执行动作
            task_type = state[0]  # 当前任务类型
            vm_id = int(task_type * NUM_VMS_PER_TYPE + action)
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

        # 记录当前回合的总奖励  好像并没有收敛  可能是收敛的方式不对  这个是不是应该放更新目标网络里面
        rewards_history.append(episode_reward)

        # 打印日志
        if episode % 500 == 0:
            avg_reward = sum(rewards_history[-500:]) / 500 if len(rewards_history) >= 500 else episode_reward
            print(f"Episode {episode}, Avg Reward (last 500): {avg_reward:.2f}")

    # 绘制奖励曲线
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.show()

train_test()