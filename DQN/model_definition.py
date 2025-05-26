import numpy as np
import random
import itertools
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.optim as optim
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# 超参数

LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.95
BATCH_SIZE = 2000
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 100
EPSILON = 0.3 # 探索率
a = 0.4 
b = 0.5

# 环境参数
NUM_TASK_TYPES = 3  # 应用类型数量
NUM_VMS_PER_TYPE = [3,3,3]  # 每种应用类型有3台虚拟机
VMS_PER_TYPE = [0,0,0,1,1,1,2,2,2]  # 每台虚拟机到应用类型的映射
NUM_PM = 3  # 实体机数量
TASK_CONFIG = {  # 不同应用类型的任务预定义参数  需求10%是为了使得离散值都能覆盖到
    0: {"demand": 10, "duration": 5},  # 类型0: 需求10%，持续8步长
    1: {"demand": 10, "duration": 6},  # 类型1: 需求10%，持续9步长
    2: {"demand": 10, "duration": 7},  # 类型2: 需求10%，持续7步长
}
VM_CAPACITY = [100,100,100]  # 虚拟机容量，执行不同应用类型任务的虚拟机资源容量
ENTITY_CAPACITY = 200  # 实体机容量（300%）


class CloudEnv:
    def __init__(self):
        # 虚拟机负载（百分比），格式：{虚拟机ID: 当前总负载}
        self.vm_load = np.zeros(sum(NUM_VMS_PER_TYPE), dtype=float)
        
        # 虚拟机到实体机的映射
        self.vm_to_entity = [0,1,2,0,1,2,0,1,2]  # 假设虚拟机i平铺部署在实体机上
        
        # 任务队列：记录每个虚拟机中正在执行的任务（剩余步长, 负载）
        self.task_queues = [deque() for _ in range(sum(NUM_VMS_PER_TYPE))]
        
    def _get_vm_level(self, load, vm_type):
        rate = load / VM_CAPACITY[vm_type] *100  # 获取对应虚拟机的容量比率
        # 将负载百分比转换为离散等级（1/2/3.../10）
        if rate < 10:
            return 1
        elif 10 <= rate < 20:
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
        #  构建状态：应用类型 + 所有虚拟机负载等级 所有虚拟机的负载等级（按应用类型分组）
        vm_levels = [self._get_vm_level(self.vm_load[i], VMS_PER_TYPE[i]) 
                    for i in range(sum(NUM_VMS_PER_TYPE))]
        return (task_type,) + tuple(vm_levels)
    
    def get_all_states(self):
        """
        枚举所有可能的状态：(task_type, vm_level_0, ..., vm_level_8)
        task_type: 0~NUM_TASK_TYPES-1
        vm_level: 1~10
        """
        all_states = []
        vm_level_range = list(range(1, 11))  # 1~11
        for task_type in range(NUM_TASK_TYPES):
            # 9台虚拟机，每台有4种level
            for vm_levels in itertools.product(vm_level_range, repeat=sum(NUM_VMS_PER_TYPE)):
                state = (task_type,) + vm_levels
                all_states.append(state)
        return all_states

    def set_state(self, state):
        """
        将环境设置为指定状态（仅设置vm_load，不处理队列）
        state: (task_type, vm_level_0, ..., vm_level_8)
        """
        # 只设置虚拟机负载，task_type不需要设置
        for i in range(sum(NUM_VMS_PER_TYPE)):
            # 反推负载百分比区间的均值
            vm_type = VMS_PER_TYPE[i]
            level = state[i + 1]
            # 反推百分比区间的中值
            self.vm_load[i] = (level*10-5) / 100 * VM_CAPACITY[vm_type]
        # 清空任务队列（注意：此处未恢复任务队列，仅用于Q表学习）
        for q in self.task_queues:
            q.clear()
    
    def step(self, task_type, vm_id):
        #  执行动作：分配任务到虚拟机，并处理任务队列
        #  处理任务队列（减少剩余步长）
        released_load = 0
        for vm in range(sum(NUM_VMS_PER_TYPE)):
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
        
        #添加新任务到队列
        task_demand = TASK_CONFIG[task_type]["demand"]
        task_duration = TASK_CONFIG[task_type]["duration"]
        
        # 检查虚拟机容量是否足够
        if self.vm_load[vm_id] + task_demand > VM_CAPACITY[VMS_PER_TYPE[vm_id]]:
            reward = -10  # 直接拒绝任务的惩罚
            return reward, False
        
        # 更新虚拟机负载
        self.vm_load[vm_id] += task_demand
        self.task_queues[vm_id].append((task_duration, task_demand))

        # 计算奖励
        # 同类型虚拟机的负载方差
        same_type_vms = [self.vm_load[i] for i in range(sum(NUM_VMS_PER_TYPE))
                       if VMS_PER_TYPE[i] == task_type]
        vm_var = np.var(same_type_vms)
        
        # 实体机负载方差
        entity_loads = []
        for e in range(NUM_PM):  # 实体机数量
            load = sum(
                self.vm_load[i] 
                for i in range(sum(NUM_VMS_PER_TYPE))
                if self.vm_to_entity[i] == e
            )
            entity_loads.append(load)
        entity_var = np.var(entity_loads)
        
        # 过载惩罚（任一实体机超载）
        overload_penalty = 10 if any(l > ENTITY_CAPACITY for l in entity_loads) else 0
        
        reward = -a * vm_var - b * entity_var - overload_penalty
        return reward, False

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



