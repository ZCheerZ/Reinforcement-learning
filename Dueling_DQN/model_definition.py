import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# 环境参数
NUM_TASK_TYPES = 7  # 任务类型数量
NUM_VMS_PER_TYPE = [2, 2, 3, 4, 2, 4, 3]  # 每种任务类型有多少台虚拟机
VMS_PER_TYPE = []  # 每台虚拟机到应用类型的映射
for i in range(NUM_TASK_TYPES):
    for j in range(NUM_VMS_PER_TYPE[i]):
        VMS_PER_TYPE.append(i)

NUM_PM = 7  # 实体机数量
VM_CAPACITY = [100, 120, 150, 150, 150, 150, 150]  # 每种应用类型对应的虚拟机容量
PM_CAPACITY = 300  # 实体机容量

TASK_CONFIG = {
    0: {"demand": 1, "duration": 5},
    1: {"demand": 2, "duration": 5},
    2: {"demand": 3, "duration": 5},
    3: {"demand": 5, "duration": 5},
    4: {"demand": 3, "duration": 5},
    5: {"demand": 4, "duration": 5},
    6: {"demand": 5, "duration": 5},
}
# 移除了多余的 7: ... 配置，因为 NUM_TASK_TYPES 为 7

# 计算每种应用类型的虚拟机前缀和
def prefix_sum(arr):
    result = [0]
    total = 0
    for num in arr:
        total += num
        result.append(total)
    return result

class CloudEnv:
    def __init__(self):
        self.vm_load = np.zeros(len(VMS_PER_TYPE), dtype=float)  # 每台虚拟机的负载
        # 初始放置策略：简单的轮询放置 (Round Robin)
        # 注意：这只是初始放置，后续不可变。若需动态迁移需修改。
        self.vm_to_pm = [i % NUM_PM for i in range(len(VMS_PER_TYPE))]  
        self.task_queues = [deque() for _ in range(len(VMS_PER_TYPE))]  # 每台虚拟机的任务队列
        self.prefix_NUM_VMS_PER_TYPE = prefix_sum(NUM_VMS_PER_TYPE)  # 每种应用类型的虚拟机前缀和
        self.reward_history = []

    def sample_next_task_type(self):
        """随机选择下一个任务类型"""
        return np.random.choice(NUM_TASK_TYPES)

    def get_action_mask(self, task_type: int):
        """生成当前任务类型下的动作掩码（合法虚拟机索引）"""
        valid = NUM_VMS_PER_TYPE[task_type]
        mask = np.zeros(max(NUM_VMS_PER_TYPE), dtype=np.float32) # 使用最大维度
        mask[:valid] = 1.0
        return mask
    
    def reset(self):
        self.vm_load.fill(0)
        for q in self.task_queues:
            q.clear()

    def get_state_vec(self, task_type: int):
        """
        紧凑型状态编码：
        [one_hot(task_type) | candidate_vm_utils | pm_utils | per_type_mean_utils]
        """
        one_hot = np.zeros(NUM_TASK_TYPES, dtype=np.float32)
        one_hot[task_type] = 1.0  # 任务类型 one-hot

        # 任务类型对应虚拟机利用率
        start = self.prefix_NUM_VMS_PER_TYPE[task_type]
        end = self.prefix_NUM_VMS_PER_TYPE[task_type + 1]
        # 注意：这里使用的是对应类型的容量
        vm_capacity = VM_CAPACITY[task_type] 
        vm_utils = [self.vm_load[i] / vm_capacity for i in range(start, end)]
        vm_utils += [0.0] * (max(NUM_VMS_PER_TYPE) - len(vm_utils))  # 填充为固定长度

        # PM 利用率
        _, pm_utils, _ = self.get_pm_info()

        # 每种任务类型的平均 VM 利用率 (全局状态信息)
        per_type_mean = np.zeros(NUM_TASK_TYPES, dtype=np.float32)
        for t in range(NUM_TASK_TYPES):
            s, e = self.prefix_NUM_VMS_PER_TYPE[t], self.prefix_NUM_VMS_PER_TYPE[t + 1]
            if e > s:
                type_capacity = VM_CAPACITY[t]
                per_type_mean[t] = np.mean([self.vm_load[i] / type_capacity for i in range(s, e)])

        return np.concatenate([one_hot, vm_utils, pm_utils, per_type_mean])

    def get_pm_info(self):
        """获取实体机的负载与资源利用率"""
        pm_utilization = []
        pm_loads = []
        for pm_id in range(NUM_PM):
            # 计算分配给该 PM 的所有 VM 的总负载
            load = sum(self.vm_load[i] for i in range(len(VMS_PER_TYPE)) if self.vm_to_pm[i] == pm_id)
            utilization = load / PM_CAPACITY
            pm_utilization.append(utilization)
            pm_loads.append(load)
        pm_var = np.var(pm_loads)
        return pm_loads, pm_utilization, pm_var

    def step_with_action(self, task_type: int, action: int, reward_clip: float = 1000.0):
        """执行动作，返回奖励和是否终止"""
        # 1. 更新所有 VM 的任务队列（模拟时间流逝，处理完成的任务）
        # 这里假设 step_with_action 代表一个时间步，或者代表一次调度事件。
        # 如果是事件驱动，duration 的含义需要明确。这里沿用原逻辑：每次调度都让所有任务剩余时间-1
        for vm in range(len(VMS_PER_TYPE)):
            new_queue = deque()
            while self.task_queues[vm]:
                remain, load = self.task_queues[vm].popleft()
                remain -= 1
                if remain > 0:
                    new_queue.append((remain, load))
                else:
                    self.vm_load[vm] -= load # 任务结束，释放负载
            self.task_queues[vm] = new_queue

        # 2. 获取目标 VM ID
        # action 是在当前 task_type 下的相对索引
        vm_id = self.prefix_NUM_VMS_PER_TYPE[task_type] + action
        
        task_demand = TASK_CONFIG[task_type]["demand"]
        task_duration = TASK_CONFIG[task_type]["duration"]
        vm_capacity = VM_CAPACITY[task_type]

        # 3. 检查约束
        # 检查是否超过 VM 容量
        if self.vm_load[vm_id] + task_demand > vm_capacity:
            reward = -reward_clip
            return reward, True  # 终止：VM 过载

        # 预分配负载
        self.vm_load[vm_id] += task_demand
        
        # 检查是否超过 PM 容量
        pm_loads, _, _ = self.get_pm_info()
        # 找到该 VM 所在的 PM
        target_pm = self.vm_to_pm[vm_id]
        if pm_loads[target_pm] > PM_CAPACITY:
             # 回滚
            self.vm_load[vm_id] -= task_demand
            reward = -reward_clip
            return reward, True # 终止：PM 过载

        # 4. 执行调度
        self.task_queues[vm_id].append((task_duration, task_demand))

        # 5. 计算奖励
        # 目标：同类型 VM 负载均衡 + 所有 PM 负载均衡
        
        # 同类型 VM 的负载方差
        s, e = self.prefix_NUM_VMS_PER_TYPE[task_type], self.prefix_NUM_VMS_PER_TYPE[task_type + 1]
        same_type_vms_load = [self.vm_load[i] for i in range(s, e)]
        # 可以归一化负载再计算方差，或者直接计算
        vm_var = np.var(same_type_vms_load)
        
        # PM 负载方差
        pm_var = np.var(pm_loads)
        
        # 奖励函数设定
        reward = -0.4 * vm_var - 0.6 * pm_var
        reward = np.clip(reward, -reward_clip, reward_clip)
        
        return reward, False

# Dueling DQN 网络结构
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        hidden = 128
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, output_dim)

    def forward(self, x):
        z = self.feature(x)
        value = self.value(z)
        advantage = self.advantage(z)
        # Dueling DQN 聚合层
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, use_double_dqn=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 0.5 # 初始 epsilon 可以高一点
        self.use_double_dqn = use_double_dqn

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state, action_mask):
        if random.random() < self.epsilon:
            valid_actions = np.where(action_mask > 0.5)[0]
            return random.choice(valid_actions)
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            q_values = self._mask_q(q_values, action_mask)
        return torch.argmax(q_values).item()

    def _mask_q(self, q_values, action_mask):
        """将 mask 为 0 的动作 Q 值设为极小值"""
        # q_values: [batch_size, action_dim] or [action_dim]
        # action_mask: [action_dim] or [batch_size, action_dim]
        
        if isinstance(action_mask, np.ndarray):
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32).to(q_values.device)
        else:
            mask_tensor = action_mask
            
        if q_values.dim() == 2 and mask_tensor.dim() == 1:
             mask_tensor = mask_tensor.unsqueeze(0)
             
        q_values[mask_tensor <= 0] = -1e9 # 使用一个足够小的数而不是 -inf 避免 NaN 问题
        return q_values

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < 64:
            return

        batch = random.sample(self.memory, 64)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # 当前状态的 Q 值
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # 计算 Mask：我们需要知道 next_states 对应的 task_type 来生成 mask
        # 假设 state 向量的前 NUM_TASK_TYPES 位是 one-hot 编码
        # 注意: 如果 state 结构改变，这里需要同步修改
        next_task_types_onehot = next_states[:, :NUM_TASK_TYPES]
        next_task_types = torch.argmax(next_task_types_onehot, dim=1)
        
        # 生成 batch mask
        batch_masks = []
        for t_type in next_task_types:
            valid = NUM_VMS_PER_TYPE[t_type.item()]
            m = torch.zeros(self.action_dim)
            m[:valid] = 1.0
            batch_masks.append(m)
        batch_masks = torch.stack(batch_masks)
        
        # 移动到同一 device
        if batch_masks.device != next_states.device:
            batch_masks = batch_masks.to(next_states.device)

        if self.use_double_dqn:
            # Double DQN: 
            # 1. 使用 Policy Net 选择动作 (argmax Q_policy)
            # 2. 使用 Target Net 评估该动作 (Q_target)
            
            with torch.no_grad():
                next_q_policy = self.policy_net(next_states)
                next_q_policy = self._mask_q(next_q_policy, batch_masks) # Mask 无效动作
                next_actions = next_q_policy.argmax(dim=1, keepdim=True)
                
                next_q_target = self.target_net(next_states)
                next_q_values = next_q_target.gather(1, next_actions)
        else:
            # Standard DQN
            with torch.no_grad():
                next_q_target = self.target_net(next_states)
                next_q_target = self._mask_q(next_q_target, batch_masks) # Mask 无效动作
                next_q_values = next_q_target.max(1)[0].unsqueeze(1)

        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
