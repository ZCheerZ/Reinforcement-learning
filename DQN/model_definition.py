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
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.8
BATCH_SIZE = 64
MEMORY_SIZE = 50000
TARGET_UPDATE_FREQ = 100
EPSILON = 0.3 # 探索率
a = 0.4 
b = 0.5
overload = 10000

# 环境参数
NUM_TASK_TYPES = 4  # 应用类型数量
NUM_VMS_PER_TYPE = [2,4,2,3]  # 每种应用类型有多少台虚拟机
VMS_PER_TYPE = [] # 每台虚拟机到应用类型的映射
for i in range(NUM_TASK_TYPES):
    for j in range(NUM_VMS_PER_TYPE[i]):
        VMS_PER_TYPE.append(i)
# print("VMS_PER_TYPE:", VMS_PER_TYPE)
# VMS_PER_TYPE = [0,0,1,1,1,1,2,2,3,3,3]  
NUM_PM = 3  # 实体机数量
TASK_CONFIG = {  # 不同应用类型的任务预定义参数  需求10%是为了使得离散值都能覆盖到
    0: {"demand": 10, "duration": 30},  # 类型0: 需求10%，持续8步长
    1: {"demand": 10, "duration": 30},  # 类型1: 需求10%，持续9步长
    2: {"demand": 10, "duration": 30},  # 类型2: 需求10%，持续7步长
    3: {"demand": 10, "duration": 30},  # 类型2: 需求10%，持续7步长

}
VM_CAPACITY = [100,100,100,100]  # 虚拟机容量，执行不同应用类型任务的虚拟机资源容量
ENTITY_CAPACITY = 200  # 实体机容量（300%）


def prefix_sum(arr):
    result = [0]
    total = 0
    for num in arr:
        total += num
        result.append(total)
    return result

class CloudEnv:
    def __init__(self):
        # 虚拟机负载（百分比），格式：{虚拟机ID: 当前总负载}
        self.vm_load = np.zeros(sum(NUM_VMS_PER_TYPE), dtype=float)
        # 虚拟机到实体机的映射  todo:到时候需要动态表示
        self.vm_to_entity = []  # 每个虚拟机对应的实体机ID 假设虚拟机i平铺部署在实体机上
        for i in range(sum(NUM_VMS_PER_TYPE)):
            self.vm_to_entity.append(i%NUM_PM) 
        print("vm_to_entity:", self.vm_to_entity)
        # self.vm_to_entity = [0,1,2,0,1,2,0,1,2,0,1] 
        # 任务队列：记录每个虚拟机中正在执行的任务（剩余步长, 负载）
        self.task_queues = [deque() for _ in range(sum(NUM_VMS_PER_TYPE))]
        # 每种应用类型对应虚拟机数量前缀和
        self.prefix_NUM_VMS_PER_TYPE = prefix_sum(NUM_VMS_PER_TYPE)
        # 初始化轮询指针
        self.rr_pointer = [0 for _ in range(NUM_TASK_TYPES)]
        
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
            # 每台虚拟机有10种level 10的n次方个组合  所以不能用q-learning
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
            reward = -1 * overload  # 直接拒绝任务的惩罚
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
        overload_penalty = overload if any(l > ENTITY_CAPACITY for l in entity_loads) else 0
        
        reward = -a * vm_var - b * entity_var - overload_penalty
        return reward, False
    
    def step_batch(self, task_types, agent , choose_function):
        """
        批量执行任务分配，不计算奖励，只更新负载和判断过载。
        若选择的虚拟机超载，尝试分配到同类型下不超载的虚拟机，并记录超载虚拟机。
        :param task_types: 任务类型数组，如 [0, 1, 2]
        :param agent:      DQNAgent 实例，用于获取动作
        :return: (vm_load, entity_loads, overload_flag, overload_vms)
        """
        overload_vms = []  # 记录本次尝试分配时超载的虚拟机ID

        #此处引出一个之前忽略的问题  就是我这个选择动作应该放到处理所有虚拟机的任务队列之后  我之前的学习都是放在之前就选完了  先看效果吧

        # 1. 处理所有虚拟机的任务队列（减少剩余步长，释放负载）  
        for vm in range(sum(NUM_VMS_PER_TYPE)):
            new_queue = deque()
            while self.task_queues[vm]:
                remain_steps, load = self.task_queues[vm].popleft()
                remain_steps -= 1
                if remain_steps > 0:
                    new_queue.append((remain_steps, load))
                else:
                    self.vm_load[vm] -= load
            self.task_queues[vm] = new_queue

        # 2. 批量添加新任务
        for task_type in (task_types):
            task_demand = TASK_CONFIG[task_type]["demand"]
            task_duration = TASK_CONFIG[task_type]["duration"]
            state = self.get_state(task_type)
            test_state = np.array(state, dtype=np.int32)
            if(choose_function == "DQN"):
                action = agent.choose_action_multi(test_state,0)
            elif(choose_function == "RR"):
                available_actions = list(range(NUM_VMS_PER_TYPE[task_type]))
                action = self.rr_pointer[task_type]
                self.rr_pointer[task_type] = (self.rr_pointer[task_type] + 1) % NUM_VMS_PER_TYPE[task_type]
            elif(choose_function == "Random"):
                available_actions = list(range(NUM_VMS_PER_TYPE[task_type]))
                action = random.choice(available_actions)


            vm_id = self.prefix_NUM_VMS_PER_TYPE[task_type] + action  # 获取全局虚拟机ID
            # 检查目标虚拟机是否超载
            if self.vm_load[vm_id] + task_demand <= VM_CAPACITY[VMS_PER_TYPE[vm_id]]:
                self.vm_load[vm_id] += task_demand
                self.task_queues[vm_id].append((task_duration, task_demand))
            else:
                overload_vms.append(vm_id)
                # 尝试分配到同类型下其他不超载的虚拟机
                vm_type = VMS_PER_TYPE[vm_id]
                # 找到所有同类型虚拟机的ID
                candidate_vms = [i for i in range(sum(NUM_VMS_PER_TYPE)) if VMS_PER_TYPE[i] == vm_type]
                allocated = False
                for alt_vm in candidate_vms:
                    if self.vm_load[alt_vm] + task_demand <= VM_CAPACITY[vm_type]:
                        self.vm_load[alt_vm] += task_demand
                        self.task_queues[alt_vm].append((task_duration, task_demand))
                        allocated = True
                        print(f"Task type {task_type} allocated to VM {alt_vm} instead of overloaded VM {vm_id}.")
                        break
                # 如果所有同类型虚拟机都超载，则该任务不分配
   

        # 3. 统计实体机负载
        entity_loads = []
        for e in range(NUM_PM):
            load = sum(
                self.vm_load[i]
                for i in range(self.prefix_NUM_VMS_PER_TYPE[-1])
                if self.vm_to_entity[i] == e
            )
            entity_loads.append(load)

        # 4. 判断是否有实体机过载
        overload_flag = any(l > ENTITY_CAPACITY for l in entity_loads)

        return self.vm_load.copy(), entity_loads, overload_flag, overload_vms

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

        self.prefix_NUM_VMS_PER_TYPE = prefix_sum(NUM_VMS_PER_TYPE)
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
    
    def choose_action_multi(self, state ,epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        """
        根据当前状态选择动作，动作空间根据任务类型动态调整。
        state: (task_type, vm_level_0, ..., vm_level_8)
        返回：全局虚拟机编号
        """
        task_type = int(state[0])
        # 当前任务类型可选虚拟机编号范围
        start = self.prefix_NUM_VMS_PER_TYPE[task_type]
        end = self.prefix_NUM_VMS_PER_TYPE[task_type + 1]
        available_vm_ids = list(range(start, end))
        available_actions = [i - start for i in available_vm_ids]  # 动作编号从0开始

        if random.random() < epsilon:
            # 随机选一个可用动作
            action = random.choice(available_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
            # 只在有效可用动作中选最大Q值   可能这一步出错了
            action = available_actions[np.argmax(q_values[available_actions])]
        # 返回全局虚拟机编号
        # return start + action
        # 返回局部虚拟机编号
        return action

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
