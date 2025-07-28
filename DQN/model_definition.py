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
DISCOUNT_FACTOR = 0.6
BATCH_SIZE = 64
MEMORY_SIZE = 50000
TARGET_UPDATE_FREQ = 100
EPSILON = 0.2 # 探索率
a = 0.4 
b = 0.5
overload = 50000

# 环境参数
NUM_TASK_TYPES = 7  # 应用类型数量
NUM_VMS_PER_TYPE = [2,2,3,4,2,4,3]  # 每种应用类型有多少台虚拟机 VMS_PER_TYPE = [0,0,1,1,1,1,2,2,3,3,3]
VMS_PER_TYPE = [] # 每台虚拟机到应用类型的映射
for i in range(NUM_TASK_TYPES):
    for j in range(NUM_VMS_PER_TYPE[i]):
        VMS_PER_TYPE.append(i)
# print("VMS_PER_TYPE:", VMS_PER_TYPE)
NUM_PM = 7  # 实体机数量
TASK_CONFIG = {  # 不同应用类型的任务预定义参数  需求10%是为了使得离散值都能覆盖到 训练的时候可以把duration拉长以覆盖更多，实际用的时候用实际值
    0: {"demand": 1, "duration": 80},  # 应用类型0: cpu需求量1%，持续80步长
    1: {"demand": 2, "duration": 70},  # 应用类型1: cpu需求量2%，持续70步长
    2: {"demand": 3, "duration": 60},  # 应用类型2: cpu需求量3%，持续60步长
    3: {"demand": 5, "duration": 50},  # 应用类型3: cpu需求量5%，持续50步长
    4: {"demand": 3, "duration": 40},  # 应用类型4: cpu需求量5%，持续40步长
    5: {"demand": 4, "duration": 60},  # 应用类型5: cpu需求量5%，持续60步长
    6: {"demand": 5, "duration": 40},  # 应用类型6: cpu需求量5%，持续40步长
    7: {"demand": 9, "duration": 5},  # 应用类型7:
}
VM_CAPACITY = [100,120,150,150,150,150,150]  # 虚拟机容量，执行不同应用类型任务的虚拟机资源容量
PM_CAPACITY = 300  # 实体机容量（300%）

def env_params_reset(num_pm=None, num_task_types=None, num_vms_per_type=None, task_config=None, vm_capacity=None, pm_capacity=None):
    """
    重置环境参数
    :param num_pm: 实体机数量
    :param num_task_types: 任务类型数量
    :param num_vms_per_type: 每种任务类型的虚拟机数量
    :param task_config: 任务配置字典，包含每种任务类型的需求和持续时间
    :param vm_capacity: 虚拟机容量列表
    :param entity_capacity: 实体机容量
    """
    global NUM_PM, NUM_TASK_TYPES, NUM_VMS_PER_TYPE, VMS_PER_TYPE, VM_CAPACITY, PM_CAPACITY ,TASK_CONFIG
    if num_pm is not None:
        NUM_PM = num_pm
    if num_task_types is not None:
        NUM_TASK_TYPES = num_task_types
    if num_vms_per_type is not None:
        NUM_VMS_PER_TYPE = num_vms_per_type
    VMS_PER_TYPE = []
    for i in range(NUM_TASK_TYPES):
        for j in range(NUM_VMS_PER_TYPE[i]):
            VMS_PER_TYPE.append(i)
    if vm_capacity is not None:
        VM_CAPACITY = vm_capacity
    if pm_capacity is not None:
        PM_CAPACITY = pm_capacity
    # 更新任务配置
    if task_config is not None:
        TASK_CONFIG = {}
        for i in range(NUM_TASK_TYPES):
            TASK_CONFIG[i] = {
                "demand": task_config[i]["demand"],
                "duration": task_config[i]["duration"]
            }
    print("环境参数已重置:")
    print(f"实体机数量: {NUM_PM}, 任务类型数量: {NUM_TASK_TYPES}, 每种任务类型虚拟机数量: {NUM_VMS_PER_TYPE}, 虚拟机映射任务类型: {VMS_PER_TYPE}")
    print(f"虚拟机容量: {VM_CAPACITY}, 实体机容量: {PM_CAPACITY}")

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
        self.vm_load = np.zeros(len(VMS_PER_TYPE), dtype=float) #sum(NUM_VMS_PER_TYPE) self.prefix_NUM_VMS_PER_TYPE[-1] 都是一样的含义
        # 虚拟机到实体机的映射  todo:到时候需要动态表示
        self.vm_to_pm = []  # 每个虚拟机对应的实体机ID 假设虚拟机i平铺部署在实体机上
        for i in range(len(VMS_PER_TYPE)): # sum(NUM_VMS_PER_TYPE) 
            self.vm_to_pm.append(i%NUM_PM) 
        print("vm_to_pm:", self.vm_to_pm) # self.vm_to_pm = [0,1,2,0,1,2,0,1,2,0,1] 
        # 任务队列：记录每个虚拟机中正在执行的任务（剩余步长, 负载）
        self.task_queues = [deque() for _ in range(len(VMS_PER_TYPE))]  # sum(NUM_VMS_PER_TYPE)
        # 每种应用类型对应虚拟机数量前缀和
        self.prefix_NUM_VMS_PER_TYPE = prefix_sum(NUM_VMS_PER_TYPE)
        # 初始化轮询指针
        self.rr_pointer = [0 for _ in range(NUM_TASK_TYPES)]
        
    def _get_vm_level(self, load, vm_type):
        rate = load / VM_CAPACITY[vm_type] *100  # 获取对应虚拟机的容量比率
        # 将负载百分比转换为离散等级（1/2/3.../10）
        level = int(rate) + 1
        if level < 1:
            level = 1
        elif level > 100:
            level = 100
        return level
        # if rate < 10:
        #     return 1
        # elif 10 <= rate < 20:
        #     return 2
        # elif 20 <= load < 30:
        #     return 3
        # elif 30 <= load < 40:
        #     return 4
        # elif 40 <= load < 50:
        #     return 5
        # elif 50 <= load < 60:
        #     return 6
        # elif 60 <= load < 70:
        #     return 7
        # elif 70 <= load < 80:
        #     return 8
        # elif 80 <= load < 90:
        #     return 9
        # else:
        #     return 10
        
    def get_state(self, task_type):
        #  构建状态：应用类型 + 所有虚拟机负载等级 所有虚拟机的负载等级（按应用类型分组）
        vm_levels = [self._get_vm_level(self.vm_load[i], VMS_PER_TYPE[i]) 
                    for i in range(len(VMS_PER_TYPE))]  # sum(NUM_VMS_PER_TYPE)
        return (task_type,) + tuple(vm_levels)
    
    def get_all_states(self):  #基本没用于DQN训练,因为是状态空间太大，无法全部枚举和存储，就不能使用遍历所有状态的方法
        """
        枚举所有可能的状态：(task_type, vm_level_0, ..., vm_level_8)
        task_type: 0~NUM_TASK_TYPES-1
        vm_level: 1~10
        """
        all_states = []
        vm_level_range = list(range(1, 101))  # 1~11
        for task_type in range(NUM_TASK_TYPES):
            # 每台虚拟机有10种level 10的n次方个组合  所以不能用q-learning
            for vm_levels in itertools.product(vm_level_range, repeat=len(VMS_PER_TYPE)): # sum(NUM_VMS_PER_TYPE)
                state = (task_type,) + vm_levels
                all_states.append(state)
        return all_states

    def generate_random_state(self,low, up):
        """
        基于NUM_VMS_PER_TYPE随机生成一个状态
        返回格式: (task_type, vm_level_0, vm_level_1, ..., vm_level_n)
        """
        # 随机生成任务类型
        task_type = random.randint(0, NUM_TASK_TYPES - 1)
        # 随机生成每台虚拟机的等级
        state = [task_type]
        for i in range(len(VMS_PER_TYPE)):
            # 随机生成虚拟机负载等级（1-100级，对应0-100%负载）
            vm_level = random.randint(low, up)
            state.append(vm_level)
        return tuple(state)

    def set_state(self, state):
        """
        将环境设置为指定状态（仅设置vm_load，不处理队列）  也就是只用于检验某一步状态的效果
        state: (task_type, vm_level_0, ..., vm_level_3...)
        """
        # 只设置虚拟机负载，task_type不需要设置
        for i in range(len(VMS_PER_TYPE)):  # sum(NUM_VMS_PER_TYPE)
            # 反推负载百分比区间的均值
            vm_type = VMS_PER_TYPE[i]
            level = state[i + 1]
            # 反推百分比区间的中值  percent = (level * 区间宽度 - 区间宽度/2) 除了1之外
            self.vm_load[i] = (level * 1) / 100 * VM_CAPACITY[vm_type]
        # 清空任务队列（注意：此处未恢复任务队列，仅用于Q表学习）
        for q in self.task_queues:
            q.clear()
    
    def step(self, task_type, vm_id): 
        #  DQN学习过程中使用的step，将每个任务分配到指定虚拟机，并计算奖励
        #  执行动作：分配任务到虚拟机，并处理任务队列
        #  处理任务队列（减少剩余步长）
        released_load = 0
        for vm in range(len(VMS_PER_TYPE)):  # sum(NUM_VMS_PER_TYPE)
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
        same_type_vms = [self.vm_load[i] for i in range(len(VMS_PER_TYPE)) 
                       if VMS_PER_TYPE[i] == task_type]
        vm_var = np.var(same_type_vms)
        
        # 实体机负载方差
        entity_loads = []
        for e in range(NUM_PM):  # 实体机数量
            load = sum(
                self.vm_load[i] 
                for i in range(len(VMS_PER_TYPE))  # sum(NUM_VMS_PER_TYPE)
                if self.vm_to_pm[i] == e
            )
            entity_loads.append(load)
        entity_var = np.var(entity_loads)
        
        # 过载惩罚（任一实体机超载）
        overload_penalty = overload if any(l > PM_CAPACITY for l in entity_loads) else 0
        
        reward = -a * vm_var - b * entity_var - overload_penalty
        return reward, False
    
    def step_training(self, task_type ,agent): 
        #  DQN学习过程中使用的step，将每个任务分配到指定虚拟机，并计算奖励
        #  执行动作：分配任务到虚拟机，并处理任务队列
        #  处理任务队列（减少剩余步长）
        released_load = 0
        for vm in range(len(VMS_PER_TYPE)):  # sum(NUM_VMS_PER_TYPE)
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

        # 选择动作
        state = self.get_state(task_type)
        state = np.array(state, dtype=np.float32)
        action = agent.choose_action_multi(state)
        vm_id = self.prefix_NUM_VMS_PER_TYPE[int(task_type)] + action  # 虚拟机ID
        
        #添加新任务到队列
        task_demand = TASK_CONFIG[task_type]["demand"]
        task_duration = TASK_CONFIG[task_type]["duration"]
        
        # 检查虚拟机容量是否足够
        if self.vm_load[vm_id] + task_demand > VM_CAPACITY[VMS_PER_TYPE[vm_id]]:
            reward = -1 * overload  # 直接拒绝任务的惩罚
            return action,reward, False
        
        # 更新虚拟机负载并检查实体机负载是否足够，不足够也直接拒绝任务并给出惩罚
        self.vm_load[vm_id] += task_demand
        # 实体机负载方差
        pm_loads , pm_utilization, pm_var = self.get_pm_info()  # 获取实体机负载信息
        if any(l > PM_CAPACITY for l in pm_loads):
            self.vm_load[vm_id] -= task_demand
            reward = -1 * overload  # 实体机过载的惩罚
            return action, reward, False
        self.task_queues[vm_id].append((task_duration, task_demand))
        # 计算奖励
        # 同类型虚拟机的负载方差
        same_type_vms = [self.vm_load[i] for i in range(len(VMS_PER_TYPE)) 
                       if VMS_PER_TYPE[i] == task_type]
        vm_var = np.var(same_type_vms)
        
        reward = -a * vm_var - b * pm_var
        return action, reward, False
    
    def step_batch(self, task_types, agent, choose_function):
        """
        是检验DQN效果过程中的真实任务到来step,学习过程中是一个个一个任务来，而这里是批量到来，实际上学习过程是更加精细的，批量到来可以看成一个一个来并选择
        批量执行任务分配，不计算奖励，只更新负载和判断过载。
        若选择的虚拟机超载，尝试分配到同类型下不超载的虚拟机，并记录超载虚拟机。
        :param task_types: 任务类型数组，如 [0, 1, 2]
        :param agent:      DQNAgent 实例，用于获取动作
        :return: (vm_load, entity_loads, overload_flag, overload_vms)
        """
        # overload_vms = []  # 记录本次尝试分配时超载的虚拟机ID
        overload_nums = 0  # 记录违规次数

        #此处引出一个之前忽略的问题  就是我这个选择动作应该放到处理所有虚拟机的任务队列之后  我之前的学习都是放在之前就选完了  先看效果吧

        # 1. 处理所有虚拟机的任务队列（减少剩余步长，释放负载）  
        for vm in range(len(VMS_PER_TYPE)):  # sum(NUM_VMS_PER_TYPE)
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
                # overload_vms.append(vm_id)
                # 尝试分配到同类型下其他不超载的虚拟机
                vm_type = VMS_PER_TYPE[vm_id]
                # 找到所有同类型虚拟机的ID
                candidate_vms = [i for i in range(len(VMS_PER_TYPE)) if VMS_PER_TYPE[i] == vm_type]
                allocated = False
                for alt_vm in candidate_vms:
                    if self.vm_load[alt_vm] + task_demand <= VM_CAPACITY[vm_type]:
                        self.vm_load[alt_vm] += task_demand
                        self.task_queues[alt_vm].append((task_duration, task_demand))
                        allocated = True
                        # print(f"Task type {task_type} allocated to VM {alt_vm} instead of overloaded VM {vm_id}.")
                        break
                # 如果所有同类型虚拟机都超载，则该任务不分配
                if not allocated:
                    overload_nums += 1

   
        pm_loads,pm_utilization, pm_var = self.get_pm_info()  # 获取实体机负载信息

        # 3. 判断是否有实体机过载
        overload_flag = any(l > PM_CAPACITY for l in pm_loads)
        if overload_flag:
            # print("Overload detected! Entity machine load exceeds capacity.")
            for pm_id, load in enumerate(pm_loads):
                if load > PM_CAPACITY:
                    overload_nums += 1
                    # print(f"Entity machine {pm_id} overloaded with load {load}.")


        return overload_flag, overload_nums

    # 虚拟机负载，资源利用率以及负载方差(同种类型虚拟机)
    def get_vm_info(self):
        vm_utilization = []
        vm_var = []  # 每种应用类型虚拟机负载方差
        for i in range(len(self.vm_load)):
            vm_type = VMS_PER_TYPE[i]
            utilization = self.vm_load[i] / VM_CAPACITY[vm_type]
            vm_utilization.append(utilization)
        for task_type in range(NUM_TASK_TYPES):
            start = self.prefix_NUM_VMS_PER_TYPE[task_type]
            end = self.prefix_NUM_VMS_PER_TYPE[task_type+1]
            vm_var.append(np.var(self.vm_load[start:end]))
        return self.vm_load.copy(),vm_utilization,vm_var

    # 实体机负载，资源利用率以及负载方差
    def get_pm_info(self):
        pm_utilization = []
        pm_loads = []
        for pm_id in range(NUM_PM):
            # 统计该实体机上所有虚拟机的负载总和
            total_load = sum(
                self.vm_load[i]
                for i in range(len(self.vm_load))
                if self.vm_to_pm[i] == pm_id
            )
            utilization = total_load / PM_CAPACITY
            pm_utilization.append(utilization)
            pm_loads.append(total_load)
        pm_var = np.var(pm_loads)
        return pm_loads,pm_utilization, pm_var
        

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
            # max_next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            # target_q_values = rewards + (1 - dones) * DISCOUNT_FACTOR * max_next_q_values
            # 获取所有 next_states 的 Q 值
            all_next_q = self.target_net(next_states).cpu().numpy()  # shape: [batch_size, action_dim]
            max_next_q_values = []
            for idx, next_state in enumerate(next_states.cpu().numpy()):
                # 获取当前 next_state 可选动作编号列表
                task_type = int(next_state[0])
                start = self.prefix_NUM_VMS_PER_TYPE[task_type]
                end = self.prefix_NUM_VMS_PER_TYPE[task_type + 1]
                available_actions = [i - start for i in range(start, end)]
                # 只在可行动作中选最大 Q 值
                q_vals = all_next_q[idx][available_actions]
                max_next_q_values.append(np.max(q_vals))
            max_next_q_values = torch.FloatTensor(max_next_q_values).unsqueeze(1)
            target_q_values = rewards + (1 - dones) * DISCOUNT_FACTOR * max_next_q_values
            

        # 更新主网络
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
