import numpy as np
import random
import itertools
from collections import defaultdict, deque

EPSILON = 0.2 # 探索率
a = 0.4 
b = 0.5

# 环境参数
NUM_TASK_TYPES = 3  # 任务类型数量
NUM_VMS_PER_TYPE = 3 # 每种任务类型有3台虚拟机 
# NUM_VMS_PER_TYPE = [3,3,3]  # 每种任务类型有3台虚拟机
TASK_CONFIG = {  # 任务类型预定义参数  需求30%是为了使得离散值都能覆盖到
    0: {"demand": 30, "duration": 5},  # 类型0: 需求10%，持续5步长
    1: {"demand": 30, "duration": 6},  # 类型1: 需求10%，持续6步长
    2: {"demand": 30, "duration": 7},  # 类型2: 需求10%，持续7步长
}
VM_CAPACITY = 100  # 虚拟机容量（100%）
ENTITY_CAPACITY = 200  # 实体机容量（200%）

class CloudEnv:
    def __init__(self):
        # 虚拟机负载（百分比），格式：{虚拟机ID: 当前总负载}
        self.vm_load = np.zeros(NUM_TASK_TYPES * NUM_VMS_PER_TYPE, dtype=float)
        # self.vm_load = np.zeros(sum(NUM_VMS_PER_TYPE), dtype=float)
        
        # 虚拟机到实体机的映射（假设虚拟机i部署在实体机i%3）
        self.vm_to_entity = [i % NUM_VMS_PER_TYPE for i in range(NUM_TASK_TYPES * NUM_VMS_PER_TYPE)]
        
        # 任务队列：记录每个虚拟机中正在执行的任务（剩余步长, 负载）
        self.task_queues = [deque() for _ in range(NUM_TASK_TYPES * NUM_VMS_PER_TYPE)]
        # self.task_queues = [deque() for _ in range(sum(NUM_VMS_PER_TYPE))]
        
    def _get_vm_level(self, load): #离散等级要尽可能细  不然会造成 如load < 30内的负载不均衡 因为它会根据离散等级1的Q表从而分配给一个虚拟机
        # 将负载百分比转换为离散等级（1/2/3.../10）
        if load < 30:
            return 1
        elif 30 <= load < 60:
            return 2
        else:
            return 3
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
        #  构建状态：任务类型 + 所有虚拟机负载等级 所有虚拟机的负载等级（按任务类型分组）
        vm_levels = [self._get_vm_level(self.vm_load[i]) 
                    for i in range(NUM_TASK_TYPES * NUM_VMS_PER_TYPE)]
        return (task_type,) + tuple(vm_levels)
    
    def get_all_states(self):
        """
        枚举所有可能的状态：(task_type, vm_level_0, ..., vm_level_8)
        task_type: 0~NUM_TASK_TYPES-1
        vm_level: 1~10
        """
        all_states = []
        vm_level_range = list(range(1, 4))  # 1~4
        for task_type in range(NUM_TASK_TYPES):
            # 9台虚拟机，每台有4种level
            for vm_levels in itertools.product(vm_level_range, repeat=NUM_TASK_TYPES * NUM_VMS_PER_TYPE):
                state = (task_type,) + vm_levels
                all_states.append(state)
        return all_states

    def set_state(self, state):
        """
        将环境设置为指定状态（仅设置vm_load，不处理队列）
        state: (task_type, vm_level_0, ..., vm_level_8)
        """
        # 只设置虚拟机负载，task_type不需要设置
        for i in range(NUM_TASK_TYPES * NUM_VMS_PER_TYPE):
            level = state[i + 1]
            # 反推百分比区间的中值
            if level == 1:
                percent = 15   # (0+30)/2
            elif level == 2:
                percent = 45   # (30+60)/2
            else:  # level == 3
                percent = 80   # (60+100)/2，假设最大100%
            self.vm_load[i] = percent / 100 * VM_CAPACITY
        # 清空任务队列（注意：此处未恢复任务队列，仅用于Q表学习）
        for q in self.task_queues:
            q.clear()
    
    def step(self, task_type, vm_id):
        #  执行动作：分配任务到虚拟机，并处理任务队列
        #  处理任务队列（减少剩余步长）
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
        
        #添加新任务到队列
        task_demand = TASK_CONFIG[task_type]["demand"]
        task_duration = TASK_CONFIG[task_type]["duration"]
        
        # 检查虚拟机容量是否足够
        if self.vm_load[vm_id] + task_demand > VM_CAPACITY:
            reward = -10  # 直接拒绝任务的惩罚
            return reward, False
        
        # 更新虚拟机负载
        self.vm_load[vm_id] += task_demand
        self.task_queues[vm_id].append((task_duration, task_demand))

        # 计算奖励
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
        return reward, False

    def reset(self):
        self.vm_load.fill(0)
        for q in self.task_queues:
            q.clear()

class QLearningAgent:
    def __init__(self):
        # Q表：状态 -> 每个动作的Q值（动作空间为3，每个任务类型只能选对应3个虚拟机）
        self.q_table = defaultdict(lambda: np.full(NUM_VMS_PER_TYPE,-10000.0))
    
    def choose_action(self, state, available_actions , EPSILON=EPSILON):
        if random.random() < EPSILON:
            return random.choice(available_actions)
        else:
            return np.argmax(self.q_table[state])



