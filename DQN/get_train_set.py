import random

NUM_TASK_TYPES = 3
SEQ_LEN = 30  # 每个序列长度
NUM_SEQS = 1000  # 生成序列数量

""" 这样生成的 train_task_set.txt 文件，
每行一个任务序列，既有均匀分布，也有极端情况，
能较好覆盖状态空间，提升神经网络泛化能力。 """

with open("train_task_set.txt", "w") as f:
    # 均匀采样
    for _ in range(NUM_SEQS // 2):
        seq = [random.randint(0, NUM_TASK_TYPES - 1) for _ in range(SEQ_LEN)]
        f.write(" ".join(map(str, seq)) + "\n")
    # 极端采样
    for _ in range(NUM_SEQS // 6):
        seq = [0] * SEQ_LEN
        f.write(" ".join(map(str, seq)) + "\n")
        seq = [1] * SEQ_LEN
        f.write(" ".join(map(str, seq)) + "\n")
        seq = [2] * SEQ_LEN
        f.write(" ".join(map(str, seq)) + "\n")
    # 交替采样
    for _ in range(NUM_SEQS // 6):
        seq = [i % NUM_TASK_TYPES for i in range(SEQ_LEN)]
        f.write(" ".join(map(str, seq)) + "\n")
        seq = [(2 - i % NUM_TASK_TYPES) for i in range(SEQ_LEN)]
        f.write(" ".join(map(str, seq)) + "\n")