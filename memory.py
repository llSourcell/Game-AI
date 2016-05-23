import random as rand
from collections import deque


class Memory:

    def __init__(self, size, batch_size):
        self.batch_size = batch_size
        self.memory = deque(maxlen=size)

    def __len__(self):
        return len(self.memory)
        
    def add(self, state, action, reward, next_state, terminal):
        if len(self.memory) >= self.memory.maxlen:
            self.memory.popleft()
        self.memory.append( (state, action, reward, next_state, terminal) )

    def getSample(self):
        return rand.sample(self.memory, self.batch_size)

    def reset(self):
        self.memory.clear()
