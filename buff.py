import numpy as np


class Buffer:

    def __init__(self, params):
        history_length = params.history_length
        width = params.width
        height = params.height
        self.dims = (width, height, history_length)
        self.buffer = np.zeros(self.dims, dtype=np.uint8)

    def add(self, state):
        self.buffer[:, :, :-1] = self.buffer[:, :, 1:]
        self.buffer[:, :, -1] = state

    def getInput(self):
        x = np.reshape(self.buffer, (1,) + self.dims)
        return x

    def getState(self):
        return self.buffer

    def reset(self):
        self.buffer.fill(0)