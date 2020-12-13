

import numpy as np


class Compose:
    def __init__(self, ops):
        self.ops = ops

    def __call__(self, *args):
        for op in self.ops:
            args = op(*args)
        return args


class RandomSelect:
    def __init__(self, ops):
        self.ops = ops

    def __call__(self, *args):
        op = np.random.choice(self.ops)
        return op(*args)
