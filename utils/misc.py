

import os

from collections import defaultdict, OrderedDict


def containerize(x, n=1):
    return x if isinstance(x, (list, tuple)) else [x] * n


def convert_path(path):
    return path.replace(r'\/'.replace(os.sep, ''), os.sep)


class Node:
    __slots__ = 'key', 'val', 'cnt'

    def __init__(self, key, val, cnt=0):
        self.key, self.val, self.cnt = key, val, cnt


class LFUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # type {key: node}
        self.cnt2node = defaultdict(OrderedDict)
        self.mincnt = 0

    def get(self, key, default=None):
        if key not in self.cache:
            return default

        node = self.cache[key]
        del self.cnt2node[node.cnt][key]

        if not self.cnt2node[node.cnt]:
            del self.cnt2node[node.cnt]

        node.cnt += 1
        self.cnt2node[node.cnt][key] = node

        if not self.cnt2node[self.mincnt]:
            self.mincnt += 1
        return node.val

    def put(self, key, value):
        if key in self.cache:
            self.cache[key].val = value
            self.get(key)
            return
        if len(self.cache) >= self.capacity:
            pop_key, _pop_node = self.cnt2node[self.mincnt].popitem(last=False)
            del self.cache[pop_key]

        self.cache[key] = self.cnt2node[1][key] = Node(key, value, 1)
        self.mincnt = 1
