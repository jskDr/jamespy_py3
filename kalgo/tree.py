"""
Tree algorithm implemented by Python
"""


class tree():
    def __init__(self, data, left, right):
        self.data = data
        self.left = left
        self.right = right

levelq = []
def levelorder_traverse(tree):
    leveq.append(tree)
    while tree:
        t = levelq
