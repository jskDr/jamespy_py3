"""
Tree algorithm implemented by Python
"""


class Tree():
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

levelq = []
def levelorder_traverse(tree):
    if tree is not None:
        levelq.append(tree)
    while len(levelq):
        t = levelq.pop(0)
        print(t.data)
        if t.left:
            levelq.append(t.left)
        if t.right:
            levelq.append(t.right)

def init_tree():
    leaf = []
    for i in range(6):
        leaf.append(Tree(i+1))
    
    left_subtree = Tree(9, Tree(7,leaf[0],leaf[1]), Tree(8,leaf[2],leaf[3]))
    right_subtree = Tree(10, leaf[4], leaf[5])
    
    root = Tree(11, left_subtree, right_subtree)
    levelorder_traverse(root)
