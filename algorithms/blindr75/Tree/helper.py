from typing import List
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
def level_order_traversal(root:Node)->List:
    res = []
    queue = [root]
    while len(queue)>0:
        curr = queue.pop(0)
        if curr:
            res.append(curr.val)
            queue.append(curr.left)
            queue.append(curr.right)
    return res

def level_order_traversal_v2(root):
    res = []
    def dfs(root):
        if root is None:
            return
        res.append(root.val)
        dfs(root.left)
        dfs(root.right)
    return res

    