"""104. Maximum Depth of Binary Tree
[Easy]
Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

input = [3,9,20,null,null,15,7]
output = 3
"""
from typing import List
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def maxDepth(root: Node) -> int:
    def dfs(root, cur_depth):
        if root is None:
            return cur_depth

        depth_left = dfs(root.left, cur_depth+1)
        depth_right = dfs(root.right, cur_depth+1)
        return max(depth_left, depth_right)

    # cur depth initialized to 0, because root.left = None is +1
    return dfs(root, cur_depth=0)

#  [3,9,20,null,null,15,7]
three = Node(3)
nine = Node(9)
twenty = Node(20)
fifteen = Node(15)
seven = Node(7)

three.left = nine
three.right = twenty
twenty.left = fifteen
twenty.right= seven

maxDepth(three)
