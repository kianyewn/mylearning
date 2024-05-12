"""100. Same Tree
[Easy]
Given the roots of two binary trees p and q, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value

Input: p = [1,2,3], q = [1,2,3]
Output: true
"""
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def isSameTree(p, q ) -> bool:
    if (p is None and q is not None) or (p is not None and q is None):
        return False

    if p is None and q is None:
        return True
        
    if p.val != q.val:
        return False

    left_true = isSameTree(p.left, q.left)
    right_true = isSameTree(p.right, q.right)
    return left_true and right_true

    
p_root = Node(1)
p_root.left = Node(2)
p_root.right = Node(3)

q_root = Node(1)
q_root.left = Node(2)
q_root.right = Node(3)

assert isSameTree(p_root,q_root) == True