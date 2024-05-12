"""226. Invert Binary Tree
[Easy]
Given the root of a binary tree, invert the tree, and return its root.
"""

class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
    def __repr__(self):
        return f'root.val: {root.val}, root.left:{root.left.val}, root.right: {root.right.val}'

def invertTree(self, root): 
    if root is None:
        return
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root

def invertTree(root):
    if root is None:
        return

    right_node = None
    left_node = None

    if root.right:
        right_node = Node(root.right.val, root.right.left, root.right.right)
    if root.left:
        left_node = Node(root.left.val, root.left.left, root.left.right)

    root.left = invertTree(right_node)
    root.right = invertTree(left_node)
    return root


root = Node(2)
root.left = Node(1)
root.right = Node(3)

new_root = invertTree(root)
print(new_root)