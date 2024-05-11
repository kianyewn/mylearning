"""
[Medium]
Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:

The left 
subtree
 of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.

root = [5,4,6,null,null,3,7]
"""


def isValidBST(root):
    def isValid(root):
        if root is None:
            left_max = float('-inf')
            right_min = float('inf')
            return left_max, right_min, True
        
        left_max, left_min, is_valid_left = isValid(root.left)
        right_max, right_min, is_valid_right = isValid(root.right)
        
        if not is_valid_left or not is_valid_right or root.val <= left_max or root.val >= right_min:
            return 0, 0, False
        
        # return max(left_max, root.val), min(right_min, root.val), True
        # find the minimum in the tree rooted at root. So need to use right_max and left_min.
        return max(right_max, root.val), min(left_min, root.val), True
    
    _, _, valid = isValid(root)
    return valid


class Node:
    def __init__(self, val, left=None,right=None):
        self.val = val
        self.left = left
        self.right = right
        
        
root = Node(val=2)
root.left = Node(val=1)
root.right = Node(val=3)

assert isValidBST(root) == True

root = Node(val=2)
root.left = Node(val=5)
root.right = Node(val=3)
assert isValidBST(root) == False

    
    