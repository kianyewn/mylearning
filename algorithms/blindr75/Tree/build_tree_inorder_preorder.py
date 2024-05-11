"""
[MEDIUM]
Given two integer arrays preorder and inorder 
where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, 
construct and return the binary tree.

Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]


"""
from algorithms.blindr75.Tree.helper import level_order_traversal

class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
def buildTree(preorder, inorder):
    if not preorder or not inorder:
        return None
    
    # Root is always the first element of the preorder traversal tree
    root = TreeNode(val=preorder[0])
    # Find index of root in the inorder list. Every elements left of the root will be in the left sub-tree
    # , elements right of root will be in the right sub-tree 
    inorder_index = inorder.index(preorder[0])
    
    # Build the left subtree
    # index starts as 1 because we already created the root using preorder[0]
    # preorder ends at :inorder_index+1, because these are all the elements that will be in the left subtree as dictated by
    # the property of the inorder traversal.
    # inorder[:inorder_index], exclude the current index, because we already got the root value.
    # inorder is used only to find the elements that should be in the left sub-tree
    root.left = buildTree(preorder=preorder[1:inorder_index+1],
                          inorder=inorder[:inorder_index])
    root.right = buildTree(preorder=preorder[inorder_index+1:],
                           inorder=inorder[inorder_index+1:])
    return root

    
preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]

tree = buildTree(preorder=preorder,
          inorder=inorder)

assert level_order_traversal(tree) == [3, 9, 20, 15, 7]
# Output: [3,9,20,null,null,15,7]
