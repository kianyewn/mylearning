"""
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.
"""

def twoSum(nums, target):
    remainder = {}
    for idx, num in enumerate(nums):
        if num in remainder:
            return (idx, remainder[num])
        remaining = target - num
        if not remaining in remainder:
            remainder[remaining] = idx
    return False
        

nums = [2,7,11,15]
target = 9
twoSum(nums=nums, target=9)