"""
Given an integer array nums, return the length of the longest strictly increasing 
subsequence
"""

from typing import List
def lengthOfLIS(nums:List):
    dp = [1] * len(nums)
    for i in range(len(dp)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j]+1)
    return max(dp) # not dp[-1]
    
    
# def lengthOfLIS_dfs(nums, index):
#     if index == len(nums)-1:
#         return 0
#     for i in range(len(nums)):
        
        
    
# O(N^2) 1+2+3+5+n == n**2
nums = [10,9,2,5,3,7,101,18]
lengthOfLIS(nums=nums)

def all_possible_subsequence(nums, index, subseq):
    if index == len(nums):
        return [subseq]

    include_current = all_possible_subsequence(nums=nums, index=index+1, subseq=subseq + nums[index])
    exclude_current = all_possible_subsequence(nums=nums, index=index+1, subseq = subseq)
    return include_current + exclude_current

nums = ['5','3','2']
# string = "abc"
all_possible_subsequence(nums=nums, index=0, subseq="")



def all_possible_subsequence(nums, res=''):
    if len(nums) == 0:
        return [res]
    
    all_res =[]
    for i in range(len(nums)):
        sub = all_possible_subsequence(nums[:i] + nums[i+1:], res=res+nums[i])
        all_res.extend(sub)
        
    return  all_res # list of list

nums = ['5','3','2']
# string = "abc"
all_possible_subsequence(nums=nums, res="")
