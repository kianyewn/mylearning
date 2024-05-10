"""You are given an integer array nums.
You are initially positioned at the array's first index,
and each element in the array represents your maximum jump length at that position.

Return true if you can reach the last index, or false otherwise.

Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
"""


# def canJump(nums):
#     pass

# def canJump_dfs(nums, index):
#     if index >= len(nums):
#         return True
    
#     if nums[index] == 0:
#         return False
    
#     for num in nums[index+1:]:
#         can_jump = canJump_dfs(nums, index+num)
#         if can_jump:
#             return True
#     return False
       

def canJump_dfs(nums, index, memo):
    if index in memo:
        return memo[index]
        
    if len(nums) == 1:
        return True

    if index >= len(nums)-1:
        return True

    for step in range(1,nums[index]+1):
        can_jump = canJump_dfs(nums, index+step, memo=memo)
        if can_jump:
            return True
    can_jump = False
    memo[index] = can_jump
    return can_jump

def canJump(nums):
    return canJump_dfs(nums, index=0, memo={})
   
# def canJump(nums):
#     goal = len(nums)-1
#     for i in range(len(nums)):
#         goal = goal - nums[i]
#         if goal <=0:
#             return True
#     return False
        
def canJump(nums):
    goal = len(nums)-1
    for i in range(len(nums)-1,-1,-1):
        if i + nums[i] >= goal:
            goal = i
    return True if goal == 0 else False
        
nums = [2,3,1,1,4] 
assert canJump_dfs(nums, index=0, memo={}) == True
nums = [3,2,1,0,4]
assert canJump_dfs(nums, index=0, memo={}) == False


# def can_jump(nums):
#     goal = len(nums)-1
#     for i in range(len(nums)-1, -1, -1):
#         if i + nums[i] <= goal:
#             goal = i
#     return True if goal == 0 else False




########################
### LEETCODE SOLUTION ###
########################
# class Solution:
#     def canJump(self, nums: List[int]) -> bool:
#         # maximum distance travelled
#         dp = nums
#         # dp[-1] = nums[-1]
#         # O (N^N)
#         for i in range(len(dp)-1, -1, -1):
#             for step_size in range(1,nums[i]+1):
#                 if i + step_size < len(dp):
#                     dp[i] = max(dp[i], dp[i+step_size])
#                 dp[i] = max(dp[i],i + step_size)
#         return dp[0] >= len(nums)-1

#     def canJump(self, nums):
#         """
#         This is wrong because if you think about the dfs, it does
#         not track what is the current index
#         """
#         dp = [0] * len(nums)
#         dp[-1] = True
#         for i in range(len(dp)-2, -1, -1):
#             for step_size in range(1, nums[i]+1):
#                 if i+step_size < len(dp):
#                     dp[i] = max(dp[i+step_size], dp[i])
#                 else:
#                     dp[i] = True
#         return dp[0]

#     def canJump(self, nums):
#         goal = len(nums)-1
#         for i in range(len(nums)-1, -1,-1):
#             if i + nums[i] >= goal:
#                 goal = i
#         return True if goal == 0 else False

#     def canJump(self, nums: List[int]) -> bool:
#         # maximum distance travelled
#         dp = nums
#         # dp[-1] = nums[-1]
#         # O (N^N)
#         for i in range(len(dp)-1, -1, -1):
#             max_size = 0
#             for step_size in range(1,nums[i]+1):
#                 if i + step_size < len(dp):
#                     max_size = max(max_size, nums[i+step_size])
#             dp[i] = max_size + nums[i]
#         return dp[0]

#     def canJump(self, nums):
#         goal = len(nums)-1
#         for i in range(len(nums)-1,-1,-1):
#             if i + nums[i] >= goal:
#                 goal = i
#         return True if goal == 0 else False

#     # def canJump(self, nums):
#     #     dp = [0] * len(nums)
#     #     dp[-1] = nums[-1]
#     #     for i in range(len(dp)-2, -1, -1):
#     #         for step_size in range(nums[i]):
#     #             dp[i] = max(dp[i], i+dp[i+step_size])

#     #     return dp[0] >= len(nums)-1


#         # for i in range(nums):
#         #     for step_size in nums[i]:
#         #         dp[i] = max(dp[i], i+step_size)
        