"""You are a professional robber planning to rob houses along a street.
Each house has a certain amount of money stashed. All houses at this place are arranged in a circle.
That means the first house is the neighbor of the last one.
Meanwhile, adjacent houses have a security system connected,
and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, 
return the maximum amount of money you can rob tonight without alerting the police.

Input: nums = [2,3,2]
Output: 3
"""

def rob(nums):
    dp = [0] * (len(nums)+2)
    for i in range(2, len(dp)):
        dp[i] = max(dp[i-1], nums[i-2] + dp[i-2])
    return dp[-1]


def rob2(nums):
    if len(nums) == 1:
        return nums[0]
    rob_first = rob(nums[:-1])
    rob_last = rob(nums[1:])
    return max(rob_first, rob_last)

nums = [2,3,2]
rob2(nums)