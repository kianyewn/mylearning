"""You are a professional robber planning to rob houses along a street. 
Each house has a certain amount of money stashed,
the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected 
and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house,
return the maximum amount of money you can rob tonight without alerting the police.

Input: nums = [1,2,3,1]
Output: 4
"""

def rob(nums):
    dp = [0] * (len(nums)+2)
    for i in range(2, len(dp)):
        dp[i] = max(dp[i-1], nums[i-2] + dp[i-2])
    return dp[-1]

nums = [1,2,3,1]
rob(nums)


    # def dfs(self, candidates, cur_sum):
    #     """
    #     This solution will not work if len(candidates) < 2:
    #     return is because when you create the new candidates, it
    #     requires i+2 to trigger the 
    #     """
    #     if len(candidates)==0:
    #         return cur_sum

    #     max_res = 0
    #     for i  in range(len(candidates)):
    #         if i == 0 or i == 1:
    #             # a = [1,2,3,4], a[10:] = []
    #             new_candidates = candidates[i+2:]
    #         else:
    #             new_candidates = candidates[:i-1]
    #             if i+2 < len(candidates):
    #                 new_candidates=new_candidates+candidates[i+2:]

    #         res = self.dfs(new_candidates, cur_sum=cur_sum+candidates[i])
    #         max_res = max(res, max_res)
    #     return max_res
        