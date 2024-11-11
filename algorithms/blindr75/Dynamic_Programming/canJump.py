from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        # maximum distance travelled
        dp = [0] * (len(nums))
        dp[-1] = nums[-1]

        for i in range(len(dp)-2, -1, -1):
            for step_size in range(1,nums[i]+1):
                if i + step_size < len(dp):
                    dp[i] = max(dp[i], i + dp[i+step_size])
                dp[i] = max(dp[i], i + step_size)
        return dp[0] >= len(nums)-1
    
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        # maximum distance travelled
        dp = nums
        # dp[-1] = nums[-1]

        for i in range(len(dp)-1, -1, -1):
            for step_size in range(1,nums[i]+1):
                if i + step_size < len(dp):
                    dp[i] = max(dp[i], dp[i+step_size])
        
                dp[i] = max(dp[i], i + step_size)
        return dp[0] >= len(nums)-1
    
    def canJump(self, nums):
        """
        This is wrong because if you think about the dfs, it does
        not track what is the current index
        """
        dp = [0] * len(nums)
        dp[-1] = nums[-1]
        for i in range(len(dp)-2, -1, -1):
            for step_size in range(nums[i]):
                dp[i] = max(dp[i], dp[i+step_size])

        return dp[0] >= len(nums)-1
    
    
    def canJump(self, nums):
        """
        This is wrong because if you think about the dfs, it does
        not track what is the current index
        """
        dp = [0] * len(nums)
        dp[-1] = nums[-1]
        for i in range(len(dp)-2, -1, -1):
            for step_size in range(nums[i]):
                dp[i] = max(dp[i], i+dp[i+step_size])

        return dp[0] >= len(nums)-1
    
    def canJump(self, nums):
        """
        This is wrong because if you think about the dfs, it does
        not track what is the current index
        """
        dp = [0] * len(nums)
        dp[-1] = True
        for i in range(len(dp)-2, -1, -1):
            if dp[i+nums[i]]:
                dp[i] = True
            
        return dp[0]
    def canJump(self, nums: List[int]) -> bool:
        # maximum distance travelled
        dp = [0] * len(nums)
        dp[-1] = nums[-1]
        # O (N^N)
        for i in range(len(dp)-2, -1, -1):
            max_size = 0
            for step_size in range(1,nums[i]+1):
                if i + step_size < len(dp):
                    max_size = max(max_size, dp[i+step_size])
            dp[i] = i + max_size 
        return dp
    
nums = [2,3,1,1,4]
Solution().canJump(nums)
    
nums = [1,2]
# nums = [3,2,1,0,4]
# nums = [1,1,1,0]
Solution().canJump(nums)


# for i in range(1,1):
#     print(i)

def canJumpNaive(nums):
    # can reach dp
    dp = [0] * (len(nums))
    # dp[0] = 1 # what is num[0] is 0. then the solution should be false
    for i in range(len(nums)):
        for step in range(1, nums[i]+1):
            max_step = min(i+step, len(nums)-1)
            if dp[max_step] == 0:
                dp[max_step] = 1

            if i == 0 and nums[i] > 0:
                dp[i] = 1

    print(dp)
    return sum(dp) == len(nums)
    
    
def canJump(nums):
    goal = len(nums)-1
    for i in range(len(nums)-2, -1, -1):
        print(i, i + nums[i])
        if i + nums[i] >= goal:
            goal = i
    return goal == 0 
        
nums = [2,3,1,1,4]
canJump(nums)
canJumpNaive(nums)


nums = [2,0,0,1,4]
print(canJump(nums))
canJumpNaive(nums)

nums = [0,1,1,1,4]
print(canJump(nums))
canJumpNaive(nums)
