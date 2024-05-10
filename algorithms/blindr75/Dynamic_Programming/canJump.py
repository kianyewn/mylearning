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