from typing import List

"""
You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
"""

def climbStairs(n:int):
    dp = [0] * (n+1)
    dp[-1] = 1 # 1 way to reach n if it is n
    for i in range(len(dp)-2,-1,-1):
        if i + 1 < len(dp):
            dp[i] += dp[i+1]
        if i+2 < len(dp):
            dp[i] += dp[i+2]
    return dp[0]

def climbStairs_dfs(n):
    if n == 0:
        return 1
    if n < 0:
        return 0
    
    one_step = climbStairs_dfs(n-1)
    two_step = climbStairs_dfs(n-2)
    return one_step + two_step

memo = {}
def climbStairs_dfs_memo(n):
    if n == 0:
        return 1
    if n < 0:
        return 0
    if n in memo:
         return memo[n]
    
    one_step = climbStairs_dfs(n-1)
    two_step = climbStairs_dfs(n-2)
    sol = one_step + two_step
    memo[n] = sol
    return sol

# O(N) timecomplexity, one loop, O(N) space complexity due to dp Array
n = 3
assert climbStairs(n) == 3

# O(2**N) time complexity, each call branches out to two branches. O(N) space complexity because max call stack is N depth of tree, O(N)
climbStairs_dfs(n)
climbStairs_dfs_memo(n)

n = 2
assert climbStairs(n=2) == 2
    