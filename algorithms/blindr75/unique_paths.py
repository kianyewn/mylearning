"""
There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]).
The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). 
The robot can only move either down or right at any point in time.

Given the two integers m and n,
return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The test cases are generated so that the answer will be less than or equal to 2 * 109.

Input: m = 3, n = 7
Output: 28
"""

def uniquePaths(m, n):
    dp = [[0] * (n+1) for _ in range(m+1)]
    dp[-2][-2] = 1
    # print(dp)

    # dp = [[0] * n for _ in range(m)]
    # dp[-1][-1] = 1

    for row in range(len(dp)-2, -1, -1):
        for col in range(len(dp[0])-2, -1, -1):
            move_right = dp[row][col+1]
            move_down = dp[row+1][col]
            dp[row][col] = max(dp[row][col], move_right + move_down)
            # if row+1 < m:
            #     dp[row][col] = dp[row+1][col]
            # if col+1 < n:
            #     dp[row][col] += dp[row][col+1]
    return dp[0][0]

m = 3
n = 7
# Output: 28

uniquePaths(m=m, n=n)