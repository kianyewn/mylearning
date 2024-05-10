"""
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

"""
from typing import List
def coinChange(coins: List, amount:int) -> int:
    dp = [float('inf')] * (amount+1)
    dp[-1] = 0
    for i in range(len(dp)-1, -1, -1):
        for coin in coins:
            if i+ coin < len(dp):
                dp[i] = min(dp[i], 1+dp[i+coin])
    return dp[0]
        
coins = [1,2,5]
amount = 11
# min 5,5,1
assert coinChange(coins=coins, amount=amount) == 3

def coinChange_dfs(coins, amount):
    if amount == 0:
        return 0
    if amount < 0:
        return float('inf')
    
    minimum = float('inf')
    for coin in coins:
        num_coins = coinChange_dfs(coins=coins, amount=amount-coin)
        minimum = min(minimum, num_coins+1)
    return minimum

def coinChange_dfs_res(coins, amount):
    res = coinChange_dfs(coins=coins, amount=amount)
    return res if res != float('inf') else -1

coins = [1,2,5]
amount = 11
# min 5,5,1
# coinChange_dfs(coins=coins, amount=amount)

coins = [2]
amount = 3
coinChange_dfs_res(coins=coins, amount=amount)

    

