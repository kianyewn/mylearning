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

#### Exercise  #####

coins = [1,2,5]
amount = 11


def dfs(coins, amount, num_coins):
    if amount == 0:
        return num_coins
    if amount < 0:
        return float('inf')
    min_coins = float('inf')
    for i in range(len(coins)):
        candidate_min_coins = dfs(coins, amount=amount-coins[i], num_coins= num_coins +1)
        min_coins = min(min_coins, candidate_min_coins)
    return min_coins

def coinChange2(coins, amount):
    return dfs(coins, amount, num_coins=0)

coinChange2(coins=coins, amount=2)

def dfs2(coins, amount, memo={}):
    """Start bottom up and think about the dfs process
    
    - time complexity is O(amount) after memo, because just have to iterate through each amount once.
    - Without memo, it will be O(numcoins**(amount)) . 
    -- imagine if we have a tree of height 2, 
    -- we need to split by 2 branches, then each of the two branch will have another 2 branch, thus 2**(h=2).
    """
    if amount == 0:
        return 0
    
    if amount < 0:
        return float('inf')
    if amount in memo:
        return memo[amount]
    
    min_coins = float('inf')
    for candidate_idx in range(len(coins)):
        candidate_min_coins = dfs2(coins, amount=amount-coins[candidate_idx]) + 1
        min_coins = min(min_coins, candidate_min_coins)
    memo[amount] = min_coins    
    return min_coins

dfs2(coins=coins, amount=128, memo={}) # 3, Correct


def coin_change_dp(coins, amount):
    """Imagine the DFS tree. When you reached `amount`, you will be using one coin
    """
    # Wrong explanation. amount must plus 1. Reason is because to reach amount, it is already 1 coin. (handled by +1 in dp)
    # Reason is because index starts from 0. So when e.g. at i =5, coin = 5, the amount is actually 10, not 11.
    dp = [float('inf')] * (amount+1) 
    # if dp is only * (amount) your solution will have one less coin
    dp[-1] = 0
    for i in range(len(dp)-2, -1, -1):
        for coin in coins:
            if i + coin < len(dp):
                dp[i] = min(dp[i], dp[i+coin] + 1)
    return dp[0]

coin_change_dp(coins=coins, amount=11)
                
                

