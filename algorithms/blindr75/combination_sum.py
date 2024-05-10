"""Given an array of distinct integers candidates and a target integer target, 
return a list of all unique combinations of candidates where the chosen numbers sum to target. 
You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the 
frequency
 of at least one of the chosen numbers is different.

The test cases are generated such that the number of unique combinations that sum up to target is less than 150 combinations for the given input.
"""

def combination_sum_dfs(candidates, target, res):
    if target  == 0:
        return [res]
    
    if len(candidates) == 0:
        return []
    
    if target < 0:
        return []
    
    all_res = []
    for i in range(len(candidates)):
        result = combination_sum_dfs(candidates[i:], target = target-candidates[i], res= res+[candidates[i]])
        all_res.extend(result)
    return all_res

def combinationSum(candidates, target):
    return combination_sum_dfs(candidates, target, res=[])

candidates = [2,3,6,7]
target = 7
res1 = combinationSum(candidates=candidates, target=target)
    
def combinationSum(candidates , target):
    dp = [[] for _ in range(target+1)]
    dp[0] = [[]]
    for candidate in candidates:
        for i in range(candidate, target+1):
            for combination in dp[i - candidate]:
                dp[i].append(combination + [candidate])
    return dp[target]


candidates = [2,3,6,7]
target = 7
res2 = combinationSum(candidates=candidates, target=target) 


def combinationSum(candidates, target, res):
    if len(candidates) == 0:
        return []
    if target == 0:
        return [res]
    all_results = []
    for i in range(len(candidates)):
        res = combinationSum(candidates = candidates[i:],
                             target = target - candidates[i],
                             res = res +[candidates[i]])
        res.extend(res)
    return all_results

res3 = combinationSum(candidates=candidates, target=target) 


res1,res2,res3