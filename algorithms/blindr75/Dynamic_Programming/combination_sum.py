"""Given an array of distinct integers candidates and a target integer target, 
return a list of all unique combinations of candidates where the chosen numbers sum to target. 
You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the 
frequency of at least one of the chosen numbers is different.

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
    
def combinationSum(candidates, target):
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

# res3 = combinationSum(candidates=candidates, target=target, res=[]) 

# res1,res2,res3


### Exercise ###
candidates = [2,3,6,7]
target = 7

def combinationSum_dfs_sol(candidates, target, res=[]):
    if target == 0:
        return [res] # list of list

    if target < 0:
        return []
    
    all_res = [] # contain a list of list
    for candidate_index in range(len(candidates)):
        i = candidate_index
        # include index i in next iteration, because qn says that the candidates can be repeated
        # the frequency will be unique, because we know that the candidates are unique
        # if it is not unique, we can sort the candidates, and then if current candidate is the same as 
        # previous candidate, we can continue.
        possible_res = combinationSum_dfs_sol(candidates=candidates[i:], 
                                              target=target-candidates[i],
                                              res=res+[candidates[i]])
        all_res.extend(possible_res)
    return all_res

assert combinationSum_dfs_sol(candidates=candidates, 
                              target=target,
                              res=[]) == [[2, 2, 3], [7]] # correct

def combinationsSum_dp(candidates, target):
    """Target = 10, = 2 + 2 + 2 + 2 + 2 = 2 + 4 + 4"""
    dp = [[] for _ in range(target + 1)] ## THIS IS VERY IMPORTANT, IF YOU MISSED THIS
    print(dp)
    dp[0] = [[]] # empty candidates
    for i in range(1, len(dp)):
        for cand in candidates:
            if i - cand >= 0 and len(dp[i-cand])>0:
                prev_solutions = dp[i-cand]
                cur_solutions = []
                for sol in prev_solutions:
                    new_sol = sol + [cand]
                    cur_solutions.append(new_sol)
                dp[i].extend(cur_solutions)
    return dp[-1]

# [3, 2, 2], [2, 3, 2], [2, 2, 3], [7]], # This will get all possible combination
# At i=7, it will look back at i==5 (which contains [2,3], [3,2])
combinationsSum_dp(candidates, target=7) 

def combinationsSum_dp(candidates, target):
    """Target = 10, = 2 + 2 + 2 + 2 + 2 = 2 + 4 + 4"""
    dp = [[] for _ in range(target + 1)] ## THIS IS VERY IMPORTANT, IF YOU MISSED THIS
    print(dp)
    dp[0] = [[]] # empty candidates
    for cand in candidates: # (1): SWAP THIS this (2)
        for i in range(1, len(dp)): # (2): SWAP THIS with (1)
            if i - cand >= 0 and len(dp[i-cand])>0:
                prev_solutions = dp[i-cand]
                cur_solutions = []
                for sol in prev_solutions:
                    new_sol = sol + [cand]
                    cur_solutions.append(new_sol)
                dp[i].extend(cur_solutions)
    return dp[-1]
# each candidate will populate the DP only once!
# Therefore at i = 5, we will only have [2,3], and not [[2,3],[3,2]]!
combinationsSum_dp(candidates, target=7)  # correct: [[2, 2, 3], [7]]




    