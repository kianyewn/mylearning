"""
[Medium]
Given a collection of candidate numbers (candidates) and a target number (target), 
find all unique combinations in candidates where the candidate numbers sum to target.

Each number in candidates may only be used once in the combination.

Note: The solution set must not contain duplicate combinations.

Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]
"""


def dfs(candidates, target, res):
    if len(candidates) == 0:
        return []
    if target == 0:
        return [res]
    
    all_res = []
    prev = None
    for i in range(len(candidates)):
        if prev == candidates[i]:
            continue
        res = dfs(candidates=candidates[i:],
                  target= target-candidates[i],
                  res=res+[candidates[i]]) # list of list
        all_res.extend(res)
        prev = candidates[i]
    
    return all_res # list of combinations:list
def combinationSum2(candidates, target):
    candidates.sort()
    return dfs(candidates, target, res=[])


