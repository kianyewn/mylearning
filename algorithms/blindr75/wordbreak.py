"""Given a string s and a dictionary of strings wordDict, 
return true if s can be segmented into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
"""

from typing import List
def wordBreak(s:str, wordDict:List):
    dp = [0] * (len(s)+1)
    dp[0] = 1 # empty can break
    for i in range(1, len(dp)):
        for word in wordDict:
            if i-len(word) >=0 and s[i-len(word):i] == word:
                # dp[i] = dp[i-len(word)] # this will fail, because if another word overwrites it, it will become 0
                if dp[i-len(word)]:
                    dp[i] = dp[i-len(word)]
    return dp[-1]

    
s = "leetcode"
wordDict = ["leet","code"]
# wordBreak(s, wordDict)
    
s = "catsandog"
wordDict = ["cats","dog","sand","and","cat"]
# wordBreak(s, wordDict)

s = "dogs"
wordDict = ["dog","s","gs"]
wordBreak(s, wordDict) # word overwrite and it becomes zero

