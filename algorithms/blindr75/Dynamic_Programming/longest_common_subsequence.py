"""
Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.

A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

For example, "ace" is a subsequence of "abcde".
A common subsequence of two strings is a subsequence that is common to both strings.

 

Example 1:

Input: text1 = "abcde", text2 = "ace" 
"""

def longestCommonSubsequence(string1, string2):
    dp = [[0] * (len(string2)+1) for _ in range(len(string1)+1)]
    
    for row in range(1,len(dp)):
        for col in range(1,len(dp[0])):
            if string1[row-1] == string2[col-1]:
                dp[row][col] = dp[row-1][col-1] + 1
            else:
                dp[row][col] = max(dp[row][col-1], dp[row-1][col])
                
    return dp[-1][-1]
            
text1 = "abcde"
text2 = "ace" 
longestCommonSubsequence(text1, text2)