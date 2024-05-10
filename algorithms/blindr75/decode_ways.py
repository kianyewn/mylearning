"""A message containing letters from A-Z can be encoded into numbers using the following mapping:

'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
To decode an encoded message, all the digits must be grouped then mapped back into letters 
using the reverse of the mapping above (there may be multiple ways). For example, "11106" can be mapped into:

"AAJF" with the grouping (1 1 10 6)
"KJF" with the grouping (11 10 6)
Note that the grouping (1 11 06) is invalid because "06" cannot be mapped into 'F' since "6" is different from "06".

Given a string s containing only digits, return the number of ways to decode it.

The test cases are generated so that the answer fits in a 32-bit integer.

Input: s = "12"
Output: 2
Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
"""

def decodeWays(string):
    # dp = [0] * (len(string)+1)
    # for i in range(1,dp):
    #     if string[i-1] in '123456789':
    #         dp[i] = dp[i]
            
    #     if string[i-1] == '0':
    #         pass
    pass

def decodeWays(string, index):
    if index == len(string):
        return 1
    
    if string[index] == '0':
        return 0
    
    # for i in range(1, len(string)):
    #     # single
    #     if string[i-1] in '123456789':
    #         decode_single = decodeWays(string[i:], index=i)
    #     single = string[i]
    #     if single in '123456789':
    #         decode_single = decodeWays(string[i+1:], index=i)
    
    # decode single
    single = decodeWays(string, index+1)
    if index+1 >0 and (string[index] in '12' and string[index+1] in '0123456'):
        double = decodeWays(string, index=index+2)
    return single + double


    