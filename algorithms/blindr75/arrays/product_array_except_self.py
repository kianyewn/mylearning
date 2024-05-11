"""238. Product of Array Except Self
[Medium]
Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.

Input: nums = [1,2,3,4]
Output: [24,12,8,6]

good read: https://leetcode.com/problems/product-of-array-except-self/solutions/1342916/3-minute-read-mimicking-an-interview/
"""

def naive_productExceptSelf(arr):
    res = []
    # naive
    for i in range(len(arr)):
        cur = 1
        for j in range(len(arr)):
            if i == j:
                continue
            else:
                cur *= arr[j]
        res.append(cur)
    return res

def productExceptSelf(arr):
    left_product =[1 for _ in range(len(arr))]
    right_product = [1 for _ in range(len(arr))]
    
    for i in range(1,len(arr)):
        left_product[i] = left_product[i-1] * arr[i-1]
    
    for j in range(len(arr)-2,-1,-1):
        right_product[j] = right_product[j+1] * arr[j+1]
    
    res = [0] * len(arr)
    for i in range(len(arr)):
        res[i] = left_product[i] * right_product[i]
    return res
        
   
def productExceptSelf(arr):
    """Optimized for memory"""
    res = [1] * len(arr)
    
    left_sofar = 1
    for i in range(1, len(arr)):
        left_sofar = left_sofar * arr[i-1]
        res[i] = left_sofar
        
    right_sofar = 1
    for j in range(len(arr)-2,-1,-1):
        right_sofar = right_sofar * arr[j+1]
        res[j] *= right_sofar
    return res

nums = [1,2,3,4]
output = [2*3*4, 1*3*4, 1*2*4, 1*2*3]
assert output == [24, 12, 8, 6]

assert naive_productExceptSelf(nums) == output
assert productExceptSelf(nums) == output

