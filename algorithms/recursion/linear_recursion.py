def recursion_sum(lst):
    if len(lst) == 0:
        return 0
    
    else:
        return lst[0] + recursion_sum(lst[1:])
    
    
 # time complexity: O(N)
 # space complexity: O(N) n + 1 memory. 1,2,3,4,5,0   
lst = [1,2,3,4,5]
recursion_sum(lst)