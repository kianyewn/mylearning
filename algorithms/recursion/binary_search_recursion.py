def binary_search(sorted_lst, target, start, end):
    mid = (start + end) // 2
    # print(mid, start, end)
    res = None
    if sorted_lst[mid] == target:
        return mid

    if start < end and start+1 < len(sorted_lst) and end-1> 0:
        res = binary_search(sorted_lst, target, start, mid-1)
        if res is None:
            res = binary_search(sorted_lst, target, mid+1, end)
        
    return res if res is not None else None
        
lst = [5,7,22,23,30,90]
# Time complexity is O(N), because in the worst case I will still search all of left. if not found, then i search all of right
# Space compelxity is O(N) because of N + 2 recursive depth (n/2 +1) for left and right
binary_search(lst, 90, start=0, end=len(lst)-1) 
    
    
def binary_search2(sorted_lst, target, start, end):
    if start > end:
        return 'cannot find'
    mid = (start+end) // 2
    if sorted_lst[mid] == target:
        return mid
    if sorted_lst[mid] < target:
        # search right
        return binary_search2(sorted_lst, target, mid+1, end)
    elif sorted_lst[mid] > target:
        # search left
        return binary_search2(sorted_lst, target, start, mid-1)
    else:
        return 'Not Found'
    
lst = [5,7,22,23,30,90]
# time complexity O(log(n)) because the number of times we can divide N by 2 until it becomes less than one is log(n))
# space complexity is o(log(n))< because we call recursion log(n) times and therefore log(n) recursive depth
binary_search2(lst, 90, start=0, end=len(lst)-1) 
    