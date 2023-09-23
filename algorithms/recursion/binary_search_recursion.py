def binary_search(sorted_lst, target, start, end):
    mid = (start + end) // 2
    print(mid, start, end)
    res = None
    if sorted_lst[mid] == target:
        return mid

    if start < end and start < len(sorted_lst) and end > 0:
        res = binary_search(sorted_lst, target, start, mid-1)
        if res is None:
            res = binary_search(sorted_lst, target, mid+1, end)
        
    return res if res is not None else None
        
lst = [5,7,22,23,30,90]
binary_search(lst, 90, start=0, end=len(lst)-1) 
    