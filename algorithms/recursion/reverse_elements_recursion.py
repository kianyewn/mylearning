def reverse(lst):
    if len(lst) == 0:
        return []
    else:
        return lst[-1:] + reverse(lst[:-1])

# time complexity O(n)
# space complexity O(n)
lst = [1,2,3,4,5]
reverse(lst)


def reverse(lst, start, stop):
    if start == stop:
        # do nothing
        return None
    else:
        to_swap = lst[start]
        lst[start] = lst[stop]
        lst[stop] = to_swap
        reverse(lst, start+1, stop-1)
    return lst

# time complexity O(N)
# space complexity N/2 +1 reversive calls = O(N)
lst = [1,2,3,4,5]
reverse(lst, 0, len(lst)-1)


def reverse(lst, start, stop):
    if start < stop:
        to_swap = lst[start]
        lst[start] = lst[stop]
        lst[stop] = to_swap
        reverse(lst, start+1, stop-1)
    return lst
lst = [1,2,3,4,5]
reverse(lst, 0, len(lst)-1)