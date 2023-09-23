def power(x, n):
    if n == 1:
        return x
    else:
        # if even
        if n % 2 == 0:
            ele = power(x, n//2)
            square = ele * ele
            return square
        else:
            ele = power(x, n//2)
            square = ele * ele * x
            return square
        
# time complexity: O(N). False this is O(log(n)) because  number of times we can devide N until it gets to one or less is O(log(N))
# space compelxity: N/2 +1 recursive calls = O(N). False, this is O(log(n)) as well because the recursive depth is O(log(N)) as well.
# space is due to recursive depth calls.
x = 2
power(x, 8)


# power(x, 3//2=1)
# power(x, 1) = 2
# 2 * 2 + 2 = 6


def power(x,n):
    if n == 0:
        return 1
    else:
        partial = power(x, n//2)
        result = partial * partial
        # if odd, add additional
        if n % 2 == 1:
            result *= x
        return result
    
x = 2
power(x, 8)
            