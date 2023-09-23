def power(x, n):
    if n == 0:
        return 1
    return x * power(x,n-1)

# time complexity O(N)
# space compelxity n+1 = O(N)
x = 2
power(x, 5)

