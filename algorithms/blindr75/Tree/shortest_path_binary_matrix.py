"""1091. Shortest Path in Binary Matrix
[Medium]
Given an n x n binary matrix grid, return the length of the shortest clear path in the matrix. If there is no clear path, return -1.

A clear path in a binary matrix is a path from the top-left cell (i.e., (0, 0)) to the bottom-right cell (i.e., (n - 1, n - 1)) such that:

All the visited cells of the path are 0.
All the adjacent cells of the path are 8-directionally connected (i.e., they are different and they share an edge or a corner).
The length of a clear path is the number of visited cells of this path.
"""

def shortest_path(grid):
    """BFS will find the shortest path, because every node in each level is touched once, before moving to nodes in the next level"""
    n = len(grid)
    visited = set()
    
    # handle edge cases, when start and end is -1, this means no solution
    if grid[0][0] or grid[-1][-1]:
        return -1
    
    # 8 directional. (row, col)
    directions = [(1,0), # move down
                  (0,1,), # move right
                  (-1,0), # move up
                   (0,-1), # move left
                   (-1,-1), # move diagonally upleft
                   (1,-1), # move diagonally downleft
                   (1,1), # move digonally downright,
                   (-1,1), # move diagonally upright
                   ]
    queue =  []
    visited.add((0,0))
    queue.append((0,0,1))

    while len(queue) > 0:
        cur_row, cur_col, cur_length = queue.pop(0) # pop from left
        
        if cur_row==n-1 and cur_col==n-1:
            return cur_length
            
        for dr, dc in directions:
            new_row = cur_row + dr
            new_col = cur_col + dc
                
            if min(new_row, new_col) >= 0 and max(new_row, new_col) < n \
                and not (new_row, new_col) in visited and grid[new_row][new_col]==0:
                    queue.append((new_row, new_col, cur_length+1))
                    visited.add((new_row, new_col))
    return -1
         

grid = [[0,0,0],[1,1,0],[1,1,0]]
assert shortest_path(grid) == 4