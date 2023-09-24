# # major tick length: length of tick for whole inch
# # minor ticks, placed at intervals of 1/2 inch, 1/4 and so forth. 
# # as the size of interval decreases by half, the tick length decreases by one

# def draw_line(length, marker, ticker='-'):
#     print('-' * length + f' {marker}')
    
# def draw_interval(length, marker=''):
#     if length > 0:
#         draw_interval(length-1, marker='')
#         draw_line(length, marker=marker)
#         draw_interval(length-1, marker='')
        

# def draw_ruler(num_inches, major_tick_length):
#     for j in range(1, 1+num_inches):
#         draw_interval(major_tick_length-1)
#         draw_line(length=major_tick_length, marker=str(j))
        
        
# num_inches = 2
# major_tick_length = 5
# draw_ruler(num_inches, major_tick_length)



def draw_line(length, marker=''):
    print('-' * length + ' ' + marker)
    
def draw_interval(length):
    if length > 0:
        draw_interval(length-1)
        draw_line(length)
        draw_interval(length-1)
        
def draw_ruler(inches, major_tick_length):
    draw_line(length=major_tick_length, marker='0')
    for i in range(1, inches+1):
        draw_interval(major_tick_length-1)
        draw_line(length=major_tick_length, marker=f'{i}')
        
    
# draw_line(2)
# draw_interval(3)a
    
# Time complexity for draw_interval: O(2**major tick length). Draw_ruler = O(2**major_tick_length * inches) because each inch calls one draw_interval
# Each draw_interval calls two recursive calls: 2 ** major_tick_length + (n-1) + (n-2) +... + 1 = 2**major_tick_length + n**2/2 + n/2 - 1 (via AP)
# Space complextiy: O(N), because the number of times it takes N to become 0 is N+1 inside draw_interval
draw_ruler(inches=2, major_tick_length=5)    
    
    