"""
import torch

# Example tensor A (can be of any type)
A = torch.tensor([[0, 1, 0, 1],[0, 1, 0, 1]])

# Replace 0s with -inf and 1s with 0
result = torch.where(A == 0, -torch.inf, 0)
print(result)
"""
import math

def func(matrix, k):
    index = [(0,0)]
    popped = ()
    while k > 0:
        k -= 1
        minimum = math.inf
        min_idx = 0
        for i in range(len(index)):
            if matrix[index[i][0]][index[i][1]] < minimum:
                minimum = matrix[index[i][0]][index[i][1]]
                min_idx = i
        popped = index.pop(min_idx)
        if popped[0] < len(matrix) - 1:
            index.append((popped[0] + 1, popped[1]))
        if popped[1] < len(matrix[0]) - 1:
            index.append((popped[0], popped[1] + 1))
    return matrix[popped[0]][popped[1]]

a = [[1,6,9],
     [2,7,11],
     [5,8,12]]
print(func(a, 5))