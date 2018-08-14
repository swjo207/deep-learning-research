from tensorflow.python.util.nest import *

print(flatten([[[[1],2],3],4]))
print(flatten([1,2,3,4,5,6]))
print(flatten([1,2,[3,4],5,[6]]))
print(flatten([[1,[2,3]], [[4,5], 6]]))

r = [[1, 0],      [0, 1]]
a = [[   [1, 2], [3, 4]   ],      [   [5, 6], [7, 8]   ]]
print(flatten_up_to(r, a))
print(flatten_up_to(a, a))
print(flatten_up_to([1, [1,1]], [[2,3], [[4,5], 6]]))

print(pack_sequence_as([[1,(1,1)],1], [4,5,6,7]))
print(map_structure(lambda a,b: a*b,  [1, (2,[3]), 4], [2, (4,[8]), 16]))
print(map_structure_up_to([1, (1,1), 1], str, [1, (2,[3]), 4]))



