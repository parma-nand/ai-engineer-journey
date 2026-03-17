import numpy as np


arr=np.array([10,20,30])
print(arr)
print(arr.size)
# print(arr.shape)
# print(arr.dtype)
# print(arr[0])
# print(arr[0:3])
# print(arr+5)
# print(arr*2)
# print(arr.sum())
# print(arr.max())
# print(arr.min())
# print(arr.mean())

# arr2=np.array([[1,2,3],[4,5,6]])
# print(arr2)
# print(arr2.ndim)
# print(arr2.shape)
# print(arr2[0,1])
# print(arr2[0])
# print(arr2[:,1])
# print(arr2.T)
# print(arr2.sum(axis=0))

# arr3 = np.array([
#     [[1,2], [3,4]],
#     [[5,6], [7,8]]
# ])

# print(arr3)
# print(arr3.ndim)
# print(arr3.shape)
# print(arr3[0])
# print(arr3[0,1,1])

# np.array([10], dtype='int8')



# arr = np.array([[1,2,3],
#                 [4,5,6]])
# print(arr.sum(axis=0))
# print(arr.sum(axis=1))


a=np.array([1,2,3,4,5])
b=np.zeros((3,3))
c=np.ones((2,3))
d=np.arange(0,20,2)
e=np.random.randint(0,100,(3,3))
# print(b)
# print(c)
# print(d)
print(e)
print("Array : ",a)
print("Shape : ",e.shape)
print("Shape : ",e.size)
print("Mean : ",e.mean())
print("Max : ",e.max())
print("Min : ",e.min())

print("Element [0,1] : ",e[0,1])
print("First Row : ",e[0,:])
print("First Column : ",e[:,0])
print("2D Array Slicing : ",e[0:2,1:5])
print("1D Array Slicing : ",a[0:3])
