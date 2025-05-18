import numpy as np

arr = np.array(([1, 2, 3], [4, 5, 6]))

print(arr.shape)
# 각 열을 기준으로 값을 더함 (차원의 수를 줄임)
print(np.add.reduce(arr))
# 각 행을 기준으로 값을 더함 (차원의 수를 더함)
print(np.add.reduce(arr, axis=1))

# 배열의 요소를 순차적으로 더하면서 그 누적합계를 반환
print(np.add.accumulate(arr))
print(np.add.accumulate(arr, axis=1))

print(np.add.accumulate([1, 2, 3, 4, 5]))
#각 원소를 곱해서 누적곱셈 반환
print(np.multiply.accumulate([1, 2, 3, 4, 5]))
# 각 원소의 누적뺄셈 진행
print(np.subtract.accumulate([1,2,3,4,5]))
print(np.divide.accumulate([1,2,3,4,5]))
# 누적 나머지 계산
print(np.remainder.accumulate([1123,18,9,5,3]))

#0~11 사이의 원소를 3x4 행렬로 생성
arr = np.arange(12).reshape((3, 4))
# 각 열을 기준으로 누적덧셈
print(np.add.accumulate(arr))
print(np.add.accumulate(arr, axis=1))

#numpy.subtract
print(np.subtract(1.0, 4.0))
# 3x3 행렬
arr1 = np.arange(9.0).reshape((3, 3))
# 1차원 배열
arr2 = np.arange(3.0)
# 뺄셈 진행 시, broadcasting 진행
print(np.subtract(arr1, arr2))
#
# 배열의 거듭제곱 계산
arr1 = range(6)
print(arr1)
# 각 arr1 요소에 3제곱
print(np.power(arr1, 3))
# arr1과 arr2의 대응하는 값들에 대해 거듭제곱 연산 수행
arr2 = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
print(np.power(arr1, arr2))

#sin function plot
import numpy as np
import matplotlib.pyplot as plt
# 0 ~ 30 까지 구간을 20001개의 균등한 점으로 나누어 연산
arr = np.linspace(0, 30, 20001)
plt.plot(arr, np.sin(arr))
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.grid()
plt.show()

# Slice View
arr = np.arange(10)
print(arr) #[0 1 2 3 4 5 6 7 8 9]
v1 = arr[1:2]
print(v1) #[1] → v1 참조
arr[1] = 2
print(v1) #[2] → v1 참조
v2 = arr[1::3]
print(v2) #[2 4 7] → v2
arr[7] = 10
print(v2) #[ 2  4 10] → v2

# Astype Copy
arr = np.ones(4, dtype=np.uint8)
print(arr) # [0 0]
print(arr.view(np.uint8)) # [0 0 0 0]
# print(arr.view(np.uint32)) # [0]

# astype : 배열의 원소를 다양한 타입으로 변환
arr = np.ones(4, dtype=np.uint8)
print(arr.dtype)
print(arr.astype(np.uint16).dtype)
print(arr.astype(np.uint32))

# 2차원 배열 viewing
arr = np.arange(4, dtype=np.uint16).reshape(2, 2)
print(arr)
# 16bit -> 8bit 변환 (각 요소는 2바이트 차지)
print(arr.view(np.uint8))
print(arr * 100)
print((arr * 100).view(np.uint8))


arr = np.arange(10, dtype='int16')
print(arr)
v1 = arr.view('int32')
print(v1)
v1 += 1
print(v1.view('int16'))
v2 = arr.view('int8')
print(v2)

# BroadCasting
a = np.array([[1.0, 2.0, 3.0]])
b = np.array([[2.0, 2.0, 2.0]])
arr = a.T * b
print(arr)


a = np.array([1.0, 2.0, 3.0])
b = 2.0
print(a * b)

arr1 = np.arange(4)
arr2 = arr1.reshape(4, 1)
arr3 = np.ones(5)
arr4 = np.ones((3, 4))
print(arr1.shape)
print(arr3.shape)
# arr1 + arr3 -> error  why? shape이 같지 않기 때문
print(arr2.shape)
print((arr2 + arr3).shape)
print(arr2 + arr3)
print(arr4.shape)
print((arr1 + arr4).shape)
print(arr1 + arr4)

# numpy.newaxis => reshaping new dimension
arr = np.array([0, 1, 2, 3])
print(arr.shape)
print(arr[np.newaxis, :])
print(arr[:, np.newaxis])

arr1 = np.array([0, 1, 2, 3])
arr2 = np.array([10, 20, 30])
arr11 = arr1[:, np.newaxis]
print(arr11 + arr2)

arr = np.arange(10)
arr_br = arr - arr[:, np.newaxis]
print(arr_br)

import numpy as np
import matplotlib.pyplot as plt
arr1 = np.arange(10)
arr2 = np.arange(7)
arr_img = np.sqrt(arr1[:, np.newaxis]**2 + arr2**2)
plt.pcolor(arr_img)
#plt.pcolor(arr_br)
plt.colorbar()
plt.axis('equal')
plt.show()


print("----------------------------")
#Array Management and alignment

#numpy concatenate
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6]])
print(np.concatenate((arr1, arr2), axis=0))
print(np.concatenate((arr1, arr2.T), axis=1))
print(np.concatenate((arr1, arr2), axis=None))

#numpy.vstack & numpy.hstack
print(np.hstack((arr1, arr2.T)))
print(np.vstack((arr1, arr2)))

# Structured array with sorting in keyword ‘height’
dtype = [('name', 'S10'), ('height', float), ('age', int)]
values = [('Jin', 175, 59), ('Suho', 185, 19),('Naeun', 162, 28)]
arr = np.array(values, dtype=dtype)
arr.sort(order=['name'])
print(arr)
print(np.sort(arr, order=['age']))

# Histogram
arr = np.array([15,16,16,17,19,20,22,35,43,45,55,59,60,75,88])
np.histogram(arr, bins=[0,20,40,60,80,100])
a, b = np.histogram(arr, bins=[0,20,40,60,80,100])

#Matplotlib
import matplotlib.pyplot as plt
import numpy as np
arr = np.array([15,16,16,17,19,20,22,35,43,45,55,59,60,75,88])
plt.hist(arr, bins=[0, 20, 40, 60, 80, 100])
plt.title('numbers depending on ages')
plt.show()

# Input Output - Save array as textfile
arr1 = arr2 = arr3 = np.arange(0.0, 5.0, 1.0)
np.savetxt('test1.txt', arr1, delimiter=',') # arr1 is array
np.savetxt('test2.txt', (arr1,arr2,arr3)) # same size of 2D array
np.savetxt('test3.txt', arr1, fmt='%1.4e') # exponential
arr4= np.loadtxt('test1.txt')
arr5= np.loadtxt('test2.txt')
arr6= np.loadtxt('test3.txt')
print(arr4)
print(arr5)
print(arr6)

# Image Processing
import scipy.misc
import matplotlib.pyplot as plt
face = scipy.misc.face()
print(face.shape)
print(face.max(), face.min(), face.mean())
print(face.dtype)
plt.gray()
plt.imshow(face)
plt.show()

# Create Numpy array from raw file of
import imageio
imageio.imwrite('face.png', face)
face1 = imageio.imread('face.png')
print(type(face1))
print(face1.dtype, face1.shape)
face1.tofile('face1.raw')
arr = np.fromfile('face1.raw', dtype=np.uint8)
print(arr)
print(arr.shape)
arr.shape = (768, 1024, 3)
print(arr)
print(arr.shape)

# Matplotlib funciton imshow - 해상도를 높일지 낮출지
import scipy.misc
import matplotlib.pyplot as plt
face1 = scipy.misc.face(gray=True)
plt.imshow(face1, cmap=plt.cm.gray)
plt.axis('off')
plt.show()

# 해상도를 낮출지 높일지 결정가능
plt.imshow(face1, cmap=plt.cm.gray, vmin=20, vmax=100)
plt.axis('off')
plt.show()

# Partial Plotting
face1 = scipy.misc.face(gray=True)
plt.imshow(face1[0:500, 400:900], cmap=plt.cm.gray, interpolation='nearest')
plt.show()

# Mirror and Rotation
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
arr = scipy.misc.face(gray=True)
arr_flipud = np.flipud(arr)
arr_rotate = scipy.ndimage.rotate(arr, 45)
plt.imshow(arr_flipud, cmap=plt.cm.gray)
plt.show()
plt.imshow(arr_rotate, cmap=plt.cm.gray)
plt.show()
