#from numpy import random


import numpy as np
# 1-9 사이에서 random 정수
x = np.random.randint(1,10)
# 0~1 의 균일분포 표준정규분포 난수를 matrix array(m,n) 생성
x1 = np.random.rand(6)
# 평균 0, 표준편차 1의 가우시안 표준정규분포 난수를 matrix array(m,n) 생성
x2 = np.random.randn(3,2)
print(x2)
print(x1)
print(x)

# Environment path 확인하는 코드
import sys
x = sys.path
for i in range(len(x)):
    print(x[i])

# Methods vs. Function
# Methods - class와 연관이 존재
class MyMath:
    def add(self, a, b):
        return a + b

p1 = MyMath()
result = p1.add(3, 4)
print(result)

# Function - 연관 없음
sm = sum([5,15,2])
print(sm)

mx = max(15, 6)
print(mx)

# Class
class MyPerson:
    i = 5
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bluechip(self):
        return "what is your name?"

# Instance
p2 = MyPerson("Kim", 22)
print(p2.name)
print(p2.age)
print(p2.i)
print(p2.bluechip())