from numpy import random

# 범위 입력하면 그 사이의 숫자 선택
x = random.randint(1, 100)
print(x)

# Environment path 확인하는 코드
import sys
x = sys.path
for i in range(len(x)):
    print(x[i])
