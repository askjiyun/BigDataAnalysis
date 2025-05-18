import numpy as np
x = np.arange(62500)

x = x.reshape(10, 10, 625)  # 10x10x625 형태로 reshape
result = np.mean(x, axis=2)  # 깊이 방향으로 평균 계산
np.savetxt("test.csv", result, delimiter=",", fmt="%.2E")  # 결과를 test.csv로 저장
