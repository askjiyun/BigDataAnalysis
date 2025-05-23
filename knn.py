# K-최근접 이웃(K-Nearest Neighbors, KNN)을 사용한 회귀 분석

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

# 1. 데이터 생성
np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()

# Add noise to targets 노이즈 추가
y[::5] += 1 * (0.5 - np.random.rand(8))

# 2. KNN 회귀 모델 선정
n_neighbors = 5

for i, weights in enumerate(["uniform", "distance"]):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X, y).predict(T)

    # 시각화
    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, color="darkorange", label="data")
    plt.plot(T, y_, color="navy", label="prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))

plt.tight_layout()
plt.show()
