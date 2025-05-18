# Gaussian Naive Bayes 를 사용한 분류

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB

#1. 데이터 로드 및 분할
X, y = load_iris(return_X_y=True) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0) 
# 2. GaussainNB 모델 학습 및 예측
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test) 
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
