from sklearn.linear_model import LogisticRegression
import pandas as pd

# 학습 데이터 로드
url = 'http://bit.ly/kaggletrain'
train = pd.read_csv(url)

# 특징 및 타겟 변수 선택
feature_cols = ['Pclass', 'Parch']
# 특징 변수
X = train.loc[:, feature_cols]
# 타겟 변수
y = train.Survived

# 로지스틱 회귀 모델 초기화
logreg = LogisticRegression()
# 학습
logreg.fit(X, y)
# 테스트 데이터 로드
url_test = 'http://bit.ly/kaggletest'
test = pd.read_csv(url_test)
# 예측
X_new = test.loc[:, feature_cols]
new_pred_class = logreg.predict(X_new)

# 예측 결과를 DF에 저장 & CSV 파일로 내보내기
kaggle_data = pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':new_pred_class}).set_index('PassengerId')
kaggle_data.to_csv('sub.csv')

# 학습 데이터를 pickle 형식으로 저장 및 불러오기
train.to_pickle('train.pkl')
pd.read_pickle('train.pkl')
