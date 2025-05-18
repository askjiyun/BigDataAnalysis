# import chardet
# # 파일 인코딩 확인
# with open("student_health_3.csv", "rb") as file:
#     result = chardet.detect(file.read())
#     print(result['encoding']) #EUC-KR

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. 데이터 로드 및 전처리
file_path = 'student_health_3.csv'
df = pd.read_csv(file_path, encoding='euc-kr')
# '학년'을 기반으로 Class를 생성
df['Class'] = df['학년'].apply(lambda x: 1 if x <= 3 else 2)


# 주요 변수 선택
data_selected = df[['수축기', '이완기', '키', '몸무게', 'Class']].rename(columns={
    '수축기': 'LowerBP',
    '이완기': 'UpperBP',
    '키': 'Height',
    '몸무게': 'Weight'
})

# 2. 독립 변수와 종속 변수 분리
X = data_selected[['LowerBP', 'UpperBP', 'Height', 'Weight']]
y = data_selected['Class'] - 1  # Logistic Regression에 맞게 0과 1로 변환

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 랜덤 포레스트 모델 학습
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# 3. 테스트 정확도 평가
y_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)


# 4. 주어진 입력 데이터 예측
input_data = [[80, 100, 140, 60]]  # 입력 데이터
input_data_df = pd.DataFrame(input_data, columns=['LowerBP', 'UpperBP', 'Height', 'Weight'])  # DataFrame 변환
predicted_class = rf_model.predict(input_data_df)
print("Predicted Class for", input_data, ":", predicted_class[0] + 1)  # 원래 Class로 변환