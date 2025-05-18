import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 데이터 불러오기
file_path = 'subway_1.csv'  # 파일 경로
df = pd.read_csv(file_path, encoding='CP949')

# 필요 열만 선택
X = df[['승차총승객수', '하차총승객수']]
y = df['노선명']

# 타겟 레이블 인코딩 (문자열을 숫자로 변환)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 데이터 분할 (train-test split)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

# 모델 학습 (RandomForestClassifier)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# 예측 함수
def predict_line_number(model, le, 승차총승객수, 하차총승객수):
    # 입력 데이터를 DataFrame으로 변환하여 feature names 유지
    input_data = pd.DataFrame([[승차총승객수, 하차총승객수]], columns=['승차총승객수', '하차총승객수'])
    prediction_encoded = model.predict(input_data)
    prediction_label = le.inverse_transform(prediction_encoded)
    return prediction_label[0]

# 주어진 입력 값에 대한 예측
print("\n### 예측 결과 ###")
input_values = [(1000, 5000), (6000, 20000)]
for 승차, 하차 in input_values:
    result = predict_line_number(clf, le, 승차, 하차)
    print(f"입력 값: 승차={승차}, 하차={하차} -> 예측 노선명: {result}")