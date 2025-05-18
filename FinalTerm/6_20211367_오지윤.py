import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# 6-1 문제 해결
df = pd.read_csv('subway_2.csv', encoding='CP949')
plt.rcParams['font.family'] = 'AppleGothic'
df.groupby('노선명')[['승차총승객수', '하차총승객수']].mean().plot(kind='bar', figsize=(8, 6), title='Mean 승차총승객수 & 하차총승객수')
plt.show()

# 6-2 문제
X, y = df[['승차총승객수', '하차총승객수']], df['노선명']
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(scaler.fit_transform(X), y, test_size=0.2, random_state=42)

input_value = scaler.transform(pd.DataFrame([[30000, 30000]], columns=['승차총승객수', '하차총승객수']))
models = {"KNN": KNeighborsClassifier(), "Logistic Regression": LogisticRegression(max_iter=5000), "Decision Tree": DecisionTreeClassifier(random_state=42)}
# 4. Train, predict, and evaluate models
acc = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    acc[name] = accuracy_score(y_test, model.predict(X_test))
    print(f"{name} Prediction: {model.predict(input_value)[0]}")
# 6-3 문제
print(f"\nModel Accuracy: {acc}\nBest Model: {max(acc, key=acc.get)}")