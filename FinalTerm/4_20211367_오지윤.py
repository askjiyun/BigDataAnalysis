import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

file_path = 'London.csv'
df = pd.read_csv(file_path, encoding='CP949')

# 선형 회귀 플롯 및 신뢰 구간 함수
def plot_regression_per_type(df, house_type_col, price_col, area_col):
    house_types = df[house_type_col].unique()

    for htype in house_types:
        subset = df[df[house_type_col] == htype]
        X = subset[[area_col]].values
        y = subset[price_col].values

        # 선형 회귀 모델 학습
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # 플롯
        plt.figure(figsize=(8, 6))
        plt.scatter(X, y, color='lightblue', label='Data Points')
        plt.plot(X, y_pred, color='red', label='Regression Line')

        # 신뢰 구간 (±1 표준편차)
        tolerance = np.std(y - y_pred)
        plt.fill_between(X.flatten(), y_pred - tolerance, y_pred + tolerance, color='red', alpha=0.2, label='Tolerance Interval')

        plt.title(f'Linear Regression for {htype}')
        plt.xlabel('Area')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

# 평균 계산 함수
def calculate_averages(df, house_type_col, price_col, area_col, rooms_col):
    summary = df.groupby(house_type_col).agg(
        Avg_Price=(price_col, 'mean'),
        Avg_Area=(area_col, 'mean'),
        Avg_Rooms=(rooms_col, 'mean')
    ).reset_index()

    print("### Averages of Price, Area, and Number of Rooms by House Type ###\n")
    for _, row in summary.iterrows():
        print(f"House Type: {row[house_type_col]}")
        print(f"  Average Price: {row['Avg_Price']:.2f}")
        print(f"  Average Area: {row['Avg_Area']:.2f}")
        print(f"  Average Rooms: {row['Avg_Rooms']:.2f}\n")

# 실행
plot_regression_per_type(df, 'House Type', 'Price', 'Area in sq ft')
calculate_averages(df, 'House Type', 'Price', 'Area in sq ft', 'No. of Bedrooms')