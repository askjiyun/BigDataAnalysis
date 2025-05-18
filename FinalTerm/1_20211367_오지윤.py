import pandas as pd
import matplotlib.pyplot as plt
# 데이터 로드 및 컬럼 정리
df = pd.read_csv('zillow.csv')
df.columns = df.columns.str.strip().str.replace('"', '').str.replace(' ', '_')
# 연도별 평균값 계산
yearly = df.groupby('Year').mean()
# 히스토그램 그리기
fig, axes = plt.subplots(3, 1, figsize=(8, 8))
axes[0].hist(yearly['List_Price_($)'], bins=10, color='blue', edgecolor='black')
axes[0].set_title('Avg Housing Price vs Year')
axes[1].hist(yearly['Living_Space_(sq_ft)'], bins=10, color='green', edgecolor='black')
axes[1].set_title('Avg Living Space vs Year')
axes[2].hist(yearly['Baths'], bins=10, color='orange', edgecolor='black')
axes[2].set_title('Avg Bathrooms vs Year')
plt.tight_layout(); plt.show()

# 피어슨 상관계수 계산 및 최대 상관관계 항목 찾기
correlations = yearly[['List_Price_($)', 'Living_Space_(sq_ft)', 'Baths', 'Beds']].corr(method='pearson')
max_corr = correlations.where(~correlations.isna() & (correlations < 1)).stack().idxmax()
print("Correlation Coefficients:\n", correlations)
print(f"\nThe highest correlation is between: {max_corr[0]} and {max_corr[1]}, Coefficient: {correlations.loc[max_corr]:.4f}")