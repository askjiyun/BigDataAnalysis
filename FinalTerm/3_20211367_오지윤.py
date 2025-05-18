import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
df = pd.read_csv('student_health_2.csv', encoding='CP949')
df.rename(columns={'키': 'Height', '학년': 'Grade', '학교ID': 'School'}, inplace=True)

# 문제 1: Grade vs School colormap
pivot_table = df.pivot_table(values='Height', index='Grade', columns='School', aggfunc='mean')
plt.imshow(pivot_table, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Mean Height')
plt.xticks(ticks=np.arange(len(pivot_table.columns)), labels=pivot_table.columns)
plt.yticks(ticks=np.arange(6), labels=[1, 2, 3, 4, 5, 6])
plt.xlabel('School'); plt.ylabel('Grade'); plt.title('Height Colormap: Grade vs School'); plt.show()

# 문제 2: Mean height and correlation calculation
grade_means = df.groupby('Grade')['Height'].mean()
correlations = {school: stats.pearsonr(grade_means, df[df['School'] == school].groupby('Grade')['Height'].mean())[0] for school in df['School'].unique()}
best_school = max(correlations, key=correlations.get)

# 결과 출력
print("Mean Height by Grade:\n", grade_means)
print("\nPearson Correlation Coefficients:")
for school, corr in correlations.items(): print(f"School {school}: {corr:.4f}")
print(f"\nBest School: {best_school}, Correlation: {correlations[best_school]:.4f}")