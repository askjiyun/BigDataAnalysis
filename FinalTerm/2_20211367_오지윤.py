import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load Data
file_path = 'hw_25000.csv'
df = pd.read_csv(file_path, encoding='CP949')

# 문제 1: 히스토그램과 PDF 비교
def plot_hist_with_pdf(data, title, ax):
    df_min, df_max = data.min(), data.max()
    df_mean, df_std = data.mean(), data.std()
    df_lins = np.linspace(df_min, df_max, 100)
    df_norm = stats.norm.pdf(df_lins, df_mean, df_std)

    ax.hist(data, bins=40, density=True, alpha=0.5, color='blue', label='Histogram')
    ax.plot(df_lins, df_norm, 'r-', label='PDF')
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plot_hist_with_pdf(df[' "Height(Inches)"'], 'Height Histogram with PDF', axes[0])
plot_hist_with_pdf(df[' "Weight(Pounds)"'], 'Weight Histogram with PDF', axes[1])
plt.tight_layout()
plt.show()

# 문제 2: 평균과 분산 계산 후 정규분포 생성 및 저장
height_mean, height_var = df[' "Height(Inches)"'].mean(), df[' "Height(Inches)"'].var()
weight_mean, weight_var = df[' "Weight(Pounds)"'].mean(), df[' "Weight(Pounds)"'].var()

height_generated = np.random.normal(height_mean, np.sqrt(height_var), 100)
weight_generated = np.random.normal(weight_mean, np.sqrt(weight_var), 100)

result_df = pd.DataFrame({'Height': height_generated, 'Weight': weight_generated})
result_df.to_csv('resultprob2.csv', index=False, encoding='CP949')