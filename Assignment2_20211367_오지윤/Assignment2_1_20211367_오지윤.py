import numpy as np
import pandas as pd

# 분포 생성
n = 100000

# 정규분포 (평균 0, 표편 1)
normal_dist = np.random.normal(0,1,n)
# 카이제곱 (정규 분포 값 제곱)
chi_square_dist = np.random.normal(0, 1, n)**2
# 균등분포 (0부터 1 사이 값)
uniform_dist = np.random.uniform(0,1,n)

# DF에 분포값 저장
df = pd.DataFrame({
    'Normal_Distribution':normal_dist,
    'Chi-Square_Distribution':chi_square_dist,
    'Uniform_Distribution': uniform_dist
})
df.to_csv('test.csv', index=False)

print("test.csv 파일이 생성되었습니다.")