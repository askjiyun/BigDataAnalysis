import numpy as np
import pandas as pd

# 통계 계산 함수 (numpy)
def calculate_statistics_numpy(data):
    stat = {
        'mean': np.mean(data),
        'std' : np.std(data),
        'variance' : np.var(data),
        'min' : np.min(data),
        'max':np.max(data),
        '10%' : np.percentile(data,10),
        '20%': np.percentile(data, 20),
        '40%': np.percentile(data, 40),
        '80%': np.percentile(data, 80),
        '90%': np.percentile(data, 90),
    }
    return stat

# 통계 계산 함수 (pandas)
def calculate_statistics_pandas(df):
    stat = {
        'mean': df.mean(),
        'std' : df.std(),
        'variance' : df.var(),
        'min' : df.min(),
        'max':df.max(),
        '10%' : df.quantile(0.10),
        '20%': df.quantile(0.20),
        '40%': df.quantile(0.40),
        '80%': df.quantile(0.80),
        '90%': df.quantile(0.90),
    }
    return stat

# sample size
sample_sizes = [10000, 20000, 30000, 50000, 100000]

# 분포 생성
n = 100000
n = 100000
normal_dist = np.random.normal(0, 1, n)
chi_square_dist = np.random.normal(0, 1, n) ** 2
uniform_dist = np.random.uniform(0, 1, n)

# 분포 저장
distributions = {'Normal_Distribution': normal_dist, 'Chi_Square_Distribution': chi_square_dist,
                 'Uniform_Distribution': uniform_dist}

# 결과 저장할 리스트
results = []

# 각 샘플 크기별로 통계 계산
for sample_size in sample_sizes:
    for dist_name, dist_data in distributions.items():
        sample = dist_data[:sample_size]

        # pandas 통계 계산
        df = pd.DataFrame(sample, columns=[dist_name])
        pandas_stats = calculate_statistics_pandas(df[dist_name])
        pandas_stats['distribution'] = dist_name
        pandas_stats['method'] = 'pandas'
        pandas_stats['sample_size'] = sample_size
        results.append({
            'distribution': dist_name,
            'method': 'pandas',
            'sample_size': sample_size,
            **pandas_stats
        })

        # numpy 통계 계산
        numpy_stats = calculate_statistics_numpy(sample)
        numpy_stats['distribution'] = dist_name
        numpy_stats['method'] = 'numpy'
        numpy_stats['sample_size'] = sample_size
        results.append({
            'distribution': dist_name,
            'method': 'numpy',
            'sample_size': sample_size,
            **numpy_stats
        })

# 결과를 DataFrame으로 변환
result_df = pd.DataFrame(results)

# CSV 파일로 저장
result_df.to_csv('result.csv', index=False)

print("result.csv 파일이 생성되었습니다.")