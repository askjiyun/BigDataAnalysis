import pandas as pd
import matplotlib.pyplot as plt

file_path = 'student_health_2.csv'
student_health_df = pd.read_csv(file_path, encoding='euc-kr')

# 평균값 계산
averages = student_health_df.groupby('학년')[['키', '몸무게', '수축기', '이완기']].mean()

# 1. 학년별 학생 수 계산
student_count = student_health_df['학년'].value_counts().sort_index()
# 2. 학생 수 플롯 추가
plt.figure(figsize=(8, 6))
student_count.plot(kind='bar', color='skyblue', alpha=0.7)
plt.title('Number of Students by Grade')
plt.xlabel('Grade')
plt.ylabel('Number of Students')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 컬럼 이름 매핑
columns = {
    '키': 'Height',
    '몸무게': 'Weight',
    '수축기': 'Systolic Blood Pressure',
    '이완기': 'Diastolic Blood Pressure'
}

# 플롯 출력
for col, label in columns.items():
    plt.figure(figsize=(8, 6))
    averages[col].plot(kind='bar', color='skyblue', alpha=0.7)
    plt.title(f'Average {label} by Grade')
    plt.xlabel('Grade')
    plt.ylabel(label)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()  # 화면에 플롯 출력