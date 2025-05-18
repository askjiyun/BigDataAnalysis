import chardet
#
# # 파일 인코딩 확인
# with open("student_health_2.csv", "rb") as file:
#     result = chardet.detect(file.read())
#     print(result['encoding']) #EUC-KR

import pandas as pd

file_path = 'student_health_2.csv'
student_health_df = pd.read_csv(file_path, encoding='euc-kr')

#print(student_health_df.head())
# count number of students per grade
students_per_grade = student_health_df['학년'].value_counts().sort_index()

print("Number of students per grade:")
print(students_per_grade)