import pandas as pd
import numpy as np

# Data Concatenation
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2'],
                     'C': ['C0', 'C1', 'C2']}, index=[0, 1, 2])
df2 = pd.DataFrame({'A': ['A3', 'A4', 'A5'],
                     'B': ['B3', 'B4', 'B5'],
                     'C': ['C3', 'C4', 'C5']}, index=[3, 4, 5])
df3 = pd.DataFrame({'A': ['A6', 'A7', 'A8'],
                     'B': ['B6', 'B7', 'B8'],
                     'C': ['C6', 'C7', 'C8']}, index=[6, 7, 8])

frame1 = [df1,df2,df3]

result = pd.concat(frame1, axis=1)
result1 = pd.concat(frame1, keys=['x','y','z'])
print(result)
print(result1)
print(result1.iloc[0])

df4 = pd.DataFrame({'B': ['B2', 'B6', 'B7'],
                          'C': ['C2', 'C6', 'C7'],
                          'E': ['E2', 'E6', 'E7']}, index=[2, 6, 7])

result = pd.concat([df1, df4], axis=1, sort=False, join='inner')
print(result)
result1 = pd.concat([df1, df4], sort=False)
print(result1)

# Merge - inner & outer
# 병합 방법에는 inner / outer / right inner / left inner 등 다양함
df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'],
                        'value': np.random.randn(4)})
df2 = pd.DataFrame({'key': ['B', 'D', 'D', 'E'],
                                                    'value': np.random.randn(4)})
print(pd.merge(df1, df2, on='key', how='outer'))

# Combine
df1 = pd.DataFrame([[np.nan, 3., 5.], [-4.6, np.nan, np.nan], [np.nan, 7., np.nan]])
df2 = pd.DataFrame([[-2.6, np.nan, -8.2], [-5., 1.6, 4]], index=[1, 2])
result = df1. combine_first(df2)
result1 = df2.combine_first(df1)
print(result)
print(result1)

# Text Data Operation
ser = pd.Series(['Suho', 'AA', np.nan, 'rabbit'])
print(type(ser.str))
print(ser.str.lower())
print(ser.str.upper())
print(ser.str.len())


ind = pd.Index(['  ha', 'hi  ', '   ho ', 'hu'])
# 공백 제거
print(ind.str.strip())
# 왼쪽 공백 제거
print(ind.str.lstrip())
# 오른쪽 공백 제거
print(ind.str.rstrip())

df = pd.DataFrame(np.random.randn(2, 2), columns=[' Column A ',' Column B '], index=range(2))
print(df)
print(df.columns)


print(df.columns.str.strip().str.lower())
print(df.columns.str.lower())


ser1 = pd.Series(['ha_a_b', 'hi_c_d', np.nan, 'ho_e_f'])
print(ser1.str.split('_'))