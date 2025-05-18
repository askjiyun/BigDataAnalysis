import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({'A': ['ha', 'hi', 'ho', 'ha', 'ho'],
                   'B': ['one', 'two', 'one', 'one', 'two'],
                   'Data1': np.random.randn(5), #무작위 정규분보에서 생성된 숫자 데이터 열
                   'Data2': np.random.randn(5)})

print(df)

# groupby() 데이터프레임 그룹화하는 과정
# 열 A를 기준으로 그룹화 (ha, hi, ho로 그룹화)
group1 = df.groupby('A')
print(group1)
# 그룹별 데이터 출력
# list : 각 그룹의 이름과 해당 그룹 데이터를 쌍으로 이루어진 리스트 반환
print(dict(list(group1)))

group_dict1 = dict(list(group1))
# 특정 그룹 데이터 출력
print(group_dict1['ha'])
# 그룹 정보 출력
print(group1.groups)
#print(group1.get_group('ha','hi'))

group2 = df.groupby('B')
print(group2)
print(dict(list(group2)))

group_dict2 = dict(list(group2))
print(group_dict2['one'])
print(group2.get_group('one'))

# B열을 기준으로 그룹화
group3 = df['Data2'].groupby(df['B'])
print(df['B'])
# 그룹별 평균 계산
print(group3.mean())

group3_1 = df['Data2'].groupby([df['A'],df['B']])
#print(df['B'])
print(group3_1.mean())


material = np.array(['water', 'oil', 'oil' ,'water', 'oil'])
time = ['1hr', '1hr', '2hr', '2hr', '1hr']
# material 과 time의 조합을 기준으로 그룹화 & 평균 계산
df['Data1'].groupby([material, time]).mean()

df2 = pd.DataFrame({'A': ['ho', 'hi', 'ha'],
                     'B': ['two', 'one', 'two'],
                      'Data1': np.random.randn(3)})
# A 기준으로 그룹화 & 합계 계산
print(df2.groupby(['A']).sum())
# 열 A를 기준으로 그룹화
# sort=False : 그룹화된 데이터의 순서를 원래 데이터 순서대로 유지
print(df2.groupby('A', sort=False).sum())

arr = [['ha', 'ha', 'hi', 'hi', 'ho', 'ho'], ['one', 'two', 'one', 'one', 'two', 'two']]
# MultiIndex : 다중 인덱스 생성
ind = pd.MultiIndex.from_arrays(arr, names = ['1st', '2nd'])
ser = pd.Series(np.random.randn(6), index=ind)
# 그룹화 이후 특정 그룹 데이터 추출
print(list(ser.groupby('1st').get_group('ha')))
print(list(ser.groupby('2nd').get_group('two')))

print("---------------------------------------------")

grouped1 = df.groupby('A')

for name, group in grouped1:
    print(name)
    print(group)

print(grouped1.agg(np.sum))
print(grouped1.size())

print(grouped1['Data1'].agg([np.sum, np.mean, np.std]))


print(grouped1['Data1'].agg([np.sum, np.mean])
      .rename(columns={'sum': '합계', 'mean': '평균'}))

# 지수값을 가지는 Series 생성
ser = pd.Series(np.exp(np.arange(1,500)))
ser
# 퍼센트 변화 계산
ser.pct_change()
# 3개 이전 값과 비교하여 변화율 계산
ser.pct_change(periods=3)

import matplotlib.pyplot as plt
ser.plot()
plt.show()

ser.pct_change().plot()
plt.show()

ser.pct_change(periods=3).plot()
plt.show()

# 두 Series 간 상관관계
ser1 = pd.Series(np.random.randn(10000))
ser2 = pd.Series(np.random.randn(10000))
print(ser1.corr(ser2))

# 공분산 및 상관계수 계산
df = pd.DataFrame(np.random.randn(1000, 3),
                   columns=['a', 'b', 'c'])
print(df.cov())
# 상관계수 계산
print(df.corr())

# 결측값 포함 데이터 프레임의 상관계수 계산
# 20행 3열 정규분포 ㅐㅇ성
df1 = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
# 처음 5행의 'a' 열에 결측값 할당
df1.loc[df1.index[:5], 'a'] = np.nan
# 5-9번째까지의 'b' 열에 결측값 NaN 할당
df1.loc[df1.index[5:10], 'b'] = np.nan
# 기본 상관계수 할당
df1.corr()
# 최소 데이터 갯수 지정
df1.corr(min_periods=12)

ind = ['a', 'b', 'c', 'd']
col = ['one', 'two', 'three']
df1 = pd.DataFrame(np.random.randn(4, 3), index=ind, columns=col)
df2 = pd.DataFrame(np.random.randn(3, 3), index=ind[:3], columns=col)
# 한 데이터프레임의 열 간 Pearson 상관계수 계싼
print(df1.corrwith(df2))
print(df2.corrwith(df1, axis=1))

print("----------------------")
# Series의 순위 계산
ser = pd.Series(np.random.randn(5), index=list('abcde'))
print(ser)
print(ser.rank())
ser['d'] = ser['b']
print(ser.rank())

# 누적합 및 롤링 평균
s = pd.Series(np.random.randn(1000),
              index=pd.date_range('1/1/2020', periods=1000))
ser = s.cumsum()

roll = ser.rolling(window=60)
roll
type(roll)
roll.mean()
# 검은색 실선으로 표시
ser.plot(style='k')
roll.mean().plot(style='k-')
plt.show()

# A,B,C 열의 값이 서로 다른 선으로 표시되는 3개의 라인
df = pd.DataFrame(np.random.randn(1000, 3), index=pd.date_range('1/1/2020', periods=1000), columns=['A', 'B', 'C'])
df.plot()
plt.show()

