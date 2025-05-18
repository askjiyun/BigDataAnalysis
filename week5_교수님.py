import numpy as np
import pandas as pd

# Series
ser = pd.Series([1,2,3,4,5], index=['a','b','c','d','e'])
# print(ser)
# print(ser.values)
# print(ser.index)
# print(ser.shape)
#
ser1 = pd.Series([2,3,4,5,6], index=['b','c','d','e','f'])
#print(ser1)
#
# # 파이썬 딕셔너리 정의 (세개의 키-값 쌍으로 구성)
# da = {'a' : 0., 'b' : 1., 'c' : 2.}
# print(pd.Series(da))
# # series 객체는 해당 순서에 맞춰 인덱스 적용
# # 'd'는 da에 존재하지 않는 키 -> NaN으로 채워짐
# print(pd.Series(da, index=['b', 'c', 'd', 'a']))
#
# # Scalar 값으로 Series 생성
# # 7.0으로 모든 인덱스 값 채우기
# print(pd.Series(7., index=['a', 'b', 'c']))
# # Same Series with ndarray
# # Series 객체의 첫번째 요소 반환
# #print(ser[0])
# print(ser[:3])
# # 지수함수 적용
# print(np.exp(ser))
#
# # dictionary
# print(ser['a'])
# ser['d'] = 7
# print(ser)
#
# # Vectorized Operation & Labels Alignments
# # Series object can do 수학적 계산
# print(ser + ser)
# print(ser*2)
# # data auto alignment with labels
# print(ser[1:]+ ser[:-1])
# # name properties - 시리즈 객체에 이름 설정
# ser = pd.Series(np.random.randn(5), name='seoul')
# print(ser)
# ser1 = ser.rename('busan')
# print(ser1.name)

# DataFrame
# Series 이용해서 DF 생성
df = pd.DataFrame([ser,ser, ser1, ser1], index=['a1', 'b1','c1','d1'])
print(df)

print(df.values)
print(df.index)


# Seires 생성
da = {'a' : 0., 'b' : 1., 'c' : 2.}
# da의 key = a,b,c
print(pd.Series(da))
# index = ['b','c','d','a']로 시리즈 생성
print(pd.Series(da, index=['b', 'c', 'd', 'a']))

# 3x4 형태의 정규분포 난수로 구성된 DF 생성
# 각 요소에 대해 10의 거듭제곱 적용
print(np.power(10,pd.DataFrame(np.random.randn(12).reshape(3,4))))

# series 슬라이싱
print(ser[0]) # 1
print(ser['a']) #1

print(ser[0:2])
print(ser['a':'c'])

# Series 객체 이름 설정
ser2 = pd.Series(np.random.randn(5),  name='seoul')
print(ser2)
# 시리즈 객체 이름 변경
ser3 = ser.rename('busan')
print(ser3)

# pd.Series 객체를 담은 딕셔너리
# pandas는 인덱스의 합집합을 이용하여 DF 생성
d = {
    'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
    'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])
}
df = pd.DataFrame(d)
print(df)

df1 = pd.DataFrame(d, index=['a','c','d'], columns=['one','two'])
print(df1)
df2 = pd.DataFrame(d, index=['b','d','e'], columns=['three','two'])
print(df2)
df3 = df1 + df2
print(df3)
#
# Numpy 구조화 배열 생성
arr = np.zeros((2,), dtype=[('A', 'i4'), ('B', 'f4'), ('C', 'a10')])
# 배열의 값을 지정하여 초기화
arr[:] = [(1, 2., 'Hello'), (2, 3., 'World')]
print(arr)
print(pd.DataFrame(arr))
# 인덱스 지정
print(pd.DataFrame(arr, index=['first', 'second']))
# 컬럼 순서 변경
print(pd.DataFrame(arr, columns=['C', 'A', 'B']))

# 리스트와 딕셔너리로 DF 생성
data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
print(pd.DataFrame(data))

# pandas 멀티 인덱스와 멀티 컬럼 구조 사용
data = pd.DataFrame({
    ('a', 'b'): {('A', 'B'): 1, ('A', 'C'): 2},
    ('a', 'a'): {('A', 'C'): 3, ('A', 'B'): 4},
    ('a', 'c'): {('A', 'B'): 5, ('A', 'C'): 6},
    ('b', 'a'): {('A', 'C'): 7, ('A', 'B'): 8},
    ('b', 'b'): {('A', 'D'): 9, ('A', 'B'): 10}
})
print(data)


# 데이터프레임 생성
d = {
    'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
    'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])
}
df = pd.DataFrame(d)
print(df)

# 개별 칼럼 접근
print(df['one'])
print(df['two'])
# 'three' 컬럼: 'one'과 'two' 컬럼의 곱
df['three'] = df['one'] * df['two']
# 'flag' 컬럼: 'one' 컬럼의 값이 2보다 큰지 여부 (True/False)
df['flag'] = df['one'] > 2
# 'four' 컬럼: 0으로 고정
df['four'] = 0
# 'whatever' 컬럼: 'A+'로 고정
df['whatever'] = 'A+'
print(df)
# 'flag' 칼럼 삭제
del df['flag']
# 'whatever' 칼럼 제거
# pop은 제거한 컬럼 반환이므로 필요한 경우 다른 변수에 저장 가능
df.pop('whatever')
print(df)
#  'truncated_one' 컬럼 추가 (첫 두 행만 복사)
df['truncated_one'] = df['one'][:2]
print(df)
#  'hi' 컬럼을 두 번째 위치에 삽입
df.insert(1, 'hi', df['one'])
print(df)
# 인덱스 'd'인 행 삭제
df=df.drop(["d"],axis=0)
print(df)
# 특정 라벨을 기반으로 행 접근
#print(df.loc('a'))
# 특정 행 접근
print(df.iloc(1))

# DF 연산 및 전치
df = pd.DataFrame(np.random.randn(5, 4), columns=['A', 'B', 'C', 'D'])
df2 = pd.DataFrame(np.random.randn(3, 3), columns=['A', 'B', 'C'])
print(df + df2)
print(df)
print(df.T)

# 2x6 형태의 Pandas DF 생성
df = pd.DataFrame(np.arange(12).reshape(2, 6), columns=list('ABCDEF'))
print(df)
print(df.index)
# 길이가 1000인 시리즈 생성
ser = pd.Series(np.random.randn(1000))
# 처음 5개 요소 반환
print(ser.head())
# 마지막 3개 요소 출력
print(ser.tail(3))

# 날짜 인덱스 생성 및 DF 생성
# 2019/1/1부터 시작해 15일간 날짜 인덱스 생성
ind = pd.date_range('1/1/2019', periods=15)
ser = pd.Series(np.random.randn(15))
df = pd.DataFrame(np.random.randn(15, 3), index=ind, columns=['A', 'B', 'C'])
print(df)
#
print("----------------------------------------------------------")

# 데이터프레임 생성
df = pd.DataFrame({
    'one': pd.Series(np.random.randn(2), index=['a', 'b']),
    'two': pd.Series(np.random.randn(3), index=['a', 'b', 'c']),
    'three': pd.Series(np.random.randn(2), index=['b', 'c'])
})
print(df)
# 특정 행 (1번째 위치) 선택
print(df.iloc[1])
# 특정 열('two') 선택
print(df['two'])
# 특정 행과 열 선택
row = df.iloc[1]
col = df['two']
print(row, col)
# 라벨 기반 인덱싱으로 인덱스 'b' 선택
print(df.loc['b'])

# 첫 번째 데이터프레임 생성
d = {
    'one': [1., 2., np.nan],
    'two': [3., 2., 1.],
    'three': [np.nan, 1., 1.]
}
df = pd.DataFrame(d, index=list('abc'))
print(df)

# 두 번째 데이터프레임 생성
d1 = {
    'one': pd.Series([1., 2.], index=['a', 'b']),
    'two': pd.Series([1., 1., 1.], index=['a', 'b', 'c']),
    'three': pd.Series([2., 2., 2.], index=['a', 'b', 'c'])
}
df1 = pd.DataFrame(d1)
print(df1)

# 덧셈 연산 - 동일 위치 값 더하기
# 두 데이터 프레임 중 하나라도 NaN이 있는 경우 해당 위치 결과도 NaN
print(df+df1)
# NaN 값을 0으로 대체 - 결측값 방지
print(df.add(df1, fill_value=0))

# 데이터프레임 생성
df = pd.DataFrame({
    'angles': [0, 3, 4],
    'degrees': [360, 180, 360]
}, index=['circle', 'triangle', 'rectangle'])

# 인덱스를 기준으로 시리즈의 값을 각 행에서 빼기
df1 = df.sub(pd.Series([1, 2, 3], index=['circle', 'triangle', 'rectangle']), axis='index')

print(df1)
#
print("-----------------------------------")
# 각 행 평균 계산
print(df1.mean(1))
# 각 열의 합계 계산
# skipna = True - 결측값 무시
print(df1.sum(0,skipna=True))
# 각 열의 표준편차 계산
print(df1.std())
# Numpy 배열로 변환 후 표편 계산
print(df1.to_numpy().std(0))

# 시리즈 생성 및 통계 연산
ser = pd.Series(np.random.randn(500))
# 20-499까지 NAN
ser[20:500] = np.nan
# 10-19 5로 설정
ser[10:20] = 5
# 고유한 값의 개수 계산 - NaN 제외
print(ser.nunique()) # 2
# 시리즈의 통계 요약 정보 - 카운트, 평균, 표편, 최솟값, 사분위 수, 최댓값 반환
print(ser.describe())

# Series의 통계 요약 (특정 백분위수 포함)
ser = pd.Series(np.random.randn(1000))
# 짝수 인덱스를 NaN으로 설정
ser[::2] = np.nan
# 특정 백분위수를 포함하여 Series의 통계 요약 출력
print(ser.describe(percentiles=[0.10, 0.25, .30, .65, .78]))

# 범주형 데이터 Series 통계 요약
ser = pd.Series(['a','a','b','c','c',np.nan, 'c', 'd'])
# count, unique, top, freq, dtype
print(ser.describe())
# DF의 통계 요약 (범주형 포함)
df = pd.DataFrame({'a': ['Yes', 'Yes', 'No', 'No'], 'b': range(4)})
# 모든 컬럼(수치형과 범주형 포함)
print(df.describe(include='all'))

# 랜덤 정수 데이터 생성 & Series 변환
data = np.random.randint(0, 7, size=30)
ser1 = pd.Series(data)
# 값의 빈도 계산
print(ser1.value_counts())
print(pd.value_counts(data))

# DF 날짜 인덱스 사용
df1 = pd.DataFrame(np.random.randn(5, 4), columns=list('ABCD'),
                       index=pd.date_range('20190701', periods=5))
print(df1)
# 특정 날짜 범위의 행 선택
print(df1.loc['20190702':'20190703'])

# 데이터 프레임 생성
df1 = pd.DataFrame(np.random.randn(5, 4), index=list('abcde'), columns=list('ABCD'))
print(df1)
# 특정 행과 열 선택
df1_row = ['a', 'b', 'd']
df1_column = ["B","C"]
df2 = df1.loc[df1_row, df1_column]
print(df2)

print("--------------------")
# Series 생성
ser = pd.Series(list('abcde'), index=[0, 3, 2, 5, 4])
print(ser)
# loc로 인덱스 3부터 5까지의 값 선택
print(ser.loc[3:5])
# iloc로 위치 기반 슬라이싱
print(ser.iloc[1:4])
# 인덱스 기준으로 정렬 - 오름차순
print(ser.sort_index())
# 값을 기준으로 정렬 - 오름차순
print(ser.sort_values())
print("--------------------")
print(df2)
# 만든 DF에서 컬럼 B 선택
print(df2.B)
# 만든 DF에서 컬럼 C 선택
print(df2.C)


df = pd.DataFrame(np.arange(9).reshape(3, 3),   columns=['A', 'B', 'C'])
print(df)
# 'D' 컬럼 추가 (컬럼 'A'를 복사하여 추가)
df.loc[:, 'D'] = df.loc[:, 'A']
# 새로운 행 추가 (3번째 행에 기존의 2번째 행을 복사)
df.loc[3, :] = df.loc[2, :]
print(df)
print("-----------------------")
# 4. DataFrame 연산 및 결측값 처리
df3 = df1 - df2
print(df3)
# 5. 결측값을 앞쪽 값으로 채우기 (ffill)
print(df3.fillna(method='ffill'))
# 6. 특정 열의 결측값이 있는 컬럼 제거
print(df3.loc[:,"B":"C"].dropna(axis=1))
# 7. 결측값을 특정 값으로 대체
print(df3.replace(np.nan,1.0))