import pandas as pd
import numpy as np
from datetime import datetime

# 1. 날짜와 시간 데이터의 생성 및 변환
dti = pd.to_datetime(['1/1/2020', np.datetime64('2020-01-02'), datetime(2020, 1, 3)])
print(dti)
df = pd.DataFrame(np.random.randn(3),index=dti)
print(df)

# 시간 로컬화 및 변환
dti2 = dti.tz_localize('UTC')
print(type(dti), type(dti2))
print(dti2)
dti3=dti2.tz_convert('US/Pacific')
print(dti3)

print("---------------------")

# Resample과 시간대 변환
ind = pd.date_range('2020-01-01',periods=50, freq='H')
ser = pd.Series(range(len(ind)), index=ind)
print(ser)
print(ser.resample('3H').mean())

# 날짜 및 시간 연산
ts = pd.Timestamp('2020-02-29')
print(ts.day_name())
ts1 = ts + pd.Timedelta('1 day')
print(ts1.day_name())

print("---------------------------")
# 다양한 시계열 생성
# 시계열 데이터 생성
ser = pd.Series(range(3), index=pd.date_range('2020',freq='D', periods=3))
print(ser)
# 기간 데이터 생성
ser = pd.Series(pd.date_range('2020', freq='D',periods=3))
print(ser)

# 리샘플링과 집계
ser = pd.Series(pd.period_range('1/1/2020', freq='M',periods=3))
print(ser)
ser = pd.Series(pd.date_range('1/1/2020', freq='M', periods=3))
print(ser)
dti = pd.date_range(start='2019-01-01 10:00', freq='H', periods=3)
print(dti)
print(dti.normalize())

pi = pd.period_range('1/1/2019', '12/31/2019', freq='M')
print(pi)
ind = pd.date_range('1/1/2020', '12/31/2020', freq='BM')
print(ind)
ser = pd.Series(np.random.randn(len(ind)), index=ind)
print(ser[:2])
print(ser[:2].index)

print("-----------------------------")

print(ser['1/31/2020'])
print(ser[datetime(2020, 11, 15):])
print(ser['10/30/2020':'12/31/2020'])

print(ser['2020'])
print(ser['2020-7'])

# 대량 데이터
df=pd.DataFrame(np.random.randn(100000),
                   columns=['Val'], index=pd.date_range('20200101',
                   periods=100000, freq='T'))
print(df)
print(df['Val']['2020-3'])

#리샘플링과 집계
ind = pd.date_range('7/1/2020', periods=20, freq='S')
ser = pd.Series(np.random.randint(0, 50, len(ind)),index=ind)
print(ser)
print(ser.resample('3S').sum())

# resample 평균 계산
ind = pd.date_range('1/1/2021', freq='S', periods=700)
df = pd.DataFrame(np.random.randn(700, 3),index=ind, columns=['A', 'B', 'C'])
rs = df.resample('3T')
print(type(rs))
print(rs.mean())