import numpy as np
import pandas as pd

# save as CSV
data = {'name': ['haena', 'naeun', 'una', 'bum', 'suho'],
                             'age': [30, 27, 28, 23, 18],
                             'address': ['dogok', 'suwon', 'mapo', 'ilsan', 'yeoyi'],
                             'grade': ['A', 'B', 'C', 'B', 'A'],
                             'score': [100, 88, 73, 83, 95]}

df = pd.DataFrame(data, columns=['name', 'age', 'address', 'score', 'grade'])
print(df)

print("-----------CSV 저장 방법-----------")
# csv 파일 저장 방법
df.to_csv("ex1.csv", index=False, header=None)

# column 이름이 없을 경우 작성해주기
# skiprows, nrows 등 다양한 argumnet 존재
# skiprow = 2 - 행 두개 건너뛰기
# names = > column 이름이 없을 경우 작성
df1=pd.read_csv("ex1.csv", index_col=[1],
                names=['No.', 'age', 'address', 'score', 'grade'], nrows=4, skiprows=2)
print(df1)

print("------------------------------------")

# JSON 파일 형식
# 데이터 보내고 받을 때 주로 사용
dfj = pd.DataFrame([['a', 'b'], ['c', 'd']], index=['row1', 'row2'], columns=['col1', 'col2'])
dfj.to_json("ex2.json")
dfj.to_json("ex2.json",orient='split')
df.to_json("ex2.json")
print(pd.read_json("ex2.json"))

# HTML 형식
url = 'http://www.fdic.gov/bank/individual/failed/banklist.html'
dfh = pd.read_html(url)

df.to_html("ex3.html")
df2 = pd.read_html("ex3.html")
print(pd.DataFrame(np.array(df2)[0]))

df.to_excel("ex4.xlsx")
df3 = pd.read_excel("ex4.xlsx")
print(df3)

#del df3["Unnamed: 0"]
df3 = df3.iloc[:,1:]
print(df3)