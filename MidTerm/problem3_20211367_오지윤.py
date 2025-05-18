import pandas as pd
import numpy as np

df1_data = np.ones((6, 7))
df1 = pd.DataFrame(df1_data, index=list('abcdef'), columns=list('ABCDEFG'))

df2_data = np.ones((7, 6))
df2 = pd.DataFrame(df2_data, index=list('abcdefg'), columns=list('ABCDEF'))

df3 = df1 + df2

# <----------- Fill in start------------>
# 추가된 4줄
df3.loc[['a', 'd', 'e'], ['A', 'D', 'E', 'G']] = np.nan
df3.loc['g', :] = np.nan
df3.loc['a':'c', ['A', 'B', 'C']] = 3
df3.loc['f', :] = 2
print(df3)
# <----------- Fill in End ------------>
