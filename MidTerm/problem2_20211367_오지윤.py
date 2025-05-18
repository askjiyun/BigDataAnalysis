import pandas as pd

import numpy as np

data = pd.read_csv("kbo_baseball_test.csv")
# <------------Fill in Start------------------>

data['Rating'] = (data['Win'] / (data['Game'] - data['Draw'])) * 100
b = data.sort_values(by='Rating', ascending=False).reset_index(drop=True)

# <------------Fill in End------------------>

print("Top 5 of Winning Percentage")

for i in range(0, 5):
    print("{}, {}".format(b["Team"][i], b["Rating"][i]))