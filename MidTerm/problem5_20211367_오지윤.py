import pandas as pd

data = pd.read_csv("seoul_ems_test.csv",)

# Calculate the sum between 2012 and 2017 for each region
data['Sum_2012_2017'] = data.loc[:, '2012':'2017'].sum(axis=1)

# Check if "Total" value matches the sum and calculate the gap if not
data['Match'] = data['Total'] == data['Sum_2012_2017']
data['Gap'] = data['Total'] - data['Sum_2012_2017']

# Displaying only relevant columns for mismatch details
mismatch_data = data[['Region ID', 'Total', 'Sum_2012_2017', 'Match', 'Gap']]
print(mismatch_data.to_string(index=False))