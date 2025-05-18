import numpy as np
import pandas as pd

# Given data
data = [
    955, 890, 519, 707, 634, 689, 177, 404, 375, 458, 607, 704, 219, 804, 983, 848,
    808, 766, 868, 981, 803, 149, 237, 778, 671, 555, 693, 429, 486, 602, 160, 808,
    382, 852, 16, 169, 4, 300, 73, 907, 199, 548, 867, 568, 242, 737, 312, 225, 170,
    975, 539, 665, 631, 286, 78, 216, 84, 127, 19, 133, 970, 296, 622, 874, 949, 809,
    616, 379, 451, 498, 420, 442, 461, 183, 760, 766, 958, 822, 2, 891, 518, 432, 363,
    517, 561, 871, 107, 744, 529, 38, 659, 413, 674, 522, 210, 711, 987, 138, 231, 53,
    382, 456, 694, 791, 14, 284, 861, 349, 761, 478, 604, 491, 582, 473, 321, 29, 706,
    525, 221, 498, 684, 241, 602, 695, 51, 266, 342
]

# Converting data to numpy array and pandas series
np_data = np.array(data)
pd_data = pd.Series(data)

# Calculations using numpy
np_results = {
    "average": np.mean(np_data),
    "variance": np.var(np_data),
    "standard_deviation": np.std(np_data),
    "median": np.median(np_data),
    "percentile_20": np.percentile(np_data, 20),
    "percentile_80": np.percentile(np_data, 80)
}

# Calculations using pandas
pd_results = {
    "average": pd_data.mean(),
    "variance": pd_data.var(),
    "standard_deviation": pd_data.std(),
    "median": pd_data.median(),
    "percentile_20": pd_data.quantile(0.20),
    "percentile_80": pd_data.quantile(0.80)
}

# Calculate differences
differences = {key: abs(np_results[key] - pd_results[key]) for key in np_results}

# Results
results_df = pd.DataFrame({
    "Numpy": np_results,
    "Pandas": pd_results,
    "Difference": differences
})
print(results_df)