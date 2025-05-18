import numpy as np
import pandas as pd
from problem1_20211367_오지윤 import CustomStatistics

# Step 1: Generate distributions and calculate statistics
# Set theoretical mean and standard deviation
mu = 0  # mean
sigma = 1  # standard deviation
target_error = 0.001  # target error

# Function to calculate mean and std until error is below target
def generate_distribution_optimized(distribution_type, mu, sigma, target_error):
    sample_size = 10000000  # Start with a large sample size for faster convergence
    if distribution_type == "normal":
        data = np.random.normal(mu, sigma, sample_size)
    elif distribution_type == "uniform":
        data = np.random.uniform(-1, 1, sample_size)
    else:
        raise ValueError("Unknown distribution type")

    stats = CustomStatistics(data)
    calculated_mean = stats.calculate_mean()
    calculated_std = stats.calculate_std()

    # Calculate errors
    mean_error = abs(calculated_mean - mu)
    std_error = abs(calculated_std - sigma if distribution_type == "normal" else 1 / np.sqrt(3))

    # Print errors to verify if they are below the target error
    print(f"{distribution_type.capitalize()} Distribution: Mean Error = {mean_error}, Std Dev Error = {std_error}")

    return {
        "distribution": distribution_type,
        "mean": calculated_mean,
        "std": calculated_std,
        "sample_size": sample_size,
        "mean_error": mean_error,
        "std_error": std_error
    }
# Regenerate stats for normal and uniform distributions
normal_stats = generate_distribution_optimized("normal", mu, sigma, target_error)
uniform_stats = generate_distribution_optimized("uniform", mu, sigma,target_error)

# save the csv
stats_df = pd.DataFrame([normal_stats, uniform_stats])
stats_df.to_csv("stat_ms.csv",  index=False)

# Step 2: Generate normal distributed data using parameters from optimized CSV
loaded_stats_optimized = pd.read_csv("stat_ms.csv")
generated_data_optimized = np.random.normal(
    loaded_stats_optimized.loc[0, 'mean'],
    loaded_stats_optimized.loc[0, 'std'],
    int(loaded_stats_optimized.loc[0, 'sample_size'])
)
# Step 3: Calculate mean and standard deviation of the newly generated normal distributed data
generated_stats = CustomStatistics(generated_data_optimized)
generated_mean = generated_stats.calculate_mean()
generated_std = generated_stats.calculate_std()

# Output results for generated data
print("Generated Normal Distribution:")
print("Mean:", generated_mean)
print("Standard Deviation:", generated_std)