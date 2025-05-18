import pandas as pd

try:
    data = pd.read_csv('attraction_list_1.csv', encoding="utf-8")
except UnicodeDecodeError:
    data = pd.read_csv('attraction_list_1.csv', encoding="cp949")

class CustomStatistics:
    def __init__(self, data):
        self.data = data

    def calculate_mean(self):
        # Calculate mean manually
        return sum(self.data) / len(self.data)

    def calculate_std(self):
        # Calculate standard deviation manually
        mean = self.calculate_mean()
        variance = sum((x - mean) ** 2 for x in self.data) / len(self.data)
        return variance ** 0.5

# Filter data to exclude rows with "합계" in "자치구별" column and fill NaN with 0
filtered_data = data[~data['자치구별'].str.contains("합계")].copy()  # Create a copy to avoid the warning
filtered_data.fillna(0, inplace=True)  # This will not raise a warning now

# Initialize lists to collect results
results = []

# Iterate over each district and each type ("유료관광지", "내국인", "외국인")
for district in filtered_data['자치구별'].unique():
    district_data = filtered_data[filtered_data['자치구별'] == district]
    for attraction_type in ["유료관광지", "내국인", "외국인"]:
        type_data = district_data[district_data['관광지별'] == attraction_type]

        if not type_data.empty:
            # Extract the yearly values as a list for custom calculations
            values = type_data.iloc[:, 2:].astype(float).values.flatten().tolist()

            # Apply custom statistics class
            stats = CustomStatistics(values)
            mean = stats.calculate_mean()
            std = stats.calculate_std()

            # Collect results for this combination
            results.append({
                "자치구": district,
                "관광지 유형": attraction_type,
                "평균": mean,
                "표준 편차": std
            })

# Create DataFrame from results and save to CSV
sightsee_df = pd.DataFrame(results)
sightsee_df.to_csv("sightsee.csv", index=False, encoding="utf-8")

print(sightsee_df.head())