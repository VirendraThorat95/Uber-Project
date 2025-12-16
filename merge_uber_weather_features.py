import pandas as pd

# ---------------------------
# Load datasets
# ---------------------------
weather_path = r"C:\Desktop\Upgrad_Python\weather_hourly_data.csv"
uber_path = r"C:\Users\dell8\Downloads\Uber Project\Dataset_Uber Traffic.csv"

weather = pd.read_csv(weather_path)
uber = pd.read_csv(uber_path)

# ---------------------------
# Inspect weather columns
# ---------------------------
print("Weather columns:", weather.columns)

# ---------------------------
# Convert DateTime columns to datetime objects
# ---------------------------
# Weather dataset
if 'time' in weather.columns:
    weather['time'] = pd.to_datetime(weather['time'])
    merge_on_weather = 'time'
elif 'DateTime' in weather.columns:
    weather['DateTime'] = pd.to_datetime(weather['DateTime'])
    merge_on_weather = 'DateTime'
else:
    raise KeyError("No suitable datetime column found in weather dataset.")

# Uber dataset
uber['DateTime'] = pd.to_datetime(uber['DateTime'], dayfirst=True, errors='coerce')

# ---------------------------
# Merge datasets on DateTime
# ---------------------------
merged_df = pd.merge(uber, weather, left_on='DateTime', right_on=merge_on_weather, how='left')

# Drop the duplicate datetime column from weather if needed
if merge_on_weather != 'DateTime':
    merged_df.drop(columns=[merge_on_weather], inplace=True)

# Fill missing values
merged_df.ffill(inplace=True)
merged_df.bfill(inplace=True)

# Sort by Junction and DateTime
merged_df.sort_values(['Junction', 'DateTime'], inplace=True)

# ---------------------------
# Create lag features for Vehicles (1h, 2h, 3h)
# ---------------------------
for lag in [1, 2, 3]:
    merged_df[f'Vehicles_lag_{lag}h'] = merged_df.groupby('Junction')['Vehicles'].shift(lag)

# ---------------------------
# Create rolling mean features for weather
# ---------------------------
weather_cols = [col for col in weather.columns if col not in ['time', 'DateTime']]
for col in weather_cols:
    for window in [3, 6]:
        merged_df[f'{col}_roll_{window}h'] = merged_df.groupby('Junction')[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

# ---------------------------
# Save enhanced merged dataset
# ---------------------------
output_path = r"C:\Desktop\Upgrad_Python\Uber Project\merged_uber_weather_lag.csv"
merged_df.to_csv(output_path, index=False)

print(f"✔️ Merged dataset with lag & rolling weather features saved: {output_path}")

# ---------------------------
# Optional: Preview first 5 rows
# ---------------------------
print("\nPreview of merged dataset:")
print(merged_df.head())
