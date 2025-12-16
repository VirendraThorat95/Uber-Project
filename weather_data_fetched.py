import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from meteostat import Point, Hourly
from datetime import datetime

# ----------------- SETTINGS -----------------
LAT, LON, ELEV = 18.6278, 73.8007, 600  # Pimpri-Chinchwad
WEATHER_START = datetime(2015, 11, 1)
WEATHER_END   = datetime(2017, 6, 30)
OUTPUT_CSV = "weather_hourly_data.csv"
# ---------------------------------------------

# 1️⃣ Fetch weather data (raw)
location = Point(LAT, LON, ELEV)
weather = Hourly(location, WEATHER_START, WEATHER_END).fetch()

# 2️⃣ Reset index + clean
weather = weather.reset_index()

# 3️⃣ Keep required columns
weather = weather[["time", "temp", "rhum", "wspd", "prcp"]]
weather.rename(columns={"time": "DateTime"}, inplace=True)

# 4️⃣ Force EXACT 1-hour intervals (no missing hours)
weather = weather.set_index("DateTime").resample("1H").mean()

# 5️⃣ Fill any missing weather values
weather = weather.interpolate(method="time").fillna(method="bfill").fillna(method="ffill")

# 6️⃣ Save final hourly weather data
weather.to_csv(OUTPUT_CSV)

print(f"✔️ Hourly weather data fetched and saved: {OUTPUT_CSV}")
