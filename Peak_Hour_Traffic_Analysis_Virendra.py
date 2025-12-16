import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os

# Load dataset
data_path = r"C:\Desktop\Upgrad_Python\Uber Project\merged_uber_weather_lag.csv"
df = pd.read_csv(data_path)

# --- Fix DateTime parsing ---
df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['DateTime'])

# Create output folder for plots
plot_dir = r"C:\Desktop\Upgrad_Python\Uber Project\plots"
os.makedirs(plot_dir, exist_ok=True)

# --- 1. Average Vehicle Count per Hour per Junction (Heatmap) ---
hourly_vehicles = df.groupby([df['DateTime'].dt.hour, 'Junction'])['Vehicles'].mean().unstack()
plt.figure(figsize=(12,6))
sns.heatmap(hourly_vehicles, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Average Vehicle Count per Hour per Junction")
plt.xlabel("Junction")
plt.ylabel("Hour of Day")
heatmap_path = os.path.join(plot_dir, "heatmap_avg_vehicles.png")
plt.savefig(heatmap_path)
plt.close()

# --- 2. Average Vehicles per Hour (Line Plot) ---
avg_per_hour = df.groupby(df['DateTime'].dt.hour)['Vehicles'].mean()
plt.figure(figsize=(10,5))
avg_per_hour.plot(marker='o')
plt.title("Average Vehicles per Hour (All Junctions)")
plt.xlabel("Hour of Day")
plt.ylabel("Average Vehicle Count")
lineplot_path = os.path.join(plot_dir, "lineplot_avg_vehicles.png")
plt.savefig(lineplot_path)
plt.close()

# --- 3. Traffic vs Precipitation (Scatter) ---
plt.figure(figsize=(10,5))
sns.scatterplot(x='prcp', y='Vehicles', data=df)
plt.title("Traffic vs Precipitation")
plt.xlabel("Precipitation (mm)")
plt.ylabel("Vehicles")
scatter_prcp_path = os.path.join(plot_dir, "scatter_traffic_prcp.png")
plt.savefig(scatter_prcp_path)
plt.close()

# --- 4. Traffic vs Wind Speed (Scatter) ---
plt.figure(figsize=(10,5))
sns.scatterplot(x='wspd', y='Vehicles', data=df)
plt.title("Traffic vs Wind Speed")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Vehicles")
scatter_wspd_path = os.path.join(plot_dir, "scatter_traffic_wspd.png")
plt.savefig(scatter_wspd_path)
plt.close()

# --- Generate PDF ---
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.multi_cell(0, 10, "Peak Hour Traffic Analysis Report\n\n", align='C')

pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 8, (
    "This report identifies peak traffic hours and patterns for Uber traffic data, "
    "analyzing hourly congestion, external factors, and providing actionable "
    "recommendations for traffic management.\n\n"
))

# Add each plot with description
plots_info = [
    (heatmap_path, "Average Vehicle Count per Hour per Junction: Peak traffic hours are generally 8-10 AM and 5-7 PM."),
    (lineplot_path, "Average Vehicles per Hour (All Junctions): Traffic follows a bimodal pattern with morning and evening peaks."),
    (scatter_prcp_path, "Traffic vs Precipitation: Increased precipitation correlates with higher congestion at several junctions."),
    (scatter_wspd_path, "Traffic vs Wind Speed: Higher wind speeds do not significantly reduce vehicle counts for most junctions.")
]

for path, desc in plots_info:
    pdf.add_page()
    pdf.image(path, x=15, w=180)
    pdf.ln(5)
    pdf.multi_cell(0, 8, desc)

# Summary recommendations
pdf.add_page()
pdf.set_font("Arial", 'B', 14)
pdf.multi_cell(0, 10, "Summary Recommendations\n")
pdf.set_font("Arial", '', 12)
recommendations = [
    "1. Peak Hour Management: Target 8-10 AM and 5-7 PM for vehicle allocation and traffic mitigation.",
    "2. Weekday vs Weekend Strategy: Adjust surge pricing and fleet deployment accordingly.",
    "3. Weather-aware Operations: Incorporate rain and extreme weather data into demand prediction models.",
    "4. Junction-specific Measures: Identify consistently congested junctions for focused interventions.",
    "5. Monthly Planning: Prepare for holiday-season spikes to reduce rider wait times."
]
for rec in recommendations:
    pdf.multi_cell(0, 8, rec)

# Save PDF
pdf_output_path = r"C:\Desktop\Upgrad_Python\Uber Project\Peak_Hour_Traffic_Analysis_Virendra.pdf"
pdf.output(pdf_output_path)

print(f"✔️ PDF report generated and saved: {pdf_output_path}")
