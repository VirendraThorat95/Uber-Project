import pandas as pd
from fpdf import FPDF
import os

# Paths
uber_csv = r"C:\Users\dell8\Downloads\Uber Project\Dataset_Uber Traffic.csv"
merged_csv = r"C:\Desktop\Upgrad_Python\Uber Project\merged_uber_weather_lag.csv"
pdf_path = r"C:\Desktop\Upgrad_Python\Uber Project\Effect_of_Traffic_on_Uber_Business_Virendra.pdf"
dejavu_font_path = r"C:\Desktop\Upgrad_Python\Uber Project\DejaVuSans.ttf"

# Load datasets
if os.path.exists(merged_csv):
    df = pd.read_csv(merged_csv)
else:
    uber = pd.read_csv(uber_csv)
    df = uber.copy()
    
# Convert DateTime
if 'DateTime' in df.columns:
    df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce')

# Create PDF
pdf = FPDF()
pdf.add_page()

# Add DejaVu Unicode font
if os.path.exists(dejavu_font_path):
    pdf.add_font("DejaVu", "", dejavu_font_path, uni=True)
else:
    print(f"Font file not found at {dejavu_font_path}, using default font.")
    dejavu_font_path = None

# Title
pdf.set_font("DejaVu" if dejavu_font_path else "Arial", "B", 16)
pdf.cell(0, 10, "Effect of Traffic on Uber's Business", ln=True, align='C')
pdf.ln(5)

# Introduction
pdf.set_font("DejaVu" if dejavu_font_path else "Arial", "", 12)
intro = (
    "This report analyzes how traffic patterns impact Uber's vehicle availability, "
    "using an integrated dataset of Uber traffic and hourly weather data. "
    "Lag features and rolling weather metrics have been computed to better understand "
    "vehicle demand in relation to environmental conditions."
)
pdf.multi_cell(0, 6, intro)
pdf.ln(5)

# Dataset preview
pdf.set_font("DejaVu" if dejavu_font_path else "Arial", "B", 14)
pdf.cell(0, 10, "Dataset Preview (first 5 rows):", ln=True)
pdf.ln(2)
pdf.set_font("DejaVu" if dejavu_font_path else "Arial", "", 12)

preview = df.head(5)
for idx, row in preview.iterrows():
    row_text = f"{row['DateTime']} | Junction: {row.get('Junction', '')} | Vehicles: {row.get('Vehicles', '')} | ID: {row.get('ID', '')}"
    pdf.multi_cell(0, 6, row_text)

pdf.ln(5)

# Dataset statistics
pdf.set_font("DejaVu" if dejavu_font_path else "Arial", "B", 14)
pdf.cell(0, 10, "Dataset Statistics:", ln=True)
pdf.set_font("DejaVu" if dejavu_font_path else "Arial", "", 12)
stats = df.describe(include='all').round(2)
pdf.multi_cell(0, 6, stats.to_string())

# Save PDF
pdf.output(pdf_path)
print(f"✔️ PDF report generated and saved: {pdf_path}")
