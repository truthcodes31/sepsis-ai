import pandas as pd
import os

# Define input and output file paths
input_file = "data/sepsis_data.csv"
output_file = "data/processed/sepsis_cleaned.csv"

# Check if file exists
if not os.path.exists(input_file):
    raise FileNotFoundError(f"❌ ERROR: '{input_file}' not found!")

# Load CSV file
print(f"📂 Loading dataset from {input_file}...")
df = pd.read_csv(input_file)

# Print initial dataset info
print("\n📊 Dataset Info (Before Processing):")
print(df.info())

# Ensure 'isSepsis' column exists
if 'isSepsis' not in df.columns:
    raise ValueError("❌ ERROR: 'isSepsis' column is missing!")

# Convert all columns to numeric (force errors to NaN)
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

# Print processed dataset info
print("\n✅ Dataset Processed Successfully!")
print(f"📊 Final Shape: {df.shape}")
print(df.head())

# Save the cleaned data
df.to_csv(output_file, index=False)
print(f"✅ Cleaned dataset saved to {output_file}")



