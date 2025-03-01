import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# ✅ Define the correct dataset path
input_file = r"C:\Users\shand\OneDrive\Desktop\sepsis-ai\data\raw\sepsis_data.csv"
output_model = r"C:\Users\shand\OneDrive\Desktop\sepsis-ai\model\sepsis_model.pkl"

# ✅ Check if file exists
if not os.path.exists(input_file):
    raise FileNotFoundError(f"❌ ERROR: '{input_file}' not found!")

# ✅ Load dataset
print(f"📂 Loading dataset from {input_file}...")
df = pd.read_csv(input_file)

# ✅ Ensure dataset is not empty
if df.empty:
    raise ValueError("❌ ERROR: The dataset is empty!")

# ✅ Print missing values per column
print("\n🔍 Missing Values Count (Before Handling):")
print(df.isnull().sum())

# ✅ Ensure 'SepsisLabel' column exists
if 'SepsisLabel' not in df.columns:
    raise ValueError("❌ ERROR: 'SepsisLabel' column is missing!")

# ✅ Convert all columns to numeric and fill missing values
df = df.apply(pd.to_numeric, errors='coerce')

# ✅ Use mean imputation for missing values
imputer = SimpleImputer(strategy="mean")
df.iloc[:, :-2] = imputer.fit_transform(df.iloc[:, :-2])

# ✅ Print missing values after handling
print("\n✅ Missing Values After Handling:")
print(df.isnull().sum())

# ✅ Split features and target variable
X = df.drop(columns=['SepsisLabel', 'Patient_ID'])  # Drop ID column too
y = df['SepsisLabel']

# ✅ Print dataset shape
print(f"\n✅ Data Loaded - Features Shape: {X.shape}, Labels Shape: {y.shape}")

# ❌ Stop if X or y is empty
if X.empty or y.empty:
    raise ValueError("❌ ERROR: X or y is empty after processing! Check dataset formatting.")

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if X_train.empty or y_train.empty:
    raise ValueError("❌ ERROR: Train dataset is empty after split!")

# ✅ Train the model
print("🔄 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# ✅ Ensure model directory exists before saving
os.makedirs(os.path.dirname(output_model), exist_ok=True)

# ✅ Save the trained model
with open(output_model, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved as '{output_model}'")
