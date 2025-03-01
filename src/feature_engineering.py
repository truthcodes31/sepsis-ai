import pandas as pd

def extract_features(df):
    """Extract relevant features for the model."""
    features = df[['heart_rate', 'blood_pressure', 'temperature', 'oxygen_level']]
    labels = df['sepsis']
    return features, labels

if __name__ == "__main__":
    df = pd.read_csv("data/processed/sepsis_cleaned.csv")
    X, y = extract_features(df)
    X.to_csv("data/processed/features.csv", index=False)
    y.to_csv("data/processed/labels.csv", index=False)
    print("Feature extraction complete!")

