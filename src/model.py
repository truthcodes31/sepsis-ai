import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    """Train a Random Forest model to detect sepsis."""
    X = pd.read_csv("data/processed/features.csv")
    y = pd.read_csv("data/processed/labels.csv")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())

    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save model
    with open("model/sepsis_model.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_model()

