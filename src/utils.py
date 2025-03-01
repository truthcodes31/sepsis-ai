# utils.py (Utility Functions)
import os
import pickle
import json
import pandas as pd

def save_model(model, filepath):
    """Save a trained model to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    """Load a trained model from a file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_json(filepath):
    """Load JSON data from a file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save dictionary data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def log_message(message, log_file="logs/general.log"):
    """Log messages to a log file."""
    with open(log_file, "a") as f:
        f.write(f"{message}\n")

def load_csv(filepath):
    """Load a CSV file into a Pandas DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    return pd.read_csv(filepath)

def save_csv(df, filepath):
    """Save a Pandas DataFrame to a CSV file."""
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")