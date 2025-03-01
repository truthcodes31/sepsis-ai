# test_feature_engineering.py
import pytest
import pandas as pd
from src.feature_engineering import extract_features

def test_extract_features():
    df = pd.read_csv("data/processed/sepsis_cleaned.csv")
    X, y = extract_features(df)
    assert X.shape[0] == y.shape[0]
