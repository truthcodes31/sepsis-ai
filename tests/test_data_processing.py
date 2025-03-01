# test_data_processing.py
import pytest
import pandas as pd
from src.data_processing import load_data, clean_data

def test_load_data():
    df = load_data("data/raw/sepsis_data.csv")
    assert isinstance(df, pd.DataFrame)

def test_clean_data():
    df = pd.DataFrame({"col1": [1, 2, None], "col2": [None, 3, 4]})
    cleaned_df = clean_data(df)
    assert cleaned_df.isnull().sum().sum() == 0
