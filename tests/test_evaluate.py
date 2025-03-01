# test_evaluate.py
import pytest
from src.evaluate import evaluate_model

def test_evaluate_model():
    results = evaluate_model("model/sepsis_model.pkl", "data/processed/sepsis_cleaned.csv")
    assert "classification_report" in results
    assert "roc_auc_score" in results