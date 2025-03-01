import pytest
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
MODEL_PATH = "model/sepsis_model.pkl"

def test_model_file_exists():
    """Check if the trained model file exists"""
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        assert isinstance(model, RandomForestClassifier)
        print("✅ Model file exists and is valid.")
    except FileNotFoundError:
        pytest.fail("❌ Model file 'sepsis_model.pkl' is missing!")
    except Exception as e:
        pytest.fail(f"❌ Error loading model: {e}")

def test_model_prediction():
    """Test if the model can make predictions"""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Create a dummy test input based on expected features
    test_input = np.array([[75, 100, 37.5, 120, 80, 60, 18, 0.5, 24, 20, 0.9, 7.4, 40, 98, 30, 15, 200, 9.5, 100, 1.2]])
    
    # Ensure input shape matches expected model input
    assert model.n_features_in_ == test_input.shape[1], "❌ Model expects a different number of features!"

    # Run prediction
    prediction = model.predict(test_input)
    
    # Check output validity
    assert prediction in [0, 1], "❌ Model should output either 0 (No Sepsis) or 1 (Sepsis)"
    print(f"✅ Model prediction successful: {prediction}")