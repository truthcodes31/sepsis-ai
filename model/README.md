# model/README.md (Model Documentation)

# Sepsis Detection AI - Model Documentation

## Overview
The **Sepsis Detection AI Model** is designed to predict sepsis risk based on patient vital signs and lab results. It uses machine learning to analyze patient data and provide early warnings for potential sepsis cases.

## Model Details
- **Algorithm:** RandomForestClassifier
- **Number of Trees (n_estimators):** 100
- **Random State:** 42
- **Training Data:** Processed dataset from `data/processed/sepsis_cleaned.csv`
- **Feature Scaling:** MinMax Scaling (pipeline_minmax.pkl)

## Files in this Directory
- **sepsis_model.pkl** → Trained machine learning model.
- **pipeline_minmax.pkl** → Feature scaling pipeline used during training.
- **README.md** → This documentation file.

## Usage
### Loading the Model
To use the trained model for predictions, load it as follows:
```python
import pickle
import numpy as np

# Load the model
with open("model/sepsis_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the scaler
with open("model/pipeline_minmax.pkl", "rb") as f:
    scaler = pickle.load(f)

# Example input data (features should match training data format)
data = np.array([[80, 120, 37.5, 95]])  # Example: Heart rate, BP, Temp, Oxygen Level
scaled_data = scaler.transform(data)

# Make a prediction
prediction = model.predict(scaled_data)
print(f"Sepsis Risk: {'High' if prediction[0] == 1 else 'Low'}")
```

## Model Performance
- **Accuracy:** Evaluated using test data.
- **Metrics Used:** Precision, Recall, F1-Score, AUC-ROC.
- **Evaluation Results:** Stored in `logs/application.log`.

## Future Improvements
- Integrate deep learning models for better accuracy.
- Collect real-time patient data for continuous learning.
- Deploy as a cloud-based API for hospital integration.

For any questions, refer to the main project **README.md**.