# Sepsis Detection AI

## Overview
Sepsis is a life-threatening medical emergency that requires early detection to improve survival rates. This project leverages **machine learning** and **AI** to predict sepsis risk based on patient vitals and laboratory results.

## Features
- **Real-time prediction:** Uses a trained model to assess sepsis risk.
- **Web dashboard:** A user-friendly interface for healthcare professionals.
- **API for integration:** Flask-based API for seamless integration with hospital systems.
- **Data processing pipeline:** Automated feature extraction and preprocessing.
- **Model evaluation:** Provides performance metrics including accuracy, precision, recall, and AUC-ROC.

## Project Structure
```
sepsis-ai/
│
├── data/                        # Dataset storage
│   ├── raw/                     # Raw input data
│   ├── processed/               # Preprocessed and cleaned data
│   ├── external/                # Additional datasets (if applicable)
│   └── README.md                # Data structure explanation
│
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb  # Data exploration & visualization
│   ├── 02_model_training.ipynb    # Model training process
│   ├── 03_model_evaluation.ipynb  # Model evaluation and analysis
│
├── src/                           # Source code
│   ├── __init__.py                # Module initialization
│   ├── data_processing.py         # Data preprocessing functions
│   ├── feature_engineering.py     # Feature transformation and selection
│   ├── model.py                   # Model training and inference functions
│   ├── evaluate.py                # Model evaluation functions
│   ├── utils.py                   # Helper functions (logging, file handling, etc.)
│   ├── api.py                     # Flask API for real-time predictions
│   ├── dashboard.py               # Streamlit-based web dashboard
│   └── train_model.py             # Script for training the ML model
│
├── tests/                         # Unit tests
│   ├── __init__.py                # Test module initialization
│   ├── test_data_processing.py    # Tests for data processing functions
│   ├── test_model.py              # Tests for model training & inference
│   ├── test_feature_engineering.py # Tests for feature engineering
│   ├── test_evaluate.py           # Tests for model evaluation
│   ├── test_api.py                # Tests for API endpoints
│   └── test_dashboard.py          # Tests for dashboard UI
│
├── logs/                          # Log files for debugging
│   ├── application.log            # Consolidated log file
│
├── model/                         # Trained model storage
│   ├── sepsis_model.pkl           # Machine learning model
│   ├── pipeline_minmax.pkl        # Feature scaling pipeline
│   ├── README.md                  # Documentation for model usage
│
├── config.yaml                    # Configuration settings
├── requirements.txt                # Dependencies list
├── Dockerfile                      # Docker configuration
├── .gitignore                      # Files to ignore in version control
├── README.md                       # Main project documentation
```

## Installation
To set up the project, follow these steps:
```bash
# Clone the repository
git clone https://github.com/yourusername/sepsis-ai.git
cd sepsis-ai

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
### 1. Train the Model
```bash
python src/train_model.py
```
### 2. Start the API
```bash
python src/api.py
```
### 3. Run the Dashboard
```bash
streamlit run src/dashboard.py
```

## API Endpoints
- **`POST /predict`** → Accepts patient vitals and returns sepsis risk.
Example request:
```json
{
    "features": [80, 120, 37.5, 95]
}
```
Example response:
```json
{
    "sepsis_risk": 1
}
```

## Model Performance
The model is evaluated using:
- **Accuracy**
- **Precision, Recall, F1-score**
- **AUC-ROC Score**

## Future Improvements
- Improve model accuracy using deep learning techniques.
- Deploy on cloud infrastructure for real-time predictions.
- Integrate with hospital EHR systems for automated monitoring.

## Contributors
- **Your Name** - *Lead Developer*
- **Contributor Name** - *Data Scientist*

## License
This project is licensed under the MIT License. See `LICENSE` for details.

