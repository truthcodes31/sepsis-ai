# evaluate.py (Model Evaluation)
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path, test_data_path):
    """Load model, evaluate performance, and generate metrics."""
    # Load test dataset and trained model
    df = pd.read_csv(test_data_path)
    X_test = df.drop(columns=['SepsisLabel'])
    y_test = df['SepsisLabel']
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Print classification report
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # Compute and print AUC-ROC score
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"AUC-ROC Score: {roc_auc:.2f}")
    
    return {
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "roc_auc_score": roc_auc
    }

if __name__ == "__main__":
    results = evaluate_model("model/sepsis_model.pkl", "data/processed/sepsis_cleaned.csv")
    print("Model evaluation completed!")

