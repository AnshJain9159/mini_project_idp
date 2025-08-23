# src/evaluate.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def evaluate_model():
    """
    Loads the trained model and evaluates its performance on the test set.
    Saves a confusion matrix visualization.
    """
    print("\n--- Starting Model Evaluation ---")

    # Define paths
    data_path = os.path.join('data', 'diabetes.csv')
    model_path = os.path.join('models', 'xgboost_diabetes_model.joblib')
    scaler_path = os.path.join('models', 'scaler.joblib')
    viz_dir = 'visualizations'

    # 1. Load Model and Scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 2. Load and Prepare Test Data
    df = pd.read_csv(data_path)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)

    # 3. Make Predictions
    y_pred = model.predict(X_test_scaled)

    # 4. Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    # 5. Generate and Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'], 
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    cm_path = os.path.join(viz_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    print("--- Model Evaluation Finished ---")

if __name__ == '__main__':
    evaluate_model()