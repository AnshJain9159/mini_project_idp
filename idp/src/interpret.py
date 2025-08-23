# src/interpret.py

import pandas as pd
import shap
import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

def interpret_model():
    """
    Loads the trained model and generates SHAP visualizations to interpret
    its predictions, saving them to the 'visualizations' directory.
    """
    print("\n--- Starting Model Interpretation ---")

    # Define paths
    data_path = os.path.join('data', 'diabetes.csv')
    model_path = os.path.join('models', 'xgboost_diabetes_model.joblib')
    scaler_path = os.path.join('models', 'scaler.joblib')
    viz_dir = 'visualizations'
    
    # Create visualizations directory if it doesn't exist
    os.makedirs(viz_dir, exist_ok=True)

    # 1. Load Model and Scaler
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded.")

    # 2. Load and Prepare Test Data
    # We need the original data to get the test set and feature names
    df = pd.read_csv(data_path)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the test data using the saved scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled data back to a DataFrame to preserve feature names for SHAP plots
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # 3. Instantiate SHAP Explainer
    print("Instantiating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    # 4. Calculate SHAP Values
    print("Calculating SHAP values for the test set...")
    shap_values = explainer.shap_values(X_test_scaled_df)

    # 5. Generate and Save Global Interpretability Plot
    print("Generating and saving global summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_test_scaled_df, show=False)
    global_plot_path = os.path.join(viz_dir, 'global_summary_plot.png')
    plt.savefig(global_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Global plot saved to {global_plot_path}")

    # 6. Generate and Save Local Interpretability Plot (for the first test instance)
    print("Generating and saving local force plot for a single instance...")
    # Create a force plot for the first prediction in the test set
    force_plot = shap.force_plot(
        explainer.expected_value, 
        shap_values[0,:], 
        X_test_scaled_df.iloc[0,:],
        matplotlib=True,
        show=False
    )
    local_plot_path = os.path.join(viz_dir, 'local_force_plot_patient_0.png')
    plt.savefig(local_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Local plot saved to {local_plot_path}")
    print("--- Model Interpretation Finished ---")

if __name__ == '__main__':
    interpret_model()