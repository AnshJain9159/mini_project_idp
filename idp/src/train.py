# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import os

def train_model():
    """
    This function trains the XGBoost model on the diabetes dataset
    and saves the trained model to the 'models' directory.
    """
    print("--- Starting Model Training ---")

    # Define file paths
    data_path = os.path.join('data', 'diabetes.csv')
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'xgboost_diabetes_model.joblib')

    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # 2. Preprocess & Split Data
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Data split into training and testing sets.")

    # 3. Feature Scaling (Optional but good practice)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # We will save the scaler to use on new data later
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    print("Feature scaler created and saved.")

    # 4. Hyperparameter Tuning with GridSearchCV
    print("Starting hyperparameter tuning with GridSearchCV...")
    
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.01, 0.05],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0]
    }

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, 
                               cv=3, n_jobs=1, verbose=2, scoring='accuracy')
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    
    # Use the best estimator found by the grid search as the final model
    model = grid_search.best_estimator_
    print("Model training with best parameters complete.")

    # 5. Save Model
    joblib.dump(model, model_path)
    print(f"Model successfully saved to {model_path}")
    print("--- Model Training Finished ---")

if __name__ == '__main__':
    train_model()