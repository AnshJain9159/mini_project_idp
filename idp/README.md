# 🩺 Interpretable AI: Diabetes Predictor

This project is a machine learning application that predicts the likelihood of a patient having diabetes based on their clinical data. Its core feature is interpretability: it doesn't just provide a prediction, it explains why the prediction was made using SHAP (SHapley Additive exPlanations).

The project is built with a standard Python data science stack and includes a simple, interactive web application created with Streamlit.

## 🎯 Problem Statement

The "black box" nature of many advanced machine learning models is a significant barrier to their adoption in critical fields like healthcare. Clinicians need to trust a model's output and understand the reasoning behind it. This project addresses that challenge by building a highly accurate predictor for diabetes that can explain its predictions on both a global (all patients) and local (individual patient) level.

## ✨ Features

- **End-to-End ML Pipeline**: Scripts to train, evaluate, and interpret the model from start to finish.
- **High-Performance Model**: Uses XGBoost, a state-of-the-art algorithm for tabular data.
- **Model Interpretability**: Integrates the SHAP library to generate clear, visual explanations for every prediction.
- **Interactive Web App**: A user-friendly interface built with Streamlit where users can input patient data and instantly receive an explained prediction.
- **Reproducible Environment**: A requirements.txt file ensures a consistent and easy setup process.

## 🛠️ Technology Stack

- **Python**: Core programming language.
- **Pandas**: For data manipulation and loading.
- **Scikit-learn**: For data preprocessing, splitting, and model evaluation.
- **XGBoost**: The machine learning algorithm used for classification.
- **SHAP**: For explaining the model's predictions.
- **Streamlit**: For building the interactive web application.
- **Matplotlib & Seaborn**: For creating visualizations like the confusion matrix.

## 📁 Project Structure

The project is organized to separate concerns, making it clean and scalable.

```
interpretable_diabetes_predictor/
│
├── data/
│   └── diabetes.csv
│
├── models/
│   ├── xgboost_diabetes_model.joblib
│   └── scaler.joblib
│
├── src/
│   ├── train.py         # Trains and saves the model
│   ├── evaluate.py      # Evaluates model performance
│   ├── interpret.py     # Generates SHAP plots from the command line
│   └── main.py          # Runs the full training pipeline
│
├── visualizations/
│   ├── confusion_matrix.png
│   └── global_summary_plot.png
│
├── .venv/               # Virtual environment directory
├── app.py               # The Streamlit web application
├── requirements.txt     # Project dependencies
└── README.md            # This file
```

## 🚀 Installation & Setup

Follow these steps to set up the project environment.

### 1. Clone the Repository (or create your project folder)

### 2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to keep dependencies isolated.

```bash
# Create the virtual environment
python -m venv .venv

# Activate it (Windows)
.\.venv\Scripts\activate

# Activate it (macOS/Linux)
source .venv/bin/activate
```

### 3. Install Dependencies

Install all the required libraries from the requirements.txt file.

```bash
pip install -r requirements.txt
```

### 4. Place the Dataset

Ensure that the diabetes.csv dataset is placed inside the `data/` directory.

### 5. Set Up Environment Variables

The app uses environment variables for API keys. You have two options to set up your `.env` file:

#### Option A: Use the Setup Script (Recommended)
```bash
python setup_env.py
```
This interactive script will securely prompt you for your Groq API key and create the `.env` file.

#### Option B: Manual Setup
Create a `.env` file in the project root with your Groq API key:

```bash
# Create .env file
echo "GROQ_API_KEY=your_actual_groq_api_key_here" > .env
```

**Important**: Replace `your_actual_groq_api_key_here` with your real Groq API key. You can get one from [Groq's website](https://console.groq.com/).

**Note**: The `.env` file should never be committed to version control. It's already included in `.gitignore`.

## ⚙️ Usage

There are two main ways to use this project: running the full training pipeline or launching the interactive web app.

### 1. Running the Full Training Pipeline

To train the model, evaluate its performance, and generate the interpretation plots from scratch, run the main.py script from the root directory.

```bash
python src/main.py
```

This command will:
- Train a new XGBoost model and save it in the `models/` folder.
- Evaluate the model and print a classification report to the console.
- Save a confusion matrix and a global SHAP summary plot in the `visualizations/` folder.

### 2. Launching the Interactive Web App

To use the pre-trained model in an interactive interface, run the Streamlit app.

```bash
streamlit run app.py
```

This will open a new tab in your web browser. You can use the sliders and input fields on the sidebar to enter a patient's data and click "Predict" to see the result and its SHAP explanation.

## 📊 Results & Interpretation

After running the pipeline, you will get several outputs that help you understand the model.

### Model Performance

The evaluation script will print a Classification Report and save a Confusion Matrix. This tells you how accurate the model is at predicting both diabetic and non-diabetic cases.

### Model Interpretation

The key feature of this project is explaining the model's predictions.

#### Global Interpretation

The `global_summary_plot.png` shows which features are most important across the entire dataset. Features at the top have the biggest impact on the model's predictions.

#### Local Interpretation (in the Web App)

The SHAP force plot in the Streamlit app explains a single prediction.
- **Red features** (like BMI) push the prediction towards "Diabetic".
- **Blue features** (like Glucose) push the prediction towards "Not Diabetic".
- The final prediction is the result of these competing forces.