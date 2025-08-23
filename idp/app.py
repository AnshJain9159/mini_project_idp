# app.py

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --- Load Model and Scaler ---
try:
    model = joblib.load('models/xgboost_diabetes_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    st.error("Error: Model or scaler file not found. Please run the training script first.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Interpretable Diabetes Prediction", layout="wide")
st.title('🩺 Interpretable Diabetes Prediction')
st.write("""
This app predicts the likelihood of a patient having diabetes based on their medical attributes. 
It also provides an explanation for the prediction using SHAP (SHapley Additive exPlanations).
""")

# --- Feature Columns ---
# The exact column names your model was trained on
feature_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# --- User Input Sidebar ---
st.sidebar.header('Patient Medical Data')
st.sidebar.write('Please provide the patient information below:')

# Create input fields for all 8 features in the sidebar
pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=1, step=1)
glucose = st.sidebar.slider('Glucose (mg/dL)', 0, 200, 120)
blood_pressure = st.sidebar.slider('Blood Pressure (mm Hg)', 0, 130, 72)
skin_thickness = st.sidebar.slider('Skin Thickness (mm)', 0, 100, 20)
insulin = st.sidebar.slider('Insulin (mu U/ml)', 0, 900, 79)
bmi = st.sidebar.slider('BMI (kg/m²)', 0.0, 70.0, 32.0, 0.1)
dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.47, 0.01)
age = st.sidebar.slider('Age (years)', 1, 120, 29)

# --- Prediction and Explanation Logic ---
if st.sidebar.button('**Predict**', use_container_width=True):
    
    # 1. Create a DataFrame from user input in the correct order
    user_data = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
        columns=feature_names
    )
    st.subheader("Patient Input Data")
    st.dataframe(user_data)

    # 2. Scale the data and make a prediction
    user_data_scaled = scaler.transform(user_data)
    prediction_proba = model.predict_proba(user_data_scaled)
    prediction = model.predict(user_data_scaled)

    # 3. Display Prediction
    st.subheader("Prediction Result")
    probability_of_diabetes = prediction_proba[0][1] * 100
    
    if prediction[0] == 1:
        st.error(f'**Diabetic** (Probability: {probability_of_diabetes:.2f}%)')
    else:
        st.success(f'**Not Diabetic** (Probability of being diabetic: {probability_of_diabetes:.2f}%)')

    # 4. Display SHAP Explanation
    st.subheader("Prediction Explanation")
    st.write("The plot below shows which features pushed the prediction higher (red) or lower (blue).")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(user_data_scaled)
    
    if isinstance(shap_values, list):
    # Case 1: Output is a list of two arrays (one for each class). We explain class 1.
        expected_value = explainer.expected_value[1] # type: ignore
        shap_values_for_plot = shap_values[1]
    else:
        # Case 2: Output is a single array. This is the explanation we need.
        expected_value = explainer.expected_value
        shap_values_for_plot = shap_values

    # Generate the plot by capturing the figure object it returns
    force_plot_fig = shap.force_plot(
        expected_value,
        shap_values_for_plot[0], # SHAP values for the single instance we are predicting
        user_data.iloc[0],       # Original feature values for labels
        matplotlib=True,
        show=False,
        text_rotation=10
    )

    # Use Streamlit to display the captured figure
    if force_plot_fig is not None:
        st.pyplot(force_plot_fig, bbox_inches='tight', use_container_width=True)
        plt.close(force_plot_fig) # Important to close the figure to free up memory

else:
    st.info("Please input patient data on the left and click 'Predict'.")