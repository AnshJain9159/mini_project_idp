# app.py

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Page Configuration ---
# Must be the first Streamlit command
st.set_page_config(
    page_title="Interpretable Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Function to Generate Medical Explanation using Groq ---
def generate_medical_explanation_groq(shap_values, feature_names, user_data):
    """
    Generates a human-readable medical explanation of the prediction using Groq.
    """
    try:
        # Initialize Groq client from environment variables
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set. Please check your .env file.")
        client = Groq(api_key=api_key)

        # Combine feature names, their SHAP values, and user inputs
        feature_effects = sorted(zip(feature_names, shap_values, user_data.iloc[0]), key=lambda x: abs(x[1]), reverse=True)

        # Identify top 3 contributing factors
        top_factors_summary = []
        for name, shap_val, user_val in feature_effects[:3]:
            direction = "increasing" if shap_val > 0 else "decreasing"
            top_factors_summary.append(f"- **{name}** (patient's value: {user_val}): Significantly **{direction}** the risk.")

        factors_str = "\n".join(top_factors_summary)

        # Create a professional medical prompt for the LLM
        prompt = f"""
        Act as a medical professional interpreting a diabetes risk prediction from a machine learning model for a patient.
        Your task is to provide a concise, professional, and easy-to-understand summary in one paragraph.

        The model's output is based on these key contributing factors:
        {factors_str}

        Based on these factors, generate a brief, one-paragraph summary.
        - Start with "Based on the provided medical data...".
        - Do not use sensational language or give direct medical advice.
        - Explain how the top factors contributed to the overall risk assessment.
        - Maintain a reassuring and professional tone.
        """

        # Make the API call to a fast, free model like Llama 3
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant providing clear explanations."},
                {"role": "user", "content": prompt},
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=200,
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Could not generate explanation due to an error. Please ensure your GROQ_API_KEY is set correctly in the .env file. Error: {e}"


# --- Load Model and Scaler ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('models/xgboost_diabetes_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model or scaler file not found. Please run the training script (`src/main.py`) first.")
        st.stop()

model, scaler = load_resources()

# --- UI Modernization & Theme Toggle ---
st.title('🩺 Interpretable Diabetes Prediction')
st.markdown("This application predicts diabetes risk and provides clear, AI-driven explanations for the results.")

# --- Sidebar for User Input and Theme ---
with st.sidebar:
    st.header('👨‍⚕️ Patient Medical Data')
    st.markdown('Please provide the patient information below:')

    # Create input fields for all 8 features
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    user_inputs = {}
    user_inputs['Pregnancies'] = st.number_input('Pregnancies', 0, 20, 1, 1)
    user_inputs['Glucose'] = st.slider('Glucose (mg/dL)', 0, 200, 120)
    user_inputs['BloodPressure'] = st.slider('Blood Pressure (mm Hg)', 0, 130, 72)
    user_inputs['SkinThickness'] = st.slider('Skin Thickness (mm)', 0, 100, 20)
    user_inputs['Insulin'] = st.slider('Insulin (mu U/ml)', 0, 900, 79)
    user_inputs['BMI'] = st.slider('BMI (kg/m²)', 0.0, 70.0, 32.0, 0.1)
    user_inputs['DiabetesPedigreeFunction'] = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.47, 0.01)
    user_inputs['Age'] = st.slider('Age (years)', 1, 120, 29)
    
    predict_button = st.button('**Get Prediction**', use_container_width=True, type="primary")

# --- Main Page Layout ---
if predict_button:
    # Create a DataFrame from user input
    user_data = pd.DataFrame([user_inputs])

    # Scale data and make predictions
    user_data_scaled = scaler.transform(user_data)
    prediction_proba = model.predict_proba(user_data_scaled)
    prediction = model.predict(user_data_scaled)
    probability_of_diabetes = prediction_proba[0][1] * 100

    # Use columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Prediction Result")
        if prediction[0] == 1:
            st.error(f'**High Risk of Diabetes** (Probability: {probability_of_diabetes:.2f}%)')
        else:
            st.success(f'**Low Risk of Diabetes** (Probability: {probability_of_diabetes:.2f}%)')
        
        st.subheader("📋 Patient Input Data")
        st.dataframe(user_data)

    with col2:
        st.subheader("🤖 AI-Generated Medical Summary")
        with st.spinner('Generating professional summary with Groq...'):
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(user_data_scaled)
            
            # Handle SHAP values format for plotting and explanation
            if isinstance(shap_values, list):
                shap_values_for_plot = shap_values[1]
            else:
                shap_values_for_plot = shap_values
            
            # Generate the text summary
            medical_summary = generate_medical_explanation_groq(shap_values_for_plot[0], feature_names, user_data)
            st.info(medical_summary)

    st.markdown("---")
    st.subheader("🔬 Technical Prediction Breakdown (SHAP Plot)")
    st.write("The plot below shows the impact of each feature on the final prediction. Red features increase the risk, while blue features decrease it.")
    
    # Generate and display SHAP force plot
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]
        
    force_plot_fig = shap.force_plot(
        expected_value, shap_values_for_plot[0], user_data.iloc[0],
        matplotlib=True, show=False, text_rotation=10
    )
    
    if force_plot_fig is not None:
        st.pyplot(force_plot_fig, bbox_inches='tight', use_container_width=True)
        plt.close(force_plot_fig)

else:
    st.info("Please input patient data in the sidebar on the left and click 'Get Prediction' to view the results.")
