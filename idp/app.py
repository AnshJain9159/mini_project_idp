# app.py

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import re
from groq import Groq
from dotenv import load_dotenv
import PyPDF2
import io

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

# --- Function to Parse Medical Report PDF ---
def parse_medical_report_pdf(pdf_file):
    """
    Parses a medical report PDF and extracts relevant diabetes-related data.
    Returns a dictionary with extracted values and the raw text for LLM analysis.
    """
    try:
        # Read PDF content
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_content = ""
        
        for page in pdf_reader.pages:
            text_content += page.extract_text()
        
        # Extract common diabetes-related values using regex patterns
        extracted_data = {}
        
        # Glucose patterns (mg/dL, mmol/L)
        glucose_patterns = [
            r'glucose[:\s]*(\d+(?:\.\d+)?)\s*(?:mg/dL|mg/dl|mg/dl|mmol/L|mmol/l)',
            r'glucose[:\s]*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:mg/dL|mg/dl|mg/dl|mmol/L|mmol/l).*glucose',
            r'glucose.*?(\d+(?:\.\d+)?)'
        ]
        
        for pattern in glucose_patterns:
            match = re.search(pattern, text_content.lower())
            if match:
                extracted_data['Glucose'] = float(match.group(1))
                break
        
        # Blood Pressure patterns
        bp_patterns = [
            r'blood\s*pressure[:\s]*(\d+)/(\d+)',
            r'bp[:\s]*(\d+)/(\d+)',
            r'(\d+)/(\d+)\s*(?:mm\s*Hg|mmHg)',
            r'(\d+)/(\d+).*blood\s*pressure'
        ]
        
        for pattern in bp_patterns:
            match = re.search(pattern, text_content.lower())
            if match:
                extracted_data['BloodPressure'] = int(match.group(1))
                break
        
        # BMI patterns
        bmi_patterns = [
            r'bmi[:\s]*(\d+(?:\.\d+)?)',
            r'body\s*mass\s*index[:\s]*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?).*bmi',
            r'(\d+(?:\.\d+)?).*body\s*mass\s*index'
        ]
        
        for pattern in bmi_patterns:
            match = re.search(pattern, text_content.lower())
            if match:
                extracted_data['BMI'] = float(match.group(1))
                break
        
        # Age patterns
        age_patterns = [
            r'age[:\s]*(\d+)',
            r'(\d+)\s*years?\s*old',
            r'(\d+)\s*yo',
            r'(\d+).*age'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text_content.lower())
            if match:
                extracted_data['Age'] = int(match.group(1))
                break
        
        # Insulin patterns
        insulin_patterns = [
            r'insulin[:\s]*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?).*insulin',
            r'insulin.*?(\d+(?:\.\d+)?)'
        ]
        
        for pattern in insulin_patterns:
            match = re.search(pattern, text_content.lower())
            if match:
                extracted_data['Insulin'] = float(match.group(1))
                break
        
        # Skin Thickness patterns
        skin_patterns = [
            r'skin\s*thickness[:\s]*(\d+(?:\.\d+)?)',
            r'triceps[:\s]*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?).*skin\s*thickness',
            r'(\d+(?:\.\d+)?).*triceps'
        ]
        
        for pattern in skin_patterns:
            match = re.search(pattern, text_content.lower())
            if match:
                extracted_data['SkinThickness'] = float(match.group(1))
                break
        
        # Pregnancies patterns
        preg_patterns = [
            r'pregnanc[yi][:\s]*(\d+)',
            r'(\d+).*pregnanc[yi]',
            r'gravidity[:\s]*(\d+)',
            r'(\d+).*gravidity'
        ]
        
        for pattern in preg_patterns:
            match = re.search(pattern, text_content.lower())
            if match:
                extracted_data['Pregnancies'] = int(match.group(1))
                break
        
        # Diabetes Pedigree Function patterns
        dpf_patterns = [
            r'diabetes\s*pedigree[:\s]*(\d+(?:\.\d+)?)',
            r'pedigree[:\s]*(\d+(?:\.\d+)?)',
            r'dpf[:\s]*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?).*diabetes\s*pedigree',
            r'(\d+(?:\.\d+)?).*pedigree'
        ]
        
        for pattern in dpf_patterns:
            match = re.search(pattern, text_content.lower())
            if match:
                extracted_data['DiabetesPedigreeFunction'] = float(match.group(1))
                break
        
        return extracted_data, text_content
        
    except Exception as e:
        st.error(f"Error parsing PDF: {str(e)}")
        return {}, ""

# --- Function to Generate Medical Report Analysis using Groq ---
def analyze_medical_report_groq(extracted_data, raw_text, feature_names):
    """
    Uses Groq LLM to analyze the medical report and provide insights.
    """
    try:
        # Initialize Groq client from environment variables
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set. Please check your .env file.")
        client = Groq(api_key=api_key)

        # Create a summary of extracted data
        extracted_summary = []
        for feature in feature_names:
            if feature in extracted_data:
                extracted_summary.append(f"- **{feature}**: {extracted_data[feature]}")
            else:
                extracted_summary.append(f"- **{feature}**: Not found in report")
        
        extracted_str = "\n".join(extracted_summary)

        # Create a professional medical analysis prompt
        prompt = f"""
        Act as a medical professional analyzing a diabetes test report. 
        Your task is to provide a comprehensive analysis in 2-3 paragraphs.

        EXTRACTED DATA FROM REPORT:
        {extracted_str}

        RAW REPORT TEXT (for context):
        {raw_text[:1000]}...

        Please provide:
        1. A summary of the key findings from the report
        2. Analysis of the extracted values in medical context
        3. Any notable patterns or concerns
        4. Recommendations for follow-up if applicable

        Keep the tone professional and medical. Do not give direct medical advice.
        Focus on interpreting the data that was successfully extracted.
        """

        # Make the API call
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a medical professional analyzing diabetes test reports."},
                {"role": "user", "content": prompt},
            ],
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=400,
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Could not analyze report due to an error: {e}"

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

# --- Create Tabs for Different Functionalities ---
tab1, tab2 = st.tabs(["📊 Manual Input Prediction", "📄 Medical Report Analysis"])

# --- Tab 1: Manual Input Prediction ---
with tab1:
    # --- Sidebar for User Input and Theme ---
    with st.sidebar:
        st.header('👨‍⚕️ Patient Medical Data')
        st.markdown('Please provide the patient information below:')

        # Create input fields for all 8 features (order must match training data exactly)
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
                # Create a DataFrame with correct feature order
        ordered_inputs = {}
        for feature in feature_names:
            ordered_inputs[feature] = user_inputs[feature]
        user_data = pd.DataFrame([ordered_inputs])
        
        # Debug: Show the ordered data
        # st.write("🔍 **Debug Info:** Feature order in DataFrame:")
        # st.write(list(user_data.columns))
        
        # Scale data and make predictions
        try:
            user_data_scaled = scaler.transform(user_data)
            prediction_proba = model.predict_proba(user_data_scaled)
            prediction = model.predict(user_data_scaled)
        except Exception as e:
            st.error(f"❌ **Error during prediction:** {str(e)}")
            st.write("🔍 **Debug Info:** DataFrame shape:", user_data.shape)
            st.write("🔍 **Debug Info:** DataFrame columns:", list(user_data.columns))
            st.write("🔍 **Debug Info:** Expected features:", feature_names)
            st.stop()
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

# --- Tab 2: Medical Report Analysis ---
with tab2:
    st.header("📄 Medical Report Analysis")
    st.markdown("Upload a diabetes test report (PDF) to automatically extract data and get AI-powered analysis.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a medical report PDF file",
        type=['pdf'],
        help="Upload a PDF file containing diabetes test results"
    )
    
    if uploaded_file is not None:
        # Parse the PDF
        with st.spinner("Parsing medical report..."):
            extracted_data, raw_text = parse_medical_report_pdf(uploaded_file)
        
        # Display extracted data
        st.subheader("📋 Extracted Data from Report")
        
        if extracted_data:
            # Create a DataFrame for display (order must match training data exactly)
            feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            extracted_df = pd.DataFrame([extracted_data])
            
            # Fill missing values with "Not found"
            for feature in feature_names:
                if feature not in extracted_data:
                    extracted_data[feature] = "Not found"
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(extracted_df, use_container_width=True)
                
                # Show missing features
                missing_features = [f for f in feature_names if f not in extracted_data or extracted_data[f] == "Not found"]
                if missing_features:
                    st.warning(f"⚠️ **Missing data for:** {', '.join(missing_features)}")
                    st.info("You can manually input missing values below to get a complete prediction.")
                
                # Manual input for missing features
                if missing_features:
                    st.subheader("🔧 Complete Missing Data")
                    for feature in missing_features:
                        if feature == 'Pregnancies':
                            extracted_data[feature] = st.number_input(f'{feature}', 0, 20, 1, 1, key=f"pdf_{feature}")
                        elif feature == 'Glucose':
                            extracted_data[feature] = st.slider(f'{feature} (mg/dL)', 0, 200, 120, key=f"pdf_{feature}")
                        elif feature == 'BloodPressure':
                            extracted_data[feature] = st.slider(f'{feature} (mm Hg)', 0, 130, 72, key=f"pdf_{feature}")
                        elif feature == 'SkinThickness':
                            extracted_data[feature] = st.slider(f'{feature} (mm)', 0, 100, 20, key=f"pdf_{feature}")
                        elif feature == 'Insulin':
                            extracted_data[feature] = st.slider(f'{feature} (mu U/ml)', 0, 900, 79, key=f"pdf_{feature}")
                        elif feature == 'BMI':
                            extracted_data[feature] = st.slider(f'{feature} (kg/m²)', 0.0, 70.0, 32.0, 0.1, key=f"pdf_{feature}")
                        elif feature == 'DiabetesPedigreeFunction':
                            extracted_data[feature] = st.slider(f'{feature}', 0.0, 2.5, 0.47, 0.01, key=f"pdf_{feature}")
                        elif feature == 'Age':
                            extracted_data[feature] = st.slider(f'{feature} (years)', 1, 120, 29, key=f"pdf_{feature}")
            
            with col2:
                # AI Analysis of the report
                st.subheader("🤖 AI Report Analysis")
                with st.spinner("Analyzing report with AI..."):
                    analysis = analyze_medical_report_groq(extracted_data, raw_text, feature_names)
                    st.info(analysis)
                
                # Prediction button for extracted data
                if st.button("🚀 Get Prediction from Report Data", type="primary", use_container_width=True):
                    # Check if we have all required data
                    complete_data = all(isinstance(extracted_data.get(f), (int, float)) for f in feature_names)
                    
                    if complete_data:
                        # Create DataFrame with correct feature order
                        ordered_data = {}
                        for feature in feature_names:
                            ordered_data[feature] = extracted_data[feature]
                        user_data = pd.DataFrame([ordered_data])
                        
                        # Debug: Show the ordered data
                        # st.write("🔍 **Debug Info:** Feature order in DataFrame:")
                        # st.write(list(user_data.columns))
                        
                        # Scale data and make predictions
                        try:
                            user_data_scaled = scaler.transform(user_data)
                            prediction_proba = model.predict_proba(user_data_scaled)
                            prediction = model.predict(user_data_scaled)
                        except Exception as e:
                            st.error(f"❌ **Error during prediction:** {str(e)}")
                            st.write("🔍 **Debug Info:** DataFrame shape:", user_data.shape)
                            st.write("🔍 **Debug Info:** DataFrame columns:", list(user_data.columns))
                            st.write("🔍 **Debug Info:** Expected features:", feature_names)
                            st.stop()
                        probability_of_diabetes = prediction_proba[0][1] * 100
                        
                        st.success("✅ **Prediction Complete!**")
                        
                        if prediction[0] == 1:
                            st.error(f'**High Risk of Diabetes** (Probability: {probability_of_diabetes:.2f}%)')
                        else:
                            st.success(f'**Low Risk of Diabetes** (Probability: {probability_of_diabetes:.2f}%)')
                        
                        # Generate SHAP explanation
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(user_data_scaled)
                        
                        if isinstance(shap_values, list):
                            shap_values_for_plot = shap_values[1]
                        else:
                            shap_values_for_plot = shap_values
                        
                        # Generate medical summary
                        medical_summary = generate_medical_explanation_groq(shap_values_for_plot[0], feature_names, user_data)
                        st.info(medical_summary)
                        
                        # SHAP plot
                        st.subheader("🔬 SHAP Analysis")
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
                        st.error("❌ **Cannot make prediction:** Some required data is missing or invalid.")
                        st.info("Please complete all missing values above to get a prediction.")
        
        else:
            st.error("❌ **No data could be extracted from the PDF.**")
            st.info("The PDF might not contain recognizable diabetes test data, or the format is not supported.")
        
        # Show raw text for debugging (collapsible)
        with st.expander("🔍 View Raw PDF Text (for debugging)"):
            st.text_area("Raw PDF Content", raw_text[:2000] + "..." if len(raw_text) > 2000 else raw_text, height=300)
    
    else:
        st.info("📤 **Upload a PDF file above to analyze your medical report.**")
        st.markdown("""
        **Supported formats:** PDF files containing diabetes test results
        
        **What we extract:**
        - Glucose levels
        - Blood pressure
        - BMI
        - Age
        - Insulin levels
        - Skin thickness
        - Pregnancy count
        - Diabetes pedigree function
        
        **Note:** The AI will analyze the extracted data and provide medical insights.
        """)
