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
import json
import sys
from sklearn.calibration import CalibratedClassifierCV
import uuid
from mem0 import MemoryClient

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from preloader import initialize_preloader
except ImportError:
    def initialize_preloader():
        pass

load_dotenv()

st.set_page_config(
    page_title="Interpretable Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_preloader()



# --- Function to Parse Medical Report PDF ---
def parse_medical_report_pdf(pdf_file):
    """
    Parses a medical report PDF and extracts relevant diabetes-related data using Groq LLM.
    Returns a dictionary with extracted values and the raw text.
    """
    try:
        # Read PDF content
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_content = ""
        
        for page in pdf_reader.pages:
            text_content += page.extract_text()
        
        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY not found.")
            return {}, text_content

        client = Groq(api_key=api_key)

        # Prompt for extraction
        prompt = f"""
        Extract the following medical values from the text below. Return ONLY a JSON object with keys:
        "Pregnancies" (int), "Glucose" (float, mg/dL), "BloodPressure" (int, mm Hg), 
        "SkinThickness" (float, mm), "Insulin" (float, mu U/ml), "BMI" (float), 
        "DiabetesPedigreeFunction" (float), "Age" (int).
        
        If a value is not found, use null.
        
        Text:
        {text_content[:4000]}
        """

        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a medical data extractor. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0,
            response_format={"type": "json_object"}
        )

        extracted_data = json.loads(completion.choices[0].message.content)
        
        # Clean up nulls
        cleaned_data = {k: v for k, v in extracted_data.items() if v is not None}
        
        return cleaned_data, text_content
        
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
            model="llama-3.1-8b-instant",
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
            model="llama-3.1-8b-instant",
            temperature=0.5,
            max_tokens=200,
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Could not generate explanation due to an error. Please ensure your GROQ_API_KEY is set correctly in the .env file. Error: {e}"

# --- Function to Generate Health Coach Plan ---
def generate_health_plan_groq(user_data, prediction_label, shap_values, feature_names):
    """
    Generates a personalized health plan using Groq.
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=api_key)

        # Identify top risk factors from SHAP
        feature_effects = sorted(zip(feature_names, shap_values, user_data.iloc[0]), key=lambda x: x[1], reverse=True)
        risk_factors = [f"{name} ({val})" for name, shap_val, val in feature_effects if shap_val > 0][:3]
        
        risk_str = ", ".join(risk_factors) if risk_factors else "General health maintenance"
        status = "High Risk" if prediction_label == 1 else "Low Risk"

        prompt = f"""
        Act as a compassionate and practical health coach.
        Patient Status: {status} of Diabetes.
        Key Risk Factors identified: {risk_str}.
        
        Provide a personalized, actionable 3-step plan (Diet, Exercise, Lifestyle).
        Keep it concise (bullet points), motivating, and specific to the risk factors.
        """

        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a health coach."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=300
        )
        return completion.choices[0].message.content

    except Exception as e:
        return f"Could not generate health plan: {e}"

# --- Function to Generate Simulation Plan ---
def generate_simulation_plan_groq(current_data, target_data, feature_names):
    """
    Generates a plan to achieve the target values in the simulator.
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=api_key)

        # Identify changes
        changes = []
        for feature in feature_names:
            curr_val = current_data[feature].iloc[0]
            target_val = target_data[feature].iloc[0]
            
            # Check for significant difference (e.g., > 1% change)
            if abs(curr_val - target_val) > 0.01:
                changes.append(f"{feature}: {curr_val} -> {target_val}")
        
        if not changes:
            return "No significant changes detected to generate a plan for."

        changes_str = "\n".join(changes)

        prompt = f"""
        Act as a medical health coach.
        The patient wants to achieve the following health targets to reduce diabetes risk:
        {changes_str}
        
        Provide a specific, actionable strategy to achieve these numbers.
        Focus ONLY on the factors that are changing.
        Give 3-4 concrete steps.
        """

        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a practical health coach."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=400
        )
        return completion.choices[0].message.content

    except Exception as e:
        return f"Could not generate simulation plan: {e}"

# --- Initialize Mem0 ---
@st.cache_resource
def initialize_mem0():
    """Initialize Mem0 client with API key from environment."""
    try:
        # Try to get key from MEMO_API_KEY (user preference) or MEM0_API_KEY (standard)
        mem0_api_key = os.getenv("MEMO_API_KEY") or os.getenv("MEM0_API_KEY")
        
        if not mem0_api_key:
            return None
        
        # Initialize MemoryClient for managed service
        memory = MemoryClient(api_key=mem0_api_key)
        return memory
    except Exception as e:
        st.warning(f"Mem0 initialization failed: {e}. Memory features will be disabled.")
        return None

# --- Chat Assistant Functions ---
def extract_patient_data_from_chat(user_message, user_id, memory_client):
    """
    Extract structured patient data from natural language using LLM.
    Returns dict with extracted values and missing fields.
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None, "GROQ_API_KEY not found"
        
        client = Groq(api_key=api_key)
        
        # Retrieve relevant memories if available
        # NOTE: Mem0 search disabled due to API issues
        # Memory is optional for extraction to work
        context = ""
        # if memory_client:
        #     try:
        #         # Use filters={"user_id": ...} for Mem0 v2
        #         memories = memory_client.search(user_message, filters={"user_id": user_id}, top_k=3)
        #         if memories and len(memories) > 0:
        #             # Handle different response formats
        #             if isinstance(memories, dict) and 'memories' in memories:
        #                 memories = memories['memories']
        #             context = "\n".join([f"- {m.get('memory', m.get('text', ''))}" for m in memories if m])
        #     except Exception as e:
        #         # Silently continue without memory context if there's an error
        #         # Memory is optional, so extraction should still work
        #         pass
        
        context_str = f"Previous context:\n{context}" if context else ""
        
        prompt = f"""Extract diabetes risk factors from this message. Return ONLY valid JSON.

Required fields (use null if not mentioned):
- Pregnancies (int) - If patient is male, ALWAYS set to 0
- Glucose (float, mg/dL)
- BloodPressure (int, mm Hg, systolic only)
- SkinThickness (float, mm)
- Insulin (float, mu U/ml)
- BMI (float)
- DiabetesPedigreeFunction (float, 0.0-2.5)
- Age (int) - Extract from patterns like "55yo", "55 years old", "age 55"

IMPORTANT:
- If you see "male" or "man", set Pregnancies to 0
- Parse age from casual formats (55yo = 55)
- Be flexible with units and formats

{context_str}

Message: {user_message}

Return JSON only."""

        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a medical data extractor. Be smart about parsing casual medical language. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(completion.choices[0].message.content)
        
        # Filter out nulls and identify missing fields
        extracted = {k: v for k, v in data.items() if v is not None}
        required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        missing = [f for f in required_fields if f not in extracted]
        
        return extracted, missing
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def generate_chat_response(user_message, extracted_data, missing_fields, prediction_result=None):
    """Generate conversational AI response."""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=api_key)
        
        if prediction_result:
            # Response after prediction
            prompt = f"""You are a helpful medical AI assistant. A diabetes risk prediction was just completed.

Prediction: {"High Risk" if prediction_result['prediction'] == 1 else "Low Risk"}
Probability: {prediction_result['probability']:.1f}%

Provide a brief, professional summary of this result in 2-3 sentences. Be reassuring and solution-focused."""
        
        elif missing_fields:
            # Still gathering data
            prompt = f"""You are a helpful medical AI assistant collecting patient data.

Extracted so far: {', '.join(extracted_data.keys()) if extracted_data else 'None yet'}
Still need: {', '.join(missing_fields)}

Ask the user for the missing information in a friendly, conversational way. Keep it brief (1-2 sentences)."""
        else:
            # All data collected
            return "Great! I have all the information needed. Click '🔬 Analyze Risk' below to run the prediction."
        
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful, concise medical assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=150
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"I encountered an error: {str(e)}"

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
tab1, tab2, tab3 = st.tabs(["📊 Manual Input Prediction", "📄 Medical Report Analysis", "💬 AI Chat Assistant"])

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
    # --- Main Page Layout ---
    if predict_button:
        # Create a DataFrame with correct feature order
        ordered_inputs = {}
        for feature in feature_names:
            ordered_inputs[feature] = user_inputs[feature]
        user_data = pd.DataFrame([ordered_inputs])
        
        # Scale data and make predictions
        try:
            user_data_scaled = scaler.transform(user_data)
            prediction_proba = model.predict_proba(user_data_scaled)
            prediction = model.predict(user_data_scaled)
            
            # Store in session state
            st.session_state['prediction_made'] = True
            st.session_state['user_data'] = user_data
            st.session_state['user_data_scaled'] = user_data_scaled
            st.session_state['prediction_proba'] = prediction_proba
            st.session_state['prediction'] = prediction
            
        except Exception as e:
            st.error(f"❌ **Error during prediction:** {str(e)}")
            st.stop()

    # Check if prediction has been made (either just now or in previous run)
    if st.session_state.get('prediction_made'):
        user_data = st.session_state['user_data']
        user_data_scaled = st.session_state['user_data_scaled']
        prediction_proba = st.session_state['prediction_proba']
        prediction = st.session_state['prediction']
        
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
                # Handle CalibratedClassifierCV for SHAP
                if isinstance(model, CalibratedClassifierCV):
                    if hasattr(model, 'estimator'):
                        explainer_model = model.estimator
                    else:
                        explainer_model = model.base_estimator
                else:
                    explainer_model = model
                
                explainer = shap.TreeExplainer(explainer_model)
                shap_values = explainer.shap_values(user_data_scaled)
                
                # Handle SHAP values format for plotting and explanation
                if isinstance(shap_values, list):
                    shap_values_for_plot = shap_values[1]
                else:
                    shap_values_for_plot = shap_values
                
                # Generate the text summary
                medical_summary = generate_medical_explanation_groq(shap_values_for_plot[0], feature_names, user_data)
                st.info(medical_summary)

                # Health Coach Section
                st.markdown("### 🧘 Personalized Health Coach")
                with st.spinner("Drafting your action plan..."):
                    health_plan = generate_health_plan_groq(user_data, prediction[0], shap_values_for_plot[0], feature_names)
                    st.success(health_plan)

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

        # --- What-If Simulator ---
        st.markdown("---")
        st.subheader("🧪 Clinical Simulation: Impact of Lifestyle Changes")
        st.markdown("Adjust the sliders below to simulate how changes in modifiable risk factors would affect the patient's risk.")

        with st.expander("🔬 Open Simulator", expanded=True):
            sim_col1, sim_col2 = st.columns(2)
            
            # Modifiable factors
            with sim_col1:
                st.markdown("**Modifiable Factors**")
                # Use session state values as defaults if available, else current user data
                sim_glucose = st.slider("Target Glucose (mg/dL)", 0, 200, int(user_data['Glucose'][0]), key="sim_glucose")
                sim_bmi = st.slider("Target BMI (kg/m²)", 0.0, 70.0, float(user_data['BMI'][0]), 0.1, key="sim_bmi")
                sim_bp = st.slider("Target Blood Pressure (mm Hg)", 0, 130, int(user_data['BloodPressure'][0]), key="sim_bp")
            
            with sim_col2:
                st.markdown("**Other Factors**")
                sim_insulin = st.slider("Target Insulin (mu U/ml)", 0, 900, int(user_data['Insulin'][0]), key="sim_insulin")
                sim_skin = st.slider("Target Skin Thickness (mm)", 0, 100, int(user_data['SkinThickness'][0]), key="sim_skin")

            # Create simulated data (copy original and update)
            sim_data = user_data.copy()
            sim_data['Glucose'] = sim_glucose
            sim_data['BMI'] = sim_bmi
            sim_data['BloodPressure'] = sim_bp
            sim_data['Insulin'] = sim_insulin
            sim_data['SkinThickness'] = sim_skin

            # Predict on simulated data
            sim_data_scaled = scaler.transform(sim_data)
            sim_prob = model.predict_proba(sim_data_scaled)[0][1] * 100
            
            # Calculate improvement
            risk_reduction = probability_of_diabetes - sim_prob
            
            # Display Results
            st.markdown("### 📉 Simulation Results")
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.metric("Original Risk", f"{probability_of_diabetes:.2f}%")
            with res_col2:
                st.metric("Simulated Risk", f"{sim_prob:.2f}%", delta=f"-{risk_reduction:.2f}%", delta_color="inverse")
            with res_col3:
                if sim_prob < 50 and probability_of_diabetes >= 50:
                    st.success("🎉 Risk dropped to Low!")
                elif risk_reduction > 0:
                    st.info("📉 Risk Reduced")
                else:
                    st.warning("⚠️ No Improvement")
            
            # Generate Plan Button
            if st.button("📝 How do I achieve this?", type="secondary", use_container_width=True):
                with st.spinner("Generating strategy to reach these targets..."):
                    sim_plan = generate_simulation_plan_groq(user_data, sim_data, feature_names)
                    st.markdown("### 🗺️ Strategy to Reach Targets")
                    st.success(sim_plan)

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
                        if isinstance(model, CalibratedClassifierCV):
                            if hasattr(model, 'estimator'):
                                explainer_model = model.estimator
                            else:
                                explainer_model = model.base_estimator
                        else:
                            explainer_model = model

                        explainer = shap.TreeExplainer(explainer_model)
                        shap_values = explainer.shap_values(user_data_scaled)
                        
                        if isinstance(shap_values, list):
                            shap_values_for_plot = shap_values[1]
                        else:
                            shap_values_for_plot = shap_values
                        
                        # Generate medical summary
                        medical_summary = generate_medical_explanation_groq(shap_values_for_plot[0], feature_names, user_data)
                        st.info(medical_summary)

                        # Health Coach Section
                        st.markdown("### 🧘 Personalized Health Coach")
                        with st.spinner("Drafting your action plan..."):
                            health_plan = generate_health_plan_groq(user_data, prediction[0], shap_values_for_plot[0], feature_names)
                            st.success(health_plan)
                        
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

# --- Tab 3: AI Chat Assistant ---
with tab3:
    st.header("💬 AI Clinical Chat Assistant")
    st.markdown("Describe your patient in natural language. The AI will extract the necessary data and provide risk analysis.")
    
    # Initialize Mem0
    memory_client = initialize_mem0()
    
    if memory_client is None:
        st.info("💡 **Memory features disabled**: Add `MEM0_API_KEY` to your `.env` file to enable conversation memory across sessions.")
    
    # Initialize session state for chat
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'chat_patient_data' not in st.session_state:
        st.session_state.chat_patient_data = {}
    if 'chat_patient_id' not in st.session_state:
        # Generate unique patient ID for each new session
        st.session_state.chat_patient_id = str(uuid.uuid4())
    
    
    # Patient ID and controls
    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input("🆔 Patient ID (unique per session)", value=st.session_state.chat_patient_id, disabled=True, key="patient_id_display")
    with col2:
        if st.button("� New Patient"):
            st.session_state.chat_messages = []
            st.session_state.chat_patient_data = {}
            st.session_state.chat_patient_id = str(uuid.uuid4())
            st.rerun()
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Type your message... (e.g., '55yo male, glucose 180, BMI 32')")
    
    if user_input:
        # Add to session state
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        
        # Extract data and generate response
        patient_id = st.session_state.chat_patient_id
        extracted, missing = extract_patient_data_from_chat(user_input, patient_id, memory_client)
        
        if extracted:
            # Update patient data (merge with existing)
            st.session_state.chat_patient_data.update(extracted)
            
            # Generate response
            response = generate_chat_response(
                user_input,
                st.session_state.chat_patient_data,
                missing
            )
            
            # Save to memory if available
            if memory_client:
                try:
                    memory_client.add(
                        f"Patient data: {extracted}",
                        user_id=patient_id
                    )
                except:
                    pass
            
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
        else:
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": f"I had trouble extracting the data: {missing}"
            })
    
    # Show current data status
    if st.session_state.chat_patient_data:
        st.markdown("---")
        st.subheader("📋 Extracted Patient Data")
        
        col1, col2 = st.columns(2)
        with col1:
            for key, value in list(st.session_state.chat_patient_data.items())[:4]:
                st.metric(key, value)
        with col2:
            for key, value in list(st.session_state.chat_patient_data.items())[4:]:
                st.metric(key, value)
        
        # Check if we have all required fields
        required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        missing = [f for f in required_fields if f not in st.session_state.chat_patient_data]
        
        if not missing:
            # All data collected, allow prediction
            if st.button("🔬 Analyze Risk", type="primary", use_container_width=True):
                # Create DataFrame with correct feature order
                feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                
                # Ensure data is in correct order
                ordered_data = {feature: st.session_state.chat_patient_data[feature] for feature in feature_names}
                user_data = pd.DataFrame([ordered_data])
                
                # Run prediction
                try:
                    user_data_scaled = scaler.transform(user_data)
                    prediction_proba = model.predict_proba(user_data_scaled)
                    prediction = model.predict(user_data_scaled)
                    probability = prediction_proba[0][1] * 100
                    
                    # Generate response
                    prediction_result = {
                        'prediction': prediction[0],
                        'probability': probability
                    }
                    
                    response = generate_chat_response(
                        None,
                        st.session_state.chat_patient_data,
                        [],
                        prediction_result
                    )
                    
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # Save to memory
                    if memory_client:
                        try:
                            memory_client.add(
                                f"Prediction result: {prediction_result}",
                                user_id=patient_id
                            )
                        except:
                            pass
                    
                    # Display result
                    if prediction[0] == 1:
                        st.error(f"⚠️ **High Risk of Diabetes** ({probability:.1f}%)")
                    else:
                        st.success(f"✅ **Low Risk of Diabetes** ({probability:.1f}%)")
                    
                    # SHAP explanation
                    st.subheader("📊 Risk Factors Breakdown")
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(user_data_scaled)
                    
                    if isinstance(shap_values, list):
                        shap_values_for_plot = shap_values[1]
                    else:
                        shap_values_for_plot = shap_values
                    
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
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
        else:
            st.info(f"ℹ️ Missing: {', '.join(missing)}")
