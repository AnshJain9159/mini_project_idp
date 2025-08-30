"""
Test script to verify feature extraction from medical report text
"""

import re

def parse_medical_report_text(text_content):
    """
    Parses medical report text and extracts relevant diabetes-related data.
    Returns a dictionary with extracted values.
    """
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
    
    return extracted_data

def test_feature_extraction():
    """Test the feature extraction with sample data"""
    
    # Sample medical report text
    sample_text = """DIABETES SCREENING REPORT
Patient ID: 12345
Date: 2024-01-15

PATIENT INFORMATION:
Name: John Doe
Age: 45 years old
Gender: Male

LABORATORY RESULTS:
Glucose: 135 mg/dL
Blood Pressure: 140/90 mmHg
BMI: 28.5 kg/m²
Insulin: 85 mu U/ml
Skin Thickness: 25 mm
Triceps: 25 mm

MEDICAL HISTORY:
Pregnancies: 0
Diabetes Pedigree Function: 0.85

CLINICAL NOTES:
Patient presents with elevated glucose levels and hypertension.
Recommend follow-up testing in 3 months.
Consider lifestyle modifications for weight management.

REFERENCE RANGES:
Normal Glucose: 70-100 mg/dL
Normal Blood Pressure: <120/80 mmHg
Normal BMI: 18.5-24.9 kg/m²"""
    
    print("Testing feature extraction...")
    print("=" * 50)
    
    # Extract features
    extracted_data = parse_medical_report_text(sample_text)
    
    # Expected feature order
    expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    print("Extracted Data:")
    for feature in expected_features:
        if feature in extracted_data:
            print(f"✅ {feature}: {extracted_data[feature]}")
        else:
            print(f"❌ {feature}: Not found")
    
    print("\n" + "=" * 50)
    
    # Test DataFrame creation
    import pandas as pd
    
    # Create ordered DataFrame
    ordered_data = {}
    for feature in expected_features:
        if feature in extracted_data:
            ordered_data[feature] = extracted_data[feature]
        else:
            # Use default values for missing features
            if feature == 'Pregnancies':
                ordered_data[feature] = 0
            elif feature == 'Glucose':
                ordered_data[feature] = 120.0
            elif feature == 'BloodPressure':
                ordered_data[feature] = 72
            elif feature == 'SkinThickness':
                ordered_data[feature] = 20.0
            elif feature == 'Insulin':
                ordered_data[feature] = 79.0
            elif feature == 'BMI':
                ordered_data[feature] = 32.0
            elif feature == 'DiabetesPedigreeFunction':
                ordered_data[feature] = 0.47
            elif feature == 'Age':
                ordered_data[feature] = 29
    
    df = pd.DataFrame([ordered_data])
    
    print("DataFrame created successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types: {df.dtypes.tolist()}")
    
    return extracted_data, df

if __name__ == "__main__":
    extracted_data, df = test_feature_extraction()
