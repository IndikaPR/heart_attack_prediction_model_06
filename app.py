import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

# Page configuration
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high { background-color: #ff4b4b; color: white; padding: 20px; border-radius: 10px; font-size: 1.5rem; text-align: center; }
    .risk-medium { background-color: #ffa500; color: white; padding: 20px; border-radius: 10px; font-size: 1.5rem; text-align: center; }
    .risk-low { background-color: #00cc66; color: white; padding: 20px; border-radius: 10px; font-size: 1.5rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    """Load model and preprocessing artifacts"""
    try:
        model = keras.models.load_model('heart_attack_model.h5')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

def create_feature_vector(input_data, feature_columns):
    """Create feature vector matching training data structure"""
    try:
        # Create a dataframe with all feature columns initialized to 0
        feature_df = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # Map input data to feature columns
        for key, value in input_data.items():
            if key in feature_df.columns:
                feature_df[key] = value
        
        return feature_df
    except Exception as e:
        st.error(f"Error creating feature vector: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Attack Risk Predictor</h1>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_columns = load_artifacts()
    
    if model is None:
        st.error("Please make sure all model files are in the same directory.")
        return
    
    # Show feature info in sidebar
    with st.sidebar:
        st.markdown("### 🔧 Model Info")
        st.write(f"Features: {len(feature_columns)}")
        
        # Show some feature examples
        st.markdown("### 📋 Feature Examples")
        for i, feat in enumerate(feature_columns[:10]):
            st.write(f"{i+1}. {feat}")
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Patient Information")
        
        # Initialize form data
        form_data = {}
        
        with st.form("patient_form"):
            # Your specific high-risk patient data
            st.subheader("🧬 Your Test Patient")
            
            # Fixed values for your test patient
            age = 24
            gender = "Female"
            smoking = "Occasionally"
            diabetes = "Yes"
            cholesterol = 256
            bmi = 33.9
            systolic_bp = 138
            diastolic_bp = 76
            stress_level = "High"
            
            st.info("Using your high-risk patient: 24F, Diabetic, High Cholesterol")
            
            st.subheader("Clinical Measurements (Pre-filled)")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Age", age)
                st.metric("Gender", gender)
                st.metric("Diabetes", diabetes)
            with col_b:
                st.metric("Cholesterol", f"{cholesterol} mg/dL")
                st.metric("BMI", f"{bmi} kg/m²")
                st.metric("Stress Level", stress_level)
            
            st.subheader("Additional Information")
            alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Regularly"], index=1)
            physical_activity = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate", "High"], index=0)
            screen_time = st.slider("Screen Time (hours/day)", 0, 15, 15)
            sleep_duration = st.slider("Sleep Duration (hours/day)", 3, 12, 3)
            diet_type = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan"], index=1)
            family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"], index=1)
            hypertension = st.selectbox("Hypertension", ["No", "Yes"], index=0)
            resting_hr = st.slider("Resting Heart Rate (bpm)", 60, 120, 86)
            region = st.selectbox("Region", ["North", "South", "East", "West", "Central", "North-East"], index=0)
            urban_rural = st.selectbox("Urban/Rural", ["Urban", "Rural"], index=0)
            ses = st.selectbox("Socioeconomic Status", ["Low", "Middle", "High"], index=0)
            
            submitted = st.form_submit_button("🔍 Predict Heart Attack Risk")
            
            # Store all data when submitted
            if submitted:
                form_data = {
                    'age': age,
                    'gender': gender,
                    'smoking': smoking,
                    'alcohol': alcohol,
                    'physical_activity': physical_activity,
                    'screen_time': screen_time,
                    'sleep_duration': sleep_duration,
                    'diet_type': diet_type,
                    'stress_level': stress_level,
                    'family_history': family_history,
                    'diabetes': diabetes,
                    'hypertension': hypertension,
                    'cholesterol': cholesterol,
                    'bmi': bmi,
                    'systolic_bp': systolic_bp,
                    'diastolic_bp': diastolic_bp,
                    'resting_hr': resting_hr,
                    'region': region,
                    'urban_rural': urban_rural,
                    'ses': ses
                }
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        if submitted and form_data:
            with st.spinner("Analyzing patient data..."):
                # Get form data
                fd = form_data
                
                # Prepare patient data - SIMPLIFIED VERSION
                patient_data = {}
                
                # Add basic numerical features
                basic_features = {
                    'Age': fd['age'],
                    'Screen Time (hrs/day)': fd['screen_time'],
                    'Sleep Duration (hrs/day)': fd['sleep_duration'],
                    'Cholesterol Levels (mg/dL)': fd['cholesterol'],
                    'BMI (kg/m²)': fd['bmi'],
                    'Resting Heart Rate (bpm)': fd['resting_hr'],
                }
                
                # Add blood pressure features (try multiple names)
                bp_features = {
                    'Blood Pressure (systolic/diastolic mmHg)_systolic': fd['systolic_bp'],
                    'Blood Pressure (systolic/diastolic mmHg)_diastolic': fd['diastolic_bp'],
                    'Systolic_BP': fd['systolic_bp'],
                    'Diastolic_BP': fd['diastolic_bp'],
                }
                
                # Add categorical features
                categorical_features = {
                    'Gender_Female': 1 if fd['gender'] == "Female" else 0,
                    'Gender_Male': 1 if fd['gender'] == "Male" else 0,
                    'Smoking Status_Occasionally': 1 if fd['smoking'] == "Occasionally" else 0,
                    'Smoking Status_Never': 1 if fd['smoking'] == "Never" else 0,
                    'Diabetes_Yes': 1 if fd['diabetes'] == "Yes" else 0,
                    'Diabetes_No': 1 if fd['diabetes'] == "No" else 0,
                    'Stress Level_High': 1 if fd['stress_level'] == "High" else 0,
                    'Stress Level_Medium': 1 if fd['stress_level'] == "Medium" else 0,
                }
                
                # Combine all features
                patient_data = {**basic_features, **bp_features, **categorical_features}
                
                # Create feature vector
                feature_vector = create_feature_vector(patient_data, feature_columns)
                
                if feature_vector is not None:
                    try:
                        # Scale and predict
                        scaled_features = scaler.transform(feature_vector)
                        prediction_proba = model.predict(scaled_features, verbose=0)
                        probability = float(prediction_proba[0][0])
                        
                        # Display results
                        st.markdown("---")
                        
                        # Risk level
                        if probability >= 0.7:
                            risk_level = "HIGH RISK"
                            risk_class = "risk-high"
                            recommendation = "🚨 Immediate medical consultation recommended!"
                            emoji = "🔴"
                        elif probability >= 0.4:
                            risk_level = "MEDIUM RISK"
                            risk_class = "risk-medium"
                            recommendation = "⚠️ Regular health monitoring advised."
                            emoji = "🟡"
                        else:
                            risk_level = "LOW RISK"
                            risk_class = "risk-low"
                            recommendation = "✅ Maintain healthy lifestyle."
                            emoji = "🟢"
                        
                        st.markdown(f'<div class="{risk_class}">{emoji} {risk_level} - {probability:.1%} Probability</div>', 
                                  unsafe_allow_html=True)
                        
                        # Progress bar
                        st.subheader("Risk Probability Gauge")
                        gauge_value = probability * 100
                        st.progress(int(gauge_value))
                        st.write(f"**{gauge_value:.1f}%** probability of heart attack risk")
                        
                        # Show what we know about the patient
                        st.subheader("🔍 Patient Risk Factors")
                        risk_factors = [
                            f"• Age: {fd['age']}",
                            f"• Gender: {fd['gender']}",
                            f"• Diabetes: {fd['diabetes']}",
                            f"• Cholesterol: {fd['cholesterol']} mg/dL",
                            f"• BMI: {fd['bmi']} kg/m²",
                            f"• Smoking: {fd['smoking']}",
                            f"• Stress Level: {fd['stress_level']}",
                        ]
                        
                        for factor in risk_factors:
                            st.write(factor)
                        
                        st.info(recommendation)
                        
                        # Debug information
                        with st.expander("🔧 Technical Details"):
                            st.write(f"Features in model: {len(feature_columns)}")
                            st.write(f"Features sent: {len(patient_data)}")
                            st.write(f"Prediction raw: {prediction_proba[0][0]}")
                            
                            if probability == 0.0:
                                st.error("""
                                **Why 0%?** 
                                - Feature names don't match model expectations
                                - Run diagnostic in Colab to see exact feature names
                                - Update the feature mapping in this app
                                """)
                    
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
        
        else:
            # Default view
            st.info("👆 Click 'Predict Heart Attack Risk' to analyze your patient")
            
            st.markdown("---")
            st.subheader("🎯 Expected Results")
            st.warning("""
            **Your High-Risk Patient:**
            - 24-year-old Female
            - Diabetes: Yes
            - High Cholesterol (256 mg/dL)  
            - High BMI (33.9 kg/m²)
            - High Stress, Occasional Smoking
            
            **Should Show:** MEDIUM to HIGH risk (60-80%)
            
            **If showing 0%:** Feature mapping issue - need to match exact feature names
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><em>⚠️ Educational tool only. Consult healthcare professionals.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
