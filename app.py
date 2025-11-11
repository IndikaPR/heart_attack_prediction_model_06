import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ff4b4b;
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 1.5rem;
        text-align: center;
    }
    .risk-medium {
        background-color: #ffa500;
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 1.5rem;
        text-align: center;
    }
    .risk-low {
        background-color: #00cc66;
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 1.5rem;
        text-align: center;
    }
    .feature-importance {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
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

def predict_heart_attack(model, scaler, feature_columns, patient_data):
    """Make prediction for a single patient"""
    try:
        # Create feature vector
        feature_vector = create_feature_vector(patient_data, feature_columns)
        
        if feature_vector is None:
            return None, None
            
        # Scale features
        scaled_features = scaler.transform(feature_vector)
        
        # Make prediction
        prediction_proba = model.predict(scaled_features, verbose=0)
        probability = float(prediction_proba[0][0])
        
        return probability, feature_vector
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def get_available_features(feature_columns):
    """Extract available feature names for debugging"""
    bp_features = [col for col in feature_columns if 'blood pressure' in col.lower() or 'bp' in col.lower() or 'systolic' in col.lower() or 'diastolic' in col.lower()]
    numerical_features = [col for col in feature_columns if col.replace('.', '').isdigit() or any(x in col.lower() for x in ['age', 'cholesterol', 'bmi', 'screen', 'sleep', 'rate'])]
    categorical_features = [col for col in feature_columns if col not in numerical_features and col not in bp_features]
    
    return {
        'bp_features': bp_features,
        'numerical_features': numerical_features[:10],  # First 10 only
        'categorical_features': categorical_features[:10]  # First 10 only
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Attack Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Early Detection for Young Adults (18-35 years)")
    
    # Load model
    model, scaler, feature_columns = load_artifacts()
    
    if model is None:
        st.error("Please make sure all model files (heart_attack_model.h5, scaler.pkl, feature_columns.pkl) are in the same directory as this app.")
        return
    
    # Debug: Show available features (optional - you can remove this section)
    with st.expander("🔧 Debug: Available Features"):
        available_features = get_available_features(feature_columns)
        st.write("Blood Pressure Features:", available_features['bp_features'])
        st.write("Numerical Features:", available_features['numerical_features'])
        st.write("Categorical Features:", available_features['categorical_features'])
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Patient Information")
        
        with st.form("patient_form"):
            # Personal Information
            st.subheader("Personal Details")
            age = st.slider("Age", 18, 35, 24)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            # Lifestyle Factors
            st.subheader("Lifestyle Factors")
            smoking = st.selectbox("Smoking Status", ["Never", "Occasionally", "Regularly"])
            alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Regularly"])
            physical_activity = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate", "High"])
            screen_time = st.slider("Screen Time (hours/day)", 0, 15, 6)
            sleep_duration = st.slider("Sleep Duration (hours/day)", 3, 12, 7)
            diet_type = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan"])
            stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
            
            # Medical History
            st.subheader("Medical History")
            family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            
            # Clinical Measurements - FIXED: Separate systolic and diastolic
            st.subheader("Clinical Measurements")
            cholesterol = st.slider("Cholesterol Levels (mg/dL)", 100, 300, 200)
            bmi = st.slider("BMI (kg/m²)", 15.0, 40.0, 25.0)
            
            # Blood pressure as separate numerical values
            col_bp1, col_bp2 = st.columns(2)
            with col_bp1:
                systolic_bp = st.slider("Systolic BP (mmHg)", 90, 180, 120)
            with col_bp2:
                diastolic_bp = st.slider("Diastolic BP (mmHg)", 60, 120, 80)
            
            resting_hr = st.slider("Resting Heart Rate (bpm)", 60, 120, 72)
            
            # Region and SES
            region = st.selectbox("Region", ["North", "South", "East", "West", "Central", "North-East"])
            urban_rural = st.selectbox("Urban/Rural", ["Urban", "Rural"])
            ses = st.selectbox("Socioeconomic Status", ["Low", "Middle", "High"])
            
            submitted = st.form_submit_button("🔍 Predict Heart Attack Risk", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        if submitted:
            with st.spinner("Analyzing patient data..."):
                # Prepare input data - FIXED VERSION
                input_data = {
                    'Age': float(age),
                    f'Gender_{gender}': 1,
                    f'Smoking Status_{smoking}': 1,
                    f'Alcohol Consumption_{alcohol}': 1,
                    f'Physical Activity Level_{physical_activity}': 1,
                    'Screen Time (hrs/day)': float(screen_time),
                    'Sleep Duration (hrs/day)': float(sleep_duration),
                    f'Diet Type_{diet_type}': 1,
                    f'Stress Level_{stress_level}': 1,
                    f'Family History of Heart Disease_{family_history}': 1,
                    f'Diabetes_{diabetes}': 1,
                    f'Hypertension_{hypertension}': 1,
                    'Cholesterol Levels (mg/dL)': float(cholesterol),
                    'BMI (kg/m²)': float(bmi),
                    f'Region_{region}': 1,
                    f'Urban/Rural_{urban_rural}': 1,
                    f'SES_{ses}': 1,
                    'Resting Heart Rate (bpm)': float(resting_hr)
                }
                
                # Handle blood pressure based on actual feature names in your model
                # Try different possible feature names for blood pressure
                bp_features_found = False
                
                # Option 1: If your model has separate systolic/diastolic features
                if 'Blood Pressure (systolic/diastolic mmHg)_systolic' in feature_columns:
                    input_data['Blood Pressure (systolic/diastolic mmHg)_systolic'] = float(systolic_bp)
                    bp_features_found = True
                
                if 'Blood Pressure (systolic/diastolic mmHg)_diastolic' in feature_columns:
                    input_data['Blood Pressure (systolic/diastolic mmHg)_diastolic'] = float(diastolic_bp)
                    bp_features_found = True
                
                # Option 2: If your model has combined BP feature (we'll skip it since it causes errors)
                # Option 3: If your model has different BP feature names
                bp_possible_names = [
                    'Systolic_BP', 'Diastolic_BP',
                    'Blood Pressure_systolic', 'Blood Pressure_diastolic',
                    'BP_Systolic', 'BP_Diastolic'
                ]
                
                for bp_name in bp_possible_names:
                    if bp_name in feature_columns:
                        if 'systolic' in bp_name.lower() or 'systolic' in bp_name:
                            input_data[bp_name] = float(systolic_bp)
                            bp_features_found = True
                        elif 'diastolic' in bp_name.lower() or 'diastolic' in bp_name:
                            input_data[bp_name] = float(diastolic_bp)
                            bp_features_found = True
                
                # If no BP features found, show warning but continue
                if not bp_features_found:
                    st.warning("⚠️ Blood pressure features not found in model. Using default values.")
                    # Add default BP values if features exist but with different names
                    for col in feature_columns:
                        if 'systolic' in col.lower():
                            input_data[col] = 120.0  # default
                        elif 'diastolic' in col.lower():
                            input_data[col] = 80.0   # default
                
                # Debug: Show what data we're sending (optional)
                with st.expander("🔍 Debug: Input Data Sent"):
                    st.write("Numerical values:", {k: v for k, v in input_data.items() if isinstance(v, (int, float))})
                    st.write("Categorical values:", {k: v for k, v in input_data.items() if not isinstance(v, (int, float))})
                
                # Make prediction
                probability, feature_vector = predict_heart_attack(model, scaler, feature_columns, input_data)
                
                if probability is not None:
                    # Display results
                    st.markdown("---")
                    
                    # Risk level determination
                    if probability >= 0.7:
                        risk_level = "HIGH RISK"
                        risk_class = "risk-high"
                        recommendation = "🚨 Immediate medical consultation recommended. Please consult a cardiologist."
                        emoji = "🔴"
                    elif probability >= 0.4:
                        risk_level = "MEDIUM RISK"
                        risk_class = "risk-medium"
                        recommendation = "⚠️ Regular health monitoring advised. Consider lifestyle changes."
                        emoji = "🟡"
                    else:
                        risk_level = "LOW RISK"
                        risk_class = "risk-low"
                        recommendation = "✅ Maintain healthy lifestyle. Regular checkups recommended."
                        emoji = "🟢"
                    
                    # Display risk box
                    st.markdown(f'<div class="{risk_class}">{emoji} {risk_level} - {probability:.1%} Probability</div>', 
                              unsafe_allow_html=True)
                    
                    # Probability gauge
                    st.subheader("Risk Probability Gauge")
                    gauge_value = probability * 100
                    st.progress(int(gauge_value))
                    st.write(f"**{gauge_value:.1f}%** probability of heart attack risk")
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    st.info(recommendation)
                    
                    # Key risk factors
                    st.subheader("🔍 Key Risk Factors Identified")
                    
                    risk_factors = []
                    if diabetes == "Yes":
                        risk_factors.append("• Diabetes")
                    if hypertension == "Yes":
                        risk_factors.append("• Hypertension")
                    if family_history == "Yes":
                        risk_factors.append("• Family History of Heart Disease")
                    if smoking != "Never":
                        risk_factors.append(f"• Smoking: {smoking}")
                    if stress_level == "High":
                        risk_factors.append("• High Stress Level")
                    if bmi >= 30:
                        risk_factors.append("• High BMI (Obesity)")
                    if cholesterol >= 240:
                        risk_factors.append("• High Cholesterol")
                    if physical_activity == "Sedentary":
                        risk_factors.append("• Sedentary Lifestyle")
                    
                    if risk_factors:
                        for factor in risk_factors:
                            st.write(factor)
                    else:
                        st.write("No major risk factors identified")
                    
                    # Protective factors
                    st.subheader("🛡️ Protective Factors")
                    protective_factors = []
                    if physical_activity == "High":
                        protective_factors.append("• High Physical Activity")
                    if smoking == "Never":
                        protective_factors.append("• Non-smoker")
                    if diet_type == "Vegetarian" or diet_type == "Vegan":
                        protective_factors.append("• Plant-based Diet")
                    if stress_level == "Low":
                        protective_factors.append("• Low Stress Level")
                    if bmi < 25:
                        protective_factors.append("• Healthy BMI")
                    if cholesterol < 200:
                        protective_factors.append("• Healthy Cholesterol")
                    
                    for factor in protective_factors:
                        st.success(factor)
        
        else:
            # Default view before prediction
            st.info("👆 Fill out the patient information form and click 'Predict Heart Attack Risk' to get started.")
            
            # Sample predictions for demonstration
            st.markdown("---")
            st.subheader("📈 Sample Risk Scenarios")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Low Risk Profile", "15-30%", "Healthy")
                st.caption("Young, active, no family history")
            
            with col_b:
                st.metric("Medium Risk Profile", "31-69%", "Monitor") 
                st.caption("Some risk factors present")
            
            with col_c:
                st.metric("High Risk Profile", "70-95%", "Alert")
                st.caption("Multiple risk factors")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><em>⚠️ Disclaimer: This tool is for educational purposes only. 
        Always consult healthcare professionals for medical advice.</em></p>
        <p>Built with ❤️ using Streamlit and TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
