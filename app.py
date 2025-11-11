import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

# Page configuration
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .debug-info { background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff; }
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

def run_feature_diagnostic(feature_columns):
    """Run diagnostic on feature names"""
    diagnostic_results = {}
    
    # Blood pressure features
    diagnostic_results['bp_features'] = [
        col for col in feature_columns 
        if any(x in col.lower() for x in ['blood', 'pressure', 'bp', 'systolic', 'diastolic'])
    ]
    
    # Key numerical features
    diagnostic_results['numerical_features'] = [
        col for col in feature_columns 
        if any(x in col.lower() for x in ['age', 'cholesterol', 'bmi', 'screen', 'sleep', 'rate', 'time', 'level'])
    ]
    
    # Categorical features
    diagnostic_results['categorical_features'] = [
        col for col in feature_columns 
        if any(x in col.lower() for x in ['gender', 'smoking', 'alcohol', 'diet', 'stress', 'diabetes', 'hypertension', 'family', 'region', 'urban', 'ses'])
    ]
    
    # First and last features
    diagnostic_results['first_10_features'] = feature_columns[:10]
    diagnostic_results['last_10_features'] = feature_columns[-10:]
    
    return diagnostic_results

def create_feature_vector(input_data, feature_columns):
    """Create feature vector matching training data structure"""
    try:
        # Create a dataframe with all feature columns initialized to 0
        feature_df = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # Map input data to feature columns
        mapped_count = 0
        for key, value in input_data.items():
            if key in feature_df.columns:
                feature_df[key] = value
                mapped_count += 1
        
        return feature_df, mapped_count
    except Exception as e:
        st.error(f"Error creating feature vector: {e}")
        return None, 0

def predict_heart_attack(model, scaler, feature_columns, patient_data):
    """Make prediction for a single patient"""
    try:
        # Create feature vector
        feature_vector, mapped_count = create_feature_vector(patient_data, feature_columns)
        
        if feature_vector is None:
            return None, None, 0
            
        # Scale features
        scaled_features = scaler.transform(feature_vector)
        
        # Make prediction
        prediction_proba = model.predict(scaled_features, verbose=0)
        probability = float(prediction_proba[0][0])
        
        return probability, feature_vector, mapped_count
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, 0

def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Attack Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Early Detection for Young Adults (18-35 years)")
    
    # Load model
    model, scaler, feature_columns = load_artifacts()
    
    if model is None:
        st.error("Please make sure all model files (heart_attack_model.h5, scaler.pkl, feature_columns.pkl) are in the same directory as this app.")
        return
    
    # Run diagnostic on features
    diagnostic_results = run_feature_diagnostic(feature_columns)
    
    # Show diagnostic information in sidebar
    with st.sidebar:
        st.markdown("### 🔧 Model Diagnostics")
        st.write(f"**Total Features:** {len(feature_columns)}")
        st.write(f"**BP Features:** {len(diagnostic_results['bp_features'])}")
        st.write(f"**Numerical Features:** {len(diagnostic_results['numerical_features'])}")
        
        with st.expander("📋 View Feature Names"):
            st.write("**Blood Pressure Features:**")
            for feat in diagnostic_results['bp_features']:
                st.write(f"• {feat}")
            
            st.write("**First 10 Features:**")
            for feat in diagnostic_results['first_10_features']:
                st.write(f"• {feat}")
    
    # Initialize session state for form data
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["🎯 Risk Prediction", "🔍 Feature Diagnostics"])
    
    with tab1:
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
                screen_time = st.slider("Screen Time (hours/day)", 0, 15, 15)  # High from your data
                sleep_duration = st.slider("Sleep Duration (hours/day)", 3, 12, 3)  # Low from your data
                diet_type = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan"])
                stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
                
                # Medical History
                st.subheader("Medical History")
                family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
                diabetes = st.selectbox("Diabetes", ["No", "Yes"])
                hypertension = st.selectbox("Hypertension", ["No", "Yes"])
                
                # Clinical Measurements
                st.subheader("Clinical Measurements")
                cholesterol = st.slider("Cholesterol Levels (mg/dL)", 100, 300, 256)
                bmi = st.slider("BMI (kg/m²)", 15.0, 40.0, 33.9)
                
                # Blood pressure
                col_bp1, col_bp2 = st.columns(2)
                with col_bp1:
                    systolic_bp = st.slider("Systolic BP (mmHg)", 90, 180, 138)
                with col_bp2:
                    diastolic_bp = st.slider("Diastolic BP (mmHg)", 60, 120, 76)
                
                resting_hr = st.slider("Resting Heart Rate (bpm)", 60, 120, 86)
                
                # Additional Information
                st.subheader("Additional Information")
                region = st.selectbox("Region", ["North", "South", "East", "West", "Central", "North-East"])
                urban_rural = st.selectbox("Urban/Rural", ["Urban", "Rural"])
                ses = st.selectbox("Socioeconomic Status", ["Low", "Middle", "High"])
                
                submitted = st.form_submit_button("🔍 Predict Heart Attack Risk", use_container_width=True)
                
                # Store form data in session state when submitted
                if submitted:
                    st.session_state.form_data = {
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
            
            if st.session_state.form_data and submitted:
                with st.spinner("Analyzing patient data..."):
                    # Get form data from session state
                    fd = st.session_state.form_data
                    
                    # Prepare patient data - TRYING ALL POSSIBLE FEATURE NAMES
                    patient_data = {}
                    
                    # Add all possible feature combinations
                    possible_features = [
                        # Age variations
                        ('Age', fd['age']),
                        ('age', fd['age']),
                        
                        # Screen time
                        ('Screen Time (hrs/day)', fd['screen_time']),
                        ('Screen Time', fd['screen_time']),
                        ('screen_time', fd['screen_time']),
                        
                        # Sleep duration
                        ('Sleep Duration (hrs/day)', fd['sleep_duration']),
                        ('Sleep Duration', fd['sleep_duration']),
                        ('sleep_duration', fd['sleep_duration']),
                        
                        # Cholesterol
                        ('Cholesterol Levels (mg/dL)', fd['cholesterol']),
                        ('Cholesterol', fd['cholesterol']),
                        ('cholesterol', fd['cholesterol']),
                        
                        # BMI
                        ('BMI (kg/m²)', fd['bmi']),
                        ('BMI', fd['bmi']),
                        ('bmi', fd['bmi']),
                        
                        # Resting heart rate
                        ('Resting Heart Rate (bpm)', fd['resting_hr']),
                        ('Resting Heart Rate', fd['resting_hr']),
                        ('resting_hr', fd['resting_hr']),
                    ]
                    
                    # Add all possible blood pressure features
                    bp_features = [
                        ('Blood Pressure (systolic/diastolic mmHg)_systolic', fd['systolic_bp']),
                        ('Blood Pressure (systolic/diastolic mmHg)_diastolic', fd['diastolic_bp']),
                        ('Systolic_BP', fd['systolic_bp']),
                        ('Diastolic_BP', fd['diastolic_bp']),
                        ('Blood Pressure_systolic', fd['systolic_bp']),
                        ('Blood Pressure_diastolic', fd['diastolic_bp']),
                        ('systolic', fd['systolic_bp']),
                        ('diastolic', fd['diastolic_bp']),
                    ]
                    
                    # Add categorical features
                    categorical_features = [
                        # Gender
                        (f'Gender_{fd["gender"]}', 1),
                        ('Gender_Female', 1 if fd['gender'] == "Female" else 0),
                        ('Gender_Male', 1 if fd['gender'] == "Male" else 0),
                        ('Gender_Other', 1 if fd['gender'] == "Other" else 0),
                        
                        # Smoking
                        (f'Smoking Status_{fd["smoking"]}', 1),
                        ('Smoking Status_Occasionally', 1 if fd['smoking'] == "Occasionally" else 0),
                        ('Smoking Status_Never', 1 if fd['smoking'] == "Never" else 0),
                        ('Smoking Status_Regularly', 1 if fd['smoking'] == "Regularly" else 0),
                        
                        # Diabetes
                        (f'Diabetes_{fd["diabetes"]}', 1),
                        ('Diabetes_No', 1 if fd['diabetes'] == "No" else 0),
                        ('Diabetes_Yes', 1 if fd['diabetes'] == "Yes" else 0),
                        
                        # Stress
                        (f'Stress Level_{fd["stress_level"]}', 1),
                        ('Stress Level_Low', 1 if fd['stress_level'] == "Low" else 0),
                        ('Stress Level_Medium', 1 if fd['stress_level'] == "Medium" else 0),
                        ('Stress Level_High', 1 if fd['stress_level'] == "High" else 0),
                    ]
                    
                    # Combine all features
                    all_possible_features = possible_features + bp_features + categorical_features
                    
                    # Add to patient_data
                    for feature_name, value in all_possible_features:
                        patient_data[feature_name] = value
                    
                    # Make prediction
                    probability, feature_vector, mapped_count = predict_heart_attack(
                        model, scaler, feature_columns, patient_data
                    )
                    
                    # Display diagnostic information
                    with st.expander("🔧 Feature Mapping Diagnostics", expanded=True):
                        st.markdown('<div class="debug-info">', unsafe_allow_html=True)
                        st.write(f"**Features in model:** {len(feature_columns)}")
                        st.write(f"**Features successfully mapped:** {mapped_count}")
                        st.write(f"**Mapping rate:** {mapped_count/len(feature_columns)*100:.1f}%")
                        
                        if mapped_count == 0:
                            st.error("❌ NO features were mapped! This is why you're getting 0% probability.")
                            st.info("Check the 'Feature Diagnostics' tab to see the exact feature names your model expects.")
                        elif mapped_count < len(feature_columns) * 0.5:  # Less than 50% mapped
                            st.warning("⚠️ Low feature mapping rate. This affects prediction accuracy.")
                        else:
                            st.success("✅ Good feature mapping rate.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    if probability is not None:
                        # Display results
                        st.markdown("---")
                        
                        # Risk level determination
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
                        
                        # Display risk box
                        st.markdown(f'<div class="{risk_class}">{emoji} {risk_level} - {probability:.1%} Probability</div>', 
                                  unsafe_allow_html=True)
                        
                        # If 0% probability with high-risk patient, show special message
                        if probability == 0.0 and fd['diabetes'] == "Yes" and fd['cholesterol'] >= 240:
                            st.error("""
                            🚨 **ISSUE DETECTED**: 
                            High-risk patient showing 0% probability indicates feature mapping problem.
                            
                            **Please check the Feature Diagnostics tab to see the exact feature names your model expects.**
                            """)
            
            elif not submitted:
                # Default view before prediction
                st.info("👆 Fill out the patient information form and click 'Predict Heart Attack Risk' to get started.")
    
    with tab2:
        st.markdown("### 🔍 Feature Diagnostics")
        st.info("This tab shows the exact feature names your model expects. Use this information to fix feature mapping issues.")
        
        st.markdown("#### 📋 All Feature Names in Your Model")
        st.write(f"**Total features:** {len(feature_columns)}")
        
        # Display features in a scrollable box
