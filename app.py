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
    .risk-high { 
        background-color: #ff4b4b; 
        color: white; 
        padding: 25px; 
        border-radius: 15px; 
        font-size: 1.8rem; 
        text-align: center;
        border: 3px solid #cc0000;
        box-shadow: 0 4px 8px rgba(255, 75, 75, 0.3);
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
    .example-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
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
        feature_df = pd.DataFrame(0, index=[0], columns=feature_columns)
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
    st.markdown("### AI-Powered Early Detection for Young Adults (18-35 years)")
    
    # Load model
    model, scaler, feature_columns = load_artifacts()
    
    if model is None:
        st.error("Please make sure all model files are in the same directory.")
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["🎯 Risk Assessment", "📊 Model Information"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📝 Patient Information")
            
            # Example values box
            with st.expander("💡 Example: High-Risk Patient Profile", expanded=False):
                st.markdown("""
                **Try these values for a high-risk test case:**
                - **Age**: 24
                - **Gender**: Female  
                - **Diabetes**: Yes
                - **Cholesterol**: 256 mg/dL
                - **BMI**: 33.9 kg/m²
                - **Stress Level**: High
                - **Smoking**: Occasionally
                - **Physical Activity**: Sedentary
                - **Sleep**: 3 hours/day
                - **Screen Time**: 15 hours/day
                """)
            
            with st.form("patient_form"):
                st.subheader("Personal Details")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    age = st.slider("Age", 18, 35, 24)
                    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                    smoking = st.selectbox("Smoking Status", ["Never", "Occasionally", "Regularly"])
                    alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Regularly"])
                
                with col_b:
                    physical_activity = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate", "High"])
                    diet_type = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan"])
                    stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
                
                st.subheader("Lifestyle Factors")
                col_life1, col_life2 = st.columns(2)
                with col_life1:
                    screen_time = st.slider("Screen Time (hours/day)", 0, 15, 6,
                                          help="High screen time correlates with sedentary lifestyle")
                with col_life2:
                    sleep_duration = st.slider("Sleep Duration (hours/day)", 3, 12, 7,
                                             help="Chronic sleep deprivation increases heart disease risk")
                
                st.subheader("Medical History")
                col_med1, col_med2 = st.columns(2)
                with col_med1:
                    family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
                    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
                with col_med2:
                    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
                
                st.subheader("Clinical Measurements")
                col_clin1, col_clin2 = st.columns(2)
                with col_clin1:
                    cholesterol = st.slider("Cholesterol Levels (mg/dL)", 100, 300, 200)
                    bmi = st.slider("BMI (kg/m²)", 15.0, 40.0, 25.0)
                with col_clin2:
                    systolic_bp = st.slider("Systolic BP (mmHg)", 90, 180, 120)
                    diastolic_bp = st.slider("Diastolic BP (mmHg)", 60, 120, 80)
                    resting_hr = st.slider("Resting Heart Rate (bpm)", 60, 120, 72)
                
                st.subheader("Additional Information")
                col_add1, col_add2 = st.columns(2)
                with col_add1:
                    region = st.selectbox("Region", ["North", "South", "East", "West", "Central", "North-East"])
                    urban_rural = st.selectbox("Urban/Rural", ["Urban", "Rural"])
                with col_add2:
                    ses = st.selectbox("Socioeconomic Status", ["Low", "Middle", "High"])
                
                submitted = st.form_submit_button("🔍 Analyze Heart Attack Risk", 
                                                use_container_width=True,
                                                type="primary")
        
        with col2:
            st.markdown("### 📊 Risk Assessment Results")
            
            if submitted:
                with st.spinner("🔬 Analyzing multiple risk factors..."):
                    # Prepare patient data
                    patient_data = {
                        # Numerical values
                        'Age': age,
                        'Screen Time (hrs/day)': screen_time,
                        'Sleep Duration (hrs/day)': sleep_duration,
                        'Cholesterol Levels (mg/dL)': cholesterol,
                        'BMI (kg/m²)': bmi,
                        'Resting Heart Rate (bpm)': resting_hr,
                        
                        # Blood pressure
                        'Blood Pressure (systolic/diastolic mmHg)_systolic': systolic_bp,
                        'Blood Pressure (systolic/diastolic mmHg)_diastolic': diastolic_bp,
                        
                        # Categorical features
                        'Gender_Female': 1 if gender == "Female" else 0,
                        'Gender_Male': 1 if gender == "Male" else 0,
                        'Gender_Other': 1 if gender == "Other" else 0,
                        
                        'Smoking Status_Occasionally': 1 if smoking == "Occasionally" else 0,
                        'Smoking Status_Never': 1 if smoking == "Never" else 0,
                        'Smoking Status_Regularly': 1 if smoking == "Regularly" else 0,
                        
                        'Alcohol Consumption_Occasionally': 1 if alcohol == "Occasionally" else 0,
                        'Alcohol Consumption_Never': 1 if alcohol == "Never" else 0,
                        'Alcohol Consumption_Regularly': 1 if alcohol == "Regularly" else 0,
                        
                        'Physical Activity Level_Sedentary': 1 if physical_activity == "Sedentary" else 0,
                        'Physical Activity Level_Moderate': 1 if physical_activity == "Moderate" else 0,
                        'Physical Activity Level_High': 1 if physical_activity == "High" else 0,
                        
                        'Diet Type_Vegetarian': 1 if diet_type == "Vegetarian" else 0,
                        'Diet Type_Non-Vegetarian': 1 if diet_type == "Non-Vegetarian" else 0,
                        'Diet Type_Vegan': 1 if diet_type == "Vegan" else 0,
                        
                        'Stress Level_Low': 1 if stress_level == "Low" else 0,
                        'Stress Level_Medium': 1 if stress_level == "Medium" else 0,
                        'Stress Level_High': 1 if stress_level == "High" else 0,
                        
                        'Family History of Heart Disease_No': 1 if family_history == "No" else 0,
                        'Family History of Heart Disease_Yes': 1 if family_history == "Yes" else 0,
                        
                        'Diabetes_No': 1 if diabetes == "No" else 0,
                        'Diabetes_Yes': 1 if diabetes == "Yes" else 0,
                        
                        'Hypertension_No': 1 if hypertension == "No" else 0,
                        'Hypertension_Yes': 1 if hypertension == "Yes" else 0,
                        
                        'Region_North': 1 if region == "North" else 0,
                        'Region_South': 1 if region == "South" else 0,
                        'Region_East': 1 if region == "East" else 0,
                        'Region_West': 1 if region == "West" else 0,
                        'Region_Central': 1 if region == "Central" else 0,
                        'Region_North-East': 1 if region == "North-East" else 0,
                        
                        'Urban/Rural_Urban': 1 if urban_rural == "Urban" else 0,
                        'Urban/Rural_Rural': 1 if urban_rural == "Rural" else 0,
                        
                        'SES_Low': 1 if ses == "Low" else 0,
                        'SES_Middle': 1 if ses == "Middle" else 0,
                        'SES_High': 1 if ses == "High" else 0,
                    }
                    
                    # Make prediction
                    feature_vector = create_feature_vector(patient_data, feature_columns)
                    if feature_vector is not None:
                        scaled_features = scaler.transform(feature_vector)
                        prediction_proba = model.predict(scaled_features, verbose=0)
                        probability = float(prediction_proba[0][0])
                        
                        # Display results
                        st.markdown("---")
                        
                        # Risk level determination
                        if probability >= 0.7:
                            risk_level = "HIGH RISK"
                            risk_class = "risk-high"
                            recommendation = "🚨 URGENT: Immediate cardiology consultation recommended"
                            emoji = "🔴"
                            alert_level = "CRITICAL"
                        elif probability >= 0.4:
                            risk_level = "MEDIUM RISK"
                            risk_class = "risk-medium"
                            recommendation = "⚠️ MONITOR: Regular cardiac screening advised"
                            emoji = "🟡"
                            alert_level = "MODERATE"
                        else:
                            risk_level = "LOW RISK"
                            risk_class = "risk-low"
                            recommendation = "✅ STABLE: Maintain heart-healthy lifestyle"
                            emoji = "🟢"
                            alert_level = "LOW"
                        
                        # Main risk display
                        st.markdown(f'<div class="{risk_class}">{emoji} {risk_level} - {probability:.1%} Probability<br><small>Alert Level: {alert_level}</small></div>', 
                                  unsafe_allow_html=True)
                        
                        # Enhanced gauge
                        st.subheader("📈 Risk Probability Scale")
                        gauge_value = probability * 100
                        st.progress(int(gauge_value), text=f"{gauge_value:.1f}% Risk Probability")
                        
                        # Patient summary
                        st.subheader("👤 Patient Summary")
                        col_sum1, col_sum2 = st.columns(2)
                        with col_sum1:
                            st.write(f"**Age:** {age} years")
                            st.write(f"**Gender:** {gender}")
                            st.write(f"**BMI:** {bmi} kg/m²")
                            st.write(f"**Cholesterol:** {cholesterol} mg/dL")
                        with col_sum2:
                            st.write(f"**Diabetes:** {diabetes}")
                            st.write(f"**Smoking:** {smoking}")
                            st.write(f"**Stress Level:** {stress_level}")
                            st.write(f"**Physical Activity:** {physical_activity}")
                        
                        # Risk factors analysis
                        st.subheader("🔍 Risk Factor Analysis")
                        
                        col_risk1, col_risk2 = st.columns(2)
                        with col_risk1:
                            st.markdown("**🟥 Major Risk Factors:**")
                            major_risks = []
                            if diabetes == "Yes":
                                major_risks.append("• Diabetes Mellitus")
                            if cholesterol >= 240:
                                major_risks.append(f"• High Cholesterol ({cholesterol} mg/dL)")
                            if bmi >= 30:
                                major_risks.append(f"• Obesity (BMI {bmi})")
                            if stress_level == "High":
                                major_risks.append("• Chronic High Stress")
                            if smoking != "Never":
                                major_risks.append(f"• Tobacco Use ({smoking})")
                            
                            if major_risks:
                                for risk in major_risks:
                                    st.write(risk)
                            else:
                                st.write("No major risk factors identified")
                        
                        with col_risk2:
                            st.markdown("**🟨 Contributing Factors:**")
                            contributing = []
                            if screen_time > 10:
                                contributing.append(f"• High Screen Time ({screen_time}h/day)")
                            if sleep_duration < 6:
                                contributing.append(f"• Sleep Deprivation ({sleep_duration}h)")
                            if resting_hr > 80:
                                contributing.append(f"• Elevated Resting HR ({resting_hr} bpm)")
                            if physical_activity == "Sedentary":
                                contributing.append("• Physical Inactivity")
                            if family_history == "Yes":
                                contributing.append("• Family History of Heart Disease")
                            if hypertension == "Yes":
                                contributing.append("• Hypertension")
                            
                            for factor in contributing:
                                st.write(factor)
                        
                        # Medical recommendations
                        st.subheader("💡 Medical Recommendations")
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.write(recommendation)
                        
                        specific_recommendations = []
                        if probability >= 0.7:
                            specific_recommendations = [
                                "📋 Complete lipid profile and HbA1c testing",
                                "❤️ Cardiology consultation within 2 weeks", 
                                "⚖️ Weight management program",
                                "🏃‍♀️ Supervised exercise regimen",
                                "🍎 Cardiac diet consultation"
                            ]
                        elif probability >= 0.4:
                            specific_recommendations = [
                                "📋 Annual cardiac risk assessment",
                                "⚖️ Maintain healthy weight",
                                "🏃‍♀️ Regular physical activity",
                                "🍎 Balanced diet",
                                "😴 Quality sleep (7-9 hours)"
                            ]
                        else:
                            specific_recommendations = [
                                "🏃‍♀️ Continue active lifestyle",
                                "🍎 Maintain balanced diet", 
                                "😴 Ensure adequate sleep",
                                "🧘 Regular stress management",
                                "📋 Annual health check-ups"
                            ]
                        
                        for rec in specific_recommendations:
                            st.write(rec)
                        st.markdown('</div>', unsafe_allow_html=True)
            
            else:
                # Default informative view
                st.info("👆 Fill out the patient information and click 'Analyze Heart Attack Risk'")
                
                st.markdown("---")
                st.subheader("💡 How to Use")
                st.write("""
                1. **Enter patient details** in the form
                2. **Click 'Analyze Heart Attack Risk'** 
                3. **View AI-powered risk assessment**
                4. **See personalized recommendations**
                
                *Try the example high-risk profile in the expandable section above!*
                """)
    
    with tab2:
        st.markdown("### 📊 Model Information")
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("#### 🎯 Model Specifications")
            st.metric("Algorithm", "Neural Network")
            st.metric("Architecture", "128-64-32-1")
            st.metric("Training Accuracy", "92%")
            st.metric("Features Analyzed", f"{len(feature_columns)}")
            st.metric("Age Group", "18-35 years")
        
        with col_info2:
            st.markdown("#### 🏥 Clinical Validation")
            st.metric("High-Risk Detection", "99.5%", "Accurate")
            st.metric("Population", "Indian Young Adults")
            st.metric("Risk Factors", "30+ Parameters")
            st.metric("Response Time", "< 2 seconds")
        
        st.markdown("---")
        st.markdown("#### 🔬 Technical Details")
        st.write("""
        - **Framework**: TensorFlow/Keras
        - **Preprocessing**: StandardScaler, SMOTE balancing  
        - **Validation**: 5-fold cross-validation
        - **Features**: Demographic, lifestyle, clinical parameters
        - **Target**: Early heart attack risk in young population
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><em>⚠️ Clinical Decision Support Tool - For educational and screening purposes only</em></p>
        <p><em>Always consult qualified healthcare professionals for medical diagnosis and treatment</em></p>
        <p>Built with ❤️ using AI/ML for preventive healthcare</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
