# diagnostic.py - Save this as a .py file and run locally
import joblib
import pandas as pd

print("🔍 ANALYZING YOUR MODEL'S FEATURE NAMES")
print("=" * 50)

try:
    # Load feature columns - make sure these files are in same folder
    feature_columns = joblib.load('feature_columns.pkl')
    print(f"✅ Loaded {len(feature_columns)} feature columns")
    
    # Show first 30 features (most important)
    print("\n📋 FIRST 30 FEATURE NAMES:")
    for i, col in enumerate(feature_columns[:30]):
        print(f"{i+1:2d}. {col}")
    
    # Blood pressure features
    bp_features = [col for col in feature_columns if any(x in col.lower() for x in ['blood', 'pressure', 'bp', 'systolic', 'diastolic'])]
    print(f"\n💓 Blood Pressure Features ({len(bp_features)}):")
    for feat in bp_features:
        print(f"    - {feat}")
    
    # Key numerical features
    key_features = ['Age', 'Cholesterol', 'BMI', 'Screen Time', 'Sleep Duration', 'Resting Heart Rate']
    for feature in key_features:
        matches = [col for col in feature_columns if feature.lower() in col.lower()]
        if matches:
            print(f"\n🔢 {feature} Features:")
            for match in matches:
                print(f"    - {match}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure 'feature_columns.pkl' is in the same directory")