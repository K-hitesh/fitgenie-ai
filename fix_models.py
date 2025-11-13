import joblib
import warnings
warnings.filterwarnings('ignore')

print("🔧 Attempting to load and re-save models with current sklearn version...")

try:
    # Load preprocessor
    preprocessor = joblib.load('artifacts/models/preprocessor_final.pkl')
    # Re-save with current version
    joblib.dump(preprocessor, 'artifacts/models/preprocessor_final.pkl')
    print("✅ Preprocessor fixed!")
except Exception as e:
    print(f"❌ Preprocessor error: {e}")

try:
    # Load RF model
    rf = joblib.load('artifacts/models/rf_model_final.pkl')
    # Re-save
    joblib.dump(rf, 'artifacts/models/rf_model_final.pkl')
    print("✅ Random Forest fixed!")
except Exception as e:
    print(f"❌ RF error: {e}")

try:
    # Load stacking meta
    meta = joblib.load('artifacts/models/stacking_meta_model_final.pkl')
    # Re-save
    joblib.dump(meta, 'artifacts/models/stacking_meta_model_final.pkl')
    print("✅ Stacking Meta fixed!")
except Exception as e:
    print(f"❌ Meta error: {e}")

print("\n🎉 Model fixing complete! Try running app.py again.")