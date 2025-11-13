import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("🔄 RETRAINING MODELS TO MATCH PREPROCESSOR")
print("="*60)

# ========== LOAD DATA ==========
print("📊 Loading data...")
X_train = pd.read_csv('artifacts/processed/X_train_final.csv')
y_train = pd.read_csv('artifacts/processed/y_train_final.csv')
X_test = pd.read_csv('artifacts/processed/X_test_final.csv')
y_test = pd.read_csv('artifacts/processed/y_test_final.csv')

print(f"✅ X_train: {X_train.shape}")
print(f"✅ y_train: {y_train.shape}")
print(f"✅ X_test: {X_test.shape}")
print(f"✅ y_test: {y_test.shape}")

# ========== LOAD PREPROCESSOR ==========
print("\⚙️ Loading preprocessor...")
preprocessor = joblib.load('artifacts/models/preprocessor_final.pkl')
label_encoder = joblib.load('artifacts/models/label_encoder_final.pkl')

# ========== PREPROCESS DATA ==========
print("🔧 Preprocessing data...")
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"✅ X_train_processed: {X_train_processed.shape}")
print(f"✅ X_test_processed: {X_test_processed.shape}")

# Encode labels
y_train_encoded = label_encoder.transform(y_train.values.ravel())
y_test_encoded = label_encoder.transform(y_test.values.ravel())

print(f"✅ y_train_encoded: {y_train_encoded.shape}")
print(f"✅ Classes: {label_encoder.classes_}")

# ========== TRAIN CATBOOST ==========
print("\n🤖 Training CatBoost...")
catboost_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    verbose=100,
    random_seed=42
)
catboost_model.fit(X_train_processed, y_train_encoded)
print("✅ CatBoost trained!")

# ========== TRAIN XGBOOST ==========
print("\n🤖 Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train_processed, y_train_encoded)
print("✅ XGBoost trained!")

# ========== TRAIN RANDOM FOREST ==========
print("\n🤖 Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_processed, y_train_encoded)
print("✅ Random Forest trained!")

# ========== CREATE STACKING ENSEMBLE ==========
print("\n🤖 Training Stacking Ensemble...")

# Get predictions from base models
print("   Getting base model predictions...")
catboost_pred_proba = catboost_model.predict_proba(X_train_processed)
xgb_pred_proba = xgb_model.predict_proba(X_train_processed)
rf_pred_proba = rf_model.predict_proba(X_train_processed)

# Stack predictions
X_meta_train = np.hstack([catboost_pred_proba, xgb_pred_proba, rf_pred_proba])
print(f"   Meta features shape: {X_meta_train.shape}")

# Train meta model
meta_model = LogisticRegression(max_iter=1000, random_state=42)
meta_model.fit(X_meta_train, y_train_encoded)
print("✅ Stacking Meta model trained!")

# ========== EVALUATE ==========
print("\n📊 Evaluating models...")

# Test predictions
catboost_test_proba = catboost_model.predict_proba(X_test_processed)
xgb_test_proba = xgb_model.predict_proba(X_test_processed)
rf_test_proba = rf_model.predict_proba(X_test_processed)

X_meta_test = np.hstack([catboost_test_proba, xgb_test_proba, rf_test_proba])
meta_predictions = meta_model.predict(X_meta_test)

accuracy = accuracy_score(y_test_encoded, meta_predictions)
print(f"✅ Stacking Model Accuracy: {accuracy*100:.2f}%")

# Individual model accuracies
cat_acc = accuracy_score(y_test_encoded, catboost_model.predict(X_test_processed))
xgb_acc = accuracy_score(y_test_encoded, xgb_model.predict(X_test_processed))
rf_acc = accuracy_score(y_test_encoded, rf_model.predict(X_test_processed))

print(f"   CatBoost: {cat_acc*100:.2f}%")
print(f"   XGBoost: {xgb_acc*100:.2f}%")
print(f"   Random Forest: {rf_acc*100:.2f}%")

# ========== SAVE MODELS ==========
print("\n💾 Saving models...")

joblib.dump(catboost_model, 'artifacts/models/catboost_model_final.pkl')
print("   ✅ CatBoost saved")

joblib.dump(xgb_model, 'artifacts/models/xgboost_model_final.pkl')
print("   ✅ XGBoost saved")

joblib.dump(rf_model, 'artifacts/models/rf_model_final.pkl')
print("   ✅ Random Forest saved")

joblib.dump(meta_model, 'artifacts/models/stacking_meta_model_final.pkl')
print("   ✅ Stacking Meta saved")

print("\n" + "="*60)
print("✅ ALL MODELS RETRAINED AND SAVED!")
print("="*60)
print("\n🚀 Now restart your Flask app:")
print("   python app.py")
print("\n🧪 Then test at: http://localhost:5000")