"""
CONTINUOUS LEARNING PIPELINE
Retrain models based on user feedback
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("="*70)
print("🔄 CONTINUOUS LEARNING PIPELINE - Model Retraining")
print("="*70)

# Check if retraining is required
retrain_flag = 'artifacts/feedback/retrain_required.flag'
learning_data_file = 'artifacts/feedback/learning_data.csv'

if not os.path.exists(retrain_flag):
    print("ℹ️  No retraining required yet.")
    print(f"   Waiting for feedback accumulation...")
    exit(0)

if not os.path.exists(learning_data_file):
    print("❌ No learning data found!")
    exit(1)

print("\n✅ Retraining triggered!")
print(f"📁 Loading learning data from: {learning_data_file}")

# ========== LOAD FEEDBACK DATA ==========
feedback_data = pd.read_csv(learning_data_file)
print(f"✅ Loaded {len(feedback_data)} feedback samples")

# ========== LOAD ORIGINAL TRAINING DATA ==========
print("\n📊 Loading original training data...")
X_train_orig = pd.read_csv('artifacts/processed/X_train_final.csv')
y_train_orig = pd.read_csv('artifacts/processed/y_train_final.csv')
X_test = pd.read_csv('artifacts/processed/X_test_final.csv')
y_test = pd.read_csv('artifacts/processed/y_test_final.csv')

print(f"✅ Original training: {len(X_train_orig)} samples")
print(f"✅ New feedback: {len(feedback_data)} samples")

# ========== PREPARE FEEDBACK DATA ==========
print("\n🔧 Preparing feedback data for training...")

# Map feedback to fit categories
feedback_mapping = {
    'perfect': 'Perfect Fit',
    'good': 'Perfect Fit',
    'acceptable': 'Slightly Small',
    'too_small': 'Too Small',
    'too_large': 'Too Large'
}

feedback_data['fit_label'] = feedback_data['fit_feedback'].map(feedback_mapping)
feedback_data = feedback_data.dropna(subset=['fit_label'])

print(f"✅ Processed {len(feedback_data)} valid feedback samples")

# Select features matching training data
feature_cols = X_train_orig.columns.tolist()
feedback_features = []

for col in feature_cols:
    if col in feedback_data.columns:
        feedback_features.append(col)
    elif col == 'final_price':
        feedback_data['final_price'] = 2000  # Default price
    elif col == 'discount_pct':
        feedback_data['discount_pct'] = 10
    elif col == 'base_price':
        feedback_data['base_price'] = 2500
    elif col == 'material':
        feedback_data['material'] = 'Cotton'
    elif col == 'color':
        feedback_data['color'] = 'Black'
    elif col == 'subcategory':
        feedback_data['subcategory'] = 'General'

# Create new training samples from feedback
X_feedback = feedback_data[feature_cols].copy()
y_feedback = feedback_data['fit_label'].copy()

print(f"✅ Feedback features prepared: {X_feedback.shape}")

# ========== COMBINE WITH ORIGINAL DATA ==========
print("\n🔗 Combining original and feedback data...")

# Sample original data to balance (use 80% original + 20% new)
sample_size = int(len(X_train_orig) * 0.8)
X_train_sampled = X_train_orig.sample(n=sample_size, random_state=42)
y_train_sampled = y_train_orig.loc[X_train_sampled.index]

# Combine
X_train_new = pd.concat([X_train_sampled, X_feedback], ignore_index=True)
y_train_new = pd.concat([y_train_sampled, y_feedback], ignore_index=True)

print(f"✅ Combined training data: {len(X_train_new)} samples")
print(f"   - Original samples: {len(X_train_sampled)}")
print(f"   - Feedback samples: {len(X_feedback)}")

# ========== LOAD PREPROCESSOR AND LABEL ENCODER ==========
print("\n⚙️ Loading preprocessor and label encoder...")
preprocessor = joblib.load('artifacts/models/preprocessor_final.pkl')
label_encoder = joblib.load('artifacts/models/label_encoder_final.pkl')

print("✅ Preprocessor loaded")

# ========== PREPROCESS DATA ==========
print("\n🔧 Preprocessing data...")
X_train_processed = preprocessor.transform(X_train_new)
X_test_processed = preprocessor.transform(X_test)

print(f"✅ Training data processed: {X_train_processed.shape}")

# Encode labels
y_train_encoded = label_encoder.transform(y_train_new.values.ravel())
y_test_encoded = label_encoder.transform(y_test.values.ravel())

# ========== RETRAIN MODELS ==========
print("\n" + "="*70)
print("🤖 RETRAINING MODELS WITH NEW DATA")
print("="*70)

# Backup old models
backup_dir = f'artifacts/models/backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(backup_dir, exist_ok=True)

print(f"\n💾 Backing up old models to: {backup_dir}")
for model_file in ['catboost_model_final.pkl', 'xgboost_model_final.pkl', 
                   'rf_model_final.pkl', 'stacking_meta_model_final.pkl']:
    src = f'artifacts/models/{model_file}'
    dst = f'{backup_dir}/{model_file}'
    if os.path.exists(src):
        joblib.dump(joblib.load(src), dst)
        print(f"   ✅ Backed up: {model_file}")

# Train CatBoost
print("\n🤖 Training CatBoost...")
catboost_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    verbose=50,
    random_seed=42
)
catboost_model.fit(X_train_processed, y_train_encoded)
print("✅ CatBoost trained!")

# Train XGBoost
print("\n🤖 Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train_processed, y_train_encoded)
print("✅ XGBoost trained!")

# Train Random Forest
print("\n🤖 Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_processed, y_train_encoded)
print("✅ Random Forest trained!")

# Train Stacking Meta Model
print("\n🤖 Training Stacking Meta Model...")
catboost_pred_proba = catboost_model.predict_proba(X_train_processed)
xgb_pred_proba = xgb_model.predict_proba(X_train_processed)
rf_pred_proba = rf_model.predict_proba(X_train_processed)

X_meta_train = np.hstack([catboost_pred_proba, xgb_pred_proba, rf_pred_proba])

meta_model = LogisticRegression(max_iter=500, random_state=42)
meta_model.fit(X_meta_train, y_train_encoded)
print("✅ Stacking Meta model trained!")

# ========== EVALUATE NEW MODELS ==========
print("\n" + "="*70)
print("📊 EVALUATING RETRAINED MODELS")
print("="*70)

# Test predictions
catboost_test_proba = catboost_model.predict_proba(X_test_processed)
xgb_test_proba = xgb_model.predict_proba(X_test_processed)
rf_test_proba = rf_model.predict_proba(X_test_processed)

X_meta_test = np.hstack([catboost_test_proba, xgb_test_proba, rf_test_proba])
meta_predictions = meta_model.predict(X_meta_test)

# Calculate accuracies
stacking_accuracy = accuracy_score(y_test_encoded, meta_predictions)
catboost_accuracy = accuracy_score(y_test_encoded, catboost_model.predict(X_test_processed))
xgb_accuracy = accuracy_score(y_test_encoded, xgb_model.predict(X_test_processed))
rf_accuracy = accuracy_score(y_test_encoded, rf_model.predict(X_test_processed))

print(f"\n📊 Model Performance:")
print(f"   • CatBoost:    {catboost_accuracy*100:.2f}%")
print(f"   • XGBoost:     {xgb_accuracy*100:.2f}%")
print(f"   • Random Forest: {rf_accuracy*100:.2f}%")
print(f"   • Stacking:    {stacking_accuracy*100:.2f}% ⭐")

# ========== SAVE RETRAINED MODELS ==========
print("\n💾 Saving retrained models...")

joblib.dump(catboost_model, 'artifacts/models/catboost_model_final.pkl')
print("   ✅ CatBoost saved")

joblib.dump(xgb_model, 'artifacts/models/xgboost_model_final.pkl')
print("   ✅ XGBoost saved")

joblib.dump(rf_model, 'artifacts/models/rf_model_final.pkl')
print("   ✅ Random Forest saved")

joblib.dump(meta_model, 'artifacts/models/stacking_meta_model_final.pkl')
print("   ✅ Stacking Meta saved")

# ========== UPDATE METRICS LOG ==========
metrics_log_file = 'artifacts/feedback/retrain_history.csv'

metrics_entry = {
    'timestamp': datetime.now().isoformat(),
    'feedback_samples': len(feedback_data),
    'total_training_samples': len(X_train_new),
    'catboost_accuracy': catboost_accuracy,
    'xgb_accuracy': xgb_accuracy,
    'rf_accuracy': rf_accuracy,
    'stacking_accuracy': stacking_accuracy,
    'backup_dir': backup_dir
}

if os.path.exists(metrics_log_file):
    metrics_log = pd.read_csv(metrics_log_file)
    metrics_log = pd.concat([metrics_log, pd.DataFrame([metrics_entry])], ignore_index=True)
else:
    metrics_log = pd.DataFrame([metrics_entry])

metrics_log.to_csv(metrics_log_file, index=False)
print(f"\n✅ Metrics logged to: {metrics_log_file}")

# ========== REMOVE RETRAIN FLAG ==========
os.remove(retrain_flag)
print("✅ Retrain flag removed")

# ========== SUMMARY ==========
print("\n" + "="*70)
print("✅ RETRAINING COMPLETE!")
print("="*70)
print(f"\n📊 Summary:")
print(f"   • New feedback samples learned: {len(feedback_data)}")
print(f"   • Total training samples: {len(X_train_new)}")
print(f"   • Stacking accuracy: {stacking_accuracy*100:.2f}%")
print(f"   • Old models backed up to: {backup_dir}")
print(f"\n🚀 Retrained models are now active!")
print("="*70)