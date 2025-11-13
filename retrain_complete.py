import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

print("🚀 Complete Model Retraining Started...")

# Load data
X_train = pd.read_csv('artifacts/processed/X_train_final.csv')
X_test = pd.read_csv('artifacts/processed/X_test_final.csv')
y_train = pd.read_csv('artifacts/processed/y_train_final.csv').iloc[:, 0]
y_test = pd.read_csv('artifacts/processed/y_test_final.csv').iloc[:, 0]

print(f"✅ Data loaded: X_train shape = {X_train.shape}")
print(f"   Training features: {X_train.shape[1]}")

# Get feature types
numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(include='object').columns.tolist()

print(f"   Numerical features: {len(numerical_features)}")
print(f"   Categorical features: {len(categorical_features)}")

# Create preprocessor
numerical_transformer = Pipeline(steps=[
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

print("\n🔧 Fitting preprocessor...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print(f"✅ Preprocessed shape: {X_train_processed.shape}")

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Save preprocessor and label encoder
joblib.dump(preprocessor, 'artifacts/models/preprocessor_final.pkl')
joblib.dump(label_encoder, 'artifacts/models/label_encoder_final.pkl')
print("✅ Preprocessor and Label Encoder saved")

# Train models
print("\n🤖 Training CatBoost...")
catboost_model = CatBoostClassifier(
    iterations=700, learning_rate=0.1, depth=6,
    eval_metric='TotalF1', random_seed=42, verbose=0
)
catboost_model.fit(X_train_processed, y_train_encoded)
joblib.dump(catboost_model, 'artifacts/models/catboost_model_final.pkl')
print("✅ CatBoost saved")

print("🤖 Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=500, learning_rate=0.1, max_depth=6,
    random_state=42, eval_metric='mlogloss', use_label_encoder=False
)
xgb_model.fit(X_train_processed, y_train_encoded)
joblib.dump(xgb_model, 'artifacts/models/xgboost_model_final.pkl')
print("✅ XGBoost saved")

print("🤖 Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
)
rf_model.fit(X_train_processed, y_train_encoded)
joblib.dump(rf_model, 'artifacts/models/rf_model_final.pkl')
print("✅ Random Forest saved")

print("\n🧪 Training Stacking Meta Model...")
# Get predictions from base models
X_meta_train = np.hstack([
    catboost_model.predict_proba(X_train_processed),
    xgb_model.predict_proba(X_train_processed),
    rf_model.predict_proba(X_train_processed)
])

stacking_meta = LogisticRegression(
    solver='lbfgs', multi_class='multinomial', max_iter=500, random_state=42
)
stacking_meta.fit(X_meta_train, y_train_encoded)
joblib.dump(stacking_meta, 'artifacts/models/stacking_meta_model_final.pkl')
print("✅ Stacking Meta Model saved")

# Test prediction
print("\n🧪 Testing prediction...")
X_meta_test = np.hstack([
    catboost_model.predict_proba(X_test_processed),
    xgb_model.predict_proba(X_test_processed),
    rf_model.predict_proba(X_test_processed)
])
y_pred = stacking_meta.predict(X_meta_test)
accuracy = (y_pred == y_test_encoded).mean()
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

print("\n🎉 RETRAINING COMPLETE!")
print("✅ All models saved with correct feature dimensions")
print("✅ You can now run: python app.py")