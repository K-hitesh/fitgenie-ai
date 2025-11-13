import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("🔧 Rebuilding Preprocessor for Current Scikit-Learn Version")
print("="*60)

# Load training data
print("📊 Loading training data...")
X_train = pd.read_csv('artifacts/processed/X_train_final.csv')
print(f"   ✅ Loaded: {X_train.shape}")

# Define feature types
numeric_features = ['age', 'height_cm', 'weight_kg', 'bmi', 
                   'chest_bust_cm', 'waist_cm', 'hip_cm',
                   'final_price', 'discount_pct', 'base_price']

categorical_features = ['gender', 'body_shape', 'brand', 'category', 
                       'material', 'color', 'subcategory']

# Filter to only existing columns
numeric_features = [f for f in numeric_features if f in X_train.columns]
categorical_features = [f for f in categorical_features if f in X_train.columns]

print(f"   Numeric features: {len(numeric_features)}")
print(f"   Categorical features: {len(categorical_features)}")

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
)

# Fit the preprocessor
print("⚙️ Fitting preprocessor...")
preprocessor.fit(X_train)
print("   ✅ Fitted!")

# Save the new preprocessor
print("💾 Saving preprocessor...")
joblib.dump(preprocessor, 'artifacts/models/preprocessor_final.pkl')
print("   ✅ Saved to: artifacts/models/preprocessor_final.pkl")

# Test the preprocessor
print("\n🧪 Testing preprocessor...")
sample = X_train.iloc[0:1]
transformed = preprocessor.transform(sample)
print(f"   ✅ Test passed! Output shape: {transformed.shape}")

print("\n" + "="*60)
print("✅ PREPROCESSOR REBUILT SUCCESSFULLY!")
print("="*60)
print("\nNow restart your Flask app:")
print("  python app.py")