import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

print("🔧 FIXING PREPROCESSOR - Handling Categorical Features")
print("="*60)

# Load training data
X_train = pd.read_csv('artifacts/processed/X_train_final.csv')
print(f"✅ Loaded training data: {X_train.shape}")
print(f"✅ Columns: {list(X_train.columns)}")

# Identify numeric and categorical columns
numeric_features = []
categorical_features = []

for col in X_train.columns:
    if X_train[col].dtype in ['int64', 'float64']:
        numeric_features.append(col)
    else:
        categorical_features.append(col)

print(f"\n📊 Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"📊 Categorical features ({len(categorical_features)}): {categorical_features}")

# Create FIXED preprocessor with proper OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(
            drop='first',
            sparse_output=False,
            handle_unknown='ignore'
        ), categorical_features)
    ],
    remainder='drop'  # Drop any remaining columns
)

# Fit preprocessor
print("\n⚙️ Fitting preprocessor...")
preprocessor.fit(X_train)
print("✅ Fitted successfully!")

# Test transformation
print("\n🧪 Testing transformation...")
sample = X_train.iloc[0:1]
transformed = preprocessor.transform(sample)
print(f"✅ Input shape: {sample.shape}")
print(f"✅ Output shape: {transformed.shape}")
print(f"✅ Output type: {type(transformed)}")
print(f"✅ All numeric: {np.issubdtype(transformed.dtype, np.number)}")

# Save
print("\n💾 Saving preprocessor...")
joblib.dump(preprocessor, 'artifacts/models/preprocessor_final.pkl')
print("✅ Saved!")

print("\n" + "="*60)
print("✅ PREPROCESSOR FIXED!")
print("="*60)