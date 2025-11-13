import pandas as pd

print("🔧 FIXING USER PROFILES")
print("="*60)

# Load user profiles
df = pd.read_csv('artifacts/processed/user_profiles_final.csv')
print(f"✅ Loaded: {df.shape}")
print(f"✅ Columns: {list(df.columns)}")

# Check for size column
size_columns = [col for col in df.columns if 'size' in col.lower()]
print(f"\n📊 Size-related columns: {size_columns}")

# Rename to 'purchased_size' if needed
if 'purchased_size' not in df.columns:
    if 'size_standardized' in df.columns:
        df['purchased_size'] = df['size_standardized']
        print("✅ Created 'purchased_size' from 'size_standardized'")
    elif 'size' in df.columns:
        df['purchased_size'] = df['size']
        print("✅ Created 'purchased_size' from 'size'")
    else:
        df['purchased_size'] = 'M'  # Default
        print("⚠️ No size column found, defaulting to 'M'")

# Ensure numeric columns are numeric
print("\n🔢 Converting numeric columns...")
numeric_cols = ['age', 'height_cm', 'weight_kg', 'bmi', 'chest_bust_cm', 
               'waist_cm', 'hip_cm', 'user_total_purchases', 'user_return_rate', 'user_avg_price']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"   ✅ {col}")

# Fill NaN values
df.fillna({
    'age': 30, 'height_cm': 165, 'weight_kg': 60, 'bmi': 22,
    'chest_bust_cm': 88, 'waist_cm': 70, 'hip_cm': 92,
    'user_total_purchases': 1, 'user_return_rate': 0.1, 'user_avg_price': 2000,
    'body_shape': 'Average', 'gender': 'Female', 'purchased_size': 'M'
}, inplace=True)

print("✅ NaN values filled")

# Save
df.to_csv('artifacts/processed/user_profiles_final.csv', index=False)
print("\n💾 Saved fixed user profiles!")

print("\n" + "="*60)
print("✅ USER PROFILES FIXED!")
print("="*60)
print(f"\nSample data:\n{df[['user_id', 'purchased_size', 'age', 'height_cm']].head()}")