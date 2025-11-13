import pandas as pd

print("🔍 CHECKING RAW DATASET")
print("="*60)

df = pd.read_csv('artifacts/raw/fashion_size_fit_dataset.csv')

print(f"✅ Rows: {len(df)}")
print(f"✅ Columns: {list(df.columns)}")
print(f"\n📋 First 5 rows:\n{df.head()}")
print(f"\n📊 Data types:\n{df.dtypes}")
print(f"\n🔍 Sample user_id values: {df.iloc[:5, 0].tolist()}")

print("\n" + "="*60)