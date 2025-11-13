import pandas as pd
import os

print("🔍 CHECKING FOR EXCEL FILES")
print("="*60)

# Find Excel files
excel_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith(('.xlsx', '.xls')):
            excel_files.append(os.path.join(root, file))

print(f"📂 Found Excel files: {excel_files}\n")

if not excel_files:
    print("❌ NO EXCEL FILES FOUND!")
    print("\n📌 Please place your Excel file in one of these locations:")
    print("  - data/fashion_dataset.xlsx")
    print("  - data/user_data.xlsx")
    print("  - users.xlsx")
else:
    for excel_file in excel_files:
        print(f"📊 Analyzing: {excel_file}")
        print("-"*60)
        
        try:
            excel_data = pd.read_excel(excel_file, sheet_name=None)
            
            print(f"✅ Sheets found: {list(excel_data.keys())}\n")
            
            for sheet_name, df in excel_data.items():
                print(f"📋 Sheet: '{sheet_name}'")
                print(f"   Rows: {len(df)}")
                print(f"   Columns: {list(df.columns)}")
                print(f"\n   First 3 rows:")
                print(df.head(3))
                print("\n")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")

print("="*60)
print("\n✅ Check complete! Now:")
print("1. Make sure your Excel file is in the right location")
print("2. Verify column names match (user_id, height_cm, weight_kg, etc.)")
print("3. Run: python app.py")