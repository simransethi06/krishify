import os
import pandas as pd

# 📂 Folder path where your CSV files are stored
folder_path = r"E:\k1\krishify\krishify\app\models"

# 🔍 Find all .csv files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

if not csv_files:
    print("❌ No CSV files found in folder!")
else:
    print(f"🧩 Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print("   -", f)

    dfs = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"✅ Loaded: {file} ({len(df)} rows)")
        except Exception as e:
            print(f"⚠️ Error loading {file}: {e}")

    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(folder_path, "crop_data_merged.csv")
        merged_df.to_csv(output_path, index=False)
        print(f"\n🎉 Merged file created successfully at:\n{output_path}")
        print(f"Total combined rows: {len(merged_df)}")
    else:
        print("❌ No dataframes loaded successfully.")
