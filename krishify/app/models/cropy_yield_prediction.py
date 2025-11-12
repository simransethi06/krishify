import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib

print("✅ Starting Crop Yield Prediction Model...")

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
try:
    df = pd.read_csv(r"E:\k1\krishify\krishify\app\models\crop_data_merged.csv")
    print("✅ Merged Dataset Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# -------------------------------
# Step 2: Detect target column automatically
# -------------------------------
possible_yield_cols = [c for c in df.columns if any(k in c.lower() for k in ["yield", "production", "output", "productivity"])]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if possible_yield_cols:
    target_col = possible_yield_cols[0]
    print(f"🌾 Using detected target column: '{target_col}'")
elif len(numeric_cols) > 0:
    target_col = numeric_cols[-1]
    print(f"⚠️ No 'Yield' column found, using last numeric column: '{target_col}'")
else:
    print("⚠️ No usable data found — generating synthetic dataset.")
    df = pd.DataFrame({
        "Crop": np.random.choice(["Rice", "Wheat", "Maize", "Sugarcane", "Cotton"], 200),
        "Temperature": np.random.uniform(20, 35, 200),
        "Humidity": np.random.uniform(50, 90, 200),
        "Rainfall": np.random.uniform(50, 300, 200),
        "pH": np.random.uniform(5.5, 7.5, 200),
        "Nitrogen": np.random.uniform(40, 120, 200),
        "Phosphorus": np.random.uniform(20, 60, 200),
        "Potassium": np.random.uniform(20, 60, 200),
        "Yield": np.random.uniform(1.5, 4.5, 200)
    })
    target_col = "Yield"

# -------------------------------
# Step 3: Encode categorical columns
# -------------------------------
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# -------------------------------
# Step 4: Prepare features and labels
# -------------------------------
if target_col not in df.columns:
    raise ValueError(f"❌ Target column '{target_col}' not found in dataset!")

X = df.drop(columns=[target_col])
y = df[target_col]

# -------------------------------
# Step 5: Split data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Step 6: Train model
# -------------------------------
model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Step 7: Evaluate
# -------------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n🎯 Model Trained Successfully!")
print(f"✅ R² Score: {r2:.2f}")
print(f"✅ MAE: {mae:.2f}")

# -------------------------------
# Step 8: Save model
# -------------------------------
model_path = r"E:\k1\krishify\krishify\app\models\crop_yield_model.pkl"
joblib.dump(model, model_path)
print(f"\n💾 Model saved successfully at: {model_path}")

# -------------------------------
# Step 9: Example prediction
# -------------------------------
sample = X.iloc[[0]]
predicted_yield = model.predict(sample)[0]
print(f"\n🌾 Predicted Crop Yield for first record: {predicted_yield:.2f}")
