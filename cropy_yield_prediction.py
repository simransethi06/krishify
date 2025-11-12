import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# -----------------------------
# ✅ Simulated dataset
# -----------------------------
df = pd.DataFrame({
    "crop": np.random.choice(["Wheat", "Rice", "Maize", "Sugarcane"], 200),
    "rainfall": np.random.uniform(200, 1000, 200),
    "temperature": np.random.uniform(15, 40, 200),
    "humidity": np.random.uniform(30, 90, 200),
    "N": np.random.uniform(10, 100, 200),
    "P": np.random.uniform(5, 60, 200),
    "K": np.random.uniform(10, 80, 200),
    "pH": np.random.uniform(5.5, 8.5, 200),
    "area": np.random.uniform(0.5, 10, 200),
    "region": np.random.choice(["Punjab", "UP", "MP", "Bihar"], 200),
    "yield": np.random.uniform(1.5, 6.0, 200)
})

# -----------------------------
# Encode categorical columns
# -----------------------------
le_crop = LabelEncoder()
le_region = LabelEncoder()

df["crop"] = le_crop.fit_transform(df["crop"])
df["region"] = le_region.fit_transform(df["region"])

# -----------------------------
# Split features and target
# -----------------------------
X = df.drop("yield", axis=1)
y = df["yield"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train RandomForestRegressor
# -----------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Evaluate model
# -----------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Model trained successfully!")
print(f"📉 MAE: {mae:.3f}")
print(f"📈 R² Score: {r2:.3f}")

# -----------------------------
# Save model and encoders
# -----------------------------
joblib.dump(model, "crop_yield_model.pkl")
joblib.dump(le_crop, "crop_encoder.pkl")
joblib.dump(le_region, "region_encoder.pkl")
print("✅ Model and encoders saved as .pkl files")

# -----------------------------
# Example prediction using saved model
# -----------------------------
# Load model and encoders (simulating future use)
model = joblib.load("crop_yield_model.pkl")
le_crop = joblib.load("crop_encoder.pkl")
le_region = joblib.load("region_encoder.pkl")

# Sample input (can be from geospatial API)
sample = pd.DataFrame({
    "crop": [le_crop.transform(["Rice"])[0]],
    "rainfall": [800],
    "temperature": [20],
    "humidity": [60],
    "N": [40],
    "P": [20],
    "K": [20],
    "pH": [7.0],
    "area": [5],
    "region": [le_region.transform(["MP"])[0]]
})

predicted_yield = model.predict(sample)[0]
print(f"🌾 Predicted Yield for sample input: {predicted_yield:.2f} tons/ha")
