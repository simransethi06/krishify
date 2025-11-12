import pandas as pd
import joblib

# -----------------------------
# Load model and encoders
# -----------------------------
model = joblib.load("fertilizer_model.pkl")
le_target = joblib.load("fertilizer_encoder.pkl")
le_dict = joblib.load("categorical_encoders.pkl")

numeric_features = ['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']
categorical_features = ['Soil Type','Crop Type']

# -----------------------------
# ICAR rule override function
# -----------------------------
def icar_rule_override(N, P, K, Ph, predicted_fertilizer):
    """
    Simple example of ICAR rules:
    - Adjust pH < 5.5: lime
    - Low Nitrogen: add Urea
    - Low Potassium: add MOP
    """
    if Ph < 5.5:
        return "Apply lime + adjust NPK accordingly"
    if N < 15:
        return "Increase N-rich fertilizer (Urea)"
    if K < 10:
        return "Add potassium-rich fertilizer (MOP)"
    return predicted_fertilizer

# -----------------------------
# Example user input
# -----------------------------
sample = pd.DataFrame({
    'Temparature': [28],
    'Humidity': [55],
    'Moisture': [40],
    'Nitrogen': [12],
    'Potassium': [0],
    'Phosphorous': [36],
    'Soil Type': ['Sandy'],
    'Crop Type': ['Maize']
})

# Encode categorical features
for col in categorical_features:
    sample[col] = le_dict[col].transform(sample[col])

# -----------------------------
# Model prediction
# -----------------------------
pred = model.predict(sample)
predicted_fertilizer = le_target.inverse_transform(pred)[0]

# -----------------------------
# Apply ICAR rules
# -----------------------------
final_fertilizer = icar_rule_override(
    N=sample['Nitrogen'][0],
    P=sample['Phosphorous'][0],
    K=sample['Potassium'][0],
    Ph=6.0,  # replace with actual soil pH from user/input
    predicted_fertilizer=predicted_fertilizer
)

print(f"🌾 Model Prediction: {predicted_fertilizer}")
print(f"🌱 ICAR Adjusted Recommendation: {final_fertilizer}")
