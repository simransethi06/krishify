import joblib
import pandas as pd
import numpy as np

# Load ML models
yield_model = joblib.load("app/ml_models/crop_yield.pkl")
recommend_model = joblib.load("app/ml_models/crop_recommendation_model.pkl")
fertilizer_model = joblib.load("app/ml_models/fertilizer_model.pkl")
water_model = joblib.load("app/ml_models/water_model.pkl")

# Load encoders (if available)
try:
    le_crop = joblib.load("app/ml_models/label_encoder.pkl")
except:
    le_crop = None

try:
    fertilizer_encoder = joblib.load("app/ml_models/fertilizer_encoder.pkl")
    categorical_encoders = joblib.load("app/ml_models/categorical_encoders.pkl")
except:
    fertilizer_encoder = None
    categorical_encoders = None


# -------------------------------
# Crop Yield Prediction
# -------------------------------
def predict_yield(features):
    """
    Input: list of numerical features
    Output: float yield value
    """
    arr = np.array([features])
    prediction = yield_model.predict(arr)
    return round(float(prediction[0]), 2)


# -------------------------------
# Crop Recommendation
# -------------------------------
def recommend_crop(features):
    df = pd.DataFrame([features])
    result = recommend_model.predict(df)
    if le_crop:
        result = le_crop.inverse_transform(result)
    return result[0]


# -------------------------------
# Fertilizer Recommendation
# -------------------------------
def predict_fertilizer(features):
    df = pd.DataFrame([features])
    for col in ["Soil Type", "Crop Type"]:
        if col in categorical_encoders:
            df[col] = categorical_encoders[col].transform(df[col])

    pred = fertilizer_model.predict(df)
    fertilizer = fertilizer_encoder.inverse_transform(pred)[0]
    return fertilizer


# -------------------------------
# Water Requirement Prediction
# -------------------------------
def predict_water(features):
    df = pd.DataFrame([features])
    prediction = water_model.predict(df)
    return round(float(prediction[0]), 2)
