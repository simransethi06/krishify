from flask import Blueprint, request, jsonify
from app.services.ml_service import (
    predict_yield,
    recommend_crop,
    predict_fertilizer,
    predict_water
)

ml_bp = Blueprint("ml", __name__)

@ml_bp.post("/predict-yield")
def predict_crop_yield():
    data = request.json["features"]
    result = predict_yield(data)
    return jsonify({"predicted_yield": result})

@ml_bp.post("/recommend-crop")
def recommend():
    data = request.json["features"]
    crop = recommend_crop(data)
    return jsonify({"recommended_crop": crop})

@ml_bp.post("/fertilizer")
def fertilizer_recommendation():
    data = request.json["features"]
    fertilizer = predict_fertilizer(data)
    return jsonify({"recommended_fertilizer": fertilizer})

@ml_bp.post("/water-requirement")
def water_requirement():
    data = request.json["features"]
    water_req = predict_water(data)
    return jsonify({"predicted_water_requirement": water_req})
