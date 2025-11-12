from flask import Blueprint, request, jsonify
from app.services.weather_service import get_weather

weather_bp = Blueprint("weather", __name__)

@weather_bp.get("/current")
def current_weather():
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    return jsonify(get_weather(lat, lon))
