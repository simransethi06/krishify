import requests
import os

API_KEY = os.getenv("WEATHER_API_KEY")

def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    return requests.get(url).json()
