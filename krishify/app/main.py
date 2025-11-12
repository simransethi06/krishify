from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Register routes
    from app.routes.auth_routes import auth_bp
    from app.routes.ml_routes import ml_bp
    from app.routes.weather_routes import weather_bp

    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(ml_bp, url_prefix="/ml")
    app.register_blueprint(weather_bp, url_prefix="/weather")

    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
