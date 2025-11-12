from flask import Blueprint, request, jsonify
from app.services.db_service import users_collection
from app.utils.jwt_handler import generate_token
import bcrypt

auth_bp = Blueprint("auth", __name__)

@auth_bp.post("/register")
def register():
    data = request.json
    hashed = bcrypt.hashpw(data["password"].encode(), bcrypt.gensalt())

    users_collection.insert_one({
        "name": data["name"],
        "email": data["email"],
        "password": hashed
    })

    return jsonify({"message": "User created"}), 201


@auth_bp.post("/login")
def login():
    data = request.json
    user = users_collection.find_one({"email": data["email"]})

    if not user or not bcrypt.checkpw(data["password"].encode(), user["password"]):
        return jsonify({"error": "Invalid credentials"}), 401

    token = generate_token(str(user["_id"]))
    return jsonify({"token": token})
