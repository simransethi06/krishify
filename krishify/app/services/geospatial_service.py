from app.services.db_service import crops_collection

def find_crops_near_location(lat, lon, distance_km):
    return crops_collection.find({
        "location": {
            "$near": {
                "$geometry": {"type": "Point", "coordinates": [lon, lat]},
                "$maxDistance": distance_km * 1000
            }
        }
    })
