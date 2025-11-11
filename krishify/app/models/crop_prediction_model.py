import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ✅ Load dataset
file_path = r"E:\krishify\krishify\krishify\app\Crop Recommendation Dataset.xlsx"
df = pd.read_excel(file_path)
print("✅ Dataset Loaded Successfully!")

# ✅ Prepare data
X = df[['Temperature', 'Humidity', 'pH', 'Rainfall']]
y = df['Label']

# ✅ Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ✅ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully with accuracy: {accuracy*100:.2f}%")

# ✅ Save model and label encoder
joblib.dump(model, "crop_recommendation_model.pkl")
joblib.dump(le, "label_encoder.pkl")

# ✅ Example Prediction
sample = pd.DataFrame({
    'Temperature': [25.0],
    'Humidity': [80.0],
    'pH': [6.5],
    'Rainfall': [200.0]
})

prediction = model.predict(sample)
predicted_label = le.inverse_transform(prediction)
print(f"🌾 Recommended Crop: {predicted_label[0]}")
