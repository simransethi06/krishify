import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib  # for saving/loading the model

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv(r"E:\krishify\krishify\krishify\app\DATASET - Sheet1.csv")

# -------------------------------
# Step 2: Convert TEMPERATURE ranges to numeric
# -------------------------------
df['TEMPERATURE'] = df['TEMPERATURE'].apply(lambda x: (int(x.split('-')[0]) + int(x.split('-')[1]))/2)

# -------------------------------
# Step 3: Encode categorical variables
# -------------------------------
categorical_cols = ['CROP TYPE', 'SOIL TYPE', 'REGION', 'WEATHER CONDITION']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# -------------------------------
# Step 4: Split features and target
# -------------------------------
X = df.drop('WATER REQUIREMENT', axis=1)
y = df['WATER REQUIREMENT']

# -------------------------------
# Step 5: Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Step 6: Train model
# -------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Step 7: Evaluate
# -------------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")

# -------------------------------
# Step 8: Save trained model to .pkl
# -------------------------------
joblib.dump(model, r"E:\krishify\krishify\krishify\app\models\water_model.pkl")
print("Trained model saved as water_model.pkl")

# -------------------------------
# Step 9: Load model and predict new input safely
# -------------------------------
# Load model
loaded_model = joblib.load(r"E:\krishify\krishify\krishify\app\models\water_model.pkl")

# New input dictionary (example)
new_input = {
    'TEMPERATURE': 25,
    'CROP TYPE_BANANA': 1,
    'SOIL TYPE_DRY': 1,
    'REGION_DESERT': 1,
    'WEATHER CONDITION_NORMAL': 1
}

# Make sure all columns seen by the model exist in new input
new_data = pd.DataFrame(columns=X.columns)
for col in new_data.columns:
    new_data[col] = [new_input.get(col, 0)]  # fill missing columns with 0

new_pred = loaded_model.predict(new_data)
print(f"Predicted water requirement for new input: {new_pred[0]:.2f}")
