import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from google.colab import drive
drive.mount('/content/drive')

# ----------------- Step 2: Set paths -----------------
PROJECT_FOLDER = "/content/drive/MyDrive/Harmony_Project"
FEATURES_FILE = os.path.join(PROJECT_FOLDER, "Output/features.xlsx")
MODEL_PATH = os.path.join(PROJECT_FOLDER, "Output/plant_model.pkl")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


FEATURES_FILE = "/content/drive/MyDrive/Harmony_Project/Output/features.xlsx"
MODEL_PATH = "Output/plant_model.pkl"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Load features
df = pd.read_excel(FEATURES_FILE)
X = df.drop("label", axis=1).values
y = df["label"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train RandomForest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")