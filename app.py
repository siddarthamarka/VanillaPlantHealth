from flask import Flask, render_template, request, url_for
import joblib
import os
from utils.preprocessing import extract_color_features  # Make sure this file exists

app = Flask(__name__)
MODEL_PATH = "models/plant_model.pkl"
model = joblib.load(MODEL_PATH)

# Ensure 'static/uploads' exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", prediction="No file uploaded", img_path=None)

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", prediction="No file selected", img_path=None)

        # Save uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        img_path = filepath  # For rendering in HTML

        # Extract features & predict
        features = extract_color_features(filepath)
        prediction = model.predict([features])[0]  # e.g., "Good/OneRooting"

    return render_template("index.html", prediction=prediction, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)