from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Cho phép kết nối từ frontend

# ===== Load model & encoder =====
model = joblib.load("model/random_forest_model.pkl")

# Nếu có label encoder (để giải mã nhãn văn bản)
encoder = None
encoder_path = "model/label_encoder.pkl"
if os.path.exists(encoder_path):
    encoder = joblib.load(encoder_path)

# ===== API Predict =====
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Lấy các thuộc tính đầu vào theo đúng thứ tự
        features = [
            data['Soil_Moisture'],
            data['Ambient_Temperature'],
            data['Soil_Temperature'],
            data['Humidity'],
            data['Light_Intensity'],
            data['Soil_pH'],
            data['Nitrogen_Level'],
            data['Phosphorus_Level'],
            data['Potassium_Level'],
            data['Chlorophyll_Content'],
            data['Electrochemical_Signal']
        ]

        # Dự đoán
        prediction = model.predict([features])[0]

        # Giải mã nếu cần
        if encoder:
            prediction = encoder.inverse_transform([prediction])[0]

        return jsonify({"prediction": str(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== Chạy server =====
if __name__ == "__main__":
    app.run(debug=True)
