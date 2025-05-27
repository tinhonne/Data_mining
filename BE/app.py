from flask import Flask, request, jsonify
import os
import pandas as pd
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_path = "model/river_random_forest.pkl"
label_map_path = "model/label_map.pkl"

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    model = None

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model chưa được train."}), 500
        data = request.get_json()
        features = {
            'Soil_Moisture': data['Soil_Moisture'],
            'Ambient_Temperature': data['Ambient_Temperature'],
            'Soil_Temperature': data['Soil_Temperature'],
            'Humidity': data['Humidity'],
            'Light_Intensity': data['Light_Intensity'],
            'Soil_pH': data['Soil_pH'],
            'Nitrogen_Level': data['Nitrogen_Level'],
            'Phosphorus_Level': data['Phosphorus_Level'],
            'Potassium_Level': data['Potassium_Level'],
            'Chlorophyll_Content': data['Chlorophyll_Content'],
            'Electrochemical_Signal': data['Electrochemical_Signal']
        }
        prediction = model.predict_one(features)
        if os.path.exists(label_map_path):
            with open(label_map_path, "rb") as f:
                label_map = pickle.load(f)
            prediction = label_map.get(prediction, prediction)
        return jsonify({"prediction": str(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    temp_path = 'uploaded.csv'
    file.save(temp_path)
    # Nối dữ liệu mới vào dữ liệu cũ (nếu có)
    if os.path.exists('your_data.csv'):
        df_old = pd.read_csv('your_data.csv')
        df_new = pd.read_csv(temp_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        # Nếu muốn loại bỏ dòng trùng lặp hoàn toàn:
        df_all = df_all.drop_duplicates()
        df_all.to_csv('your_data.csv', index=False)
    else:
        os.rename(temp_path, 'your_data.csv')
    if os.path.exists(temp_path):
        os.remove(temp_path)
    os.system('python train_model_from_csv.py')
    return jsonify({'status': 'Đã cập nhật và train model với dữ liệu mới!'})

@app.route('/model_status', methods=['GET'])
def model_status():
    last_row_path = "model/last_trained_row.txt"
    if os.path.exists(last_row_path):
        with open(last_row_path, "r") as f:
            last_trained_row = int(f.read().strip())
    else:
        last_trained_row = 0
    if os.path.exists(label_map_path):
        with open(label_map_path, "rb") as f:
            label_map = pickle.load(f)
        labels = list(label_map.values())
    else:
        labels = []
    return jsonify({
        'so_mau_train': last_trained_row,
        'cac_nhan': labels
    })

@app.route('/reset_train', methods=['POST'])
def reset_train():
    last_row_path = "model/last_trained_row.txt"
    if os.path.exists(last_row_path):
        os.remove(last_row_path)
    os.system('python train_model_from_csv.py')
    return jsonify({'status': 'Đã train lại toàn bộ dữ liệu!'})

if __name__ == "__main__":
    app.run(debug=True)
