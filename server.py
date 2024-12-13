from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from typing import List, Any
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import joblib
import os
import tensorflow as tf
from material_calculator import get_materials_by_year_and_area

app = Flask(__name__)
CORS(app)

MODEL_SAVE_DIR = "./models"
SCALER_SAVE_DIR = "./scalers"
TARGET_SIZE = (256, 256)

def predict_and_denormalize(model_name, image, scaler):
    model_path = os.path.join(MODEL_SAVE_DIR, model_name)
    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(image)
    print(f"Prediction: {prediction}")
    denormalized_prediction = scaler.inverse_transform(prediction)
    print(f"Denormalized prediction: {denormalized_prediction[0][0]}")
    return denormalized_prediction[0][0]

def load_and_predict(model_name, image):
    model_path = os.path.join(MODEL_SAVE_DIR, model_name)
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(image)
    return predictions.astype(int)  # Ensure the output is an integer

@app.route("/", methods=["GET"])
def home():
    return "", 200

@app.route("/", methods=["POST"])
def process():
    data = request.get_json()
    processed_data = _process(data)
    return jsonify(processed_data)

def _process(data) -> List[Any]:
    image_data = data.get("image")

    # Decode the base64 image data
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
    image = image.resize(TARGET_SIZE)  # Resize image to the expected input size
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict using the models
    area_scaler = joblib.load(os.path.join(SCALER_SAVE_DIR, "area_scaler.pkl"))
    year_scaler = joblib.load(os.path.join(SCALER_SAVE_DIR, "year_scaler.pkl"))

    area = int(predict_and_denormalize("model_area.h5", image_array, area_scaler))
    year = int(predict_and_denormalize("model_year.h5", image_array, year_scaler))
    print("Area and year")
    print(area)
    print(year)
    
    if year < 1953:
        return "the year is too old to get accurate material data"
    elif year < 1963:
        year = 1963        
    materials = get_materials_by_year_and_area(year, area)
    print(f"Materials for year {year} and surface area {area}m²:")
    for material, amount in materials.items():
        print(f"{material}: {amount*area} kg")

    return [int(area), int(year), materials]

if __name__ == "__main__":
    app.run(host="localhost", port=9090)
