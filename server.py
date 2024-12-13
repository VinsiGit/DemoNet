from http.server import BaseHTTPRequestHandler, HTTPServer
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


HOST = "0.0.0.0"
PORT = 9090
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

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"")  # Send an empty response body

    def do_POST(self):
        # Read incoming sent data
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        # Process the image data
        processed_data = self._process(post_data.decode("utf-8"))

        # Prepare the (json) response
        jsonbytes = self._prepare_json_response(processed_data)

        # Send the (json) response back
        self.wfile.write(jsonbytes)

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-type")
        self.end_headers()

    def _process(self, data) -> List[Any]:
        data_dict = json.loads(data)  # Parse the data into a Python object
        image_data = data_dict.get("image")

        # Decode the base64 image data
        image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
        image = image.resize(TARGET_SIZE)  # Resize image to the expected input size
        image_array = np.array(image) / 255.0  # Normalize the image
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict using the models
        # area = load_and_predict('model_area.h5', image_array)
        # year = load_and_predict('model_year.h5', image_array)
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

    def _prepare_json_response(self, response: List[Any]) -> bytes:
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")  # Add CORS header
        self.end_headers()
        jsonstr = json.dumps(response, indent=4)
        return jsonstr.encode("utf-8")  # Encode to bytes

server = HTTPServer((HOST, PORT), Handler)

def main():
    print(f"Starting server at http://{HOST}:{PORT}")
    server.serve_forever()

if __name__ == "__main__":
    main()
