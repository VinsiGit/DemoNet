from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from typing import List, Any
import base64
import numpy as np
from PIL import Image
from io import BytesIO

from aiFunctions import load_and_predict

HOST = "0.0.0.0"
PORT = 9090


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"")  # Send an empty response body

    def do_POST(self):
        # read incoming sent data
        data = self.rfile.read(self._sent_data_size)

        # process the image data
        processed_data = self._process(data.decode("utf-8"))

        # prepare the (json) response
        jsonbytes = self._prepare_json_response(processed_data)

        # send the (json) response back ...
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
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image = image.resize((224, 224))  # Resize image to the expected input size
        image_array = np.array(image) / 255.0  # Normalize the image
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict using the models
        opp = load_and_predict('path_to_save_model_opp.h5', image_array)
        year = load_and_predict('path_to_save_model_year.h5', image_array)

        return [int(opp[0][0]), int(year[0][0])]

    def _prepare_json_response(self, response: List[Any]) -> bytes:
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")  # Add CORS header
        self.end_headers()
        jsonstr = json.dumps(response, indent=4)
        return jsonstr.encode("utf-8")  # Encode to bytes

    @property
    def _sent_data_size(self) -> int:
        return int(self.headers.get("Content-Length"))


server = HTTPServer((HOST, PORT), Handler)




def main():
    server.serve_forever()


if __name__ == "__main__":
    main()