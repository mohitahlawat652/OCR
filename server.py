from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from ocr import NeuralNetwork

model = NeuralNetwork(400, 30, 10)
model.load_weights()

class Handler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))

        if self.path == "/train":
            model.train(data['pixels'], data['label'])
            model.save_weights()
            response = {"status": "trained"}
        elif self.path == "/predict":
            output = model.predict(data['pixels'])
            response = {
                "prediction": output.index(max(output)),
                "confidence": max(output)
            }
        else:
            response = {"error": "Invalid endpoint"}

        self._set_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

if __name__ == "__main__":
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, Handler)
    print("Server started on http://localhost:8000")
    httpd.serve_forever()
