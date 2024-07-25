import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from phi_3_vision_mlx import load, generate

preload = load()

class SimpleAPIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request = json.loads(post_data.decode('utf-8'))
            prompts = request.get('prompt', '')
            max_tokens = request.get('max_tokens', 512)
            if isinstance(prompts, str):
                prompts = [prompts]
            responses = generate(prompts, preload=preload, max_tokens=max_tokens)
            if isinstance(responses, str):
                responses = [responses]
            response = {
                "model": "phi-3-vision",
                "responses": responses
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_error(404, "Not Found")

def run(server_class=HTTPServer, handler_class=SimpleAPIHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()

"""
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, world!",
    "max_tokens": 50}'

curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": [
        "Hello, world!",
        "Guten tag!"
    ],
    "max_tokens": 50}'
"""
