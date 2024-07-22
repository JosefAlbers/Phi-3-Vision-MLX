import json
from http.server import BaseHTTPRequestHandler, HTTPServer

class DummyModel:
    def generate(self, prompt):
        return prompt + " This is a dummy response."

class SimpleAPIHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.model = DummyModel()
        super().__init__(*args, **kwargs)

    def do_POST(self):
        if self.path == "/v1/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request = json.loads(post_data.decode('utf-8'))

            prompt = request.get('prompt', '')
            response_text = self.model.generate(prompt)

            response = {
                "id": "cmpl-123",
                "object": "text_completion",
                "created": 1234567890,
                "model": "dummy-model",
                "choices": [
                    {
                        "text": response_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "length"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(prompt.split()) + len(response_text.split())
                }
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
(phi) phi % curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!"}'
{"id": "cmpl-123", "object": "text_completion", "created": 1234567890, "model": "dummy-model", "choices": [{"text": "Hello, world! This is a dummy response.", "index": 0, "logprobs": null, "finish_reason": "length"}], "usage": {"prompt_tokens": 2, "completion_tokens": 7, "total_tokens": 9}}%
"""
