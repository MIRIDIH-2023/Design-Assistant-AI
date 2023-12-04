"""
    This file is the main file for the API.

    The API is used to serve the following models:
        - Sentence-BERT (SBERT)

    Usage:
        - Run the API with `docker run -p 5000:5000 naye971012/my-image-name`
"""


from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from core.models.sbert.sbert_main import SBERT

app = Flask(__name__)
CORS(app)

# Test route
@app.route('/')
def index():
    return 'SW Festival API'


# SBERT route
@app.route('/models/sbert', methods=['POST'])
def udop_main():
    response = Response()
    if request.method == 'POST':
        response.headers.add("Access-Control-Allow-Origin", "*")
        data = request.get_json()

        image_url_list = SBERT(data=data)
        
        response = jsonify(image_url_list)
        return response


if __name__ == '__main__':
    SBERT = SBERT()
    app.run(host="0.0.0.0", port=5000)
