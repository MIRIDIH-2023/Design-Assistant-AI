"""
    This file is the main file for the API.

    The API is used to serve the following models:
        - Universal Document Processing (UDOP)

    Usage:
        - Run the API with `python udop.py`
"""


from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from core.models.udop.udop_main import UDOP

app = Flask(__name__)
CORS(app)


# UDOP route
@app.route('/models/udop', methods=['POST'])
def udop_main():
    response = Response()
    if request.method == 'POST':
        response.headers.add("Access-Control-Allow-Origin", "*")
        data = request.get_json()

        layout = UDOP(data=data)

        response = jsonify(layout)
        return response


if __name__ == '__main__':
    UDOP = UDOP()
    app.run(host="0.0.0.0", port=5001)
