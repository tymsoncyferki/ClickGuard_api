from flask import Flask, request, jsonify, render_template
from pydantic import ValidationError
from flask_cors import CORS

from service import handle_predict, handle_extract, handle_extract_and_predict, handle_predetection
from models import Article, HTMLPayload

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def root():
    return render_template("root.html")

@app.route("/extract", methods=["POST"])
def extract():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.get_json()
        html_payload = HTMLPayload(**data)
        article = handle_extract(html_payload)
        return jsonify(article.model_dump())
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'OPTIONS':
        return '', 204  # Return 204 No Content for OPTIONS requests
    try:
        data = request.get_json()
        article = Article(**data)
        prediction = handle_predict(article)
        return jsonify(prediction.model_dump())
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

@app.route("/extract_and_predict", methods=["POST"])
def extract_and_predict():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.get_json()
        html_payload = HTMLPayload(**data)
        prediction = handle_extract_and_predict(html_payload)
        return jsonify(prediction.model_dump())
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    
@app.route("/predetect", methods=["POST"])
def detect():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.get_json()
        html_payload = HTMLPayload(**data)
        prediction = handle_predetection(html_payload)
        return jsonify(prediction.model_dump())
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
